#!/usr/bin/env python3
"""
1D Nernst-Planck + Poisson (NP+P) — Equilibrium EDL
PINN benchmark with nonlinear coupling and analytical reference

System (nondimensional, steady-state):
  NP+: c+'' + (c+ * phi')' = 0     i.e. c+'' + c+'*phi' + c+*phi'' = 0
  NP-: c-'' - (c- * phi')' = 0     i.e. c-'' - c-'*phi' - c-*phi'' = 0
  P:   eps^2 * phi'' + (c+ - c-) = 0

Domain: x in [0, 1]
BCs:    x=0: phi=zeta, c+=exp(-zeta), c-=exp(zeta)
        x=1: phi=0, c+=1, c-=1

Analytical solution: c± = exp(-+phi),
  where phi solves Poisson-Boltzmann: eps^2*phi'' = 2*sinh(phi)

Coupling parameter: eps = lambda_D / L (Debye length ratio)
  eps -> 0: thin boundary layer, stronger coupling, harder problem
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import json
import argparse
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# ───────────────────────────── CLI ─────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='1D NP+P PINN with independent networks')
    p.add_argument('--epsilon', type=float, default=0.1,
                   help='Debye length ratio (default: 0.1)')
    p.add_argument('--zeta', type=float, default=1.0,
                   help='Wall potential in thermal voltage units (default: 1.0)')
    p.add_argument('--optimizer', choices=['adam', 'soap'], default='soap')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=30000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-interior', type=int, default=300,
                   help='Number of interior collocation points')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=4)
    # Surgery
    p.add_argument('--surgery', choices=['none', 'pcgrad'], default='none',
                   help='Gradient surgery method')
    # Loss weighting
    p.add_argument('--weighting', choices=['none', 'lra', 'gradnorm'],
                   default='none',
                   help='Adaptive loss weighting: none, lra, or gradnorm')
    p.add_argument('--lra-alpha', type=float, default=0.1)
    p.add_argument('--gn-update-freq', type=int, default=1000)
    p.add_argument('--gn-momentum', type=float, default=0.9)
    return p.parse_args()


# ───────────────────────── Reference Solution ─────────────────
def solve_reference(epsilon, zeta, n_eval=2000):
    """Solve Poisson-Boltzmann for the reference phi, then c± = exp(-+phi)."""
    def ode(x, y):
        return [y[1], 2.0 * np.sinh(y[0]) / epsilon**2]

    def bc(ya, yb):
        return [ya[0] - zeta, yb[0] - 0.0]

    n_mesh = 500
    x_mesh = np.linspace(0, 1, n_mesh)
    y_init = np.zeros((2, n_mesh))
    y_init[0] = zeta * (1.0 - x_mesh)

    sol = solve_bvp(ode, bc, x_mesh, y_init, tol=1e-8, max_nodes=50000)
    if not sol.success:
        # Retry with exponential initial guess
        y_init[0] = zeta * np.exp(-x_mesh / max(epsilon, 0.01))
        sol = solve_bvp(ode, bc, x_mesh, y_init, tol=1e-8, max_nodes=50000)
    if not sol.success:
        print(f"WARNING: BVP solver did not converge (eps={epsilon}, zeta={zeta})")

    x_eval = np.linspace(0, 1, n_eval)
    phi_ref = sol.sol(x_eval)[0]
    cp_ref = np.exp(-phi_ref)
    cm_ref = np.exp(phi_ref)
    return x_eval, phi_ref, cp_ref, cm_ref


# ───────────────────────── Network ────────────────────────────
class MLP(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, n_out))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(2.0 * x - 1.0)


# ───────────────────────── Collocation ────────────────────────
def make_collocation(n_interior, epsilon, device):
    """Generate collocation points with concentration near the wall (x=0)
    where the boundary layer lives for small epsilon."""
    # Half uniform, half concentrated near x=0
    n_half = n_interior // 2
    x_uniform = np.linspace(0, 1, n_half + 2)[1:-1]  # exclude endpoints
    # Exponential concentration: more points in [0, 5*eps]
    bl_width = min(5.0 * epsilon, 0.5)
    x_bl = bl_width * (1.0 - np.cos(np.linspace(0, np.pi/2, n_half))) 
    x_all = np.unique(np.concatenate([x_uniform, x_bl]))
    x_all = np.sort(x_all)
    # Remove points too close to boundaries
    x_all = x_all[(x_all > 1e-6) & (x_all < 1.0 - 1e-6)]
    x_t = torch.tensor(x_all, dtype=torch.float32, device=device).reshape(-1, 1)
    return x_t


# ────────────────────────── PCGrad ────────────────────────────
def pcgrad_project(ga, gb):
    """Project ga to remove component conflicting with gb, and vice versa."""
    dot = torch.dot(ga, gb)
    if dot < 0:
        ga_new = ga - (dot / (gb.norm()**2 + 1e-30)) * gb
        gb_new = gb - (dot / (ga.norm()**2 + 1e-30)) * ga
        return ga_new, gb_new, True
    return ga, gb, False


# ══════════════════════════ MAIN ══════════════════════════════
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"epsilon={args.epsilon}, zeta={args.zeta}")
    print(f"Optimizer: {args.optimizer}, lr={args.lr}")
    print(f"Surgery: {args.surgery}")
    print(f"Weighting: {args.weighting}" +
          (f" (alpha={args.lra_alpha})" if args.weighting == 'lra' else "") +
          (f" (freq={args.gn_update_freq}, mom={args.gn_momentum})"
           if args.weighting == 'gradnorm' else ""))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Reference solution ──
    x_ref_np, phi_ref_np, cp_ref_np, cm_ref_np = solve_reference(
        args.epsilon, args.zeta)
    print(f"Reference: phi=[{phi_ref_np.min():.4f}, {phi_ref_np.max():.4f}], "
          f"c+=[{cp_ref_np.min():.4f}, {cp_ref_np.max():.4f}], "
          f"c-=[{cm_ref_np.min():.4f}, {cm_ref_np.max():.4f}]")

    # Interpolators for evaluation
    phi_interp = interp1d(x_ref_np, phi_ref_np, kind='cubic')
    cp_interp = interp1d(x_ref_np, cp_ref_np, kind='cubic')
    cm_interp = interp1d(x_ref_np, cm_ref_np, kind='cubic')

    # ── Networks ──
    net_names = ['cp', 'cm', 'phi']
    nets = {
        'cp':  MLP(1, 1, args.hidden, args.layers).to(device),
        'cm':  MLP(1, 1, args.hidden, args.layers).to(device),
        'phi': MLP(1, 1, args.hidden, args.layers).to(device),
    }
    all_params = []
    param_groups = {}
    for name in net_names:
        params = list(nets[name].parameters())
        all_params.extend(params)
        param_groups[name] = params

    # ── Optimizer ──
    if args.optimizer == 'soap':
        try:
            from soap import SOAP
            optimizer = SOAP(all_params, lr=args.lr, betas=(0.95, 0.95),
                             precondition_frequency=2, weight_decay=0.0)
        except ImportError:
            print("SOAP not found, falling back to Adam")
            optimizer = torch.optim.Adam(all_params, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(all_params, lr=args.lr)

    # ── Collocation ──
    x_int = make_collocation(args.n_interior, args.epsilon, device)
    x_bc0 = torch.zeros(1, 1, device=device)
    x_bc1 = torch.ones(1, 1, device=device)
    print(f"Collocation: {x_int.shape[0]} interior points")

    # ── BC values ──
    zeta = args.zeta
    eps = args.epsilon
    bc_cp_0 = np.exp(-zeta)   # c+(0)
    bc_cm_0 = np.exp(zeta)    # c-(0)
    bc_phi_0 = zeta            # phi(0)
    bc_cp_1 = 1.0              # c+(1)
    bc_cm_1 = 1.0              # c-(1)
    bc_phi_1 = 0.0             # phi(1)

    # ── PDE names ──
    PDE_NAMES = ['R_NP_p', 'R_NP_m', 'R_P']
    ALL_PAIRS = [('R_NP_p', 'R_NP_m'), ('R_NP_p', 'R_P'), ('R_NP_m', 'R_P')]

    # ── Tracking ──
    loss_history = {k: [] for k in PDE_NAMES + ['BC', 'total']}
    L2_history = []
    best_L2_avg = float('inf')
    best_epoch = 0
    best_state = {n: None for n in net_names}
    best_pde_L2_avg = float('inf')
    best_pde_epoch = 0
    best_pde_state = {n: None for n in net_names}

    neg_ratio_hist = {f'{a}-{b}': [] for a, b in ALL_PAIRS}
    surgery_count = {f'{a}-{b}': 0 for a, b in ALL_PAIRS}
    mag_ratio_hist = {f'{a}-{b}': [] for a, b in ALL_PAIRS}
    backward_times = []
    eval_times = []

    # Adaptive weights
    ALL_LOSS_NAMES = PDE_NAMES + ['BC']
    lra_weights = {n: 1.0 for n in ALL_LOSS_NAMES}
    lra_weight_hist = {n: [] for n in ALL_LOSS_NAMES}
    gn_weights = {n: 1.0 for n in ALL_LOSS_NAMES}

    t0 = time.time()
    LOG_EVERY = 500

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # ── Forward pass ──
        x = x_int.clone().requires_grad_(True)
        cp_pred = nets['cp'](x)
        cm_pred = nets['cm'](x)
        phi_pred = nets['phi'](x)

        # First derivatives
        ones = torch.ones_like(cp_pred)
        cp_x = torch.autograd.grad(cp_pred, x, ones, create_graph=True)[0]
        cm_x = torch.autograd.grad(cm_pred, x, ones, create_graph=True)[0]
        phi_x = torch.autograd.grad(phi_pred, x, ones, create_graph=True)[0]

        # Second derivatives
        cp_xx = torch.autograd.grad(cp_x, x, ones, create_graph=True)[0]
        cm_xx = torch.autograd.grad(cm_x, x, ones, create_graph=True)[0]
        phi_xx = torch.autograd.grad(phi_x, x, ones, create_graph=True)[0]

        # ── PDE residuals ──
        # NP+: c+'' + c+'*phi' + c+*phi'' = 0
        r_np_p = cp_xx + cp_x * phi_x + cp_pred * phi_xx
        # NP-: c-'' - c-'*phi' - c-*phi'' = 0
        r_np_m = cm_xx - cm_x * phi_x - cm_pred * phi_xx
        # P: eps^2 * phi'' + (c+ - c-) = 0
        r_p = eps**2 * phi_xx + (cp_pred - cm_pred)

        L_np_p = r_np_p.pow(2).mean()
        L_np_m = r_np_m.pow(2).mean()
        L_p = r_p.pow(2).mean()
        pde_total = L_np_p + L_np_m + L_p

        # ── BC losses ──
        bc_loss = (
            (nets['phi'](x_bc0) - bc_phi_0).pow(2).mean() +
            (nets['phi'](x_bc1) - bc_phi_1).pow(2).mean() +
            (nets['cp'](x_bc0) - bc_cp_0).pow(2).mean() +
            (nets['cp'](x_bc1) - bc_cp_1).pow(2).mean() +
            (nets['cm'](x_bc0) - bc_cm_0).pow(2).mean() +
            (nets['cm'](x_bc1) - bc_cm_1).pow(2).mean()
        )

        pde_losses = {'R_NP_p': L_np_p, 'R_NP_m': L_np_m, 'R_P': L_p}

        loss_history['R_NP_p'].append(L_np_p.item())
        loss_history['R_NP_m'].append(L_np_m.item())
        loss_history['R_P'].append(L_p.item())
        loss_history['BC'].append(bc_loss.item())
        loss_history['total'].append((pde_total + bc_loss).item())

        # ── Compute per-PDE gradients ──
        t_bw = time.time()
        grads = {}
        for pde_name in PDE_NAMES:
            optimizer.zero_grad()
            pde_losses[pde_name].backward(retain_graph=True)
            g = torch.cat([p.grad.detach().clone().flatten()
                           if p.grad is not None
                           else torch.zeros(p.numel(), device=device)
                           for p in all_params])
            grads[pde_name] = g

        optimizer.zero_grad()
        bc_loss.backward(retain_graph=True)
        g_bc = torch.cat([p.grad.detach().clone().flatten()
                          if p.grad is not None
                          else torch.zeros(p.numel(), device=device)
                          for p in all_params])

        # ── Gradient diagnostics ──
        for (a, b) in ALL_PAIRS:
            key = f'{a}-{b}'
            dot = torch.dot(grads[a], grads[b]).item()
            na = grads[a].norm().item()
            nb = grads[b].norm().item()
            cos = dot / (na * nb + 1e-30)
            neg_ratio_hist[key].append(1 if cos < 0 else 0)
            if na > 0 and nb > 0:
                mag_ratio_hist[key].append(min(na, nb) / max(na, nb))

        # ── Surgery ──
        if args.surgery == 'pcgrad':
            for (a, b) in ALL_PAIRS:
                ga_new, gb_new, did = pcgrad_project(grads[a], grads[b])
                if did:
                    surgery_count[f'{a}-{b}'] += 1
                grads[a] = ga_new
                grads[b] = gb_new

        # ── LRA weighting ──
        if args.weighting == 'lra':
            max_grads = {}
            for name in PDE_NAMES:
                max_grads[name] = grads[name].abs().max().item()
            max_grads['BC'] = g_bc.abs().max().item()
            mean_max = np.mean([max_grads[n] for n in ALL_LOSS_NAMES])
            for name in ALL_LOSS_NAMES:
                if max_grads[name] > 1e-30:
                    lra_hat = mean_max / max_grads[name]
                else:
                    lra_hat = 1.0
                lra_weights[name] = ((1 - args.lra_alpha) * lra_weights[name]
                                     + args.lra_alpha * lra_hat)
            for name in ALL_LOSS_NAMES:
                lra_weight_hist[name].append(lra_weights[name])
            g_total = (sum(lra_weights[n] * grads[n] for n in PDE_NAMES)
                       + lra_weights['BC'] * g_bc)

        # ── GradNorm weighting ──
        elif args.weighting == 'gradnorm':
            if epoch % args.gn_update_freq == 1 or args.gn_update_freq == 1:
                l2_norms = {}
                for name in PDE_NAMES:
                    l2_norms[name] = grads[name].norm().item()
                l2_norms['BC'] = g_bc.norm().item()
                mean_norm = np.mean([l2_norms[n] for n in ALL_LOSS_NAMES])
                for name in ALL_LOSS_NAMES:
                    if l2_norms[name] > 1e-30:
                        gn_hat = mean_norm / l2_norms[name]
                    else:
                        gn_hat = 1.0
                    gn_weights[name] = (args.gn_momentum * gn_weights[name]
                                        + (1 - args.gn_momentum) * gn_hat)
            for name in ALL_LOSS_NAMES:
                lra_weight_hist[name].append(gn_weights[name])
            g_total = (sum(gn_weights[n] * grads[n] for n in PDE_NAMES)
                       + gn_weights['BC'] * g_bc)
        else:
            g_total = sum(grads[n] for n in PDE_NAMES) + g_bc

        # ── Apply manual gradient ──
        idx = 0
        for p in all_params:
            numel = p.numel()
            p.grad = g_total[idx:idx+numel].reshape(p.shape).clone()
            idx += numel
        optimizer.step()
        backward_times.append(time.time() - t_bw)

        # ── Evaluate L2 error ──
        t_ev = time.time()
        with torch.no_grad():
            x_eval = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
            cp_eval = nets['cp'](x_eval).cpu().numpy().flatten()
            cm_eval = nets['cm'](x_eval).cpu().numpy().flatten()
            phi_eval = nets['phi'](x_eval).cpu().numpy().flatten()
            x_np = x_eval.cpu().numpy().flatten()

            cp_exact = cp_interp(x_np)
            cm_exact = cm_interp(x_np)
            phi_exact = phi_interp(x_np)

            L2_cp = np.sqrt(np.mean((cp_eval - cp_exact)**2)) / (np.sqrt(np.mean(cp_exact**2)) + 1e-30)
            L2_cm = np.sqrt(np.mean((cm_eval - cm_exact)**2)) / (np.sqrt(np.mean(cm_exact**2)) + 1e-30)
            L2_phi = np.sqrt(np.mean((phi_eval - phi_exact)**2)) / (np.sqrt(np.mean(phi_exact**2)) + 1e-30)
            L2_avg = (L2_cp + L2_cm + L2_phi) / 3.0

        eval_times.append(time.time() - t_ev)

        L2_history.append({
            'epoch': epoch, 'L2_cp': L2_cp, 'L2_cm': L2_cm,
            'L2_phi': L2_phi, 'L2_avg': L2_avg
        })

        # ── Track best (oracle) ──
        if L2_avg < best_L2_avg:
            best_L2_avg = L2_avg
            best_epoch = epoch
            for n in net_names:
                best_state[n] = {k: v.cpu().clone()
                                 for k, v in nets[n].state_dict().items()}

        # ── Track best PDE-loss ──
        pde_val = (pde_total + bc_loss).item()
        if pde_val < best_pde_L2_avg:
            best_pde_L2_avg = pde_val
            best_pde_epoch = epoch
            for n in net_names:
                best_pde_state[n] = {k: v.cpu().clone()
                                     for k, v in nets[n].state_dict().items()}

        # ── Logging ──
        if epoch % LOG_EVERY == 0 or epoch == 1:
            elapsed = (time.time() - t0) / 60.0
            print(f"  ep {epoch:5d} ({elapsed:.1f}min)  R_NP+={L_np_p.item():.2e}  "
                  f"R_NP-={L_np_m.item():.2e}  R_P={L_p.item():.2e}  "
                  f"BC={bc_loss.item():.2e}")
            print(f"        L2_avg={L2_avg:.3e}  "
                  f"(best={best_L2_avg:.3e} @ ep {best_epoch})")
            if args.weighting == 'lra':
                w_str = ' '.join(f'{n}:{lra_weights[n]:.2f}'
                                 for n in ALL_LOSS_NAMES)
                print(f"        lra_w=[{w_str}]")
            elif args.weighting == 'gradnorm':
                w_str = ' '.join(f'{n}:{gn_weights[n]:.2f}'
                                 for n in ALL_LOSS_NAMES)
                print(f"        gn_w=[{w_str}]")

    # ═══════════════════ Evaluation ═══════════════════

    # ── Oracle best ──
    for n in net_names:
        nets[n].load_state_dict(best_state[n])
    x_eval = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
    with torch.no_grad():
        cp_eval = nets['cp'](x_eval).cpu().numpy().flatten()
        cm_eval = nets['cm'](x_eval).cpu().numpy().flatten()
        phi_eval = nets['phi'](x_eval).cpu().numpy().flatten()
        x_np = x_eval.cpu().numpy().flatten()

    cp_exact = cp_interp(x_np)
    cm_exact = cm_interp(x_np)
    phi_exact = phi_interp(x_np)

    L2_cp_best = np.sqrt(np.mean((cp_eval - cp_exact)**2)) / (np.sqrt(np.mean(cp_exact**2)) + 1e-30)
    L2_cm_best = np.sqrt(np.mean((cm_eval - cm_exact)**2)) / (np.sqrt(np.mean(cm_exact**2)) + 1e-30)
    L2_phi_best = np.sqrt(np.mean((phi_eval - phi_exact)**2)) / (np.sqrt(np.mean(phi_exact**2)) + 1e-30)
    L2_avg_best = (L2_cp_best + L2_cm_best + L2_phi_best) / 3.0

    # ── PDE-select best ──
    for n in net_names:
        nets[n].load_state_dict(best_pde_state[n])
    with torch.no_grad():
        cp_pde = nets['cp'](x_eval).cpu().numpy().flatten()
        cm_pde = nets['cm'](x_eval).cpu().numpy().flatten()
        phi_pde = nets['phi'](x_eval).cpu().numpy().flatten()

    L2_cp_pde = np.sqrt(np.mean((cp_pde - cp_exact)**2)) / (np.sqrt(np.mean(cp_exact**2)) + 1e-30)
    L2_cm_pde = np.sqrt(np.mean((cm_pde - cm_exact)**2)) / (np.sqrt(np.mean(cm_exact**2)) + 1e-30)
    L2_phi_pde = np.sqrt(np.mean((phi_pde - phi_exact)**2)) / (np.sqrt(np.mean(phi_exact**2)) + 1e-30)
    L2_avg_pde = (L2_cp_pde + L2_cm_pde + L2_phi_pde) / 3.0

    # ── Final ──
    final_L2 = L2_history[-1]['L2_avg']

    # ── Print results ──
    total_time = (time.time() - t0) / 60.0
    print()
    print("=" * 70)
    print(f"RESULTS (eps={args.epsilon}, zeta={args.zeta}, "
          f"{args.surgery}, {args.weighting}, "
          f"seed={args.seed}, {args.optimizer})")
    print("=" * 70)
    print(f"  === Best-ever (epoch {best_epoch}) [oracle] ===")
    print(f"  L2_cp  = {L2_cp_best:.4e}")
    print(f"  L2_cm  = {L2_cm_best:.4e}")
    print(f"  L2_phi = {L2_phi_best:.4e}")
    print(f"  L2_avg = {L2_avg_best:.4e}")
    print(f"  === Best-PDE-loss (epoch {best_pde_epoch}) [practical] ===")
    print(f"  L2_cp  = {L2_cp_pde:.4e}")
    print(f"  L2_cm  = {L2_cm_pde:.4e}")
    print(f"  L2_phi = {L2_phi_pde:.4e}")
    print(f"  L2_avg = {L2_avg_pde:.4e}  (PDE_total={best_pde_L2_avg:.4e})")
    print(f"  === Final epoch ({args.epochs}) ===")
    print(f"  L2_avg = {final_L2:.4e}")
    print(f"  === Gap ===")
    print(f"  Final/Best ratio = {final_L2 / (L2_avg_best + 1e-30):.2f}x")
    print(f"  Avg backward: {np.mean(backward_times)*1000:.2f} ms/step")
    print(f"  Avg L2 eval:  {np.mean(eval_times)*1000:.2f} ms/step (every epoch)")
    print(f"  Total time: {total_time:.1f} min")

    for key in [f'{a}-{b}' for a, b in ALL_PAIRS]:
        nr = neg_ratio_hist[key]
        pct = 100.0 * sum(nr) / len(nr) if nr else 0
        mr = mag_ratio_hist[key]
        mr_mean = np.mean(mr) if mr else 0
        print(f"  {key}: neg={pct:.1f}%, mag_ratio={mr_mean:.4f}")

    # ── Save results ──
    results = {
        'args': vars(args),
        'best_epoch': best_epoch,
        'best_L2': {'cp': L2_cp_best, 'cm': L2_cm_best,
                     'phi': L2_phi_best, 'avg': L2_avg_best},
        'pde_select_epoch': best_pde_epoch,
        'pde_select_L2': {'cp': L2_cp_pde, 'cm': L2_cm_pde,
                           'phi': L2_phi_pde, 'avg': L2_avg_pde},
        'final_L2': final_L2,
        'fb_ratio': final_L2 / (L2_avg_best + 1e-30),
        'L2_history': L2_history,
        'loss_history': {k: v for k, v in loss_history.items()},
        'neg_ratios': {k: sum(v)/len(v) if v else 0
                       for k, v in neg_ratio_hist.items()},
        'mag_ratios': {k: float(np.mean(v)) if v else 0
                       for k, v in mag_ratio_hist.items()},
        'total_time_min': total_time,
    }

    tag = (f"npp_eps{args.epsilon}_z{args.zeta}_s{args.seed}"
           f"_{args.surgery}_{args.weighting}_{args.optimizer}")
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{tag}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    for n in net_names:
        torch.save(best_state[n], os.path.join(run_dir, f'best_{n}.pt'))

    print(f"  Saved to: {run_dir}/")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 1: Solutions (oracle best)
    for n in net_names:
        nets[n].load_state_dict(best_state[n])
    with torch.no_grad():
        cp_plot = nets['cp'](x_eval).cpu().numpy().flatten()
        cm_plot = nets['cm'](x_eval).cpu().numpy().flatten()
        phi_plot = nets['phi'](x_eval).cpu().numpy().flatten()

    axes[0, 0].plot(x_np, phi_exact, 'k-', label='Exact', lw=2)
    axes[0, 0].plot(x_np, phi_plot, 'r--', label='PINN', lw=1.5)
    axes[0, 0].set_title(f'phi (eps={args.epsilon})')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('x')

    axes[0, 1].plot(x_np, cp_exact, 'k-', label='Exact c+', lw=2)
    axes[0, 1].plot(x_np, cp_plot, 'r--', label='PINN c+', lw=1.5)
    axes[0, 1].plot(x_np, cm_exact, 'b-', label='Exact c-', lw=2)
    axes[0, 1].plot(x_np, cm_plot, 'b--', label='PINN c-', lw=1.5)
    axes[0, 1].set_title('Concentrations')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('x')

    axes[0, 2].plot(x_np, np.abs(phi_plot - phi_exact), 'r-', label='|err phi|')
    axes[0, 2].plot(x_np, np.abs(cp_plot - cp_exact), 'g-', label='|err c+|')
    axes[0, 2].plot(x_np, np.abs(cm_plot - cm_exact), 'b-', label='|err c-|')
    axes[0, 2].set_yscale('log')
    axes[0, 2].set_title('Pointwise error')
    axes[0, 2].legend()
    axes[0, 2].set_xlabel('x')

    # Row 2: Training history
    lh = loss_history
    for key in PDE_NAMES:
        axes[1, 0].semilogy(lh[key], label=key, alpha=0.7)
    axes[1, 0].semilogy(lh['BC'], label='BC', alpha=0.7)
    axes[1, 0].set_title('Loss history')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Epoch')

    L2h = L2_history
    epochs_arr = [h['epoch'] for h in L2h]
    axes[1, 1].semilogy(epochs_arr, [h['L2_cp'] for h in L2h], label='L2 c+')
    axes[1, 1].semilogy(epochs_arr, [h['L2_cm'] for h in L2h], label='L2 c-')
    axes[1, 1].semilogy(epochs_arr, [h['L2_phi'] for h in L2h], label='L2 phi')
    axes[1, 1].semilogy(epochs_arr, [h['L2_avg'] for h in L2h], 'k-', label='L2 avg', lw=2)
    axes[1, 1].axvline(best_epoch, color='red', ls='--', alpha=0.5, label=f'Best@{best_epoch}')
    axes[1, 1].set_title('L2 error history')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Epoch')

    # Summary text
    axes[1, 2].axis('off')
    summary = (
        f"eps={args.epsilon}, zeta={args.zeta}\n"
        f"Optimizer: {args.optimizer}\n"
        f"Surgery: {args.surgery}\n"
        f"Weighting: {args.weighting}\n"
        f"Seed: {args.seed}\n\n"
        f"Best L2_avg: {L2_avg_best:.3e} @ ep {best_epoch}\n"
        f"PDE-select:  {L2_avg_pde:.3e} @ ep {best_pde_epoch}\n"
        f"Final L2:    {final_L2:.3e}\n"
        f"F/B ratio:   {final_L2/(L2_avg_best+1e-30):.1f}x\n"
        f"Time: {total_time:.1f} min"
    )
    axes[1, 2].text(0.1, 0.5, summary, fontsize=11, family='monospace',
                    verticalalignment='center', transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'summary.png'), dpi=150)
    plt.close()
    print(f"  Plot saved: {run_dir}/summary.png")


if __name__ == '__main__':
    main()
