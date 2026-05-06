#!/usr/bin/env python3
"""
2D Nernst-Planck + Poisson + Stokes (NP+P+S) — EDL-Resolved EOF
PINN benchmark: 6 coupled PDEs, 4 independent networks, no HS slip

Architecture (following Sun et al. 2024):
  φ-net:    (x,y) → φ           [Poisson]
  c+-net:   (x,y) → c+          [Nernst-Planck +]
  c--net:   (x,y) → c-          [Nernst-Planck -]
  flow-net: (x,y) → (u, v, p)   [Stokes + Continuity]

System (nondimensional, steady-state):
  Poisson:    eps^2 * laplacian(phi) + (c+ - c-) = 0
  NP+:        laplacian(c+) + div(c+ * grad(phi)) = 0
  NP-:        laplacian(c-) - div(c- * grad(phi)) = 0
  Stokes-x:   -p_x + mu * laplacian(u) + Ex*(c+ - c-) = 0
  Stokes-y:   -p_y + mu * laplacian(v) = 0
  Continuity: u_x + v_y = 0

Domain: [0, 1] x [0, 1]
BCs:
  Walls  (y=0, y=1): phi=zeta, c+=exp(-zeta), c-=exp(zeta), u=0, v=0
  Sides  (x=0, x=1): Dirichlet from reference, p=0

Reference (fully developed, y-dependent only, NO HS slip):
  phi(y): symmetric Poisson-Boltzmann
  c±(y)  = exp(-+phi(y))
  u(y)   = Ex*eps^2/mu * (phi(y) - zeta)   [exact EOF velocity]
  v = 0, p = 0
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import json
import copy
import argparse
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ───────────────────────────── CLI ─────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='2D NP+P+Stokes PINN — EDL-resolved EOF')
    p.add_argument('--epsilon', type=float, default=0.2)
    p.add_argument('--zeta', type=float, default=1.0)
    p.add_argument('--Ex', type=float, default=1.0)
    p.add_argument('--mu', type=float, default=1.0)
    p.add_argument('--optimizer', choices=['adam', 'soap'], default='soap')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=50000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-interior', type=int, default=3000)
    p.add_argument('--n-boundary', type=int, default=200)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=5)
    p.add_argument('--weighting', choices=['none', 'lra', 'gradnorm'],
                   default='gradnorm')
    p.add_argument('--lra-alpha', type=float, default=0.1)
    p.add_argument('--gn-update-freq', type=int, default=1000)
    p.add_argument('--gn-momentum', type=float, default=0.9)
    return p.parse_args()


# ───────────────────────── Reference ──────────────────────────
def solve_reference(epsilon, zeta, Ex, mu, n_eval=500):
    def ode(y, u):
        return [u[1], 2.0 * np.sinh(u[0]) / epsilon**2]
    def bc(ua, ub):
        return [ua[0] - zeta, ub[0] - zeta]

    y_mesh = np.linspace(0, 1, 500)
    y_init = np.zeros((2, 500))
    y_init[0] = zeta * np.ones(500)
    sol = solve_bvp(ode, bc, y_mesh, y_init, tol=1e-10, max_nodes=50000)
    if not sol.success:
        y_init[0] = zeta * (1.0 - 4*(y_mesh - 0.5)**2)
        sol = solve_bvp(ode, bc, y_mesh, y_init, tol=1e-8, max_nodes=50000)

    y_eval = np.linspace(0, 1, n_eval)
    phi_ref = sol.sol(y_eval)[0]
    cp_ref = np.exp(-phi_ref)
    cm_ref = np.exp(phi_ref)
    u_ref = Ex * epsilon**2 / mu * (phi_ref - zeta)
    return y_eval, phi_ref, cp_ref, cm_ref, u_ref


# ───────────────────────── Networks ───────────────────────────
class MLP(nn.Module):
    """Generic MLP. n_out=1 for scalar fields, n_out=3 for flow (u,v,p)."""
    def __init__(self, n_in=2, n_out=1, n_hidden=128, n_layers=5):
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

    def forward(self, xy):
        return self.net(2.0 * xy - 1.0)


# ───────────────────────── Collocation ────────────────────────
def make_collocation(n_interior, n_boundary, epsilon, device):
    n_half = n_interior // 2
    xy_uniform = np.random.rand(n_half, 2) * 0.998 + 0.001
    bl = min(5.0 * epsilon, 0.3)
    xy_bl = np.random.rand(n_half, 2)
    xy_bl[:, 0] = xy_bl[:, 0] * 0.998 + 0.001
    n_q = n_half // 2
    xy_bl[:n_q, 1] = np.random.rand(n_q) * bl + 0.001
    xy_bl[n_q:, 1] = 1.0 - np.random.rand(n_half - n_q) * bl - 0.001

    x_int = torch.tensor(np.vstack([xy_uniform, xy_bl]),
                          dtype=torch.float32, device=device)

    nb = n_boundary
    bc_wall = np.vstack([
        np.column_stack([np.linspace(0, 1, nb), np.zeros(nb)]),
        np.column_stack([np.linspace(0, 1, nb), np.ones(nb)]),
    ])
    bc_side = np.vstack([
        np.column_stack([np.zeros(nb), np.linspace(0, 1, nb)]),
        np.column_stack([np.ones(nb), np.linspace(0, 1, nb)]),
    ])
    bc_wall_t = torch.tensor(bc_wall, dtype=torch.float32, device=device)
    bc_side_t = torch.tensor(bc_side, dtype=torch.float32, device=device)
    return x_int, bc_wall_t, bc_side_t


# ───────────────────────── Derivatives ────────────────────────
def laplacian_and_grads(f, xy):
    """Returns (lap_f, f_x, f_y) for scalar f."""
    ones = torch.ones_like(f)
    g = torch.autograd.grad(f, xy, ones, create_graph=True)[0]
    f_x, f_y = g[:, 0:1], g[:, 1:2]
    g_xx = torch.autograd.grad(f_x, xy, ones, create_graph=True)[0]
    g_yy = torch.autograd.grad(f_y, xy, ones, create_graph=True)[0]
    return g_xx[:, 0:1] + g_yy[:, 1:2], f_x, f_y


# ══════════════════════════ MAIN ══════════════════════════════
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*70}")
    print(f"2D NP+P+Stokes — EDL-Resolved EOF (4-network architecture)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"epsilon={args.epsilon}, zeta={args.zeta}, Ex={args.Ex}, mu={args.mu}")
    print(f"Optimizer: {args.optimizer}, lr={args.lr}, epochs={args.epochs}")
    print(f"Network: {args.layers}×{args.hidden}, Weighting: {args.weighting}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eps = args.epsilon
    zeta = args.zeta
    Ex = args.Ex
    mu = args.mu

    # ── Reference ──
    y_ref, phi_ref, cp_ref, cm_ref, u_ref = solve_reference(eps, zeta, Ex, mu)
    phi_interp = interp1d(y_ref, phi_ref, kind='cubic')
    cp_interp = interp1d(y_ref, cp_ref, kind='cubic')
    cm_interp = interp1d(y_ref, cm_ref, kind='cubic')
    u_interp = interp1d(y_ref, u_ref, kind='cubic')
    print(f"Reference: phi=[{phi_ref.min():.4f}, {phi_ref.max():.4f}], "
          f"u_max={u_ref.max():.6e}")

    # ── 4 Networks ──
    # 3 scalar nets (φ, c+, c-) + 1 flow net (u, v, p)
    net_phi = MLP(2, 1, args.hidden, args.layers).to(device)
    net_cp  = MLP(2, 1, args.hidden, args.layers).to(device)
    net_cm  = MLP(2, 1, args.hidden, args.layers).to(device)
    net_flow = MLP(2, 3, args.hidden, args.layers).to(device)  # outputs: u, v, p

    nets = {'phi': net_phi, 'cp': net_cp, 'cm': net_cm, 'flow': net_flow}
    net_names = list(nets.keys())

    all_params = []
    for n in net_names:
        all_params.extend(list(nets[n].parameters()))
    total_params = sum(p.numel() for p in all_params)
    print(f"Networks: phi(1), cp(1), cm(1), flow(3) — {total_params:,} params total")

    # ── Optimizer ──
    if args.optimizer == 'soap':
        try:
            from soap import SOAP
            optimizer = SOAP(all_params, lr=args.lr, betas=(0.95, 0.95),
                             precondition_frequency=2, weight_decay=0.0)
        except ImportError:
            print("SOAP not found, using Adam")
            optimizer = torch.optim.Adam(all_params, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(all_params, lr=args.lr)

    # ── Collocation ──
    x_int, bc_wall, bc_side = make_collocation(
        args.n_interior, args.n_boundary, eps, device)
    print(f"Collocation: {x_int.shape[0]} int, {bc_wall.shape[0]} wall, {bc_side.shape[0]} side")

    # ── Eval grid ──
    nx_ev, ny_ev = 50, 200
    x_ev = np.linspace(0, 1, nx_ev)
    y_ev = np.linspace(0, 1, ny_ev)
    xx_ev, yy_ev = np.meshgrid(x_ev, y_ev)
    xy_ev_t = torch.tensor(
        np.column_stack([xx_ev.ravel(), yy_ev.ravel()]),
        dtype=torch.float32, device=device)
    phi_exact_ev = phi_interp(yy_ev.ravel())
    cp_exact_ev = cp_interp(yy_ev.ravel())
    cm_exact_ev = cm_interp(yy_ev.ravel())
    u_exact_ev = u_interp(yy_ev.ravel())

    # ── PDE / loss names ──
    PDE_NAMES = ['R_Poisson', 'R_NP_p', 'R_NP_m', 'R_Stokes_x', 'R_Stokes_y', 'R_cont']
    ALL_LOSS_NAMES = PDE_NAMES + ['BC']

    # ── Tracking ──
    loss_history = {k: [] for k in ALL_LOSS_NAMES + ['total']}
    L2_history = []
    best_L2_avg = float('inf')
    best_epoch = 0
    best_state = {n: None for n in net_names}
    best_pde_total = float('inf')
    best_pde_epoch = 0
    best_pde_state = {n: None for n in net_names}
    gn_weights = {n: 1.0 for n in ALL_LOSS_NAMES}
    lra_weights = {n: 1.0 for n in ALL_LOSS_NAMES}
    backward_times = []
    eval_times = []

    t0 = time.time()
    LOG_EVERY = 500

    # ══════════════════════ TRAINING LOOP ══════════════════════
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        xy = x_int.clone().requires_grad_(True)

        # ── Forward ──
        phi = nets['phi'](xy)
        cp = nets['cp'](xy)
        cm = nets['cm'](xy)
        flow_out = nets['flow'](xy)
        u_vel = flow_out[:, 0:1]
        v_vel = flow_out[:, 1:2]
        p_pres = flow_out[:, 2:3]

        # ── Derivatives ──
        ones = torch.ones_like(phi)

        lap_phi, phi_x, phi_y = laplacian_and_grads(phi, xy)
        lap_cp, cp_x, cp_y = laplacian_and_grads(cp, xy)
        lap_cm, cm_x, cm_y = laplacian_and_grads(cm, xy)
        lap_u, u_x, u_y = laplacian_and_grads(u_vel, xy)
        lap_v, v_x, v_y = laplacian_and_grads(v_vel, xy)

        p_grad = torch.autograd.grad(p_pres, xy, ones, create_graph=True)[0]
        p_x, p_y = p_grad[:, 0:1], p_grad[:, 1:2]

        # ── PDE residuals ──
        r_poisson = eps**2 * lap_phi + (cp - cm)
        r_np_p = lap_cp + cp_x * phi_x + cp_y * phi_y + cp * lap_phi
        r_np_m = lap_cm - cm_x * phi_x - cm_y * phi_y - cm * lap_phi
        r_stokes_x = -p_x + mu * lap_u + Ex * (cp - cm)
        r_stokes_y = -p_y + mu * lap_v
        r_cont = u_x + v_y

        pde_losses = {
            'R_Poisson': r_poisson.pow(2).mean(),
            'R_NP_p': r_np_p.pow(2).mean(),
            'R_NP_m': r_np_m.pow(2).mean(),
            'R_Stokes_x': r_stokes_x.pow(2).mean(),
            'R_Stokes_y': r_stokes_y.pow(2).mean(),
            'R_cont': r_cont.pow(2).mean(),
        }

        # ── BC losses ──
        # Walls: phi=zeta, c+=exp(-zeta), c-=exp(zeta), u=0, v=0
        flow_wall = nets['flow'](bc_wall)
        bc_loss_wall = (
            (nets['phi'](bc_wall) - zeta).pow(2).mean() +
            (nets['cp'](bc_wall) - np.exp(-zeta)).pow(2).mean() +
            (nets['cm'](bc_wall) - np.exp(zeta)).pow(2).mean() +
            flow_wall[:, 0:1].pow(2).mean() +   # u=0
            flow_wall[:, 1:2].pow(2).mean()      # v=0
        )

        # Sides: Dirichlet from reference, p=0
        y_side = bc_side[:, 1:2].cpu().numpy().flatten()
        phi_ref_s = torch.tensor(phi_interp(y_side), dtype=torch.float32,
                                  device=device).reshape(-1, 1)
        cp_ref_s = torch.tensor(cp_interp(y_side), dtype=torch.float32,
                                 device=device).reshape(-1, 1)
        cm_ref_s = torch.tensor(cm_interp(y_side), dtype=torch.float32,
                                 device=device).reshape(-1, 1)
        u_ref_s = torch.tensor(u_interp(y_side), dtype=torch.float32,
                                device=device).reshape(-1, 1)
        flow_side = nets['flow'](bc_side)

        bc_loss_side = (
            (nets['phi'](bc_side) - phi_ref_s).pow(2).mean() +
            (nets['cp'](bc_side) - cp_ref_s).pow(2).mean() +
            (nets['cm'](bc_side) - cm_ref_s).pow(2).mean() +
            (flow_side[:, 0:1] - u_ref_s).pow(2).mean() +  # u
            flow_side[:, 1:2].pow(2).mean() +               # v=0
            flow_side[:, 2:3].pow(2).mean()                  # p=0
        )

        bc_loss = bc_loss_wall + bc_loss_side
        pde_losses['BC'] = bc_loss

        pde_total = sum(pde_losses[k] for k in PDE_NAMES)

        for k in ALL_LOSS_NAMES:
            loss_history[k].append(pde_losses[k].item())
        loss_history['total'].append((pde_total + bc_loss).item())

        # ── Per-PDE gradients ──
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
        grads['BC'] = torch.cat([p.grad.detach().clone().flatten()
                                  if p.grad is not None
                                  else torch.zeros(p.numel(), device=device)
                                  for p in all_params])

        # ── Weighting ──
        if args.weighting == 'gradnorm':
            if epoch % args.gn_update_freq == 1 or args.gn_update_freq == 1:
                l2_norms = {n: grads[n].norm().item() for n in ALL_LOSS_NAMES}
                mean_norm = np.mean(list(l2_norms.values()))
                for name in ALL_LOSS_NAMES:
                    if l2_norms[name] > 1e-30:
                        gn_hat = mean_norm / l2_norms[name]
                    else:
                        gn_hat = 1.0
                    gn_weights[name] = (args.gn_momentum * gn_weights[name]
                                        + (1 - args.gn_momentum) * gn_hat)
            g_total = sum(gn_weights[n] * grads[n] for n in ALL_LOSS_NAMES)

        elif args.weighting == 'lra':
            max_grads = {n: grads[n].abs().max().item() for n in ALL_LOSS_NAMES}
            mean_max = np.mean(list(max_grads.values()))
            for name in ALL_LOSS_NAMES:
                if max_grads[name] > 1e-30:
                    lra_hat = mean_max / max_grads[name]
                else:
                    lra_hat = 1.0
                lra_weights[name] = ((1 - args.lra_alpha) * lra_weights[name]
                                     + args.lra_alpha * lra_hat)
            g_total = sum(lra_weights[n] * grads[n] for n in ALL_LOSS_NAMES)
        else:
            g_total = sum(grads[n] for n in ALL_LOSS_NAMES)

        # Apply
        optimizer.zero_grad()
        idx = 0
        for p in all_params:
            numel = p.numel()
            p.grad = g_total[idx:idx+numel].reshape(p.shape).clone()
            idx += numel
        backward_times.append(time.time() - t_bw)
        optimizer.step()

        # ── Evaluation ──
        t_ev = time.time()
        with torch.no_grad():
            phi_pred = nets['phi'](xy_ev_t).cpu().numpy().flatten()
            cp_pred = nets['cp'](xy_ev_t).cpu().numpy().flatten()
            cm_pred = nets['cm'](xy_ev_t).cpu().numpy().flatten()
            flow_pred = nets['flow'](xy_ev_t).cpu().numpy()
            u_pred = flow_pred[:, 0]

            L2_phi = np.sqrt(np.mean((phi_pred - phi_exact_ev)**2)) / \
                     (np.sqrt(np.mean(phi_exact_ev**2)) + 1e-30)
            L2_cp = np.sqrt(np.mean((cp_pred - cp_exact_ev)**2)) / \
                    (np.sqrt(np.mean(cp_exact_ev**2)) + 1e-30)
            L2_cm = np.sqrt(np.mean((cm_pred - cm_exact_ev)**2)) / \
                    (np.sqrt(np.mean(cm_exact_ev**2)) + 1e-30)
            L2_u = np.sqrt(np.mean((u_pred - u_exact_ev)**2)) / \
                   (np.sqrt(np.mean(u_exact_ev**2)) + 1e-30)
            L2_avg = (L2_phi + L2_cp + L2_cm + L2_u) / 4.0

        eval_times.append(time.time() - t_ev)
        L2_history.append({'epoch': epoch, 'L2_avg': L2_avg,
                           'L2_phi': L2_phi, 'L2_cp': L2_cp,
                           'L2_cm': L2_cm, 'L2_u': L2_u})

        if L2_avg < best_L2_avg:
            best_L2_avg = L2_avg
            best_epoch = epoch
            best_state = {n: copy.deepcopy(nets[n].state_dict()) for n in net_names}

        total_loss_val = (pde_total + bc_loss).item()
        if total_loss_val < best_pde_total:
            best_pde_total = total_loss_val
            best_pde_epoch = epoch
            best_pde_state = {n: copy.deepcopy(nets[n].state_dict()) for n in net_names}

        # ── Print ──
        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == args.epochs:
            elapsed = (time.time() - t0) / 60
            improved = ' ***' if epoch == best_epoch else ''
            print(f"[{epoch:>6}/{args.epochs}] ({elapsed:5.1f}min) "
                  f"L2={L2_avg:.3e}{improved}")
            pde_str = '  '.join(f'{k.replace("R_","")}={pde_losses[k].item():.1e}'
                                for k in PDE_NAMES)
            print(f"  {pde_str}  BC={bc_loss.item():.1e}")
            print(f"  best={best_L2_avg:.3e}@{best_epoch}  "
                  f"(phi={L2_phi:.1e} cp={L2_cp:.1e} cm={L2_cm:.1e} u={L2_u:.1e})")

    # ══════════════════════ FINAL EVAL ══════════════════════
    for n in net_names:
        nets[n].load_state_dict(best_state[n])
    with torch.no_grad():
        phi_b = nets['phi'](xy_ev_t).cpu().numpy().flatten()
        cp_b = nets['cp'](xy_ev_t).cpu().numpy().flatten()
        cm_b = nets['cm'](xy_ev_t).cpu().numpy().flatten()
        u_b = nets['flow'](xy_ev_t).cpu().numpy()[:, 0]

        L2s = {}
        for name, pred, exact in [('phi', phi_b, phi_exact_ev),
                                   ('cp', cp_b, cp_exact_ev),
                                   ('cm', cm_b, cm_exact_ev),
                                   ('u', u_b, u_exact_ev)]:
            L2s[name] = np.sqrt(np.mean((pred - exact)**2)) / \
                        (np.sqrt(np.mean(exact**2)) + 1e-30)
        L2s['avg'] = np.mean([L2s[k] for k in ['phi', 'cp', 'cm', 'u']])

    # ── Best PDE model eval ──
    for n in net_names:
        nets[n].load_state_dict(best_pde_state[n])
    with torch.no_grad():
        phi_p = nets['phi'](xy_ev_t).cpu().numpy().flatten()
        cp_p = nets['cp'](xy_ev_t).cpu().numpy().flatten()
        cm_p = nets['cm'](xy_ev_t).cpu().numpy().flatten()
        u_p = nets['flow'](xy_ev_t).cpu().numpy()[:, 0]

        L2s_pde = {}
        for name, pred, exact in [('phi', phi_p, phi_exact_ev),
                                   ('cp', cp_p, cp_exact_ev),
                                   ('cm', cm_p, cm_exact_ev),
                                   ('u', u_p, u_exact_ev)]:
            L2s_pde[name] = np.sqrt(np.mean((pred - exact)**2)) / \
                            (np.sqrt(np.mean(exact**2)) + 1e-30)
        L2s_pde['avg'] = np.mean([L2s_pde[k] for k in ['phi', 'cp', 'cm', 'u']])

    final_L2 = L2_history[-1]['L2_avg']
    total_time = (time.time() - t0) / 60.0

    print()
    print("=" * 70)
    print(f"RESULTS (eps={eps}, Ex={Ex}, {args.weighting}, "
          f"seed={args.seed}, {args.optimizer})")
    print("=" * 70)
    print(f"  === Best-ever (epoch {best_epoch}) [oracle] ===")
    for k in ['phi', 'cp', 'cm', 'u']:
        print(f"  L2_{k:>3} = {L2s[k]:.4e}")
    print(f"  L2_avg = {L2s['avg']:.4e}")
    print(f"  === Best-PDE (epoch {best_pde_epoch}) [oracle-free] ===")
    print(f"  PDE_total = {best_pde_total:.4e}")
    for k in ['phi', 'cp', 'cm', 'u']:
        print(f"  L2_{k:>3} = {L2s_pde[k]:.4e}")
    print(f"  L2_avg = {L2s_pde['avg']:.4e}")
    print(f"  === Final epoch ({args.epochs}) ===")
    print(f"  L2_avg = {final_L2:.4e}")
    print(f"  === Gap ===")
    print(f"  Final/Best ratio = {final_L2 / (L2s['avg'] + 1e-30):.2f}x")
    print(f"  PDE-select/Best ratio = {L2s_pde['avg'] / (L2s['avg'] + 1e-30):.2f}x")
    print(f"  Total time: {total_time:.1f} min")

    # ── Save ──
    tag = (f"npps2d_eps{eps}_Ex{Ex}_s{args.seed}"
           f"_{args.weighting}_{args.optimizer}")
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{tag}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    results = {
        'args': vars(args),
        'best_epoch': best_epoch, 'best_L2': L2s,
        'pde_select_epoch': best_pde_epoch,
        'pde_select_total': best_pde_total,
        'pde_select_L2': L2s_pde,
        'final_L2': final_L2,
        'fb_ratio': final_L2 / (L2s['avg'] + 1e-30),
        'pde_best_ratio': L2s_pde['avg'] / (L2s['avg'] + 1e-30),
        'total_time_min': total_time,
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    for n in net_names:
        torch.save(best_state[n], os.path.join(run_dir, f'best_{n}.pt'))
        torch.save(best_pde_state[n], os.path.join(run_dir, f'pde_best_{n}.pt'))

    # ── Plot ──
    try:
        for n in net_names:
            nets[n].load_state_dict(best_state[n])
        with torch.no_grad():
            phi_plot = nets['phi'](xy_ev_t).cpu().numpy().reshape(ny_ev, nx_ev)
            cp_plot = nets['cp'](xy_ev_t).cpu().numpy().reshape(ny_ev, nx_ev)
            cm_plot = nets['cm'](xy_ev_t).cpu().numpy().reshape(ny_ev, nx_ev)
            flow_plot = nets['flow'](xy_ev_t).cpu().numpy()
            u_plot = flow_plot[:, 0].reshape(ny_ev, nx_ev)
            v_plot = flow_plot[:, 1].reshape(ny_ev, nx_ev)
            p_plot = flow_plot[:, 2].reshape(ny_ev, nx_ev)

            phi_ex_2d = phi_exact_ev.reshape(ny_ev, nx_ev)
            cp_ex_2d = cp_exact_ev.reshape(ny_ev, nx_ev)
            cm_ex_2d = cm_exact_ev.reshape(ny_ev, nx_ev)
            u_ex_2d = u_exact_ev.reshape(ny_ev, nx_ev)
            v_ex_2d = np.zeros_like(u_ex_2d)
            p_ex_2d = np.zeros_like(u_ex_2d)

        fig = plt.figure(figsize=(28, 13))
        gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.8],
                              hspace=0.35, wspace=0.45)
        fig.suptitle(f'2D NP+P+Stokes (EDL-resolved)  |  ε={eps}, Ex={Ex}, '
                     f'{args.optimizer}+{args.weighting}',
                     fontsize=15, fontweight='bold')

        # ── Row 0: PINN predictions ──
        fields = [
            ('φ', phi_plot, 'RdBu_r'),
            ('c⁺', cp_plot, 'YlOrRd'),
            ('c⁻', cm_plot, 'YlOrRd'),
            ('u', u_plot, 'viridis'),
            ('v', v_plot, 'RdBu_r'),
            ('p', p_plot, 'RdBu_r'),
        ]
        for j, (label, data, cmap) in enumerate(fields):
            ax = fig.add_subplot(gs[0, j])
            im = ax.pcolormesh(xx_ev, yy_ev, data, cmap=cmap, shading='auto')
            ax.set_title(f'{label} (PINN)', fontsize=11)
            plt.colorbar(im, ax=ax, format='%.2e')
            ax.set_aspect('equal')

        # ── Row 1: Error maps ──
        errors = [
            ('|φ err|', phi_plot, phi_ex_2d),
            ('|c⁺ err|', cp_plot, cp_ex_2d),
            ('|c⁻ err|', cm_plot, cm_ex_2d),
            ('|u err|', u_plot, u_ex_2d),
            ('|v err|', v_plot, v_ex_2d),
            ('|p err|', p_plot, p_ex_2d),
        ]
        for j, (label, pred, exact) in enumerate(errors):
            ax = fig.add_subplot(gs[1, j])
            err = np.abs(pred - exact)
            im = ax.pcolormesh(xx_ev, yy_ev, err, cmap='inferno', shading='auto')
            ax.set_title(f'{label}', fontsize=11)
            plt.colorbar(im, ax=ax, format='%.1e')
            ax.set_aspect('equal')

        # ── Row 2: L2 history (spanning all columns) ──
        ax_l2 = fig.add_subplot(gs[2, :])
        ep_arr = [h['epoch'] for h in L2_history]
        ax_l2.semilogy(ep_arr, [h['L2_avg'] for h in L2_history], 'k-', lw=0.8,
                        alpha=0.7, label='L2_avg')
        ax_l2.axhline(L2s['avg'], color='r', ls='--', lw=1.5,
                       label=f'Oracle best = {L2s["avg"]:.2e} (ep {best_epoch})')
        ax_l2.axhline(L2s_pde['avg'], color='b', ls='--', lw=1.5,
                       label=f'PDE best = {L2s_pde["avg"]:.2e} (ep {best_pde_epoch})')
        ax_l2.set_title('L2_avg history', fontsize=12)
        ax_l2.set_xlabel('Epoch'); ax_l2.set_ylabel('L2 relative error')
        ax_l2.legend(fontsize=10); ax_l2.grid(True, alpha=0.3)

        plt.savefig(os.path.join(run_dir, 'summary.png'), dpi=150,
                    bbox_inches='tight')
        plt.close()
        print(f"  Plot: {run_dir}/summary.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    print(f"  Saved to: {run_dir}/")


if __name__ == '__main__':
    main()
