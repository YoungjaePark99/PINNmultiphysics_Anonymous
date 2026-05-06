"""
1D Steady-State Thermoelasticity — Multiphysics PINN Benchmark
================================================================
Genuine cross-physics coupling: Heat conduction + Solid mechanics.

PDEs (physically exact steady-state thermoelasticity):
  R_T:  -κ T''(x)              = f_T(x)     (heat conduction)
  R_u:  -E u''(x) + γ T'(x)   = f_u(x)     (mechanical equilibrium with thermal stress)

  γ = E·α_T  (thermal stress coefficient)
  One-way coupling: T → u  (temperature gradient drives thermal stress)

Gradient conflict mechanism:
  In θ_T space, ∂L_RT/∂θ_T wants T to satisfy heat equation,
  while ∂L_Ru/∂θ_T wants T' to provide correct thermal stress for u.
  These can conflict — classic Case A (structural inter-PDE conflict).

Exact solutions (MMS):
  T(x) = sin(πx) + 1      →  T(0) = 1, T(1) = 1
  u(x) = sin(2πx)          →  u(0) = 0, u(1) = 0

Source terms:
  f_T = κ π² sin(πx)
  f_u = E(2π)² sin(2πx) + γ π cos(πx)

Usage:
  python thermoelasticity_1d.py --gamma 10 --optimizer soap
  python thermoelasticity_1d.py --gamma 10 --optimizer adam
  python thermoelasticity_1d.py --gamma 10 --optimizer soap --surgery pcgrad
"""

import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import time
import os
import json
import math
import matplotlib
import matplotlib.pyplot as plt
from soap import SOAP

matplotlib.use('Agg')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
PI = math.pi


# ══════════════════════════════════════════════════════════════════════
# Arguments
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description='1D Thermoelasticity (Multiphysics)')
    # Physics
    p.add_argument('--kappa', type=float, default=1.0,
                   help='Thermal conductivity κ')
    p.add_argument('--E-modulus', type=float, default=1.0,
                   help="Young's modulus E")
    p.add_argument('--gamma', type=float, default=1.0,
                   help='Thermal stress coefficient γ = E·α_T (coupling strength)')
    # Training
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=60000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--N-domain', type=int, default=200)
    # Architecture
    p.add_argument('--n-hidden', type=int, default=4)
    p.add_argument('--n-neurons', type=int, default=64)
    # Optimizer
    p.add_argument('--optimizer', choices=['adam', 'soap'], default='adam')
    p.add_argument('--soap-beta1', type=float, default=0.99)
    p.add_argument('--soap-beta2', type=float, default=0.999)
    p.add_argument('--soap-precond-freq', type=int, default=2)
    # Surgery
    p.add_argument('--surgery', choices=['none', 'pcgrad'],
                   default='none',
                   help='none or pcgrad (only 1 PDE pair, so no physics_aware variant)')
    # Loss weighting
    p.add_argument('--weighting', choices=['none', 'lra', 'gradnorm'],
                   default='none',
                   help='Adaptive loss weighting: none, lra (Wang et al. 2021), '
                        'or gradnorm (L2-norm, Wang et al. 2025)')
    p.add_argument('--lra-alpha', type=float, default=0.1,
                   help='LRA EMA smoothing factor (default: 0.1)')
    p.add_argument('--gn-update-freq', type=int, default=1000,
                   help='GradNorm update frequency in steps (default: 1000)')
    p.add_argument('--gn-momentum', type=float, default=0.9,
                   help='GradNorm EMA momentum (default: 0.9)')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Network
# ══════════════════════════════════════════════════════════════════════
class SubNet(nn.Module):
    def __init__(self, n_hidden, n_neurons):
        super().__init__()
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(2.0 * x - 1.0)


# ══════════════════════════════════════════════════════════════════════
# Exact solutions and source terms
# ══════════════════════════════════════════════════════════════════════
def exact_T(x):
    return torch.sin(PI * x) + 1.0

def exact_u(x):
    return torch.sin(2 * PI * x)

def compute_sources(x, args):
    """Source terms from MMS: f = PDE(exact)."""
    sx = torch.sin(PI * x)
    cx = torch.cos(PI * x)
    s2x = torch.sin(2 * PI * x)

    # f_T = κ π² sin(πx)     [from -κ T'' = f_T]
    f_T = args.kappa * PI**2 * sx

    # f_u = E(2π)² sin(2πx) + γ π cos(πx)   [from -E u'' + γ T' = f_u]
    f_u = args.E_modulus * (2*PI)**2 * s2x + args.gamma * PI * cx

    return f_T.detach(), f_u.detach()


# ══════════════════════════════════════════════════════════════════════
# PDE residuals
# ══════════════════════════════════════════════════════════════════════
def compute_residuals(nets, x, src, args):
    T = nets['T'](x)
    u = nets['u'](x)

    T_x = torch.autograd.grad(T, x, torch.ones_like(T),
                               create_graph=True, retain_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x),
                                create_graph=True, retain_graph=True)[0]

    u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]

    f_T, f_u = src

    # R_T: -κ T'' - f_T = 0
    R_T = -args.kappa * T_xx - f_T

    # R_u: -E u'' + γ T' - f_u = 0   (T' is the coupling term!)
    R_u = -args.E_modulus * u_xx + args.gamma * T_x - f_u

    return {
        'RT': torch.mean(R_T**2),
        'Ru': torch.mean(R_u**2),
    }


def compute_bc_loss(nets, args, device):
    x0 = torch.zeros(1, 1, device=device)
    x1 = torch.ones(1, 1, device=device)

    loss = 0.0
    for x_bc in [x0, x1]:
        loss += (nets['T'](x_bc) - exact_T(x_bc))**2
        loss += (nets['u'](x_bc) - exact_u(x_bc))**2
    return loss.squeeze()


# ══════════════════════════════════════════════════════════════════════
# Gradient utilities
# ══════════════════════════════════════════════════════════════════════
def get_grad_vec(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    vecs = []
    for g, p in zip(grads, params):
        if g is None:
            vecs.append(torch.zeros_like(p).reshape(-1))
        else:
            vecs.append(g.reshape(-1))
    return torch.cat(vecs)


def cosine_sim(g1, g2):
    n1, n2 = g1.norm(), g2.norm()
    if n1 < 1e-30 or n2 < 1e-30:
        return 0.0
    return (g1 @ g2 / (n1 * n2)).item()


def pcgrad_project(ga, gb):
    dot = ga @ gb
    if dot < 0:
        ga = ga - (dot / (gb @ gb + 1e-30)) * gb
        return ga, True
    return ga, False


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"κ={args.kappa}, E={args.E_modulus}, γ={args.gamma}")
    print(f"Surgery: {args.surgery}")
    print(f"Weighting: {args.weighting}" +
          (f" (α={args.lra_alpha})" if args.weighting == 'lra' else "") +
          (f" (freq={args.gn_update_freq}, mom={args.gn_momentum})" if args.weighting == 'gradnorm' else ""))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Networks (independent: T and u)
    net_names = ['T', 'u']
    nets = {n: SubNet(args.n_hidden, args.n_neurons).to(device) for n in net_names}

    all_params = []
    for n in net_names:
        all_params.extend(list(nets[n].parameters()))

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(all_params, lr=args.lr)
    elif args.optimizer == 'soap':
        optimizer = SOAP(
            all_params, lr=args.lr,
            betas=(args.soap_beta1, args.soap_beta2),
            weight_decay=0.0,
            precondition_frequency=args.soap_precond_freq,
            precondition_1d=False,
        )
        print(f"SOAP: betas=({args.soap_beta1}, {args.soap_beta2}), "
              f"precond_freq={args.soap_precond_freq}")

    # Collocation
    x_int = torch.linspace(0.01, 0.99, args.N_domain, device=device).reshape(-1, 1)
    x_int.requires_grad_(True)

    # Precompute sources
    src = compute_sources(x_int, args)
    src_mags = [s.abs().max().item() for s in src]
    print(f"Source magnitudes: |f_T|={src_mags[0]:.2e}, |f_u|={src_mags[1]:.2e}")

    # ─── Best-ever checkpoint ────
    best_L2_avg = float('inf')
    best_epoch = -1
    best_state = None

    # ─── Best-PDE-loss checkpoint (no oracle needed) ────
    best_pde_total = float('inf')
    best_pde_epoch = -1
    best_pde_state = None
    L2_history = []
    x_eval_mon = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
    with torch.no_grad():
        T_e_mon = exact_T(x_eval_mon)
        u_e_mon = exact_u(x_eval_mon)

    # Monitoring
    PDE_NAMES = ['RT', 'Ru']
    ALL_PAIRS = [('RT', 'Ru')]

    loss_hist = {k: [] for k in PDE_NAMES + ['BC', 'total']}
    cosine_hist = {'RT-Ru': []}
    neg_count = {'RT-Ru': 0}
    surgery_count = {'RT-Ru': 0}
    mag_ratio_hist = {'RT-Ru': []}
    backward_times = []
    eval_times = []

    # Adaptive weights
    ALL_LOSS_NAMES = PDE_NAMES + ['BC']
    lra_weights = {n: 1.0 for n in ALL_LOSS_NAMES}
    lra_weight_hist = {n: [] for n in ALL_LOSS_NAMES}

    # GradNorm weights
    gn_weights = {n: 1.0 for n in ALL_LOSS_NAMES}

    t0 = time.time()
    LOG_EVERY = 500

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        pde_losses = compute_residuals(nets, x_int, src, args)
        bc_loss = compute_bc_loss(nets, args, device)

        t_back = time.time()

        grads = {}
        for name in PDE_NAMES:
            grads[name] = get_grad_vec(pde_losses[name], all_params)
        g_bc = get_grad_vec(bc_loss, all_params)

        # Cosine monitoring (before surgery)
        cos_val = cosine_sim(grads['RT'], grads['Ru'])
        cosine_hist['RT-Ru'].append(cos_val)
        if cos_val < 0:
            neg_count['RT-Ru'] += 1

        # Magnitude ratio (masking diagnostic)
        norm_RT = grads['RT'].norm().item()
        norm_Ru = grads['Ru'].norm().item()
        mag_r = min(norm_RT, norm_Ru) / (max(norm_RT, norm_Ru) + 1e-30)
        mag_ratio_hist['RT-Ru'].append(mag_r)

        # Surgery
        if args.surgery == 'pcgrad':
            ga, gb = grads['RT'], grads['Ru']
            ga_new, did_a = pcgrad_project(ga, gb)
            gb_new, did_b = pcgrad_project(gb, ga)
            if did_a or did_b:
                surgery_count['RT-Ru'] += 1
            grads['RT'] = ga_new
            grads['Ru'] = gb_new

        # LRA weighting (Wang et al. 2021) — L∞ norm, every step
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

        # GradNorm weighting (Wang et al. 2025) — L2 norm, every gn_update_freq steps
        # Satisfies equal-norm assumption: ||λ_i ∇L_i||_2 = mean for all i
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

        # Apply
        idx = 0
        for p in all_params:
            numel = p.numel()
            p.grad = g_total[idx:idx+numel].reshape(p.shape).clone()
            idx += numel

        backward_times.append(time.time() - t_back)
        optimizer.step()

        # ─── Best-ever eval (every epoch) ────
        t_ev = time.time()
        with torch.no_grad():
            T_pred = nets['T'](x_eval_mon)
            u_pred = nets['u'](x_eval_mon)
            l2_T = torch.norm(T_pred - T_e_mon) / torch.norm(T_e_mon)
            l2_u = torch.norm(u_pred - u_e_mon) / torch.norm(u_e_mon)
            l2_avg = 0.5 * (l2_T.item() + l2_u.item())
        eval_times.append(time.time() - t_ev)
        L2_history.append(l2_avg)

        improved = l2_avg < best_L2_avg
        if improved:
            best_L2_avg = l2_avg
            best_epoch = epoch
            best_state = {n: copy.deepcopy(nets[n].state_dict()) for n in net_names}

        # Logging
        total_loss = sum(pde_losses[n].item() for n in PDE_NAMES) + bc_loss.item()
        for n in PDE_NAMES:
            loss_hist[n].append(pde_losses[n].item())
        loss_hist['BC'].append(bc_loss.item())
        loss_hist['total'].append(total_loss)

        # Best-PDE-loss update (unweighted raw total — no oracle needed)
        if total_loss < best_pde_total:
            best_pde_total = total_loss
            best_pde_epoch = epoch
            best_pde_state = {n: copy.deepcopy(nets[n].state_dict()) for n in net_names}

        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == args.epochs:
            elapsed = (time.time() - t0) / 60
            avg_back = np.mean(backward_times[-LOG_EVERY:]) * 1000
            avg_eval = np.mean(eval_times[-LOG_EVERY:]) * 1000
            neg_pct = 100 * neg_count['RT-Ru'] / epoch

            marker = "[best=" if improved else "(best="
            end_marker = "]" if improved else ")"
            print(f"[{epoch:>6}/{args.epochs}] ({elapsed:5.1f}min) "
                  f"Total={total_loss:.3E}  "
                  f"RT={pde_losses['RT'].item():.1E} "
                  f"Ru={pde_losses['Ru'].item():.1E} "
                  f"BC={bc_loss.item():.1E}")
            print(f"        neg=[RT-Ru:{neg_pct:.0f}%]  "
                  f"mag_r={np.mean(mag_ratio_hist['RT-Ru'][-LOG_EVERY:]):.3f}  "
                  f"back={avg_back:.1f}ms eval={avg_eval:.2f}ms")
            print(f"        L2_avg={l2_avg:.3e}  "
                  f"{marker}{best_L2_avg:.3e} @ ep {best_epoch}{end_marker}")
            if args.weighting == 'lra':
                w_str = ' '.join(f'{n}:{lra_weights[n]:.2f}' for n in ALL_LOSS_NAMES)
                print(f"        lra_w=[{w_str}]")
            elif args.weighting == 'gradnorm':
                w_str = ' '.join(f'{n}:{gn_weights[n]:.2f}' for n in ALL_LOSS_NAMES)
                print(f"        gn_w=[{w_str}]")

    # ─── Load best and evaluate ────
    for n in net_names:
        nets[n].load_state_dict(best_state[n])

    x_eval = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
    with torch.no_grad():
        T_pred = nets['T'](x_eval)
        u_pred = nets['u'](x_eval)
        T_ex = exact_T(x_eval)
        u_ex = exact_u(x_eval)
        l2_T = torch.norm(T_pred - T_ex) / torch.norm(T_ex)
        l2_u = torch.norm(u_pred - u_ex) / torch.norm(u_ex)
        l2_avg = 0.5 * (l2_T.item() + l2_u.item())

    # ─── Load best-PDE-loss and evaluate ────
    if best_pde_state is not None:
        for n in net_names:
            nets[n].load_state_dict(best_pde_state[n])
        with torch.no_grad():
            T_pred_pde = nets['T'](x_eval)
            u_pred_pde = nets['u'](x_eval)
            l2_T_pde = torch.norm(T_pred_pde - T_ex) / torch.norm(T_ex)
            l2_u_pde = torch.norm(u_pred_pde - u_ex) / torch.norm(u_ex)
            l2_avg_pde = 0.5 * (l2_T_pde.item() + l2_u_pde.item())
        # Reload best-L2 state back (for plotting)
        for n in net_names:
            nets[n].load_state_dict(best_state[n])
    else:
        l2_T_pde, l2_u_pde, l2_avg_pde = l2_T, l2_u, l2_avg

    # Final epoch eval
    for n in net_names:
        nets[n].load_state_dict(best_state[n])
    # Reload latest for final eval
    # (need to re-run forward pass with latest weights)
    # Actually we need to save final state too
    # Let me reconsider - we already have L2_history[-1] as final

    final_l2 = L2_history[-1]
    ratio = final_l2 / best_L2_avg if best_L2_avg > 0 else float('inf')

    print()
    print("=" * 70)
    print(f"RESULTS (γ={args.gamma}, {args.surgery}, {args.weighting}, seed={args.seed}, {args.optimizer})")
    print("=" * 70)
    print(f"  === Best-ever (epoch {best_epoch}) [oracle] ===")
    print(f"  L2_T   = {l2_T.item():.4e}")
    print(f"  L2_u   = {l2_u.item():.4e}")
    print(f"  L2_avg = {l2_avg:.4e}")
    print(f"  === Best-PDE-loss (epoch {best_pde_epoch}) [practical] ===")
    print(f"  L2_T   = {l2_T_pde.item():.4e}")
    print(f"  L2_u   = {l2_u_pde.item():.4e}")
    print(f"  L2_avg = {l2_avg_pde:.4e}  (PDE_total={best_pde_total:.4e})")
    print(f"  === Final epoch ({args.epochs}) ===")
    print(f"  L2_avg = {final_l2:.4e}")
    print(f"  === Gap ===")
    print(f"  Final/Best ratio = {ratio:.2f}x")
    print(f"  Avg backward: {np.mean(backward_times)*1000:.2f} ms/step")
    print(f"  Avg L2 eval:  {np.mean(eval_times)*1000:.2f} ms/step (every epoch)")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"  Neg ratio: RT-Ru={100*neg_count['RT-Ru']/args.epochs:.1f}%")
    print(f"  Mag ratio (mean): RT-Ru={np.mean(mag_ratio_hist['RT-Ru']):.4f}")

    # ─── Save ────
    tag = f"thermo_g{args.gamma:.4g}_s{args.seed}_{args.surgery}_{args.weighting}_{args.optimizer}"
    run_dir = os.path.join("runs", f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    results = {
        'L2_T_best': l2_T.item(),
        'L2_u_best': l2_u.item(),
        'L2_avg_best': l2_avg,
        'best_epoch': best_epoch,
        'L2_T_pde_best': l2_T_pde.item(),
        'L2_u_pde_best': l2_u_pde.item(),
        'L2_avg_pde_best': l2_avg_pde,
        'best_pde_epoch': best_pde_epoch,
        'best_pde_total': best_pde_total,
        'L2_avg_final': final_l2,
        'ratio': ratio,
        'neg_count': neg_count,
        'surgery_count': surgery_count,
        'L2_history': L2_history,
        'loss_total': loss_hist['total'],
        'loss_RT': loss_hist['RT'],
        'loss_Ru': loss_hist['Ru'],
        'loss_BC': loss_hist['BC'],
        'cosine_RT_Ru': cosine_hist['RT-Ru'],
        'mag_ratio_RT_Ru': mag_ratio_hist['RT-Ru'],
    }

    config = {
        'kappa': args.kappa, 'E_modulus': args.E_modulus, 'gamma': args.gamma,
        'optimizer': args.optimizer, 'surgery': args.surgery,
        'weighting': args.weighting, 'lra_alpha': args.lra_alpha,
        'seed': args.seed, 'epochs': args.epochs, 'lr': args.lr,
        'N_domain': args.N_domain, 'n_hidden': args.n_hidden,
        'n_neurons': args.n_neurons,
    }

    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump({'config': config, 'results': results}, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"1D Thermoelasticity  |  γ={args.gamma}  |  "
        f"{args.optimizer} | {args.surgery} | seed={args.seed}",
        fontsize=14, fontweight='bold')

    xx = x_eval.cpu().numpy().ravel()

    # T comparison
    ax = axes[0, 0]
    ax.plot(xx, T_ex.cpu().numpy().ravel(), 'k-', lw=2, label='Exact T')
    ax.plot(xx, T_pred.cpu().numpy().ravel(), 'r--', lw=1.5, label=f'PINN T (L2={l2_T.item():.2e})')
    ax.set_title('Temperature T(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # u comparison
    ax = axes[0, 1]
    ax.plot(xx, u_ex.cpu().numpy().ravel(), 'k-', lw=2, label='Exact u')
    ax.plot(xx, u_pred.cpu().numpy().ravel(), 'b--', lw=1.5, label=f'PINN u (L2={l2_u.item():.2e})')
    ax.set_title('Displacement u(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # L2 history
    ax = axes[0, 2]
    ax.semilogy(L2_history, alpha=0.5, lw=0.5)
    ax.axhline(best_L2_avg, color='r', ls='--', lw=1, label=f'Best={best_L2_avg:.2e} @{best_epoch}')
    ax.set_title('L2_avg History')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss history
    ax = axes[1, 0]
    ax.semilogy(loss_hist['RT'], alpha=0.4, lw=0.5, label='RT (heat)')
    ax.semilogy(loss_hist['Ru'], alpha=0.4, lw=0.5, label='Ru (mech)')
    ax.semilogy(loss_hist['BC'], alpha=0.4, lw=0.5, label='BC')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cosine history
    ax = axes[1, 1]
    cos_arr = np.array(cosine_hist['RT-Ru'])
    window = min(500, len(cos_arr)//10)
    if window > 0:
        cos_smooth = np.convolve(cos_arr, np.ones(window)/window, mode='valid')
        ax.plot(cos_smooth, lw=1)
    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.set_title(f'Cosine RT-Ru (neg={100*neg_count["RT-Ru"]/args.epochs:.1f}%)')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)

    # Error distribution
    ax = axes[1, 2]
    with torch.no_grad():
        err_T = (T_pred - T_ex).abs().cpu().numpy().ravel()
        err_u = (u_pred - u_ex).abs().cpu().numpy().ravel()
    ax.semilogy(xx, err_T, 'r-', lw=1, label='|T_err|')
    ax.semilogy(xx, err_u, 'b-', lw=1, label='|u_err|')
    ax.set_title('Pointwise Error (Best Checkpoint)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'summary.png'), dpi=150)
    plt.close()

    print(f"\n  Saved to: {run_dir}/")
    return results


if __name__ == '__main__':
    args = parse_args()
    train(args)
