"""
Reaction-Diffusion Tier 1: 1D 3-Species A ⇌ B ⇌ C
=====================================================
Reversible reaction-diffusion with linear kinetics (MMS).

PDEs:
  R1: D1·c_A,xx - k1·c_A + k2·c_B                     = f1
  R2: D2·c_B,xx + k1·c_A - (k2+k3)·c_B + k4·c_C      = f2
  R3: D3·c_C,xx + k3·c_B - k4·c_C                     = f3

Exact solutions (MMS):
  c_A = sin(πx) + 2
  c_B = cos(πx) + 2
  c_C = sin(2πx) + 2

Best-ever checkpoint tracking:
  L2 error is computed EVERY epoch on a monitoring grid (following
  AL-PINNs convention from Son et al. 2023 — see github.com/HwijaeSon/AL-PINNs).
  Whenever L2_avg improves, the model state_dicts are deep-copied.
  Final evaluation is performed after loading the best state.
  Print/log is throttled to every LOG_EVERY epochs to avoid I/O overhead.

Usage:
  python rxn_diff_3species.py --k-rxn 5 --surgery none
  python rxn_diff_3species.py --k-rxn 5 --surgery pcgrad
  python rxn_diff_3species.py --k-rxn 5 --surgery physics_aware
  python rxn_diff_3species.py --k-rxn 10 --optimizer soap --surgery physics_aware
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
    p = argparse.ArgumentParser(description='Reaction-Diffusion 3-Species A⇌B⇌C')
    # Physics
    p.add_argument('--D1', type=float, default=1.0, help='Diffusivity of A')
    p.add_argument('--D2', type=float, default=1.0, help='Diffusivity of B')
    p.add_argument('--D3', type=float, default=1.0, help='Diffusivity of C')
    p.add_argument('--k-rxn', type=float, default=1.0,
                   help='Symmetric reaction rate k=k1=k2=k3=k4')
    p.add_argument('--k1', type=float, default=None, help='A→B rate (overrides k-rxn)')
    p.add_argument('--k2', type=float, default=None, help='B→A rate (overrides k-rxn)')
    p.add_argument('--k3', type=float, default=None, help='B→C rate (overrides k-rxn)')
    p.add_argument('--k4', type=float, default=None, help='C→B rate (overrides k-rxn)')
    # Training
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=60000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--N-domain', type=int, default=200)
    p.add_argument('--N-bc', type=int, default=2)
    # Architecture
    p.add_argument('--n-hidden', type=int, default=4)
    p.add_argument('--n-neurons', type=int, default=64)
    # Optimizer
    p.add_argument('--optimizer', choices=['adam', 'soap'], default='adam')
    p.add_argument('--soap-beta1', type=float, default=0.99,
                   help='SOAP beta1 (Wang et al. 2025 PINN-optimal: 0.99)')
    p.add_argument('--soap-beta2', type=float, default=0.999,
                   help='SOAP beta2 (Wang et al. 2025 PINN-optimal: 0.999)')
    p.add_argument('--soap-precond-freq', type=int, default=2,
                   help='SOAP precondition frequency (Wang et al. 2025: 2)')
    # Surgery
    p.add_argument('--surgery', choices=['none', 'pcgrad', 'physics_aware'],
                   default='none')
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


def resolve_rates(args):
    """Set individual rates from symmetric k-rxn if not overridden."""
    k = args.k_rxn
    args.k1_val = args.k1 if args.k1 is not None else k
    args.k2_val = args.k2 if args.k2 is not None else k
    args.k3_val = args.k3 if args.k3 is not None else k
    args.k4_val = args.k4 if args.k4 is not None else k


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
def exact_cA(x):
    return torch.sin(PI * x) + 2.0

def exact_cB(x):
    return torch.cos(PI * x) + 2.0

def exact_cC(x):
    return torch.sin(2 * PI * x) + 2.0

def compute_sources(x, args):
    """Source terms from MMS: f_k = PDE_k(exact)."""
    k1, k2, k3, k4 = args.k1_val, args.k2_val, args.k3_val, args.k4_val
    sx, cx, s2x = torch.sin(PI*x), torch.cos(PI*x), torch.sin(2*PI*x)
    cA, cB, cC = sx + 2.0, cx + 2.0, s2x + 2.0

    lap_cA = -PI**2 * sx
    lap_cB = -PI**2 * cx
    lap_cC = -4*PI**2 * s2x

    f1 = args.D1 * lap_cA - k1 * cA + k2 * cB
    f2 = args.D2 * lap_cB + k1 * cA - (k2 + k3) * cB + k4 * cC
    f3 = args.D3 * lap_cC + k3 * cB - k4 * cC

    return f1.detach(), f2.detach(), f3.detach()


# ══════════════════════════════════════════════════════════════════════
# PDE residuals
# ══════════════════════════════════════════════════════════════════════
def compute_residuals(nets, x, src, args):
    k1, k2, k3, k4 = args.k1_val, args.k2_val, args.k3_val, args.k4_val

    cA = nets['cA'](x)
    cB = nets['cB'](x)
    cC = nets['cC'](x)

    cA_x = torch.autograd.grad(cA, x, torch.ones_like(cA),
                                create_graph=True, retain_graph=True)[0]
    cA_xx = torch.autograd.grad(cA_x, x, torch.ones_like(cA_x),
                                 create_graph=True, retain_graph=True)[0]

    cB_x = torch.autograd.grad(cB, x, torch.ones_like(cB),
                                create_graph=True, retain_graph=True)[0]
    cB_xx = torch.autograd.grad(cB_x, x, torch.ones_like(cB_x),
                                 create_graph=True, retain_graph=True)[0]

    cC_x = torch.autograd.grad(cC, x, torch.ones_like(cC),
                                create_graph=True, retain_graph=True)[0]
    cC_xx = torch.autograd.grad(cC_x, x, torch.ones_like(cC_x),
                                 create_graph=True, retain_graph=True)[0]

    f1, f2, f3 = src

    R1 = args.D1 * cA_xx - k1 * cA + k2 * cB - f1
    R2 = args.D2 * cB_xx + k1 * cA - (k2 + k3) * cB + k4 * cC - f2
    R3 = args.D3 * cC_xx + k3 * cB - k4 * cC - f3

    return {
        'R1': torch.mean(R1**2),
        'R2': torch.mean(R2**2),
        'R3': torch.mean(R3**2),
    }


def compute_bc_loss(nets, args, device):
    x0 = torch.zeros(1, 1, device=device)
    x1 = torch.ones(1, 1, device=device)

    loss = 0.0
    for x_bc in [x0, x1]:
        loss += (nets['cA'](x_bc) - exact_cA(x_bc))**2
        loss += (nets['cB'](x_bc) - exact_cB(x_bc))**2
        loss += (nets['cC'](x_bc) - exact_cC(x_bc))**2
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
    resolve_rates(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"k1={args.k1_val}, k2={args.k2_val}, k3={args.k3_val}, k4={args.k4_val}")
    print(f"Surgery: {args.surgery}")
    print(f"Weighting: {args.weighting}" +
          (f" (α={args.lra_alpha})" if args.weighting == 'lra' else "") +
          (f" (freq={args.gn_update_freq}, mom={args.gn_momentum})" if args.weighting == 'gradnorm' else ""))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Networks
    net_names = ['cA', 'cB', 'cC']
    nets = {n: SubNet(args.n_hidden, args.n_neurons).to(device) for n in net_names}

    all_params = []
    for n in net_names:
        all_params.extend(list(nets[n].parameters()))

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(all_params, lr=args.lr)
    elif args.optimizer == 'soap':
        optimizer = SOAP(
            all_params,
            lr=args.lr,
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
    print(f"Source magnitudes: |f1|={src_mags[0]:.2e}, |f2|={src_mags[1]:.2e}, |f3|={src_mags[2]:.2e}")

    # ─── Best-ever checkpoint tracking (by L2 on monitoring grid) ────
    # Following AL-PINNs (Son et al. 2023) convention: evaluate every epoch.
    best_L2_avg = float('inf')
    best_epoch = -1
    best_state = None

    # ─── Best-PDE-loss checkpoint tracking (no oracle needed) ────────
    best_pde_total = float('inf')
    best_pde_epoch = -1
    best_pde_state = None
    L2_history = []
    x_eval_mon = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
    with torch.no_grad():
        cA_e_mon = exact_cA(x_eval_mon)
        cB_e_mon = exact_cB(x_eval_mon)
        cC_e_mon = exact_cC(x_eval_mon)

    # Monitoring
    PDE_NAMES = ['R1', 'R2', 'R3']
    ALL_PAIRS = [('R1','R2'), ('R1','R3'), ('R2','R3')]
    SURGERY_PAIRS_PA = [('R1','R2'), ('R2','R3')]  # negative feedback only

    loss_hist = {k: [] for k in PDE_NAMES + ['BC', 'total']}
    cosine_hist = {f'{a}-{b}': [] for a, b in ALL_PAIRS}
    neg_count = {f'{a}-{b}': 0 for a, b in ALL_PAIRS}
    surgery_count = {f'{a}-{b}': 0 for a, b in ALL_PAIRS}
    mag_ratio_hist = {f'{a}-{b}': [] for a, b in ALL_PAIRS}
    backward_times = []
    eval_times = []

    # Adaptive weights (shared structure for LRA and GradNorm)
    ALL_LOSS_NAMES = PDE_NAMES + ['BC']
    lra_weights = {n: 1.0 for n in ALL_LOSS_NAMES}
    lra_weight_hist = {n: [] for n in ALL_LOSS_NAMES}

    # GradNorm weights (separate from LRA)
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
        for a, b in ALL_PAIRS:
            key = f'{a}-{b}'
            cos_val = cosine_sim(grads[a], grads[b])
            cosine_hist[key].append(cos_val)
            if cos_val < 0:
                neg_count[key] += 1
            # Magnitude ratio (masking diagnostic)
            norm_a = grads[a].norm().item()
            norm_b = grads[b].norm().item()
            mag_r = min(norm_a, norm_b) / (max(norm_a, norm_b) + 1e-30)
            mag_ratio_hist[key].append(mag_r)

        # Surgery
        if args.surgery == 'pcgrad':
            target_pairs = ALL_PAIRS
        elif args.surgery == 'physics_aware':
            target_pairs = SURGERY_PAIRS_PA
        else:
            target_pairs = []

        for a, b in target_pairs:
            key = f'{a}-{b}'
            ga, gb = grads[a], grads[b]
            ga_new, did_a = pcgrad_project(ga, gb)
            gb_new, did_b = pcgrad_project(gb, ga)
            if did_a or did_b:
                surgery_count[key] += 1
            grads[a] = ga_new
            grads[b] = gb_new

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

        # Log loss
        with torch.no_grad():
            vals = {k: pde_losses[k].item() for k in PDE_NAMES}
            vals['BC'] = bc_loss.item()
            vals['total'] = sum(vals.values())
        for k in vals:
            loss_hist[k].append(vals[k])

        # Best-PDE-loss update (unweighted raw total — no oracle needed)
        raw_total = vals['total']
        if raw_total < best_pde_total:
            best_pde_total = raw_total
            best_pde_epoch = epoch
            best_pde_state = {
                'cA': copy.deepcopy(nets['cA'].state_dict()),
                'cB': copy.deepcopy(nets['cB'].state_dict()),
                'cC': copy.deepcopy(nets['cC'].state_dict()),
                'pde_total': raw_total,
                'epoch': epoch,
            }

        # ═══════════════════════════════════════════════════════════
        # Evaluate L2 EVERY epoch (AL-PINNs convention)
        # ═══════════════════════════════════════════════════════════
        t_eval = time.time()
        with torch.no_grad():
            cA_p_mon = nets['cA'](x_eval_mon)
            cB_p_mon = nets['cB'](x_eval_mon)
            cC_p_mon = nets['cC'](x_eval_mon)
            L2_cA_ep = torch.sqrt(torch.mean((cA_p_mon - cA_e_mon)**2)).item()
            L2_cB_ep = torch.sqrt(torch.mean((cB_p_mon - cB_e_mon)**2)).item()
            L2_cC_ep = torch.sqrt(torch.mean((cC_p_mon - cC_e_mon)**2)).item()
            L2_avg_ep = (L2_cA_ep + L2_cB_ep + L2_cC_ep) / 3
        eval_times.append(time.time() - t_eval)

        # Store L2 history (every epoch)
        L2_history.append({
            'epoch': epoch,
            'cA': L2_cA_ep,
            'cB': L2_cB_ep,
            'cC': L2_cC_ep,
            'avg': L2_avg_ep,
        })

        # Best-ever update (every epoch)
        if L2_avg_ep < best_L2_avg:
            best_L2_avg = L2_avg_ep
            best_epoch = epoch
            best_state = {
                'cA': copy.deepcopy(nets['cA'].state_dict()),
                'cB': copy.deepcopy(nets['cB'].state_dict()),
                'cC': copy.deepcopy(nets['cC'].state_dict()),
                'L2_cA': L2_cA_ep,
                'L2_cB': L2_cB_ep,
                'L2_cC': L2_cC_ep,
                'L2_avg': L2_avg_ep,
                'epoch': epoch,
            }

        # ═══════════════════════════════════════════════════════════
        # Print only every LOG_EVERY epochs
        # ═══════════════════════════════════════════════════════════
        if epoch % LOG_EVERY == 0 or epoch == 1:
            elapsed = (time.time() - t0) / 60
            avg_back = 1000 * np.mean(backward_times[-LOG_EVERY:])
            avg_eval = 1000 * np.mean(eval_times[-LOG_EVERY:])

            neg_strs = []
            for a, b in ALL_PAIRS:
                key = f'{a}-{b}'
                pct = 100 * neg_count[key] / epoch
                neg_strs.append(f"{key}:{pct:.0f}%")

            print(f"[{epoch:6d}/{args.epochs}] ({elapsed:5.1f}min) "
                  f"Total={vals['total']:.3E}  "
                  f"R1={vals['R1']:.1E} R2={vals['R2']:.1E} R3={vals['R3']:.1E} "
                  f"BC={vals['BC']:.1E}")
            print(f"        neg=[{', '.join(neg_strs)}]  "
                  f"back={avg_back:.1f}ms eval={avg_eval:.2f}ms")
            mag_strs = []
            for a, b in ALL_PAIRS:
                key = f'{a}-{b}'
                mr = np.mean(mag_ratio_hist[key][-LOG_EVERY:])
                mag_strs.append(f"{key}:{mr:.3f}")
            print(f"        mag_r=[{', '.join(mag_strs)}]")
            # Show best status
            if best_epoch == epoch or (epoch - best_epoch) < LOG_EVERY:
                # Best was updated within the last LOG_EVERY window
                print(f"        L2_avg={L2_avg_ep:.3e}  "
                      f"[best={best_L2_avg:.3e} @ ep {best_epoch}]")
            else:
                print(f"        L2_avg={L2_avg_ep:.3e}  "
                      f"(best={best_L2_avg:.3e} @ ep {best_epoch})")
            if args.weighting == 'lra':
                w_str = ' '.join(f'{n}:{lra_weights[n]:.2f}' for n in ALL_LOSS_NAMES)
                print(f"        lra_w=[{w_str}]")
            elif args.weighting == 'gradnorm':
                w_str = ' '.join(f'{n}:{gn_weights[n]:.2f}' for n in ALL_LOSS_NAMES)
                print(f"        gn_w=[{w_str}]")

    elapsed_total = (time.time() - t0) / 60

    # ─── Final-epoch L2 (current network state, 500-pt grid) ─────────
    x_eval = torch.linspace(0, 1, 500, device=device).reshape(-1, 1)
    with torch.no_grad():
        cA_ex = exact_cA(x_eval)
        cB_ex = exact_cB(x_eval)
        cC_ex = exact_cC(x_eval)

        cA_pr_final = nets['cA'](x_eval)
        cB_pr_final = nets['cB'](x_eval)
        cC_pr_final = nets['cC'](x_eval)

        L2_final = {}
        for name, pred, exact in [('cA', cA_pr_final, cA_ex),
                                   ('cB', cB_pr_final, cB_ex),
                                   ('cC', cC_pr_final, cC_ex)]:
            L2_final[name] = torch.sqrt(torch.mean((pred - exact)**2)).item()

    # ─── Best-ever L2 (load best state, evaluate on 500-pt grid) ────
    if best_state is not None:
        nets['cA'].load_state_dict(best_state['cA'])
        nets['cB'].load_state_dict(best_state['cB'])
        nets['cC'].load_state_dict(best_state['cC'])

        with torch.no_grad():
            cA_pr = nets['cA'](x_eval)
            cB_pr = nets['cB'](x_eval)
            cC_pr = nets['cC'](x_eval)

            L2 = {}
            for name, pred, exact in [('cA', cA_pr, cA_ex),
                                       ('cB', cB_pr, cB_ex),
                                       ('cC', cC_pr, cC_ex)]:
                L2[name] = torch.sqrt(torch.mean((pred - exact)**2)).item()
    else:
        print("[WARN] No best checkpoint recorded; using final-epoch state.")
        L2 = L2_final
        cA_pr, cB_pr, cC_pr = cA_pr_final, cB_pr_final, cC_pr_final
        best_epoch = args.epochs

    # ─── Best-PDE-loss L2 (load best-pde state, evaluate on 500-pt grid) ─
    if best_pde_state is not None:
        nets['cA'].load_state_dict(best_pde_state['cA'])
        nets['cB'].load_state_dict(best_pde_state['cB'])
        nets['cC'].load_state_dict(best_pde_state['cC'])

        with torch.no_grad():
            cA_pr_pde = nets['cA'](x_eval)
            cB_pr_pde = nets['cB'](x_eval)
            cC_pr_pde = nets['cC'](x_eval)

            L2_pde = {}
            for name, pred, ex in [('cA', cA_pr_pde, cA_ex),
                                    ('cB', cB_pr_pde, cB_ex),
                                    ('cC', cC_pr_pde, cC_ex)]:
                L2_pde[name] = torch.sqrt(torch.mean((pred - ex)**2)).item()

        # Reload best-L2 state back (for plotting)
        if best_state is not None:
            nets['cA'].load_state_dict(best_state['cA'])
            nets['cB'].load_state_dict(best_state['cB'])
            nets['cC'].load_state_dict(best_state['cC'])
    else:
        L2_pde = L2_final

    avg_backward_ms = 1000 * np.mean(backward_times)
    avg_eval_ms = 1000 * np.mean(eval_times)

    print(f"\n{'='*70}")
    print(f"RESULTS (k={args.k_rxn}, {args.surgery}, {args.weighting}, seed={args.seed}, {args.optimizer})")
    print(f"{'='*70}")
    print(f"  === Best-ever (epoch {best_epoch}) [oracle] ===")
    for name in ['cA', 'cB', 'cC']:
        print(f"  L2_{name} = {L2[name]:.4e}")
    print(f"  L2_avg  = {(L2['cA']+L2['cB']+L2['cC'])/3:.4e}")
    pde_avg = (L2_pde['cA'] + L2_pde['cB'] + L2_pde['cC']) / 3
    print(f"  === Best-PDE-loss (epoch {best_pde_epoch}) [practical] ===")
    for name in ['cA', 'cB', 'cC']:
        print(f"  L2_{name} = {L2_pde[name]:.4e}")
    print(f"  L2_avg  = {pde_avg:.4e}  (PDE_total={best_pde_total:.4e})")
    print(f"  === Final epoch ({args.epochs}) ===")
    for name in ['cA', 'cB', 'cC']:
        print(f"  L2_{name} = {L2_final[name]:.4e}")
    print(f"  L2_avg  = {(L2_final['cA']+L2_final['cB']+L2_final['cC'])/3:.4e}")
    best_avg = (L2['cA'] + L2['cB'] + L2['cC']) / 3
    final_avg = (L2_final['cA'] + L2_final['cB'] + L2_final['cC']) / 3
    gap = final_avg / max(best_avg, 1e-30)
    print(f"  === Gap ===")
    print(f"  Final/Best ratio = {gap:.2f}x")
    print(f"  Avg backward: {avg_backward_ms:.2f} ms/step")
    print(f"  Avg L2 eval:  {avg_eval_ms:.2f} ms/step (every epoch)")
    print(f"  Total time: {elapsed_total:.1f} min")
    print(f"  Neg ratios: " + ", ".join(
        f"{k}={100*neg_count[k]/args.epochs:.1f}%"
        for k in neg_count))
    print(f"  Mag ratios (mean): " + ", ".join(
        f"{k}={np.mean(mag_ratio_hist[k]):.4f}"
        for k in mag_ratio_hist))

    # Save
    results = {
        'args': {k: v for k, v in vars(args).items()
                 if not k.startswith('k') or k in ['k_rxn','k1_val','k2_val','k3_val','k4_val']},
        'L2': L2,                     # Best-ever L2 (oracle — upper bound)
        'L2_pde_best': L2_pde,        # Best-PDE-loss L2 (practical metric)
        'L2_final_epoch': L2_final,   # Final-epoch L2 (stability metric)
        'best_epoch': best_epoch,
        'best_pde_epoch': best_pde_epoch,
        'best_pde_total': best_pde_total,
        'L2_history': L2_history,
        'timing': {
            'avg_backward_ms': avg_backward_ms,
            'avg_eval_ms': avg_eval_ms,
            'total_min': elapsed_total,
        },
        'neg_count': neg_count,
        'surgery_count': surgery_count,
        'loss_hist': loss_hist,
        'cosine_hist': cosine_hist,
        'mag_ratio_hist': {k: v for k, v in mag_ratio_hist.items()},
    }

    tag = f"rxndiff_k{args.k_rxn:.4g}_s{args.seed}_{args.surgery}_{args.weighting}_{args.optimizer}"
    run_dir = os.path.join('runs', tag)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if best_state is not None:
        torch.save(best_state, os.path.join(run_dir, 'best_checkpoint.pt'))
    if best_pde_state is not None:
        torch.save(best_pde_state, os.path.join(run_dir, 'best_pde_checkpoint.pt'))

    # Plot (uses best state which is currently loaded)
    plot_results(results, x_eval.cpu().numpy(),
                 {n: v.cpu().numpy() for n, v in zip(['cA','cB','cC'], [cA_pr, cB_pr, cC_pr])},
                 {n: v.cpu().numpy() for n, v in zip(['cA','cB','cC'], [cA_ex, cB_ex, cC_ex])},
                 run_dir, args)

    return results


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════
def plot_results(results, x, pred, exact, run_dir, args):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Row 1: Solutions (best-ever state)
    colors = {'cA': 'tab:red', 'cB': 'tab:blue', 'cC': 'tab:green'}
    labels = {'cA': '$c_A$', 'cB': '$c_B$', 'cC': '$c_C$'}
    L2 = results['L2']
    best_ep = results['best_epoch']

    for i, name in enumerate(['cA', 'cB', 'cC']):
        ax = axes[0, i]
        ax.plot(x, exact[name], 'k-', lw=2, label='Exact')
        ax.plot(x, pred[name], '--', color=colors[name], lw=1.5,
                label=f'PINN (best@{best_ep})')
        ax.set_title(f'{labels[name]}  (L2_best={L2[name]:.2e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2: Loss history, L2 history, summary
    ax = axes[1, 0]
    lh = results['loss_hist']
    for key in ['R1', 'R2', 'R3']:
        ax.semilogy(lh[key], label=key, alpha=0.7)
    ax.set_title('PDE Losses')
    ax.set_xlabel('epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # L2 history + best epoch marker
    ax = axes[1, 1]
    L2h = results['L2_history']
    if len(L2h) > 0:
        # L2 history is now every epoch, so downsample for plotting
        stride = max(1, len(L2h) // 500)
        L2h_plot = L2h[::stride]
        eps = [e['epoch'] for e in L2h_plot]
        for name in ['cA', 'cB', 'cC']:
            ax.semilogy(eps, [e[name] for e in L2h_plot], label=name,
                        color=colors[name], alpha=0.7)
        ax.semilogy(eps, [e['avg'] for e in L2h_plot], 'k-', label='avg', lw=1.5)
        ax.axvline(best_ep, color='red', linestyle=':', alpha=0.6,
                   label=f'best ep={best_ep}')
    ax.set_title('L2 history (every epoch, stride-downsampled)')
    ax.set_xlabel('epoch')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    info = [
        f"System: A ⇌ B ⇌ C",
        f"k = {args.k_rxn}",
        f"optimizer: {args.optimizer}",
        f"surgery: {args.surgery}",
        f"seed: {args.seed}",
        f"back: {results['timing']['avg_backward_ms']:.1f} ms",
        f"eval: {results['timing']['avg_eval_ms']:.2f} ms",
        f"time: {results['timing']['total_min']:.1f} min",
        "",
        f"=== BEST (ep {best_ep}) ===",
    ]
    for name in ['cA', 'cB', 'cC']:
        info.append(f"L2_{name} = {L2[name]:.3e}")
    info.append("")
    info.append(f"=== FINAL (ep {args.epochs}) ===")
    L2f = results['L2_final_epoch']
    for name in ['cA', 'cB', 'cC']:
        info.append(f"L2_{name} = {L2f[name]:.3e}")
    info.append("")
    nc = results['neg_count']
    total = args.epochs
    for key in nc:
        info.append(f"neg {key}: {100*nc[key]/total:.1f}%")
    info.append("")
    sc = results['surgery_count']
    for key in sc:
        if sc[key] > 0:
            info.append(f"surgery {key}: {sc[key]}")
    ax.text(0.05, 0.95, '\n'.join(info), transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace')
    ax.axis('off')
    ax.set_title('Summary')

    fig.suptitle(f"Reaction-Diffusion A⇌B⇌C | k={args.k_rxn} | "
                 f"{args.optimizer} | {args.surgery} | seed={args.seed}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'results.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)
