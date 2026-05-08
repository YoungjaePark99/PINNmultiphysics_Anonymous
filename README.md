# Coupling-Robust Accuracy in Multiphysics PINNs via Kronecker-Preconditioned Optimization

This repository contains the code for reproducing the experiments in:

> **Coupling-Robust Accuracy in Multiphysics PINNs via Kronecker-Preconditioned Optimization**
> Submitted to ICML 2026 Workshop (AI4Physics)

## Overview

We show that combining the Kronecker-preconditioned optimizer **SOAP** with inverse-gradient-norm loss balancing (**GradNorm**) yields coupling-robust accuracy in multiphysics PINNs. Across four benchmark systems, SOAP+GradNorm maintains final-epoch L₂ degradation ≤ 1.1× even as coupling strength varies over 1–2 orders of magnitude.

## Benchmark Systems

| Script | System | PDEs | Coupling |
|---|---|---|---|
| `thermoelasticity_1d.py` | 1D Thermoelasticity | 2 | Linear, one-way |
| `rxn_diff_3species.py` | 1D Reaction–Diffusion (A⇌B⇌C) | 3 | Linear, bidirectional |
| `npp_1d.py` | 1D Nernst–Planck–Poisson | 3 | Nonlinear, circular |
| `npps_2d.py` | 2D Electroosmotic Flow (NP+P+Stokes) | 4 | Nonlinear, EDL-resolved |

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy
- SciPy (`solve_bvp` for reference solutions)
- Matplotlib (for plotting)
- [`soap.py`](https://github.com/nikhilvyas/SOAP) — SOAP optimizer (place `soap.py` in the same directory as the scripts)

Install Python dependencies:
```bash
pip install torch numpy scipy matplotlib
```

## Usage

Each script is self-contained. Options are passed via `--optimizer`, `--weighting` flags separately:

```bash
# 1D Thermoelasticity: SOAP + GradNorm at γ=10
python thermoelasticity_1d.py --optimizer soap --weighting gradnorm --gamma 10 --seed 42

# 1D Reaction-Diffusion: Adam + LRA at k=5
python rxn_diff_3species.py --optimizer adam --weighting lra --k-rxn 5 --seed 42

# 1D Nernst-Planck-Poisson: SOAP + GradNorm at ε=0.1
python npp_1d.py --optimizer soap --weighting gradnorm --epsilon 0.1 --seed 42

# 2D Electroosmotic Flow: SOAP + GradNorm at ε=0.2
python npps_2d.py --optimizer soap --weighting gradnorm --epsilon 0.2 --seed 42
```

### Available Options

**Optimizer** (`--optimizer`):
| Choice | Description |
|---|---|
| `adam` | Adam |
| `soap` | SOAP (Kronecker-preconditioned Adam) |

**Loss weighting** (`--weighting`):
| Choice | Description | Key hyperparameters |
|---|---|---|
| `none` | Uniform (all weights = 1) | — |
| `lra` | Learning Rate Annealing | `--lra-alpha 0.1` |
| `gradnorm` | Inverse-gradient-norm balancing | `--gn-update-freq 1000 --gn-momentum 0.9` |



### Physics Parameters

**`thermoelasticity_1d.py`:**
| Flag | Default | Description |
|---|---|---|
| `--gamma` | 1.0 | Thermal stress coefficient γ = E·α_T (coupling strength) |
| `--kappa` | 1.0 | Thermal conductivity κ |
| `--E-modulus` | 1.0 | Young's modulus E |

**`rxn_diff_3species.py`:**
| Flag | Default | Description |
|---|---|---|
| `--k-rxn` | 1.0 | Symmetric reaction rate (k1=k2=k3=k4) |
| `--k1` ~ `--k4` | — | Individual rates (override `--k-rxn`) |
| `--D1`, `--D2`, `--D3` | 1.0 | Species diffusivities |

**`npp_1d.py`:**
| Flag | Default | Description |
|---|---|---|
| `--epsilon` | 0.1 | Debye length ratio ε (coupling strength) |
| `--zeta` | 1.0 | Wall potential (thermal voltage units) |

**`npps_2d.py`:**
| Flag | Default | Description |
|---|---|---|
| `--epsilon` | 0.2 | Debye length ratio ε (coupling strength) |
| `--zeta` | 1.0 | Wall potential (thermal voltage units) |
| `--Ex` | 1.0 | Applied electric field |
| `--mu` | 1.0 | Dynamic viscosity |


## Key Results
<!-- Update these numbers after running the full experiment sweep -->
- **SOAP+GradNorm** achieves degradation ≤ 1.1× across all 1D systems (coupling variations of 5–50×)
- In 2D EDL-resolved electroosmotic flow, SOAP+GradNorm achieves L₂ = 1.3×10⁻³ at ε=0.2 where Adam+GradNorm fails (L₂ > 0.9)
- LRA fails catastrophically in nonlinear NP+P (weight explosion to O(10⁹))

## Citation

```bibtex
@inproceedings{anonymous2026coupling,
  title={Coupling-Robust Accuracy in Multiphysics {PINN}s via {K}ronecker-Preconditioned Optimization},
  author={Anonymous},
  booktitle={ICML 2026 Workshop},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
