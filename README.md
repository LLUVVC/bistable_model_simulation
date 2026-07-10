# Bistable Reaction Systems: From Well-Mixed to Spatial Dynamics

Simulation code for my master's thesis *"Bridging Rates Across Scales in Bistable Reaction Systems: From Well-Mixed to Spatial Dynamics"*, and a potential paper, Freie Universität Berlin / Zuse Institute Berlin (ZIB), 2026.

Supervisor: Mauricio del Razo

---

## Overview

This repository implements a multi-scale computational study of the **Schlögl bistable reaction system** — one of the simplest chemical models exhibiting stochastic switching between two stable states.

The project proceeds in three stages:

1. **Well-mixed stochastic simulation** — tau-leaping simulation of the Schlögl model and a derived full bimolecular model; comparison against the numerically solved Chemical Master Equation (CME).
2. **Spatially-resolved particle simulation** — 3D particle-based simulation using the λ–ρ framework (Erban & Chapman, 2009), with Strang operator splitting for reaction–diffusion dynamics.
3. **Discrete-time rate analysis (Model 2)** — numerical solver for the Erban–Chapman discrete-time integral equations, used to analytically predict the effective macroscopic rates achieved by the simulation and explain discretisation-induced distribution shifts.

---

## Background

The Schlögl model undergoes a cubic nonlinearity that produces bistability: two stable steady states coexist, and stochastic fluctuations drive switching between them. To simulate this with spatial resolution, the original trimolecular reaction is decomposed into two sequential bimolecular steps. The Pre-Equilibrium Approximation (PEA) and mass conservation are then used to derive the parameter relations that recover the Schlögl behaviour.

The spatial simulation uses the λ–ρ framework: particles diffuse by Brownian motion and react when within a reactive radius σ of each other. A key finding is that the interaction parameter **γ = s/σ** (where s = √(2(D₁+D₂)τ) is the characteristic diffusion step) plays a significant role in determining the stationary distribution.

---

## Repository Structure

```
bistable_model_simulation/
│
├── simulation/                  # Core simulation library
│   ├── solvers/
│   │   ├── well_mixed_process.py          # Tau-leaping solver for the full well-mixed model
│   │   ├── general_well_mixed_process.py  # Generic tau-leaping (Schlögl model)
│   │   ├── spatial_process.py             # 3D particle-based spatial solver (Strang splitting)
│   │   └── rate_conversions.py            # Macro ↔ micro rate conversions
│   ├── models/
│   │   └── analytical_curve.py            # Analytical stationary distribution (Schlögl CME)
│   └── utils/
│
├── scripts/
│   ├── runners/
│   │   ├── run_well_mixed.py    # Run well-mixed simulation batches
│   │   └── run_spatial.py       # Run spatial simulation batches
│   ├── analysis/
│   │   ├── erban_model2_coupled.py        # Model 2 integral solver (achieved rate computation)
│   │   ├── analyze_distributions.py       # Wasserstein distance & KDE analysis
│   │   ├── wd_convergence_analysis.py     # Wasserstein convergence over batch size
│   │   └── data_loader.py                 # Trajectory data loading utilities
│   └── plotting/                          # Plotting scripts for figures
│
├── notebooks/
│   ├── 01_well_mixed_case_workflow.ipynb          # well-mixed data processing
│   ├── 02_spatial_case_workflow.ipynb             # spatial data processing
│   ├── 03_schloegl_model_wd_convergence.ipynb     # Wasserstein convergence study
│   ├── 04_bifurcation_and_distribution_curve.ipynb # Bifurcation diagrams & sensitivity
│   └── 05_model_description_analytical_curve.ipynb # Analytical curve & Model 2 predictions
│
├── results/                     # Output data (not tracked by git)
├── simulation_data/             # Raw trajectory data (not tracked by git)
├── run_spatial_batch.sh         # Batch script for HPC cluster runs
└── requirements.txt
```

---

## Installation

Python 3.11+ is recommended.

```bash
git clone https://github.com/LLUVVC/bistable_model_simulation.git
cd bistable_model_simulation
pip install -r requirements.txt
```

Key dependencies: `numpy`, `scipy`, `numba`, `matplotlib`, `scikit-learn`, `pandas`, `jupyter`.

---

## Usage

### 1. Well-mixed simulation

Run a batch of tau-leaping trajectories for the full well-mixed model:

```bash
python scripts/runners/run_well_mixed.py
```

### 2. Spatial simulation

Run a batch of 3D particle-based trajectories:

```bash
python scripts/runners/run_spatial.py
```

For HPC/cluster batch submission:

```bash
bash run_spatial_batch.sh
```

### 3. Model 2 — achieved rate computation

Compute the effective macroscopic rates achieved by the discrete-time spatial simulation, given specific κ, τ, and σ values:

```bash
python scripts/analysis/erban_model2_coupled.py
```

This numerically solves the Erban–Chapman discrete-time integral equations and outputs the achieved ℓ₁⁺, ℓ₂⁺, ℓ₂⁻ values, which can then be used to predict the stationary distribution analytically.

### 4. Distribution analysis

Compute Wasserstein distances between simulation output and the analytical Schlögl distribution:

```bash
python scripts/analysis/analyze_distributions.py
python scripts/analysis/wd_convergence_analysis.py
```

---

## Key Results

- The well-mixed full model reproduces the Schlögl stationary distribution with Wasserstein distance W_d ≈ 1.93 (100 trajectories, τ = 10⁻⁶).
- The spatially-resolved simulation shows distribution shifts that are related to the interaction parameter **γ = s/σ**: cases with similar γ produce similar distributions regardless of D and τ individually.
- The discrete-time **Model 2** framework analytically predicts the distribution shift, explaining it as a consequence of finite-timestep discretisation rather than a numerical error.

---

## References

- Erban, R. & Chapman, S.J. (2009). Stochastic modelling of reaction–diffusion processes: algorithms for bimolecular reactions. *Physical Biology*, 6(4), 046001.
- Schlögl, F. (1972). Chemical reaction models for non-equilibrium phase transitions. *Zeitschrift für Physik*, 253(2), 147–161.
- Gillespie, D.T. (1977). Exact stochastic simulation of coupled chemical reactions. *The Journal of Physical Chemistry*, 81(25), 2340–2361.

---

## License

MIT License. See `LICENSE` for details.

---

## Citation

If you use this code, please cite:

```
Ming Lu (2026). Bridging Rates Across Scales in Bistable Reaction Systems:
From Well-Mixed to Spatial Dynamics. Master's thesis,
Freie Universität Berlin / Zuse Institute Berlin.
```
