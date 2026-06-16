"""
solve_erban_model2_coupled_exact.py
===================================
Adapts the Erban & Chapman Model 2 integral equation solver for the 
COUPLED reversible reaction: X2 + A ⇌ X2 + X.

Mathematical Insight
--------------------
Unlike previous scripts that forced g_A(r) + g_X(r) = 2 (which implicitly 
assumed c_A = c_X), this script computes the exact macroscopic rates 
independent of the bulk concentrations. 

It splits the linear concentration fields into two shape functions:
    A(r) = c_A * phi_A(r) + c_X * phi_X(r)
    
We solve two N x N matrix systems independently:
    phi_A has boundary condition = 1 at infinity
    phi_X has boundary condition = 0 at infinity

The resulting macroscopic rates l2+ and l2- are then truly constant
macroscopic observables, completely detached from the local concentration.

Usage
-----
    python solve_erban_model2_coupled_exact.py
"""

import numpy as np
from scipy.special import erf
import time

from simulation.solvers.rate_conversions import calculate_kappas

# ─── Kernel ──────────────────────────────────────────────────────────────

def K(r, rp, gamma):
    """The integral kernel K(r, r'; gamma) from Erban Eq (35)."""
    if r == 0:
        return 0.0
    term1 = np.exp(-((r - rp)**2) / (2 * gamma**2))
    term2 = np.exp(-((r + rp)**2) / (2 * gamma**2))
    return (rp / (r * gamma * np.sqrt(2 * np.pi))) * (term1 - term2)


def build_grid(N1, N2, S=10.0):
    """Construct the piecewise grid from Erban Appendix F.
    N1 points in [0, 1], N2 points in [1, S]."""
    r_grid = np.zeros(N1 + N2)
    for j in range(1, N1 + 1):
        r_grid[j - 1] = j / N1
    for j in range(N1 + 1, N1 + N2 + 1):
        r_grid[j - 1] = 1.0 + (S - 1.0) * (j - N1) / N2
    return r_grid


def build_tail_vector(r_grid, gamma, S=10.0):
    """Compute the analytic tail integral b[i] = ∫_S^∞ K(r_i, r') dr'
    using the exact 'erf' formula from Erban Appendix F."""
    N = len(r_grid)
    b = np.zeros(N)
    for i in range(N):
        r_i = r_grid[i]
        term1 = (gamma**2 * K(r_i, S, gamma)) / S
        term2 = 1.0
        term3 = -0.5 * erf((S - r_i) / (gamma * np.sqrt(2)))
        term4 = -0.5 * erf((S + r_i) / (gamma * np.sqrt(2)))
        b[i] = term1 + term2 + term3 + term4
    return b


# ─── Decoupled solver (original Erban) ──────────────────────────────────

def solve_decoupled(gamma, P_lambda, is_homoreaction=True, S=10.0, N1=4000, N2=2000):
    """
    Standard Erban Model 2: solves one reaction independently.
    Returns r_grid, g, kappa_dimensionless.
    """
    r_grid = build_grid(N1, N2, S)
    N = N1 + N2

    M = np.zeros((N, N))
    for i in range(N):
        r_i = r_grid[i]
        for j in range(N1):
            M[i, j] = ((1.0 - P_lambda) / N1) * K(r_i, r_grid[j], gamma)
        for j in range(N1, N):
            M[i, j] = ((S - 1.0) / N2) * K(r_i, r_grid[j], gamma)

    b = build_tail_vector(r_grid, gamma, S)
    g = np.linalg.solve(np.eye(N) - M, b)

    integral = 0.0
    for j in range(N1):
        integral += 4 * np.pi * (r_grid[j] ** 2) * g[j] * (1.0 / N1)

    if is_homoreaction:
        kappa_val = P_lambda * integral / 2.0
    else:
        kappa_val = P_lambda * integral

    return r_grid, g, kappa_val

# ─── Exact Coupled solver ───────────────────────────────────────────────

def solve_coupled_exact(gamma, P_lambda_plus, P_lambda_minus, S=10.0, N1=4000, N2=2000):
    """
    Coupled Erban Model 2 for X2 + A ⇌ X2 + X using phi_A and phi_X.

    Returns r_grid, phi_A, phi_X, kappa_plus_achieved, kappa_minus_achieved.
    """
    r_grid = build_grid(N1, N2, S)
    N = N1 + N2
    P_total = P_lambda_plus + P_lambda_minus

    M = np.zeros((N, N))
    Kr_ones = np.zeros(N)  # K_r · 1  (kernel integral over reactive zone)

    for i in range(N):
        r_i = r_grid[i]
        for j in range(N1):  # inside reaction radius
            K_val = K(r_i, r_grid[j], gamma)
            w_j = 1.0 / N1
            M[i, j] = (1.0 - P_total) * K_val * w_j
            Kr_ones[i] += K_val * w_j
        for j in range(N1, N):  # outside reaction radius
            M[i, j] = ((S - 1.0) / N2) * K(r_i, r_grid[j], gamma)

    # b implements the boundary condition of 1.0 at infinity
    b = build_tail_vector(r_grid, gamma, S)

    # The source term created by P_lambda_minus
    source = P_lambda_minus * Kr_ones

    # Solve for phi_A (boundary condition = 1)
    rhs_A = b + source
    phi_A = np.linalg.solve(np.eye(N) - M, rhs_A)
    
    # Solve for phi_X (boundary condition = 0, so b is absent)
    rhs_X = source
    phi_X = np.linalg.solve(np.eye(N) - M, rhs_X)

    # Achieved dimensionless kappas
    int_A = 0.0
    int_X = 0.0
    for j in range(N1):
        w_j = 1.0 / N1
        dV_j = 4 * np.pi * r_grid[j] ** 2 * w_j
        
        # Forward net element: (P+ + P-) * phi_A - P-
        int_A += (P_total * phi_A[j] - P_lambda_minus) * dV_j
        
        # Backward net element: P- - (P+ + P-) * phi_X
        int_X += (P_lambda_minus - P_total * phi_X[j]) * dV_j

    kappa_plus = int_A
    kappa_minus = int_X

    return r_grid, phi_A, phi_X, kappa_plus, kappa_minus


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    rho = 0.1       # σ (reaction radius)
    D = 1500.0      # all diffusion coefficients equal
    D_tot = 2 * D   # D_X2 + D_A = D_X2 + D_X = 3000

    # Microscopic rates from the coupled Doi solver
    diffusions = np.ones(4) * 1500
    ls = np.array((1.5, 1500., 150., 25., 5.75, 25.))
    sigmas = np.ones(4) * 0.1
    kappas = calculate_kappas(ls, diffusions[2], diffusions[0], diffusions[1], sigmas)
    kappa_1  = kappas[0]      # R1:  X + X  → X2
    kappa_2p = kappas[2]      # R2f: X2 + A → X2 + X
    kappa_2m = kappas[3]      # R2b: X2 + X → X2 + A

    # Target macroscopic rates
    target_l2p = 150.0
    target_l2m = 25.0
    target_ratio = target_l2p / target_l2m

    # Grid resolution
    N1 = 1000
    N2 = 500

    print("=" * 70)
    print("Erban Model 2: EXACT COUPLED solver (Concentration Independent)")
    print(f"Grid: N1={N1}, N2={N2}")
    print(f"Target: l2+ = {target_l2p}, l2- = {target_l2m}")
    print(f"Target ratio l2+/l2- = {target_ratio:.10f}")
    print("=" * 70)

    for tau in [1e-6, 2e-7]:
        gamma = np.sqrt(2 * D_tot * tau) / rho
        P_1  = 1.0 - np.exp(-kappa_1 * tau)
        P_2p = 1.0 - np.exp(-kappa_2p * tau)
        P_2m = 1.0 - np.exp(-kappa_2m * tau)

        print(f"\n{'─' * 70}")
        print(f"tau = {tau:.1e},  gamma = {gamma:.4f}")
        print(f"P_lambda:  R2f = {P_2p:.6f},  R2b = {P_2m:.6f}")
        print(f"{'─' * 70}")

        # ── R1 DECOUPLED EXACT ──
        t0 = time.time()
        _, _, kd1 = solve_decoupled(gamma, P_1, is_homoreaction=True, N1=N1, N2=N2)
        k1 = kd1 * rho**3 / tau
        print(f"  R1 (X+X→X2):  k_macro = {k1:.6f}  [{time.time()-t0:.1f}s]")

        # ── R2 COUPLED EXACT ──
        t0 = time.time()
        _, phi_A, phi_X, kc2p, kc2m = solve_coupled_exact(gamma, P_2p, P_2m, N1=N1, N2=N2)
        l2p_cpl = kc2p * rho**3 / tau
        l2m_cpl = kc2m * rho**3 / tau
        ratio_cpl = l2p_cpl / l2m_cpl
        
        print(f"  EXACT COUPLED RATES:    [{time.time()-t0:.1f}s]")
        print(f"    Forward  l2+ = {l2p_cpl:.6f}")
        print(f"    Reverse  l2- = {l2m_cpl:.6f}")
        print(f"    Ratio        = {ratio_cpl:.10f}")


if __name__ == '__main__':
    main()