"""
solve_erban_model2_coupled.py
==============================
Adapts the Erban & Chapman Model 2 integral equation solver for the 
COUPLED reversible reaction: X2 + A ⇌ X2 + X.

Compares the DECOUPLED and COUPLED achieved macroscopic rates to determine
whether the coupling affects the l2+/l2- ratio differently at different τ.

Mathematical Insight
--------------------
In the decoupled Erban model 2, each reaction is verified independently:
one test particle diffuses near one reactive target.

In the coupled version, BOTH reactions happen simultaneously around X2:
  - A particle near X2 can react (prob P_λ⁺) and become X
  - X particle near X2 can react (prob P_λ⁻) and become A

The coupled integral equations are:

    g_A(r) = ∫₀¹ [(1-P⁺)g_A(r') + P⁻ g_X(r')] K dr' + ∫₁^∞ g_A K dr'
    g_X(r) = ∫₀¹ [(1-P⁻)g_X(r') + P⁺ g_A(r')] K dr' + ∫₁^∞ g_X K dr'

Key simplification: since reactions conserve total particle count (A ↔ X),
    h(r) = g_A(r) + g_X(r) = 2  (everywhere)

This reduces the 2N×2N system to a single N×N system:
    (I - M_total) g_A = b + 2·P_λ⁻·K_r·1
    g_X = 2 - g_A

where M_total uses P_total = P_λ⁺ + P_λ⁻ as the combined reaction probability.

Usage
-----
    python -m scripts.analysis.solve_erban_model2_coupled
"""

import numpy as np
from scipy.special import erf
import time


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


# ─── Coupled solver (new) ───────────────────────────────────────────────

def solve_coupled(gamma, P_lambda_plus, P_lambda_minus, S=10.0, N1=4000, N2=2000):
    """
    Coupled Erban Model 2 for X2 + A ⇌ X2 + X.

    Since reactions conserve total particle number (A ↔ X conversion),
    h(r) = g_A(r) + g_X(r) = 2 everywhere. Substituting g_X = 2 - g_A
    into the coupled integral equations yields a single N×N system:

        (I - M_total) g_A = b + 2·P⁻·K_r·1

    where M_total uses P_total = P⁺ + P⁻ as the combined reaction probability,
    and K_r·1 is the kernel integral over the reactive zone evaluated at g=1.

    NOTE: This reduction requires D_A = D_X (same gamma for both species).

    Returns r_grid, g_A, g_X, kappa_plus_achieved, kappa_minus_achieved.
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

    b = build_tail_vector(r_grid, gamma, S)

    # Modified RHS: the extra term accounts for X→A conversion near X2
    rhs = b + 2.0 * P_lambda_minus * Kr_ones

    g_A = np.linalg.solve(np.eye(N) - M, rhs)
    g_X = 2.0 - g_A

    # Achieved dimensionless kappas (heteroreaction, no 1/2 factor)
    int_A = 0.0
    int_X = 0.0
    for j in range(N1):
        w_j = 1.0 / N1
        int_A += 4 * np.pi * r_grid[j] ** 2 * g_A[j] * w_j
        int_X += 4 * np.pi * r_grid[j] ** 2 * g_X[j] * w_j

    kappa_plus = P_lambda_plus * int_A    # Forward: A reacts near X2
    kappa_minus = P_lambda_minus * int_X   # Reverse: X reacts near X2

    return r_grid, g_A, g_X, kappa_plus, kappa_minus


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    rho = 0.1       # σ (reaction radius)
    D = 1500.0      # all diffusion coefficients equal
    D_tot = 2 * D   # D_X2 + D_A = D_X2 + D_X = 3000

    # Microscopic rates from the coupled Doi solver (rate_conversions.py)
    kappa_1  = 716.88       # R1:  X + X  → X2
    kappa_2p = 3.7921e4     # R2f: X2 + A → X2 + X
    kappa_2m = 6.3201e3     # R2b: X2 + X → X2 + A

    # Target macroscopic rates
    target_l2p = 150.0
    target_l2m = 25.0
    target_ratio = target_l2p / target_l2m

    # Grid resolution — increase for cluster runs
    # Laptop:  N1=4000,  N2=2000  (~288 MB per matrix, ~minutes)
    # Cluster: N1=10000, N2=5000  (~1.8 GB per matrix, ~hours)
    N1 = 4000
    N2 = 2000

    print("=" * 70)
    print("Erban Model 2: DECOUPLED vs COUPLED comparison")
    print(f"Grid: N1={N1}, N2={N2}")
    print(f"Target: l2+ = {target_l2p}, l2- = {target_l2m}")
    print(f"Target ratio l2+/l2- = {target_ratio:.10f}")
    print("=" * 70)

    results = {}

    for tau in [1e-6, 2e-7]:
        gamma = np.sqrt(2 * D_tot * tau) / rho

        P_1  = 1.0 - np.exp(-kappa_1 * tau)
        P_2p = 1.0 - np.exp(-kappa_2p * tau)
        P_2m = 1.0 - np.exp(-kappa_2m * tau)

        print(f"\n{'─' * 70}")
        print(f"tau = {tau:.1e},  gamma = {gamma:.4f}")
        print(f"P_lambda:  R1 = {P_1:.6f},  R2f = {P_2p:.6f},  R2b = {P_2m:.6f}")
        print(f"{'─' * 70}")
        """
        # ── R1 (decoupled, homoreaction — unchanged) ──
        t0 = time.time()
        _, _, kd1 = solve_decoupled(gamma, P_1, is_homoreaction=True, N1=N1, N2=N2)
        k1 = kd1 * rho**3 / tau
        print(f"  R1 (X+X→X2):  k_macro = {k1:.6f}  [{time.time()-t0:.1f}s]")

        # ── R2 DECOUPLED ──
        t0 = time.time()
        _, _, kd2p = solve_decoupled(gamma, P_2p, is_homoreaction=False, N1=N1, N2=N2)
        _, _, kd2m = solve_decoupled(gamma, P_2m, is_homoreaction=False, N1=N1, N2=N2)
        l2p_dec = kd2p * rho**3 / tau
        l2m_dec = kd2m * rho**3 / tau
        ratio_dec = l2p_dec / l2m_dec
        print(f"  R2 DECOUPLED:  [{time.time()-t0:.1f}s]")
        print(f"    Forward  l2+ = {l2p_dec:.6f}")
        print(f"    Reverse  l2- = {l2m_dec:.6f}")
        print(f"    Ratio        = {ratio_dec:.10f}")
        """
        # ── R2 COUPLED ──
        t0 = time.time()
        _, g_A, g_X, kc2p, kc2m = solve_coupled(gamma, P_2p, P_2m, N1=N1, N2=N2)
        l2p_cpl = kc2p * rho**3 / tau
        l2m_cpl = kc2m * rho**3 / tau
        ratio_cpl = l2p_cpl / l2m_cpl
        print(f"  R2 COUPLED:    [{time.time()-t0:.1f}s]")
        print(f"    Forward  l2+ = {l2p_cpl:.6f}")
        print(f"    Reverse  l2- = {l2m_cpl:.6f}")
        print(f"    Ratio        = {ratio_cpl:.10f}")

        # results[tau] = {
        #     'ratio_dec': ratio_dec, 'ratio_cpl': ratio_cpl,
        #     'l2p_dec': l2p_dec, 'l2m_dec': l2m_dec,
        #     'l2p_cpl': l2p_cpl, 'l2m_cpl': l2m_cpl,
        # }

    # ── Summary ──
    # print(f"\n{'=' * 70}")
    # print("SUMMARY: Ratio l2+/l2- comparison")
    # print(f"{'=' * 70}")
    # print(f"  Target ratio:                    {target_ratio:.10f}")
    # print()
    # for tau in [1e-6, 2e-7]:
    #     r = results[tau]
    #     print(f"  tau = {tau:.1e}:")
    #     print(f"    Decoupled ratio:  {r['ratio_dec']:.10f}  "
    #           f"(Δ from target = {r['ratio_dec'] - target_ratio:+.10f})")
    #     print(f"    Coupled   ratio:  {r['ratio_cpl']:.10f}  "
    #           f"(Δ from target = {r['ratio_cpl'] - target_ratio:+.10f})")
    # print()

    # r1 = results[1e-6]
    # r2 = results[2e-7]
    # print(f"  Ratio gap between timesteps (tau=2e-7 minus tau=1e-6):")
    # print(f"    Decoupled:  {r2['ratio_dec'] - r1['ratio_dec']:+.10f}")
    # print(f"    Coupled:    {r2['ratio_cpl'] - r1['ratio_cpl']:+.10f}")
    # print()
    # print("  If the coupled gap is LARGER than the decoupled gap,")
    # print("  the coupling amplifies the ratio difference between timesteps.")
    # print("  If it is SMALLER, the coupling dampens it.")


if __name__ == '__main__':
    main()