import numpy as np
from scipy.integrate import quad

def K(r, rp, gamma):
    """The integral kernel K(r, r'; gamma) from Erban Eq (35)."""
    if r == 0:
        return 0.0 # Limit as r->0 is 0
    
    term1 = np.exp(-((r - rp)**2) / (2 * gamma**2))
    term2 = np.exp(-((r + rp)**2) / (2 * gamma**2))
    return (rp / (r * gamma * np.sqrt(2 * np.pi))) * (term1 - term2)

def solve_model2(gamma, P_lambda, is_homoreaction=True, Rmax=10.0, N=500):
    """
    Numerically solves the integral Equation (34) for g(r)
    and then computes the achieved macroscopic kappa using Eq (36).
    """
    r_grid = np.linspace(1e-5, Rmax, N)
    dr = r_grid[1] - r_grid[0]
    
    # We want to solve: g(r) = int_0^infty K(r,r') g(r') dr' - P_lambda * int_0^1 K(r,r') g(r') dr'
    # Build matrix M such that g = M g + b
    M = np.zeros((N, N))
    for i, r in enumerate(r_grid):
        for j, rp in enumerate(r_grid):
            val = K(r, rp, gamma) * dr
            if rp <= 1.0:
                val *= (1.0 - P_lambda)
            M[i, j] = val
            
    # Calculate b vector: int_{Rmax}^\infty K(r,r') * 1 dr'
    # We use scipy.integrate for the infinite tail part where g(r') = 1
    b = np.zeros(N)
    for i, r in enumerate(r_grid):
        res, _ = quad(lambda rp: K(r, rp, gamma), Rmax, 100.0, limit=200)
        b[i] = res
        
    # Solve the linear system (I - M) g = b
    I = np.eye(N)
    g = np.linalg.solve(I - M, b)
    
    # Calculate achieved dimensionless kappa from Eq (36)
    integral = 0.0
    for i, r in enumerate(r_grid):
        if r <= 1.0:
            integral += 4 * np.pi * (r**2) * g[i] * dr
            
    # Erban Eq 36: For homoreaction it's 2*kappa = ...
    # For heteroreaction it's kappa = ...
    if is_homoreaction:
        kappa_val = P_lambda * integral / 2.0
    else:
        kappa_val = P_lambda * integral
        
    return r_grid, g, kappa_val

def main():
    # --- Parameters for R1 (X + X -> X2) ---
    rho = 0.1
    D_tot = 1500.0 + 1500.0 # D_A + D_B
    kappa_micro = [716.88, 3.7921e4, 6.3201e3]
    reaction_str = ['=== Reaction 1 f (X + X -> X2) ===', '=== Reaction 2 f (X2 + A -> X2 + X) ===',
                    '=== Reaction 2 b (X2 + X -> X2 + A) ===']
    is_homoreaction = True
    for cnt in range(3):
        # Timestep 1
        tau1 = 1e-6
        tau_effective1 = tau1  # For R1, you use the full tau step
        gamma1 = np.sqrt(2 * D_tot * tau_effective1) / rho
        P_lambda1 = 1.0 - np.exp(-kappa_micro[cnt] * tau_effective1)
        
        _, _, kappa_dim1 = solve_model2(gamma1, P_lambda1, is_homoreaction)
        k_macro1 = kappa_dim1 * (rho**3) / tau_effective1
        
        # Timestep 2
        tau2 = 2e-7
        tau_effective2 = tau2
        gamma2 = np.sqrt(2 * D_tot * tau_effective2) / rho
        P_lambda2 = 1.0 - np.exp(-kappa_micro[cnt] * tau_effective2)
        
        _, _, kappa_dim2 = solve_model2(gamma2, P_lambda2, is_homoreaction)
        k_macro2 = kappa_dim2 * (rho**3) / tau_effective2
        
        print(f"{reaction_str[cnt]}")
        print(f"Target k1: 1.5")
        print(f"tau = {tau1}: gamma = {gamma1:.4f}, P_lambda = {P_lambda1:.4e}, k_macro_achieved = {k_macro1:.6f}")
        print(f"tau = {tau2}: gamma = {gamma2:.4f}, P_lambda = {P_lambda2:.4e}, k_macro_achieved = {k_macro2:.6f}")
        print(f"Ratio k_macro(2e-7) / k_macro(1e-6) = {k_macro2 / k_macro1:.6f}")

        is_homoreaction = False

if __name__ == '__main__':
    main()