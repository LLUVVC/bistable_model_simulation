import numpy as np
from scipy.special import erf

def K(r, rp, gamma):
    """The integral kernel K(r, r'; gamma) from Erban Eq (35)."""
    if r == 0:
        return 0.0 # Limit as r->0 is 0
    
    term1 = np.exp(-((r - rp)**2) / (2 * gamma**2))
    term2 = np.exp(-((r + rp)**2) / (2 * gamma**2))
    return (rp / (r * gamma * np.sqrt(2 * np.pi))) * (term1 - term2)

def solve_model2_erban_exact(gamma, P_lambda, is_homoreaction=True, S=10.0, N1=5000, N2=2500):
    """
    Solves Eq (34) using the exact numerical scheme from Appendix F.
    - N1 points in [0, 1]
    - N2 points in [1, S]
    """
    # 1. Create the piece-wise grid exactly as defined in the paper
    r_grid = np.zeros(N1 + N2)
    for j in range(1, N1 + 1):
        r_grid[j-1] = j / N1
    for j in range(N1 + 1, N1 + N2 + 1):
        r_grid[j-1] = 1.0 + (S - 1.0) * (j - N1) / N2
        
    N = N1 + N2
    M = np.zeros((N, N))
    
    # 2. Build the matrix M according to the discretization
    for i in range(N):
        r_i = r_grid[i]
        
        # Sum over j=1 to N1 (inside reaction radius)
        for j in range(N1):
            r_j = r_grid[j]
            M[i, j] = ((1.0 - P_lambda) / N1) * K(r_i, r_j, gamma)
            
        # Sum over j=N1+1 to N1+N2 (outside reaction radius)
        for j in range(N1, N):
            r_j = r_grid[j]
            M[i, j] = ((S - 1.0) / N2) * K(r_i, r_j, gamma)
            
    # 3. Calculate the infinite tail vector using the analytical 'erf' formula
    b = np.zeros(N)
    for i in range(N):
        r_i = r_grid[i]
        
        term1 = (gamma**2 * K(r_i, S, gamma)) / S
        term2 = 1.0
        term3 = -0.5 * erf((S - r_i) / (gamma * np.sqrt(2)))
        term4 = -0.5 * erf((S + r_i) / (gamma * np.sqrt(2)))
        
        b[i] = term1 + term2 + term3 + term4
        
    # 4. Solve the linear system
    I = np.eye(N)
    g = np.linalg.solve(I - M, b)
    
    # 5. Calculate achieved dimensionless kappa from Eq (36)
    # The integral is only from 0 to 1, which corresponds to the first N1 points
    integral = 0.0
    for j in range(N1):
        r_j = r_grid[j]
        integral += 4 * np.pi * (r_j**2) * g[j] * (1.0 / N1)
            
    if is_homoreaction:
        kappa_val = P_lambda * integral / 2.0
    else:
        kappa_val = P_lambda * integral
        
    return r_grid, g, kappa_val

def main():
    rho = 0.1
    D_tot = 1500.0 + 1500.0 # D_A + D_B
    kappa_micro = [716.88, 3.7604e4, 6.0162e3] # [716.88, 3.7921e4, 6.3201e3]
    reaction_str = ['=== Reaction 1 f (X + X -> X2) ===', '=== Reaction 2 f (X2 + A -> X2 + X) ===',
                    '=== Reaction 2 b (X2 + X -> X2 + A) ===']
    is_homoreaction = True
    for cnt in range(3):
        for tau in [1e-6, 2e-7]:
            gamma = np.sqrt(2 * D_tot * tau) / rho
            P_lambda = 1.0 - np.exp(-kappa_micro[cnt] * tau)
            
            _, _, kappa_dim = solve_model2_erban_exact(gamma, P_lambda, is_homoreaction)
            k_macro_achieved = kappa_dim * (rho**3) / tau
            print(f"{reaction_str[cnt]}")
            print(f"tau = {tau:.1e}: gamma = {gamma:.4f}, k_macro_achieved = {k_macro_achieved:.6f}")
        is_homoreaction = False


if __name__ == '__main__':
    main()