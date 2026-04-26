import numpy as np
from scipy.optimize import root_scalar, root

"""

Reaction rates of the well-mixed full model -> Reaction rates of the spatial full model 

"""

import numpy as np
from scipy.optimize import root_scalar, root


def calculate_k_from_l(l):
    keq = l[0]/l[1]
    k = np.array((keq*l[2], keq*l[3], l[4], l[5]))
    return k

def get_reaction_volume(sigma):
    """Calculates the volume of the reaction sphere."""
    return (4.0/3.0) * np.pi * (sigma**3)


# --- Formula definitions ---

def l1_plus_formula(kappa_1_plus, D, sigma):
     # Calculate the term inside the tanh function
    sqrt_term = np.sqrt(kappa_1_plus / (2 * D))
    
    # Calculate the Left-Hand Side (LHS) of the equation
    # This is the expression for the effective rate l_1^+
    tanh_val = np.tanh(sigma * sqrt_term)
    lhs = 4 * np.pi * D * (sigma - (1 / sqrt_term) * tanh_val)
    return lhs

def find_kappa_1_plus(kappa_1_plus, l1_plus_input, D, sigma):
    """
    Defines the self-consistency equation for kappa_1_plus.
    This function will be zero when the correct kappa_1_plus is found.
    """
    # Ensure kappa_1_plus is positive to avoid math errors
    if kappa_1_plus <= 0:
        return np.inf # Return a large number if the guess is non-physical

    lhs = l1_plus_formula(kappa_1_plus, D, sigma)
    # Calculate the Right-Hand Side (RHS) of the equation
    rhs = l1_plus_input

    return lhs - rhs
def calculate_l2_rates(kappa_2_plus, kappa_2_minus, DA, DX, DX2, sigma_3):
    if kappa_2_plus <= 0 or kappa_2_minus <= 0: 
        return np.inf, np.inf
    
    alpha_sq = kappa_2_plus / (DX2 + DA) + kappa_2_minus / (DX2 + DX)
    alpha = np.sqrt(alpha_sq)
    common_factor = 4 * np.pi * (1 / alpha_sq) * (sigma_3 - np.tanh(alpha * sigma_3) / alpha)
    l2_plus = kappa_2_plus * common_factor
    l2_minus = kappa_2_minus * common_factor

    return l2_plus, l2_minus

# --- ROBUST SOLVER FOR REACTION 2 ---
def find_kappa_2_pair_robust(l2_plus_target, l2_minus_target, DA, DX, DX2, sigma_3):
    """
    Robust solver that uses physical estimates to initialize and verify the root.
    """
    # 1. Calculate the 'Safe' Reaction-Limited Estimate
    #    In the limit of fast diffusion, lambda = k_macro / V_sphere, Eq.(32)
    # Add this step because sometimes it return unreasonable values for the model
    V_sph = get_reaction_volume(sigma_3)
    k2p_est = l2_plus_target / V_sph
    k2m_est = l2_minus_target / V_sph
    
    initial_guess = [k2p_est, k2m_est] 

    # 2. Define Objective Function
    def objective_function(vars):
        kappa_2_plus, kappa_2_minus = vars
        # Prevent negative values causing math errors in sqrt
        if kappa_2_plus <= 0: kappa_2_plus = 1e-12
        if kappa_2_minus <= 0: kappa_2_minus = 1e-12
        
        l2_plus_cal, l2_minus_cal = calculate_l2_rates(kappa_2_plus, kappa_2_minus, DA, DX, DX2, sigma_3)
        
        # Calculate residuals
        return [l2_plus_cal - l2_plus_target, l2_minus_cal - l2_minus_target]

    # 3. Run Solver
    # We use 'lm' (Levenberg-Marquardt) as it is robust for least-squares
    sol = root(objective_function, initial_guess, method='lm')
    
    if not sol.success:
        print("⚠️ Solver did not converge. Using estimates.")
        return k2p_est, k2m_est

    k2p_sol, k2m_sol = sol.x

    # 4. VALIDATION: Check if the root is "Wrong"
    
    # Criteria A: Must be positive
    if k2p_sol <= 0 or k2m_sol <= 0:
        print("⚠️ κ₂ Solver found non-positive rates. Discarding. Using estimates.")
        return k2p_est, k2m_est 

    # Criteria B: Ratio Check
    # The micro ratio should be close to macro ratio (within factor of 2 tolerance)
    # Theoretically, those two ratios should be equal
    target_ratio = l2_plus_target / l2_minus_target
    sol_ratio = k2p_sol / k2m_sol
    
    # Check if ratio is preserved within a tolerance (e.g., factor of 2)
    if not (0.5 * target_ratio < sol_ratio < 2.0 * target_ratio):
        print(f"⚠️ Solver found wrong root for R2! Ratio {sol_ratio:.2f} != Target {target_ratio:.2f}")
        print(f"   Discarding solution: {k2p_sol:.2e}, {k2m_sol:.2e}")
        print(f"   Using reaction-limited estimate instead.")
        return k2p_est, k2m_est

    # Criteria C: Magnitude Check
    # Ensure result is within 1 order of magnitude of the estimate
    if np.abs(np.log10(k2p_sol) - np.log10(k2p_est)) > 1.0:
         print(f"⚠️ Solver result for R2 is orders of magnitude off. Discarding.")
         return k2p_est, k2m_est

    print("✅ R2 Solver converged and passed physics checks.")
    return k2p_sol, k2m_sol



def calculate_kappas(ls, DA, DX, DX2, sigma):
    kappas = np.zeros(6)
    
    # --- REACTION 1 (R1) ---
    # R1 Plus is Bimolecular (Needs Volume scaling)
    # R1 Minus is Unimolecular (NO Volume scaling)
    
    V_sph_R1 = get_reaction_volume(sigma[0])
    
    # 1. Estimate for R1
    est_k1p = 2 * ls[0] / V_sph_R1
    est_k1m = ls[1] # Unimolecular, so kappa approx equals macro rate
    
    # 2. Try solving for R1 Plus
    try:
        solution = root_scalar(
            find_kappa_1_plus,
            args=(ls[0], DX, sigma[0]), 
            bracket=[est_k1p * 0.01, est_k1p * 100], # Search around the physical estimate
            method='brentq'
        )
        k1p_sol = solution.root
        
        # 3. Check Magnitude for R1
        if np.abs(np.log10(k1p_sol) - np.log10(est_k1p)) > 1.0:
            print("⚠️ R1 Solver result orders of magnitude off. Using estimate.")
            kappas[0] = est_k1p
        else:
            print("✅ R1 Solver converged.")
            kappas[0] = k1p_sol
            
    except ValueError:
        print(f"❌ R1 Solver failed. Using estimate.")
        kappas[0] = est_k1p

    kappas[1] = ls[1] # Unimolecular (Reverse R1)

    # --- REACTION 2 (R2) ---
    k2p_sol, k2m_sol = find_kappa_2_pair_robust(ls[2], ls[3], DA, DX, DX2, sigma[2])
    kappas[2] = k2p_sol
    kappas[3] = k2m_sol

    # --- REACTION 3 (R3) ---
    kappas[4], kappas[5] = ls[4], ls[5] # Usually assumed 0th/1st order, effectively unchanged

    # Print Summary
    print("\n--- Final Intrinsic Rates (Kappa) ---")
    print(f"κ₁⁺ = {kappas[0]:.4e}") # (Est: {est_k1p:.4e})
    print(f"κ₁⁻ = {kappas[1]:.4e}") # (Est: {est_k1m:.4e})
    print(f"κ₂⁺ = {kappas[2]:.4e}")
    print(f"κ₂⁻ = {kappas[3]:.4e}")
    print(f"κ₃⁺ = {kappas[4]:.4e}")
    print(f"κ₃⁻ = {kappas[5]:.4e}")
    print("The estimation is calculated based on the Eq. (32) in Erban's paper.")

    return kappas