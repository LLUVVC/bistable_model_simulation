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

    
changed the calculation of phi_A to delta_A=1-phi_A, to improve the precision!!!
"""

import numpy as np
from scipy.special import erf
import time
import pandas as pd
from scipy.optimize import root_scalar, root
from simulation.solvers.rate_conversions import calculate_kappas
from pathlib import Path
from datetime import datetime

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


# ─────── Decoupled solver (original one in Appendix F on Erban and Chapman's paper) ────────

def solve_decoupled(gamma, P_lambda, is_homoreaction=True, S=10.0, N1=4000, N2=2000):
    """
    Standard Erban Model 2: solves one reaction independently.
    Uses delta = 1 - g to bypass catastrophic cancellation and tail integrals.
    """
    r_grid = build_grid(N1, N2, S)
    print(r_grid[0])
    print(r_grid[-1])
    N = N1 + N2

    M = np.zeros((N, N))
    Kr_ones = np.zeros(N)

    for i in range(N):
        r_i = r_grid[i]
        for j in range(N1):
            K_val = K(r_i, r_grid[j], gamma)
            w_j = 1.0 / N1
            M[i, j] = (1.0 - P_lambda) * K_val * w_j
            Kr_ones[i] += K_val * w_j
        for j in range(N1, N):
            M[i, j] = ((S - 1.0) / N2) * K(r_i, r_grid[j], gamma)

    # solve for delta = 1 - g directly 
    rhs_delta = P_lambda * Kr_ones
    delta = np.linalg.solve(np.eye(N) - M, rhs_delta)

    # calculate integral of g(r) = 1 - delta(r) over the reactive volume
    integral = 0.0
    for j in range(N1):
        w_j = 1.0 / N1
        r_prev = r_grid[j] - w_j
        dV_j = 4 * np.pi * r_grid[j] ** 2 * w_j
        # dV_j = 4 * np.pi * r_prev ** 2 * w_j
        # dV_j = (4.0 / 3.0) * np.pi * (r_grid[j]**3 - r_prev**3)
        integral += (1.0 - delta[j]) * dV_j
        # integral += 4 * np.pi * (r_grid[j] ** 2) * (1.0 - delta[j]) * w_j

    if is_homoreaction:
        kappa_val = P_lambda * integral / 2.0
    else:
        kappa_val = P_lambda * integral

    return r_grid, delta, kappa_val

# ─── Exact Coupled solver ───────────────────────────────────────────────

def solve_coupled_exact_2(gamma, P_lambda_plus, P_lambda_minus, S=10.0, N1=4000, N2=2000):
    """
    Coupled Erban Model 2 for X2 + A ⇌ X2 + X using delta_A and phi_X.
    delta_A = 1 - phi_A (the deviation from the boundary condition at infinity).

    Returns r_grid, delta_A, phi_X, kappa_plus_achieved, kappa_minus_achieved.
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

    # solve directly for the deviations
    # The tail boundary vector 'b' cancels out analytically. (see the calculations in the appendix)
    rhs_delta_A = P_lambda_plus * Kr_ones
    rhs_phi_X   = P_lambda_minus * Kr_ones
    
    delta_A = np.linalg.solve(np.eye(N) - M, rhs_delta_A)
    phi_X   = np.linalg.solve(np.eye(N) - M, rhs_phi_X)
    
    # achieved dimensionless kappas
    int_A = 0.0
    int_X = 0.0
    for j in range(N1):
        w_j = 1.0 / N1
        dV_j = 4 * np.pi * r_grid[j] ** 2 * w_j # the outer shell
        r_prev = r_grid[j] - w_j
        # dV_j = 4 * np.pi * r_prev ** 2 * w_j # the inner shell
        # dV_j = (4.0 / 3.0) * np.pi * (r_grid[j]**3 - r_prev**3) # the shell volume
        # forward net element: P+ - (P+ + P-) * delta_A
        int_A += (P_lambda_plus - P_total * delta_A[j]) * dV_j
        # backward net element: P- - (P+ + P-) * phi_X
        int_X += (P_lambda_minus - P_total * phi_X[j]) * dV_j

    kappa_plus = int_A
    kappa_minus = int_X
    ratio = kappa_plus/kappa_minus
    return r_grid, delta_A, phi_X, kappa_plus, kappa_minus, ratio # phi_X

def recover_lambda(p, tau):
    return -np.log(1-p)/tau

def micro_to_macro(diffusions, tau, N1=4000, N2=2000): # tau_list = [2e-6,]#2e-7
    # Microscopic rates from the coupled Doi solver
    
    print(f"diffusions dx, dx2, da, db = {diffusions}")
    
    print(f"timestep = {tau}")
    rho = 0.1       # sigma/rho (reaction radius)
    ls = np.array((1.5, 1500., 150., 25., 5.75, 25.))
    sigmas = np.ones(4)*rho
    kappas = calculate_kappas(ls, diffusions[2], diffusions[0], diffusions[1], sigmas)
    D_tot = diffusions[1]+diffusions[0] # 2 * D   # D_X2 + D_A = D_X2 + D_X = 3000

    kappa_1  = kappas[0]      # R1:  X + X  → X2
    kappa_2p = kappas[2]      # R2f: X2 + A → X2 + X
    kappa_2m = kappas[3]      # R2b: X2 + X → X2 + A

    # Target macroscopic rates
    target_l2p = ls[2]
    target_l2m = ls[3]
    target_ratio = target_l2p / target_l2m

    print("=" * 70)
    print("Erban Model 2: EXACT COUPLED solver (Concentration Independent)")
    print(f"Grid: N1={N1}, N2={N2}")
    print(f"Target: l2+ = {target_l2p}, l2- = {target_l2m}")
    print(f"Target ratio l2+/l2- = {target_ratio:.10f}")
    print("=" * 70)

    # for tau in tau_list: # 2e-7
    gamma_2 = np.sqrt(2 * D_tot * 0.5 * tau) / rho
    gamma_1 = np.sqrt(2 * (diffusions[0]+ diffusions[0]) * tau) /rho
    print(f"gamma equals to {gamma_1}.")
    P_1  = 1.0 - np.exp(-kappa_1 * tau)
    P_2p = 1.0 - np.exp(-kappa_2p * 0.5 * tau)
    P_2m = 1.0 - np.exp(-kappa_2m * 0.5 * tau)

    print(f"\n{'─' * 70}")
    print(f"P_lambda:  R1f = {P_1:.6f}, R2f = {P_2p:.6f},  R2b = {P_2m:.6f}")
    print(f"{'─' * 70}")

    # ── R1 DECOUPLED EXACT ──
    t0 = time.time()
    print(f"tau = {tau:.1e},  gamma_1 = {gamma_1:.4f}")
    _, _, kd1 = solve_decoupled(gamma_1, P_1, is_homoreaction=True, N1=N1, N2=N2)
    k1 = kd1 * rho**3 / tau
    print(f"  R1 (X+X→X2):  k_macro = {k1:.6f}  [{time.time()-t0:.1f}s]")
    # ── R2 COUPLED EXACT ──
    t0 = time.time()
    #####_, delta_A, phi_X, kc2p, kc2m = solve_coupled_asymmetric(gamma_in, gamma_out, P_2p, P_2m, N1=N1, N2=N2)
    print(f"tau = {tau:.1e},  gamma_2 = {gamma_2:.4f}")
    _, delta_A, phi_X, kc2p, kc2m, ratio = solve_coupled_exact_2(gamma_2, P_2p, P_2m, N1=N1, N2=N2) # gamma_2
    # _, phi_A, phi_X, kc2p, kc2m = solve_coupled_exact(gamma, P_2p, P_2m, N1=N1, N2=N2)
    l2p_cpl = 2 * kc2p * rho**3 / tau
    l2m_cpl = 2 * kc2m * rho**3 / tau
    ratio_cpl = l2p_cpl / l2m_cpl
    
    print(f"  EXACT COUPLED RATES:    [{time.time()-t0:.1f}s]")
    print(f"    Forward  l2+ = {l2p_cpl:.6f}")
    print(f"    Reverse  l2- = {l2m_cpl:.6f}")
    print(f"    Ratio        = {ratio:.10f}")
    print(f"    Ratio_cpl    = {ratio_cpl:.10f}")



def micro_to_P_needed(tau_list, diff_list, tau_fixed=True, N1=1000, N2=500):
    '''
    when tau_fixed == True, varing diffusion to change gamma and see the how the  
    P_needed (to reproduce the exact macroscopic rates) changes wrt gamma, by varing the diffusion size;

    tau_fixed == False, varying tau with a fixed diffusion to see how P_needed goes vs. gamma, by varying
    the tau size. 

    gamma = np.sqrt(2*(D1+D2)*tau)/sigma
    '''
    ### parameter setting
    rho = 0.1
    ls = np.array((1.5, 1500., 150., 25., 5.75, 25.))
    sigmas = np.ones(4)*rho

    results = []
    # define the functions
    def find_P_1(P, tau, gamma):
        _,_,lhs = solve_decoupled(gamma, P, is_homoreaction=True, N1=N1, N2=N2)
        rhs = ls[0]*tau/(rho**3)
        return rhs - lhs

    def find_P_2(P_vars, tau, gamma):
        
        print(f"Testing P_plus={P_vars[0]:.6f}, P_minus={P_vars[1]:.6f}")
        P_plus, P_minus = P_vars
        _,_,_,lhs_plus,lhs_minus = solve_coupled_exact_2(gamma, P_plus, P_minus, N1, N2)
        rhs_plus = ls[2]*tau/(rho**3)
        rhs_minus = ls[3]*tau/(rho**3)
        return [lhs_plus-rhs_plus, lhs_minus-rhs_minus]
    
    t0 = time.time()

    print("Started calculating P needed...")
    if tau_fixed:
        tau = tau_list[0]
        print("Timestep is fixed at {tau}, diffusions are varying.")
        for i in range(len(diff_list)):
            print(f"Current diffusion size is {diff_list[i]}")
            diffusions = np.ones(4) * diff_list[i]
            gamma_1 = np.sqrt(2 * (diffusions[0]+ diffusions[0]) * tau) /rho
            # ── R1 DECOUPLED EXACT ──
            solution_1 = root_scalar(
            find_P_1,
            args=(tau,gamma_1),
            bracket=[0.0, 1.0],
            method='brentq'
        )
            P_1 = solution_1.root
            # ── R2 COUPLED EXACT ──
            kappas = calculate_kappas(ls, diffusions[2], diffusions[0], diffusions[1], sigmas)
            initial_guess = [1-np.exp(-kappas[2]*tau), 1-np.exp(-kappas[3]*tau)]
            solution_2 = root(
                find_P_2, initial_guess, args=(tau, gamma_1), method='lm'
            ) # gamma_2
            P_2_plus, P_2_minus = solution_2.x
            results.append({'gamma':gamma_1,'timestep':tau,'macro_full':ls, 'micro_full':kappas, 
                            'diffusions':diff_list[i], 'P_needed_1': P_1, 'P_needed_2_plus': P_2_plus, 
                            'P_needed_2_minus': P_2_minus})   
            file_str =  f'p_lambda_regimes_tau{tau}.csv'

    else:
        diffusions = np.ones(4) * diff_list[0]
        print(f"diffusions dx, dx2, da, db = {diffusions}")
        kappas = calculate_kappas(ls, diffusions[2], diffusions[0], diffusions[1], sigmas)
        for i in range(len(tau_list)):
            print(f"Current time step size is {tau_list[i]}")
            gamma_1 = np.sqrt(2 * (diffusions[0]+ diffusions[0]) * tau_list[i]) /rho
            # ── R1 DECOUPLED EXACT ──
            solution_1 = root_scalar(
            find_P_1,
            args=(tau_list[i],gamma_1),
            bracket=[0.0, 1.0],
            method='brentq'
        )
            P_1 = solution_1.root
            # ── R2 COUPLED EXACT ──
            initial_guess = [1-np.exp(-kappas[2]*tau_list[i]), 1-np.exp(-kappas[3]*tau_list[i])]
            solution_2 = root(
                find_P_2, initial_guess, args=(tau_list[i], gamma_1), method='lm'
            ) # gamma_2
            P_2_plus, P_2_minus = solution_2.x
            results.append({'gamma':gamma_1,'timestep':tau_list[i],'macro_full':ls, 'micro_full':kappas, 
                            'diffusions':diffusions, 'P_needed_1': P_1, 'P_needed_2_plus': P_2_plus, 
                            'P_needed_2_minus': P_2_minus})   
            file_str =  f'p_lambda_regimes_D{diffusions[0]}.csv'

    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = project_root/'results'/'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir/file_str
    print(f"------ Time taken is [{time.time()-t0:.1f}s] ------")
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=True)
    print("------ csv saved to {file_path} ------")

# ─── Main ───────────────────────────────────────────────────────────────
def main():
    
    model1_kappa_known = True

    if model1_kappa_known:
        
        diffusion_list = [6000, 12000, 24000,] # [1500.0, 1500.0, 750.0, 750.0]
        tau_list = [1e-7, 1e-8, 1e-9,] #[1e-6, 2e-7, 2e-6, 1e-6]#[1e-6, 5e-6, 2e-7,1e-7, 5e-8]
        for i in range(len(tau_list)):
            tau = tau_list[i]
            for j in range(len(diffusion_list)):
                diffusions = np.ones(4) * diffusion_list[j]
                micro_to_macro(diffusions, tau, N1=20000, N2=5000)

    ## the following timestep is for cluster running
    
    else:
        tau_fixed = True
        if tau_fixed:
            # create 30 evenly spaced gamma values from 0.2 up to 3.0
            gamma_sweep = np.linspace(0.2, 3.0, 30)

            # Back-calculate the required diffusion constants for those gammas
            fixed_tau = 1e-6
            rho = 0.1
            d_sweep = (gamma_sweep * rho)**2 / (4 * fixed_tau)
            tau_sweep = [fixed_tau, ]
            print("Gamma values:", gamma_sweep)
            print("Diffusion values:", d_sweep)

        else:
            # create 30 evenly spaced gamma values from 0.2 up to 3.0
            gamma_sweep = np.linspace(0.2, 3.0, 30)

            # Back-calculate the required diffusion constants for those gammas
            fixed_D = 1500.0
            rho = 0.1
            tau_sweep = (gamma_sweep * rho)**2 / (4 * fixed_D)
            d_sweep = [fixed_D, ]
            print("Gamma values:", gamma_sweep)
            print("Diffusion values:", tau_sweep)
            
            
        
        micro_to_P_needed(tau_sweep, d_sweep, tau_fixed=tau_fixed)

if __name__ == '__main__':
    main()

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################

"""
def solve_coupled_asymmetric(gamma_in, gamma_out, P_lambda_plus, P_lambda_minus, S=10.0, N1=4000, N2=2000):
    '''
    Coupled Erban Model 2 for X2 + A ⇌ X2 + X when D_in != D_out.
    '''
    r_grid = build_grid(N1, N2, S)
    N = N1 + N2
    
    # 1. Build the kernels for both diffusion regimes
    Min_1 = np.zeros((N, N))
    Min_2 = np.zeros((N, N))
    Mout_1 = np.zeros((N, N))
    Mout_2 = np.zeros((N, N))
    
    Kr_ones_in = np.zeros(N)
    Kr_ones_out = np.zeros(N)
    
    for i in range(N):
        r_i = r_grid[i]
        for j in range(N1):
            w_j = 1.0 / N1
            kin = K(r_i, r_grid[j], gamma_in) * w_j
            kout = K(r_i, r_grid[j], gamma_out) * w_j
            
            Min_1[i, j] = kin
            Mout_1[i, j] = kout
            
            Kr_ones_in[i] += kin
            Kr_ones_out[i] += kout
            
        for j in range(N1, N):
            w_j = (S - 1.0) / N2
            kin = K(r_i, r_grid[j], gamma_in) * w_j
            kout = K(r_i, r_grid[j], gamma_out) * w_j
            
            Min_2[i, j] = kin
            Mout_2[i, j] = kout
            
    # 2. Build the Block Matrices
    I = np.eye(N)
    
    A11 = I - ((1.0 - P_lambda_plus) * Min_1 + Min_2)
    A12 = P_lambda_minus * Min_1
    
    A21 = P_lambda_plus * Mout_1
    A22 = I - ((1.0 - P_lambda_minus) * Mout_1 + Mout_2)
    
    # Assemble the full 2N x 2N matrix
    A_full = np.block([
        [A11, A12],
        [A21, A22]
    ])
    
    # 3. Solve STATE 1 (Forward Rate, A -> 1, X -> 0)
    rhs_delta_A_fwd = P_lambda_plus * Kr_ones_in
    rhs_phi_X_fwd   = P_lambda_plus * Kr_ones_out 
    rhs_full_fwd    = np.concatenate([rhs_delta_A_fwd, rhs_phi_X_fwd])
    
    sol_fwd = np.linalg.solve(A_full, rhs_full_fwd)
    delta_A_fwd = sol_fwd[:N]
    phi_X_fwd   = sol_fwd[N:]
    
    # 4. Solve STATE 2 (Backward Rate, A -> 0, X -> 1)
    rhs_phi_A_bwd   = P_lambda_minus * Kr_ones_in
    rhs_delta_X_bwd = P_lambda_minus * Kr_ones_out 
    rhs_full_bwd    = np.concatenate([rhs_phi_A_bwd, rhs_delta_X_bwd])
    
    sol_bwd = np.linalg.solve(A_full, rhs_full_bwd)
    phi_A_bwd   = sol_bwd[:N]
    delta_X_bwd = sol_bwd[N:]
    
    # 5. Calculate Independent Achieved Rates
    int_A = 0.0
    int_X = 0.0
    for j in range(N1):
        w_j = 1.0 / N1
        dV_j = 4 * np.pi * r_grid[j] ** 2 * w_j
        
        # l_plus comes strictly from the net forward flux in State 1
        net_fwd = P_lambda_plus * (1.0 - delta_A_fwd[j]) - P_lambda_minus * phi_X_fwd[j]
        
        # l_minus comes strictly from the net backward flux in State 2
        net_bwd = P_lambda_minus * (1.0 - delta_X_bwd[j]) - P_lambda_plus * phi_A_bwd[j]
        
        int_A += net_fwd * dV_j
        int_X += net_bwd * dV_j
        
    kappa_plus = int_A
    kappa_minus = int_X
    
    return r_grid, delta_A_fwd, phi_X_fwd, kappa_plus, kappa_minus
"""