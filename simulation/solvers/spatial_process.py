import numpy as np
import tqdm as tqdm
from numba import njit
from numba_progress import ProgressBar
from simulation.utils.geometry_fast import generate_position_box_numba, generate_sphere_offsets_numba, diffusion_periodic_step_numba
from simulation.utils.reactions_fast import (maintain_bath_numba, unimolecularSelectReactant_numba, bimolecular_hetero_candidates_update,
                                             bimolecular_homo_candidates_update, AddParticleHomoMid_numba_update, SubstituteParticle_numba)

# ==========================================
# REACTION LOGIC
# ==========================================

@njit
def reaction_R1_forward_numba(pos_x, pos_x2, sigma, kappa, h, box_shape):
    """
    X + X -> X2 (Homogeneous)
    Pairs of X become one X2 at midpoint.
    """
    n_x = len(pos_x)
    if n_x < 2:
        return pos_x, pos_x2

    react_xa, react_xb = bimolecular_homo_candidates_update(pos_x, sigma, kappa, h, box_shape)
    pos_x, pos_x2 = AddParticleHomoMid_numba_update(pos_x, pos_x2, react_xa, react_xb, box_shape)

    return pos_x, pos_x2

@njit
def reaction_R1_backward_numba(pos_x, pos_x2, sigma, kappa, h, box_shape):
    """
    X2 -> X + X (Unimolecular)
    X2 disappears, 2 Xs appear near it.
    """
    n_x2 = len(pos_x2)
    if n_x2 == 0:
        return pos_x, pos_x2
        
    reacted_list = unimolecularSelectReactant_numba(pos_x2, kappa, h) # get the reacted X2
    num_r = len(reacted_list) # number of X2 that engages in the reaction
    if num_r == 0:
        return pos_x, pos_x2
    
    reacted_r_coords = pos_x2[reacted_list]
    base_pos = np.empty((num_r * 2, 3), dtype= pos_x.dtype)
    base_pos[::2] = reacted_r_coords
    base_pos[1::2] = reacted_r_coords
    
    offsets = generate_sphere_offsets_numba(2*num_r, sigma)
    pos_new_products = base_pos + offsets
    ## ensure the added particles are inside the simulation box
    pos_new_products = pos_new_products - np.floor(pos_new_products / box_shape) * box_shape
    
    '''
    ### axis=0 is not supported in numba
    expanded_reacted_r_coords = np.repeat(reacted_r_coords, repeats=2, axis=0)
    pos_new_products = expanded_reacted_r_coords + generate_sphere_offsets_numba(2*num_r, sigma)
    '''
    
    if pos_x.size==0:
        pos_x = pos_new_products
    else:
        pos_x = np.vstack((pos_x, pos_new_products))

    mask = np.ones(n_x2, dtype=np.bool_)
    mask[reacted_list] = False   
    pos_x2 = pos_x2[mask]
    
    return pos_x, pos_x2


@njit
def reaction_hetero_replacement_numba(pos_catalyst, pos_substrate, pos_product, sigma, kappa, h, box_shape):
    """
    Generic Catalyst + Substrate -> Catalyst + Product
    (Used for R2: X2 + A -> X2 + X and X2 + X -> X2 + A)
    Returns: Updated pos_substrate (removed), List of new product positions
    """
    n_cat = len(pos_catalyst)
    n_sub = len(pos_substrate)
    if n_cat == 0 or n_sub == 0:
        return pos_substrate, pos_product

    _, react_sub = bimolecular_hetero_candidates_update(pos_catalyst, pos_substrate, sigma, kappa, h, box_shape)
    if len(react_sub) == 0:
        return pos_substrate, pos_product
    else:
        pos_substrate, pos_product = SubstituteParticle_numba(pos_substrate, pos_product, react_sub)
        return pos_substrate, pos_product

@njit
def reaction_unimolecular_replacement_numba(pos_r, pos_p, kappa, h):
    """
    R -> P (Simple replacement)
    (Used for R3: B -> X and X -> B)
    """
    n_r = len(pos_r)
    if n_r == 0:
        return pos_r, pos_p # np.empty((0,3), dtype=np.float64)
        
    react_list = unimolecularSelectReactant_numba(pos_r, kappa, h)
    if len(react_list) == 0:
        return pos_r, pos_p
    else:
        pos_r, pos_p = SubstituteParticle_numba(pos_r, pos_p, react_list)
        return pos_r, pos_p


# ==========================================
# THE MAIN STRANG SPLITTING STEP
# ==========================================

@njit
def run_one_step_numba(pos_x, pos_x2, pos_a, pos_b,
                       sigmas, kappas, h, box_shape,
                       num_a_target, num_b_target):
    
    '''
    use Strang splitting
    Bruno Sportisse: An Analysis of Operator Splitting Techniques in the Stiff Case
    -> slow reaction first then fast reactions
    '''
    half_dt = h * 0.5
    full_dt = h
    
    # === PART 1: SLOW (R2, R3) ===
    
    # --- R2 Forward: X2 + A -> X2 + X ---
    # pos_a is reduced, new_x_from_a created
    pos_a, pos_x = reaction_hetero_replacement_numba(pos_x2, pos_a, pos_x, sigmas[2], kappas[2], half_dt, box_shape)
    pos_a = maintain_bath_numba(pos_a, box_shape, num_a_target)
   

    # --- R2 Backward: X2 + X -> X2 + A ---
    # pos_x is reduced, new_a_from_x created
    pos_x, pos_a = reaction_hetero_replacement_numba(pos_x2, pos_x, pos_a, sigmas[3], kappas[3], half_dt, box_shape)
    pos_a = maintain_bath_numba(pos_a, box_shape, num_a_target) # We just discard excess A here effectively via maintain
    
    # --- R3 Forward: B -> X ---
    pos_b, pos_x = reaction_unimolecular_replacement_numba(pos_b, pos_x, kappas[4], half_dt)
    pos_b = maintain_bath_numba(pos_b, box_shape, num_b_target)

    # --- R3 Backward: X -> B ---
    pos_x, pos_b = reaction_unimolecular_replacement_numba(pos_x, pos_b, kappas[5], half_dt)
    pos_b = maintain_bath_numba(pos_b, box_shape, num_b_target)
    
    
    # === PART 2: FAST (R1) ===
    
    # --- R1 Forward: X + X -> X2 ---
    pos_x, pos_x2 = reaction_R1_forward_numba(pos_x, pos_x2, sigmas[0], kappas[0], full_dt, box_shape)
    
    # --- R1 Backward: X2 -> X + X ---
    '''change 1.'''
    pos_x, pos_x2 = reaction_R1_backward_numba(pos_x, pos_x2, sigmas[1], kappas[1], full_dt, box_shape)

    
    # === PART 3: SLOW REPEAT (Reversed Order) ===
    
    # --- R3 Backward: X -> B ---
    pos_x, pos_b = reaction_unimolecular_replacement_numba(pos_x, pos_b, kappas[5], half_dt)
    pos_b = maintain_bath_numba(pos_b, box_shape, num_b_target)

    # --- R3 Forward: B -> X ---
    pos_b, pos_x = reaction_unimolecular_replacement_numba(pos_b, pos_x, kappas[4], half_dt)
    pos_b = maintain_bath_numba(pos_b, box_shape, num_b_target)

    # --- R2 Backward: X2 + X -> X2 + A ---
    # pos_x is reduced, new_a_from_x created
    pos_x, pos_a = reaction_hetero_replacement_numba(pos_x2, pos_x, pos_a, sigmas[3], kappas[3], half_dt, box_shape)
    pos_a = maintain_bath_numba(pos_a, box_shape, num_a_target) 

    # --- R2 Forward: X2 + X -> X2 + A ---
    # pos_a is reduced, new_x_from_a created
    pos_a, pos_x = reaction_hetero_replacement_numba(pos_x2, pos_a, pos_x, sigmas[2], kappas[2], half_dt, box_shape)
    pos_a = maintain_bath_numba(pos_a, box_shape, num_a_target)
   
    return pos_x, pos_x2, pos_a, pos_b


# ==========================================
#      THE COMPILED STEP WRAPPER
# ==========================================
@njit
def run_single_step_compiled(pos_x, pos_x2, pos_a, pos_b, sigmas, kappas, 
                            diffusions, h, box_shape, num_a_target, num_b_target):
    # ========================================
    #                Reaction 
    # ========================================
    pos_x, pos_x2, pos_a, pos_b = run_one_step_numba(pos_x, pos_x2, pos_a, pos_b,
                                                     sigmas, kappas, h, box_shape, 
                                                     num_a_target, num_b_target)
    # ========================================
    #         Diffusion + Periodic
    # ========================================

    pos_x = diffusion_periodic_step_numba(pos_x, diffusions[0], h, box_shape)
    pos_x2 = diffusion_periodic_step_numba(pos_x2, diffusions[1], h, box_shape)
    pos_a = diffusion_periodic_step_numba(pos_a, diffusions[2], h, box_shape)
    pos_b = diffusion_periodic_step_numba(pos_b, diffusions[3], h, box_shape)

    return pos_x, pos_x2, pos_a, pos_b


# ==========================================
#        INITIALIZATION HELPER
# ==========================================
@njit
def simul_initialize(initial_num:np.array, box_shape):
    '''
    initialize the particle positions in the simulation box
    '''
    n_x, n_x2, n_a, n_b = initial_num
    pos_x = generate_position_box_numba(n_x, box_shape)
    pos_x2 = generate_position_box_numba(n_x2, box_shape)
    pos_a = generate_position_box_numba(n_a, box_shape)
    pos_b = generate_position_box_numba(n_b, box_shape)

    return pos_x, pos_x2, pos_a, pos_b



'''
PROGRESSBAR
'''
def simul_run(t_f_steps, pos_x, pos_x2, pos_a, pos_b, sigmas, kappas, 
            diffusions, h, box_shape, num_a_target, num_b_target):
    
    # x_log, x2_log = [], [] 
    # leave the positions aside for now
    # could add the animations later but not necessary

    xandx2_log = np.empty((t_f_steps,2))
    
    with ProgressBar(total=t_f_steps) as progress:
        for i in range(t_f_steps):
            pos_x, pos_x2, pos_a, pos_b = run_single_step_compiled(pos_x, pos_x2, 
                    pos_a, pos_b, sigmas, kappas, diffusions, h, box_shape, num_a_target, num_b_target)
            xandx2_log[i, 0] = len(pos_x)
            xandx2_log[i, 1] = len(pos_x2)
            progress.update(1)

    return xandx2_log
