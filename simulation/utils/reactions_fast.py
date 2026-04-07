import numpy as np
from numba import njit
from numba_progress import ProgressBar
from simulation.utils.geometry_fast import generate_position_box_numba



@njit
def unimolecularSelectReactant_numba(pos_r, kappa, h):
    num_r = len(pos_r)
    react_prob = 1.0 - np.exp(-kappa * h)
    
    # 1. Generate random numbers
    rands = np.random.random(num_r)
    
    # 2. Boolean mask (True where reaction happens)
    mask = rands <= react_prob
    
    # 3. Get indices
    # np.nonzero returns a tuple of arrays, we want the first one
    return np.nonzero(mask)[0]


@njit
def SubstituteParticle_numba(pos_ori, pos_sub, replace_list):
    '''
    substitute the particles at positions of pos_ori on replace_list with 
    pos_sub, return the updated positions of pos_ori and pos_sub
    '''

    if len(replace_list) == 0:
        return pos_ori, pos_sub
    
    pos_new_products = pos_ori[replace_list]

    if pos_sub.size==0:
        pos_sub = pos_new_products
    else:
        pos_sub = np.vstack((pos_sub, pos_new_products))
    mask = np.ones(len(pos_ori), dtype=np.bool_)
    mask[replace_list] = False
    pos_ori = pos_ori[mask]

    return pos_ori, pos_sub


# ==========================================
# BATH MAINTENANCE
# ==========================================

@njit
def maintain_bath_numba(pos, box_shape, target_num):
    current = len(pos)
    diff = int(target_num - current)
    
    if diff == 0:
        return pos
    
    if diff > 0:
        # Add particles
        new_parts = generate_position_box_numba(diff, box_shape)
        if len(pos) == 0:
            return new_parts
        return np.vstack((pos, new_parts))
        
    else: 
        # Remove particles (diff is negative)
        num_remove = abs(diff)
        # Random shuffle indices to pick survivors
        indices = np.arange(current)
        np.random.shuffle(indices)
        
        # Keep the first (current - num_remove)
        n_keep = current - num_remove
        keep_indices = indices[:n_keep]
        pos = pos[keep_indices]

        return pos
    

######### the original function does not take into account the interaction with the 
######### 'image' of a particle introduced by periodic boundary condition
### ================================================================================
###                             below is updated code
### ================================================================================

@njit
def bimolecular_hetero_candidates_update(pos_r1, pos_r2, sigma, kappa, h, box_shape):
    
    sigma_sq = sigma * sigma
    prob = 1.0 - np.exp(-kappa * h)

    # --- PHASE REACT (Linear Loop) ---
    n1 = len(pos_r1)
    n2 = len(pos_r2)
    mask_1 = np.zeros(n1, dtype=np.bool_) # False if not reacted, True once reacted
    mask_2 = np.zeros(n2, dtype=np.bool_)
    
    react_idx_1 = []
    react_idx_2 = []

    candidates = np.empty((n1*n2, 2), dtype=np.int64)
    num_pair = 0
    for i in range(n1):
        for j in range(n2):
            diff = pos_r1[i] - pos_r2[j]
            diff = diff - np.round(diff/box_shape) * box_shape
            # Calculate Squared Distances (N1, N2)
            dist_sq = np.sum(diff**2)
            if dist_sq <= sigma_sq:
                candidates[num_pair] = [i, j]
                num_pair = num_pair + 1

    # --- OPTIONAL: SHUFFLE (Removes Bias) ---
    # Shuffling ensures particle 0 doesn't always react with neighbor 1 before neighbor 2.
    pair_indices = np.arange(num_pair)
    np.random.shuffle(pair_indices)

    # Iterate through the LIST of close pairs
    for k in pair_indices:#
        i = candidates[k, 0]
        j = candidates[k, 1]
        
        # Check if they are still available
        if mask_1[i] or mask_2[j]: 
            continue
            
        # Roll the dice
        if np.random.random() < prob:
            mask_1[i] = True
            mask_2[j] = True
            react_idx_1.append(i)
            react_idx_2.append(j)
            
    return np.array(react_idx_1), np.array(react_idx_2)

@njit
def bimolecular_homo_candidates_update(pos_r, sigma, kappa, h, box_shape):
    sigma_sq = sigma * sigma
    prob = 1.0 - np.exp(-kappa * h)
    
    # --- PHASE 1: SEARCH (Vectorized) ---
    # 1. Calculate vector differences (Broadcasting)
    # Shape: (N, N, 3) -> Warning: Memory heavy for N > 2000!
    diff = pos_r[:, np.newaxis, :] - pos_r[np.newaxis, :, :]
    diff = diff - np.round(diff / box_shape) * box_shape

    # 2. Calculate Squared Distances (N, N)
    dist_sq = np.sum(diff**2, axis=2)
    
    # 3. Get all indices where distance is valid
    # Returns an array of pairs [[i, j], [i, j], ...]
    candidates = np.argwhere(dist_sq < sigma_sq)
    
    # --- OPTIONAL: SHUFFLE (Removes Bias) ---
    # Shuffling ensures particle 0 doesn't always react with neighbor 1 before neighbor 2.
    np.random.shuffle(candidates)
    

    n = len(pos_r)
    mask = np.zeros(n, dtype=np.bool_)
    react_i = []
    react_j = []

    # Iterate through the LIST of close pairs
    for k in range(len(candidates)):
        i = candidates[k, 0]
        j = candidates[k, 1]
        # Skip cases when j>=i to avoid double counting 
        # in particular, when j==i we are dealing with the same particle 'self-reaction'
        # which we must get rid of
        if j >= i: 
            continue
        # Check if they are still available
        if mask[i] or mask[j]: 
            continue
            
        # Roll the dice
        if np.random.random() < prob:
            mask[i] = True
            mask[j] = True
            # instead of a list of tuples (pairs), return the indices separately
            # numba might not handle the tuples well
            react_i.append(i)
            react_j.append(j)
            
    return np.array(react_i), np.array(react_j)

@njit
def AddParticleHomoMid_numba_update(pos_r, pos_p, idx_1, idx_2, box_shape):
    '''
    add the product particle at the middle point of two reactants of the 
    same species.
    pos_r: positions of reactants
    pos_p: positions of products
    idx_1, idx_2: indices of reacted particles pair - (i,j), i in idx_1 and j in idx_2
    '''

    n_reactions = len(idx_1)
    if n_reactions == 0:
        return pos_r, pos_p
    
    pos1 = pos_r[idx_1]
    pos2 = pos_r[idx_2]
     
    diff = pos2 - pos1
    pos1 = pos1 + np.round(diff / box_shape) * box_shape # move particle to pos1 prime, in the 'box' adjacent to pos2
                                                         # (in the original box).
    pos_new_products = (pos1 + pos2) * 0.5
    pos_new_products = pos_new_products - np.floor(pos_new_products / box_shape) * box_shape
    
    if pos_p.size == 0:
        pos_p = pos_new_products
    else:
        pos_p = np.vstack((pos_p, pos_new_products))
    mask = np.ones(len(pos_r), dtype=np.bool_)
    
    # Mark as Dead
    mask[idx_1] = False
    mask[idx_2] = False
    pos_r = pos_r[mask]

    return pos_r, pos_p