import numpy as np
from numba import njit
from numba_progress import ProgressBar

"""
Functions for the position update of particles in the system

with numba so the simulation is faster

the system is a simulation box with periodic boundary conditions
"""

# ==========================================
# CORE GEOMETRY & MATH HELPERS
# ==========================================


@njit
def generate_position_box_numba(num, box_shape):
    """Generates random positions inside the simulation box."""
    if num <= 0:
        return np.empty((0, 3), dtype=np.float64)
    
    pos = np.empty((num, 3), dtype=np.float64)
    # Assuming box_shape is [Lx, Ly, Lz]
    pos[:, 0] = np.random.uniform(0.0, box_shape[0], num)
    pos[:, 1] = np.random.uniform(0.0, box_shape[1], num)
    pos[:, 2] = np.random.uniform(0.0, box_shape[2], num)
    return pos


@njit
def generate_sphere_offsets_numba(num, R):
    """
    Generates random offsets within a sphere of radius R.
    """
    if num <= 0:
        return np.empty((0, 3), dtype=np.float64)

    # 1. Generate random directions (Gaussian)
    dirs = np.random.normal(0.0, 1.0, (num, 3))
    
    # 2. Normalize manually (Avoids np.linalg.norm with axis)
    for i in range(num):
        # Calculate norm squared
        sq_norm = dirs[i, 0]**2 + dirs[i, 1]**2 + dirs[i, 2]**2
        norm = np.sqrt(sq_norm)
        
        # Avoid division by zero (unlikely but safe)
        if norm > 0:
            dirs[i, 0] /= norm
            dirs[i, 1] /= norm
            dirs[i, 2] /= norm
        else:
            dirs[i, 0] = 1.0 # fallback
    
    # 3. Radius (Cube root for uniform volume distribution)
    u = np.random.uniform(0.0, 1.0, (num, 1))
    radii = R * u**(1.0/3.0)
    
    # 4. Scale
    return dirs * radii


@njit
def get_dist_sq(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2


@njit
def diffusion_periodic_step_numba(pos, diffusion, h, box_shape):
    """
    Apply Diffusion + Periodic Boundaries in a single, fast pass.
    """
    n = len(pos)
    
    # 1. Diffusion Kick
    # scale = sqrt(2 * D * dt)
    scale = np.sqrt(2.0 * diffusion * h)
    
    # Generate random kicks for (N, 3)
    # explicitly passing size=(n, 3) is safer/clearer than pos.shape in Numba
    kicks = np.random.normal(0.0, 1.0, size=(n, 3)) * scale
    
    # 2. Update Position & Check Boundaries (One Loop)
    for i in range(n):
        for dim in range(3):
            # Apply Kick
            new_val = pos[i, dim] + kicks[i, dim]
            
            # Apply Periodic Boundary Condition (Wrap)
            if new_val < 0:
                new_val = new_val + box_shape[dim]
            elif new_val > box_shape[dim]:
                new_val = new_val - box_shape[dim]
                
            pos[i, dim] = new_val

    return pos



##### The following method is discarded here
##### Because this specific method may change the dynamic of the simulation 
##### of the corresponding reaction
"""
@njit
def generate_shell_offsets_numba(num, R):
    '''
    Generates random offsets ON THE SURFACE of a sphere of radius R.
    (Fixed distance, random direction)

    according to andrew & bray's paper
    '''
    if num <= 0:
        return np.empty((0, 3), dtype=np.float64)

    # 1. Generate random directions (Gaussian)
    dirs = np.random.normal(0.0, 1.0, (num, 3))
    
    # 2. Normalize manually to get unit vectors
    for i in range(num):
        sq_norm = dirs[i, 0]**2 + dirs[i, 1]**2 + dirs[i, 2]**2
        norm = np.sqrt(sq_norm)
        
        if norm > 0:
            dirs[i, 0] /= norm
            dirs[i, 1] /= norm
            dirs[i, 2] /= norm
        else:
            # Fallback for extremely rare (0,0,0) vector
            dirs[i, 0] = 1.0 
            dirs[i, 1] = 0.0
            dirs[i, 2] = 0.0
    
    # 3. Scale by fixed Radius R (No random u needed!)
    # This places everyone exactly at distance R
    return dirs * R
"""