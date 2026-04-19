import numpy as np
from numba import njit
from numba_progress import ProgressBar

"""
Module: well_mixed_process.py
Description: High-performance, JIT-compiled stochastic solver for the well-mixed Schlögl model
            and well-mixed full model only.

PERFORMANCE & ARCHITECTURE NOTE:
This solver is explicitly hardcoded for two models above rather than 
using a generalized, N-dimensional stoichiometry matrix. This is a deliberate optimization 
designed to support large-scale spatial/batch simulations.

This module is strictly optimized for production-grade execution speed.
"""


'''
this is only for the full model reaction
'''
@njit(nogil=True)
def full_fast_tau_leaping_loop(
    seed_value, # in the batch run, a new seed_value is generated for each round of simulation
    initial_state: np.array,
    t_f_steps: int,
    h: float,
    kappa: np.array,
    reaction_coef: np.array,
    volume: float,
    num_reactions: int,
    num_species: int,
    progress_hook,
    save_every_n_steps: int = 10000 # <-- We only save data this often
):
    """
    This is a Numba-compiled, high-speed version of the tau-leaping loop.
    It hard-codes the 6 propensities for the full-schloegl model.
    """
    
    np.random.seed(seed_value)

    # --- 1. Pre-allocate arrays for SAVED data ONLY ---
    num_saves = t_f_steps // save_every_n_steps
    saved_particles = np.zeros((num_saves + 1, num_species))
    saved_times = np.zeros(num_saves + 1)
    
    # --- 2. Set initial state ---
    current_state = initial_state.copy()
    current_time = 0.0
    saved_particles[0] = current_state
    saved_times[0] = current_time
    save_index = 1
    
    # --- 3. Main simulation loop (this will be C-code fast) ---
    props = np.zeros(num_reactions)
    poisson_jumps = np.zeros(num_reactions)

    l1p, l1m, l2p, l2m, l3p, l3m = kappa
    # --- PRE-CALCULATE CONSTANTS ---
    """
    Division is computationally expensive compared to multiplication.
    """
    c1p = l1p / volume
    c1m = l1m           # No volume term
    c2p = l2p / volume
    c2m = l2m / volume
    c3p = l3p
    c3m = l3m
    # print(reaction_coef)

    for i in range(t_f_steps):
        
        # --- 4. Fast Propensity Calculation (Hard-coded for your model) ---
        # This replaces your slow _generate_intensity_func
        # This logic exactly matches your code's scaling: kappa/V**(m-1)
        
        X, X2, A, B = current_state
        
        
        # R I: 2X -> X2 (m=2, V**(m-1) = V)
        props[0] = c1p * X * (X - 1.0)
        # R I: X2 -> 2X (m=1, V**(m-1) = 1)
        props[1] = c1m * X2
        
        # R II: X2 + A -> X2 + X (m=2, V**(m-1) = V)
        props[2] = c2p * X2 * A
        # R II: X2 + X -> X2 + A (m=2, V**(m-1) = V)
        props[3] = c2m * X2 * X
        
        # R III: B -> X (m=1, V**(m-1) = 1)
        props[4] = c3p * B
        # R III: X -> B (m=1, V**(m-1) = 1)
        props[5] = c3m * X
        
        while True: ### Rejection sampling
            # --- 5. Calculate Poisson numbers ---
            for j in range(num_reactions):
                rate = props[j] * h
                if rate < 0:
                    rate = 0.0  # Ensure rate is non-negative
                poisson_jumps[j] = np.random.poisson(rate)

            # --- 6. Update state ---
            delta = np.zeros(num_species)
            for j in range(num_reactions):
                delta += reaction_coef[j] * poisson_jumps[j]
          
            # Check if this step causes negative populations
            temp_state = current_state + delta
            if np.any(temp_state < 0):
                # REJECT: Do nothing, the loop will repeat and draw new numbers
                continue 
            else:
                # ACCEPT: Update state and break the loop
                # Ensure no negative particles
                current_state = temp_state
                break

        current_time += h
        
        # --- 7. Save data ONLY periodically ---
        if (i + 1) % save_every_n_steps == 0:
            # update the bar by batch size
            progress_hook.update(save_every_n_steps)
            if save_index < num_saves + 1:
                saved_particles[save_index] = current_state
                saved_times[save_index] = current_time
                save_index += 1
                
    return saved_particles, saved_times



@njit # ---> STILL SUPER SLOW, so not actually used
def full_fast_gillespie_loop(
    initial_state: np.array,
    t_f: float,
    kappa: np.array,
    bath_reaction_coef: np.array,
    volume: float,
    num_reactions: int,
    num_species: int,
    log_dt: float = 0.1
):
    """
    This is a stand-alone, Numba-compiled, hard-coded Gillespie algorithm
    specifically for the 6-reaction L-model.

    ASSUMPTIONS:
    - Species order in `initial_state` is: [A, B, X, X2]
    - kappa order is: [l1p, l1m, l2p, l2m, l3p, l3m]
    - bath_reaction_coef is the stoichiometry for non-bath species
    """
    
    # --- 1. Initialize State ---
    current_state = initial_state.copy()
    t = 0.0
    
    # --- 2. Pre-allocate Logging Arrays ---
    # We log at fixed time intervals (log_dt)
    num_logs = int(t_f / log_dt) + 2
    x_log = np.zeros((num_logs, num_species), dtype=np.float64)
    t_log = np.zeros(num_logs, dtype=np.float64)
    
    x_log[0] = current_state
    t_log[0] = t
    log_index = 1
    last_log_t = 0.0
    
    # Propensity array
    props = np.zeros(num_reactions, dtype=np.float64)
    
    # --- 3. Main Gillespie Loop ---
    l1p, l1m, l2p, l2m, l3p, l3m = kappa
    
    while t < t_f:
        
        # --- 4. Hard-coded Propensity Calculation ---
        # This is the fast part
        A, B, X, X2 = current_state
        
        # R1 fwd: 2X -> X2 (m=2)
        props[0] = (l1p / volume) * X * (X - 1.0)
        # R1 rev: X2 -> 2X (m=1)
        props[1] = l1m * X2
        
        # R2 fwd: X2+A -> X2+X (m=2)
        props[2] = (l2p / volume) * X2 * A
        # R2 rev: X2+X -> X2+A (m=2)
        props[3] = (l2m / volume) * X2 * X
        
        # R3 fwd: B -> X (m=1)
        props[4] = l3p * B
        # R3 rev: X -> B (m=1)
        props[5] = l3m * X
        
        # Ensure non-negative propensities
        for i in range(num_reactions):
            if props[i] < 0:
                props[i] = 0.0
                
        # --- 5. Calculate Tau and Mu ---
        props_sum = np.sum(props)
        if props_sum <= 0:
            # No reactions possible, simulation is over
            t = t_f
            break
            
        # Draw random numbers
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        # Calculate time step
        delta_t = -np.log(r1) / props_sum
        
        # Choose reaction
        props_cumsum = np.cumsum(props)
        mu = 0
        while props_cumsum[mu] < r2 * props_sum:
            mu += 1
            
        # --- 6. Update State ---
        t = t + delta_t
        
        # We apply the stoichiometry *only* to the non-bath species
        # (X and X2), as defined by bath_reaction_coef
        current_state = current_state + bath_reaction_coef[mu]
        
        # Ensure no negative particles
        # We only need to check X and X2 (indices 2 and 3)
        if current_state[2] < 0:
            current_state[2] = 0.0
        if current_state[3] < 0:
            current_state[3] = 0.0

        # --- 7. Log Data ---
        if t - last_log_t > log_dt:
            if log_index < num_logs:
                # Log the full state (A, B, X, X2)
                x_log[log_index] = current_state
                t_log[log_index] = t
                log_index += 1
            last_log_t = t
            
    # Add final state to log
    if log_index < num_logs:
        x_log[log_index] = current_state
        t_log[log_index] = t
        log_index += 1
        
    return x_log[:log_index], t_log[:log_index]



'''
this is only for the schloegl model reaction
'''

@njit(nogil=True)
def schloegl_fast_tau_leaping_loop(
    seed_value,
    initial_state: np.array,
    t_f_steps: int,
    h: float,
    kappa: np.array,
    reaction_coef: np.array,
    volume: float,
    num_reactions: int,
    num_species: int,
    progress_hook,
    save_every_n_steps: int = 10000 # <-- We only save data this often
):
    """
    This is a Numba-compiled, high-speed version of the tau-leaping loop.
    It hard-codes the 4 propensities for the schloegl model.
    """
    
    np.random.seed(seed_value)

    # --- 1. Pre-allocate arrays for SAVED data ONLY ---
    num_saves = t_f_steps // save_every_n_steps
    saved_particles = np.zeros((num_saves + 1, num_species))
    saved_times = np.zeros(num_saves + 1)
    
    # --- 2. Set initial state ---
    current_state = initial_state.copy()
    current_time = 0.0
    saved_particles[0] = current_state
    saved_times[0] = current_time
    save_index = 1
    
    # --- 3. Main simulation loop (this will be C-code fast) ---
    props = np.zeros(num_reactions)
    poisson_jumps = np.zeros(num_reactions)

    k1, k2, k3, k4 = kappa
    # --- PRE-CALCULATE CONSTANTS ---
    """
    Division is computationally expensive compared to multiplication.
    """
    c1p = k1 / (volume**2)
    c1m = k2 / (volume**2)         
    c2p = k3  # No volume term
    c2m = k4  # No volume term

    # print(reaction_coef)

    for i in range(t_f_steps):
        
        # --- 4. Fast Propensity Calculation (Hard-coded for your model) ---
        # This replaces your slow _generate_intensity_func
        # This logic exactly matches your code's scaling: kappa/V**(m-1)
        
        X, A, B = current_state
        
        
        # R I: 2X + A -> 3X (m=3, V**(m-1) = V^2)
        props[0] = c1p * X * (X - 1.0) * A
        # R I: 3X -> 2X + A (m=3, V**(m-1) = V^2)
        props[1] = c1m * X * (X - 1.0) * (X - 2.0)
        
        # R II: B -> X (m=1, V**(m-1) = 1)
        props[2] = c2p * B
        # R II: X -> B (m=1, V**(m-1) = 1)
        props[3] = c2m * X
        
        
        # --- 5. Calculate Poisson numbers ---
        for j in range(num_reactions):
            rate = props[j] * h
            if rate < 0:
                rate = 0.0  # Ensure rate is non-negative
            poisson_jumps[j] = np.random.poisson(rate)
        """
        if not np.all(poisson_jumps[2:]==0):
            print(poisson_jumps)
        """
        # --- 6. Update state ---
        delta = np.zeros(num_species)
        for j in range(num_reactions):
            delta += reaction_coef[j] * poisson_jumps[j]
        # print(delta)
        current_state += delta

        # for j in range(num_reactions):
        #    current_state += reaction_coef[j] * poisson_jumps[j]
        
        # Ensure no negative particles
        
        checkflag = False

        for j in range(num_species):
            if current_state[j] < 0: 
                checkflag = True
                break  # Optimization: Stop checking once finding one negative
        
        
        if checkflag:
            current_state -= delta # = current_state - delta
            checkflag = False

        current_time += h
        
        # --- 7. Save data ONLY periodically ---
        if (i + 1) % save_every_n_steps == 0:
            # update the bar by batch size
            progress_hook.update(save_every_n_steps)
            if save_index < num_saves + 1:
                saved_particles[save_index] = current_state
                saved_times[save_index] = current_time
                save_index += 1
                
    return saved_particles, saved_times



class StochasticModelBase:
    """
    Base class that handles the mathematics of initializing any stochastic reaction system.
    Do not instantiate this directly. Use the specific model subclasses.
    """
    def __init__(self, kappa: np.ndarray, R_reactant: np.ndarray, R_product: np.ndarray, 
                 x: np.ndarray, bath: np.ndarray, volume: float) -> None:
        
        self.kappa = kappa # the macroscopic reaction rates
        self.R_reactant = R_reactant
        self.R_product = R_product
        self.bath = bath
        self.volume = volume
        self.x_ini = x * self.volume

        self.reaction_coef = R_product - R_reactant
        self.num_reactions = int(len(kappa))
        self.species = int(len(R_reactant[0]))
        
        temp_ = R_product - R_reactant
        temp_ind = np.nonzero(bath)[0]
        temp_[:, temp_ind] = 0
        self.bath_reaction_coef = temp_
        self.bath_boolean = True if bath.any() > 0 else False
        self.threshold = 5e9
        self.coef_to_use = self.bath_reaction_coef if self.bath_boolean else self.reaction_coef



class Reaction_Full(StochasticModelBase):

    """Specific solver methods for the Full 6-Reaction Model."""
    
    ##### the actual solver I 
    def full_tau_leaping(self, seed_value, t_f_steps, h, save_steps):
        '''
        This is now a wrapper that calls the fast, Numba-compiled loop.
        '''
        
        # Define how often to save the data
        # Example: if t_f_steps=100M and h=0.00001, 
        # saving every 10000 steps = save 10,000 data points.
        # save_every = 10000 
        with ProgressBar(total=t_f_steps) as progress:

            # Call the fast JIT-compiled function
            x_log, time_log = full_fast_tau_leaping_loop(
                seed_value=seed_value,
                initial_state=self.x_ini,
                t_f_steps=t_f_steps,
                h=h,
                kappa=self.kappa,
                reaction_coef=self.coef_to_use,
                volume=self.volume,
                num_reactions=self.num_reactions,
                num_species=self.species,
                progress_hook=progress,
                save_every_n_steps = save_steps
            )
        
        return x_log, time_log

    ##### in fact this is not used in practical simulation, too slow
    def full_gillespies(self, t_f, log_dt=0.1):
        '''
        This is now a wrapper that calls the fast, Numba-compiled Gillespie loop.
        t_ini is assumed to be 0 for the fast loop.
        t_f is the final simulation time.
        log_dt is the time interval for saving data.
        '''
    
        
        # Call the fast JIT-compiled function
        # We pass self.x_ini (particle counts)
        x_log, time_log = full_fast_gillespie_loop(
            initial_state=self.x_ini,
            t_f=t_f,
            kappa=self.kappa,
            bath_reaction_coef=self.coef_to_use,
            volume=self.volume,
            num_reactions=self.num_reactions,
            num_species=self.species,
            log_dt=log_dt
        )
        
        # Return the logs. The third return value (x_dist) is
        # removed as the Numba function doesn't calculate it.
        return x_log, time_log




class Reaction_Schloegl(StochasticModelBase):

    """Specific solver methods for the Bistable Schlögl Model."""
    
    def schloegl_tau_leaping(self, seed_value, t_f_steps, h, save_steps):
        '''
        This is now a wrapper that calls the fast, Numba-compiled loop.
        '''
        
        # Define how often to save the data
        # Example: if t_f_steps=100M and h=0.00001, 
        # saving every 10000 steps = save 10,000 data points.
        # save_every = 10000 
        with ProgressBar(total=t_f_steps) as progress:

            # Call the fast JIT-compiled function
            x_log, time_log = schloegl_fast_tau_leaping_loop(
                seed_value=seed_value,
                initial_state=self.x_ini,
                t_f_steps=t_f_steps,
                h=h,
                kappa=self.kappa,
                reaction_coef=self.coef_to_use,
                volume=self.volume,
                num_reactions=self.num_reactions,
                num_species=self.species,
                progress_hook=progress,
                save_every_n_steps = save_steps
            )
        
        return x_log, time_log

    





    