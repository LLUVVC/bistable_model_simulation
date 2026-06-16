"""

Run the diffusion influenced simulations by batch and collect the data to draw the distribution

Intial conditions: the number of Particle X is uniformly sampled from possible states of the system
                   (this step not done yet)
                   the number of Particle X2 is sampled from a poisson distribution of the calculated equilibrated X

"""

import numpy as np
import glob
import os
import re
import math
from numba import njit

from simulation.solvers.rate_conversions import calculate_kappas, build_piecewise_kappa_table
from simulation.solvers.spatial_process import simul_initialize, simul_run
from pathlib import Path


from scipy.optimize import root_scalar, root


def get_data_dir(file_str: str) -> Path:
    """
    Safely resolves the data directory whether run from a terminal script
    or imported inside a Jupyter Notebook.
    """
    try:
        # 1. SCENARIO: Running as a standard Python script (.py)
        # Assuming script is in project_root_folder/scripts/runners/
        project_root = Path(__file__).resolve().parent.parent.parent # runner > scripts > root folder
        
    except NameError:
        # 2. SCENARIO: Running interactively in a Jupyter Notebook!
        # Notebooks don't have __file__. We use the Current Working Directory.
        current_dir = Path(os.getcwd()).resolve()
        
        project_root = current_dir.parent # notebooks are inside a 'notebooks' folder, go up one level
        

    # 3. Build the exact requested path to save simulated data
    data_dir = project_root /"simulation_data"/"spatial_data"/ file_str
    
    # 4. Create the folders automatically if they don't exist yet!
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory set to: {data_dir}")

    return data_dir



def initialization(box_shape, keq, c_a, c_b):
    """
    c_a and c_b are the initial concentrations
    keq=l1+/l1-
    """
    vol = np.prod(box_shape)
    print(f"The simulation volume is {vol}.")
    x_ini = np.random.randint(low=0, high=int(vol*50))
    ### sample the initial number of X2 based on a poisson distribution of the initial number of X
    x2_ini = np.random.poisson(x_ini**2*keq/vol)
    
    initial_num = np.array((x_ini, x2_ini, c_a*vol, c_b*vol),dtype=int)
    print(f"The initial numbers of particle X is {initial_num[0]}, particle X2 is {initial_num[1]}.")
    print(f"The initial numbers of particles in bath are: A = {initial_num[2]}, B = {initial_num[3]}.")
    pos_x, pos_x2, pos_a, pos_b = simul_initialize(initial_num, box_shape)

    return pos_x, pos_x2, pos_a, pos_b
    
def decoupled_l2_formula(kappa_1_plus, D, sigma):
     # Calculate the term inside the tanh function
    sqrt_term = np.sqrt(kappa_1_plus / (2 * D))
    
    # Calculate the Left-Hand Side (LHS) of the equation
    # This is the expression for the effective rate l_1^+
    tanh_val = np.tanh(sigma * sqrt_term)
    lhs = 8 * np.pi * D * (sigma - (1 / sqrt_term) * tanh_val)
    return lhs

def find_kappa_2(kappa_1_plus, l1_plus_input, D, sigma):
    """
    Defines the self-consistency equation for kappa_1_plus.
    This function will be zero when the correct kappa_1_plus is found.
    """
    # Ensure kappa_1_plus is positive to avoid math errors
    if kappa_1_plus <= 0:
        return np.inf # Return a large number if the guess is non-physical

    lhs = decoupled_l2_formula(kappa_1_plus, D, sigma)
    # Calculate the Right-Hand Side (RHS) of the equation
    rhs = l1_plus_input
    return lhs -rhs

def run_save_spatial(num, t_f, tau, ls, sigmas, diffusions, c_a, c_b, box_shape, result_dir, p, q, hetero=False):
    
    keq = ls[0]/ls[1]
    DA, DX, DX2 = diffusions[2], diffusions[0], diffusions[1]
    
    print(f"initializing the simulation box...")

    NUM_RUNS_TO_DO = num       # Number of simulations to run *this time*
    DATA_DIR = result_dir      # the folder to save the simulated data

    # This is the "burn-in" time. We throw away the first 't_brun_in' period of simulation
    # of each simulation to let it reach the stable distribution.
    t_burn_in = 0.0 # 0.05, for test 
    t_burn_in_steps = int(np.ceil(t_burn_in/tau))
    t_f_steps = int(np.ceil(t_f/tau))


    # Create the data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # --- PREVENT OVERWRITING ---
    # 1. Find all existing data files
    existing_files = glob.glob(os.path.join(DATA_DIR, "run_data_spatial_*.npz")) # diff stands for diffusion
    
    if not existing_files:
        # If no files, start at index 0
        start_index = 0
    else:
        # 2. Find the highest number in the existing filenames
        max_index = -1
        for f in existing_files:
            # Use regex to find the number in 'run_data_0004.npz'
            match = re.search(r'run_data_spatial_(\d+)', os.path.basename(f))
            if match:
                max_index = max(max_index, int(match.group(1)))
        
        # 3. Start at the *next* available index
        start_index = max_index + 1

        
    print(f"--- Starting Data Collection ---")
    print(f"Will run {NUM_RUNS_TO_DO} simulations, starting from index {start_index}.")

    for i in range(start_index, start_index + NUM_RUNS_TO_DO):
        print(f"Run {i}...")
        
        pos_x, pos_x2, pos_a, pos_b, = initialization(box_shape, keq, c_a, c_b)

        n_a, n_b = len(pos_a), len(pos_b)
        print(f"  Starting the spatial-resolution simulation of the full model (t_f={t_f}, tau={tau})...")

        if hetero:
            """
            The Diffusion is no longer a constant in the simulation box, but now
            D = D(x), x represents the x-axis of the simulation box (2 * 2 * 2).
            Here I set D(x) to be a sinuous function of half cycle on x-axis,
            -> D(x) = -p * cos(omega*x) + q, where p, q are constants.
            Note: D(x) has to be @njit
            """
            ls_2 = ls.copy()
            ls_2[2] = 148
            ls_list = [ls, ls_2]
            # 1. build the table
            kappas = build_piecewise_kappa_table(ls_list, DA, DX, DX2, sigmas) 
            num_log, pos_x_log, pos_x2_log = simul_run(t_f_steps, pos_x, pos_x2, pos_a, pos_b, sigmas, kappas, 
                                           diffusions, tau, box_shape, num_a_target=n_a, num_b_target=n_b)
            time_log = np.arange(t_f_steps)

        else:

            kappas = calculate_kappas(ls, DA, DX, DX2, sigmas)
            solution = root_scalar(find_kappa_2, bracket=[1e4, 10e4], args=(150., 1500, 0.1), )
            k2p_sol = solution.root
            solution = root_scalar(find_kappa_2, bracket=[1e3, 10e3], args=(25., 1500, 0.1),  )
            k2m_sol = solution.root
            kappas[2] = k2p_sol
            kappas[3] = k2m_sol
            print(kappas) 
            kappas = kappas[:,np.newaxis]
            num_log, pos_x_log, pos_x2_log = simul_run(t_f_steps, pos_x, pos_x2, pos_a, pos_b, sigmas, kappas, 
                    diffusions, tau, box_shape, num_a_target=n_a, num_b_target=n_b)
            time_log = np.arange(t_f_steps)
            
        pos_x_log_timestep = int((len(time_log)-1)/(len(pos_x_log)-1)) # remove the first entry
        # print(f" ---- Test: the positions of X in the system are recorded every {pos_x_log_timestep} steps ----")

        print(" ... simulation finished.")
        # --- Data Collection ---
        # the data without the simulation within 't_burn_in'
        burn_in_indices = np.where(time_log >= t_burn_in_steps)[0] 


        if len(burn_in_indices) == 0:
            # Simulation was shorter than burn-in time
            print(f"Warning: Simulation time ({t_f}s) is shorter than burn-in time ({t_burn_in}s). Saving no data.")
            data_to_save_X = np.array([])
            data_to_save_X2 = np.array([])
            # Package the jagged list into an 'object' array, so it can be compressed into .npz 
            pos_x_obj_array = np.empty(1, dtype=object)
            pos_x2_obj_array = np.empty(1, dtype=object)
            
            return None 
        
        else:
            burn_in_index = burn_in_indices[0] # Get the first index *after* burn-in
            data_to_save_X = num_log[burn_in_index:, 0]  # Save X data
            data_to_save_X2 = num_log[burn_in_index:, 1] # Save X2 data
            pos_burn_in_index = int(np.ceil(burn_in_index/pos_x_log_timestep))
            pos_x_obj_array = np.empty(len(pos_x_log), dtype=object)
            
            # check if pos_x2_log exists and isn't empty
            if pos_x2_log is None:
                pos_x2_obj_array = np.empty(1, dtype=object)
                pos_x_log_burned = pos_x_log[pos_burn_in_index:]
                for k in range(len(pos_x_log_burned)):
                    pos_x_obj_array[k] = pos_x_log_burned[k]
                    
                    
            else:
                pos_x2_obj_array = np.empty(len(pos_x2_log), dtype=object)

                pos_x_log_burned = pos_x_log[pos_burn_in_index:]
                pos_x2_log_burned = pos_x2_log[pos_burn_in_index:]
                for k in range(len(pos_x_log_burned)):
                    pos_x_obj_array[k] = pos_x_log_burned[k]
                    
                for k in range(len(pos_x2_log_burned)):
                    pos_x2_obj_array[k] = pos_x2_log_burned[k]

            time_log_pos = time_log[::pos_x_log_timestep]


        # --- Save to File ---
        # Use 4-digit padding for nice filenames (0000, 0001, 0002, ...)
        pid = os.getpid() # the simulation with spatial resolution takes a long time, add the "pid" in case the data gets overwritten
                          # when we run multiple simulations at the same time
        output_filename = os.path.join(DATA_DIR, f"run_data_spatial_{i:04d}_pid{pid}.npz")                 

        np.savez_compressed(output_filename, X=data_to_save_X, X2=data_to_save_X2, Time=time_log[burn_in_index:],
                            # metadata
                            pos_X=pos_x_obj_array, pos_Time=time_log_pos[pos_burn_in_index:],
                            l=ls, kappa=kappas, tau=tau, box_shape=box_shape, t_f=t_f, a=c_a, b=c_b,
                            sigma=sigmas[0], D=diffusions[0], p=p, q=q, pos_X2=pos_x2_obj_array) # For simplicity, only one scalar for sigma and for D are saved
                                                              # since we assume they are all equal for different chemicals and reactions
        
        print(f"  Successfully saved {len(data_to_save_X)} data points to {output_filename}")


    print("---")
    print(f"Finished {NUM_RUNS_TO_DO} runs.")
    print(f"All data saved in '{DATA_DIR}' folder.")
    print("Run the data_loader to read the data saved.")


def main():
    """ Main execution block containing all physics parameters. """
    ###### ================================== 1. parameter setting =====================================
    L = 2. # cubic box length

    diff_scale = 1500. 
    DA = 1. 
    DB = 1. 
    DX = 1.  
    DX2 = 1. 

    diffusions = np.array((DX, DX2, DA, DB)) * diff_scale 
    diff_max = np.max(diffusions)
    ##### There are 6 reactions but only 4 sigma values
    ##### because the reactions B <-> X involve no sigma value
    sigmas = np.array((1., 1., 1., 1.)) * 0.1 # sigma_r1f, sigma_r1b, sigma_r2f, sigma_r2b

    box_shape = np.array((L, L, L,))
    
    ##### the Part to change freely for the corresponding simulation
    # Schloegl's model reaction rates
    k = np.array((0.15, 0.025, 5.75, 25.))
    print("Reaction rates for bistable schloegl's model: ",k)
    # full model reaction rates
    ls = np.array((1.5, 1500., 150., 25., 5.75, 25.))
    print("Reaction rates for bistable full model: ",ls)
    ##### as for the varied_l -> varied_kappa, i packed it into the run_spatial function

    ###### ---------- diffusion function -----------
    ###### obey the form: D(x) = - p * cos(omega * x) + q
    ###### continuous even at boundary -> smooth D' -> have to consider the spurious drift
    p = 0.
    q = 0.
    ###### we discard the heterogeneous diffusion implementation now so p=q=0


    
    state = "homo"
    ###### ---------- decide if kappa is homogeneous in space -----------
    hetero = False # True or False
    if hetero:
        state = "hetero"

    ###### ============================ actually start the simulations ===============================

    num_run = 1
    t_f = 12. # 20.
    tau = 1e-6 # 1e-6
    
    ###### ---------- check if the timestep is small enough for the Diffusion setup -----------
    simulation_is_feasible = True if np.sqrt((2*diff_max)*2*tau) < np.min(sigmas) else False
    file_str = f"{state}_kp_tf_{t_f}_D_{diff_max}_tau_{tau}"
    DATA_DIR = get_data_dir(file_str)

    if simulation_is_feasible:
        run_save_spatial(num_run, t_f, tau, ls, sigmas, diffusions, 10, 20, box_shape, DATA_DIR, p, q, hetero=hetero)
    else:
        print("Please reduce the simulation time intervals.")


if __name__ == "__main__":
    main()


"""

# Compiled ONCE when the script loads, not once per simulation run
def make_diff_func(p, q, box_shape_0):
    @njit
    def diff_func(pos):
        # D(x) = -p * cos(omega * x) + q
        # omega = (2*pi)/box_shape_0 
        omega = 2 * math.pi / box_shape_0
        D = -p * math.cos(omega * pos) + q
        # Derivative: dD/dx = omega* p * sin(omega * x)
        dD_dx = omega * p * math.sin(omega * pos)
        return D, dD_dx 
    return diff_func
    
    """
