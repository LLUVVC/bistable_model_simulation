import numpy as np
import glob
import os
import re
import sys
from pathlib import Path


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
    data_dir = project_root /"simulation_data"/"well_mixed_data"/ file_str
    
    # 4. Create the folders automatically if they don't exist yet!
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory set to: {data_dir}")

    return data_dir



from simulation.solvers.well_mixed_process import Reaction_Full, Reaction_Schloegl

def run_save_well_mixed_full(NUM_RUNS_TO_DO, save_step, t_f, tau, ls, a, b, vol, DATA_DIR):
    
    print(f"the simulation reaction rates are l={ls}.")

    # This is the "burn-in" time. the simulations in [0, t_burn_in] are thrown away
    # of each simulation to let it reach the stable distribution.
    t_burn_in = 0.0 
    t_burn_in_steps = int(np.ceil(t_burn_in/tau))
    t_f_steps = int(np.ceil(t_f/tau))


    # parameters for the system
    # in the order X, X2, A, B
    reactant_full = np.array(((2,0,0,0),(0,1,0,0),(0,1,1,0),(1,1,0,0),(0,0,0,1),(1,0,0,0)))
    product_full = np.array(((0,1,0,0),(2,0,0,0),(1,1,0,0),(0,1,1,0),(1,0,0,0),(0,0,0,1)))
    bath_full = np.array((0,0,1,1)) # "1" denotes the bath's on, "0" off
    # reaction_full = product_full-reactant_full


    # Create the data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # --- PREVENT OVERWRITING ---
    # 1. Find all existing data files
    existing_files = glob.glob(os.path.join(DATA_DIR, "run_data_*.npz"))
    
    if not existing_files:
        # If no files, start at index 0
        start_index = 0
    else:
        # 2. Find the highest number in the existing filenames
        max_index = -1
        for f in existing_files:
            # Use regex to find the number in 'run_data_0004.npz'
            match = re.search(r'run_data_(\d+).npz', os.path.basename(f))
            if match:
                max_index = max(max_index, int(match.group(1)))
        
        # 3. Start at the *next* available index
        start_index = max_index + 1
        
    print(f"--- Starting Data Collection ---")
    print(f"Will run {NUM_RUNS_TO_DO} simulations, starting from index {start_index}.")

    keq = ls[0]/ls[1]

    for i in range(start_index, start_index + NUM_RUNS_TO_DO):
        print(f"Run {i}...")
        seed_value = np.random.randint(0,10000)
        print(f"the current seed value is {seed_value}.") # not necessary, keep for checking
        
        x_init = np.random.randint(low=0, high=400)
        ##### POISSON SAMPLING #####
        x2_init = np.random.poisson(x_init**2*keq/vol) # add the poisson distribution sampling
        initial_num = np.array((x_init, x2_init, a*vol, b*vol),dtype=int)
        ini_con = initial_num/vol # initial concentrations of chemicals
        full_model_bath = Reaction_Full(ls, reactant_full, product_full, ini_con, bath_full, volume=vol)
  
        print(f"  Starting the simulation of full model with Tau-Leaping algorithm (t_f={t_f}, tau={tau})...")
        particle_run_data, time_run_data = full_model_bath.full_tau_leaping(seed_value, t_f_steps, tau, save_steps=save_step)
        print(" ... the simulation of full model with Tau-Leaping algorithm finished.")

        # --- Data Collection ---
        burn_in_indices = np.where(time_run_data >= t_burn_in_steps)[0] # the data without the simulation within 't_burn_in'
        

        if len(burn_in_indices) == 0:
            # Simulation was shorter than burn-in time
            print(f"Warning: Simulation time ({t_f}s) is shorter than burn-in time ({t_burn_in}s). Saving no data.")
            data_to_save_X = np.array([])
            data_to_save_X2 = np.array([])
        else:
            burn_in_index = burn_in_indices[0] # Get the first index *after* burn-in
            data_to_save_X = particle_run_data[burn_in_index:, 0]  # Save X data
            data_to_save_X2 = particle_run_data[burn_in_index:, 1] # Save X2 data

        # --- Save to File ---
        # Use 4-digit padding for nice filenames (0000, 0001, 0002, ...)
        output_filename = os.path.join(DATA_DIR, f"run_data_{i:04d}.npz")
        # np.savez_compressed(output_filename, X=data_to_save_X, X2=data_to_save_X2, Time=time_run_data[burn_in_index:])
        np.savez_compressed(output_filename, X=data_to_save_X, X2=data_to_save_X2, Time=time_run_data[burn_in_index:],
        # --- parameters ---
        l=ls, tau=tau, vol=vol, t_f=t_f, a=a, b=b)
        # store "ls" with the name l
        print(f"  Successfully saved {len(data_to_save_X)} data points to {output_filename}")


    print("---")
    print(f"Finished {NUM_RUNS_TO_DO} runs.")
    print(f"All data saved in '{DATA_DIR}' folder.")
    print("Run the data_loader to read the data saved.")



def run_save_well_mixed_schloegl(NUM_RUNS_TO_DO, save_step, t_f, tau, k, a, b, vol, DATA_DIR):
    
    print(f"the simulation reaction rates are k={k}.")

    # This is the "burn-in" time. the simulations in [0, t_burn_in] are thrown away
    # of each simulation to let it reach the stable distribution.
    t_burn_in = 0.0 
    t_burn_in_steps = int(np.ceil(t_burn_in/tau))
    t_f_steps = int(np.ceil(t_f/tau))


    # parameters for the system
    # in the order [X, A, B]
    reactant_schloegl = np.array(((2,1,0),(3,0,0),(0,0,1),(1,0,0))) 
    product_schloegl = np.array(((3,0,0),(2,1,0),(1,0,0),(0,0,1)))
    bath_schloegl = np.array((0,1,1)) # in the order [X, A, B], with bath denoted as '1'
  

    # Create the data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # --- PREVENT OVERWRITING ---
    # 1. Find all existing data files
    existing_files = glob.glob(os.path.join(DATA_DIR, "run_data_*.npz"))
    
    if not existing_files:
        # If no files, start at index 0
        start_index = 0
    else:
        # 2. Find the highest number in the existing filenames
        max_index = -1
        for f in existing_files:
            # Use regex to find the number in 'run_data_0004.npz'
            match = re.search(r"run_data_(\d+).npz", os.path.basename(f))
            if match:
                max_index = max(max_index, int(match.group(1)))
        
        # 3. Start at the *next* available index
        start_index = max_index + 1
        
    print(f"--- Starting Data Collection ---")
    print(f"Will run {NUM_RUNS_TO_DO} simulations, starting from index {start_index}.")


    for i in range(start_index, start_index + NUM_RUNS_TO_DO):
        print(f"Run {i}...")
        seed_value = np.random.randint(0,10000)
        print(f"the current seed value is {seed_value}.") # not necessary, keep for checking
        
        ###### the "400" is specifically for the current model
        ###### this part can be improved ... 
        x_init = np.random.randint(low=0, high=400) 
        ini_con = np.array((x_init/vol, a, b)) # initial concentrations of chemicals
        ###### schloegl model
        schloegl_model_bath = Reaction_Schloegl(k, reactant_schloegl, product_schloegl, ini_con, bath_schloegl, volume=vol)
        print(f"  Starting simulation of Schloegl model with Tau-Leaping algorithm (t_f={t_f}, tau={tau})...")
        particle_run_data, time_run_data = schloegl_model_bath.schloegl_tau_leaping(seed_value, t_f_steps, tau, save_step)
        
        
        print(" ... the simulation of Schloegl model with Tau-Leaping algorithm finished.")

        # --- Data Collection ---
        burn_in_indices = np.where(time_run_data >= t_burn_in_steps)[0] # the data without the simulation within 't_burn_in' 
        

        if len(burn_in_indices) == 0:
            # Simulation was shorter than burn-in time
            print(f"Warning: Simulation time ({t_f}s) is shorter than burn-in time ({t_burn_in}s). Saving no data.")
            data_to_save_X = np.array([])
        else:
            burn_in_index = burn_in_indices[0] # Get the first index *after* burn-in
            data_to_save_X = particle_run_data[burn_in_index:, 0]  # Save X data

        # --- Save to File ---
        # Use 4-digit padding for nice filenames (0000, 0001, 0002, ...)
        output_filename = os.path.join(DATA_DIR, f"run_data_{i:04d}.npz")
        # np.savez_compressed(output_filename, X=data_to_save_X, X2=data_to_save_X2, Time=time_run_data[burn_in_index:])
        np.savez_compressed(output_filename, X=data_to_save_X, Time=time_run_data[burn_in_index:], 
                            # metadata
                            k=k, tau=tau, vol=vol, t_f=t_f, a=a, b=b)

        print(f"  Successfully saved {len(data_to_save_X)} data points to {output_filename}")


    print("---")
    print(f"Finished {NUM_RUNS_TO_DO} runs.")
    print(f"All data saved in '{DATA_DIR}' folder.")
    print("Run the data_loader to read the data saved.")


def main():
    # ==========================================
    #           CONFIGURATION SWITCH
    # ==========================================
    MODEL_TO_RUN = "full"  # Options: "full" or "schloegl"
    # change the model to run manually

    # ==========================================
    # 1. SHARED SIMULATION PARAMETERS
    # ==========================================
    L = 2.0
    vol = L**3
    a = 10.0
    b = 20.0
    num_runs = 2
    t_f = 10
    tau = 1e-5
    save_every_n_steps = int(10000)

    print(f"--- Starting Batch Run for {MODEL_TO_RUN.upper()} model ---")

    # ==========================================
    # 2. MODEL-SPECIFIC EXECUTION
    # ==========================================
    if MODEL_TO_RUN == "full":
        # Parameters unique to the Full model
        l = np.array((1.5, 1500., 150., 25., 5.75, 25.)) # check it again 
        
        print(f"The simulation parameter is: {l}")
        
        # Safely create the folder name using f-strings
        file_str = f"full_model_{l[0]}_{l[1]}"
        DATA_DIR = get_data_dir(file_str)
        
        run_save_well_mixed_full(num_runs, save_every_n_steps, t_f, tau, l, a, b, vol, DATA_DIR)

    elif MODEL_TO_RUN == "schloegl":
        # Parameters unique to the Schloegl model
        k = np.array((0.15, 0.025, 5.75, 25.)) # corresponding to the "l" setting
        print(f"The simulation parameter is: {k}")
        
        # Safely create the folder name
        file_str = f"schloegl_model_{k[0]}_{k[1]}"
        DATA_DIR = get_data_dir(file_str)
        
        run_save_well_mixed_schloegl(num_runs, save_every_n_steps, t_f, tau, k, a, b, vol, DATA_DIR)

    else:
        print(f"Error: Unknown model type '{MODEL_TO_RUN}'")

if __name__ == "__main__":
    main()

    