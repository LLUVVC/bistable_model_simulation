"""
Load the simulation data

for further quantitative analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import random
from datetime import datetime # Import the datetime module
from simulation.models.analytical_curve import get_analytical_curve


from pathlib import Path


def get_data_dir(file_str: str) -> Path:
    """
    Safely resolves the data directory whether run from a terminal script
    or imported inside a Jupyter Notebook.
    """
    try:
        # 1. SCENARIO: Running as a standard Python script (.py)
        # Assuming script is in project_root_folder/scripts/analysis/
        project_root = Path(__file__).resolve().parent.parent.parent # analysis > scripts > root folder
        
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


def load_well_mixed_data(file_str):
    """
    Universal loader for any well-mixed model (Schlögl or Full).
    Returns a dictionary of combined arrays and a list of individual trajectories.
    """
    DATA_DIR = get_data_dir(file_str)
    file_pattern = os.path.join(DATA_DIR, "run_data_*.npz")
    data_files = glob.glob(file_pattern)

    if not data_files:
        print(f" Error: No files found in {DATA_DIR}")
        return None, None, None

    # This dictionary will store lists of arrays for every key found in the files
    # e.g., storage['X'] = [array_run1, array_run2...]
    storage = {} # store the concatenated data of all trajectories for one species, separate by species name
    trajectories = [] # store (time, species_dict) for plotting lines
    metadata = {} # information about parameter setting

    for f in data_files:
        try:
            with np.load(f) as data: # equals to: data = np.load(f)
                # 1. Automatically detect what species are in this file
                # species_keys = 'X' for Schloegl; 'X', 'X2' for Full model
                species_keys = [i for i in data.files if i not in ['Time', 'ls', 'k', 'tau', 'vol']]
                
                
                
                # 2. Extract Species Data
                run_species_log = {}
                for s in species_keys:
                    
                    # Initialize storage list if it's the first time we see this species
                    if s not in storage: storage[s] = []
                    species_data = data[s]
                    storage[s].append(species_data)
                    run_species_log[s] = species_data

                # 3. Extract Time (used for trajectory plots)
                t_data = data['Time']

                # 4. Save individual trajectory
                trajectories.append({'timescale': t_data, 'species_log': run_species_log})
                
                # 5. Capture Metadata (ls or k) - only need to do this once
                if not metadata:
                    metadata['macrorates'] = data['l'] if 'l' in data.files else data['k']
                    metadata['timestep'] = data['tau'], metadata['timespan'] = data['t_f']

        except Exception as e:
            print(f" Error loading {os.path.basename(f)}: {e}") # return the file name only, rather than the full path 

    # Combine all data points for the Histogram/KDE
    combined_data = {s: np.concatenate(storage[s]) for s in storage}

    print(f" Loaded {len(data_files)} runs. Combined {len(combined_data['X'])} points.")
    return trajectories, combined_data, metadata



def load_spatial_full_data(file_str):
    """
    data loader for the spatial simulation data
    """

    DATA_DIR = get_data_dir(file_str)
    file_pattern = os.path.join(DATA_DIR, "run_data_diff_*.npz")
    data_files = glob.glob(file_pattern)

    if not data_files:
        print(f" Error: No files found in {DATA_DIR}")
        return None, None, None
    
    all_data_X, all_data_X2, trajectories = [], [], []
    metadata = {}
    for f in data_files:
        try:
            with np.load(f) as data:
                t_data = data['Time']

                run_species_log = {}

                all_data_X.append(data['X'])
                all_data_X2.append(data['X2'])
                run_species_log['X'] = data['X']
                run_species_log['X2'] = data['X2']
                trajectories.append({'timescale': t_data, 'species_log': run_species_log})

            if not metadata:
                metadata['macrorates'] = data['l'], metadata['microrates'] = data['kappa']
                metadata['timestep'] = data['tau'], metadata['timespan'] = data['t_f']
                # skip the a, b, volume values at the moment
                # they are all fixed for the current simulations

        except Exception as e:
            print(f" Error loading {os.path.basename(f)}: {e}") # return the file name only, rather than the full path 

    # Combine all data points for the Histogram/KDE
    combined_data = {}
    combined_data['X'] = all_data_X
    combined_data['X2'] = all_data_X2

    print(f" Loaded {len(data_files)} runs. Combined {len(combined_data['X'])} points.")
    return trajectories, combined_data, metadata


