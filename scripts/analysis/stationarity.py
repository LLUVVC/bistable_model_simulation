"""
Read the simulation data from particle-based simulation or well-mixed simulatin

select certain time step from all trajs, which would be used to make distribution plots

"""

import matplotlib.pyplot as plt
import numpy as np
import random 
from scripts.analysis.data_loader import load_well_mixed_data
from scripts.analysis.analyze_distributions import find_the_best_bw, hist_np, kde_sk, get_pretty_upper_bound
from simulation.models.analytical_curve import get_analytical_curve
from simulation.solvers.rate_conversions import calculate_k_from_l
from scipy.stats import wasserstein_distance
from datetime import datetime

import os
from pathlib import Path

def get_data_dir(file_str: str) -> Path:
    try:
        # 1. running a python script
        project_root = Path(__file__).resolve().parent.parent.parent

    except NameError:
        # 2. running interactively in a jupyter notebook
        current_dir = Path(os.getcwd()).resolve() # get current working directory
        project_root = current_dir.parent
    
    data_dir = project_root /"results"/file_str
    data_dir.mkdir(parents=True, exist_ok=True) # if the parent folder does not exist yet, create them
                                                # if the folder already exist, move on to next step without crash
    print(f"Data directory set to: {data_dir}")

    return data_dir



    """
Project Default:
- We use bw=2.5714 (calculated via GridSearchCV on our reference dataset).
"""
file_str = "well_mixed_data/" + file_str 
trajectories, combined_data, metadata = load_well_mixed_data(file_str=file_str)

tau = metadata['timestep']
a = metadata['a']
b = metadata['b']
vol = metadata['vol']
macrorates = metadata['macrorates']

# slice the data according to the size of tau
slice_val = int(1e-5/tau)
print(f"For test, the slice size is {slice_val}") # comment out later
# ============================================================
# =============== DISTRIBUTION + TRAJ PLOTS ==================
# ============================================================
ori_len_traj = len(trajectories)
num_traj = min(ori_len_traj, num_traj)
if num_traj < ori_len_traj:
    trajectories = random.sample(trajectories, num_traj)
