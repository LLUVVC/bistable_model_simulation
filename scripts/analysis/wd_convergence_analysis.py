"""
Plot the trend of mean value and standard deviation of W_d to different
batch size;

Compare the convergence of W_d wrt batch size as well as the simulation
timestep tau;

A method to decide whether we have enough data from the simulation to gauge
the quality of the current model with its specific paramter settings.

"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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


from scripts.analysis.data_loader import load_well_mixed_data, load_spatial_full_data
from scipy.stats import wasserstein_distance
from simulation.models.analytical_curve import get_analytical_curve
from simulation.solvers.rate_conversions import calculate_k_from_l
from scipy.stats import wasserstein_distance
from scripts.analysis.analyze_distributions import get_pretty_upper_bound, kde_sk

def calculate_wd_for_sample(p_states, stat_dist, combined_data_X, upper_bound, band_width):
    """
    Calculates the Wasserstein distance for a specific sample of data.
    This function ONLY does math. No plotting.
    """
    # 1. Compute KDE for this specific sample
    x_axis_plot, kde_X = kde_sk(combined_data_X, upper_bound, band_width)
        
    # 2. Calculate and return W_d
    W_d = wasserstein_distance(p_states, x_axis_plot, stat_dist, kde_X)
    
    return W_d



def run_bootstrap_Wd(file_str, num_traj_per_batch, resolution, iterations=100):

    """
    Runs the bootstrap analysis for a given batch size.

    resolution = 'well_mixed' or 'spatial'

    """
    # 1. Load the data ONCE
    # AND also get the analytical curve to avoid repetitiously calculating the analytical results
    if resolution == 'well_mixed':
        trajectories, combined_data, metadata = load_well_mixed_data(file_str=file_str)

        # Extract parameters
        tau = metadata['timestep']
        macrorates = metadata['macrorates']
        a = metadata['a']
        b = metadata['b']
        vol = metadata['vol']
        upper_bound = get_pretty_upper_bound(combined_data['X'])

        slice_val = int(1e-5/tau)
        # get analytical curve
        model_type="Full" if len(trajectories[0]['species_log']) == 2 else "Schlögl"
        if model_type == "Full":
            macrorates_k = calculate_k_from_l(macrorates)
            p_states, stat_dist = get_analytical_curve(upper_bound, macrorates_k, a, b, vol)
        else:
            p_states, stat_dist = get_analytical_curve(upper_bound, macrorates, a, b, vol)
    
    elif resolution == 'spatial':
        trajectories, _, metadata = load_spatial_full_data(file_str=file_str)
        ###################################################################
        ##### haven't thought about the slicing for the spatial model #####
        ###################################################################
    else:
        print("wrong resolution!")
        return None

    results = []
    # 2. Run the bootstrap loop
    for _ in tqdm(range(iterations)):
        # Randomly select 'num_traj_per_batch' trajectories
        sampled_trajs = np.random.choice(trajectories, size=num_traj_per_batch, replace=True) # originally replace=False, however that introduces
                                                                                              # a problem of stds=0 when batch_size is equal to the pool size
        # Combine the data for just this sample
        sampled_X = np.concatenate([traj['species_log']['X'][::slice_val] for traj in sampled_trajs])
        
        # Calculate W_d using our lean math function
        wd = calculate_wd_for_sample(
            p_states, stat_dist, sampled_X, upper_bound, band_width=2.5714
        )
        results.append(wd)
        
    # 4. Return or save the results (mean, std, etc. -> move this part into calculation)
    return np.array(results), metadata


def main():
    resolution = "well_mixed"
    filestr =  resolution + "_data/" + "full_model_1.5_1500.0_tf_500_tau5e-7"
    batch_size = np.arange(50,101,20) # could adapt it to the actual number of files available, keep it as it is now 
    
    means = []
    stds = []
    n_iterations = 10 # trial number for iterations
    for bs in batch_size:

        results, metadata = run_bootstrap_Wd(filestr, bs, resolution, iterations=n_iterations) 
        means.append(np.mean(results))
        stds.append(np.std(results))
        # NOT SURE if I should plot the metadata

    means = np.array(means)
    stds = np.array(stds)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(batch_size, means, '-o', color="#62BF04", label='Mean $W_d$', linewidth=2)
    ax.fill_between(batch_size, means - stds, means + stds, 
                    color="#47E10A", alpha=0.2, label='± 1 Std Dev')
    # Formatting the plot
    ax.set_xlabel('Batch Size (Number of Trajectories)')
    ax.set_ylabel('Wasserstein Distance ($W_d$)')
    ax.set_title(r'Convergence of $W_d$ with respect to Batch Size')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    filestr = "Wd_analysis/" + filestr
    DATA_DIR = get_data_dir(filestr)
    os.makedirs(DATA_DIR, exist_ok=True)
    output_filename = os.path.join(DATA_DIR, f"iter_{n_iterations}.npz")
    # np.savez_compressed(output_filename, X=data_to_save_X, X2=data_to_save_X2, Time=time_run_data[burn_in_index:])
    np.savez_compressed(output_filename, params=metadata, n_iter=n_iterations, mean=means, std=stds)
    filename_wd = f"wd_analysis_iter_{n_iterations}"
    output_plot_path_wd = os.path.join(DATA_DIR, filename_wd)
    fig.savefig(output_plot_path_wd)
    print(f"  Successfully saved analysis results to {output_filename}")
    plt.tight_layout()
    plt.show()

"""
TO DO:
Change the function to add the comparison of W_d between simulations of different tau(s).

"""   

if __name__ == "__main__":
    main()