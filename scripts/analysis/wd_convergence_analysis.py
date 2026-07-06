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
import glob
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
from joblib import Parallel, delayed



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


def _single_bootstrap_iteration(trajectories, num_traj_per_batch, slice_val, 
                                  upper_bound, p_states, stat_dist, band_width):
    """One single iteration — parallelisable."""
    sampled_trajs = np.random.choice(trajectories, size=num_traj_per_batch, replace=True)
    sampled_X = np.concatenate([traj['species_log']['X'][::slice_val] for traj in sampled_trajs])
    return calculate_wd_for_sample(p_states, stat_dist, sampled_X, upper_bound, band_width)




def run_bootstrap_Wd(file_str, num_traj_per_batch, resolution, iterations=100, n_jobs=5): # or: n_jobs=-1, but i dont wanna exhaust my laptop

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


    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap_iteration)(
            trajectories, num_traj_per_batch, slice_val,
            upper_bound, p_states, stat_dist, band_width=2.5714
        )
        for _ in tqdm((range(iterations)))
    )

    # 4. Return or save the results (mean, std, etc. -> move this part into calculation)
    return np.array(results), metadata


def single_group_wd_analysis(resolution, data_file, n_iterations = 100):
    
    filestr =  resolution + "_data/" + data_file
    batch_size = np.arange(20,101,20) # could adapt it to the actual number of files available, keep it as it is now 
    
    means = []
    stds = []
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
    filestr = "wd_analysis/" + filestr
    DATA_DIR = get_data_dir(filestr)
    os.makedirs(DATA_DIR, exist_ok=True)
    output_filename = os.path.join(DATA_DIR, f"iter_{n_iterations}_tau_{metadata['timestep']}.npz")
    # np.savez_compressed(output_filename, X=data_to_save_X, X2=data_to_save_X2, Time=time_run_data[burn_in_index:])
    np.savez_compressed(output_filename, params=metadata, n_iter=n_iterations, mean=means, std=stds)
    filename_wd = f"wd_analysis_iter_{n_iterations}"
    output_plot_path_wd = os.path.join(DATA_DIR, filename_wd)
    fig.savefig(output_plot_path_wd)
    print(f"  Successfully saved analysis results to {output_filename}")
    plt.tight_layout()
    plt.show()

def multi_group_wd_comparison(resolution, n_iterations=100, base_dir='.'):
    resolution = resolution + "_data"
    search_pattern = os.path.join(base_dir, "results/wd_analysis", resolution, "*", f"iter_{n_iterations}_tau_*.npz")
    files_to_read = glob.glob(search_pattern)

    if not files_to_read:
        print(f" Error: No files found.")
        return None, None, None
    
    all_data = {}
    for f in files_to_read:
        filename = os.path.basename(f)
        tau_value = filename.split("tau_")[1].replace(".npz", "")
        all_data[tau_value] = np.load(f, allow_pickle=True)
        print(f"Successfully loaded data for tau={tau_value}")

    first_key = list(all_data.keys())[0]
    params_dict = all_data[first_key]['params'].item()
    macrorates = params_dict['macrorates']

    batch_size = np.arange(20, 101, 20)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # A beautiful sequential palette (Dark Navy -> Purple -> Blue -> Teal)
    # Visually represents the parameter \tau shrinking!
    # colors = ['#1B2A4A', '#215F88', '#2B939B', '#63C3A1']
    # colors = ['#1D2D44', '#414868', '#745174', '#E06A6B']
    colors = ['#1A365D', '#2B6CB0', '#48BB78', "#532C83"]
    line_styles = ['-', '-', '--', ':'] 
    markers = ['o', 's', '^', 'D']
    # 1. SORT the data so the legend matches the top-to-bottom visual order
    # reverse=True makes the largest tau (1e-5) plot first
    sorted_taus = sorted(all_data.keys(), key=lambda x: float(x), reverse=True)

    for i, tau_str in enumerate(sorted_taus):
        mean = all_data[tau_str]['mean']
        std = all_data[tau_str]['std']
        
        # 2. Format Scientific Notation beautifully for the legend
        parts = tau_str.split('e')
        base = float(parts[0])
        exp = int(parts[1])
        if base == 1.0:
            latex_tau = rf'10^{{{exp}}}'
        else:
            latex_tau = rf'{base:g} \times 10^{{{exp}}}'

        # Make the last line (teal) dashed and use a square marker ('s')
        # line_style = '--' if i == 3 else '-'
        # marker_style = 's' if i == 3 else 'o'
        # Plot the main line with the dynamic styles
        ax.plot(batch_size, mean, linestyle=line_styles[i], marker=markers[i], 
                color=colors[i], linewidth=2.5, markersize=6, label=rf'$\tau = {latex_tau}$')
        # 3. Fix the shading (remove the harsh dashed lines, use clean soft alpha)
        ax.fill_between(batch_size, mean - std, mean + std, color=colors[i], alpha=0.1, edgecolor='none')


    ax.set_xlabel('Batch Size (Number of Trajectories)', fontsize=13)
    ax.set_ylabel(r'Wasserstein Distance $(W_d)$', fontsize=13)
    
    # Optional: Remove title for thesis, put it in LaTeX caption
    # ax.set_title(r'$W_d$ Convergence with Respect to Batch Size', fontsize=14, fontweight='bold')

    ax.set_xlim(batch_size[0] - 5, batch_size[-1] + 5)
    ax.set_ylim(bottom=1.0) 

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Move legend slightly out of the way of the data if needed
    ax.legend(fontsize=12, framealpha=0.9, loc='upper right')

    filestr = "wd_analysis/" + resolution
    DATA_DIR = get_data_dir(filestr)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    filename_wd = f"wd_comparison_iter_{n_iterations}_{macrorates}.png" # Save as PDF!
    output_plot_path_wd = os.path.join(DATA_DIR, filename_wd)
    fig.savefig(output_plot_path_wd)
    # fig.savefig(output_plot_path_wd, dpi=300, bbox_inches='tight')
    plt.show()    

def main():
    resolution = "well_mixed"
    n_iterations = 150
    data_file = "full_model_1.5_1500.0_tf_500_tau1e-5" # "full_model_1.5_1500.0_tf_500_tau5e-7"
    
    # single_group_wd_analysis(resolution, data_file, n_iterations=n_iterations) # or: "spatial"
    multi_group_wd_comparison(resolution, n_iterations=n_iterations)

"""
TO DO:
Change the function to add the comparison of W_d between simulations of different tau(s).

"""   

if __name__ == "__main__":
    main()