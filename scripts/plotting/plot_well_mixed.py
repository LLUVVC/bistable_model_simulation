import matplotlib.pyplot as plt
import numpy as np
import random 
from scripts.analysis.data_loader import load_well_mixed_data
from scripts.analysis.analyze_distributions import find_the_best_bw, hist_np, kde_sk, get_pretty_upper_bound
from simulation.models.analytical_curve import get_analytical_curve
from simulation.solvers.rate_conversions import calculate_k_from_l
from scipy.stats import wasserstein_distance
from datetime import datetime
from scripts.analysis.cme2d_full_model import solve_cme_2d
from scripts.analysis.stationarity import snapshot

"""
1. Plot the distribution of species X throughout the simulation process, using data aggregated from all trajectories 

generated under the same parameter settings.

2. Compare the histograms and KDEs of the empirical distribution against the analytical distribution curve of the referenced 

Schlögl model.

3. Quantitatively measure the model fit by calculating the Wasserstein distance between the KDE and the analytical curve.

4. For the well-mixed simulation, data is recorded in continuous simulation time over the interval [0, t_f], while the spatial 

simulation are recorded at discrete time-step intervals [0, t_f/tau]. To facilitate a direct and intuitive comparison, the x-axes 

of all trajectory plots are normalized to represent the actual simulation timespan.

"""

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


def select_samples(X, times, snapshot_time=None):
    if snapshot_time is None:
        return X.ravel()               # as before: everything that was saved
    return X[:, np.argmin(np.abs(times - snapshot_time))]

def plot_traj(trajectories, axs, step_val, color_x, color_x2, full=False):
    for i, traj in enumerate(trajectories):
        ax = axs[i]
        x_time = traj['timescale'][::step_val] 
        y_x = traj['species_log']['X'][::step_val]
        ax.plot(x_time, y_x, color=color_x, label='X', drawstyle='steps-post', alpha=0.9, linewidth=1.5, zorder=1)
        # add a subtle shaded area under the curves
        ax.fill_between(x_time, y_x, color=color_x, step='post', alpha=0.1)

        if full:
            
            y_x2 = traj['species_log']['X2'][::step_val]
            ax.plot(x_time, y_x2, color=color_x2, label='X2', drawstyle='steps-post', alpha=0.9, linewidth=1.5, zorder=2)
            ax.fill_between(x_time, y_x2, color=color_x2, step='post', alpha=0.1)
        
        ax.set_xlabel('Time ($t$)')
        ax.set_ylabel('Particle Count')
        ax.set_title(f'Trajectory {i+1}')
        ax.legend(fontsize='small', loc='upper right') # Or 'upper left', etc.
        ax.grid(True, linestyle='--', alpha=0.4, which='both')
        # clean up the frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(left=traj['timescale'][0])

    return axs

def plot_time_average(combined_data_X, upper_bound, bin_width, band_width, 
                      macrorates, a, b, vol, ax, ori_len_traj, full=False):
    hist_bin, density_hist = hist_np(combined_data_X, upper_bound, bin_width)
    x_axis_plot, kde_X = kde_sk(combined_data_X, upper_bound, band_width)

    # analytical results
    # calculate the corresponding macroscopic reaction rates for the Schlögl model to get the analytical curve
    if full:
        macrorates_k = calculate_k_from_l(macrorates)
    else:
        macrorates_k = macrorates

    p_states, stat_dist = get_analytical_curve(upper_bound, macrorates_k, a, b, vol)
    W_d = wasserstein_distance(p_states, x_axis_plot, stat_dist, kde_X) # W(asserstein)_d(istance)
    print(f"the wasserstein distance is {W_d}")
    ax.bar(hist_bin, density_hist, width=bin_width, 
        color='#a9cce3', alpha=0.6, label='Simulation') # '#d5d8dc'# edgecolor='white',
    ax.plot(x_axis_plot, kde_X, color='#1f77b4', linewidth=3.5, 
            zorder=3, label='KDE') # '#2e4053'
    ax.plot(p_states, stat_dist, color='#e74c3c', 
            linewidth=2.5, zorder=4, label='Steady-state solution') # '#f39c12' # linestyle='--'
    ax.set_title(f'Combined trajectories: {ori_len_traj}')
    ax.set_xlabel('Particle Count')
    ax.set_ylabel('Probability')
    ax.set_xlim(0, upper_bound) # could change this, depending on the setting in simulation.model
    ax.legend(fontsize='small', loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    # clean up the frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax

def plot_snapshot(trajs, snap_list, axs_snap, upper_bound, bin_width, band_width):
    for i in range(len(snap_list)):
        ax=axs_snap[i]
        snapshot_time = snap_list[i]
        x_snap = snapshot(trajs, snapshot_time)
        hist_bin, density_hist = hist_np(x_snap, upper_bound, bin_width)
        x_axis_plot, kde_X = kde_sk(x_snap, upper_bound, band_width)
        ax.bar(hist_bin, density_hist, width=bin_width, 
                        color='#a9cce3', edgecolor='white', alpha=0.6, label='Simulation')
        ax.plot(x_axis_plot, kde_X, color='#1f77b4', linewidth=3.5, 
                    zorder=3, label='KDE') # '#2e4053'
        ax.set_title(f'time={snapshot_time}')
        ## where to put the number of trajs?
    return axs_snap

def plot_well_mixed(file_str, num_traj, bin_width=2., band_width=2.5714, optimize_bw=False, 
                    snap_list=None):
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
    step_val = max(1, len(trajectories[0]['timescale']) // 5000)   # about 5000 points per curve
    # print(len(trajectories[0]['timescale']), len(trajectories[0]['timescale'][::step_val]))
    print(f"For test, the slice size is {slice_val}") # comment out later
    # ============================================================
    # =============== DISTRIBUTION + TRAJ PLOTS ==================
    # ============================================================
    ori_len_traj = len(trajectories)
    num_traj = min(ori_len_traj, num_traj)
    if num_traj < ori_len_traj:
        selected_trajectories = random.sample(trajectories, num_traj)
    else:
        selected_trajectories = trajectories

    ncols = 2
    nrows = int(np.ceil(num_traj / ncols))
    plt.rcParams.update({
        'font.size': 18,          # General font size
        'axes.titlesize': 22,     # Subtitles & top case labels
        'axes.labelsize': 20,     # Axis labels (X and Y labels)
        'xtick.labelsize': 18,    # X-axis tick numbers
        'ytick.labelsize': 18,    # Y-axis tick numbers
        'legend.fontsize': 18     # Legend font size
    })
    fig_dist, ax = plt.subplots(figsize=(12, 6))
    header_height = 2.0  # Reserve exactly 2 inches for the title and text box
    fig_height = 4 * nrows + header_height
    fig_traj, axs = plt.subplots(nrows, ncols, figsize=(12, fig_height))
    # Flatten axes to 1D array for easy iteration
    axs = np.array(axs).reshape(-1)

    color_x = "#0072B2" 
    color_x2 = "#D55E00" 

    species_num = len(combined_data)

    ##### ----- preprocessing the data ----- #####
    combined_data_X = combined_data['X']
    combined_data_X = combined_data_X[::slice_val]
    upper_bound = get_pretty_upper_bound(combined_data_X)
    print(f"The calculated upper bound for #X is {upper_bound}")
    print(f"--- Total data points for distribution is {len(combined_data_X)} ---")

    if optimize_bw:
    
        print("Optimizing bandwidth via GridSearchCV. This might take a moment...")
        bw_X_data = combined_data_X[:, np.newaxis]
        band_width = find_the_best_bw(bw_X_data)
        
        print("Consider updating your default band_width parameter to this new value.")
    if species_num == 1:
        ##### model = 'Schlögl'

        if snap_list is None:

            plot_time_average(combined_data_X, upper_bound, bin_width, band_width, 
                      macrorates, a, b, vol, ax, ori_len_traj)
        
        else:
           fig_snap, axs_snap = plt.subplots(ncols=len(snap_list), figsize=(3*len(snap_list), 3))
           plot_snapshot(trajectories, snap_list, axs_snap, upper_bound, bin_width, band_width)

        ##### plot the trajctories 
        plot_traj(selected_trajectories, axs, step_val, color_x, color_x2)

    elif species_num == 2:
        ##### model = 'Full'
        
        if snap_list is None:
            plot_time_average(combined_data_X, upper_bound, bin_width, band_width, 
                                  macrorates, a, b, vol, ax, ori_len_traj, full=True)
        else:
            fig_snap, axs_snap = plt.subplots(ncols=len(snap_list), figsize=(3*len(snap_list), 3))
            plot_snapshot(trajectories, snap_list, axs_snap, upper_bound, bin_width, band_width)

        ##### plot the trajctories 
        plot_traj(selected_trajectories, axs, step_val, color_x, color_x2, full=True)
    else:
        print("Error.")
        return 0

    # t_f = metadata['timespan']
    # Create a string of formatted numbers separated by commas
    # formatted_rates = ", ".join([f"{x:.2f}" for x in macrorates])

    # exponent = int(np.floor(np.log10(tau)))
    # base = tau / 10**exponent
    # textstr = '\n'.join((
    #     rf"$\mathbf{{Rates_{{{model}}}}}$: {formatted_rates}",
    #     rf"$c_a ={a:.1f}\quad | \quad c_b ={b:.1f} \quad | \quad V={vol:.1f}$",
    #     rf"$\tau: {base:.2f} \times 10^{{{exponent}}} \quad | \quad T_{{final}}: {t_f:.1f} \quad | \quad W_d:{W_d:.5f}$" 
    # ))

    # props = dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.8)

    # fig_dist.text(0.5, 0.85, textstr, transform=fig_dist.transFigure, fontsize=8,
    #         ha='center', va='top', multialignment='left', bbox=props, linespacing=1.2)
    # fig_dist.suptitle(
    #     r"$\bf{Bistable\ System\ Analysis}$" + "\n" + 
    #     f"Well-Mixed {model} Distribution",
    #     fontsize=16, 
    #     y=0.98
    # )
    # fig_dist.tight_layout(rect=[0, 0.05, 1, 0.75])
    fig_dist.tight_layout()

    # --- Get timestamp for filename ---
    # Format as YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_str = f"{file_str}_{timestamp}"
    filename_dist = f"distribution.png" # timestamp as a benchmark
    filename_traj = f"trajectories.png"
    filename_snap = f"snapshots.png"
    # --- Save plot to the 'results' folder ---
    DATA_DIR = get_data_dir(file_str)

    # --- Create the directory if it doesn't exist ---
    os.makedirs(DATA_DIR, exist_ok=True)

    output_plot_path_dist = os.path.join(DATA_DIR, filename_dist)
    output_plot_path_traj = os.path.join(DATA_DIR, filename_traj)
    fig_traj.savefig(output_plot_path_traj)
    output_plot_path_snap = os.path.join(DATA_DIR, filename_snap)
    if snap_list:
        fig_snap.savefig(output_plot_path_snap)
    else:
        fig_dist.savefig(output_plot_path_dist)
    print(f"Saved trajectories and distribution plots to {DATA_DIR}")
    plt.close('all')



def main():
    """
    By default, the optimize_bw is set to False,

    it can be manually change to True if needed.
    
    """
    filestr =  "schloegl_model_0.15_0.025_tf_500.0_tau5e-07_initX_300" # "schloegl_model_0.15_0.025"

    snap_list = [50, 100, 250, 450]

    plot_well_mixed(filestr, num_traj=10, snap_list=snap_list, optimize_bw=False) # optimize_bw as an input




if __name__ == "__main__":
    main()