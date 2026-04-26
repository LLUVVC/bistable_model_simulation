import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.data_loader import load_spatial_full_data
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



# Function to format a list of rates into LaTeX scientific notation
def format_rate_list(rates):
    formatted = []
    for r in rates:
        if 0.1 <= r <= 1000:
            formatted.append(rf"{r:.2f}")
        else:
            exp = int(np.floor(np.log10(r)))
            base = r / 10**exp
            formatted.append(rf"${base:.2f}{{\times}}10^{{{exp}}}$")
    return ", ".join(formatted)



def plot_spatial(file_str, bin_width=2., band_width=2.5714):
    """
    Project Default:
    - We use bw=2.5714 (calculated via GridSearchCV on our reference dataset).
    """
    file_str = "spatial_data/" + file_str
    trajectories, combined_data, metadata = load_spatial_full_data(file_str=file_str)

    # --- read the data about parameter settings ---
    macrorates = metadata['macrorates']
    microrates = metadata['microrates']
    t_f = metadata['timespan']
    tau = metadata['timestep']
    D = metadata['D']
    sigma = metadata['sigma']
    box_shape = metadata['box_shape']
    vol = np.prod(box_shape)
    a = metadata['a']
    b = metadata['b']

    # --- select three examples to show the trajectories ---
    # --- plot all the trajectories for now ---

    # ============================================================
    # =============== DISTRIBUTION + TRAJ PLOTS ==================
    # ============================================================

    combined_data_X = combined_data['X']
    upper_bound = get_pretty_upper_bound(combined_data_X)
    print(f"The calculated upper bound for #X is {upper_bound}")

    hist_bin, density_hist = hist_np(combined_data_X, upper_bound, bin_width)
    x_axis_plot, kde_X = kde_sk(combined_data_X, upper_bound, band_width)
    # --- plot the distribution first, otherwise the 'ax' would get overwritten when iterate through axs in trajectories
    fig_dist, ax = plt.subplots(figsize=(10,8))
    # analytical results
    # calculate the corresponding macroscopic reaction rates for the Schlögl model to get the analytical curve
    macrorates_k = calculate_k_from_l(macrorates)
    p_states, stat_dist = get_analytical_curve(upper_bound, macrorates_k, a, b, vol)

    W_d = wasserstein_distance(p_states, x_axis_plot, stat_dist, kde_X) # W(asserstein)_d(istance)
    
    ax.bar(hist_bin, density_hist, width=bin_width, 
            color='#a9cce3', edgecolor='white', alpha=0.6, label='Simulation') # '#d5d8dc'
    ax.plot(x_axis_plot, kde_X, color='#1f77b4', linewidth=2.5, 
            zorder=3, label='KDE') # '#2e4053'
    ax.plot(p_states, stat_dist, color='#e74c3c', linestyle='--', 
            linewidth=2, zorder=4, label='Analytical') # '#f39c12'
    

    ax.set_title(f'Combined trajectories: {len(trajectories)}')
    ax.set_xlabel('Particle Count')
    ax.set_ylabel('Probability')
    ax.set_xlim(0, upper_bound) # could change this, depending on the setting in simulation.model
    ax.legend(fontsize='small', loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    # clean up the frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- for trajectories plots ---
    num_traj = len(trajectories)
    ncols = 2
    nrows = int(np.ceil(num_traj / ncols))

    fig_traj, axs = plt.subplots(nrows, ncols, figsize=(12, 5*nrows))

    # Flatten axes to 1D array for easy iteration
    axs = np.array(axs).reshape(-1)

    step = 1 # Only plot every 1000th data point if step = 1000
    # set step small in the test phase, bc large step would slice out all the data

    # Modern color palette (Blue and Orange/Coral)
    color_x = "#0072B2" 
    color_x2 = "#D55E00" 

    for i, traj in enumerate(trajectories):
        ax = axs[i]
        x_time = traj['timescale'][::step] * tau
        y_x = traj['species_log']['X'][::step]
        y_x2 = traj['species_log']['X2'][::step]
        ax.plot(x_time, y_x, color=color_x, label='X', drawstyle='steps-post', alpha=0.9, linewidth=1.5, zorder=1)
        ax.plot(x_time, y_x2, color=color_x2, label='X2', drawstyle='steps-post', alpha=0.9, linewidth=1.5, zorder=2)
        # add a subtle shaded area under the curves
        ax.fill_between(x_time, y_x, color=color_x, step='post', alpha=0.1)
        ax.fill_between(x_time, y_x2, color=color_x2, step='post', alpha=0.1)
        ax.legend(fontsize='small', loc='upper right') # Or 'upper left', etc.
        ax.grid(True, linestyle='--', alpha=0.4, which='both')
        ax.set_xlabel('Time ($t$)')
        ax.set_ylabel('Particle Count')
        ax.set_title(f'Trajectory {i+1}')
        # clean up the frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(left=traj['timescale'][0])


    exponent = int(np.floor(np.log10(tau)))
    base = tau / 10**exponent
    
    textstr = '\n'.join((
        rf"$\mathbf{{Rates_{{macro}}}}$: {format_rate_list(macrorates)}",
        rf"$\mathbf{{Rates_{{micro}}}}$: {format_rate_list(microrates)}",
        rf"$\sigma: {sigma:.2f}\quad  | \quad D:{D:.1f} \quad  | \quad $Domain$: {box_shape[0]}\times {box_shape[1]}\times {box_shape[2]}\ (V={vol:.1f})\quad | \quad$BC: Periodic",
        rf"$c_a ={a:.1f}\quad | \quad c_b ={b:.1f} \quad | \quad \tau: {base:.2f} \times 10^{{{exponent}}} \quad | \quad T_{{final}}: {t_f:.1f} \quad | \quad W_d:{W_d:.5f}$" 
    ))

    props = dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.8)
    fig_traj.text(0.5, 0.88, textstr, transform=fig_traj.transFigure, fontsize=8,
            ha='center', va='top', multialignment='left', bbox=props, linespacing=1.2)

    fig_traj.subplots_adjust(top=0.7, bottom=0.15, hspace=0.3, wspace=0.3)
    
    fig_traj.tight_layout(rect=[0, 0, 1, 0.75])
    fig_traj.suptitle("Bistable System Dynamics\n" + rf"$\mathrm{{Spatially\ Resolved\ Full\ Trajectories}}$", 
            fontsize=16, y=0.98, fontweight='bold') 

    # --- Push the plots down ---
    # --- for distribution plot ---
    fig_dist.text(0.5, 0.88, textstr, transform=fig_dist.transFigure, fontsize=8,
            ha='center', va='top', multialignment='left', bbox=props, linespacing=1.2)
    # top=0.7 means the subplots only occupy the bottom 70% of the figure
    fig_dist.subplots_adjust(top=0.7, bottom=0.15, hspace=0.3, wspace=0.3)
    
    fig_dist.tight_layout(rect=[0, 0, 1, 0.75])
    fig_dist.suptitle("Bistable System Distribution\n" + rf"$\mathrm{{Spatially\ Resolved\ Full\ Distribution}}$",
                    fontsize=16, y=0.98, fontweight='bold')

    # --- Get timestamp for filename ---
    # Format as YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_str = f"{file_str}_{timestamp}"
    filename_dist = f"distribution.png" # timestamp as a benchmark
    filename_traj = f"trajectories.png" # timestamp as a benchmark
    # --- Save plot to the 'results' folder ---
    DATA_DIR = get_data_dir(file_str)
    # --- Create the directory if it doesn't exist ---
    os.makedirs(DATA_DIR, exist_ok=True)

    output_plot_path_dist = os.path.join(DATA_DIR, filename_dist)
    fig_dist.savefig(output_plot_path_dist)
    output_plot_path_traj = os.path.join(DATA_DIR, filename_traj)
    fig_traj.savefig(output_plot_path_traj)
    print(f"Saved trajectoris and distribution plots to {DATA_DIR}")

    plt.show()



def main():
    ####### edit the part for the best bandwidth
    ####### to do
    # new_bw = False
    # if new_bw:
    #     find_the_best_bw

    filestr =  "diff_equals_1500"

    plot_spatial(filestr)




if __name__ == "__main__":
    main()