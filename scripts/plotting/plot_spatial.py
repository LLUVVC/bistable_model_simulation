import matplotlib.pyplot as plt
import numpy as np
import random 

from scripts.analysis.data_loader import load_spatial_full_data
from scripts.analysis.analyze_distributions import find_the_best_bw, hist_np, kde_sk, get_pretty_upper_bound
from simulation.models.analytical_curve import get_analytical_curve
from simulation.solvers.rate_conversions import calculate_k_from_l
from scipy.stats import wasserstein_distance
from datetime import datetime
from scripts.runners.run_spatial import make_diff_func

import os
from pathlib import Path


"""

For the well-mixed simulation, data is recorded in continuous simulation time over the interval [0, t_f], while the spatial 

simulation are recorded at discrete time-step intervals [0, t_f/tau]. To facilitate a direct and intuitive comparison, the x-axes 

of all trajectory plots are normalized to represent the actual simulation timespan.

"""


def get_data_dir(file_str: str) -> Path:
    try:
        # 1. running a python script
        project_root = Path(__file__).resolve().parent.parent.parent

    except NameError:
        # 2. running interactively in a jupyter notebook
        current_dir = Path(os.getcwd()).resolve() # get current working directory
        project_root = current_dir.parent
    
    data_dir = project_root /"results"/file_str
    # data_dir.mkdir(parents=True, exist_ok=True) # if the parent folder does not exist yet, create them
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



def plot_spatial(file_str, num_traj, slice_val, plot_spatial_dist=False, bin_width=2., band_width=2.5714, optimize_bw=False):
    """
    Project Default:
    - We use bw=2.5714 (calculated via GridSearchCV on our reference dataset).

    plot_spatial_dist: 
            Set to False as default. It will and only plot the distributions of particle X in space if
            it is set to be True.
                
    """

    print(f"------ The output is sliced every {slice_val} steps ------")
    print(f"------ for both trajs and distributions ------")

    ##### if no slice the data size is too large, slow to process ####


    file_str = "spatial_data/" + file_str

    trajectories, combined_data, collective_pos_X, pos_Time, metadata = load_spatial_full_data(file_str=file_str, slice_val=slice_val)
    
    ori_len_traj = len(trajectories)
    num_traj = min(ori_len_traj, num_traj)
    if num_traj < ori_len_traj:
        trajectories = random.sample(trajectories, num_traj)

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
    p = metadata['p']
    q = metadata['q']

    # plot_spatial_dist = False if p==0 else True # if p=0 it is homogeneous diffusion simulation 
    
    macrorates_k = calculate_k_from_l(macrorates)
    # --- the result folder to save plots ---
    DATA_DIR = get_data_dir(file_str)

    # --- select three examples to show the trajectories ---
    # --- plot all the trajectories for now ---

    # ============================================================
    # =============== DISTRIBUTION + TRAJ PLOTS ==================
    # ============================================================
    if not plot_spatial_dist:

        combined_data_X = combined_data['X']
        upper_bound = get_pretty_upper_bound(combined_data_X)
        print(f"The calculated upper bound for #X is {upper_bound}")

        if optimize_bw:
            
                print("Optimizing bandwidth via GridSearchCV. This might take a moment...")
                band_width = find_the_best_bw(combined_data_X[:, np.newaxis]) 
                print("Consider updating your default band_width parameter to this new value.")

        hist_bin, density_hist = hist_np(combined_data_X, upper_bound, bin_width)
        x_axis_plot, kde_X = kde_sk(combined_data_X, upper_bound, band_width)
        # --- plot the distribution first, otherwise the 'ax' would get overwritten when iterate through axs in trajectories
        fig_dist, ax = plt.subplots(figsize=(10,8))
        # analytical results
        # calculate the corresponding macroscopic reaction rates for the Schlögl model to get the analytical curve
        
        p_states, stat_dist = get_analytical_curve(upper_bound, macrorates_k, a, b, vol)

        W_d = wasserstein_distance(p_states, x_axis_plot, stat_dist, kde_X) # W(asserstein)_d(istance)
        
        ax.bar(hist_bin, density_hist, width=bin_width, 
                color='#a9cce3', edgecolor='white', alpha=0.6, label='Simulation')
        ax.plot(x_axis_plot, kde_X, color='#1f77b4', linewidth=2.5, 
                zorder=3, label='KDE')
        ax.plot(p_states, stat_dist, color='#e74c3c', linestyle='--', 
                linewidth=2, zorder=4, label='Analytical')
        

        ax.set_title(f'Combined trajectories: {ori_len_traj}')
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

        header_height = 2.0  # Reserve exactly 2 inches for the title and text box
        fig_height = 4 * nrows + header_height
        fig_traj, axs = plt.subplots(nrows, ncols, figsize=(12, fig_height))

        # Flatten axes to 1D array for easy iteration
        axs = np.array(axs).reshape(-1)

        # Modern color palette (Blue and Orange/Coral)
        color_x = "#0072B2" 
        color_x2 = "#D55E00" 

        for i, traj in enumerate(trajectories):
            ax = axs[i]
            x_time = traj['timescale'] * tau
            y_x = traj['species_log']['X']
            y_x2 = traj['species_log']['X2']
            ax.plot(x_time, y_x, color=color_x, label='X', drawstyle='steps-post', alpha=0.9, linewidth=1.5, zorder=1)
            ax.plot(x_time, y_x2, color=color_x2, label='X2', drawstyle='steps-post', alpha=0.9, linewidth=1.5, zorder=2)
            # add a subtle shaded area under the curves
            ax.fill_between(x_time, y_x, color=color_x, step='post', alpha=0.1)
            ax.fill_between(x_time, y_x2, color=color_x2, step='post', alpha=0.1)
            ax.legend(fontsize='small', loc='upper right') # Or 'upper left', etc.
            ax.grid(True, linestyle='--', alpha=0.4, which='both')
            ax.set_xlabel('Timespan ($t$)')
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
        
        # --- Dynamically calculate relative positions based on figure height ---
        title_y = 1.0 - (0.2 / fig_height)         # Title is always 0.2 inches from the top
        text_y = 1.0 - (1.0 / fig_height)          # Text box is always 1.0 inch from the top
        rect_top = 1.0 - (header_height / fig_height) # Plot grid stops exactly 2 inches from the top
        
        fig_traj.text(0.5, text_y, textstr, transform=fig_traj.transFigure, fontsize=8,
                ha='center', va='top', multialignment='left', bbox=props, linespacing=1.2)

        fig_traj.tight_layout(rect=[0, 0, 1, rect_top])
        fig_traj.suptitle("Bistable System Dynamics\n" + rf"$\mathrm{{Spatially\ Resolved\ Full\ Trajectories\ (Examples)}}$", 
                fontsize=16, y=title_y, fontweight='bold') 
        
        # --- Push the plots down ---
        # --- for distribution plot ---
        fig_dist.text(0.5, 0.88, textstr, transform=fig_dist.transFigure, fontsize=8,
                ha='center', va='top', multialignment='left', bbox=props, linespacing=1.2)
        # top=0.7 means the subplots only occupy the bottom 70% of the figure
        # fig_dist.subplots_adjust(top=0.7, bottom=0.15, hspace=0.3, wspace=0.3)
        
        fig_dist.tight_layout(rect=[0, 0, 1, 0.75])
        fig_dist.suptitle("Bistable System Analysis\n" + rf"$\mathrm{{Spatially\ Resolved\ Full\ Distribution}}$",
                        fontsize=16, y=0.98, fontweight='bold')

        # --- Get timestamp for filename ---
        # Format as YYYY-MM-DD_HH-MM-SS
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_str = f"{DATA_DIR}_{slice_val}_{timestamp}"
        filename_dist = f"distribution.png"
        filename_traj = f"trajectories.png" 
        
        # --- Create the directory if it doesn't exist ---
        os.makedirs(file_str, exist_ok=True)

        output_plot_path_dist = os.path.join(file_str, filename_dist)
        fig_dist.savefig(output_plot_path_dist)
        output_plot_path_traj = os.path.join(file_str, filename_traj)
        fig_traj.savefig(output_plot_path_traj)
        print(f"Saved trajectoris and distribution plots to {file_str}")

        # plt.show()

    if plot_spatial_dist: # if not plot_spatial_dist: -> change to this when I try to see if the system is well-mixed with smaller D.
        print(f" ----- The Diffusion function is: D(x) = -{p}*cos(pi*x) + {q}. -----")
        print(f" ----- Test: the input box shape is {box_shape}. -----")
        count_divisions = hetero_diffusion_dist(collective_pos_X, pos_Time, box_shape)
        plot_3d_spatial_distributions(DATA_DIR, count_divisions, a, b, macrorates_k, p, q, box_shape, bin_width=2.) 



def hetero_diffusion_dist(collective_pos_X, pos_Time, box_shape, num_division=5):
    """
    num_division = 5, default

    It decides the x-axis scale of the distribution in space
    [0, 400] -> (rescale by num_division) [0, 80]

    suppose the Diffusion function is along the X-axis -> index [:,0] in space
    """
    
    all_snapshots = []
    for traj in collective_pos_X:     # Loop through each run -> each entry is the saved "pos_x_obj_array"
        for snapshot in traj:         # Loop through all the snapshots in the run
                                      # corresponds to every entry in "pos_x_obj_array"
            all_snapshots.append(snapshot)
    
    ### If plot the snapshots of the particles in space -> add pos_Time
    ### Currently only look at the distribution in space

    total_snapshots = len(all_snapshots)
    x_axis_divisions = np.linspace(0, box_shape[0], num_division+1)
    count_divisions = np.zeros((total_snapshots, num_division))

    for j in range(total_snapshots):
        
        for i in range(num_division):
            
            left_bound = x_axis_divisions[i]
            right_bound = x_axis_divisions[i+1]
            
            x0 = (all_snapshots[j])[:,0] # x-axis
            within_range = np.where((left_bound <= x0) & (x0 < right_bound), x0, -1)
            count_divisions[j][i] = np.sum(within_range>0, axis=0)

    # print(f"--- Test: the test count_divisions in space is {count_divisions} ---")
    
    return count_divisions

def plot_3d_spatial_distributions(filestr, count_divisions, a, b, k, p, q,
                                  box_shape, bin_width=2.): # slice_val=1
    
    """
    plot the COUNT(X) distributions in each subspace,

    and compare them to the analytical curve.

    slice_val=1 -> there is no need to slice data with exact positions, bc it 
                   is alreay sliced when being saved.
    """

    # fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': '3d'})
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax_2d = fig.add_subplot(1, 2, 2)

    num_divisions = count_divisions.shape[1]
    dist_plot_pad = 0.5

    x_axis_divisions = np.linspace(0, box_shape[0], num_divisions+1)
    diff_func = make_diff_func(p, q)
    # find min and max D in each division
    diff_ranges = []
    for i in range(num_divisions):
        dense_x = np.linspace(x_axis_divisions[i], x_axis_divisions[i+1], 100)
        dense_D = np.array([diff_func(x)[0] for x in dense_x])
        diff_ranges.append((np.min(dense_D), np.max(dense_D)))
    
    # Modern colors for the 5 divisions
    colors = ['#4A90E2', '#F5A623', '#7ED321', '#D0021B', '#9013FE']
    linewidth_list = [5.0, 4.0, 3.0, 2.0, 1.5]
    
    upper_bound = get_pretty_upper_bound(count_divisions.reshape(-1,1)) # data from the highest diffusion division
    print(f"The calculated upper bound for #X is {upper_bound}")
    vol = np.prod(box_shape)/num_divisions 
    p_states, stat_dist = get_analytical_curve(upper_bound, k, a, b, vol)

    for i in range(num_divisions):
        raw_data = count_divisions[:, i]

        # average particle count for this specific region
        avg_count = np.mean(raw_data)

        hist_bin, density_hist = hist_np(raw_data, upper_bound, bin_width)
        # print(f"Test: for the round {i}, the sum of density hist is {np.sum(density_hist*bin_width)}")

        label_str = f"D: {diff_ranges[i][0]:.0f} ~ {diff_ranges[i][1]:.0f} | Avg X: {avg_count:.1f}" if p else ""
        ax.bar(hist_bin, density_hist, zs=num_divisions-i-dist_plot_pad, zdir='x', width=bin_width, color=colors[i], 
               edgecolor='white', linewidth=0.3, alpha=0.85, label=label_str)

        ax_2d.step(hist_bin, density_hist, where='mid', color=colors[i], 
                   linewidth=linewidth_list[i], alpha=0.85, label=label_str)

    ax.plot(p_states, stat_dist, zs=0, zdir='x', color='#c0392b', 
            linestyle='--', linewidth=2.5, zorder=1, alpha=0.9, label='Analytical Distribution')   
    ax_2d.plot(p_states, stat_dist, color='#c0392b', 
            linestyle='--', linewidth=2.5, alpha=0.6, label='Analytical Distribution')  

    ax.set_xlabel('Spatial Division', labelpad=10)
    ax.set_ylabel('Particle Count', labelpad=10)
    ax.set_zlabel('Probability Density', labelpad=10)

    ax.set_ylim(0, upper_bound)
    x_axis_boundaries = np.linspace(0, box_shape[0], num_divisions+1)
    x_labels = [f"[{x_axis_boundaries[i]:.1f}, {x_axis_boundaries[i+1]:.1f})" 
                for i in range(num_divisions)]
    ax.set_xticks(range(num_divisions))
    ax.set_xticklabels(x_labels)
    ax.set_title("3D Spatial Distribution of (X)", pad=20, fontsize=14, fontweight='bold')
    ax.view_init(elev=25, azim=-55)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.legend(loc='upper right')

    ax_2d.set_title("Analytical Stationary Distribution")
    ax_2d.set_xlabel("Particle Count")
    ax_2d.set_ylabel("Probability Density")
    ax_2d.grid(True)
    ax_2d.legend(loc='upper right')

    plt.tight_layout()
    print(f"Test: filestr is {filestr}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # --- Create the directory if it doesn't exist ---
    os.makedirs(filestr, exist_ok=True)
    filename = f"spatial_dist_{timestamp}.png"
    output_plot_path = os.path.join(filestr, filename)
    fig.savefig(output_plot_path)
    print(f"Saved trajectoris and distribution plots to {filestr}")

    # plt.show()


def main():

    filestr = "hetero_tf_4.0_D_750.0" # "homo_tf_24.0_D_1500.0" # "hetero_tf_12.0_D_1600.0"

    slice_val = 10000 # 1 10 100 1000 10000 
                     # The slice_val only affect the analysis of 
                     # simulations with homogeneous Diffusion coefficients
    plot_spatial(filestr, num_traj=10, slice_val=slice_val, plot_spatial_dist=True)




if __name__ == "__main__":
    main()