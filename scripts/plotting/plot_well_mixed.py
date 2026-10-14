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

def count_switches(X, lo=100, hi=180):      # hysteresis band around the unstable point
    n = 0
    for traj in X:
        state = None
        for x in traj['species_log']['X']:
            if x > hi and state != 'u': state, n = 'u', n + (state is not None)
            elif x < lo and state != 'l': state, n = 'l', n + (state is not None)
    return n/len(X)

# def select_samples(X, times, snapshot_time=None):
#     if snapshot_time is None:
#         return X.ravel()               # as before: everything that was saved
#     return X[:, np.argmin(np.abs(times - snapshot_time))]


def pooled_samples(trajs, t_start=0.0, stride=1):
    """X values with t >= t_start from every trajectory, thinned by `stride`."""
    xs = []
    for traj in trajs:
        #print(traj.keys())
        i0 = np.searchsorted(traj['timescale'], t_start)       # first saved point with t >= t_start
        xs.append(np.asarray(traj['species_log']['X'][i0::stride]))
    return np.concatenate(xs)


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


    
def plot_time_average(trajectories, upper_bound, bin_width, band_width, 
                      ax, ori_len_traj, t_start=0.0,stride=1, stat_sch=None, stat_full=None):
    
    combined_data_X = pooled_samples(trajectories, t_start, stride)
    hist_bin, density_hist = hist_np(combined_data_X, upper_bound, bin_width)
    x_axis_plot, kde_X = kde_sk(combined_data_X, upper_bound, band_width)

    # analytical results
    # calculate the corresponding macroscopic reaction rates for the Schlögl model to get the analytical curve
    if stat_sch is not None:
        p_states, stat_dist = stat_sch
        W_d = wasserstein_distance(p_states, x_axis_plot, stat_dist, kde_X) # W(asserstein)_d(istance)
        print(f"the wasserstein distance is {W_d}")
        ax.plot(p_states, stat_dist*100, color='#e74c3c', 
                    linewidth=2.5, zorder=4, label='Schlögl model') # '#f39c12' # linestyle='--'
    ax.bar(hist_bin, density_hist*100, width=bin_width, 
        color='#a9cce3', alpha=0.6) # '#d5d8dc'# edgecolor='white',
    ax.plot(x_axis_plot, kde_X*100, color='#1f77b4', linewidth=2.5, 
            zorder=3, label='Stationary dist. (Simulation KDE)') # '#2e4053'
    
    if stat_full is not None:
        stat_x, stat_p = stat_full
        ax.plot(stat_x, stat_p*100, color='black', linestyle='--', linewidth=3.5, label='Stationary dist. (2D-CME)')

    ax.set_title(f'N = {ori_len_traj}, t ≥ {t_start}')
    # ax.set_title(f'Combined trajectories: {ori_len_traj}')
    ax.set_xlabel('Particle Count X')
    ax.set_ylabel('Probability (%)')
    ax.set_xlim(0, upper_bound) # could change this, depending on the setting in simulation.model
    ax.legend(fontsize='small', loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    # clean up the frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax

def plot_snapshot(trajs, snap_list, axs_snap, upper_bound, bin_width, band_width, stat_sch=None, stat_full=None):
    
    for ax, t in zip(axs_snap, snap_list):
        x_snap = snapshot(trajs, t)
        hist_bin, density_hist = hist_np(x_snap, upper_bound, bin_width)
        x_axis_plot, kde_X = kde_sk(x_snap, upper_bound, band_width)
        print(f'The simulated KDE low peak area is {np.mean(x_snap<135)}.')
        if stat_sch is not None:
            p_states, stat_dist = stat_sch
            ax.plot(p_states, stat_dist*100, color='#e74c3c', 
                        linewidth=2.5, zorder=4, label='Schlögl model') # '#f39c12' # linestyle='--'
            if stat_full is None:
                print(f'The calculated low peak area is {stat_dist[:135].sum()}.') # , the high peak area is {stat_dist[:150].sum()}
            W_d = wasserstein_distance(p_states, x_axis_plot, stat_dist, kde_X) # W(asserstein)_d(istance)
            print(f"the wasserstein distance to the original Schlögl model is {W_d}.")
        ax.bar(hist_bin, density_hist*100, width=bin_width, 
                        color='#a9cce3', edgecolor='white', alpha=0.6)
        ax.plot(x_axis_plot, kde_X*100, color='#1f77b4', linewidth=2.5, 
                    zorder=3, label='Stationary dist. (Simulation KDE)') # '#2e4053'
        if stat_full is not None:
            stat_x, stat_p = stat_full
            print(f'The calculated low peak area is {stat_p[:135].sum()}.')   # lower-state weight, two thresholds # , the high peak area is {stat_p[150:].sum()}
            ax.plot(stat_x, stat_p*100, color='black', linewidth=3.5, linestyle='--', label='Stationary dist. (2D-CME)')  
            W_d = wasserstein_distance(stat_x[:upper_bound], x_axis_plot, stat_p[:upper_bound], kde_X) # W(asserstein)_d(istance)
            print(f"the wasserstein distance to the extended Schlögl model is {W_d}.")
            """ should i normalize wd by two peak difference: ~220 ???"""
        ax.set_title(f'time={t}')
        ax.set_ylabel('Probability (%)')
        ax.set_xlabel('Particle Count X')
        ax.set_xlim(0, upper_bound)
        ax.grid(True, linestyle='--', alpha=0.4, which='both')
        if t == snap_list[-1]:
            ax.legend()

    return axs_snap

def find_equilibrate_time(trajs, p_exact):

    times = np.arange(5, 500, 5)
    p_low = np.array([np.mean(snapshot(trajs, t) < 135) for t in times])
    se = np.sqrt(p_exact*(1-p_exact)/len(trajs))
    inside = np.abs(p_low - p_exact) < 3*se # 3 std
    outside = np.where(~inside)[0]
    if len(outside) == 0:
        t_eq = times[0]
    elif outside.max() == len(times) - 1:
        t_eq = None                          # still outside the band at the last time
    else:
        t_eq = times[outside.max() + 1]

    if t_eq is None:
        print(f"Not equilibrated within t_f = {times[-1]:.0f}: low fraction {p_low[-1]:.4f} vs exact {p_exact:.4f}")
    else:
        print(f"... equilibrated from t = {t_eq}")
    # t_eq = times[np.where(~inside)[0].max() + 1] if (~inside).any() else times[0]

    return t_eq

def plot_well_mixed(file_str, num_traj, bin_width=10., band_width=2.5714, optimize_bw=False, 
                    snap_list=None, traj_example=False):
    """
    Project Default:
    - We use bw=2.5714 (calculated via GridSearchCV on our reference dataset).
    
    Snap_list: time points we take a snapshot of each trajectory in the pool
    traj_example: if True, we print the randomly selected "num_traj" example trajs
    """
    file_str = "well_mixed_data/" + file_str 
    trajectories, combined_data, metadata = load_well_mixed_data(file_str=file_str)
    
    tau = metadata['timestep']
    a = metadata['a']
    b = metadata['b']
    vol = metadata['vol']
    macrorates = metadata['macrorates']
    tf = metadata['timespan']

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
    
    color_x = "#0072B2" 
    color_x2 = "#D55E00" 

    species_num = len(combined_data)

    ##### ----- preprocessing the data ----- #####
    combined_data_X = combined_data['X']
    combined_data_X = combined_data_X[::slice_val]
    upper_bound = get_pretty_upper_bound(combined_data_X)
    print(f"The calculated upper bound for #X is {upper_bound}")
    print(f"--- Total data points for distribution is {len(combined_data_X)} ---")

    average_switch = count_switches(trajectories, lo=100, hi=180)
    print(f'for current params, the average switch over timespan={tf} per traj is {average_switch},'+
          f'corresponding to {average_switch*24/tf} over timespan= 24.')
    
    if optimize_bw:
    
        print("Optimizing bandwidth via GridSearchCV. This might take a moment...")
        bw_X_data = combined_data_X[:, np.newaxis]
        band_width = find_the_best_bw(bw_X_data)
        
        print(f"Consider updating your default band_width parameter to this new value band width={band_width}.")
    if species_num == 1:
        ##### model = 'Schlögl'
        [p_states, stat_dist] = get_analytical_curve(upper_bound, macrorates, a, b, vol)
        teq = find_equilibrate_time(trajectories, stat_dist[:135].sum())
        print(f"For the current model and initial state, the equilibrated time for the system is around t={teq}.")
        if snap_list is None:
            fig_dist, ax = plt.subplots(figsize=(12, 6))
            plot_time_average(selected_trajectories, upper_bound, bin_width, band_width, 
                    ax, ori_len_traj, t_start=teq, stat_sch=[p_states, stat_dist])
            fig_dist.tight_layout()

        else:
            fig_snap, axs_snap = plt.subplots(ncols=len(snap_list), figsize=(6*len(snap_list), 6), sharey=True)
            plot_snapshot(trajectories, snap_list, axs_snap, upper_bound, bin_width, band_width, stat_sch=[p_states, stat_dist])
            fig_snap.suptitle(f'N={ori_len_traj}', fontsize=16)
        if traj_example:
            header_height = 2.0  # Reserve exactly 2 inches for the title and text box
            fig_height = 4 * nrows + header_height
            fig_traj, axs = plt.subplots(nrows, ncols, figsize=(12, fig_height), sharey=True)
            # Flatten axes to 1D array for easy iteration
            axs = np.array(axs).reshape(-1)
        ##### plot the trajctories 
            plot_traj(selected_trajectories, axs, step_val, color_x, color_x2)

    elif species_num == 2:
        ##### model = 'Full'
        Xmax = int(upper_bound+20)
        m = Xmax**2 * macrorates[0] / (vol*macrorates[1])
        Ymax = int(np.ceil(m + 6*np.sqrt(m))) + 10 # Ymax = int(np.ceil(Xmax**2*macrorates[0]/(macrorates[1]*vol)))
        stat_x = np.arange(0, Xmax+1)
        stat_p,_ = solve_cme_2d(macrorates, a, b, vol, Xmax=Xmax, Ymax=Ymax)
        teq = find_equilibrate_time(trajectories, stat_p[:135].sum())
        print(f"For the current model and initial state, the equilibrated time for the system is around t={teq}.")
        macrorates_k = calculate_k_from_l(macrorates)  
        p_states, stat_dist = get_analytical_curve(upper_bound, macrorates_k, a, b, vol)
        if snap_list is None:
            fig_dist, ax = plt.subplots(figsize=(12, 6))
            plot_time_average(selected_trajectories, upper_bound, bin_width, band_width, 
                                ax, ori_len_traj, t_start=teq,
                                stat_sch=[p_states, stat_dist], stat_full=[stat_x, stat_p])
            fig_dist.tight_layout()
        else:
            x_init = trajectories[0]['species_log']['X'][0]
            fig_snap, axs_snap = plt.subplots(ncols=len(snap_list), figsize=(6*len(snap_list), 6), sharey=True)
            plot_snapshot(trajectories, snap_list, axs_snap, upper_bound, bin_width, band_width, [p_states, stat_dist], [stat_x, stat_p])
            fig_snap.suptitle(rf'N={ori_len_traj}', fontsize=16) # , $X_0$={x_init}
        if traj_example:
            header_height = 2.0  # Reserve exactly 2 inches for the title and text box
            fig_height = 4 * nrows + header_height
            fig_traj, axs = plt.subplots(nrows, ncols, figsize=(12, fig_height), sharey=True)
            # Flatten axes to 1D array for easy iteration
            axs = np.array(axs).reshape(-1)
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
    output_plot_path_snap = os.path.join(DATA_DIR, filename_snap)

    if traj_example:
        fig_traj.savefig(output_plot_path_traj)
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
    filestr =  "schloegl_model_0.15_0.025_tf_500.0_tau5e-07_initX_100" # 'full_model_7.5_1500.0_tf_500.0_tau5e-7_initX_100' # 

    snap_list =  None # [50, 100, 250, 450, 495] # None 

    plot_well_mixed(filestr, num_traj=200, snap_list=snap_list, optimize_bw=False) # optimize_bw as an input



if __name__ == "__main__":
    main()