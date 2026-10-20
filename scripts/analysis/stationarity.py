"""
Read the simulation data from particle-based simulation or well-mixed simulatin

select certain time step from all trajs, which would be used to make distribution plots

NOT SURE IF NECESSARY
"""

import numpy as np
import random 
from scripts.analysis.analyze_distributions import hist_np, kde_sk
from scipy.stats import wasserstein_distance

def snapshot(trajectories, t):
    xs = []
    for traj in trajectories:
        i = min(np.searchsorted(traj['timescale'], t), len(traj['timescale']) - 1)
        xs.append(traj['species_log']['X'][i])   # first saved point at or after t

    return np.array(xs)


def count_switches(X, lo=100, hi=180):      # hysteresis band around the unstable point
    n = 0
    for traj in X:
        state = None
        for x in traj['species_log']['X']:
            if x > hi and state != 'u': state, n = 'u', n + (state is not None)
            elif x < lo and state != 'l': state, n = 'l', n + (state is not None)
    return n/len(X)

def pooled_samples(trajs, t_start=0.0, stride=1):
    """X values with t >= t_start from every trajectory, thinned by `stride`."""
    xs = []
    for traj in trajs:
        #print(traj.keys())
        i0 = np.searchsorted(traj['timescale'], t_start)       # first saved point with t >= t_start
        xs.append(np.asarray(traj['species_log']['X'][i0::stride]))
    return np.concatenate(xs)

def plot_time_average(trajectories, upper_bound, bin_width, band_width, 
                      ax, ori_len_traj, t_start=0.0, stride=1, stat_sch=None, stat_full=None):
    
    # Ensure t_start is treated as a list for uniform iteration
    is_list = isinstance(t_start, list)
    t_list = t_start if is_list else [t_start]
    
    # Map ax to a list/array of axes matching t_list length
    if is_list:
        axes = ax if isinstance(ax, (list, np.ndarray)) else [ax] * len(t_list)
    else:
        axes = [ax]

    for i, t in enumerate(t_list):
        combined_data_X = pooled_samples(trajectories, t, stride)
        hist_bin, density_hist = hist_np(combined_data_X, upper_bound, bin_width)
        x_axis_plot, kde_X = kde_sk(combined_data_X, upper_bound, band_width)

        ax = axes[i]
        # analytical results
        # calculate the corresponding macroscopic reaction rates for the Schlögl model to get the analytical curve
        if stat_sch is not None:
            p_states, stat_dist = stat_sch
            W_d = wasserstein_distance(p_states, x_axis_plot, stat_dist, kde_X) # W(asserstein)_d(istance)
            print(f"the wasserstein distance is {W_d}")
            ax.plot(p_states, stat_dist*100, color='#e74c3c', 
                        linewidth=1.5, zorder=1, label='Schlögl model') # '#f39c12' # linestyle='--'
        ax.bar(hist_bin, density_hist*100, width=bin_width, 
            color='#a9cce3', alpha=0.6) # '#d5d8dc'# edgecolor='white',
        ax.plot(x_axis_plot, kde_X*100, color='#1f77b4', linewidth=1.5, 
                zorder=2, label='Stationary dist. (Simulation KDE)') # '#2e4053'
        
        if stat_full is not None:
            stat_x, stat_p = stat_full
            ax.plot(stat_x, stat_p*100, color='black', linestyle='--', linewidth=2., 
                    zorder=999, label='Stationary dist. (2D-CME)')

        ax.set_title(f'N = {ori_len_traj}, t ≥ {t}')
        # ax.set_title(f'Combined trajectories: {ori_len_traj}')
        ax.set_xlabel('Particle Count X')
        ax.set_ylabel('Probability (%)')
        ax.set_xlim(0, upper_bound) # could change this, depending on the setting in simulation.model
        if i==0:
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
            ax.plot(stat_x, stat_p*100, color='black', linewidth=3., linestyle='--', label='Stationary dist. (2D-CME)')  
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
    """
    from trajectories
    """
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


def ks_to_analytical(samples, p_states, stat_dist):
    cdf_ref = np.cumsum(stat_dist) / np.sum(stat_dist)
    xs = np.sort(samples)
    cdf_emp = np.searchsorted(xs, p_states, side='right') / len(xs)
    return np.max(np.abs(cdf_emp - cdf_ref))
#### tau-int
#### half-windows