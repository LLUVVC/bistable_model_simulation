"""
Read the simulation data from particle-based simulation or well-mixed simulatin

select certain time step from all trajs, which would be used to make distribution plots

NOT SURE IF NECESSARY
"""

import numpy as np
import random 

def snapshot(trajectories, t):
    xs = []
    for traj in trajectories:
        i = min(np.searchsorted(traj['timescale'], t), len(traj['timescale']) - 1)
        xs.append(traj['species_log']['X'][i])   # first saved point at or after t
    return np.array(xs)

### this is for the relaxation time. But i dont feel so sure about it, yet.
def p_up_curve(trajectories, times, thr=135):
    return np.array([np.mean(snapshot(trajectories, t) > thr) for t in times])

def ks_to_analytical(samples, p_states, stat_dist):
    cdf_ref = np.cumsum(stat_dist) / np.sum(stat_dist)
    xs = np.sort(samples)
    cdf_emp = np.searchsorted(xs, p_states, side='right') / len(xs)
    return np.max(np.abs(cdf_emp - cdf_ref))
#### tau-int
#### half-windows