import os
import numpy as np
from scipy.signal import find_peaks
from scripts.analysis.cme2d_full_model import solve_cme_2d

def k_to_l(keq, l1m):
    k = np.array((0.15, 0.025, 5.75, 25.))
    return np.array((l1m*keq, l1m, k[0]/keq, k[1]/keq, k[2], k[3]))

Xmax, a, b, vol = 420, 10, 20, 8
keq_grid = np.geomspace(1e-5, 5e-2, 40)          # same point density as before, extended left
l1m_list = np.array([300, 1500, 7500, 37500, 150000])
data_file = "schloegl_grid_results.npz"
shape = (len(l1m_list), len(keq_grid))

if os.path.exists(data_file):
    d = np.load(data_file)
    tau2_grid, tau3_grid = d["tau2_grid"], d["tau3_grid"]
    px_grid, n_peaks, p_low = d["px_grid"], d["n_peaks"], d["p_low"]
else:
    tau2_grid = np.full(shape, np.nan); tau3_grid = np.full(shape, np.nan)
    n_peaks = np.zeros(shape, dtype=int); p_low = np.full(shape, np.nan)
    px_grid = np.full(shape + (Xmax + 1,), np.nan)
    bnd_x = np.full(shape, np.nan); bnd_y = np.full(shape, np.nan)

    for i, l1m in enumerate(l1m_list):
        for j, keq in enumerate(keq_grid):
            l = k_to_l(keq, l1m)
            m = Xmax**2 * l[0] / (vol * l[1])
            Ymax = int(np.ceil(m + 6*np.sqrt(m))) + 10 ### Y refers to the number of X2
            px, P, t2, t3 = solve_cme_2d(l, a, b, vol, Xmax=Xmax, Ymax=Ymax)
            px = np.clip(px, 0, None)
            pk, _ = find_peaks(px, prominence=0.05*px.max())
            px_grid[i, j] = px
            n_peaks[i, j] = len(pk)
            p_low[i, j]   = px[:135].sum()
            tau2_grid[i, j], tau3_grid[i, j] = t2, t3
            bnd_x[i, j] = P[-3:, :].sum()      # mass in the last 3 X rows
            bnd_y[i, j] = P[:, -3:].sum()      # mass in the last 3 Y columns

        np.savez_compressed(data_file, keq_grid=keq_grid, l1m_list=l1m_list,
                            tau2_grid=tau2_grid, tau3_grid=tau3_grid,
                            px_grid=px_grid, n_peaks=n_peaks, p_low=p_low, bnd_x=bnd_x, bnd_y=bnd_y,
                            Xmax=Xmax, a=a, b=b, vol=vol)   # saved after each l1m row