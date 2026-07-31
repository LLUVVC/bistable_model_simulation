import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from pathlib import Path

def get_project_root():
    """Finds the root directory of the project assuming the script is run from inside the project."""
    try:
        # If running as a script (e.g. from scripts/runners/calc_gr.py)
        return Path(__file__).resolve().parent.parent.parent
    except NameError:
        # If running interactively
        return Path(os.getcwd()).resolve().parent

def calc_gr(pos_list, box_size, r_max, bins):
    """
    Calculates the radial distribution function g(r) from a list of particle positions.
    """
    edges = np.linspace(0, r_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    
    hist_sum = np.zeros(bins)
    pair_expected_sum = np.zeros(bins)
    vol = box_size**3
    
    for pos in pos_list:
        N = len(pos)
        if N < 2: continue
        
        # Calculate pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # Apply Periodic Boundary Conditions (PBC)
        diff = diff - box_size * np.round(diff / box_size)
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Extract upper triangle (unique pairs)
        i, j = np.triu_indices(N, k=1)
        d = dist[i, j]
        
        # Histogram the distances
        h, _ = np.histogram(d, bins=edges)
        hist_sum += h
        
        # Calculate expected number of pairs in an ideal gas for this shell
        shell_vol = (4.0/3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
        expected = 0.5 * N * (N-1) * (shell_vol / vol)
        pair_expected_sum += expected
        
    # Prevent division by zero
    valid = pair_expected_sum > 0
    gr = np.zeros_like(centers)
    gr[valid] = hist_sum[valid] / pair_expected_sum[valid]
    
    return centers, gr

def main():
    print("Starting g(r) calculation... This might take a few minutes.")
    project_root = get_project_root()
    base_dir = project_root / "simulation_data" / "spatial_data"
    
    # =========================================================
    # FLEXIBLE INPUT: Define your folder names and labels here
    # =========================================================
    folder_1 = 'homo_tf_24.0_D_1500.0'
    folder_2 = 'homo_tf_24.0_D_1500.0_tau_2e-07'
    
    label_1 = r'$\tau = 10^{-6}$ (Numerical Smearing)'
    label_2 = r'$\tau = 2\times 10^{-7}$ (Physical Spatial Correlations)'
    
    # How many of the final snapshots to average over (for steady state)
    num_snapshots = 2000 # 500 
    # =========================================================
    
    dir_1 = os.path.join(base_dir, folder_1)
    dir_2 = os.path.join(base_dir, folder_2)
    
    files_1 = glob.glob(os.path.join(dir_1, "*.npz"))
    files_2 = glob.glob(os.path.join(dir_2, "*.npz"))
    
    if not files_1:
        print(f"Error: Could not find any .npz files in {dir_1}")
        return
    if not files_2:
        print(f"Error: Could not find any .npz files in {dir_2}")
        return
    
    r_max = .2  # compute up to 5 sigma (since sigma=0.1)
    bins = 100
    box_size = 2.0
    sigma = 0.1
    
    # We will average the histograms over all files
    total_hist_1 = np.zeros(bins)
    total_hist_2 = np.zeros(bins)
    
    print(f"Processing {len(files_1)} files from {folder_1}...")
    for f in files_1:
        data = np.load(f, allow_pickle=True)
        pos = data['pos_X']
        # Take the final `num_snapshots`
        subset = pos[-num_snapshots:] if len(pos) > num_snapshots else pos
        centers, gr_part = calc_gr(subset, box_size, r_max, bins)
        total_hist_1 += gr_part 
        
    print(f"Processing {len(files_2)} files from {folder_2}...")
    num_snapshots = int(num_snapshots * 5)
    for f in files_2:
        data = np.load(f, allow_pickle=True)
        pos = data['pos_X']
        subset = pos[-num_snapshots:] if len(pos) > num_snapshots else pos
        subset = subset[::5]
        centers, gr_part = calc_gr(subset, box_size, r_max, bins)
        total_hist_2 += gr_part

    # Average the g(r) curves
    gr_1_avg = total_hist_1 / len(files_1)
    gr_2_avg = total_hist_2 / len(files_2)
    
    # ==========================
    # Plotting
    # ==========================
    print("Generating plot...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(centers / sigma, gr_1_avg, label=label_1, color='#1f77b4', linewidth=2.5)
    plt.plot(centers / sigma, gr_2_avg, label=label_2, color='#d62728', linewidth=2.5)
    
    plt.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, label=r'Reaction Radius ($\sigma$)')
    plt.axhline(y=1.0, color='black', linestyle=':', linewidth=1.5, label='Well-mixed Ideal ($g(r)=1$)')
    
    plt.title('Radial Distribution Function $g(r)$ of Particles X', fontsize=16, fontweight='bold')
    plt.xlabel(r'Distance $r$ (in units of $\sigma$)', fontsize=14)
    plt.ylabel(r'Pair Correlation $g(r)$', fontsize=14)
    
    plt.xlim(0, r_max / sigma)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    save_dir = project_root / "results" / "spatial_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / "gr_comparison_all_trajs_rmax.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Successfully saved smoothed plot to: {plot_path}")
    
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()