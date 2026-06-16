import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from pathlib import Path

def get_project_root():
    """Finds the root directory of the project assuming the script is run from inside the project."""
    try:
        # If running as a script (e.g. from scripts/runners/calc_cross_gr.py)
        return Path(__file__).resolve().parent.parent.parent
    except NameError:
        # If running interactively
        return Path(os.getcwd()).resolve().parent

def calc_cross_gr(pos_x_list, pos_x2_list, box_size, r_max, bins):
    """
    Calculates the cross-radial distribution function g(r) between X and X2.
    """
    edges = np.linspace(0, r_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    
    hist_sum = np.zeros(bins)
    pair_expected_sum = np.zeros(bins)
    vol = box_size**3
    
    num_frames = min(len(pos_x_list), len(pos_x2_list))
    
    for k in range(num_frames):
        pos_x = pos_x_list[k]
        pos_x2 = pos_x2_list[k]
        
        N1 = len(pos_x)
        N2 = len(pos_x2)
        if N1 == 0 or N2 == 0: 
            continue
        
        # Calculate pairwise distances between X and X2
        diff = pos_x[:, np.newaxis, :] - pos_x2[np.newaxis, :, :]
        # Apply Periodic Boundary Conditions (PBC)
        diff = diff - box_size * np.round(diff / box_size)
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Flatten the matrix because all X to X2 pairs are unique valid pairs
        d = dist.flatten()
        
        # Histogram the distances
        h, _ = np.histogram(d, bins=edges)
        hist_sum += h
        
        # Calculate expected number of cross-pairs in an ideal gas for this shell
        shell_vol = (4.0/3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
        expected = N1 * N2 * (shell_vol / vol)
        pair_expected_sum += expected
        
    # Prevent division by zero
    valid = pair_expected_sum > 0
    gr = np.zeros_like(centers)
    gr[valid] = hist_sum[valid] / pair_expected_sum[valid]
    
    return centers, gr

def main():
    print("Starting cross-g(r) calculation... This might take a few minutes.")
    project_root = get_project_root()
    base_dir = project_root / "simulation_data" / "spatial_data"
    
    # =========================================================
    # FLEXIBLE INPUT: Define your folder names and labels here
    # =========================================================
    folder_1 = 'homo_kp_tf_0.1_D_1500.0_tau_1e-06'  # UPDATE these to your actual short-run folders
    folder_2 = 'homo_kp_tf_0.1_D_1500.0_tau_2e-07'
    
    label_1 = r'$\tau = 10^{-6}$'
    label_2 = r'$\tau = 2\times 10^{-7}$'
    
    # How many of the final snapshots to average over
    num_snapshots = 10000 # Adjust if necessary based on your short runs
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
    
    r_max = 0.5  # compute up to 5 sigma (since sigma=0.1)
    bins = 50
    box_size = 2.0
    sigma = 0.1
    
    edges = np.linspace(0, r_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    total_hist_1 = np.zeros(bins)
    total_hist_2 = np.zeros(bins)
    
    print(f"Processing {len(files_1)} files from {folder_1}...")
    for f in files_1:
        data = np.load(f, allow_pickle=True)
        if 'pos_X2' not in data:
            print(f"Skipping {f} because it does not contain 'pos_X2'")
            continue
            
        pos_x = data['pos_X']
        pos_x2 = data['pos_X2']
        print(pos_x.shape)
        print(pos_x2.shape)
        subset_x = pos_x[-num_snapshots:] if len(pos_x) > num_snapshots else pos_x
        subset_x2 = pos_x2[-num_snapshots:] if len(pos_x2) > num_snapshots else pos_x2
        
        centers, gr_part = calc_cross_gr(subset_x, subset_x2, box_size, r_max, bins)
        total_hist_1 += gr_part 
        
    print(f"Processing {len(files_2)} files from {folder_2}...")
    # NOTE: You may or may not need the `int(num_snapshots * 5)` depending on 
    # if you changed the logging interval for the tau=2e-7 run!
    slice_val = 1
    num_snapshots_2 = int(num_snapshots * slice_val) 
    
    for f in files_2:
        data = np.load(f, allow_pickle=True)
        if 'pos_X2' not in data:
            continue
            
        pos_x = data['pos_X']
        pos_x2 = data['pos_X2']
        
        subset_x = pos_x[-num_snapshots_2:] if len(pos_x) > num_snapshots_2 else pos_x
        subset_x2 = pos_x2[-num_snapshots_2:] if len(pos_x2) > num_snapshots_2 else pos_x2
        
        # If your tau=2e-7 logged 5x more often, slice every 5th to match
        subset_x = subset_x[::slice_val]
        subset_x2 = subset_x2[::slice_val]
        
        centers, gr_part = calc_cross_gr(subset_x, subset_x2, box_size, r_max, bins)
        total_hist_2 += gr_part

    # Average the cross-g(r) curves
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
    
    plt.title(r'Cross-Radial Distribution $g_{X, X_2}(r)$ (Catalyst-Product Clustering)', fontsize=16, fontweight='bold')
    plt.xlabel(r'Distance $r$ (in units of $\sigma$)', fontsize=14)
    plt.ylabel(r'Cross Pair Correlation $g(r)$', fontsize=14)
    
    plt.xlim(0, r_max / sigma)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    save_dir = project_root / "results" / "spatial_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / "cross_gr_X_X2_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Successfully saved plot to: {plot_path}")
    
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()