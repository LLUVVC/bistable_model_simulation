import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import seaborn as sns

def calculate_fano_for_trajectory(file_path, R=0.2, num_test_spheres=5000, num_frames=50):
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
        
    if 'pos_X' not in data:
        return None
        
    pos_X_all = data['pos_X']
    box_shape = data['box_shape']
    L = box_shape[0]
    
    # Ensure we don't take more frames than exist
    frames_to_analyze = pos_X_all[-min(num_frames, len(pos_X_all)):]
    
    fano_factors = []
    
    for pos_x in frames_to_analyze:
        if len(pos_x) == 0:
            continue
            
        # Drop random test spheres
        test_points = np.random.uniform(0, L, size=(num_test_spheres, 3))
        tree = cKDTree(pos_x, boxsize=box_shape)
        neighbors = tree.query_ball_point(test_points, r=R)
        counts = np.array([len(n) for n in neighbors])
        
        mean_count = np.mean(counts)
        var_count = np.var(counts)
        
        if mean_count > 0:
            fano_factors.append(var_count / mean_count)
            
    if len(fano_factors) == 0:
        return None
        
    return np.mean(fano_factors)

def analyze_folder(folder_path, label, R=0.2):
    file_pattern = os.path.join(folder_path, "*.npz")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No .npz files found in {folder_path}")
        return [], []
        
    print(f"\nFound {len(files)} files in {folder_path} for {label}")
    
    fano_list = []
    
    for i, f in enumerate(files):
        if i % 10 == 0:
            print(f"  Processing file {i+1}/{len(files)}...")
            
        fano = calculate_fano_for_trajectory(f, R=R)
        if fano is not None:
            fano_list.append(fano)
            
    avg = np.mean(fano_list) if fano_list else 0
    print(f"  -> Average Fano Factor for {label}: {avg:.4f}")
    
    return fano_list, [label] * len(fano_list)

if __name__ == "__main__":
    # =======================================================
    # TODO: Update these paths to the exact folder names 
    # where your laptop stores the downloaded .npz data
    # =======================================================
    
    base_dir = "./simulation_data/spatial_data"
    folder_1e6 = os.path.join(base_dir, "homo_tf_24.0_D_1500.0_1")
    folder_2e7 = os.path.join(base_dir, "homo_2e-07_136")
    
    all_fanos = []
    all_labels = []
    
    # Process tau = 1e-6
    if os.path.exists(folder_1e6):
        fano_1e6, labels_1e6 = analyze_folder(folder_1e6, r"$\tau = 10^{-6}$")
        all_fanos.extend(fano_1e6)
        all_labels.extend(labels_1e6)
    else:
        print(f"Directory not found: {folder_1e6}")

    # Process tau = 2e-7
    if os.path.exists(folder_2e7):
        fano_2e7, labels_2e7 = analyze_folder(folder_2e7, r"$\tau = 2\times 10^{-7}$")
        all_fanos.extend(fano_2e7)
        all_labels.extend(labels_2e7)
    else:
        print(f"Directory not found: {folder_2e7}")
        
    if not all_fanos:
        print("No data processed. Exiting.")
        exit()
        
    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    
    # Seaborn boxplot + swarmplot combo
    sns.boxplot(x=all_labels, y=all_fanos, palette="Set2", showfliers=False)
    sns.swarmplot(x=all_labels, y=all_fanos, color=".25", alpha=0.7)
    
    # Add a horizontal line at 1.0 (perfectly well-mixed Poisson baseline)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Perfectly Well-Mixed (Poisson)')
    
    plt.ylabel("Fano Factor (Variance / Mean)")
    plt.title("Spatial Clustering of X: Fano Factor Comparison")
    plt.legend()
    
    plt.tight_layout()
    output_filename = "fano_comparison.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot saved successfully as '{output_filename}'")
    plt.show()