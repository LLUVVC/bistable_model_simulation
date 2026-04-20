import matplotlib.pyplot as plt
import numpy as np
from scripts.analysis.data_loader import load_well_mixed_data, load_spatial_full_data
from scripts.analysis.analyze_distributions import find_the_best_bw, hist_np, kde_sk
from datetime import datetime

"""
1. Plot the distribution of species X throughout the simulation process, using data aggregated from all

trajectories generated under the same parameter settings.

2. Compare the histograms and KDEs of the empirical distribution against the analytical distribution curve of the

referenced Schlögl model.

3. Quantitatively measure the model fit by calculating the Wasserstein distance between the KDE and the analytical

curve.

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


def plot_well_mixed_dist(file_str, bin_width=2., band_width=2.5714):
    """
    Project Default:
    - We use bw=2.5714 (calculated via GridSearchCV on our reference dataset).
    """
    traj, combined_data, metadata = load_well_mixed_data(file_str=file_str)
    

    # ============================================================
    # ================= DISTRIBUTION PLOTS =======================
    # ============================================================

    fig_dist, ax = plt.subplots(figsize=(12,6))

    species_num = len(combined_data)
    print(species_num)

    if species_num == 1:
        model = 'Schlögl'
        combined_data_X = combined_data['X']
        hist_bin, density_hist = hist_np(combined_data_X, bin_width=bin_width)
        hist_bin_center = hist_bin[:-1] + 0.5
        x_axis_plot, kde_X = kde_sk(combined_data_X, bw=band_width)
        ax.bar(
            hist_bin_center,
            density_hist, 
            color='royalblue',
            width=bin_width * 0.9,
            # label=f'tau-leaping at t={t_f}',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        # ax.set_title(f'Combined X Histogram ({len(traj)} trajectories)')
        # ax.set_xlabel('Number of X particles')
        # ax.set_ylabel('Probability')
        # ax.set_xlim(0, 400)
        # ax.legend()
        # ax.grid(True)

    elif species_num == 2:
        model = 'Full'
        combined_data_X = combined_data['X']
        hist_bin, density_hist = hist_np(combined_data_X, bin_width=bin_width)
        x_axis_plot, kde_X = kde_sk(combined_data_X, bw=band_width)
    else:
        print("Error.")
        return 0
    
    tau = metadata['timestep']
    macrorates = metadata['macrorates']
    t_f = metadata['timespan']

    # Create a string of formatted numbers separated by commas
    formatted_rates = ", ".join([f"{x:.2f}" for x in macrorates])

    # Plug that string into your label

    textstr = '\n'.join((
        rf"$\mathbf{{Rates_{{{model}}}}}$: {formatted_rates}",
        rf"$\tau: {tau:.2e} \quad | \quad T_{{final}}: {t_f}$" 
    ))

    props = dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.8)
    fig_dist.text(0.5, 0.85, textstr, transform=fig_dist.transFigure, fontsize=9,
            ha='center', va='top', multialignment='left', bbox=props, linespacing=1.5)
    fig_dist.suptitle("Bistable System Distribution\n" + rf"$mathit{{(Well-Mixed\ {model}\ Distribution)}}$",
                      fontsize=16, y=0.98, fontweight='bold')
    plt.subplots_adjust(top=0.7, bottom=0.15, hspace=0.3, wspace=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.82])

    # --- Get timestamp for filename ---
    # Format as YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_dist = f"distribution_{timestamp}.png" # timestamp as a benchmark

    # --- Save plot to the 'results' folder ---
    DATA_DIR = get_data_dir(file_str)

    # --- Create the directory if it doesn't exist ---
    os.makedirs(DATA_DIR, exist_ok=True)

    output_plot_path_dist = os.path.join(DATA_DIR, filename_dist)
    fig_dist.savefig(output_plot_path_dist)

    print(f"Saved distribution plot plot to {DATA_DIR}")

    plt.show()


def plot_spatial_dist(file_str):
    return None


def main():
    filestr = "schloegl_model_0.15_0.025" # 
    model_resolution = "well-mixed" # "spatial" or "well-mixed" 

    if model_resolution == "spatial":
        filestr = "spatial_data/" + filestr
        plot_spatial_dist(filestr)
    elif model_resolution == "well-mixed":
        filestr = "well_mixed_data/" + filestr
        plot_well_mixed_dist(filestr)
    else:
        print("Error.")



if __name__ == "__main__":
    main()