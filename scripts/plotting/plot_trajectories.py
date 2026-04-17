import matplotlib.pyplot as plt
import numpy as np
from scripts.analysis.data_loader import load_well_mixed_data, load_spatial_full_data
from datetime import datetime

import os
from pathlib import Path

def get_data_dir(file_str: str) -> Path:
    """
    Safely resolves the data directory whether run from a terminal script
    or imported inside a Jupyter Notebook.
    """
    try:
        # 1. SCENARIO: Running as a standard Python script (.py)
        # Assuming script is in project_root_folder/scripts/plotting/
        project_root = Path(__file__).resolve().parent.parent.parent # plotting > scripts > root folder
        
    except NameError:
        # 2. SCENARIO: Running interactively in a Jupyter Notebook!
        # Notebooks don't have __file__. We use the Current Working Directory.
        current_dir = Path(os.getcwd()).resolve()
        
        project_root = current_dir.parent # notebooks are inside a 'notebooks' folder, go up one level
        

    # 3. Build the exact requested path to save simulated data
    data_dir = project_root /"results"/file_str
    
    # 4. Create the folders automatically if they don't exist yet!
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory set to: {data_dir}")

    return data_dir



def plot_well_mixed(file_str):

    trajectories, _, metadata = load_well_mixed_data(file_str=file_str)

    title_str = "Bistable System Dynamics"
    # --- select three examples to show the trajectories ---
    # --- plot all the trajectories for now ---

    # ============================================================
    # ================= TRAJECTORY PLOTS =========================
    # ============================================================
    tau = metadata['timestep']

    num_traj = len(trajectories)
    ncols = 2
    nrows = int(np.ceil(num_traj / ncols))

    fig_traj, axs = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))
    fig_traj.suptitle(title_str + " Trajectories", fontsize=16)

    # Flatten axes to 1D array for easy iteration
    axs = np.array(axs).reshape(-1)

    step = 1 # Only plot every 1000th data point if step = 1000
    # set step small in testing phases, bc large step would slice out all the data

    species_num = len(trajectories[0]['species_log']) # return the number of keys in the dictionary 
                                              # run_species_log. return 1 or 2.
    if species_num == 1: # only X: Schloegl model
        model = 'Schlögl'
        for i, traj in enumerate(trajectories):
            ax = axs[i]
            ax.plot(traj['timescale'][::step]*tau, traj['species_log']['X'][::step], 'b-', label='X', alpha=0.8, linewidth=1.2)
            ax.set_xlabel(r'$t$')
            ax.set_ylabel('Number of Particles')
            ax.legend(fontsize='small', loc='upper right') # Or 'upper left', etc.
            ax.grid(True)
            ax.set_xlim(left=traj['timescale'][0])

    elif species_num == 2: # X and X2: full model
        model = 'full'
        for i, traj in enumerate(trajectories):
            ax = axs[i]
            ax.plot(traj['timescale'][::step]*tau, traj['species_log']['X'][::step], 'b-', label='X', alpha=0.8, linewidth=1.2)
            ax.plot(traj['timescale'][::step]*tau, traj['species_log']['X2'][::step], 'r-', label='X2', alpha=0.8, linewidth=1.2)
            ax.set_xlabel(r'$t$')
            ax.set_ylabel('Number of Particles')
            ax.legend(fontsize='small', loc='upper right') # Or 'upper left', etc.
            ax.grid(True)
            ax.set_xlim(left=traj['timescale'][0])
    else:
        print("Error.")
        return 0
    
    macrorats = metadata['macrorates']
    t_f = metadata['timespan']

    # Create a string of formatted numbers separated by commas
    formatted_rats = ", ".join([f"{x:.2f}" for x in macrorats])

    # Plug that string into your label

    textstr = '\n'.join((
        r"$Rates_{%s}=[%s]$" % (model, formatted_rats),
        r'$\tau=%.4f$' % (tau, ),
        r'$T_{final}=%d$' % (t_f, )))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout() # rect=[0, 0, 0.85, 1]

    # --- Get timestamp for filename ---
    # Format as YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_traj = f"trjactories_{timestamp}.png" # timestamp as a benchmark

    # --- Save plot to the 'results' folder ---
    DATA_DIR = get_data_dir(file_str)

    # --- Create the directory if it doesn't exist ---
    os.makedirs(DATA_DIR, exist_ok=True)

    output_plot_path_traj = os.path.join(DATA_DIR, filename_traj)
    fig_traj.savefig(output_plot_path_traj)

    print(f"Saved trajectories plot to {DATA_DIR}")

    plt.show()

def plot_spatial(file_str): 

    return None



def main():
    filestr = "schloegl_model_0.15_0.025"
    model = "well-mixed" # "spatial" or "well-mixed" 

    if model == "spatial":
        filestr = "spatial" # need changing
        plot_spatial(filestr)
    elif model == "well-mixed":
        filestr = "well_mixed_data/" + filestr
        plot_well_mixed(filestr)
    else:
        print("Error.")



if __name__ == "__main__":
    main()



