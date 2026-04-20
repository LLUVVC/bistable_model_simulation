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



def plot_well_mixed_traj(file_str):

    trajectories, _, metadata = load_well_mixed_data(file_str=file_str)

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
    
    # Flatten axes to 1D array for easy iteration
    axs = np.array(axs).reshape(-1)

    step = 1 # Only plot every 1000th data point if step = 1000
    # set step small in testing phases, bc large step would slice out all the data
    
    # Modern color palette (Blue and Orange/Coral)
    # color_x = "#20d1db" 
    # color_x2 = "#efb20b" 
    color_x = "#0072B2" 
    color_x2 = "#D55E00" 
    species_num = len(trajectories[0]['species_log']) # return the number of keys in the dictionary 
                                              # run_species_log. return 1 or 2.
    if species_num == 1: # only X: Schloegl model
        model = 'Schlögl'
        for i, traj in enumerate(trajectories):
            ax = axs[i]
            x_time = traj['timescale'][::step] * tau
            y_x = traj['species_log']['X'][::step]
            
            ax.plot(x_time, y_x, color=color_x, label='X', drawstyle='steps-post', alpha=0.9, linewidth=1.5, zorder=1)
            # add a subtle shaded area under the curves
            ax.fill_between(x_time, y_x, color=color_x, step='post', alpha=0.1)
        
            ax.set_xlabel('Time ($t$)')
            ax.set_ylabel('Particle Count')
            ax.set_title(f'Trajectory {i+1}')
            ax.legend(fontsize='small', loc='upper right') # Or 'upper left', etc.
            ax.grid(True, linestyle='--', alpha=0.4, which='both')
            # clean up the frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(left=traj['timescale'][0])

    elif species_num == 2: # X and X2: full model
        model = 'Full'
        for i, traj in enumerate(trajectories):
            ax = axs[i]
            x_time = traj['timescale'][::step] * tau
            y_x = traj['species_log']['X'][::step]
            y_x2 = traj['species_log']['X2'][::step]
            ax.plot(x_time, y_x, color=color_x, label='X', drawstyle='steps-post', alpha=0.9, linewidth=1.5, zorder=1)
            ax.plot(x_time, y_x2, color=color_x2, label='X2', drawstyle='steps-post', alpha=0.9, linewidth=1.5, zorder=2)
            # add a subtle shaded area under the curves
            ax.fill_between(x_time, y_x, color=color_x, step='post', alpha=0.1)
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
    else:
        print("Error.")
        return 0
    
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
    fig_traj.text(0.5, 0.85, textstr, transform=fig_traj.transFigure, fontsize=9,
            ha='center', va='top', multialignment='left', bbox=props, linespacing=1.5)
    
    fig_traj.suptitle("Bistable System Dynamics\n" + rf"$\mathit{{(Well-Mixed \ {model}\ Trajectories)}}$", 
             fontsize=16, y=0.98, fontweight='bold') # use backslash before a space for a physical gap between words
    # --- THE KEY FIX: Push the plots down ---
    # top=0.7 means the subplots only occupy the bottom 70% of the figure
    plt.subplots_adjust(top=0.7, bottom=0.15, hspace=0.3, wspace=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.82])

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


##########
########## ONLY tested with the well-mixed data so far ######
########## Next step: spatial simulation 


def plot_spatial_traj(file_str): 
        
        trajectories, _, metadata = load_spatial_full_data(file_str=file_str)
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
        fig_traj.suptitle("Bistable System Dynamics\n" + r"$\mathit{(Spatially\ Resolved\ Full\ Trajectories)}$", 
             fontsize=16, y=0.98, fontweight='bold') 
        # fig_traj.suptitle(title_str, fontsize=16, y=0.96) # title_str + " Trajectories"

        # Flatten axes to 1D array for easy iteration
        axs = np.array(axs).reshape(-1)

        step = 1 # Only plot every 1000th data point if step = 1000
        # set step small in testing phases, bc large step would slice out all the data

        # Modern color palette (Blue and Orange/Coral)
        color_x = "#0072B2" 
        color_x2 = "#D55E00" 

        for i, traj in enumerate(trajectories):
            ax = axs[i]
            x_time = traj['timescale'][::step] * tau
            y_x = traj['species_log']['X'][::step]
            y_x2 = traj['species_log']['X2'][::step]
            ax.plot(x_time, y_x, color=color_x, label='X', drawstyle='steps-post', alpha=0.9, linewidth=1.5, zorder=1)
            ax.plot(x_time, y_x2, color=color_x2, label='X2', drawstyle='steps-post', alpha=0.9, linewidth=1.5, zorder=2)
            # add a subtle shaded area under the curves
            ax.fill_between(x_time, y_x, color=color_x, step='post', alpha=0.1)
            ax.fill_between(x_time, y_x2, color=color_x2, step='post', alpha=0.1)
            ax.legend(fontsize='small', loc='upper right') # Or 'upper left', etc.
            ax.grid(True, linestyle='--', alpha=0.4, which='both')
            ax.set_xlabel('Time ($t$)')
            ax.set_ylabel('Particle Count')
            ax.set_title(f'Trajectory {i+1}')
            # clean up the frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(left=traj['timescale'][0])

        macrorates = metadata['macrorates']
        microrates = metadata['microrates']
        t_f = metadata['timespan']

        # Create a string of formatted numbers separated by commas
        macro_formatted_rates = ", ".join([f"{x:.2f}" for x in macrorates])
        micro_formatted_rates = ", ".join([f"{x:.2e}" for x in microrates])
        # Plug that string into your label

        textstr = '\n'.join((
            rf"$\mathbf{{Rates_{{macro}}}}$: {macro_formatted_rates}",
            rf"$\mathbf{{Rates_{{micro}}}}$: {micro_formatted_rates}",
            rf"$\tau: {tau:.2e} \quad | \quad T_{{final}}: {t_f}$" 
        ))

        props = dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.8)
        fig_traj.text(0.5, 0.81, textstr, transform=fig_traj.transFigure, fontsize=9,
                ha='center', va='top', multialignment='left', bbox=props, linespacing=1.5)
        
        # --- THE KEY FIX: Push the plots down ---
        # top=0.7 means the subplots only occupy the bottom 70% of the figure
        plt.subplots_adjust(top=0.7, bottom=0.15, hspace=0.3, wspace=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.82])
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




def main():
    filestr = "full_model_1.5_1500.0" # "schloegl_model_0.15_0.025"
    model_resolution = "well-mixed" # "spatial" or "well-mixed" 

    if model_resolution == "spatial":
        filestr = "spatial_data/" + filestr
        plot_spatial_traj(filestr)
    elif model_resolution == "well-mixed":
        filestr = "well_mixed_data/" + filestr
        plot_well_mixed_traj(filestr)
    else:
        print("Error.")



if __name__ == "__main__":
    main()



