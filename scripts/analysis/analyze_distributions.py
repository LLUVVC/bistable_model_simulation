"""
Histogram

KDE

Wasserstein distance

The data loader return the data in the form of:
    trajectories(dictionary): store single dynamics of X or X and X2;
    combined_data(dictionary): put all trajectories with the same parameter setting together (timespan could vary, but the way I dealt with it
    seems to confine the timespan to be the same for all trajectory simulations);
    metadata(dictionary): paramter setting about the current data
"""
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


def find_the_best_bw(x_data):
    """
    use the from sklearn to find the most suitable bandwidth for current data
    
    kernel: Gaussian

    instead of Scipy KDE, we adopt the sklearn one bc it offers the function to find an optimal bandwidth value

    input: the combined data of X from different trjactories

    note: (at the moment) only need to run this function once, and keep it the same throughout all simulation data across scales;
    for the simulation with spatial resolution, the bandwidth shall be the same as it is in the well-mixed simulation cases when diffusion 
    coefficients are big enough.

    for spatial process simulations with smaller diffusion coeff. -> TO DO...
    """

    # --- 1. Sample the data ---
    # originally I tried to use all the data, the process of searching for the best bw is incredibly slow
    # randomly select 50000 indicies without replacement
    sample_size = 50000
    x_sample = x_data[sample_size]

    # --- 2. Define the search grid ---
    bandwidth_options = np.linspace(2.0, 30.0, 50) # identified the range through trial

    # --- 3. Configure the grid search --- 
    grid = GridSearchCV(
        KernelDensity(kernel='gaussian'),
        param_grid = {'bandwidth':bandwidth_options},
        cv = 3, # tried cv=5, but super slow
        n_jobs = -1 # use all cores available
        )
    

    # --- 4. Run the search on the sample ---
    print(f"Running GridSearch on {sample_size} points...")
    grid.fit(x_sample)

    print(f"Best bandwidth found: {grid.best_params_['bandwidth']}")

    return grid.best_params_['bandwidth']


# # Example of a clean metadata box
# textstr = '\n'.join((
#     r'$\kappa=%.2f$' % (microrate, ),
#     r'$\tau=%.4f$' % (tau, ),
#     r'$T_{final}=%d$' % (t_f, )))

# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
#         verticalalignment='top', bbox=props)


def get_pretty_upper_bound(data, pad_percent=0.05, snap_to=100):
    """
    Finds a clean upper bound for the x-axis.
    data: Your particle count array
    pad_percent: How much empty space to leave on the right (10%)
    snap_to: The 'nice' number to round up to (e.g., 50 or 100)
    """
    raw_max = np.max(data)
    # Add 10% breathing room
    padded_max = raw_max * (1 + pad_percent)
    # Round up to the nearest 'snap_to' interval
    pretty_max = np.ceil(padded_max / snap_to) * snap_to
    
    return int(pretty_max)


def hist(combined_data_X, bin_width):
    """
    Plot the histogram of the distribution of species X collected in all the trajectories

    input: combined_data_X = combined_data['X']
    """
    
    upper_bound = get_pretty_upper_bound(combined_data_X)
    bins = np.arange(0, upper_bound, bin_width)
    return None