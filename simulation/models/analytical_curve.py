import scipy.sparse
from scipy.linalg import null_space
from scipy.sparse.linalg import eigs
import numpy as np


# generate Q, Qp = dp/dt
# specifically for the schoegl's model
# with the specific setting of parameters used in simulation


"""
Modify the code so it is general for different parameter settings.
"""

def generate_shoegl_Q(x_state:np.array, kappa:np.array):

    a, b, vol = 10.0, 20.0, 8.0
    
    rank = x_state.size
    Q = np.zeros((rank, rank))

    lamb_n = lambda y: a*kappa[0]*y*(y-1)/vol + b*kappa[2]*vol
    mu_n = lambda y: y*kappa[3] + kappa[1]*y*(y-1)*(y-2)/(vol**2)
  
    Q_mu, Q_lamb = mu_n(x_state), lamb_n(x_state)    
    # Q_mu = np.roll(mu_n(x_state),-1)
    
    Q[np.eye(rank, k=1, dtype='bool')] = Q_mu[1:]
    Q[np.eye(rank, k=-1, dtype='bool')] = Q_lamb[:-1]
    Q_lamb[-1] = 0. # otherwise the Q does not satisfy column-wise or row-wise sum up to 1.
    np.fill_diagonal(Q, -(Q_lamb+Q_mu))

    return Q


def get_analytical_curve():

    n_x_max = 400
    p_states = np.arange(0, n_x_max + 1, 1)
    
    k = np.array((6.,1.,230.,1000.))
    Q = generate_shoegl_Q(p_states, k)

    eigenvalues, eigenvectors = np.linalg.eig(Q)
    id_max = np.argmax(np.isclose(eigenvalues, 0))

    stat_dist = np.real(eigenvectors[:, id_max])
    stat_dist /= stat_dist.sum()  # Normalize to ensure it's a probability distribution
    
    return p_states, stat_dist


##### Not used function
'''
def get_stationary_distribution(Q, method='dense'):
    """
    Find stationary distribution of Q-matrix.
    
    Parameters:
    - Q: transition rate matrix
    - method: 'dense' or 'sparse'
    """
    if method == 'dense' or not scipy.sparse.issparse(Q):
        # Use null space method for dense matrices
        stat_dist = null_space(Q.T)
        
        if stat_dist.shape[1] == 0:
            # Fallback to eigenvalue method
            eigenvalues, eigenvectors = np.linalg.eig(Q.T)
            id_max = np.argmax(np.isclose(eigenvalues, 0))
            stat_dist = eigenvectors[:, id_max]
        else:
            stat_dist = stat_dist[:, 0]
    else:
        # Use sparse eigenvalue solver
        eigenvalues, eigenvectors = eigs(Q.T, k=1, sigma=0, which='SM')
        stat_dist = eigenvectors[:, 0]
    
    # Normalize
    stat_dist = np.real(stat_dist)
    if np.sum(stat_dist) < 0:
        stat_dist = -stat_dist
    stat_dist = np.maximum(stat_dist, 0)
    stat_dist /= stat_dist.sum()
    
    return stat_dist
'''




