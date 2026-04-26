
import numpy as np


# generate Q, Qp = dp/dt
# specifically for the schoegl's model
# with the specific setting of parameters used in simulation


"""
Modify the code so it is general for different parameter settings (in progress: 22.04)
"""

def generate_schloegl_Q(x_state:np.array, k:np.array, a, b, vol):
    
    rank = x_state.size
    Q = np.zeros((rank, rank))

    lamb_n = lambda y: a*k[0]*y*(y-1)/vol + b*k[2]*vol
    mu_n = lambda y: y*k[3] + k[1]*y*(y-1)*(y-2)/(vol**2)
  
    Q_mu, Q_lamb = mu_n(x_state), lamb_n(x_state)    
    # Q_mu = np.roll(mu_n(x_state),-1)
    
    Q[np.eye(rank, k=1, dtype='bool')] = Q_mu[1:]
    Q[np.eye(rank, k=-1, dtype='bool')] = Q_lamb[:-1]
    Q_lamb[-1] = 0. # otherwise the Q does not satisfy column-wise or row-wise sum up to 1.
    np.fill_diagonal(Q, -(Q_lamb+Q_mu))

    return Q



"""

n_x_max as an input which we get from the get_pretty_upper_bound function in analyze_distributions???

not sure, really need to try this first and see what the upper bound calculation function returns.

"""

def get_analytical_curve(n_x_max, k, a, b, vol):

    p_states = np.arange(0, n_x_max, 1)
    Q = generate_schloegl_Q(p_states, k, a, b, vol)

    eigenvalues, eigenvectors = np.linalg.eig(Q)
    id_max = np.argmax(np.isclose(eigenvalues, 0))

    stat_dist = np.real(eigenvectors[:, id_max])

    stat_dist /= stat_dist.sum()  # Normalize to ensure it's a probability distribution
    stat_dist = stat_dist[stat_dist>=0] 
    index = int(len(p_states) - len(stat_dist))
    p_states = p_states[index:]
    return p_states, stat_dist






