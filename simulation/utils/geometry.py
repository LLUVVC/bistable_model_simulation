import numpy as np
import inspect
from tqdm import tqdm

'''
All the functions are for 3-dimensional simulations
'''


##### generate positions of particles #####
##### cartesian coordinate #####

def generate_position_box(num:int, box_shape:np.array):
    '''
    define the box_shape as [0,L1],[0,L2],[0,L3]
    '''
    positions = np.empty((num,3)) # dimension 3
    
    positions[:,0] = np.random.uniform(low=0.0, high=box_shape[0], size=num)
    positions[:,1] = np.random.uniform(low=0.0, high=box_shape[1], size=num)
    positions[:,2] = np.random.uniform(low=0.0, high=box_shape[2], size=num)

    return positions



def generate_position_sphere(num:int, R:float):
    '''
    R: the radius of the ball
    '''
    random_directions = np.random.normal(0,1,(num,3)) # 3-dimensional
    magnitude = np.linalg.norm(random_directions, axis=1)[:, np.newaxis] # keepdim = True
    unit_vectors = random_directions/magnitude # normalize the coordinates
    # generate random radii
    # take the cube root of a uniform variable to account for spherical volume scaling
    u = np.random.uniform(low=0.0, high=1.0, size=(num,1))
    radii = R * u**(1/3)
    positions = radii * unit_vectors

    return positions



##### diffusion process #####

def Diffusion(x:np.array,tau:float,diff:float):
    '''
    one time step diffusion
    x: input position coords
    D: diffusion coefficient
    tau: simulation step
    '''
    x_step = np.random.normal(size=x.shape) * np.sqrt(2*diff*tau)
    return x + x_step

# check if particles are outside the box
def CheckInBox(box_coords, pos):

    dim = box_coords.shape[0]

    if dim==3:       
        x_inside = (pos[:, 0] >= box_coords[0][0]) & (pos[:, 0] <= box_coords[0][1])
        y_inside = (pos[:, 1] >= box_coords[1][0]) & (pos[:, 1] <= box_coords[1][1])
        z_inside = (pos[:, 2] >= box_coords[2][0]) & (pos[:, 2] <= box_coords[2][1])

        pos_outside = np.argwhere((x_inside&y_inside&z_inside)==False).flatten()
        return pos_outside
    else:
        print('Dimension Wrong')
        return 0
    



# check if particles is within a specific range of the selected "center" particle
def CheckInCenter(pos_center,pos,sigma):
    if pos_center.shape[0] and pos.shape[0]:
        pos_center_exp = pos_center[:,np.newaxis,:]
        pos_exp = pos[np.newaxis,:,:]
        pair_dist = np.sqrt(np.sum((pos_center_exp - pos_exp)**2,axis=-1))

        pair_dist = pair_dist-sigma
        InRange_ind = np.argwhere((pair_dist<0)&(pair_dist>-sigma))
        return InRange_ind
    else:
        return np.zeros((0,2)) # or np.array([]) 1-d array



# injection uniformly
def InjectUniform(number:int,box_shape:np.array,dim:int):
    '''
    Add particles to the reaction region at a random position uniformly.
    number: number of particles to be added
    '''
    num = np.array((number,))
    pos, = generate_position_box(num,box_shape,dim=dim)
    return pos