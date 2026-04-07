import numpy as np
from scipy.spatial import cKDTree as KDTree # instead of KDTree which is conducted in python
from simulation.utils.geometry import generate_position_box

'''
Algorithm used are based on Figure 10 in the paper del_Razo_2025_J._Phys._A__Math._Theor._58_145001_diffusion

Functions here are not adopted in the actual simulation due to the low speed, so I haven't check them carefully.
'''


def BimolecularHomo(pos_r:np.array, sigma:float, kappa:float, h:float):
    '''
    pos_r: positions of reactants
    pos_p: positions of products
    sigma: active reaction radius
    kappa: microscopic reaction rate
    h: time step
    '''

    num_r = len(pos_r)
    if num_r < 2:
        return np.array([])
    reacted_set = set()
    reacted_pair = []


    tree = KDTree(pos_r)
    pair_set = tree.query_pairs(r=sigma)
    pair_list = list(pair_set)

    np.random.shuffle(pair_list)
    num_pairs = len(pair_list)

    # if r<= react_prob then reaction happens
    # which is react_propensity >= 0
    # r is a random number placeholder according to the algorithm used

    react_prob = 1 - np.exp(-kappa*h)
    react_propensity = react_prob - np.random.uniform(low=0., high=1., size=num_pairs)

    for cnt, (i,j) in enumerate(pair_list):
        if react_propensity[cnt]<0:
            continue
        if i in reacted_set or j in reacted_set:
            continue
        
        reacted_set.add(i)
        reacted_set.add(j)
        reacted_pair.append([i,j])

    return np.array(reacted_pair)


def BimolecularHetero(pos_ra:np.array, pos_rb:np.array, sigma:float, kappa:float, h:float):
    '''
    pos_r1: positions of reactants species A
    pos_r2: positions of reactants specoes B
    sigma: active reaction radius
    kappa: microscopic reaction rate
    h: time step
    '''

    num_ra, num_rb = len(pos_ra), len(pos_rb)
    if num_ra < 1 or num_rb < 1:
        return np.array([]), np.array([])
    a_reacted_set, b_reacted_set = set(), set()
    
    tree_a, tree_b = KDTree(pos_ra), KDTree(pos_rb)
    '''
    each point on tree_a is a base,
    and we check how many points on tree_b is within the
    radius of sigma to a specific point on tree_a
    
    EXAMPLE on tree_a: 
        tree_a indexes: [[4], [2, 3], [], [], []]
        for tree_a, point b4 is close to a0, and b2&b3 close to a1.
    EXAMPLE on tree_b: 
        tree_b indexes: [[], [], [1], [1], [0]]
    '''
    indexes = tree_a.query_ball_tree(tree_b, r=sigma)

    react_prob = 1 - np.exp(-kappa*h)
    ''' To do: add shuffle '''
    for a, neighbor_list in enumerate(indexes):
        
        # if len(neighbor_list)<1:
        #     continue
        if a in a_reacted_set or not neighbor_list:
            continue

        react_propensity = react_prob - np.random.uniform(low=0., high=1., size=len(neighbor_list))
        for cnt, b in enumerate(neighbor_list):
            if react_propensity[cnt]<0:
                continue
            if b in b_reacted_set:
                continue
            
            a_reacted_set.add(a)
            b_reacted_set.add(b)
            break  # STOP CHECKING !!! reason: One reaction for per 'a' particle

    return np.array(list(a_reacted_set)), np.array(list(b_reacted_set))


def UnimolecularSelectReactant(pos_r, kappa, h):
    '''
    select the reactant for unimolecular reaction
    pos_r: positions of all reactants
    kappa: microscopic reaction rate for the reaction
    h: timestep
    '''
    num_r = len(pos_r)

    react_prob = 1 - np.exp(-kappa*h)
    react_propensity = react_prob - np.random.uniform(low=0., high=1., size=num_r)
    # if react_prob < r, then there is no reaction -> set to False
    # otherwise there is a reaction -> set to True
    if_react = np.where(react_propensity<0, False, True)
    reacted_list = np.nonzero(if_react)[0]
    return np.array(reacted_list) 


def AddMoleculeMidpoint(pos_r, pos_p, pairs):
    
    reacted_list = np.reshape(pairs, -1) # one-dimentional
    
    pos1 = pos_r[pairs[:,0]]    
    pos2 = pos_r[pairs[:,1]]
    pos_new_products = (pos1+pos2)*0.5
    
    if pos_p.size==0:
        # set the pos_p = np.empty((0,3)) as in put
        # if there is no existing products in the system
        pos_p = pos_new_products
    else:
        pos_p = np.vstack((pos_p, pos_new_products))
        # pos_p = np.concatenate((pos_p, pos_new_products), axis=0)

    pos_r = np.delete(pos_r, reacted_list, axis=0)

    return pos_r, pos_p

def SubstituteParticle(pos_ori, pos_sub, replace_list):
    '''
    substitute the particles at positions of pos_ori on replace_list with 
    pos_sub, return the updated positions of pos_ori and pos_sub
    '''
    pos_new_products = pos_ori[replace_list]

    if pos_sub.size==0:
        pos_sub = pos_new_products
    else:
        pos_sub = np.vstack((pos_sub, pos_new_products))
    pos_ori = np.delete(pos_ori, replace_list, axis=0)

    return pos_ori, pos_sub


def AddParticleBath(pos, box_shape, num_2add):
    '''
    pos: positions of the particle
    box_shape: shape of the simulation box 
    num_2add: number of the particle to be added
    '''
    if num_2add <= 0:
        return pos
    
    pos_new = generate_position_box(num_2add, box_shape)
    if pos.size == 0:
        return pos_new
    pos = np.vstack((pos, pos_new))
    return pos


def RemoveParticleBath(pos, num_2remove):
    '''
    pos: positions of the particle
    num_2remove: number of the particle to be removed
    '''

    current_num = len(pos)
    
    # Safety Check: Can't remove more than we have !!!
    if num_2remove >= current_num:
        return np.empty((0, 3))
    
    if num_2remove <= 0:
        return pos
    
    # generate a uniform random sample from pos of size num_2remove without replacement:
    pos_2remove_index = np.random.choice(current_num, num_2remove, replace=False)
    pos = np.delete(pos, pos_2remove_index, axis=0)

    return pos
