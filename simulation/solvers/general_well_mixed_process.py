import numpy as np
import inspect
import tqdm as tqdm


"""
Generalized stochastic simulation for an arbitrary chemical reaction system.
    
Implements Gillespie's SSA and the Explicit Tau-Leaping Algorithm.

"""


class Reaction:
    def __init__(self, kappa:np.narray, R_reactant:np.narray, R_product:np.narray, x:np.narray, 
                 bath:np.narray, volume:float) -> None:
        '''
        R_reactant & R_product: 
            set of stoichiomertric number
            example:
            
                mA+nB -> lC
                R_reactant = np.array((m,n,0)), R_product = np.array((0,0,l))

        kappa: reaction constants in terms of concentration
        bath: an array
        V: volume of the system
        x: initial concentration of different species
        '''
        self.kappa = kappa
        self.R_reactant = R_reactant
        self.R_product = R_product
        
        self.bath = bath
        self.volume = volume
        self.x_ini = x*self.volume # get the number of particles, NOT the concentration

        self.reaction_coef = R_product - R_reactant
        self.num_reactions = int(len(kappa))
        self.species = int(len(R_reactant[0]))
        
        temp_ = R_product - R_reactant
        temp_ind = np.nonzero(bath)[0]
        temp_[:,temp_ind] = 0
        self.bath_reaction_coef = temp_
        self.bath_boolean = True if bath.any()>0 else False
        self.threshold = 5e9



    def _generate_intensity_func(obj):
        inten_func_list = []
        for k_out in range(0,obj.num_reactions):
            # print('outside k=',k)
            def intensity_func(x:np.array,k=k_out):
                '''
                return the intensity function for 1 reaction
                at a time
                '''

            
                # rescale the reaction constant to adapt to stochastic model
                # according to kappa_hat = kappa/V**(m-1)

                num_reactants = int((obj.R_reactant[k]>0).sum())
                num_reactants_total = int(obj.R_reactant[k].sum()) # number of reactants 
                                                                   # rather than the num of species
                # rescale the system
                lamb_k = obj.kappa[k]/(obj.volume**(num_reactants_total-1))
                reactant_index = np.nonzero(obj.R_reactant[k])[0]   # np.where also work
        
                for i in range(0,num_reactants):
                    indi = reactant_index[i]
                    for j in range(0,obj.R_reactant[k][indi]):# if an element = 0 
                                                           # it don't go into the statement 
                        lamb_k = lamb_k*(x[indi]-j)

                return lamb_k
            # print(inspect.getsource(oneA_oneB_to_oneC))
            inten_func_list.append(intensity_func)
        return inten_func_list
    
    def run_tau_leaping(self, t_ini, t_f_steps, h):
        '''
        kappa: chemical reaction constant
        x: input is the initial number of particles
        t: input the initial time
        reaction_coef: how the reaction changes the number of particles
        h: the simulation step, need to be small
           it is the tau
        '''
        x_log = []
        time_log = []
        lamb_func_list = self._generate_intensity_func()
        x = self.x_ini
        t = t_ini
        x_log.append(x)
        time_log.append(t)
        
        lamb = np.zeros(self.num_reactions)
        cnt_step = 0

        while (cnt_step<t_f_steps): # can add another condition of equilibrium
            cnt_step = cnt_step + 1
            lamb_num_jumps = np.zeros(self.num_reactions) # (obj.num_reactions,1)
            t = t + h
            for j in enumerate(lamb_func_list):
                lamb[j[0]] = j[1](x)
                lamb_num_jumps[j[0]] = np.random.poisson(lamb[j[0]]*h)
                if (lamb[j[0]]*h > self.threshold):
                    print("change to smaller tau.")
                    break
            
            #if (lamb_num_jumps.any()>0):
            time_log.append(t)
            if self.bath_boolean:
                x = x + np.matmul(np.transpose(lamb_num_jumps), self.bath_reaction_coef)
            else:
                x = x + np.matmul(np.transpose(lamb_num_jumps), self.reaction_coef)
            x_log.append(x)
 
        x_log = np.array(x_log)
        time_log = np.array(time_log)
        
        return x_log, time_log

    def run_gillespie(self, t_ini, t_f): # t_f_steps
        '''
        kappa: chemical reaction constant
        x: input is the initial number of particles
        t: input the initial time
        t_f_steps: simulation steps
        t_f: the actual final simulation time
        reaction_coef: how the reaction changes the number of particles
        '''

        # num_reactants = int((reaction_coef<0).sum())
        # num_reactions_R = int(len(kappa))
        # cnt = 0
        x_log = []
        time_log = []
        lamb_func_list = self._generate_intensity_func()
        x = self.x_ini # number of particles
        t = t_ini
        x_log.append(x)
        time_log.append(t)
        x_dist = [x[2],] # x sample at fixed time interval
        
        dt = 0.0001 # the sample time interval
        last_log_t = 0. # initial sample time
        

        
        lamb_log = []
        # cnt_step = 0
        while (t<t_f): # (x[:num_reactants]>0).all() and 
                              # cnt_step<t_f_steps
            
            # cnt_step = cnt_step + 1
            r_1 = np.random.rand() # jump time
                                   # np.random.rand return a number from uniformly distributed [0,1)
            r_2 = np.random.rand() # decide which direction to go
            
            lamb = np.ones(self.num_reactions)
            for j in enumerate(lamb_func_list):
                lamb[j[0]] = j[1](x)
            lamb_log.append(lamb)
            lamb_for_compare = lamb.cumsum()/lamb.sum()
            
            '''
            for k in enumerate(direct_lamb_compare): # lamb_for_compare 1
                if not(r_2>k[1]):
                    mu = k[0] # return the numbering of the reaction that is taken
                    break
            '''
            mu = np.searchsorted(lamb_for_compare, r_2) # direct_lamb_compare

            delta_time = np.log(1/r_1)/(lamb.sum()) # lamb.sum() 2  direct_lamb.sum()
            t = t + delta_time
            time_log.append(t)
            jump_chi = 1 # np.random.poisson(lamb[mu]*delta_time)
                         # in Gillespie's each step exactly one step of mu-th reaction 
            if self.bath_boolean:
                x = x + jump_chi * self.bath_reaction_coef[mu]
            else:
                x = x + jump_chi * self.reaction_coef[mu]
            x_log.append(x)

            if (t-last_log_t)>dt:
                x_dist.append(x[2])
                last_log_t = t
                pass

        x_log = np.array(x_log)
        time_log = np.array(time_log)

        return x_log, time_log, np.array(x_dist)

    