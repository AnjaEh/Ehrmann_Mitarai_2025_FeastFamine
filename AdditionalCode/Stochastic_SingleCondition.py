# -*- coding: utf-8 -*-
"""Feast Famine model for Ehrmann & Mitarai (2025) with drawing random alpha values. Running single parameter examples, generates plots that are represented in Supplementary Figure S6.
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage OB (obligate burst)"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



def system_of_equations(t, y, pars):
    # Unpack the variables
    bacteria, infected_T4, infected_T7, phage_T4, phage_T7, substrate = y
        
    K_s = pars['K_s']
    max_mu = pars['max_mu']
    adsorption = pars['adsorption'] 
    latency_T4 = pars['latency_T4'] 
    latency_T7 = pars['latency_T7'] 
    burst_T7_max = pars['burst_T7_max'] 
    burst_T7_low = pars['burst_T7_low'] 
    burst_T4 = pars['burst_T4'] 
 
    p_deg = pars['p_deg']

    #set condition for infection in stationary phase
    if ((substrate)/(K_s+substrate)) <= 0.2:
        burst_T7 = burst_T7_low
    else:
        burst_T7 = burst_T7_max
    # Define the system of differential equations

    dbacteria_dt = max_mu * ((substrate)/(K_s+substrate))*bacteria - adsorption*bacteria*phage_T4 - adsorption*bacteria*phage_T7
    
    dinfected_T4_dt = (adsorption*bacteria*phage_T4)  - infected_T4/latency_T4* ((substrate)/(K_s+substrate))
    dinfected_T7_dt = (adsorption*bacteria*phage_T7)  - infected_T7/latency_T7
    
        
    dphage_T4_dt = burst_T4*infected_T4/latency_T4* ((substrate)/(K_s+substrate))  - adsorption*phage_T4*(bacteria + infected_T4 + infected_T7) - p_deg * phage_T4
    dphage_T7_dt = burst_T7*infected_T7/latency_T7  - adsorption*phage_T7*(bacteria + infected_T4 + infected_T7) - p_deg * phage_T7
    
    dsubstrate_dt = (-1)*max_mu * ((substrate)/(K_s+substrate)) * (bacteria)
    
       
    return [dbacteria_dt, dinfected_T4_dt, dinfected_T7_dt, dphage_T4_dt, dphage_T7_dt, dsubstrate_dt]


def simulation(initial_conditions, pars):
    #iterate over different parameter sets


    #initializing t0 and output variables
    t0 = 0

    t_final = np.array([0])
    bacteria_final = np.array([0])
    infected_T4_final = np.array([0])
    infected_T7_final = np.array([0])

    phage_T4_final = np.array([0])
    phage_T7_final = np.array([0])
    substrate_final = np.array([0])
    bacteria_sum_final = np.array([0])
    starvation_time = np.array([0])

    alpha_values = []
    

    
    for cycle in range(0,80):
        
        # Time span for simulation
        time_span_simulate = (0,100)
        
        #Unpack needed parameters
        dilution = pars['dilution']
        S_1 = pars['S_1']
        fresh_bacteria = pars['fresh_bacteria']
        #fluctuating parameter

        #Picking a random alpha from list
        alpha_list = np.logspace(-3, 0.3, num =20)
        fresh_bacteria = np.random.choice(alpha_list)*S_1
        # Save the picked alpha value
                    
        alpha_values.append(fresh_bacteria / S_1)


        # Solve the system of differential equations
        solution = solve_ivp(system_of_equations, time_span_simulate, initial_conditions, max_step = 1, args=(pars,))
        
        # Extract the results
        t = solution.t
        bacteria = solution.y[0]
        infected_T4 = solution.y[1]
        infected_T7 = solution.y[2]        
        phage_T4 = solution.y[3]
        phage_T7 = solution.y[4]
        substrate = solution.y[5]
        
        bacteria_total = np.sum([bacteria, infected_T4, infected_T7], axis = 0)
        
        
        #determine time until substrate exhaustion
        if np.any(substrate <= 2.5e4):
            starvation_idx = np.asarray(substrate <= 2.5e4).nonzero()[0][0]
        else:
            starvation_idx = substrate.shape[0] -1
               
        #convert starvation index to actual timepoint
        starvation = t[starvation_idx]
        #append to a starvation array (starvation_time)
        
        #append results to longer time frame
        t = solution.t + t_final[-1]
        t_final = np.append(t_final,t)
        bacteria_final = np.append(bacteria_final, bacteria)
        infected_T4_final = np.append(infected_T4_final, infected_T4)
        infected_T7_final = np.append(infected_T7_final, infected_T7)
      
        phage_T4_final = np.append(phage_T4_final, phage_T4)
        phage_T7_final = np.append(phage_T7_final, phage_T7)
        bacteria_sum_final = np.append(bacteria_sum_final, bacteria_total)
        substrate_final = np.append(substrate_final, substrate)
        starvation_time = np.append(starvation_time, starvation)

                
        #change input for second cycle
 
        B_cycle = bacteria[-1] * dilution + fresh_bacteria
        P_cycle_T4 = phage_T4[-1] *dilution
        P_cycle_T7 = phage_T7[-1] *dilution
        I_cycle_T4 = infected_T4[-1] *dilution
        I_cycle_T7 = infected_T7[-1] *dilution
        S_cycle = S_1
        #Generate time point for reset event
        t0 = t[-1] + 1

        bacteria_0 = np.sum([B_cycle, I_cycle_T4, I_cycle_T7])

        t_final = np.append(t_final,t0)
        bacteria_final = np.append(bacteria_final, B_cycle)
        infected_T4_final = np.append(infected_T4_final, I_cycle_T4)
        infected_T7_final = np.append(infected_T7_final, I_cycle_T7)        
        phage_T4_final = np.append(phage_T4_final, P_cycle_T4)
        phage_T7_final = np.append(phage_T7_final, P_cycle_T7)
        bacteria_sum_final = np.append(bacteria_sum_final, bacteria_0)
        substrate_final = np.append(substrate_final, S_cycle)
        
        initial_conditions = [B_cycle, I_cycle_T4, I_cycle_T7, P_cycle_T4, P_cycle_T7, S_cycle]
        

    return [t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final,
            infected_T7_final, bacteria_sum_final, substrate_final, starvation_time, alpha_values]


#Parameters and initial conditions ###########################################################

B_0 = 1.0e06
I_0 = 0
S_0 = 1.0e06
MOI = 1
P_0 = MOI * B_0
initial_conditions = [B_0, I_0, I_0, P_0, P_0,S_0]

pars = {}

pars['S_1'] = S_0
pars['K_s']  = 1e5
pars['max_mu']  = 1
pars['adsorption'] = 1.5e-8
pars['latency_T4'] = 0.8*(S_0/(pars['K_s']+S_0))
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 90
pars['burst_T4'] = 150

pars['p_deg'] = 0 
pars['fresh_bacteria'] = 0.35*pars['S_1']
pars['dilution'] = 0.1


################################################################################################


#calling simulation and extracting final result


# %%

# Running the simulation 100 times

result_T4_par_i = np.array([])
result_T7_par_i = np.array([])

for i in range(0,100):

        
    output = simulation(initial_conditions, pars)

    t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, infected_T7_final, \
        bacteria_sum_final, substrate_final, starvation_time, alpha_values = output
            
            
    # Extracting information which phage wins
    t_n = 8000
    t_x = t_n - 2000
    x = np.argmin(np.abs(t_final - t_x))

    #Calculate time steps
    delta_T = np.insert(np.diff(t_final), 0, 0)

    phage_T4_average = np.mean(phage_T4_final[x:]*delta_T[x:])
    phage_T7_average = np.mean(phage_T7_final[x:]*delta_T[x:])
    result_T4_par_i = np.append(result_T4_par_i, phage_T4_average)
    result_T7_par_i = np.append(result_T7_par_i, phage_T7_average)

    if i % 10 == 0:
        plt.figure()
        plt.plot(t_final[1:], phage_T4_final[1:], label = 'Phage DL')
        plt.plot(t_final[1:], phage_T7_final[1:], label = 'Phage OB')
        plt.yscale('log')
        plt.xlabel('Time [cycles]')
        plt.ylabel('Phage concentration [1/ml]')
        plt.legend()
        plt.ylim(bottom = 1e0, top = 1e9)
        #plt.savefig('Plots/StochExamples/stochastic_burst_low'+str(pars['burst_T7_low'])+'example'+str(i)+'.svg', format ='svg', dpi = 300, bbox_inches = 'tight')
        plt.show()


 

