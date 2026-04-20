# -*- coding: utf-8 -*-
"""Feast Famine Model (Single condition) for Ehrmann & Mitarai (2025). Generates plots for figure 3, and supplementary figures 4, 5 
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
    
    dbacteria_dt = max_mu * ((substrate)/(K_s+substrate))*bacteria  - adsorption*bacteria*phage_T4 - adsorption*bacteria*phage_T7
    
    dinfected_T4_dt = (adsorption*bacteria*phage_T4)  - infected_T4/latency_T4* ((substrate)/(K_s+substrate))
    dinfected_T7_dt = (adsorption*bacteria*phage_T7)  - infected_T7/latency_T7
            
    dphage_T4_dt = burst_T4*infected_T4/latency_T4* ((substrate)/(K_s+substrate)) - adsorption*phage_T4*(bacteria + infected_T4 + infected_T7 ) - p_deg * phage_T4
    dphage_T7_dt = burst_T7*infected_T7/latency_T7 - adsorption*phage_T7*(bacteria + infected_T4 + infected_T7) - p_deg * phage_T7
    
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
    infected_T4_starv_time = np.array([0])
    infected_T7_starv_time = np.array([0])

    for cycle in range(0,80):
        
        # Time span for simulation
        # assuming one generation time as time unit
        # One cycle is 100 units
        
        time_span_simulate = (0,100)
        
        #Unpack needed parameters
        dilution = pars['dilution']
        S_1 = pars['S_1']
        fresh_bacteria = pars['fresh_bacteria']
        
        # Solve the system of differential equations
        solution = solve_ivp(system_of_equations, time_span_simulate, initial_conditions, max_step =1, args=(pars,))
        
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
        
        infected_T4_starv = infected_T4[starvation_idx]
        infected_T7_starv = infected_T7[starvation_idx]
        #convert starvation index to actual timepoint
        starvation = t[starvation_idx]
        #append to a starvation array (starvation_time)

        
        #append results to longer time frame
        # change t0 to the end of the current cycle - Simulation time runs from 0 to 100 every time
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
        infected_T4_starv_time = np.append(infected_T4_starv_time, infected_T4_starv)
        infected_T7_starv_time = np.append(infected_T7_starv_time, infected_T7_starv)

      
        #change input for second cycle
        
        B_cycle = bacteria[-1]* dilution + fresh_bacteria
        P_cycle_T4 = phage_T4[-1] *dilution
        P_cycle_T7 = phage_T7[-1] *dilution
        I_cycle_T4 = infected_T4[-1] *dilution
        I_cycle_T7 = infected_T7[-1] *dilution
        
        S_cycle = S_1
        # Create time point for reset event
        t0 = t[-1] + 1
       

        t_final = np.append(t_final,t0)
        bacteria_final = np.append(bacteria_final, B_cycle)
        infected_T4_final = np.append(infected_T4_final, I_cycle_T4)
        infected_T7_final = np.append(infected_T7_final, I_cycle_T7)        
        phage_T4_final = np.append(phage_T4_final, P_cycle_T4)
        phage_T7_final = np.append(phage_T7_final, P_cycle_T7)        
        substrate_final = np.append(substrate_final, S_cycle)
        
        initial_conditions = [B_cycle, I_cycle_T4, I_cycle_T7, P_cycle_T4, P_cycle_T7, S_cycle]
        

    return [t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final,
            infected_T7_final, substrate_final, starvation_time, infected_T4_starv_time, infected_T7_starv_time]


#Parameters and initial conditions ###########################################################

B_0 = 1.0e06
I_0 = 0
R_0 = 0
S_0 = 1.0e06
MOI = 1
P_0 = MOI * B_0


#One substrate unit is what is needed for one bacterium to devide once
#The substrate amount therefore defines the capacity of the culture

pars = {}

pars['S_1'] = S_0

pars['K_s']  = 1e5
pars['max_mu']  = 1
pars['adsorption'] = 1.5e-8 # ml/Tg
pars['latency_T4'] = 0.8*(S_0/(pars['K_s']+S_0))
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 65
pars['burst_T4'] = 150


pars['p_deg'] = 0 
pars['fresh_bacteria'] = 0.347*pars['S_1']
pars['dilution'] = 0.1

# Calling simulation

initial_conditions_T7 = [B_0, I_0, I_0, P_0, P_0, S_0]


# %%
# Iterate over the parameter sets

# Run the simulation
output = simulation(initial_conditions_T7, pars)
t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, infected_T7_final, \
    substrate_final, starvation_time, infected_T4_starv_time, infected_T7_starv_time = output

starvation_average = np.median(starvation_time)
print('Median time of starvation:', starvation_average)


# %%
#Plot all the different components of the fate cycle over time
plt.figure()
plt.plot(t_final, phage_T4_final, label = 'phage DL', color = '#2B6DC3')
plt.plot(t_final, phage_T7_final, label = 'phage OB', color = '#AB4C1D')
plt.plot(t_final, infected_T7_final, label = 'infected OB', color = '#AB4C1D', linestyle = '--', linewidth = 2)
plt.plot(t_final, substrate_final, label = 'substrate', color = '#21AB61')
plt.plot(t_final, bacteria_final, label = 'bacteria', color = '#E0C465')
plt.plot(t_final, infected_T4_final, label = 'infected DL', color = '#2B6DC3', linestyle = '--', linewidth = 2)
plt.yscale('log')
plt.xlabel(r"Time [$\mathrm{T_g}$]")  
plt.ylabel('Concentration [1/ml]')
plt.ylim(bottom = 1e0, top = 1e9)
plt.xlim(left = 0, right = 400)
plt.legend(loc='lower right')
plt.title(r"$\gamma_{stat} = 0.433$, $\alpha = 0.347$")
#plt.savefig('Plots/Supp_FeastFamine_130_0-01_zoom.png', dpi=600, bbox_inches='tight')
#plt.savefig('Plots/FeastFamine_50_01_zoom_ext.png', dpi=600, bbox_inches='tight')
plt.show()

# %%
#Plot all the different components of the fate cycle over time
plt.figure()
plt.plot(t_final, phage_T4_final, label = 'phage DL', color = '#2B6DC3')
plt.plot(t_final, phage_T7_final, label = 'phage OB', color = '#AB4C1D')
plt.yscale('log')
plt.xlabel(r"Time [$\mathrm{T_g}$]")   
plt.ylabel('Concentration [1/ml]')
plt.ylim(bottom = 1e0, top = 1e9)
plt.xlim(left = 0, right = 8000)
plt.legend(loc='lower right')
plt.title(r"$\gamma_{stat} = 0.433$, $\alpha = 0.347$")
#plt.savefig('Plots/Supp_FeastFamine_130_0-01_full.png', dpi=600, bbox_inches='tight')
plt.show()






