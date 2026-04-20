# -*- coding: utf-8 -*-

"""Feast Famine Model with paired alpha (Single condition) for Ehrmann & Mitarai (2025). Generates plots supplementary figures 4, 5
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage OB (obligate burst)"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import pandas as pd



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

     
    #adding a growth rate dependency term to both the latency time and burst rate
    dbacteria_dt = max_mu * ((substrate)/(K_s+substrate))*bacteria - adsorption*bacteria*phage_T4 - adsorption*bacteria*phage_T7
    
    dinfected_T4_dt = (adsorption*bacteria*phage_T4)  - infected_T4/latency_T4* ((substrate)/(K_s+substrate))
    dinfected_T7_dt = (adsorption*bacteria*phage_T7)  - infected_T7/latency_T7
       
    dphage_T4_dt = burst_T4*infected_T4/latency_T4* ((substrate)/(K_s+substrate)) - adsorption*phage_T4*(bacteria + infected_T4 + infected_T7) - p_deg * phage_T4
    dphage_T7_dt = burst_T7*infected_T7/latency_T7 - adsorption*phage_T7*(bacteria + infected_T4 + infected_T7) - p_deg * phage_T7
    
    dsubstrate_dt = (-1)*max_mu * ((substrate)/(K_s+substrate)) * (bacteria)
    
       
    return [dbacteria_dt, dinfected_T4_dt, dinfected_T7_dt, dphage_T4_dt, dphage_T7_dt, dsubstrate_dt]


def simulation(initial_conditions, pars):
    #iterate over different parameter sets

    
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
    
    bacteria_starv_time = np.array([0])
    
    phage_T4_trend = np.array([initial_conditions[3]])
    phage_T7_trend = np.array([initial_conditions[4]])
    
    for cycle in range(0,80):

        # One cycle is 100 units

        time_span_simulate = (0,100)
        
        #Unpack needed parameters
        dilution = pars['dilution']
        S_1 = pars['S_1']
        fresh_bacteria = pars['fresh_bacteria']

        #fluctuating parameter
        alpha1 = pars['alpha1']
        alpha2 = pars['alpha2']

        #Changing between alpha 1 and alpha 2 in even / uneven cycles
        if cycle % 2 == 0:
            fresh_bacteria = alpha1*S_1
            
        elif cycle % 2 != 0:
            fresh_bacteria = alpha2*S_1
            
        
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
        
        infected_T4_starv = infected_T4[starvation_idx]
        infected_T7_starv = infected_T7[starvation_idx]
        
        bacteria_starv = bacteria[starvation_idx]


        #convert starvation index to actual timepoint
        starvation = t[starvation_idx]
        
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
        infected_T4_starv_time = np.append(infected_T4_starv_time, infected_T4_starv)
        infected_T7_starv_time = np.append(infected_T7_starv_time, infected_T7_starv)
        
        bacteria_starv_time = np.append(bacteria_starv_time, bacteria_starv)
                
        #Output for trend analysis

        phage_T4_trend = np.append(phage_T4_trend, phage_T4[-1])
        phage_T7_trend = np.append(phage_T7_trend, phage_T7[-1])
        
        #change input for second cycle       
        
        B_cycle = bacteria[-1] * dilution + fresh_bacteria
        P_cycle_T4 = phage_T4[-1] *dilution
        P_cycle_T7 = phage_T7[-1] *dilution
        I_cycle_T4 = infected_T4[-1] *dilution
        I_cycle_T7 = infected_T7[-1] *dilution
        
        S_cycle = S_1
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
            infected_T7_final, bacteria_sum_final, substrate_final, starvation_time, infected_T4_starv_time, infected_T7_starv_time,
            bacteria_starv_time, phage_T4_trend, phage_T7_trend]


#Parameters and initial conditions ###########################################################

B_0 = 1.0e06
I_0 = 0
S_0 = 1.0e06


MOI = 1
P_0 = MOI * B_0
initial_conditions = [B_0, I_0, I_0, P_0, P_0,S_0]

#One substrate unit is what is needed for one bacterium to devide once
#The substrate amount therefore defines the capacity of the culture

pars = {}

pars['S_1'] = S_0
pars['K_s']  = 1e5
pars['max_mu']  = 1
pars['adsorption'] = 1.5e-8

pars['latency_T4'] = 0.8*(S_0/(pars['K_s']+S_0)) # Corrected so that latent period at S_0 is 0.8
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 65
pars['burst_T4'] = 150


pars['deg_bacteria'] = 0 
pars['p_deg'] = 0 
pars['fresh_bacteria'] = 1.0e6
pars['dilution'] = 0.1

pars['alpha1'] = 0.246
pars['alpha2'] = 0.491


#calling simulation and extracting final result

output = simulation(initial_conditions, pars)

t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, infected_T7_final, \
    bacteria_sum_final, substrate_final, starvation_time, infected_T4_starv_time, infected_T7_starv_time, \
        bacteria_starv_time, phage_T4_trend, phage_T7_trend = output


#this is only necessary if simulation time is adjusted
#set to fixed value of target cycles, exact time points might be off

t_n = 8000
t_x = t_n - 2000
x = np.argmin(np.abs(t_final - t_x))

#Calculate time steps
delta_T = np.insert(np.diff(t_final), 0, 0)

phage_T4_average = np.median(phage_T4_final[x:]*delta_T[x:])
phage_T7_average = np.median(phage_T7_final[x:]*delta_T[x:])

# Calculate log of phage trend values
log_phage_T4 = np.log10(phage_T4_final)
log_phage_T7 = np.log10(phage_T7_final)

# Prepare data for linear regression
# Set x values as cycle number

# Fit linear regression for T4 trend
model_T4 = LinearRegression().fit(t_final[500:].reshape(-1,1), log_phage_T4[500:])
trend_T4_slope = model_T4.coef_[0]

# Fit linear regression for T7 trend
model_T7 = LinearRegression().fit(t_final[500:].reshape(-1,1), log_phage_T7[500:])
trend_T7_slope = model_T7.coef_[0]

X = np.arange(0, t_n, 1).reshape(-1, 1)

# Plotting the results
# %%
# Plot the results
plt.figure()

plt.plot(t_final[1:], substrate_final[1:], label = 'substrate', color = '#21AB61')

plt.plot(t_final[1:], phage_T4_final[1:], label = 'phage DL', color = '#2B6DC3')
plt.plot(t_final[1:], phage_T7_final[1:], label = 'phage OB', color = '#AB4C1D')

plt.yscale('log')
plt.xlabel('Time [$T_g$]')
plt.ylabel('Phage concentration [1/ml]')
plt.legend()
plt.ylim(bottom = 1e0, top = 1e9)

#plt.savefig('Plots/Stoch_Supp_95_004_04_switch.png', dpi=600, bbox_inches='tight')
plt.show()

# %%
# Plot the results
plt.figure()
plt.plot(t_final[1:], phage_T4_final[1:], label = 'phage DL', color = '#2B6DC3')
plt.plot(t_final[1:], phage_T7_final[1:], label = 'phage OB', color = '#AB4C1D')
plt.plot(t_final[1:], substrate_final[1:], label = 'substrate', color = '#21AB61')
plt.plot(t_final[1:], bacteria_final[1:], label = 'bacteria', color = '#E0C465')
plt.plot(t_final[1:], infected_T4_final[1:], label = 'infected DL', color = '#2B6DC3', linestyle = '--', alpha = 0.7)
plt.plot(t_final[1:], infected_T7_final[1:], label = 'infected OB', color = '#AB4C1D', linestyle = '--', alpha = 0.7)

plt.yscale('log')
plt.xlabel('Time [$T_g$]')
plt.ylabel('Phage concentration [1/ml]')
plt.legend()
plt.ylim(bottom = 1e0, top = 1e9)
plt.xlim(left = 590, right = 680)
#plt.savefig('Plots/Stoch_Supp_95_0-004_0-4_paired.png', dpi=600, bbox_inches='tight')
plt.show()


# %%
