# -*- coding: utf-8 -*-
"""Feast Famine model for Ehrmann & Mitarai (2025). Script to screen paired alpha fluctuations. Generates Figures 4B
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage OB (obligate burst)"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os as os

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
    
    dphage_T4_dt = burst_T4*infected_T4/latency_T4* ((substrate)/(K_s+substrate))  - adsorption*phage_T4*(bacteria + infected_T4 + infected_T7) - p_deg * phage_T4
    dphage_T7_dt = burst_T7*infected_T7/latency_T7  - adsorption*phage_T7*(bacteria + infected_T4 + infected_T7 ) - p_deg * phage_T7
    
    dsubstrate_dt = (-1)*max_mu * ((substrate)/(K_s+substrate)) * (bacteria)
    
       
    return [dbacteria_dt, dinfected_T4_dt, dinfected_T7_dt, dphage_T4_dt, dphage_T7_dt, dsubstrate_dt]


def simulation(initial_conditions, pars):

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
    
    bacteria_starv_time = np.array([0])
    
    phage_T4_trend = np.array([initial_conditions[3]])
    phage_T7_trend = np.array([initial_conditions[4]])
    
    for cycle in range(0,80):
        
        # Time span for simulation
        # assuming one generation time as time unit
        # One cycle is 100 units
        #time_span = (t0, t0+100)
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
R_0 = 0
S_0 = 1.0e06
#S_1 = S_0
#t0 = 0
MOI = 1
P_0 = MOI * B_0
initial_conditions = [B_0, I_0, I_0, P_0, P_0,S_0]


#One substrate unit is what is needed for one bacterium to devide once

pars = {}

pars['S_1'] = S_0
pars['K_s']  = 1e5
pars['max_mu']  = 1
pars['adsorption'] = 1.5e-8

pars['latency_T4'] = 0.8*(S_0/(pars['K_s']+S_0))
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 65
pars['burst_T4'] = 150

pars['deg_bacteria'] = 0 
pars['p_deg'] = 0 
pars['fresh_bacteria'] = 1.0e6
pars['dilution'] = 0.1

pars['alpha1'] = 0.1
pars['alpha2'] = 0.1



# Define the timepoints range
start_time = 404
end_time = 504




################################################################################################


#calling simulation and extracting final result

#2D parameter scan


scan_parameter_1 = 'alpha 1'
list_parameter_1 = np.logspace(-3, 0.3, num =20)

scan_parameter_2 = 'alpha 2'
list_parameter_2 = np.logspace(-3, 0.3, num =20)

result_T4 = np.empty_like(list_parameter_1)
result_T7 = np.empty_like(list_parameter_1)
result_substrate = np.empty_like(list_parameter_1)
result_starvation = np.empty_like(list_parameter_1)
result_infected_T4_starv = np.empty_like(list_parameter_1)
result_infected_T7_starv = np.empty_like(list_parameter_1)

result_bacteria_starv = np.empty_like(list_parameter_1)


# Trend results
result_trend_T4 = np.empty_like(list_parameter_1)
result_trend_T7 = np.empty_like(list_parameter_1)


for j in list_parameter_2:
    pars['alpha2'] = j
    
    
    #Initalize temporary result arrays
    
    result_T4_par_i = np.array([])
    result_T7_par_i = np.array([])
    result_starv_i = np.array([])
    result_inf_T4_i = np.array([])
    result_inf_T7_i = np.array([])
   
    result_bac_i = np.array([])

    result_trend_T4_i = np.array([])
    result_trend_T7_i = np.array([])

    print(f'alpha2 {j}')
    
    for i in list_parameter_1:
        print(f'Screening alpha1 {i}')
        pars['alpha1'] = i
       
        output = simulation(initial_conditions, pars)
        
        t_final,  phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, infected_T7_final, \
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
        result_T4_par_i = np.append(result_T4_par_i, phage_T4_average)
        result_T7_par_i = np.append(result_T7_par_i, phage_T7_average)
        
        #stravation time and infected cells at starvation averages
        starvation_average = np.median(starvation_time[-20:])
        infected_T4_starv_average = np.median(infected_T4_starv_time[-20:])
        infected_T7_starv_average = np.median(infected_T7_starv_time[-20:])
        
        bacteria_starv_average = np.median(bacteria_starv_time[-30:])
        result_starv_i = np.append(result_starv_i, starvation_average)
        result_inf_T4_i = np.append(result_inf_T4_i, infected_T4_starv_average)
        result_inf_T7_i = np.append(result_inf_T7_i, infected_T7_starv_average)
        
        result_bac_i = np.append(result_bac_i, bacteria_starv_average)
        
        #trend analysis
        # Calculate log of phage trend values
        log_phage_T4_trend = np.log10(phage_T4_trend)
        log_phage_T7_trend = np.log10(phage_T7_trend)

        # Prepare data for linear regression
        # Set x values as cycle number
        X = np.arange(len(log_phage_T4_trend)).reshape(-1, 1)

        # Fit linear regression for T4 trend
        model_T4 = LinearRegression().fit(X, log_phage_T4_trend)
        trend_T4_slope = model_T4.coef_[0]

        # Fit linear regression for T7 trend
        model_T7 = LinearRegression().fit(X, log_phage_T7_trend)
        trend_T7_slope = model_T7.coef_[0]

        result_trend_T4_i = np.append(result_trend_T4_i, trend_T4_slope)
        result_trend_T7_i = np.append(result_trend_T7_i, trend_T7_slope)


    # Return to outer loop    
    result_T4 = np.vstack((result_T4, result_T4_par_i))
    result_T7 = np.vstack((result_T7, result_T7_par_i))
    result_starvation = np.vstack((result_starvation, result_starv_i))
    result_infected_T4_starv = np.vstack((result_infected_T4_starv, result_inf_T4_i))
    result_infected_T7_starv = np.vstack((result_infected_T7_starv, result_inf_T7_i))
   
    result_bacteria_starv = np.vstack((result_bacteria_starv, result_bac_i))

    #Append trend results
    result_trend_T4 = np.vstack((result_trend_T4, result_trend_T4_i))
    result_trend_T7 = np.vstack((result_trend_T7, result_trend_T7_i))
    

#Summarize trend results
result_trend_T4 = result_trend_T4[1:,:]
result_trend_T7 = result_trend_T7[1:,:]

result_trend_diff = np.subtract(result_trend_T7, result_trend_T4)

ratio_phage = np.log10(np.divide(result_T7[1:,:], result_T4[1:,:]))
        


fig, ax = plt.subplots()
im = ax.imshow(np.flipud((result_trend_diff)),vmin = -0.5, vmax = 0.5, interpolation = 'none', cmap = 'coolwarm')
fig.colorbar(im, ax=ax)
ax.set_title('Trend difference')
#plt.savefig(os.path.join(directory, '20250314_paired_burst65_trend.png'), dpi=600, bbox_inches='tight')
plt.show()







