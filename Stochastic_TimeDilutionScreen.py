# -*- coding: utf-8 -*-
"""Feast Famine model for Ehrmann & Mitarai (2025). Adjusted script to screen length of the feast famine cycle and dilution factor. Generates data for supplementary figure S7.
    Takes very long to run. Data is exported as csv and can be plotted separately.
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage OB (obligate burst)"""



import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import pandas as pd
import os as os

def Load_data(directory, file):
    path_data = os.path.join(directory, file)
  
    data = pd.read_csv(path_data)
    data = data.fillna(value = 0)
    
    return data


def exportData(frame, directory, file):
    path_data = os.path.join(directory, file)
    
    if isinstance(frame, np.ndarray):
        df = pd.DataFrame(frame)
    else:
        df = frame

    df.to_csv(path_data, index = False)
    print('Exported to ' + directory + ' as csv as ' + file)
    return

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

    #Ensuring non-negativity of all variables, needed for solver stability for the extreme cases of short cycles / low dilution

    bacteria = max(0, bacteria)
    infected_T4 = max(0, infected_T4)
    infected_T7 = max(0, infected_T7)
    phage_T4 = max(0, phage_T4)
    phage_T7 = max(0, phage_T7)
    substrate = max(0, substrate)

    #adding a growth rate dependency term to both the latency time and burst rate
    dbacteria_dt = max_mu * ((substrate)/(K_s+substrate))*bacteria - adsorption*bacteria*phage_T4 - adsorption*bacteria*phage_T7
    
    dinfected_T4_dt = (adsorption*bacteria*phage_T4)  - infected_T4/latency_T4* ((substrate)/(K_s+substrate))
    dinfected_T7_dt = (adsorption*bacteria*phage_T7)  - infected_T7/latency_T7
    
    
    dphage_T4_dt = burst_T4*infected_T4/latency_T4* ((substrate)/(K_s+substrate)) - adsorption*phage_T4*(bacteria + infected_T4 + infected_T7 ) - p_deg * phage_T4
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
    infected_T4_starv_time = np.array([0])
    infected_T7_starv_time = np.array([0])
    alpha_values = []

    phage_T4_trend = np.array([initial_conditions[3]])
    phage_T7_trend = np.array([initial_conditions[4]])

    time_span = pars['time_span']

    
    for cycle in range(0,80):
        
        # Time span for simulation
        # One cycle is 100 units
        time_span_simulate = (0,time_span)
        
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
        
        infected_T4_starv = infected_T4[starvation_idx]
        infected_T7_starv = infected_T7[starvation_idx]
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
            infected_T7_final, bacteria_sum_final, substrate_final, starvation_time,
              infected_T4_starv_time, infected_T7_starv_time, alpha_values, phage_T4_trend, phage_T7_trend]


#Parameters and initial conditions ###########################################################

B_0 = 1.0e06
I_0 = 0
R_0 = 0
S_0 = 1.0e06

MOI = 1
P_0 = MOI * B_0
initial_conditions = [B_0, I_0, I_0, P_0, P_0,S_0]

# Defining parameters and default values
pars = {}

pars['S_1'] = S_0

pars['K_s']  = 1.0e5
pars['max_mu']  = 1
pars['adsorption'] = 1.5e-8
pars['latency_T4'] = 0.8*(S_0/(pars['K_s']+S_0)) # Adjustment so that latent period is the same at S_0
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 85
pars['burst_T4'] = 150
pars['p_deg'] = 0 
pars['fresh_bacteria'] = 0.35*pars['S_1']
pars['dilution'] = 0.1
pars['time_span'] = 100


################################################################################################


#calling simulation and extracting final result


# %%

#Calling simulation, extracting information on which phage wins 
# Running the simulation 100 times

scan_parameter_1 = 'burst size T7stat'
list_parameter_1 = np.linspace(20,150, num =13)


list_parameter_2 = np.linspace(10, 200, num = 10) # Duration of growth cycle
list_parameter_3 = np.logspace(-3, -0.02, num = 12) # Dilution rate
dilution_factor = 1/list_parameter_3

result_competitive_burst = np.empty_like(list_parameter_3)

for k in list_parameter_2:
    pars['time_span'] = k

    result_competitive_burst_par_k = np.array([])

    print(f'Scanning growth cycle duration: {k}')

    for m in list_parameter_3:
        pars['dilution'] = m
        
        # Initialize result arrays
        #result_median = np.array([])
        #result_all = np.empty(100)

        result_trend_T4 = np.empty(100)
        result_trend_T7 = np.empty(100)

        print(f'Scanning dilution rate: {m}')

        for j in list_parameter_1:
            # Scanning all stationary phase burst sizes for T7 phage 
            pars['burst_T7_low'] = j

            result_T4_par_i = np.array([])
            result_T7_par_i = np.array([])

            result_trend_T4_i = np.array([])
            result_trend_T7_i = np.array([])

            alpha_values_all = np.array([])

            for i in range(0,100):

                    
                output = simulation(initial_conditions, pars)

                t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, infected_T7_final, \
                    bacteria_sum_final, substrate_final, starvation_time, infected_T4_starv_time, infected_T7_starv_time, alpha_values, phage_T4_trend, phage_T7_trend = output
                        
                        
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

                alpha_values_all = np.append(alpha_values_all, alpha_values)

                                


            #ratio_phage = np.log10(np.divide(result_T7_par_i, result_T4_par_i))
            #median_ratio = np.median(ratio_phage)

            #result_median = np.append(result_median, median_ratio)
            #result_all = np.vstack((result_all, ratio_phage))

            result_trend_T4 = np.vstack((result_trend_T4, result_trend_T4_i))
            result_trend_T7 = np.vstack((result_trend_T7, result_trend_T7_i))

            # Plot histogram of alpha values
            #might need to add specific bins 
 

        #Elementwise trend difference
        trend_diff = result_trend_T7 - result_trend_T4

        median_trend_diff = np.median(trend_diff[1:], axis = 1)

        # Find the final entry in median_trend_diff with a negative result --> maximum burst size where T4 outcompetes T7
        negative_indices = np.where(median_trend_diff < 0)[0]
        if len(negative_indices) > 0:
            final_negative_index = negative_indices[-1]
            
        else:
            final_negative_index = 0
        result_competitive_burst_par_k = np.append(result_competitive_burst_par_k, final_negative_index)

        #Convert arrays to list of lists so that it can be plotted as violin plot
        #result_list = [result_all[i, :] for i in range(result_all.shape[0])]
        #result_list = result_list[1:]

        result_trend_list_T4 = [result_trend_T4[i, :] for i in range(result_trend_T4.shape[0])]
        result_trend_list_T4 = result_trend_list_T4[1:]

        result_trend_list_T7 = [result_trend_T7[i, :] for i in range(result_trend_T7.shape[0])]
        result_trend_list_T7 = result_trend_list_T7[1:]

        result_trend_diff_list = [trend_diff[i, :] for i in range(trend_diff.shape[0])] 
        result_trend_diff_list = result_trend_diff_list[1:]

    result_competitive_burst = np.vstack((result_competitive_burst, result_competitive_burst_par_k))

# Add list_parameter_2 and list_parameter_3 as row and column to result_competitive_burst
result_competitive_burst = np.vstack((list_parameter_3, result_competitive_burst[1:]))
result_competitive_burst = np.column_stack((list_parameter_2, result_competitive_burst))



# %%
# export results to csv
#Export results
directory = os.path.join(os.getcwd(), 'data')
print(directory)
##exportData(result_all[1:], directory, '20250227_stochastic_alpha_result_all.csv')
#exportData(trend_diff[1:], directory, '20250227_stochastic_alpha_trend_diff.csv')
#exportData(result_competitive_burst, directory, '20250228_stochastic_alpha_competitive_burst.csv')

 
