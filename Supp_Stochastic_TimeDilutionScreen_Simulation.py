# -*- coding: utf-8 -*-
"""Feast Famine model for Ehrmann & Mitarai (2026). Adjusted script to screen length of the feast famine cycle and dilution factor. Generates data for supplementary figure S9.
    Takes very long to run on a regular laptop. Data is exported as csv and can be plotted separately.
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage RB (reduced burst)"""



import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import pandas as pd
import os as os
from datetime import date

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
    bacteria, phage_T4, phage_T7, substrate, infected_T4_1, infected_T4_2, infected_T4_3, infected_T4_4, infected_T4_5, infected_T7_1, infected_T7_2, infected_T7_3, infected_T7_4, infected_T7_5 = y
    
    K_s = pars['K_s']
    max_mu = pars['max_mu']
    adsorption = pars['adsorption'] 
    latency_T4 = pars['latency_T4'] 
    latency_T7 = pars['latency_T7'] 
    burst_T7_max = pars['burst_T7_max'] 
    burst_T7_low = pars['burst_T7_low'] 
    burst_T4 = pars['burst_T4'] 
    p_deg = pars['p_deg']
    gamma_delay = pars['gamma_delay']

    #set condition for infection in stationary phase
    if ((substrate)/(K_s+substrate)) <= 0.2:
        burst_T7 = burst_T7_low
    else:
        burst_T7 = burst_T7_max
    # Define the system of differential equations

    latency_delay = latency_T4/(((substrate/(K_s + substrate))/(S_0/(K_s+S_0)))**gamma_delay)
    n = 5

    #Ensuring non-negativity of all variables, needed for solver stability for the extreme cases of short cycles / low dilution

    bacteria = max(0, bacteria)
    infected_T4_1 = max(0, infected_T4_1)
    infected_T4_2 = max(0, infected_T4_2)
    infected_T4_3 = max(0, infected_T4_3)
    infected_T4_4 = max(0, infected_T4_4)
    infected_T4_5 = max(0, infected_T4_5)
    infected_T7_1 = max(0, infected_T7_1)
    infected_T7_2 = max(0, infected_T7_2)
    infected_T7_3 = max(0, infected_T7_3)
    infected_T7_4 = max(0, infected_T7_4)
    infected_T7_5 = max(0, infected_T7_5)
    phage_T4 = max(0, phage_T4)
    phage_T7 = max(0, phage_T7)
    substrate = max(0, substrate)

    #adding a growth rate dependency term to both the latency time and burst rate
    dbacteria_dt = max_mu * ((substrate)/(K_s+substrate))*bacteria - adsorption*bacteria*phage_T4 - adsorption*bacteria*phage_T7
    
    d_infected_T4_1 = (adsorption*bacteria*phage_T4)  - n/latency_delay * infected_T4_1

    d_infected_T4_2 = n/latency_delay * (infected_T4_1 - infected_T4_2)
    d_infected_T4_3 = n/latency_delay * (infected_T4_2 - infected_T4_3)
    d_infected_T4_4 = n/latency_delay * (infected_T4_3 - infected_T4_4)
    d_infected_T4_5 = n/latency_delay * (infected_T4_4 - infected_T4_5)


    d_infected_T7_1 = (adsorption*bacteria*phage_T7)  - n/latency_T7 * infected_T7_1
    d_infected_T7_2 = n/latency_T7 * (infected_T7_1 - infected_T7_2)
    d_infected_T7_3 = n/latency_T7 * (infected_T7_2 - infected_T7_3)
    d_infected_T7_4 = n/latency_T7 * (infected_T7_3 - infected_T7_4)
    d_infected_T7_5 = n/latency_T7 * (infected_T7_4 - infected_T7_5)

    sum_infected = infected_T4_1 + infected_T4_2 + infected_T4_3 + infected_T4_4 + infected_T4_5 + infected_T7_1 + infected_T7_2 + infected_T7_3 + infected_T7_4 + infected_T7_5
    
    dphage_T4_dt = burst_T4*infected_T4_5*n/latency_delay - adsorption*phage_T4*(bacteria + sum_infected) - p_deg * phage_T4
    dphage_T7_dt = burst_T7*infected_T7_5*n/latency_T7 - adsorption*phage_T7*(bacteria + sum_infected) - p_deg * phage_T7
    
    

    dsubstrate_dt = (-1)*max_mu * ((substrate)/(K_s+substrate)) * (bacteria)     
       
    return [dbacteria_dt, dphage_T4_dt, dphage_T7_dt, dsubstrate_dt, d_infected_T4_1, d_infected_T4_2, d_infected_T4_3, d_infected_T4_4, d_infected_T4_5, d_infected_T7_1, d_infected_T7_2, d_infected_T7_3, d_infected_T7_4, d_infected_T7_5]


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

    phage_T4_trend = np.array([initial_conditions[1]])
    phage_T7_trend = np.array([initial_conditions[2]])

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
                    

        # Solve the system of differential equations
        solution = solve_ivp(system_of_equations, time_span_simulate, initial_conditions, max_step = 0.1, args=(pars,))
        
        # Extract the results
        t = solution.t
        bacteria = solution.y[0]
        infected_T4 = solution.y[8] + solution.y[7] + solution.y[6] + solution.y[5] + solution.y[4]
        infected_T7 = solution.y[13] + solution.y[12] + solution.y[11] + solution.y[10] + solution.y[9]
        phage_T4 = solution.y[1]
        phage_T7 = solution.y[2]
        substrate = solution.y[3]  
        
        
        #append results to longer time frame
        t = solution.t + t_final[-1]
        t_final = np.append(t_final,t)
        bacteria_final = np.append(bacteria_final, bacteria)
        infected_T4_final = np.append(infected_T4_final, infected_T4)
        infected_T7_final = np.append(infected_T7_final, infected_T7)
        phage_T4_final = np.append(phage_T4_final, phage_T4)
        phage_T7_final = np.append(phage_T7_final, phage_T7)
        substrate_final = np.append(substrate_final, substrate)

                      
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

        #bacteria_0 = np.sum([B_cycle, I_cycle_T4, I_cycle_T7])

        t_final = np.append(t_final,t0)
        bacteria_final = np.append(bacteria_final, B_cycle)
        infected_T4_final = np.append(infected_T4_final, I_cycle_T4)
        infected_T7_final = np.append(infected_T7_final, I_cycle_T7)
        phage_T4_final = np.append(phage_T4_final, P_cycle_T4)
        phage_T7_final = np.append(phage_T7_final, P_cycle_T7)
        substrate_final = np.append(substrate_final, S_cycle)


        initial_conditions = [B_cycle, P_cycle_T4, P_cycle_T7, S_cycle, solution.y[4][-1]*dilution, solution.y[5][-1]*dilution, solution.y[6][-1]*dilution, solution.y[7][-1]*dilution, solution.y[8][-1]*dilution,
                              solution.y[9][-1]*dilution, solution.y[10][-1]*dilution, solution.y[11][-1]*dilution, solution.y[12][-1]*dilution, solution.y[13][-1]*dilution]        

    return [t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final,
            infected_T7_final, substrate_final, phage_T4_trend, phage_T7_trend]


#Parameters and initial conditions ###########################################################

B_0 = 1.0e06
I_0 = 0
R_0 = 0
S_0 = 1.0e06

MOI = 1
P_0 = MOI * B_0
initial_conditions = [B_0, P_0, P_0, S_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0]

# Defining parameters and default values
pars = {}

pars['S_1'] = S_0

pars['K_s']  = 1.0e5
pars['max_mu']  = 1
pars['adsorption'] = 1.5e-8
pars['latency_T4'] = 0.8
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 85
pars['burst_T4'] = 150
pars['p_deg'] = 0 
pars['fresh_bacteria'] = 0.35*pars['S_1']
pars['dilution'] = 0.1
pars['time_span'] = 100
pars['gamma_delay'] = 1.0


################################################################################################


#calling simulation and extracting final result


# %%

#Calling simulation, extracting information on which phage wins 
# Running the simulation 100 times

scan_parameter_1 = 'burst size T7stat'
list_parameter_1 = np.linspace(22,150, num =14)

list_parameter_2 = np.array([10, 31, 52, 75, 100, 150, 200, 250])

list_parameter_3 = np.array([1/100, 1/50, 1/25, 1/10, 1/6, 1/4, 1/2, 1/1.1])
dilution_factor = 1/list_parameter_3

result_competitive_burst = np.empty_like(list_parameter_3)

for k in list_parameter_2:
    pars['time_span'] = k

    result_competitive_burst_par_k = np.array([])

    print(f'Scanning growth cycle duration: {k}')

    for m in list_parameter_3:
        pars['dilution'] = m
        

        result_trend_T4 = np.empty(100)
        result_trend_T7 = np.empty(100)

        print(f'Scanning dilution rate: {m}')

        for j in list_parameter_1:
            # Scanning all stationary phase burst sizes for T7 phage 
            pars['burst_T7_low'] = j
            print(f'Scanning burst size T7 in stationary phase: {j}')


            result_trend_T4_i = np.array([])
            result_trend_T7_i = np.array([])


            for i in range(0,100):

                    
                output = simulation(initial_conditions, pars)

                t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, \
                    infected_T7_final, substrate_final, phage_T4_trend, phage_T7_trend = output
                        
                        
                # Extracting information which phage wins
                t_n = 8000
                t_x = t_n - 2000
                x = np.argmin(np.abs(t_final - t_x))

                #Calculate time steps
                delta_T = np.insert(np.diff(t_final), 0, 0)


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

                


            result_trend_T4 = np.vstack((result_trend_T4, result_trend_T4_i))
            result_trend_T7 = np.vstack((result_trend_T7, result_trend_T7_i))

 

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
      

    result_competitive_burst = np.vstack((result_competitive_burst, result_competitive_burst_par_k))

# Add list_parameter_2 and list_parameter_3 as row and column to result_competitive_burst
result_competitive_burst = np.vstack((list_parameter_3, result_competitive_burst[1:]))
result_competitive_burst = np.column_stack((list_parameter_2, result_competitive_burst))



# %%
# export results to csv
#Export results
directory = os.path.join(os.getcwd(), 'Data')
print(directory)
gamma_delay = pars['gamma_delay']
today = date.today().strftime('%Y%m%d')

#Uncomment to export data as csv, can be plotted separately
#exportData(result_competitive_burst, directory, f'{today}_stochastic_alpha_TimeDilutionScreen_competitive_burst.csv')

 
