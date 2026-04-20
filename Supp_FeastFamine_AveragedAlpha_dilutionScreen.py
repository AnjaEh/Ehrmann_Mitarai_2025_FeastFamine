# -*- coding: utf-8 -*-
"""

@author: Anja Ehrmann
Feast Famine Model for Ehrmann & Mitarai (2026). Generates plots for Supplementary Figure S10B. Testing contant alpha with different dilution factors. 
Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage RB (reduced burst)


"""


import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import gmean
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import pandas as pd
import os as os
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

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
    #Code for one simulation of the feast famine system with (usually) 80 growth cycles.  

    #initializing t0 and empty result arrays
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


    #Initalizing vector that stores final phage concentration from each cycle for regression and slope analysis
    phage_T4_trend = np.array([initial_conditions[1]])
    phage_T7_trend = np.array([initial_conditions[2]])
    
        
    for cycle in range(0,80):
        
        # Time span for simulation
        time_span_simulate = (0,100) #Length of each feast famine cycle can be adjusted here
        
        #Unpack needed parameters
        dilution = pars['dilution']
        S_1 = pars['S_1']
        fresh_bacteria = pars['fresh_bacteria']
        
        # Solve the system of differential equations
        solution = solve_ivp(system_of_equations, time_span_simulate, initial_conditions, max_step = 0.11, args=(pars,))
        
        # Extract the results
        t = solution.t
        bacteria = solution.y[0]
        infected_T4 = solution.y[8] + solution.y[7] + solution.y[6] + solution.y[5] + solution.y[4]
        infected_T7 = solution.y[13] + solution.y[12] + solution.y[11] + solution.y[10] + solution.y[9]
        phage_T4 = solution.y[1]
        phage_T7 = solution.y[2]
        substrate = solution.y[3] 
        
        #bacteria_total = np.sum([bacteria, infected_T4, infected_T7], axis = 0)
        
        
        #determine time until substrate drops below the starvation threshold
        if np.any(substrate <= 2.5e4):
            starvation_idx = np.asarray(substrate <= 2.5e4).nonzero()[0][0]
        else:
            starvation_idx = substrate.shape[0] -1
       
        #convert starvation index to actual timepoint
        starvation = t[starvation_idx]

                
        #append results from single cycle to full results array
        t = solution.t + t_final[-1]
        t_final = np.append(t_final,t)
        bacteria_final = np.append(bacteria_final, bacteria)
        infected_T4_final = np.append(infected_T4_final, infected_T4)
        infected_T7_final = np.append(infected_T7_final, infected_T7)
        
        phage_T4_final = np.append(phage_T4_final, phage_T4)
        phage_T7_final = np.append(phage_T7_final, phage_T7)
        substrate_final = np.append(substrate_final, substrate)
        starvation_time = np.append(starvation_time, starvation)
               
        # Append phage concentration at the end of the feast famine cycle to array for regression and slope analysis

        phage_T4_trend = np.append(phage_T4_trend, phage_T4[-1])
        phage_T7_trend = np.append(phage_T7_trend, phage_T7[-1])        
        
        # Reset initial conditions for the next feast famine cycle    
                
        B_cycle = bacteria[-1] * dilution + fresh_bacteria
        P_cycle_T4 = phage_T4[-1] *dilution
        P_cycle_T7 = phage_T7[-1] *dilution
        I_cycle_T4 = infected_T4[-1] *dilution
        I_cycle_T7 = infected_T7[-1] *dilution
        
        S_cycle = S_1
        t0 = t[-1] + 1

        #bacteria_0 = np.sum([B_cycle, I_cycle_T4, I_cycle_T7])

        #Append new initial conditions to results array in order to have clear record of the reset point
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
            infected_T7_final, substrate_final, starvation_time, phage_T4_trend, phage_T7_trend]


#Parameters and initial conditions ###########################################################

B_0 = 1.0e06
I_0 = 0
S_0 = 1.0e06

MOI = 1 #initial multiplicity of infection, sets initial condition for both phages
P_0 = MOI * B_0
initial_conditions = [B_0, P_0, P_0, S_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0]

pars = {}

pars['S_1'] = S_0
pars['K_s']  = 1e5 # K_s = S_0/10
pars['max_mu']  = 1 
pars['adsorption'] = 1.5e-8
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 50
pars['burst_T4'] = 150

pars['p_deg'] = 0

pars['fresh_bacteria'] = 0.045*1.0e6
pars['dilution'] = 0.1
pars['gamma_delay'] = 1.0

pars['latency_T4'] = 0.8


# 0.8 * 30 min generation time = 24 min 

################################################################################################


#calling simulation and extracting final result
# Defining parameter ranges
scan_parameter_1 = 'fresh bacteria/substrate'
list_parameter_1 = np.logspace(-3, 0.3, num =20)

scan_parameter_2 = 'burst size T7stat'
list_parameter_2 = np.linspace(22, 150, 14)

scan_parameter_3 = 'dilution'
list_parameter_3 = np.array([1/100, 1/50, 1/25, 1/10, 1/6, 1/4])


# Initalize empty result arrays

result_T4 = np.empty_like(list_parameter_3)
result_T7 = np.empty_like(list_parameter_3)
result_trend_T4 = np.empty_like(list_parameter_3)
result_trend_T7 = np.empty_like(list_parameter_3)



for j in list_parameter_2:
    pars['burst_T7_low'] = j
    
    # Initialize temporary result arrays for each inner loop   
    
    result_T4_par_i = np.array([])
    result_T7_par_i = np.array([])
    result_starv_i = np.array([])
    result_sub_i = np.array([])
    result_trend_T4_i = np.array([])
    result_trend_T7_i = np.array([])  


    print(f'Screening burst T7 low {j}')
    
    #for i in list_parameter_1:
    for i in list_parameter_3:
        print(f'Screening dilution {i}')
        #print(f'Screening alpha {i}')
        #pars['fresh_bacteria'] = i * pars[ 'S_1']

        pars['dilution'] = i
        
        output = simulation(initial_conditions, pars)
        
        t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, infected_T7_final, \
            substrate_final, starvation_time, phage_T4_trend, phage_T7_trend = output
        
        
        #this is only necessary if simulation time is adjusted
        #set to fixed value of target cycles, exact time points might be off
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

        
        
    result_T4 = np.vstack((result_T4, result_T4_par_i))
    result_T7 = np.vstack((result_T7, result_T7_par_i))
    
        
    #Append trend results
    result_trend_T4 = np.vstack((result_trend_T4, result_trend_T4_i))
    result_trend_T7 = np.vstack((result_trend_T7, result_trend_T7_i))

 
#Summarize trend results
result_trend_T4 = result_trend_T4[1:,:]
result_trend_T7 = result_trend_T7[1:,:]

result_trend_diff = np.subtract(result_trend_T7, result_trend_T4)

directory = os.path.join(os.getcwd(), 'Data')
print(directory)
today = date.today().strftime('%Y%m%d')

#%%
# Start and end index can be adjusted to plot only a subset of the dilution factors, e.g. from 1/25 to 1/4
start_idx = 0
end_idx = len(list_parameter_3)
param2_range = np.arange(start_idx, end_idx)
param2_values = 1/list_parameter_3[start_idx:end_idx]

# Reverse the order for plotting: left = end_idx, right = start_idx
param2_range_rev = param2_range[::-1]
param2_values_rev = param2_values[::-1]

# Normalize colors for the number of list_parameter_1 values
cmap = cm.get_cmap('twilight_shifted', len(list_parameter_2))
colors = [cmap(i) for i in range(len(list_parameter_2))]

# Wider violins and staggered positions
#violin_width = 3 / len(list_parameter_1)
offset = np.linspace(-0.3, 0.3, len(list_parameter_2))

for idx1, param1 in enumerate(list_parameter_2):
    print(idx1, param1)
    data = [result_trend_diff[idx1, idx2] for idx2 in param2_range]
    positions = param2_range_rev + offset[idx1]
    plt.scatter(positions, data, color=colors[idx1], s=15)

#Coexistence and slow exclusion lines    
plt.axhline(0.07, color='black', linestyle=':', linewidth=1, alpha = 0.5)
plt.axhline(-0.07, color='black', linestyle=':', linewidth=1, alpha = 0.5)
# Shaded neutral band around zero
plt.axhspan(-0.0025, 0.0025, color='grey', alpha=0.2, zorder=0)

plt.xticks(param2_range, [f"{v:.2g}" for v in param2_values_rev], rotation=45)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Parameter 2 (dilution rate)')
plt.ylabel('Delta m (OB - DL trend)')
plt.title('Violin plot of trend values by dilution rate and burst size')
plt.tight_layout()
plt.savefig(f'Plots/{today}_DilutionFactor_fixedAlpha.svg', format = 'svg',  dpi=600, bbox_inches='tight')
plt.show()


