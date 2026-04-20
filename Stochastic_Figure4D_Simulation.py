# -*- coding: utf-8 -*-
"""Feast Famine model for Ehrmann & Mitarai (2026) with drawing random alpha values. Generating data for plot 4D, plot with seperate script. 
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage RB (reduced burst)"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib import cm

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
    starvation_time = np.array([0])


    phage_T4_trend = np.array([initial_conditions[1]])
    phage_T7_trend = np.array([initial_conditions[2]])

    
    for cycle in range(0,80):
        
        # Time span for simulation
        # One cycle is 100 units
        time_span_simulate = (0,100)
        
        #Unpack needed parameters
        dilution = pars['dilution']
        S_1 = pars['S_1']
        fresh_bacteria = pars['fresh_bacteria']
        #fluctuating parameter

        #Picking a random alpha from list
        alpha_list = np.logspace(-3, 0.3, num =20)
        fresh_bacteria = np.random.choice(alpha_list)*S_1

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

        #determine time until substrate exhaustion
        if np.any(substrate <= 1000):
            starvation_idx = np.asarray(substrate <= 2.5e4).nonzero()[0][0]
        else:
            starvation_idx = substrate.shape[0] -1

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
        substrate_final = np.append(substrate_final, substrate)
        starvation_time = np.append(starvation_time, starvation)    
               
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
        # Create time point for the reset event
        t0 = t[-1] + 1


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
R_0 = 0
S_0 = 1.0e06
MOI = 1
P_0 = MOI * B_0
initial_conditions = [B_0, P_0, P_0, S_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0]

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
#pars['fresh_bacteria'] = 0.35*pars['S_1']
pars['fresh_bacteria'] = 2.0e4
pars['dilution'] = 0.1
pars['gamma_delay'] = 1.0

################################################################################################


#calling simulation and extracting final result


# %%

#Calling simulation, extracting information on which phage wins 
# Running the simulation 100 times

scan_parameter_1 = 'burst size T7stat'
list_parameter_1 = np.linspace(22,150, num =14)
#If running in multiple pars:
#list_parameter_1 = list_parameter_1[9:]

scan_parameter_2 = 'dilution rate'
list_parameter_2 = np.array([1/100, 1/50, 1/25, 1/10, 1/6, 1/4]) # Dilution rate


result_trend_T4 = np.empty(100)
result_trend_T7 = np.empty(100)

results_3D = np.empty((len(list_parameter_1), len(list_parameter_2), 100))

#plt.figure()



for idx1, j in enumerate(list_parameter_1):
    pars['burst_T7_low'] = j
    print(f'Simulating for {scan_parameter_1}: {j}')

    
    for idx2, k in enumerate(list_parameter_2):
        pars['dilution'] = k

        print(f'Running parameter set {idx1+1} of {len(list_parameter_1)} with {idx2+1} of {len(list_parameter_2)}')
        print(f'Current parameters: {scan_parameter_1} = {j}, {scan_parameter_2} = {k}')
    
        for i in range(0,100):

                
            output = simulation(initial_conditions, pars)

            t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, infected_T7_final, \
                substrate_final, starvation_time, phage_T4_trend, phage_T7_trend = output
                    


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


            result_trend_diff = trend_T7_slope - trend_T4_slope
            results_3D[idx1, idx2, i] = result_trend_diff


# %%
# export results to csv
#Export results
directory = os.path.join(os.getcwd(), 'Data')
print(directory)
gamma_delay = pars['gamma_delay']
today = date.today().strftime('%Y%m%d')

print(f"Working directory: {os.getcwd()}")
exportData(results_3D.reshape(len(list_parameter_1), len(list_parameter_2)*100), directory, f'{today}_Stochastic_DilutionScreen_GammaDelay_{gamma_delay}_results3D_reshape_Part2.csv')

# Export results_3D so that entries along the third axis are summarized in a list in one field
results_3D_list = [[list(results_3D[i, j, :]) for j in range(results_3D.shape[1])] for i in range(results_3D.shape[0])]
df_results_3D = pd.DataFrame(results_3D_list)
df_results_3D.to_csv(os.path.join(directory, f'{today}_Stochastic_DilutionScreen_GammaDelay_{gamma_delay}_results3D_list_Part2.csv'), sep=';', index=False)
#print('Exported results_3D as list-of-lists to 250711_results3D_list.csv')

#%%

### Preliminary Figure, real figure generated with Stochastic_Figure4D_plotting.py

plt.figure(figsize=(12, 6))

# Normalize colors for the number of list_parameter_1 values
cmap = cm.get_cmap('viridis', len(list_parameter_1))
colors = [cmap(i) for i in range(len(list_parameter_1))]

for idx1, param1 in enumerate(list_parameter_1):
    data = [results_3D[idx1, idx2, :] for idx2 in range(len(list_parameter_2))]
    parts = plt.violinplot(
        data,
        positions=np.arange(len(list_parameter_2)),
        showmeans=False,
        showmedians=True,
        widths=0.8 / len(list_parameter_1),  # Make violins narrower for overlay
    )
    for pc in parts['bodies']:
        pc.set_facecolor(colors[idx1])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
    # Color the median lines
    if 'cmedians' in parts:
        parts['cmedians'].set_color(colors[idx1])

# Set x-ticks to correspond to list_parameter_2 values
plt.xticks(np.arange(len(list_parameter_2)), [f"{v:.2g}" for v in list_parameter_2], rotation=45)
plt.xlabel('Parameter 2 (dilution rate)')
plt.ylabel('Delta m (OB - DL trend)')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(list_parameter_1), vmax=max(list_parameter_1)))
cbar = plt.colorbar(sm, ticks=[min(list_parameter_1), max(list_parameter_1)])
cbar.ax.set_yticklabels([f"{min(list_parameter_1):.2g}", f"{max(list_parameter_1):.2g}"])
cbar.set_label('Parameter 1 (burst size T7stat)')
plt.title('Violin plot of trend values by dilution rate and burst size')
plt.tight_layout()
plt.show()



