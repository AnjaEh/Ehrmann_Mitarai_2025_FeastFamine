# -*- coding: utf-8 -*-
"""Feast Famine model for Ehrmann & Mitarai (2025) with drawing random alpha values. Generating data for plot 4D, plot with seperate script. 
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage OB (obligate burst)"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib import cm

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

    #Ensuring non-negativity of all variables

    #bacteria = max(0, bacteria)
    #infected_T4 = max(0, infected_T4)
    #infected_T7 = max(0, infected_T7)
    #phage_T4 = max(0, phage_T4)
    #phage_T7 = max(0, phage_T7)
    #substrate = max(0, substrate)

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
    alpha_values = []

    phage_T4_trend = np.array([initial_conditions[3]])
    phage_T7_trend = np.array([initial_conditions[4]])

    
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
        bacteria_sum_final = np.append(bacteria_sum_final, bacteria_total)
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
            infected_T7_final, bacteria_sum_final, substrate_final, starvation_time, alpha_values, phage_T4_trend, phage_T7_trend]


#Parameters and initial conditions ###########################################################

B_0 = 1.0e06
I_0 = 0
R_0 = 0
S_0 = 1.0e06
MOI = 1
P_0 = MOI * B_0
initial_conditions = [B_0, I_0, I_0, P_0, P_0,S_0]

pars = {}

pars['S_1'] = S_0
pars['K_s']  = 1.0e5
pars['max_mu']  = 1
pars['adsorption'] = 1.5e-8
pars['latency_T4'] = 0.8*(S_0/(pars['K_s']+S_0))
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 85
pars['burst_T4'] = 150

pars['p_deg'] = 0 
#pars['fresh_bacteria'] = 0.35*pars['S_1']
pars['fresh_bacteria'] = 2.0e4
pars['dilution'] = 0.1


################################################################################################


#calling simulation and extracting final result


# %%

#Calling simulation, extracting information on which phage wins 
# Running the simulation 100 times

scan_parameter_1 = 'burst size T7stat'
list_parameter_1 = np.linspace(22,150, num =14)

scan_parameter_2 = 'dilution rate'
list_parameter_2 = np.logspace(-3, -0.02, num = 12) # Dilution rate

#result_median = np.array([])
#result_all = np.empty(100)

result_trend_T4 = np.empty(100)
result_trend_T7 = np.empty(100)

results_3D = np.empty((len(list_parameter_1), len(list_parameter_2), 100))

#plt.figure()



for idx1, j in enumerate(list_parameter_1):
    pars['burst_T7_low'] = j
    print(f'Simulating for {scan_parameter_1}: {j}')

    
    for idx2, k in enumerate(list_parameter_2):
        pars['dilution'] = k

        #result_T4_par_i = np.array([])
        #result_T7_par_i = np.array([])

        #result_trend_T4_i = np.array([])
        #result_trend_T7_i = np.array([])

        #alpha_values_all = np.array([])
        print(f'Running parameter set {idx1+1} of {len(list_parameter_1)} with {idx2+1} of {len(list_parameter_2)}')
    
        for i in range(0,100):

                
            output = simulation(initial_conditions, pars)

            t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, infected_T7_final, \
                bacteria_sum_final, substrate_final, starvation_time, alpha_values, phage_T4_trend, phage_T7_trend = output
                    
                    
            # Extracting information which phage wins
            #t_n = 8000
            #t_x = t_n - 2000
            #x = np.argmin(np.abs(t_final - t_x))

            #Calculate time steps
            delta_T = np.insert(np.diff(t_final), 0, 0)

            #phage_T4_average = np.mean(phage_T4_final[x:]*delta_T[x:])
            #phage_T7_average = np.mean(phage_T7_final[x:]*delta_T[x:])
            #result_T4_par_i = np.append(result_T4_par_i, phage_T4_average)
            #result_T7_par_i = np.append(result_T7_par_i, phage_T7_average)

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

            #result_trend_T4_i = np.append(result_trend_T4_i, trend_T4_slope)
            #result_trend_T7_i = np.append(result_trend_T7_i, trend_T7_slope)

            result_trend_diff = trend_T7_slope - trend_T4_slope
            results_3D[idx1, idx2, i] = result_trend_diff

            #alpha_values_all = np.append(alpha_values_all, alpha_values)

            #if i == 0:
            #    print(f'T4 trend slope: {trend_T4_slope}')
            #    print(f'T7 trend slope: {trend_T7_slope}')
            #    fig, ax = plt.subplots()
            #    ax.plot(X, log_phage_T4_trend, label='T4')
            #    ax.plot(X, log_phage_T7_trend, label='T7')
            #    ax.plot(X, model_T4.predict(X), linestyle='--', color='blue')
            #    ax.plot(X, model_T7.predict(X), linestyle='--', color='orange')
            #    ax.set_xlabel('Cycle')
            #    ax.set_ylabel('log10(Phage)')
            #    ax.legend()
            #    plt.show()
            


        #result_T4_par_i[result_T4_par_i < 0.001] = 0.001
        #result_T7_par_i[result_T7_par_i < 0.001] = 0.001

        #ratio_phage = np.log10(np.divide(result_T7_par_i, result_T4_par_i))
        #median_ratio = np.median(ratio_phage)

        #result_median = np.append(result_median, median_ratio)
        #result_all = np.vstack((result_all, ratio_phage))

        #result_trend_T4 = np.vstack((result_trend_T4, result_trend_T4_i))
        #result_trend_T7 = np.vstack((result_trend_T7, result_trend_T7_i))

    

#Elementwise trend difference
#trend_diff = result_trend_T7 - result_trend_T4

#Convert arrays to list of lists so that it can be plotted as violin plot
#result_list = [result_all[i, :] for i in range(result_all.shape[0])]
#result_list = result_list[1:]

#result_trend_list_T4 = [result_trend_T4[i, :] for i in range(result_trend_T4.shape[0])]
#result_trend_list_T4 = result_trend_list_T4[1:]

#result_trend_list_T7 = [result_trend_T7[i, :] for i in range(result_trend_T7.shape[0])]
#result_trend_list_T7 = result_trend_list_T7[1:]

#result_trend_diff_list = [trend_diff[i, :] for i in range(trend_diff.shape[0])] 
#result_trend_diff_list = result_trend_diff_list[1:]

# %%
# export results to csv
#Export results
directory = os.path.join(os.getcwd())
print(directory)
#exportData(result_all[1:], directory, '20250314_stochastic_alpha_result_all.csv')
#exportData(trend_diff[1:], directory, '20250314_stochastic_alpha_trend_diff_t200.csv')

print(f"Working directory: {os.getcwd()}")
#exportData(results_3D.reshape(len(list_parameter_1), len(list_parameter_2)*100), os.getcwd(), '250711_results3D_reshape.csv')

# Export results_3D so that entries along the third axis are summarized in a list in one field
results_3D_list = [[list(results_3D[i, j, :]) for j in range(results_3D.shape[1])] for i in range(results_3D.shape[0])]
df_results_3D = pd.DataFrame(results_3D_list)
#df_results_3D.to_csv(os.path.join(os.getcwd(), '250711_results3D_list.csv'), sep=';', index=False)
print('Exported results_3D as list-of-lists to 250711_results3D_list.csv')

#%%

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





#%%
#Calculate row wise median of trend_diff
median_trend_diff = np.median(trend_diff[1:], axis = 1)

# Find the final entry in median_trend_diff with a negative result
negative_indices = np.where(median_trend_diff < 0)[0]
if len(negative_indices) > 0:
    final_negative_index = negative_indices[-1]
    corresponding_parameter = list_parameter_1[final_negative_index]
    print(f'Final negative median trend difference: {median_trend_diff[final_negative_index]}')
    print(f'Corresponding parameter in list_parameter_1: {corresponding_parameter}')
else:
    final_negative_index = 0

# %%
#Plotting the results as violin plot
plt.figure()
plt.violinplot(result_list, positions = list_parameter_1, showmedians = True, widths= 10)
plt.xlabel('Burst size T7stat')
plt.ylabel('log10(T7/T4)')
plt.show()

# %%
#Plotting the results as violin plot
plt.figure()
plt.boxplot(result_list, positions = list_parameter_1)
plt.xlabel('Burst size T7stat')
plt.ylabel('log10(T7/T4)')
plt.show()

# %%
#Plotting the results as violin plot
plt.figure()
plt.violinplot(result_trend_list_T4, positions = list_parameter_1, showmedians = True, widths= 10)
plt.xlabel('Burst size T7stat')
plt.ylabel('trend T4')
plt.show()

# %%
#Plotting the results as violin plot
plt.figure()
plt.violinplot(result_trend_list_T7, positions = list_parameter_1, showmedians = True, widths= 10)
plt.xlabel('Burst size T7stat')
plt.ylabel('trend T7')
plt.show()

# %%
#Plotting the results as violin plot
directory = os.path.join(os.getcwd(), 'Plots')
plt.figure()
plt.violinplot(result_trend_diff_list, positions = list_parameter_1, showmedians = True, widths= 10)
plt.xlabel('Burst size T7stat')
plt.ylabel('trend difference')
plt.hlines(0, 10, 160, color = '#800080', linestyle = '--')
plt.ylim(-1, 0.6)
#plt.savefig(os.path.join(directory, '20250314_stochastic_alpha_trend_diff_violin_t200.png'), dpi=600, bbox_inches='tight')
plt.show()


#%%
#Plotting the results
plt.figure()
plt.hist(ratio_phage, bins = 20)
plt.xlabel('log10(T7/T4)')
plt.ylabel('Frequency')
plt.show()


# %%
# Plot the results
plt.figure()
plt.plot(t_final[1:]/100, phage_T4_final[1:], label = 'Phage DL')
plt.plot(t_final[1:]/100, phage_T7_final[1:], label = 'Phage OB')
plt.yscale('log')
plt.xlabel('Time [cycles]')
plt.ylabel('Phage concentration [1/ml]')
plt.legend()
plt.ylim(bottom = 1e0, top = 1e9)

plt.show()


# %%
plt.figure()
plt.plot(t_final[1:], phage_T4_final[1:], label = 'Phage DL')
plt.plot(t_final[1:], phage_T7_final[1:], label = 'Phage OB')
#plt.plot(t_final[1:], (infected_T4_final[1:] + resis_T7_inf_T4_final[1:]), label = 'Infected DL')
#plt.plot(t_final[1:], (infected_T7_final[1:]+ resis_T4_inf_T7_final[1:]), label = 'Infected OB')
#plt.plot(t_final[1:], resistant_T4_final[1:], label = 'Resistant to DL')
#plt.plot(t_final[1:], resistant_T7_final[1:], label = 'Resistant to OB')
plt.plot(t_final[1:], bacteria_final[1:], label = 'Bacteria')
plt.plot(t_final[1:], substrate_final[1:], label = 'Substrate')
plt.yscale('log')
plt.xlabel('Time [generations]')
plt.ylabel('Concentration [1/ml]')
plt.legend()
plt.ylim(bottom = 1e0, top = 1e9)
plt.xlim(left = 400, right = 500)

plt.show()
# %%
