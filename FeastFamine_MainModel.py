# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:14:48 2024

@author: Anja Ehrmann
Feast Famine Model (Parameter Screens) for Ehrmann & Mitarai (2025). Generates plots for figure 3, and supplementary figures 2, 3 
Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage OB (obligate burst)

"""


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
  
    #Set condition for phage_T7 (OB) to reduce burst size when S (and bacterial growth rate) drops below starvation threshold
    if ((substrate)/(K_s+substrate)) <= 0.2:
        burst_T7 = burst_T7_low
    else:
        burst_T7 = burst_T7_max
    # Define the system of differential equations

    
    dbacteria_dt = max_mu * ((substrate)/(K_s+substrate))*bacteria - adsorption*bacteria*phage_T4 - adsorption*bacteria*phage_T7
    
    dinfected_T4_dt = (adsorption*bacteria*phage_T4)  - infected_T4/latency_T4* ((substrate)/(K_s+substrate))
    dinfected_T7_dt = (adsorption*bacteria*phage_T7)  - infected_T7/latency_T7
    
    dphage_T4_dt = burst_T4*infected_T4/latency_T4* ((substrate)/(K_s+substrate))  - adsorption*phage_T4*(bacteria + infected_T4 + infected_T7) - p_deg * phage_T4
    dphage_T7_dt = burst_T7*infected_T7/latency_T7 - adsorption*phage_T7*(bacteria + infected_T4 + infected_T7) - p_deg * phage_T7
    
    #Alternative phage equations without adsorption to infected cells
    #dphage_T4_dt = burst_T4*infected_T4/latency_T4* ((substrate)/(K_s+substrate)) - adsorption*phage_T4*bacteria - p_deg * phage_T4
    #dphage_T7_dt = burst_T7*infected_T7/latency_T7 - adsorption*phage_T7*bacteria - p_deg * phage_T7
    

    dsubstrate_dt = (-1)*max_mu * ((substrate)/(K_s+substrate)) * (bacteria)
    
       
    return [dbacteria_dt, dinfected_T4_dt, dinfected_T7_dt, dphage_T4_dt, dphage_T7_dt, dsubstrate_dt]


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
    infected_T4_starv_time = np.array([0])
    infected_T7_starv_time = np.array([0])
    
    bacteria_starv_time = np.array([0])

    latency_delay = np.array([0])
    latency_delay_mean = np.array([0])

    infected_T4_end = np.array([0])
    phage_T4_cycle1 = np.array([0])
    phage_T7_cycle1 = np.array([0])

    #Initalizing vector that stores final phage concentration from each cycle for regression and slope analysis
    phage_T4_trend = np.array([initial_conditions[3]])
    phage_T7_trend = np.array([initial_conditions[4]])
    
        
    for cycle in range(0,80):
        
        # Time span for simulation
        # assuming one (minimum) generation time as time unit
        # One cycle is usually 100 units

        time_span_simulate = (0,100) #Length of each feast famine cycle can be adjusted here
        
        #Unpack needed parameters
        dilution = pars['dilution']
        S_1 = pars['S_1']
        fresh_bacteria = pars['fresh_bacteria']
        
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
           
        #determine time until substrate drops below the starvation threshold
        if np.any(substrate <= 2.5e4):
            starvation_idx = np.asarray(substrate <= 2.5e4).nonzero()[0][0]
        else:
            starvation_idx = substrate.shape[0] -1
        
        # Extract the concentration of each type of bacterial cell at the timepoint when S < 2.5e4
        infected_T4_starv = infected_T4[starvation_idx] 
        infected_T7_starv = infected_T7[starvation_idx]        
        bacteria_starv = bacteria[starvation_idx]

        #convert starvation index to actual timepoint
        starvation = t[starvation_idx]

        # Calculate typical latency time for T4
        delta_T_cycle = np.insert(np.diff(t), 0, 0)
        substrate_window_cycle = np.where(infected_T4 > 1, substrate, np.nan) # filter substrate values where infected T4 > 1 (cells with potentially delayed lysis available), otherwise add np.nan
        growth_rate_window_cycle = substrate_window_cycle/(pars['K_s']+substrate_window_cycle) # calculate growth rate at the filtered timepoints
        latency_T4_window_cycle = pars['latency_T4']/growth_rate_window_cycle # Calculate latency time for T4/phage DL at the filtered timepoints

        # Weigh values in the window by the time step and calculate the median / mean
        typical_latency_T4_cycle = np.nanmedian((latency_T4_window_cycle*delta_T_cycle))        
        typical_latency_T4_mean_c = np.nanmean((latency_T4_window_cycle*delta_T_cycle))

        # If the result is nan, which means that the window is empty, set the latency to the initial value, phage DL is then not exposed to lowered substrate concentration.
        # The effective latency of phage DL is equal to phage OB at S = S_0, = latency_T7
        if np.isnan(typical_latency_T4_cycle):
            typical_latency_T4_cycle = pars['latency_T7']            
            typical_latency_T4_mean_c = pars['latency_T7']

        
        #append results from single cycle to full results array
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

        # Append latency delay result and normalize by latency of OB phage
        latency_delay = np.append(latency_delay, typical_latency_T4_cycle/pars['latency_T7'])
        latency_delay_mean = np.append(latency_delay_mean, typical_latency_T4_mean_c/pars['latency_T7']) 

        infected_T4_end = np.append(infected_T4_end, infected_T4[-1])
        
               
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
        #Generate time point entry for the reset
        t0 = t[-1] + 1

        bacteria_0 = np.sum([B_cycle, I_cycle_T4, I_cycle_T7])

        #Append new initial conditions to results array in order to have clear record of the reset point
        t_final = np.append(t_final,t0)
        bacteria_final = np.append(bacteria_final, B_cycle)
        infected_T4_final = np.append(infected_T4_final, I_cycle_T4)
        infected_T7_final = np.append(infected_T7_final, I_cycle_T7)
        
        phage_T4_final = np.append(phage_T4_final, P_cycle_T4)
        phage_T7_final = np.append(phage_T7_final, P_cycle_T7)
        bacteria_sum_final = np.append(bacteria_sum_final, bacteria_0)
        substrate_final = np.append(substrate_final, S_cycle)

        if cycle == 0:
        #    print(bacteria_sum_final)
        #    print(bacteria_final)
            phage_T4_cycle1 = phage_T4[-1]
            phage_T7_cycle1 = phage_T7[-1]
        
        initial_conditions = [B_cycle, I_cycle_T4, I_cycle_T7, P_cycle_T4, P_cycle_T7, S_cycle]
        
        

    return [t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final,
            infected_T7_final, bacteria_sum_final, substrate_final, starvation_time, infected_T4_starv_time, infected_T7_starv_time,
            bacteria_starv_time, phage_T4_trend, phage_T7_trend, latency_delay, latency_delay_mean, infected_T4_end, phage_T4_cycle1, phage_T7_cycle1]


#Parameters and initial conditions ###########################################################

B_0 = 1.0e06
I_0 = 0
S_0 = 1.0e06

MOI = 1 #initial multiplicity of infection, sets initial condition for both phages
P_0 = MOI * B_0
initial_conditions = [B_0, I_0, I_0, P_0, P_0,S_0]

pars = {}

pars['S_1'] = S_0
pars['K_s']  = 1e5 # K_s = S_0/10
pars['max_mu']  = 1 
pars['adsorption'] = 1.5e-8 # ml/Tg
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 50
pars['burst_T4'] = 150
pars['p_deg'] = 0

pars['fresh_bacteria'] = 1.0e6
pars['dilution'] = 0.1

pars['latency_T4'] = 0.8*(S_0/(pars['K_s']+S_0)) #Adjustment of latency_T4 to ensure that effectively phage OB and phage DL have equal latency time at S = S_0


# 0.8 * 30 min generation time = 24 min 

################################################################################################


#calling simulation and extracting final result

#2D parameter scan
#index i : screen x axis (paramter 1)
#index j : screen y axis (parameter 2)

scan_parameter_1 = 'fresh bacteria/substrate'
list_parameter_1 = np.logspace(-3, 0.3, num =20)

scan_parameter_2 = 'burst size T7stat'
list_parameter_2 = np.linspace(22, 150, 14)

# theoretical time until substrate is fully consumed
#t_starv = (1/pars['max_mu'])*np.log(pars['S_1']/(pars['K_s']*0.2 + pars['S_1']))
time_substrate_v2 = 1 * np.log2((list_parameter_1 + 1) / list_parameter_1)
time_substrate_0 = 1 * np.log((list_parameter_1 + 1) / list_parameter_1)
time_substrate_norm = time_substrate_0/pars['latency_T7']


# Initalize empty result arrays

result_T4 = np.empty_like(list_parameter_1)
result_T7 = np.empty_like(list_parameter_1)
result_substrate = np.empty_like(list_parameter_1)
result_starvation = np.empty_like(list_parameter_1)
result_infected_T4_starv = np.empty_like(list_parameter_1)
result_infected_T7_starv = np.empty_like(list_parameter_1)
result_bacteria_starv = np.empty_like(list_parameter_1)
result_trend_T4 = np.empty_like(list_parameter_1)
result_trend_T7 = np.empty_like(list_parameter_1)
result_latency_T4 = np.empty_like(list_parameter_1)
result_latency_T4_mean = np.empty_like(list_parameter_1)
result_infected_T4_cycle5 = np.empty_like(list_parameter_1)
result_infected_T4_cycle30 = np.empty_like(list_parameter_1)
result_T4_cycle1 = np.empty_like(list_parameter_1)
result_T7_cycle1 = np.empty_like(list_parameter_1)


for j in list_parameter_2:
    pars['burst_T7_low'] = j
    
    # Initialize temporary result arrays for each inner loop   
    
    result_T4_par_i = np.array([])
    result_T7_par_i = np.array([])
    result_starv_i = np.array([])
    result_inf_T4_i = np.array([])
    result_inf_T7_i = np.array([])    
    result_bac_i = np.array([])
    result_sub_i = np.array([])
    result_trend_T4_i = np.array([])
    result_trend_T7_i = np.array([])
    result_latency_T4_i = np.array([])
    result_latency_T4_mean_i = np.array([])
    result_infected_T4_5_par_i = np.array([])
    result_infected_T4_30_par_i = np.array([])
    result_phage_T4_cycle1_i = np.array([])
    result_phage_T7_cycle1_i = np.array([])


    print(f'Screening burst T7 low {j}')
    
    for i in list_parameter_1:
        print(f'Screening alpha {i}')
        pars['fresh_bacteria'] = i * pars[ 'S_1']
        
        
        output = simulation(initial_conditions, pars)
        
        t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, infected_T7_final, \
            bacteria_sum_final, substrate_final, starvation_time, infected_T4_starv_time, infected_T7_starv_time, \
                 bacteria_starv_time, phage_T4_trend, phage_T7_trend, latency_delay, latency_delay_mean, infected_T4_end, \
                     phage_T4_cycle1, phage_T7_cycle1 = output
        
        
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
        

        starvation_average = np.median(starvation_time)
        infected_T4_starv_average = np.median(infected_T4_starv_time)
        infected_T7_starv_average = np.median(infected_T7_starv_time)
        
        bacteria_starv_average = np.median(bacteria_starv_time)
        result_starv_i = np.append(result_starv_i, starvation_average)
        result_inf_T4_i = np.append(result_inf_T4_i, infected_T4_starv_average)
        result_inf_T7_i = np.append(result_inf_T7_i, infected_T7_starv_average)
        
        result_bac_i = np.append(result_bac_i, bacteria_starv_average)
        result_sub_i = np.append(result_sub_i, np.mean(substrate_final[x:]*delta_T[x:]))


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

        result_latency_T4_i = np.append(result_latency_T4_i, np.nanmedian(latency_delay))
        result_latency_T4_mean_i = np.append(result_latency_T4_mean_i, np.nanmedian(latency_delay_mean))
        
        result_infected_T4_5_par_i = np.append(result_infected_T4_5_par_i, infected_T4_end[5])
        result_infected_T4_30_par_i = np.append(result_infected_T4_30_par_i, infected_T4_end[30])

        result_phage_T4_cycle1_i = np.append(result_phage_T4_cycle1_i, phage_T4_cycle1)
        result_phage_T7_cycle1_i = np.append(result_phage_T7_cycle1_i, phage_T7_cycle1)
                       
        
    result_T4 = np.vstack((result_T4, result_T4_par_i))
    result_T7 = np.vstack((result_T7, result_T7_par_i))
    result_starvation = np.vstack((result_starvation, result_starv_i))
    result_infected_T4_starv = np.vstack((result_infected_T4_starv, result_inf_T4_i))
    result_infected_T7_starv = np.vstack((result_infected_T7_starv, result_inf_T7_i))
    
    result_bacteria_starv = np.vstack((result_bacteria_starv, result_bac_i))
    result_substrate = np.vstack((result_substrate, result_sub_i))
    
        
    #Append trend results
    result_trend_T4 = np.vstack((result_trend_T4, result_trend_T4_i))
    result_trend_T7 = np.vstack((result_trend_T7, result_trend_T7_i))

    # Append latency results
    result_latency_T4 = np.vstack((result_latency_T4, result_latency_T4_i))
    result_latency_T4_mean = np.vstack((result_latency_T4_mean, result_latency_T4_mean_i))

    # Append infected T4 results
    result_infected_T4_cycle5 = np.vstack((result_infected_T4_cycle5, result_infected_T4_5_par_i))
    result_infected_T4_cycle30 = np.vstack((result_infected_T4_cycle30, result_infected_T4_30_par_i))

    # Append phage concentration at cycle 1 results
    result_T4_cycle1 = np.vstack((result_T4_cycle1, result_phage_T4_cycle1_i))
    result_T7_cycle1 = np.vstack((result_T7_cycle1, result_phage_T7_cycle1_i))


#Summarize trend results
result_trend_T4 = result_trend_T4[1:,:]
result_trend_T7 = result_trend_T7[1:,:]

result_trend_diff = np.subtract(result_trend_T7, result_trend_T4)

ratio_phage = np.log10(np.divide(result_T7[1:,:], result_T4[1:,:]))

        

#Export results
directory = os.path.join(os.getcwd(), 'data')
print(directory)
#exportData(ratio_phage, directory, '20250114_constant_alpha_result_all.csv')
#exportData(result_trend_diff, directory, '20250114_constant_alpha_trend_diff.csv')



kwargs={'vmin': -12, 'vmax': 12}

#Ratio of phage concentrations
fig, ax = plt.subplots()
im = ax.imshow(np.flipud(ratio_phage), interpolation = 'none', vmin = -12, vmax = 12, cmap = 'coolwarm')
fig.colorbar(im, ax=ax, ticks = [-12,-4, 0,4,12], shrink = 0.75)
#plt.savefig('Plots/Feast_Supp_PhageRatio.png', dpi=600, bbox_inches='tight')
plt.show()

#Difference of phage slopes
fig, ax = plt.subplots()
im = ax.imshow(np.flipud((result_trend_diff)),vmin = -0.5, vmax = 0.5, interpolation = 'none', cmap = 'coolwarm')
fig.colorbar(im, ax=ax, shrink = 0.75)
#plt.savefig('Plots/Feast_TrendDiff_DLinvading.svg', format = 'svg', dpi=600, bbox_inches='tight')
plt.show()

plt.figure()
plt.scatter(list_parameter_1, time_substrate_norm, label = 'theoretical starvation time, normalized to latent period')
plt.scatter(list_parameter_1, time_substrate_0, label = 'theoretical starvation time ln')
plt.xlabel('alpha - bacteria/substrate')
plt.ylabel('t')
plt.legend()
plt.xscale('log')
plt.show()



# Outcome concentration phage DL
fig, ax = plt.subplots()
im = ax.imshow(np.flipud(np.log10(result_T4[1:,:])), interpolation = 'none', vmin = 5, vmax = 8, cmap = 'YlGnBu')
fig.colorbar(im, ax=ax, ticks = [-3,0,3,6,9], shrink = 0.75)
ax.set_title('T4')
#plt.savefig('Plots/Feast_Supp_PhageLogDL.png', dpi=600, bbox_inches='tight')
plt.show()

#Outcome concentration phage OB
fig, ax = plt.subplots()
im = ax.imshow(np.flipud((result_T7[1:,:])), interpolation = 'none', vmin = 5e7, vmax = 3e8, cmap = 'YlGnBu')
fig.colorbar(im, ax=ax, shrink = 0.75)
ax.set_title('T7')
#plt.savefig('Plots/Feast_Supp_PhageLogOB.png', dpi=600, bbox_inches='tight')
plt.show()

# Cycle 1 phage concentration
fig, ax = plt.subplots()
im = ax.imshow(np.flipud(np.log10(result_T7_cycle1[1:,:])), interpolation = 'none', vmin = 7, vmax = 8.5, cmap = 'YlGnBu')
fig.colorbar(im, ax=ax, ticks = [7,7.5, 8,8.5], shrink = 0.75)
ax.set_title('T7 after cycle 1')
#plt.savefig('Plots/Feast_Supp_T7afterCycle1.svg', format = 'svg', dpi=600, bbox_inches='tight')
plt.show()

# Cycle 1 phage concentration
fig, ax = plt.subplots()
im = ax.imshow(np.flipud((result_T4_cycle1[1:,:])), interpolation = 'none', cmap = 'YlGnBu')
fig.colorbar(im, ax=ax, shrink = 0.75)
ax.set_title('T4 after cycle 1')
#plt.savefig('Plots/Feast_Supp_PhageLogDL.png', dpi=600, bbox_inches='tight')
plt.show()

# Cycle 1 total phage concentration
fig, ax = plt.subplots()
im = ax.imshow(np.flipud(np.log10(result_T4_cycle1[1:,:] + result_T7_cycle1[1:,:])), interpolation = 'none', vmin = 7, vmax = 8.5, cmap = 'YlGnBu')
fig.colorbar(im, ax=ax, ticks = [-3,0,3,7,8], shrink = 0.75)
ax.set_title('Total phage titer after cycle 1 (log)')
#plt.savefig('Plots/Feast_Supp_PhageLogDL.png', dpi=600, bbox_inches='tight')
plt.show()



#Typical time until starvation
fig, ax = plt.subplots()
im = ax.imshow(np.flipud(result_starvation[1:,:]), interpolation = 'none', vmin = 0, vmax = 20, cmap = 'RdYlBu')
fig.colorbar(im, ax=ax, shrink = 0.75)
ax.set_title('starvation time')
#plt.savefig('Plots/Feast_StarvationTime.svg', format = 'svg', dpi=600, bbox_inches='tight')
plt.show()

#Effective latent period of phage DL
fig, ax = plt.subplots()
im = ax.imshow(np.flipud(result_latency_T4_mean[1:,:]), interpolation = 'none', vmin = 0, vmax = 30, cmap = 'RdYlBu')
fig.colorbar(im, ax=ax, shrink = 0.75)
ax.set_title('Relative latency T4')
#plt.savefig('Plots/Feast_RelativeLatency.png', dpi=600, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(np.flipud((result_latency_T4[1:,:])), interpolation = 'none', vmin = 0, vmax = 30, cmap = 'RdYlBu')
fig.colorbar(im, ax=ax, shrink = 0.75)
ax.set_title('Relative latency median log T4')
#plt.savefig('Plots/Feast_RelativeLatencyMedian.png', dpi=600, bbox_inches='tight')
plt.show()

# Infected with phage DL at cycle 5 and 30
fig, ax = plt.subplots()
im = ax.imshow(np.flipud(np.log10(result_infected_T4_cycle5[1:,:])), interpolation = 'none', vmin = -1, vmax = 6, cmap = 'RdYlBu')
fig.colorbar(im, ax=ax, shrink = 0.75)
ax.set_title('Infected DL cells at the end of cycle 5')
#plt.savefig('Plots/Feast_Supp_InfectedT4Cycle5.png', dpi=600, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(np.flipud(np.log10(result_infected_T4_cycle30[1:,:])), interpolation = 'none', vmin = -1, vmax = 6, cmap = 'RdYlBu')
fig.colorbar(im, ax=ax, shrink = 0.75)
ax.set_title('Infected DL cells at the end of cycle 30')
#plt.savefig('Plots/Feast_InfectedT4Cycle30.png', dpi=600, bbox_inches='tight')
plt.show()


#Substrate concentration at the end of each simulation
fig, ax = plt.subplots()
im = ax.imshow(np.flipud(np.log10(result_substrate[1:,:])), interpolation = 'none', vmin = 3, vmax = 6, cmap = 'coolwarm')
fig.colorbar(im, ax=ax, shrink = 0.8)
ax.set_title('substrate')
plt.show()

