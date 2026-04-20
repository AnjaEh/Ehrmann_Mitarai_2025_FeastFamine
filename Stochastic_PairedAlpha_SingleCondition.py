# -*- coding: utf-8 -*-

"""Feast Famine Model with paired alpha (Single condition) for Ehrmann & Mitarai (2026). Generates plots supplementary figures S7
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage RB (reduced burst)"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import pandas as pd
import os as os
from datetime import date



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
    S_zero = pars['S_zero']
    

    #set condition for infection in stationary phase
    if ((substrate)/(K_s+substrate)) <= 0.2:
        burst_T7 = burst_T7_low
    else:
        burst_T7 = burst_T7_max
    
    latency_delay = latency_T4/(((substrate/(K_s + substrate))/(S_zero/(K_s+S_zero)))**gamma_delay)
    n = 5

    # Define the system of differential equations

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
        if np.any(substrate <= 2.5e4):
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
S_0 = 1.0e06


MOI = 1
P_0 = MOI * B_0
initial_conditions = [B_0, P_0, P_0,S_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0]

#One substrate unit is what is needed for one bacterium to devide once
#The substrate amount therefore defines the capacity of the culture

pars = {}

pars['S_1'] = S_0
pars['K_s']  = 1e5
pars['max_mu']  = 1
pars['adsorption'] = 1.5e-8

pars['latency_T4'] = 0.8
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 0.4*150
pars['burst_T4'] = 150


pars['deg_bacteria'] = 0 
pars['p_deg'] = 0 
pars['fresh_bacteria'] = 1.0e6
pars['dilution'] = 0.1

pars['alpha1'] = 0.01
pars['alpha2'] = 0.01

pars['gamma_delay'] = 1.0
pars['S_zero'] = S_0


#calling simulation and extracting final result

output = simulation(initial_conditions, pars)

t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, infected_T7_final, substrate_final, starvation_time, phage_T4_trend, phage_T7_trend = output


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

# Values for fit of trend line
predict_T4 = model_T4.predict(X)
predict_T7 = model_T7.predict(X)


print(f"Trend T4 slope: {trend_T4_slope:.8f}")
print(f"Trend T7 slope: {trend_T7_slope:.8f}")
print(f"Trend diff T7 - T4: {trend_T7_slope - trend_T4_slope:.8f}")

directory = os.path.join(os.getcwd(), 'Data')
print(directory)
#exportData(ratio_phage, directory, '20250114_constant_alpha_result_all.csv')
gamma_delay = pars['gamma_delay']
#alpha = 0.001
today = date.today().strftime('%Y%m%d')
#experiment_name = "checkCorrect"

# Plotting the results
# %%
# Plot the results
plt.figure()

#plt.plot(t_final[1:], substrate_final[1:], label = 'substrate', color = '#21AB61')

plt.plot(t_final[1:], phage_T4_final[1:], label = 'phage DL', color = '#2B6DC3')
plt.plot(t_final[1:], phage_T7_final[1:], label = 'phage RB', color = '#AB4C1D')

plt.yscale('log')
plt.xlabel('Time [$T_g$]')
plt.ylabel('Phage concentration [1/ml]')
plt.legend()
plt.ylim(bottom = 1e0, top = 1e9)

#plt.savefig(f"Plots/{today}_FeastFamine_PairedAlpha_reduced_BurstRed{pars['burst_T7_low']}_gammaDelay_{gamma_delay:.3g}_Alpha1_{pars['alpha1']}_Alpha2_{pars['alpha2']}.svg", format = 'svg', dpi=600, bbox_inches='tight')
plt.show()

#%%
# Plotting for definition of trend analysis / definition of slope
plt.figure()


plt.plot(t_final[1:], phage_T4_final[1:], label = 'phage DL', color = '#2B6DC3')
plt.plot(t_final[1:], phage_T7_final[1:], label = 'phage RB', color = '#AB4C1D')
plt.plot(X*100, 10**predict_T4, label = 'DL fit', color = "#2D058A", linestyle = '--')
plt.plot(X*100, 10**predict_T7, label = 'RB fit', color = "#490220", linestyle = '--')

plt.yscale('log')
plt.xlabel('Time [$T_g$]')
plt.ylabel('Phage concentration [1/ml]')
plt.legend()
plt.ylim(bottom = 1e0, top = 1e9)

plt.savefig(f"Plots/{today}_SupplementModelFit_BurstRed{pars['burst_T7_low']}_gammaDelay_{gamma_delay:.3g}_Alpha1_{pars['alpha1']}_Alpha2_{pars['alpha2']}.svg", format = 'svg', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Plotting for definition of trend analysis / definition of slope
plt.figure()


plt.scatter(X, log_phage_T4_trend, label = 'phage DL', color = '#2B6DC3')
plt.scatter(X, log_phage_T7_trend, label = 'phage RB', color = '#AB4C1D')
plt.plot(X, predict_T4, label = 'DL fit', color = "#2D058A", linestyle = '--')
plt.plot(X, predict_T7, label = 'RB fit', color = "#490220", linestyle = '--')

#plt.yscale('log')
plt.xlabel('Time [cycles]')
plt.ylabel('Log(Phage concentration [1/ml])')
plt.legend()
#plt.ylim(bottom = 1e0, top = 1e9)


plt.savefig(f"Plots/{today}_SupplementModelFit_log_BurstRed{pars['burst_T7_low']}_Alpha1_{pars['alpha1']}_Alpha2_{pars['alpha2']}.svg", format = 'svg', dpi=300, bbox_inches='tight')
plt.show()


# %%
# Plot the results
plt.figure()
plt.plot(t_final[1:], phage_T4_final[1:], label = 'phage DL', color = '#2B6DC3')
plt.plot(t_final[1:], phage_T7_final[1:], label = 'phage RB', color = '#AB4C1D')
plt.plot(t_final[1:], substrate_final[1:], label = 'substrate', color = '#21AB61')
plt.plot(t_final[1:], bacteria_final[1:], label = 'bacteria', color = '#E0C465')
plt.plot(t_final[1:], infected_T4_final[1:], label = 'infected DL', color = '#2B6DC3', linestyle = '--', alpha = 0.7)
plt.plot(t_final[1:], infected_T7_final[1:], label = 'infected RB', color = '#AB4C1D', linestyle = '--', alpha = 0.7)

plt.yscale('log')
plt.xlabel('Time [$T_g$]')
plt.ylabel('Phage concentration [1/ml]')
plt.legend()
plt.ylim(bottom = 1e0, top = 1e9)
plt.xlim(left = 590, right = 680)
#plt.savefig(f"Plots/{today}_FeastFamine_PairedAlpha_full_BurstRed{pars['burst_T7_low']}_gammaDelay_{gamma_delay:.3g}_Alpha1_{pars['alpha1']}_Alpha2_{pars['alpha2']}.svg", format = 'svg', dpi=600, bbox_inches='tight')
plt.show()


# %%
