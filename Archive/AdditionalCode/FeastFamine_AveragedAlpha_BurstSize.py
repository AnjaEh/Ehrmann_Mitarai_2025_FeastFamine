# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:11:07 2024


@author: Anja Ehrmann
Feast Famine Model for Ehrmann & Mitarai (2025). Generates plots for Supplementary Figure 8. Testing contant alpha . 
Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage OB (obligate burst)


"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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


    #Initalizing vector that stores final phage concentration from each cycle for regression and slope analysis
    phage_T4_trend = np.array([initial_conditions[3]])
    phage_T7_trend = np.array([initial_conditions[4]])
    
    
    for cycle in range(0,80):
        
        # Time span for simulation
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
        
              
        #determine time until substrate exhaustion
        if np.any(substrate <= 2.5e4):
            starvation_idx = np.asarray(substrate <= 2.5e4).nonzero()[0][0]
        else:
            starvation_idx = substrate.shape[0] -1
     
        #convert starvation index to actual timepoint
        starvation = t[starvation_idx]

        #append results to longer time frame
        # change t0 to the end of the current cycle - Simulation time runs from 0 to 100 every time
        t = solution.t + t_final[-1]
        t_final = np.append(t_final,t)
        bacteria_final = np.append(bacteria_final, bacteria)
        infected_T4_final = np.append(infected_T4_final, infected_T4)
        infected_T7_final = np.append(infected_T7_final, infected_T7)
        
        phage_T4_final = np.append(phage_T4_final, phage_T4)
        phage_T7_final = np.append(phage_T7_final, phage_T7)

        substrate_final = np.append(substrate_final, substrate)
        starvation_time = np.append(starvation_time, starvation)

        phage_T4_trend = np.append(phage_T4_trend, phage_T4[-1])
        phage_T7_trend = np.append(phage_T7_trend, phage_T7[-1]) 


        
        #change input for second cycle
      
        B_cycle = bacteria[-1]* dilution + fresh_bacteria
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

        
        initial_conditions = [B_cycle, I_cycle_T4, I_cycle_T7, P_cycle_T4, P_cycle_T7, S_cycle]
        

    return [t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final,
            infected_T7_final, substrate_final, starvation_time, phage_T4_trend, phage_T7_trend]


#Parameters and initial conditions ###########################################################

B_0 = 1.0e06
I_0 = 0
R_0 = 0
S_0 = 1.0e06
#S_1 = S_0
#t0 = 0
MOI = 1
P_0 = MOI * B_0

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


pars['p_deg'] = 0 
pars['fresh_bacteria'] = 0.075*pars['S_1']
pars['dilution'] = 0.1

# Calling simulation

initial_conditions = [B_0, I_0, I_0, P_0, P_0, S_0]



# %%
# Iterate over the parameter sets
scan_parameter_2 = 'burst size T7stat'
list_parameter_2 = np.linspace(22, 150, 14)

result_trend_T4 = np.empty_like(list_parameter_2)
result_trend_T7 = np.empty_like(list_parameter_2)
# Run the simulation

for j in range(len(list_parameter_2)):
    pars['burst_T7_low'] = list_parameter_2[j]

    output = simulation(initial_conditions, pars)
    t_final, phage_T4_final, phage_T7_final, bacteria_final, infected_T4_final, infected_T7_final, \
    substrate_final, starvation_time,  phage_T4_trend, phage_T7_trend = output

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

    result_trend_T4[j] = trend_T4_slope
    result_trend_T7[j] = trend_T7_slope

    

starvation_average = np.median(starvation_time[-20:])
print('Median time of starvation:', starvation_average)

#Calculate difference in trends
result_trend_diff = np.subtract(result_trend_T7, result_trend_T4)

#%%
#Plot trend results
plt.figure()
cmap = plt.cm.get_cmap('twilight_shifted')
norm = plt.Normalize(vmin=min(list_parameter_2), vmax=max(list_parameter_2))


# light connecting line + colored markers along the twilight_shifted scale
#plt.plot(list_parameter_2, result_trend_diff, color='lightgray', linewidth=1)
plt.scatter(list_parameter_2, result_trend_diff, c=list_parameter_2, cmap=cmap, norm=norm, edgecolor='k', zorder=3, s = 100)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Burst size T7 in stationary phase')
plt.ylabel('Delta slope')
plt.ylim(-0.6, 0.4)
plt.title(r"Feast-Famine with fixed $\alpha = 0.075$")
#plt.savefig('Plots/FeastFamine_TrendDifference_vs_BurstT7stat_fixedAlpha_75.svg', format = 'svg', dpi=600, bbox_inches='tight')
plt.show()


##random alpha draw

#Picking a random alpha from list
from scipy.stats import gmean
alpha_list = np.logspace(-3, 0.3, num =20)
# pick random alpha from list 100 times

mean_alpha = []
for trail in range(100):

    distribution = []

    for i in range(100):
        distribution.append(np.random.choice(alpha_list))
    mean_alpha.append(gmean(distribution))
#plot histogram of median alphas
plt.figure()
plt.hist(mean_alpha, bins = 10, color = 'gray', edgecolor = 'black')
plt.xlabel('Median alpha')
plt.ylabel('Frequency')
plt.title('Distribution of median alpha from 100 random draws')
plt.show()

print('Mean of gmean alphas:', np.mean(mean_alpha))
print('Minimum of gmean alphas:', np.min(mean_alpha))
print('Maximum of gmean alphas:', np.max(mean_alpha))



# %%
