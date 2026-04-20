"""Chemostat model with bacteria influx. Generates plots for supplementary figure s3."
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage RB (reduced burst)"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import pandas as pd
import os as os
from datetime import date

today = date.today().strftime('%Y%m%d')
#experiment_name = "RB-invading-DL"


def system_of_equations(t,y, pars):
    """ System of differential equations"""

    bacteria, phage_T4, phage_T7, substrate, infected_T4_1, infected_T4_2, infected_T4_3, infected_T4_4, infected_T4_5, infected_T7_1, infected_T7_2, infected_T7_3, infected_T7_4, infected_T7_5 = y

    #Unpack parameters
    max_mu = pars['max_mu']
    K_s = pars['K_s']
    flow_rate = pars['flow_rate']
    substrate_in = pars['substrate_in']
    adsorption = pars['adsorption'] 
    latency_T4 = pars['latency_T4']
    burst_T4 = pars['burst_T4'] 
    burst_T7_max = pars['burst_T7_max'] 
    burst_T7_low = pars['burst_T7_low'] 
    latency_T7 = pars['latency_T7']
    p_deg = pars['p_deg'] #phage degradation rate
    gamma_delay = pars['gamma_delay']
    S_zero = pars['S_zero']
    bacteria_in = pars['bacteria_in']

    #Set condition for phage_T7 (RB) to reduce burst size when S (and bacterial growth rate) drops below starvation threshold
    if ((substrate)/(K_s+substrate)) <= 0.2:
        burst_T7 = burst_T7_low
    else:
        burst_T7 = burst_T7_max

    latency_delay = latency_T4/(((substrate/(K_s + substrate))/(S_zero/(K_s+S_zero)))**gamma_delay)
    n = 5  #number of infection steps

    dB_dt = flow_rate * (bacteria_in - bacteria) + max_mu * (substrate / (K_s + substrate)) * bacteria - adsorption * bacteria * phage_T4 - adsorption * bacteria * phage_T7


    d_infected_T4_1 = (adsorption*bacteria*phage_T4)  - n/latency_delay * infected_T4_1 - flow_rate * infected_T4_1
    d_infected_T4_2 = n/latency_delay * (infected_T4_1 - infected_T4_2) - flow_rate * infected_T4_2
    d_infected_T4_3 = n/latency_delay * (infected_T4_2 - infected_T4_3) - flow_rate * infected_T4_3
    d_infected_T4_4 = n/latency_delay * (infected_T4_3 - infected_T4_4) - flow_rate * infected_T4_4
    d_infected_T4_5 = n/latency_delay * (infected_T4_4 - infected_T4_5) - flow_rate * infected_T4_5

    d_infected_T7_1 = (adsorption*bacteria*phage_T7)  - n/latency_T7 * infected_T7_1 - flow_rate * infected_T7_1
    d_infected_T7_2 = n/latency_T7 * (infected_T7_1 - infected_T7_2) - flow_rate * infected_T7_2
    d_infected_T7_3 = n/latency_T7 * (infected_T7_2 - infected_T7_3) - flow_rate * infected_T7_3
    d_infected_T7_4 = n/latency_T7 * (infected_T7_3 - infected_T7_4) - flow_rate * infected_T7_4
    d_infected_T7_5 = n/latency_T7 * (infected_T7_4 - infected_T7_5) - flow_rate * infected_T7_5

    sum_infected = infected_T4_1 + infected_T4_2 + infected_T4_3 + infected_T4_4 + infected_T4_5 + infected_T7_1 + infected_T7_2 + infected_T7_3 + infected_T7_4 + infected_T7_5


    dP4_dt = burst_T4*infected_T4_5*n/latency_delay - adsorption*phage_T4*(bacteria + sum_infected) - p_deg * phage_T4 - flow_rate * phage_T4
    dP7_dt = burst_T7 * infected_T7_5 *n/latency_T7 - adsorption * phage_T7 * (bacteria + sum_infected) - flow_rate * phage_T7 -p_deg * phage_T7

    dS_dt = flow_rate * (substrate_in - substrate) - max_mu * substrate / (K_s + substrate) * bacteria

    return [dB_dt, dP4_dt, dP7_dt, dS_dt, d_infected_T4_1, d_infected_T4_2, d_infected_T4_3, d_infected_T4_4, d_infected_T4_5, d_infected_T7_1, d_infected_T7_2, d_infected_T7_3, d_infected_T7_4, d_infected_T7_5]



# Parameters
S_0 = 1e6 #initial condition for substrate, equal to concentration of S in influx (substrate_in)
B_0 = 1e4
MOI = 1 #initial multiplicity of infection, sets initial condition for both phages
P_0 = MOI * B_0
I_0 = 0
pars = {'max_mu': 1, 'K_s': 1e5, 'flow_rate': 0.08, 'substrate_in': 1e6, 'adsorption': 1.5e-8, 'burst_T4': 150}
pars['latency_T4'] = 0.8
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 75 #Example value of gamma_stat * burst_T7_max,
pars['p_deg'] = 0 #phage degradation rate
pars['bacteria_in'] = 1e2
pars['gamma_delay'] = 1.0 #Parameter for scaling of latency with growth rate
pars['S_zero'] = S_0

# Initial conditions
y0 = [B_0, P_0, P_0, S_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0]

# Time points
time_span = (0,50000)

# Simulate
sol = solve_ivp(system_of_equations, time_span, y0, max_step = 0.1, args=(pars,))
        
infected_T4 = sol.y[4] + sol.y[5] + sol.y[6] + sol.y[7] + sol.y[8]
infected_T7 = sol.y[9] + sol.y[10] + sol.y[11] + sol.y[12] + sol.y[13]
#Generate example plot (Figure 2A)
alpha = pars['bacteria_in'] / pars['substrate_in'] #influx ratio

#%%      
plt.plot(sol.t, sol.y[1], label = 'phage DL', color = '#2B6DC3')
plt.plot(sol.t, sol.y[2], label = 'phage RB', color = '#AB4C1D')
plt.plot(sol.t, sol.y[3], label = 'substrate', color = '#21AB61')
plt.plot(sol.t, sol.y[0], label = 'bacteria', color = '#E0C465')
plt.plot(sol.t, infected_T4, label = 'infected DL', color = '#2B6DC3', linestyle = '--', alpha = 0.7)
plt.plot(sol.t, infected_T7, label = 'infected RB', color = '#AB4C1D', linestyle = '--', alpha = 0.7)
plt.legend(loc = 'lower right')
plt.yscale('log')
plt.ylim([1e0, 1e10])
plt.title('Burst T7 low: ' + str(pars['burst_T7_low']) + 'Flow rate: ' + str(pars['flow_rate']))
plt.axhline(y=2.5e4, color='#800080', linestyle=':', alpha = 0.5)
plt.xlabel('Time [generations]')
plt.xlim([000, 3000])
plt.ylabel('Concentration [1/mL]')

plt.savefig(f"Plots/{today}_Chemostat_SingleCondition_Influx_Flow{pars['flow_rate']:.3g}_Delay{pars['gamma_delay']:.3g}_Burst{pars['burst_T7_low']:.3g}_Alpha{alpha:.3g}.svg", format = 'svg', dpi=600, bbox_inches='tight')
plt.show()

# Plot of substrate alone, analyze if substrate drops below starvation threshold
#%%

plt.plot(sol.t, sol.y[3], label = 'substrate', color = '#21AB61')

plt.legend(loc = 'lower right')
plt.yscale('log')
plt.ylim([1e2, 1e7])
plt.title('Burst T7 low: ' + str(pars['burst_T7_low']) + 'Flow rate: ' + str(pars['flow_rate']))
plt.axhline(y=2.5e4, color='#800080', linestyle='--')
plt.xlabel('Time [generations]')
plt.xlim([500, 10500])
plt.ylabel('Concentration [1/mL]')

plt.show()

#Fit linear regression to log phage concentration to extract overall slope 
#log of phage results

log_phage_T4 = np.log10(sol.y[1])
log_phage_T7 = np.log10(sol.y[2])
#FIt linear regression
model_T4 = LinearRegression().fit(sol.t[200:].reshape(-1,1), log_phage_T4[200:])
trend_T4_slope = model_T4.coef_[0]
model_T7 = LinearRegression().fit(sol.t[200:].reshape(-1,1), log_phage_T7[200:])
trend_T7_slope = model_T7.coef_[0]
print('Trend T4 slope:', trend_T4_slope)
print('Trend T7 slope:', trend_T7_slope)   
print('Trend difference:', trend_T7_slope - trend_T4_slope)

#### end for single condition example ####







