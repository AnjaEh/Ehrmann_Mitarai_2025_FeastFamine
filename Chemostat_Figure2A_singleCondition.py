"""Chemostat model for paper. Generates plot 2A"
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage OB (obligate burst)"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def system_of_equations(t,y, pars):
    """ System of differential equations"""

    substrate, bacteria, phage_T4, infected_T4, phage_T7, infected_T7 = y

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

    #Set condition for phage_T7 (OB) to reduce burst size when S (and bacterial growth rate) drops below starvation threshold
    if ((substrate)/(K_s+substrate)) <= 0.2:
        burst_T7 = burst_T7_low
    else:
        burst_T7 = burst_T7_max

    dS_dt = flow_rate * (substrate_in - substrate) - max_mu * substrate / (K_s + substrate) * bacteria
    dB_dt = max_mu * (substrate / (K_s + substrate)) * bacteria - flow_rate * bacteria - adsorption * bacteria * phage_T4 - adsorption * bacteria * phage_T7

    dP4_dt = burst_T4 * infected_T4 / latency_T4 * (substrate / (K_s + substrate)) - adsorption * phage_T4 * (bacteria + infected_T4 + infected_T7) - flow_rate * phage_T4 - p_deg * phage_T4
    dI4_dt = adsorption * bacteria * phage_T4 - infected_T4 / latency_T4 * (substrate / (K_s + substrate)) - flow_rate * infected_T4
    
    #PT4 equation without latency adjustment
    #dP4_dt = burst_T4 * infected_T4 / latency_T4 - adsorption * phage_T4 * (bacteria + infected_T4 + infected_T7) - flow_rate * phage_T4
    #dI4_dt = adsorption * bacteria * phage_T4 - infected_T4 / latency_T4 - flow_rate * infected_T4

    dP7_dt = burst_T7 * infected_T7 / latency_T7 - adsorption * phage_T7 * (bacteria + infected_T7 + infected_T4) - flow_rate * phage_T7 - p_deg * phage_T7
    dI7_dt = adsorption * bacteria * phage_T7 - infected_T7 / latency_T7 - flow_rate * infected_T7

    return [dS_dt, dB_dt, dP4_dt, dI4_dt, dP7_dt, dI7_dt]



# Parameters
S_0 = 1e6 #initial condition for substrate, equal to concentration of S in influx (substrate_in)
B_0 = 1e4
MOI = 1 #initial multiplicity of infection, sets initial condition for both phages
P_0 = MOI * B_0
I_0 = 0
pars = {'max_mu': 1, 'K_s': 1e5, 'flow_rate': 0.05, 'substrate_in': 1e6, 'adsorption': 1.5e-8, 'burst_T4': 150}
pars['latency_T4'] = 0.8*(S_0/(pars['K_s']+S_0))
#pars['latency_T4'] = 8
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 75 #Example value of gamma_stat * burst_T7_max,
pars['p_deg'] = 0 #phage degradation rate

# Initial conditions
y0 = [S_0, B_0, P_0, I_0, P_0, I_0]

# Time points
time_span = (0,50000)

# Simulate
sol = solve_ivp(system_of_equations, time_span, y0, max_step = 1, args=(pars,))
        
#Generate example plot (Figure 2A)
        
plt.plot(sol.t, sol.y[2], label = 'phage DL', color = '#2B6DC3')
plt.plot(sol.t, sol.y[4], label = 'phage OB', color = '#AB4C1D')
plt.plot(sol.t, sol.y[0], label = 'substrate', color = '#21AB61')
plt.plot(sol.t, sol.y[1], label = 'bacteria', color = '#E0C465')
plt.plot(sol.t, sol.y[3], label = 'infected DL', color = '#2B6DC3', linestyle = '--', alpha = 0.7)
plt.plot(sol.t, sol.y[5], label = 'infected OB', color = '#AB4C1D', linestyle = '--', alpha = 0.7)
plt.legend(loc = 'lower right')
plt.yscale('log')
plt.ylim([1e0, 1e10])
plt.title('Burst T7 low: ' + str(pars['burst_T7_low']) + 'Flow rate: ' + str(pars['flow_rate']))
plt.axhline(y=2.5e4, color='#800080', linestyle='--')
plt.xlabel('Time [generations]')
plt.xlim([0, 500])
plt.ylabel('Concentration [1/mL]')
#plt.savefig('Plots/Chemostat_burst_75_flow_rate_005.png', dpi=600, bbox_inches='tight')
plt.show()

# Plot of substrate alone, analyze if substrate drops below starvation threshold

plt.plot(sol.t, sol.y[0], label = 'substrate', color = '#21AB61')

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

log_phage_T4 = np.log10(sol.y[2])
log_phage_T7 = np.log10(sol.y[4])
#FIt linear regression
model_T4 = LinearRegression().fit(sol.t[200:].reshape(-1,1), log_phage_T4[200:])
trend_T4_slope = model_T4.coef_[0]
model_T7 = LinearRegression().fit(sol.t[200:].reshape(-1,1), log_phage_T7[200:])
trend_T7_slope = model_T7.coef_[0]
print('Trend T4 slope:', trend_T4_slope)
print('Trend T7 slope:', trend_T7_slope)   
print('Trend difference:', trend_T7_slope - trend_T4_slope)

#### end for single condition example ####







