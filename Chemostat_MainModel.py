"""Chemostat model for Ehrmann & Mitarai (2025). Generates plots for figure 2, supplementary figure 1 and 2.
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage OB (obligate burst)"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from scipy.signal import argrelmin, argrelmax
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable



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
    p_deg = pars['p_deg']

    #Set condition for phage_T7 (OB) to reduce burst size when S (and bacterial growth rate) drops below starvation threshold
    if ((substrate)/(K_s+substrate)) <= 0.2:
        burst_T7 = burst_T7_low
    else:
        burst_T7 = burst_T7_max

    dS_dt = flow_rate * (substrate_in - substrate) - max_mu * substrate / (K_s + substrate) * bacteria
    dB_dt = max_mu * (substrate / (K_s + substrate)) * bacteria - flow_rate * bacteria - adsorption * bacteria * phage_T4 - adsorption * bacteria * phage_T7

    dP4_dt = burst_T4 * infected_T4 / latency_T4 * ((substrate) / (K_s + substrate)) - adsorption * phage_T4 * (bacteria + infected_T4 + infected_T7) - flow_rate * phage_T4 - p_deg * phage_T4
    #Euqation without secondary adsorption to already infected cells
    #dP4_dt = burst_T4*infected_T4/latency_T4* ((substrate)/(K_s+substrate)) - adsorption*phage_T4*bacteria - flow_rate * phage_T4

    dI4_dt = adsorption * bacteria * phage_T4 - infected_T4 / latency_T4 * ((substrate) / (K_s + substrate)) - flow_rate * infected_T4

    dP7_dt = burst_T7 * infected_T7 / latency_T7 - adsorption * phage_T7 * (bacteria + infected_T7 + infected_T4) - flow_rate * phage_T7 -p_deg * phage_T7
    #dP7_dt = burst_T7*infected_T7/latency_T7 - adsorption*phage_T7*bacteria - flow_rate * phage_T7
    dI7_dt = adsorption * bacteria * phage_T7 - infected_T7 / latency_T7 - flow_rate * infected_T7

    return [dS_dt, dB_dt, dP4_dt, dI4_dt, dP7_dt, dI7_dt]



# Parameters
S_0 = 1e6 #initial condition for substrate, equal to concentration of S in influx (substrate_in)
B_0 = 1e4
MOI = 1 #initial multiplicity of infection, sets initial condition for both phages
P_0 = MOI * B_0
I_0 = 0
pars = {'max_mu': 1, 'K_s': 1e5, 'flow_rate': 0.01, 'substrate_in': 1e6, 'adsorption': 1.5e-8, 'burst_T4': 150}
pars['latency_T4'] = 0.8*(S_0/(pars['K_s']+S_0))
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 150 #Default value of gamma_stat * burst_T7_max, different values for this parameter screened below
pars['p_deg'] = 0 #phage degradation rate (Not applied in the main simulations)

# Initial conditions
y0 = [S_0, B_0, P_0, I_0, P_0, I_0]

# Time points, Default for chemostat is 50000 Tg time units. Depending on the flow rate this translates to different number of system cycles. 
time_span = (0,50000)

#Scan flow rate parameter
flow_rates = np.logspace(-3, -1, 15)
#Scan burst size parameter
burst_T7_stat = np.linspace(22, 150, 14)

# Initialize emtpy arrays to store results
result_trend_T4 = np.empty((len(burst_T7_stat), len(flow_rates)))
result_trend_T7 = np.empty((len(burst_T7_stat), len(flow_rates)))
result_trend_T4_norm = np.empty((len(burst_T7_stat), len(flow_rates)))
result_trend_T7_norm = np.empty((len(burst_T7_stat), len(flow_rates)))
result_substrate = np.empty((len(burst_T7_stat), len(flow_rates)))
result_bacteria = np.empty((len(burst_T7_stat), len(flow_rates)))
result_infectedT4 = np.empty((len(burst_T7_stat), len(flow_rates)))
result_infectedT7 = np.empty((len(burst_T7_stat), len(flow_rates)))
result_T4 = np.empty((len(burst_T7_stat), len(flow_rates)))
result_T7 = np.empty((len(burst_T7_stat), len(flow_rates)))
oscillation_count = np.empty((len(burst_T7_stat), len(flow_rates)))
result_starvation_cycles = np.empty((len(burst_T7_stat), len(flow_rates)))



f = 0 # Loop counter for printed output

for j in burst_T7_stat:
    pars['burst_T7_low'] = j

    print(f'Screening burst T7 low {j}, counter {f}')
    
    n = 0

    for flow_rate in flow_rates:
        pars['flow_rate'] = flow_rate
        #Solve system of differential equations for combination of burst_T7_low and flow_rate
        sol = solve_ivp(system_of_equations, time_span, y0, max_step = 1, args=(pars,))
        
        #Generate example plots for single conditions
        #if j == burst_T7_stat[-2]: #and flow_rate == flow_rates[13]:
        if flow_rate == flow_rates[7]: #or flow_rate == flow_rates[11]: #and j == burst_T7_stat[3]:
            plt.plot(sol.t, sol.y[2], label = 'phage DL', color = '#2B6DC3')
            plt.plot(sol.t, sol.y[4], label = 'phage OB', color = '#AB4C1D')
            plt.plot(sol.t, sol.y[0], label = 'substrate', color = '#21AB61')
            plt.plot(sol.t, sol.y[1], label = 'bacteria', color = '#E0C465')
            #plt.plot(sol.t, sol.y[3], label = 'infected DL', color = '#2B6DC3', linestyle = '--', alpha = 0.7)
            #plt.plot(sol.t, sol.y[5], label = 'infected OB', color = '#AB4C1D', linestyle = '--', alpha = 0.7)
            plt.legend(loc = 'lower right')
            plt.yscale('log')
            plt.ylim([1e0, 1e10])
            plt.title('Burst T7 low: ' + str(j) + ' Flow rate: ' + str(flow_rate))
            plt.axhline(y=2.5e4, color='#800080', linestyle='--')
            plt.xlabel('Time [generations]')
            plt.xlim([0, 500])
            plt.ylabel('Concentration [1/mL]')
            #plt.savefig('Plots/Chemostat_burst_51_flow_rate_016.png', dpi=600, bbox_inches='tight')
            plt.show()

        # Find first local minimum of infected T4 
        infected_T4 = sol.y[3]
        infected_T7 = sol.y[5]
        t_index_20 = np.where(sol.t >= 20)[0][0]
        
        # Extract the bacteria time series
        bacteria = sol.y[1]

        # Find indices of local maxima and minima
        maxima_indices = argrelmax(bacteria)[0]
        minima_indices = argrelmin(bacteria)[0]

        # Combine maxima and minima indices and sort them
        extrema_indices = np.sort(np.concatenate((maxima_indices, minima_indices)))

        # Calculate the number of oscillations
        num_oscillations = len(extrema_indices) // 2  # Each oscillation has one max and one min // creates integer result
        # Could probably calculate this just on the number of maxima or minima
        print(f"Number of oscillations: {num_oscillations}")

        #Calculate number of oscillations in first 1/3 of the time series
        index_t_third = np.where(sol.t >= (sol.t[-1]/3))[0][0]
        extrema_indices_third = extrema_indices[extrema_indices < index_t_third]
        num_oscillations_third = len(extrema_indices_third) // 2        
        
  
        #Fit linear regression to log phage concentration to extract overall slope 
        #log of phage results
        log_phage_T4 = np.log10(sol.y[2])
        log_phage_T7 = np.log10(sol.y[4])
        #FIt linear regression
        if np.isnan(log_phage_T4[500:]).any():
            trend_T4_slope = -0.1
        if np.isinf(log_phage_T4[500:]).any():
            trend_T4_slope = -0.1
        else:
            model_T4 = LinearRegression().fit(sol.t[500:].reshape(-1,1), log_phage_T4[500:])
            trend_T4_slope = model_T4.coef_[0]
        if np.isnan(log_phage_T7[500:]).any():
            trend_T7_slope = -0.1
        if np.isinf(log_phage_T7[500:]).any():
            trend_T7_slope = -0.1
        else:
            model_T7 = LinearRegression().fit(sol.t[500:].reshape(-1,1), log_phage_T7[500:])
            trend_T7_slope = model_T7.coef_[0]

 
       
        #Find the number of oscillations where substrate is below starvation threshold (2.5e4) in first third of time series
        # Values later in the time series are not that representative, once one phage is excluded the system stabilizes
        #Find time index of t/3
        index_t_third = np.where(sol.t >= (sol.t[-1]/3))[0][0]

        below_starvation = np.where(sol.y[0][:index_t_third] < 2.5e4, 1, 0)
        starvation_cycles = np.sum(np.diff(below_starvation) == 1)

        # Store in result arrays
        result_trend_T4[f, n] = trend_T4_slope
        result_trend_T7[f, n] = trend_T7_slope
        result_trend_T4_norm[f, n] = trend_T4_slope / flow_rate
        result_trend_T7_norm[f, n] = trend_T7_slope / flow_rate
        oscillation_count[f, n] = num_oscillations_third #Only counts in the first third of the time series, check definition above
        result_starvation_cycles[f, n] = starvation_cycles

        result_substrate[f, n] = np.median(sol.y[0][-500:])
        result_bacteria[f, n] = np.median(sol.y[1][-500:])  #
        result_T4[f, n] = np.median(sol.y[2][-500:])
        result_T7[f, n] = np.median(sol.y[4][-500:])  #
        result_infectedT4[f, n] = np.median(sol.y[3][-500:])
        result_infectedT7[f, n] = np.median(sol.y[5][-500:])




        n = n+ 1

    
    #Update counter
    f = f + 1

#Remove first row, which were generated when initializing the array



# Difference in area under the curve between OB and DL phage in first infection cycle, this is only calculating difference of infected cells? Not area under curve
result_infected_diff = np.subtract(result_infectedT7, result_infectedT4)

result_trend_diff = np.subtract(result_trend_T7, result_trend_T4)
result_trend_norm_diff = np.subtract(result_trend_T7_norm, result_trend_T4_norm)

# Figures 
plt.imshow(np.flipud(result_trend_diff), extent = [np.log10(flow_rates[0]), np.log10(flow_rates[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'coolwarm', vmin = -0.0005, vmax = 0.0005)
plt.colorbar()
plt.xlabel('log10(flow rate)')
plt.ylabel('burst size T7stat')
plt.title('Trend difference')
plt.show()

plt.imshow(np.flipud(result_trend_norm_diff), extent = [np.log10(flow_rates[0]), np.log10(flow_rates[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'coolwarm', vmin = -0.05, vmax = 0.05)
plt.colorbar()
plt.xlabel('log10(flow rate)')
plt.ylabel('burst size T7stat')
plt.title('Trend difference, normalized by flow rate')
#plt.savefig('Plots/Chemostat_trend_normalized_base.png', dpi=600, bbox_inches='tight')
plt.show()



plt.imshow(np.flipud(result_trend_T4), extent = [np.log10(flow_rates[0]), np.log10(flow_rates[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'coolwarm', vmin = -0.0005, vmax = 0.0005)
plt.colorbar()
plt.xlabel('log10(flow rate)')
plt.ylabel('burst size T7stat')
plt.title('Trend T4')
plt.show()

plt.imshow(np.flipud(result_trend_T7), extent = [np.log10(flow_rates[0]), np.log10(flow_rates[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'coolwarm', vmin = -0.0005, vmax = 0.0005)
plt.colorbar()
plt.xlabel('log10(flow rate)')
plt.ylabel('burst size T7stat')
plt.title('Trend T7')
plt.show()


#Oscillation count might represent only the first third of the time series, check which value is stored in the array above

plt.imshow(np.flipud(oscillation_count), extent = [np.log10(flow_rates[0]), np.log10(flow_rates[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'RdYlBu', vmin = 0, vmax = 500)
plt.colorbar()
plt.xlabel('log10(flow rate)')
plt.ylabel('burst size T7stat')
plt.title('Number of oscillations')
plt.show()

plt.imshow(np.flipud(result_starvation_cycles/(oscillation_count)), extent = [np.log10(flow_rates[0]), np.log10(flow_rates[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'RdYlBu', vmin = 0, vmax = 1)
plt.colorbar()
plt.xlabel('log10(flow rate)')
plt.ylabel('burst size T7stat')
plt.title('Relative number of oscillations where S < 2.5e4 (in the first third of time series)')
#plt.savefig('Plots/Chemostat_StarvationCycles_OBinvasion.svg', format = 'svg', dpi=600, bbox_inches='tight')
plt.show()




#%%
#Plotting slope differences vs. flow rate with color gradient representing the different burst size for phage OB (T7stat)
#colors = plt.cm.Blues(np.linspace(0.3, 1, len(burst_T7_stat)))
colors = plt.cm.rainbow(np.linspace(0, 1, len(burst_T7_stat)))
colors_flow = plt.cm.Reds(np.linspace(0.3, 1, len(flow_rates)))



#%%


fig, ax = plt.subplots(figsize=(8, 5))
sm = ScalarMappable(cmap=plt.cm.rainbow, norm=Normalize(vmin=0.3, vmax=1))
#sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(vmin=0.3, vmax=1))
sm.set_array([])

for i, color in enumerate(colors):
    ax.scatter(flow_rates, result_trend_diff[i], label=f'Trend T7, burst size row {i}', alpha=0.8, color=color)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel('Flow rate')
ax.set_ylabel('delta Slope m')
ax.set_title('Slope diff vs. Flow rate')
ax.set_ylim([-0.00025, 0.00025])
ax.hlines(0, flow_rates[0], flow_rates[-1], color = '#800080', linestyle = '--')
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Burst T7stat')
cbar.set_ticks(np.linspace(0.3, 1, len(burst_T7_stat)))
cbar.set_ticklabels([f'{int(burst)}' for burst in burst_T7_stat])

#plt.savefig('Plots/Chemostat_TrendDiff_Flowrate.png', dpi=600, bbox_inches='tight')

plt.show()



fig, ax = plt.subplots(figsize=(8, 5))
sm = ScalarMappable(cmap=plt.cm.rainbow, norm=Normalize(vmin=0.3, vmax=1))
#sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(vmin=0.3, vmax=1))
sm.set_array([])

for i, color in enumerate(colors):
    ax.scatter(flow_rates, result_trend_norm_diff[i], label=f'Trend T7, burst size row {i}', alpha=0.8, color=color)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel('Flow rate')
ax.set_ylabel('delta Slope m')
ax.set_title('Slope diff (normalized) vs. Flow rate')
ax.set_ylim([-0.04, 0.04])
ax.hlines(0, flow_rates[0], flow_rates[-1], color = '#800080', linestyle = '--')
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Burst T7stat')
cbar.set_ticks(np.linspace(0.3, 1, len(burst_T7_stat)))
cbar.set_ticklabels([f'{int(burst)}' for burst in burst_T7_stat])

#plt.savefig('Plots/Chemostat_TrendDiff_FlowRate_norm.png', dpi=600, bbox_inches='tight')

plt.show()













