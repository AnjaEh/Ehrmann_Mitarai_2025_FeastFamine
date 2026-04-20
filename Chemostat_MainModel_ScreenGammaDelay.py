"""Chemostat model for Ehrmann & Mitarai (2025). Generates plots for figure 2.
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage RB (reduced burst)"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from scipy.signal import argrelmin, argrelmax
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd
import os as os
from datetime import date

def exportData(frame, directory, file):
    path_data = os.path.join(directory, file)
    
    if isinstance(frame, np.ndarray):
        df = pd.DataFrame(frame)
    else:
        df = frame

    df.to_csv(path_data, index = False)
    print('Exported to ' + directory + ' as csv as ' + file)
    return


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
    p_deg = pars['p_deg']
    gamma_delay = pars['gamma_delay']

    #Set condition for phage_T7 (OB) to reduce burst size when S (and bacterial growth rate) drops below starvation threshold
    if ((substrate)/(K_s+substrate)) <= 0.2:
        burst_T7 = burst_T7_low
    else:
        burst_T7 = burst_T7_max

    latency_delay = latency_T4/(((substrate/(K_s + substrate))/(S_0/(K_s+S_0)))**gamma_delay)
    n = 5  #number of infection steps

    
    dB_dt = - flow_rate * bacteria + max_mu * (substrate / (K_s + substrate)) * bacteria - adsorption * bacteria * phage_T4 - adsorption * bacteria * phage_T7

    
    
    #Euqation without secondary adsorption to already infected cells
    #dP4_dt = burst_T4*infected_T4/latency_T4* ((substrate)/(K_s+substrate)) - adsorption*phage_T4*bacteria - flow_rate * phage_T4

    #dI4_dt = adsorption * bacteria * phage_T4 - infected_T4 / latency_T4 * ((substrate) / (K_s + substrate)) - flow_rate * infected_T4
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
    #dP7_dt = burst_T7*infected_T7/latency_T7 - adsorption*phage_T7*bacteria - flow_rate * phage_T7
    

    dS_dt = flow_rate * (substrate_in - substrate) - max_mu * substrate / (K_s + substrate) * bacteria

    return [dB_dt, dP4_dt, dP7_dt, dS_dt, d_infected_T4_1, d_infected_T4_2, d_infected_T4_3, d_infected_T4_4, d_infected_T4_5, d_infected_T7_1, d_infected_T7_2, d_infected_T7_3, d_infected_T7_4, d_infected_T7_5]



# Parameters
S_0 = 1e6 #initial condition for substrate, equal to concentration of S in influx (substrate_in)
B_0 = 1e4
MOI = 1 #initial multiplicity of infection, sets initial condition for both phages
P_0 = MOI * B_0
I_0 = 0
pars = {'max_mu': 1, 'K_s': 1e5, 'flow_rate': 0.01, 'substrate_in': 1e6, 'adsorption': 1.5e-8, 'burst_T4': 150}
#pars['latency_T4'] = 0.8*(S_0/(pars['K_s']+S_0))
pars['latency_T4'] = 0.8
pars['gamma_delay'] = 1.5
pars['latency_T7'] = 0.8
pars['burst_T7_max'] = 150
pars['burst_T7_low'] = 150 #Default value of gamma_stat * burst_T7_max, different values for this parameter screened below
pars['p_deg'] = 0 #phage degradation rate (Not applied in the main simulations)
pars['bacteria_in'] = 0

# Initial conditions
y0 = [B_0, P_0, P_0, S_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0, I_0]

# Time points, Default for chemostat is 50000 Tg time units. Depending on the flow rate this translates to different number of system cycles. 
time_span = (0,50000)

#Scan flow rate parameter
#flow_rates = np.logspace(-3, -1, 15)
gamma_values = np.linspace(0.1, 2, 10)
#Scan burst size parameter
burst_T7_stat = np.linspace(22, 150, 14)

# Initialize emtpy arrays to store results
result_trend_T4 = np.empty((len(burst_T7_stat), len(gamma_values)))
result_trend_T7 = np.empty((len(burst_T7_stat), len(gamma_values)))
result_trend_T4_norm = np.empty((len(burst_T7_stat), len(gamma_values)))
result_trend_T7_norm = np.empty((len(burst_T7_stat), len(gamma_values)))
result_substrate = np.empty((len(burst_T7_stat), len(gamma_values)))
result_bacteria = np.empty((len(burst_T7_stat), len(gamma_values)))
result_infectedT4 = np.empty((len(burst_T7_stat), len(gamma_values)))
result_infectedT7 = np.empty((len(burst_T7_stat), len(gamma_values)))
result_T4 = np.empty((len(burst_T7_stat), len(gamma_values)))
result_T7 = np.empty((len(burst_T7_stat), len(gamma_values)))
oscillation_count = np.empty((len(burst_T7_stat), len(gamma_values)))
result_starvation_cycles = np.empty((len(burst_T7_stat), len(gamma_values)))
result_typical_latency = np.empty((len(burst_T7_stat), len(gamma_values)))
result_median_latency = np.empty((len(burst_T7_stat), len(gamma_values)))



f = 0 # Loop counter for printed output

for j in burst_T7_stat:
    pars['burst_T7_low'] = j

    print(f'Screening burst T7 low {j}, counter {f}')
    
    n = 0

    #for flow_rate in flow_rates:
    for gamma_delay in gamma_values:
        pars['gamma_delay'] = gamma_delay
        #pars['flow_rate'] = flow_rate
        #Solve system of differential equations for combination of burst_T7_low and flow_rate
        sol = solve_ivp(system_of_equations, time_span, y0, max_step = 0.1, args=(pars,))
        
        #Generate example plots for single conditions
        #if j == burst_T7_stat[-2]: #and flow_rate == flow_rates[13]:
        if gamma_delay == gamma_values[7]: #or flow_rate == flow_rates[11]: #and j == burst_T7_stat[3]:
            plt.plot(sol.t, sol.y[1], label = 'phage DL', color = '#2B6DC3')
            plt.plot(sol.t, sol.y[2], label = 'phage OB', color = '#AB4C1D')
            plt.plot(sol.t, sol.y[3], label = 'substrate', color = '#21AB61')
            plt.plot(sol.t, sol.y[0], label = 'bacteria', color = '#E0C465')
            #plt.plot(sol.t, sol.y[3], label = 'infected DL', color = '#2B6DC3', linestyle = '--', alpha = 0.7)
            #plt.plot(sol.t, sol.y[5], label = 'infected OB', color = '#AB4C1D', linestyle = '--', alpha = 0.7)
            plt.legend(loc = 'lower right')
            plt.yscale('log')
            plt.ylim([1e0, 1e10])
            plt.title('Burst T7 low: ' + str(j) + ' Gamma delay: ' + str(gamma_delay))
            plt.axhline(y=2.5e4, color='#800080', linestyle='--')
            plt.xlabel('Time [generations]')
            plt.xlim([0, 3000])
            plt.ylabel('Concentration [1/mL]')
            #plt.savefig('Plots/Chemostat_burst_51_flow_rate_016.png', dpi=600, bbox_inches='tight')
            plt.show()

        # Find first local minimum of infected T4 
        infected_T4 = sol.y[8] + sol.y[7] + sol.y[6] + sol.y[5] + sol.y[4]
        infected_T7 = sol.y[13] + sol.y[12] + sol.y[11] + sol.y[10] + sol.y[9]
        t_index_20 = np.where(sol.t >= 20)[0][0]
        
        # Extract the bacteria time series
        bacteria = sol.y[0]

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
        log_phage_T4 = np.log10(sol.y[1])
        log_phage_T7 = np.log10(sol.y[2])
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

        # Calculate typical latency time for T4
        delta_T_cycle = np.insert(np.diff(sol.t), 0, 0)
        substrate_window_cycle = np.where(infected_T4 > 1, sol.y[3], np.nan) # filter substrate values where infected T4 > 1 (cells with potentially delayed lysis available), otherwise add np.nan
        growth_rate_window_cycle = substrate_window_cycle/(pars['K_s']+substrate_window_cycle) # calculate growth rate at the filtered timepoints
        latency_T4_window_cycle = pars['latency_T4']/((growth_rate_window_cycle/(S_0/(pars['K_s']+S_0)))**pars['gamma_delay'])

        # Only sum weights where latency values are valid (not NaN)
        valid_mask = ~np.isnan(latency_T4_window_cycle)
        if np.any(valid_mask):
            weighted_mean_latency = np.nansum(latency_T4_window_cycle * delta_T_cycle) / np.sum(delta_T_cycle[valid_mask]) / pars['latency_T4']
            median_latency = np.nanmedian(latency_T4_window_cycle) / pars['latency_T4']
        else:
            # If the window is empty, set the latency to nan, no infected cells to evaluate
            weighted_mean_latency = np.nan
            median_latency = np.nan

 
       
        #Find the number of oscillations where substrate is below starvation threshold (2.5e4) in first third of time series
        # Values later in the time series are not that representative, once one phage is excluded the system stabilizes
        #Find time index of t/3
        index_t_third = np.where(sol.t >= (sol.t[-1]/3))[0][0]

        below_starvation = np.where(sol.y[3][:index_t_third] < 2.5e4, 1, 0)
        starvation_cycles = np.sum(np.diff(below_starvation) == 1)

        # Store in result arrays
        result_trend_T4[f, n] = trend_T4_slope
        result_trend_T7[f, n] = trend_T7_slope
        result_trend_T4_norm[f, n] = trend_T4_slope / pars['flow_rate']
        result_trend_T7_norm[f, n] = trend_T7_slope / pars['flow_rate']
        oscillation_count[f, n] = num_oscillations_third #Only counts in the first third of the time series, check definition above
        result_starvation_cycles[f, n] = starvation_cycles

        result_substrate[f, n] = np.median(sol.y[3][-500:])
        result_bacteria[f, n] = np.median(sol.y[0][-500:])  #
        result_T4[f, n] = np.median(sol.y[1][-500:])
        result_T7[f, n] = np.median(sol.y[2][-500:])  #
        result_infectedT4[f, n] = np.median(sol.y[8][-500:])
        result_infectedT7[f, n] = np.median(sol.y[13][-500:])
        result_typical_latency[f, n] = weighted_mean_latency
        result_median_latency[f, n] = median_latency




        n = n+ 1

    
    #Update counter
    f = f + 1

#Remove first row, which were generated when initializing the array



# Difference in area under the curve between OB and DL phage in first infection cycle, this is only calculating difference of infected cells? Not area under curve
result_infected_diff = np.subtract(result_infectedT7, result_infectedT4)

result_trend_diff = np.subtract(result_trend_T7, result_trend_T4)
result_trend_norm_diff = np.subtract(result_trend_T7_norm, result_trend_T4_norm)

#Export results
directory = os.path.join(os.getcwd(), 'Data')
print(directory)

flow_rate = pars['flow_rate']

today = date.today().strftime('%Y%m%d')

exportData(result_trend_diff, directory, f"{today}_Chemostat_GammaScreen_trend_diff_FlowRate_{flow_rate:.3g}.csv")
exportData(result_trend_norm_diff, directory, f"{today}_Chemostat_GammaScreen_trend_norm_diff_FlowRate_{flow_rate:.3g}.csv")
exportData((result_starvation_cycles/(oscillation_count)), directory, f"{today}_Chemostat_GammaScreen_StarvationCycles_FlowRate_{flow_rate:.3g}.csv")

exportData(result_typical_latency, directory, f"{today}_Chemostat_GammaScreen_LatencyT4_FlowRate_{flow_rate:.3g}.csv")
exportData(result_substrate, directory, f"{today}_Chemostat_GammaScreen_SubstrateEnd_FlowRate_{flow_rate:.3g}.csv")
exportData(result_median_latency, directory, f"{today}_Chemostat_GammaScreen_MedianLatencyT4_FlowRate_{flow_rate:.3g}.csv")



# Figures 
plt.imshow(np.flipud(result_trend_diff), extent = [gamma_values[0], gamma_values[-1], (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'coolwarm', vmin = -0.0005, vmax = 0.0005)
plt.colorbar()
plt.xlabel('Gamma delay')
plt.ylabel('burst size T7stat')
plt.title('Trend difference')
plt.show()

plt.imshow(np.flipud(result_trend_norm_diff), extent = [(gamma_values[0]), (gamma_values[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'coolwarm', vmin = -0.05, vmax = 0.05)
plt.colorbar()
plt.xlabel('Gamma delay')
plt.ylabel('burst size T7stat')
plt.title('Trend difference, normalized by flow rate')
plt.savefig(f"Plots/{today}_Chemostat_GammaScreen_trend_norm_diff_FlowRate_{flow_rate:.3g}.svg", format = 'svg', dpi=600, bbox_inches='tight')
plt.show()



plt.imshow(np.flipud(result_trend_T4), extent = [(gamma_values[0]), (gamma_values[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'coolwarm', vmin = -0.0005, vmax = 0.0005)
plt.colorbar()
plt.xlabel('Gamma delay')
plt.ylabel('burst size T7stat')
plt.title('Trend T4')
plt.show()

plt.imshow(np.flipud(result_trend_T7), extent = [(gamma_values[0]), (gamma_values[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'coolwarm', vmin = -0.0005, vmax = 0.0005)
plt.colorbar()
plt.xlabel('Gamma delay')
plt.ylabel('burst size T7stat')
plt.title('Trend T7')
plt.show()


#Oscillation count might represent only the first third of the time series, check which value is stored in the array above

plt.imshow(np.flipud(oscillation_count), extent = [(gamma_values[0]), (gamma_values[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'RdYlBu', vmin = 0, vmax = 500)
plt.colorbar()
plt.xlabel('Gamma delay')
plt.ylabel('burst size T7stat')
plt.title('Number of oscillations')
plt.show()

plt.imshow(np.flipud(result_starvation_cycles/(oscillation_count)), extent = [(gamma_values[0]), (gamma_values[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', cmap = 'RdYlBu', vmin = 0, vmax = 1)
plt.colorbar()
plt.xlabel('Gamma delay')
plt.ylabel('burst size T7stat')
plt.title('Relative number of oscillations where S < 2.5e4 (in the first third of time series)')
plt.savefig(f"Plots/{today}_Chemostat_GammaScreen_StarvationCycles_FlowRate_{flow_rate:.3g}.svg", format = 'svg', dpi=600, bbox_inches='tight')
plt.show()

#%%
plt.imshow(np.flipud(result_typical_latency), extent = [(gamma_values[0]), (gamma_values[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', vmin = 0, vmax = 100, cmap = 'RdYlBu')
plt.colorbar()
plt.xlabel('Gamma delay')
plt.ylabel('burst size T7stat')
plt.title('Typical latency T4')
plt.savefig(f"Plots/{today}_Chemostat_GammaScreen_TypicalLatency_FlowRate_{flow_rate:.3g}.svg", format = 'svg', dpi=600, bbox_inches='tight')
plt.show()

#%%
plt.imshow(np.flipud(result_median_latency), extent = [(gamma_values[0]), (gamma_values[-1]), (burst_T7_stat[0]), burst_T7_stat[-1]], aspect = 'auto', vmin = 0, vmax = 100, cmap = 'RdYlBu')
plt.colorbar()
plt.xlabel('Gamma delay')
plt.ylabel('burst size T7stat')
plt.title('Median latency T4')
plt.savefig(f"Plots/{today}_Chemostat_GammaScreen_MedianLatency_FlowRate_{flow_rate:.3g}.svg", format = 'svg', dpi=600, bbox_inches='tight')
plt.show()


