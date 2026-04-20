# -*- coding: utf-8 -*-
"""Feast Famine model for Ehrmann & Mitarai (2026) with drawing random alpha values. Importing previously simulated data to generate Figure 4D. 
    Nomenclature: phage_T4 = phage DL (delayed lysis), phage_T7 = phage RB (reduced burst)"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import os as os



# %%

# Defining parameter ranges

scan_parameter_1 = 'burst size T7stat'
list_parameter_1 = np.linspace(22,150, num =14)

scan_parameter_2 = 'dilution rate'
list_parameter_2 = np.array([1/100, 1/50, 1/25, 1/10, 1/6, 1/4])


# %%
# Import results
directory = os.path.join(os.getcwd(), 'Data')
print(directory)

#re-import data
results_3D = np.genfromtxt('Data/20260313_Stochastic_DilutionScreen_GammaDelay_1.0_results3D_reshape_complete.csv', delimiter=',', skip_header=1)
results_3D = results_3D.reshape(len(list_parameter_1), len(list_parameter_2), 100)



#%%

plt.figure(figsize=(8, 6))
### Preliminary Figure #### 
# Can adjust to plot only selected data subset
start_idx = 0
end_idx = len(list_parameter_2)
param2_range = np.arange(start_idx, end_idx)
param2_values = list_parameter_2[start_idx:end_idx]

# Normalize colors for the number of list_parameter_1 values
cmap_base = cm.get_cmap('twilight_shifted')
# Skip the first 10% of the colormap
skip = 0.1
cmap = lambda x: cmap_base(skip + (1 - skip) * (x / (len(list_parameter_1) - 1)))
colors = [cmap(i) for i in range(len(list_parameter_1))]

# Wider violins and staggered positions
violin_width = 4 / len(list_parameter_1)  # wider than before
offset = np.linspace(-0.5, 0.5, len(list_parameter_1))  # stagger offsets

for idx1, param1 in enumerate(list_parameter_1):
    print(idx1, param1)
    data = [results_3D[idx1, idx2, :] for idx2 in param2_range]
    # Stagger positions for each parameter1
    positions = param2_range + offset[idx1]
    parts = plt.violinplot(
        data,
        positions=positions,
        showmeans=False,
        showmedians=True,
        widths=violin_width,
    )
    for pc in parts['bodies']:
        pc.set_facecolor(colors[idx1])
        pc.set_edgecolor(colors[idx1])
        pc.set_alpha(0.6)
    # Set median and errorbar colors to match violin body
    if 'cmedians' in parts:
        parts['cmedians'].set_color(colors[idx1])
    if 'cbars' in parts:
        parts['cbars'].set_color(colors[idx1])
    if 'cmins' in parts:
        parts['cmins'].set_color(colors[idx1])
    if 'cmaxes' in parts:
        parts['cmaxes'].set_color(colors[idx1])

plt.xticks(param2_range, [f"{v:.2g}" for v in param2_values], rotation=45)
plt.axhline(0, color='black', linestyle='--', linewidth=1)

plt.xlabel('Parameter 2 (dilution rate)')
plt.ylabel('Delta m (OB - DL trend)')
plt.title('Violin plot of trend values by dilution rate and burst size')
plt.tight_layout()
plt.show()

# %%

#%%
################ Final figure ##########
# Reverse x-axis and convert dilution rate to dilution factor

plt.figure(figsize=(10, 6))

# Only plot from parameter2[4] to parameter2[-1]
start_idx = 0
end_idx = len(list_parameter_2)
param2_range = np.arange(start_idx, end_idx)
param2_values = list_parameter_2[start_idx:end_idx]

# Reverse the order for plotting: left = end_idx, right = start_idx
param2_range_rev = param2_range[::-1]
param2_values_rev = param2_values[::-1]

# Normalize colors for the number of list_parameter_1 values
cmap = cm.get_cmap('twilight_shifted', len(list_parameter_1))
colors = [cmap(i) for i in range(len(list_parameter_1))]

# Wider violins and staggered positions
violin_width = 3 / len(list_parameter_1)
offset = np.linspace(-0.3, 0.3, len(list_parameter_1))
plt.axhline(0.07, color='black', linestyle=':', linewidth=1, alpha = 0.5)
plt.axhline(-0.07, color='black', linestyle=':', linewidth=1, alpha = 0.5)
# Shaded neutral band around zero
plt.axhspan(-0.0025, 0.0025, color='grey', alpha=0.2, zorder=0)

for idx1, param1 in enumerate(list_parameter_1):
    print(idx1, param1)
    data = [results_3D[idx1, idx2, :] for idx2 in param2_range]
    positions = param2_range_rev + offset[idx1]
    parts = plt.violinplot(
        data,
        positions=positions,
        showmeans=False,
        showmedians=True,
        widths=violin_width,
    )
    for pc in parts['bodies']:
        pc.set_facecolor(colors[idx1])
        pc.set_edgecolor(colors[idx1])
        pc.set_alpha(0.6)
    if 'cmedians' in parts:
        parts['cmedians'].set_color(colors[idx1])
    if 'cbars' in parts:
        parts['cbars'].set_color(colors[idx1])
    if 'cmins' in parts:
        parts['cmins'].set_color(colors[idx1])
    if 'cmaxes' in parts:
        parts['cmaxes'].set_color(colors[idx1])

plt.xticks(param2_range, [f"{v:.2g}" for v in param2_values_rev], rotation=45)
plt.axhline(0, color='black', linestyle='--', linewidth=1)

plt.xlabel('Parameter 2 (dilution rate)')
plt.ylabel('Delta m (OB - DL trend)')
plt.title('Violin plot of trend values by dilution rate and burst size')
plt.tight_layout()
plt.savefig('Plots/Stochastic_DilutionFactor_Violins.svg', format = 'svg',  dpi=600, bbox_inches='tight')
plt.show()




# %%
# Plot colorbar only for reference
# Add colorbar to the right of the existing plot
fig, ax = plt.subplots(figsize=(8, 6))
cmap = cm.get_cmap('twilight_shifted', len(list_parameter_1))
norm = plt.Normalize(vmin=0, vmax=len(list_parameter_1)-1)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# Hide the axes for the colorbar-only figure
ax.axis('off')

# Place colorbar to the right
cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation='vertical',
    fraction=0.05,
    pad=0.04,
    ticks=[0, len(list_parameter_1)-1]
)
cbar.ax.set_yticklabels([f"{list_parameter_1[0]:.2g}", f"{list_parameter_1[-1]:.2g}"])
cbar.set_label('Parameter 1 (burst size T7stat)')
#plt.savefig('Plots/Colobar_Violins.png', dpi=600, bbox_inches='tight')
plt.show()
# %%
