"""Load data from stochastic burst size dilution screen and plot heatmap of competitive burst size. Supplementary Figure S9"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os

# Helper functions
def Load_data(directory, file):
    path_data = os.path.join(directory, file)
  
    data = pd.read_csv(path_data, sep = ';', header = None)
    data = data.fillna(value = 0)
    
    return data


def exportData(frame, directory, file):
    path_data = os.path.join(directory, file)
  
    frame.to_csv(path_data, index = True, sep = ';')
    print('Exported to ' + directory + ' as csv as ' + file)
    return


scan_parameter_1 = 'burst size T7stat'
list_parameter_1 = np.linspace(22,150, num =14)


list_parameter_2 =  np.array([10, 31 ,52, 75, 100, 150, 200, 250]) # Duration of growth cycle
list_parameter_3 = np.array([1/100, 1/50, 1/25, 1/10, 1/6, 1/4, 1/2, 1/1.1]) # Dilution rate

dilution_factor = 1/list_parameter_3


# Import results
directory = os.path.join(os.getcwd(), 'Data')
filename = '260414_result_competitive_burst_allData_TimeDilutionScreen.csv'

results = Load_data(directory, filename).values
results = results[2:, 1:].astype(float) # Remove header and index, convert to float

output_directory = os.path.join(os.getcwd(), 'Plots')
if not os.path.exists(output_directory):
    print(f'Output directory does not exist')

output_filename = 'heatmap_competitive_burst_size_flipped.svg'
output_path = os.path.join(output_directory, output_filename)

#Convert index to parameter values
results_burst_size = list_parameter_1[results.astype(int)]

# Plot heatmap
plt.figure(figsize=(8, 6))  # Make the figure wider
plt.imshow(np.flipud(np.fliplr(results_burst_size)), cmap= 'coolwarm_r', aspect='auto', vmin = 20, vmax= 150)
cbar = plt.colorbar()
cbar.set_label(scan_parameter_1)
cbar.set_ticks(np.linspace(list_parameter_1[0], list_parameter_1[-1], num=5))
plt.xlabel('Dilution rate')
plt.ylabel('Duration of growth cycle')
plt.title('Heatmap of Competitive Burst Size')
#plt.xticks(ticks=np.linspace(list_parameter_3[3], list_parameter_3[-1], num=len(list_parameter_3)-3), labels=[f'{x:.2f}' for x in list_parameter_3[3:]])
#plt.savefig(output_path, format = 'svg', dpi=600, bbox_inches='tight')
plt.show()

# Export the figure


print(f'Figure saved as {output_path}')
