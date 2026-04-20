"""Load data from stochastic burst size dilution screen and plot heatmap of competitive burst size."""


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
list_parameter_1 = np.linspace(20,150, num =13)


list_parameter_2 = np.linspace(10, 200, num = 10) # Duration of growth cycle
list_parameter_3 = np.logspace(-3, -0.02, num = 12) # Dilution rate

dilution_factor = 1/list_parameter_3


# Import results
directory = os.path.join(os.getcwd(), 'data')
filename = '250306_result_competitive_burst.csv'

results = Load_data(directory, filename).values

output_directory = os.path.join(os.getcwd(), 'Plots')
if not os.path.exists(output_directory):
    print(f'Output directory does not exist')

output_filename = 'heatmap_competitive_burst_size_flipped.svg'
output_path = os.path.join(output_directory, output_filename)

# Plot heatmap
plt.figure(figsize=(8, 6))  # Make the figure wider
plt.imshow(np.flipud(np.fliplr(results[:,3:-1])), cmap= 'coolwarm_r', aspect='auto', vmin = 20, vmax= 150)
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
