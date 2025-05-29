import numpy as np
import os
from scipy.io import savemat
import mat73
from neural_data_object import NeuralData

# Parameters
use_l = -1
sigma = 0
easy = True
mice = ['42R']
dates = {'42R': ['072224', '072324', '072424', '072524']}
base_name = '/media/maclean/Storage/'

# Load CellReg data
cellreg_file = os.path.expanduser('/home/maclean/CellReg/cellreg42R/081224/cellRegistered_20240812_115023.mat')
if not os.path.exists(cellreg_file):
    raise FileNotFoundError(f"CellReg file not found: {cellreg_file}")
cellreg_data = mat73.loadmat(cellreg_file)

# Extract cell indices from the correct key
cellreg_indices = cellreg_data['cell_registered_struct']['cell_to_index_map'].astype(int) - 1  # Adjusted for zero-indexing

# Filter out only the overlapping cells (cells present in all days)
num_days = len(dates['42R'])
overlapping_cells_mask = np.all(cellreg_indices >= 0, axis=1)
overlapping_cellreg_indices = cellreg_indices[overlapping_cells_mask]

# Initialize lists for normalization indices and noise correlations
norm_indices_all = []
noise_corr_all = []
noise_corr_opposite = []

for mouse in mice:
    # Load up each dataset
    datas = [NeuralData(mouse, date, sure=True) for date in dates[mouse]]
    
    for cell_idx, cell_indices_per_day in enumerate(overlapping_cellreg_indices):
        norm_indices_per_cell = []
        noise_corr_per_cell_all = []
        noise_corr_per_cell_opposite = []

        for day_idx, cell_day_idx in enumerate(cell_indices_per_day):
            # Ensure that the cell index is valid for the day's dataset
            if cell_day_idx < 0 or cell_day_idx >= len(datas[day_idx].iscell):
                print(f"Skipping invalid cell index {cell_day_idx} on day {day_idx} for cell {cell_idx}.")
                continue

            # Get only the cells that are marked as 'iscell'
            iscell_idx = np.where(datas[day_idx].iscell)[0]
            if cell_day_idx >= len(iscell_idx):
                print(f"Skipping cell {cell_idx} on day {day_idx} due to index mismatch.")
                continue

            actual_cell_idx = iscell_idx[cell_day_idx]

            # Calculate normalization index for this cell
            norm_idx = datas[day_idx].normalization_indices()[actual_cell_idx]
            if np.isnan(norm_idx):  # Skip cells with invalid normalization indices
                print(f"Skipping cell {cell_idx} on day {day_idx} due to NaN in normalization index.")
                continue
            norm_indices_per_cell.append(norm_idx)

            # Align red_labels with iscell for the current day
            aligned_red_labels = datas[day_idx].red_labels[datas[day_idx].iscell]

            # Calculate noise correlations with all SST/red labeled cells
            noise_corr_matrix = datas[day_idx].noise_correlations()
            noise_corr_all_cells = noise_corr_matrix[actual_cell_idx, aligned_red_labels]
            noise_corr_per_cell_all.append(np.nanmean(noise_corr_all_cells))

            # Get preferred orientation of the cell
            _, pref_oris, _ = datas[day_idx].normalization_indices(return_prefs=True)
            pref_ori = pref_oris[actual_cell_idx]

            # Identify SST cells tuned to the opposite orientation
            opposite_ori_cells = (pref_oris != pref_ori) & aligned_red_labels
            noise_corr_opposite_cells = noise_corr_matrix[actual_cell_idx, opposite_ori_cells]
            noise_corr_per_cell_opposite.append(np.nanmean(noise_corr_opposite_cells))

        # Store data for this cell if all days' data is valid
        if norm_indices_per_cell and noise_corr_per_cell_all and noise_corr_per_cell_opposite:
            norm_indices_all.append(np.nanmean(norm_indices_per_cell))
            noise_corr_all.append(np.nanmean(noise_corr_per_cell_all))
            noise_corr_opposite.append(np.nanmean(noise_corr_per_cell_opposite))

# Convert lists to numpy arrays for easier manipulation
norm_indices_all = np.array(norm_indices_all)
noise_corr_all = np.array(noise_corr_all)
noise_corr_opposite = np.array(noise_corr_opposite)

# Save results for further analysis (optional)
result_file = '/home/maclean/CellReg/cellreg42R2/noise_corr_analysis_results.npy'
np.save(result_file, {'norm_indices_all': norm_indices_all,
                      'noise_corr_all': noise_corr_all,
                      'noise_corr_opposite': noise_corr_opposite}, allow_pickle=True)

print(f'Saved analysis results to {result_file}')

# Plotting

# Optionally, create plots or further analysis of the results
# For example, you could plot the distribution of normalization indices
# across all cells and sessions, or plot the functional connectivity
# as a function of normalization indices.

# Example: Plot normalization indices distribution
#import matplotlib.pyplot as plt
"""plt.figure()
plt.hist(norm_indices_all_sessions.flatten(), bins='doane')
plt.xlabel('Normalization Index')
plt.ylabel('Count')
plt.title('Normalization Index Distribution Across All Sessions')
plt.show()

# Example: Plot functional connectivity vs normalization indices
plt.figure()
plt.scatter(norm_indices_all_sessions.flatten(), func_connectivity_all_sessions.flatten())
plt.xlabel('Normalization Index')
plt.ylabel('Functional Connectivity with Red-Labeled Cells')
plt.title('Functional Connectivity vs Normalization Index')
plt.show()"""
