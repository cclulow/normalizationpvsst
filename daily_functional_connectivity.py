import numpy as np
import matplotlib.pyplot as plt
import os
from neural_data_object import NeuralData
from scipy.stats import linregress

# Define base path and parameters
mice = ['42R']
dates = {'42R': ['072224', '072324', '072424', '072524']}
base_path = '/media/maclean/Storage/'

# Initialize lists to collect data from all days
all_norm_idx_list = []
all_noise_corr_all_list = []
all_noise_corr_opposite_list = []

# Save plots to a new folder
save_dir = '/home/maclean/data_proc/figures/ni_noise_corr_analysis'
os.makedirs(save_dir, exist_ok=True)

# Part 1: Individual Day Analysis
for mouse in mice:
    for date in dates[mouse]:
        print(f"Processing mouse {mouse}, date {date}...")
        
        # Load data for the current day
        data = NeuralData(mouse, date, sure=True)
        
        # Load the red labels directly from the saved file
        red_labels_file = os.path.join(data.proc_dir, 'sure_red_labels.npy')
        if not os.path.exists(red_labels_file):
            print(f"Red labels file not found for {date}. Skipping this date.")
            continue
        
        red_labels = np.load(red_labels_file)
        
        # Ensure that red_labels length matches the number of detected cells
        if len(red_labels) != len(data.stat):
            print(f"Red labels and stat file mismatch for {date}. Skipping this date.")
            continue
        
        # Get indices of visually responsive cells
        vis_resp = data.vis_responsive_cells()

        # Calculate normalization indices
        norm_indices = data.normalization_indices()

        # Calculate noise correlations for all cells
        noise_corr_matrix = data.noise_correlations()

        # Convert to 2D if the matrix is 3D (e.g., average across the last dimension)
        if noise_corr_matrix.ndim == 3:
            noise_corr_matrix = np.mean(noise_corr_matrix, axis=0)

        # Initialize a padded noise correlation matrix with NaN values
        n_rois = len(data.stat)
        padded_noise_corr_matrix = np.full((n_rois, n_rois), np.nan)

        # Copy the noise correlation matrix into the padded matrix
        padded_noise_corr_matrix[:noise_corr_matrix.shape[0], :noise_corr_matrix.shape[1]] = noise_corr_matrix

        # Initialize lists to store results for plotting
        norm_idx_list = []
        noise_corr_all_list = []
        noise_corr_opposite_list = []

        # Get preferred orientations for all cells
        _, pref_oris, _ = data.normalization_indices(return_prefs=True)

        # Loop over each visually responsive cell
        for cell_idx in np.where(vis_resp)[0]:
            # Ensure cell_idx is within bounds of the noise correlation matrix
            if cell_idx >= noise_corr_matrix.shape[0]:
                print(f"Cell index {cell_idx} is out of bounds for noise_corr_matrix. Skipping this cell.")
                continue
            
            # Get normalization index for this cell
            norm_idx = norm_indices[cell_idx]
            
            # Calculate noise correlations with all SST/red-labeled cells
            noise_corr_all_cells = padded_noise_corr_matrix[cell_idx, red_labels]

            # Get preferred orientation of the cell
            pref_ori = pref_oris[cell_idx]

            # Identify SST cells tuned to the opposite orientation
            opposite_ori_cells = (pref_oris != pref_ori) & red_labels
            noise_corr_opposite_cells = padded_noise_corr_matrix[cell_idx, opposite_ori_cells]

            # Store results for this cell
            norm_idx_list.append(norm_idx)
            noise_corr_all_list.append(np.nanmean(noise_corr_all_cells))
            noise_corr_opposite_list.append(np.nanmean(noise_corr_opposite_cells))


        # Convert lists to arrays for plotting
        norm_idx_array = np.array(norm_idx_list)
        noise_corr_all_array = np.array(noise_corr_all_list)
        noise_corr_opposite_array = np.array(noise_corr_opposite_list)

        # Plot the results
        plt.figure()
        plt.scatter(norm_idx_array, noise_corr_all_array, color='blue', label='All SST Cells')
        plt.scatter(norm_idx_array, noise_corr_opposite_array, color='red', label='Opposite Orientation SST Cells')
        plt.xlabel('Normalization Index')
        plt.ylabel('Noise Correlation')
        plt.title(f'Noise Correlation vs Normalization Index on {date}')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'ni_vs_noise_corr_{date}.png'))
        plt.show()

        # Append current day data to the combined lists
        all_norm_idx_list.extend(norm_idx_array)
        all_noise_corr_all_list.extend(noise_corr_all_array)
        all_noise_corr_opposite_list.extend(noise_corr_opposite_array)

# Convert lists to arrays for combined plotting
all_norm_idx_array = np.array(all_norm_idx_list)
all_noise_corr_all_array = np.array(all_noise_corr_all_list)
all_noise_corr_opposite_array = np.array(all_noise_corr_opposite_list)

# Plot the combined results with a trend line for all days
plt.figure(figsize=(10, 6))
plt.scatter(all_norm_idx_array, all_noise_corr_all_array, color='blue', label='All SST Cells')
plt.scatter(all_norm_idx_array, all_noise_corr_opposite_array, color='red', label='Opposite Orientation SST Cells')

# Calculate and plot trend lines for both sets of correlations
slope_all, intercept_all, _, _, _ = linregress(all_norm_idx_array, all_noise_corr_all_array)
plt.plot(all_norm_idx_array, slope_all * all_norm_idx_array + intercept_all, color='blue', linestyle='--')

slope_opposite, intercept_opposite, _, _, _ = linregress(all_norm_idx_array, all_noise_corr_opposite_array)
plt.plot(all_norm_idx_array, slope_opposite * all_norm_idx_array + intercept_opposite, color='red', linestyle='--')

plt.xlabel('Normalization Index')
plt.ylabel('Noise Correlation')
plt.title('Combined Noise Correlation vs Normalization Index Across All Days')
plt.legend()
plt.savefig(os.path.join(save_dir, 'combined_ni_vs_noise_corr_all_days.png'))
plt.show()

