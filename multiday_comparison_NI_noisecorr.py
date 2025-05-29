import numpy as np
import os
import mat73
from neural_data_object import NeuralData

# Load the filtered aligned ROIs matrix
save_dir = '/home/maclean/data_proc/figures/ni_noise_corr_analysis'
aligned_rois = np.load(os.path.join(save_dir, 'filtered_aligned_rois_matrix.npy'))

# Define your dates and load cell registration data
mice = ['42R']
dates = ['072224', '072324', '072424', '072524']
base_path = '/media/maclean/Storage/'
cellreg_file = '/home/maclean/CellReg/cellreg42R/081224/cellRegistered_20240812_115023.mat'
cellreg_data = mat73.loadmat(cellreg_file)
cell_to_index_map = cellreg_data['cell_registered_struct']['cell_to_index_map'].astype(int)

# Initialize NeuralData objects for each day
neural_data_objects = [NeuralData('42R', date, sure=True) for date in dates]

# Initialize matrices for normalization indices, noise correlations, tuning strength, and visual responsiveness
n_cells = aligned_rois.shape[0]
n_days = len(dates)
norm_indices_matrix = np.full((n_cells, n_days), np.nan)
noise_corr_matrix_opposite_sst = np.full((n_cells, n_days), np.nan)
tuning_strength_matrix = np.full((n_cells, n_days), np.nan)
vis_resp_matrix = np.full((n_cells, n_days), np.nan)

# Process each cell across the days
for cell_idx in range(n_cells):
    for date_idx, date in enumerate(dates):
        roi_idx = aligned_rois[cell_idx, date_idx]
        
        # If the ROI is valid (not NaN) for that day, process it
        if not np.isnan(roi_idx):
            roi_idx = int(roi_idx)  # Convert to integer for indexing
            data = neural_data_objects[date_idx]
            
            # Get normalization index
            norm_indices = data.normalization_indices()
            if 0 <= roi_idx < len(norm_indices):
                norm_indices_matrix[cell_idx, date_idx] = norm_indices[roi_idx]
            else:
                norm_indices_matrix[cell_idx, date_idx] = np.nan  # Pad with NaN if no NI

            # Get tuning strength
            tuning_strength = data.get_osis()
            if 0 <= roi_idx < len(tuning_strength):
                tuning_strength_matrix[cell_idx, date_idx] = tuning_strength[roi_idx]
            else:
                tuning_strength_matrix[cell_idx, date_idx] = np.nan  # Pad with NaN if no tuning strength
            
            # Get red labels and noise correlations
            red_labels_file = os.path.join(data.proc_dir, 'sure_red_labels.npy')
            if not os.path.exists(red_labels_file):
                continue
            
            red_labels = np.load(red_labels_file)
            if len(red_labels) != len(data.stat):
                continue
            
            # Calculate noise correlations
            noise_corr_matrix = data.noise_correlations(vis_resp_only=True)
            if noise_corr_matrix.ndim == 3:
                noise_corr_matrix = np.mean(noise_corr_matrix, axis=0)

            # Initialize a padded noise correlation matrix with NaN values
            n_rois = len(data.stat)
            padded_noise_corr_matrix = np.full((n_rois, n_rois), np.nan)
            padded_noise_corr_matrix[:noise_corr_matrix.shape[0], :noise_corr_matrix.shape[1]] = noise_corr_matrix
            
            # Preferred orientations
            _, pref_oris, _ = data.normalization_indices(return_prefs=True)
            pref_ori = pref_oris[roi_idx]
            
            if roi_idx < padded_noise_corr_matrix.shape[0]:
                # Identify SST cells tuned to the opposite orientation
                opposite_ori_cells = (pref_oris != pref_ori) & red_labels
                if len(opposite_ori_cells) > 0:
                    noise_corr_opposite_cells = padded_noise_corr_matrix[roi_idx, opposite_ori_cells]
                    noise_corr_matrix_opposite_sst[cell_idx, date_idx] = np.nanmean(noise_corr_opposite_cells)
                else:
                    noise_corr_matrix_opposite_sst[cell_idx, date_idx] = np.nan  # Pad with NaN if no noise corr found
            
            # Check visually responsive neurons
            vis_resp_cells = data.vis_responsive_cells()
            if roi_idx < len(vis_resp_cells):
                vis_resp_matrix[cell_idx, date_idx] = vis_resp_cells[roi_idx]

# Now count how many cells are visually responsive on how many days
vis_resp_count_per_cell = np.sum(vis_resp_matrix == 1, axis=1)

# Calculate how many cells are visually responsive on 1, 2, 3, or all 4 days
num_vis_resp_1_day = np.sum(vis_resp_count_per_cell == 1)
num_vis_resp_2_days = np.sum(vis_resp_count_per_cell == 2)
num_vis_resp_3_days = np.sum(vis_resp_count_per_cell == 3)
num_vis_resp_4_days = np.sum(vis_resp_count_per_cell == 4)

# Print the results
print(f"Number of cells visually responsive on 1 day: {num_vis_resp_1_day}")
print(f"Number of cells visually responsive on 2 days: {num_vis_resp_2_days}")
print(f"Number of cells visually responsive on 3 days: {num_vis_resp_3_days}")
print(f"Number of cells visually responsive on all 4 days: {num_vis_resp_4_days}")

# Print the shape of each matrix
print(f"Shape of norm_indices_matrix: {norm_indices_matrix.shape}")
print(f"Shape of noise_corr_matrix_opposite_sst: {noise_corr_matrix_opposite_sst.shape}")
print(f"Shape of tuning_strength_matrix: {tuning_strength_matrix.shape}")
print(f"Shape of vis_resp_matrix: {vis_resp_matrix.shape}")

# Save the matrices for normalization indices, noise correlations, tuning strength, and visual responsiveness
np.save(os.path.join(save_dir, 'norm_indices_matrix.npy'), norm_indices_matrix)
np.save(os.path.join(save_dir, 'noise_corr_matrix_opposite_sst.npy'), noise_corr_matrix_opposite_sst)
np.save(os.path.join(save_dir, 'tuning_strength_matrix.npy'), tuning_strength_matrix)
np.save(os.path.join(save_dir, 'vis_resp_matrix.npy'), vis_resp_matrix)