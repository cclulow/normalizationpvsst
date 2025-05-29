### 102324 trying to rewrite charlotte's cellreg alignment script
import os
import shutil
import numpy as np
import mat73
from neural_data_object import NeuralData
from scipy.spatial.distance import cdist

# Define the base path and mouse/days
mouse = '42R'
dates = ['072224', '072324', '072424', '072524']
base_path = '/media/maclean/Storage/42R/'
neural_data_objects = [NeuralData(mouse, date, sure=True) for date in dates]
distance_threshold = 10 # pixels, defined here I guess

# Load the .mat file for transformations
cellreg_file = '/home/maclean/CellReg/cellreg42R/091324/cellRegistered_20240913_140427.mat'
cellreg_data = mat73.loadmat(cellreg_file)

# Extract necessary data from cell_registered_struct
x_translations = np.array(cellreg_data['cell_registered_struct']['alignment_x_translations'])
y_translations = np.array(cellreg_data['cell_registered_struct']['alignment_y_translations'])
rotations = np.array(cellreg_data['cell_registered_struct']['alignment_rotations'])
cell_to_index_map = cellreg_data['cell_registered_struct']['cell_to_index_map'].astype(int)
index_map_copy = np.copy(cell_to_index_map)
centroid_locations_corrected = cellreg_data['cell_registered_struct']['centroid_locations_corrected']

# Count the number of missing (zero) entries for each cell
num_zeros_per_row = np.sum(cell_to_index_map == 0, axis=1)

# Focus on rows to convert, first
valid_cells = np.where(num_zeros_per_row == 1)[0]  # Rows with precisely 1 missing entry

# keep track of flipped cells
flipped_cells = []
converted_flags = [] # in C's format
for date_idx, data in enumerate(neural_data_objects):
    # load stat file for this date (iscell is already in data object)
    stat = np.load(os.path.join(data.s2p_dir, 'stat.npy'), allow_pickle=True)

    # get mask to index all days but this one
    date_mask = np.array([True] * len(dates))
    date_mask[date_idx] = False

    # cells that don't appear, only on this day
    potential_conversions = np.logical_and(cell_to_index_map[:, date_idx] == 0, num_zeros_per_row == 1)

    # now for each of those cells, we want to try to convert it
    # keeping track in a new iscell
    new_iscell = np.copy(data.iscell)
    for cell in np.where(potential_conversions)[0]:
        idxs = cell_to_index_map[cell][date_mask] # index on each other day (1-indexed)
        coords = np.array([centroid_locations_corrected[dt][0][cell_to_index_map[cell, dt]-1] for dt in np.where(date_mask)[0]])
        avg_coords = np.mean(coords, axis=0) # average coordinates of cell on other days

        # apply transformation to average coords, copied from Charlotte's code
        rotation_angle = rotations[date_idx]  # Get the missing day rotation
        x_translation = x_translations[date_idx]  # Get the missing day translation
        y_translation = y_translations[date_idx]

        cos_theta = np.cos(np.deg2rad(rotation_angle))
        sin_theta = np.sin(np.deg2rad(rotation_angle))
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Apply the transformation to the averaged coordinates
        # apparently the cellreg coords are x,y unlike the y,x of suite2p
        # and the translations also appear to be in microns, not pixels
        transformed_coords = np.dot(rotation_matrix, avg_coords)[::-1] + np.array([y_translation, x_translation]) / 1.5

        # get the centroids for all cells today
        centroids = np.array([cel['med'] for cel in stat])
        dists = cdist([transformed_coords], centroids).squeeze()
        min_noncell_dist = np.argmin(dists[~new_iscell]) # want the closest noncell, not the closest ROI overall
        min_dist_idx = np.where(np.cumsum(~new_iscell) == min_noncell_dist + 1)[0][0] # flip back to full
        min_dist = dists[min_dist_idx]

        # if low enough, flip iscell
        if min_dist < distance_threshold:
            new_iscell[min_dist_idx] = True
            # then need to update the index map later based on what we flipped
            cell_idx = np.cumsum(new_iscell)[min_dist_idx] # off by one? should be matlab index -- is it?
            index_map_copy[:, date_idx][index_map_copy[:, date_idx] >= cell_idx] += 1
            index_map_copy[cell, date_idx] = cell_idx
            converted_flags.append((cell_idx, date_idx)) # wrong but whatevsies
            # and keep track for later plotting
            flipped_cells.append(cell)
        
    # save the new iscell for this day
    np.save(data.s2p_dir + 'new_iscell.npy', np.column_stack([new_iscell, np.zeros_like(new_iscell)]))

# keeping track of which cells are flipped, make something like C's "three_day_rois" and four
four_day_rois = index_map_copy[num_zeros_per_row == 0]
three_day_rois = index_map_copy[flipped_cells]

# and save like C does
# Create a folder for saving aligned ROIs matrices
base_dir = '/home/maclean/data_proc/'
mouse_dir_matrix = os.path.join(base_dir, mouse)

# Create the mouse-specific directory if it doesn't exist
if not os.path.exists(mouse_dir_matrix):
    os.makedirs(mouse_dir_matrix)

# Save the matrices
np.save(os.path.join(mouse_dir_matrix, '4aligned_rois.npy'), four_day_rois)
np.save(os.path.join(mouse_dir_matrix, '3aligned_rois.npy'), three_day_rois)
# Save the converted flags as a .npy file
converted_flags_file = os.path.join(mouse_dir_matrix, 'converted_flags.npy')
np.save(converted_flags_file, converted_flags)
