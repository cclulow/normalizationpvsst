
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

# Create a new folder for each day in the "plane0" directory and copy necessary files
def prepare_new_folder(neural_data):
    plane0_dir = neural_data.s2p_dir  # Directly point to the correct 'plane0' folder
    new_dir = os.path.join(plane0_dir, 'new')

    # Create the "new" subfolder if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # List of necessary files to copy
    necessary_files = ['stat.npy', 'iscell.npy', 'F.npy', 'Fneu.npy', 'ops.npy', 'spks.npy']

    # Copy necessary files to the new subfolder
    for file_name in necessary_files:
        src_file = os.path.join(plane0_dir, file_name)
        dest_file = os.path.join(new_dir, file_name)

        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
            print(f"Copied {file_name} to {new_dir}.")
        else:
            print(f"{file_name} not found in {plane0_dir}.")

# Prepare folders and copy files for each day
for neural_data in neural_data_objects:
    prepare_new_folder(neural_data)

# Load the .mat file for transformations
cellreg_file = '/home/maclean/CellReg/cellreg42R/091324/cellRegistered_20240913_140427.mat'
cellreg_data = mat73.loadmat(cellreg_file)

# Extract necessary data from cell_registered_struct
x_translations = np.array(cellreg_data['cell_registered_struct']['alignment_x_translations'])
y_translations = np.array(cellreg_data['cell_registered_struct']['alignment_y_translations'])
rotations = np.array(cellreg_data['cell_registered_struct']['alignment_rotations'])
cell_to_index_map = cellreg_data['cell_registered_struct']['cell_to_index_map'].astype(int)
centroid_locations_corrected = cellreg_data['cell_registered_struct']['centroid_locations_corrected']

# Set 0's to -1 in cell_to_index_map to handle missing cells (indicating NaN in final matrix)
cell_to_index_map[cell_to_index_map == 0] = -1
cell_to_index_map = np.where(cell_to_index_map > 0, cell_to_index_map - 1, cell_to_index_map)

# Count the number of missing (zero) entries for each cell
num_zeros_per_row = np.sum(cell_to_index_map == -1, axis=1)

# Only keep rows with 1 or fewer missing ROIs
valid_cells = np.where(num_zeros_per_row <= 1)[0]  # Rows with 1 or no missing entries

# Initialize matrix with NaN for missing values
aligned_rois = np.full((len(valid_cells), len(dates)), np.nan)
converted_flags = []

# Reference session index (e.g., assuming day 1 as reference)
ref_day_idx = 1
converted_rois = []
pixel_distance_threshold = 10  # Set a threshold distance
n_days = len(dates)

# Process each cell that is detected in 3 or more days
for cell_counter, cell_idx in enumerate(valid_cells):
    for date_idx, date in enumerate(dates):
        roi_idx = cell_to_index_map[cell_idx, date_idx]
        
        if roi_idx >= 0:  # If ROI is present on this day
            aligned_rois[cell_counter, date_idx] = roi_idx
        else:
            # Cell is missing on this day, average the coordinates from the available days
            transformed_coords_list = []

            for other_day_idx in range(n_days):
                if other_day_idx == date_idx:
                    continue
                other_roi_idx = cell_to_index_map[cell_idx, other_day_idx]
                if other_roi_idx >= 0:
                    centroids_for_day = centroid_locations_corrected[other_day_idx][0]  # Step-by-step indexing
                    print(f"other_roi_idx: {other_roi_idx}, centroids_for_day size: {len(centroids_for_day)}")
                    # Add bounds check to avoid out-of-bounds error
                    if other_roi_idx + 1 < len(centroids_for_day):  # Check if index is within bounds
                        ref_coords = centroids_for_day[other_roi_idx]  # Adjust for Matlab's 1-indexing
                        transformed_coords_list.append(ref_coords)
                    else:
                        print(f"Skipping out-of-bounds index: {other_roi_idx + 1}")    
                        

            # Step 1: Average the coordinates from all available days
            if transformed_coords_list:
                avg_transformed_coords = np.mean(transformed_coords_list, axis=0)

                # Step 2: Apply affine transformation for the missing day
                rotation_angle = rotations[date_idx]  # Get the missing day rotation
                x_translation = x_translations[date_idx]  # Get the missing day translation
                y_translation = y_translations[date_idx]

                cos_theta = np.cos(np.deg2rad(rotation_angle))
                sin_theta = np.sin(np.deg2rad(rotation_angle))
                rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

                # Apply the transformation to the averaged coordinates
                transformed_coords = np.dot(rotation_matrix, avg_transformed_coords) + np.array([x_translation, y_translation])

                # Step 3: Use the transformed coordinates to search for nearest non-cell ROI in stat.npy
                stat_file = os.path.join(neural_data_objects[date_idx].s2p_dir, 'stat.npy')
                stat = np.load(stat_file, allow_pickle=True)

                iscell_file = os.path.join(neural_data_objects[date_idx].s2p_dir, 'iscell.npy')
                iscell = np.load(iscell_file)[:, 0].astype(bool)
                stat_centroids = np.array([cell['med'] for idx, cell in enumerate(stat) if not iscell[idx]])

                # Find the nearest ROI to the transformed coordinates
                distances = cdist([transformed_coords], stat_centroids)
                nearest_idx = np.argmin(distances)

                # Only proceed if the nearest distance is within the pixel threshold
                min_distance = distances[0][nearest_idx]
                if min_distance <= pixel_distance_threshold:
                    # Check if the nearest ROI is not marked as a cell
                    print(f"Avg transformed coords: {transformed_coords}, Stat centroids: {stat_centroids}")
                    print(f"Checking ROI {nearest_idx} on day {dates[date_idx]} with distance {min_distance}")
                    print(f"Is cell status before conversion: {iscell[nearest_idx]}")


                    if not iscell[nearest_idx]:  # If it's not labeled as a cell, convert it
                        # Update the aligned_rois with the new ROI index
                        aligned_rois[cell_counter, date_idx] = nearest_idx
                        converted_flags.append((nearest_idx, date_idx))  # Mark this ROI as converted

                        # Track the converted ROI and day
                        converted_rois.append((date_idx, nearest_idx))

                        # Update iscell.npy to mark this ROI as a cell
                        iscell[nearest_idx] = 1  # Convert to cell
                        new_iscell_file = os.path.join(neural_data_objects[date_idx].s2p_dir, 'new_iscell.npy')
                        np.save(new_iscell_file, np.column_stack([iscell, np.zeros_like(iscell)]))

# Post-processing to save aligned ROIs based on converted ROIs
converted_counts = np.sum(converted_flags, axis=1)  # Count the number of converted ROIs per row

rows_with_converted_rois = []

for cell_idx in range(aligned_rois.shape[0]): 
    for day_idx in range(len(dates)):
        roi_number = aligned_rois[cell_idx, day_idx]
        if not np.isnan(roi_number):
            if (roi_number, day_idx) in converted_flags:
                rows_with_converted_rois.append(cell_idx)
                break 

rows_with_converted_rois = np.array(rows_with_converted_rois)

# Cells detected on 4 days, without any converted ROIs -- NEED TO CHANGE THIS SO MATRIX ACTUALLY REFLECTS
cells_4days = np.where((np.sum(~np.isnan(aligned_rois), axis=1) == 4))[0]

# Cells detected on 3 days initially (including converted ROIs), now detected on all 4 days
cells_3days = rows_with_converted_rois

# All cells detected across all days, including converted ROIs
all_cells = np.where(np.sum(~np.isnan(aligned_rois), axis=1) >=3)[0]

# Filter out the cells based on conversion
four_day_rois = aligned_rois[cells_4days]
three_day_rois = aligned_rois[cells_3days]
all_rois = aligned_rois[all_cells]

# Create a folder for saving aligned ROIs matrices
base_dir = '/home/maclean/data_proc/'
mouse_dir_matrix = os.path.join(base_dir, mouse)

# Create the mouse-specific directory if it doesn't exist
if not os.path.exists(mouse_dir_matrix):
    os.makedirs(mouse_dir_matrix)

# Save the matrices
np.save(os.path.join(mouse_dir_matrix, '4aligned_rois.npy'), four_day_rois)
np.save(os.path.join(mouse_dir_matrix, '3aligned_rois.npy'), three_day_rois)
np.save(os.path.join(mouse_dir_matrix, 'all_aligned_rois.npy'), all_rois)
# Save the converted flags as a .npy file
converted_flags_file = os.path.join(mouse_dir_matrix, 'converted_flags.npy')
np.save(converted_flags_file, converted_flags)

# Print shapes of each matrix
print(f"4aligned_rois shape: {four_day_rois.shape}")
print(f"3aligned_rois shape: {three_day_rois.shape}")
print(f"all_aligned_rois shape: {all_rois.shape}")
# Print the first 10 rows of each matrix
print("First 10 rows of 4aligned_rois matrix:")
print(four_day_rois[:10])

print("First 10 rows of 3aligned_rois matrix:")
print(three_day_rois[:10])

print("First 10 rows of all_aligned_rois matrix:")
print(all_rois[:10])