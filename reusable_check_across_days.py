import os
import shutil
import numpy as np
import mat73
from neural_data_object import NeuralData
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.io import loadmat

def prepare_new_folder(neural_data):
    """
    Prepares a new folder in the 'plane0' directory and copies necessary files.
    """
    plane0_dir = neural_data.s2p_dir
    new_dir = os.path.join(plane0_dir, 'new')

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    necessary_files = ['stat.npy', 'iscell.npy', 'F.npy', 'Fneu.npy', 'ops.npy', 'spks.npy']

    """for file_name in necessary_files:
        src_file = os.path.join(plane0_dir, file_name)
        dest_file = os.path.join(new_dir, file_name)

        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
            print(f"Copied {file_name} to {new_dir}.")
        else:
            print(f"{file_name} not found in {plane0_dir}.")"""

def load_cellreg_data(cellreg_file):
    """
    Loads CellReg data from the given file and processes it.
    """
    cellreg_data = mat73.loadmat(cellreg_file)
    cell_registered_struct = cellreg_data['cell_registered_struct']

    x_translations = np.array(cell_registered_struct['alignment_x_translations'])
    y_translations = np.array(cell_registered_struct['alignment_y_translations'])
    rotations = np.array(cell_registered_struct['alignment_rotations'])
    cell_to_index_map = cell_registered_struct['cell_to_index_map'].astype(int)
    centroid_locations_corrected = cell_registered_struct['centroid_locations_corrected']

    cell_to_index_map[cell_to_index_map == 0] = -1
    cell_to_index_map = np.where(cell_to_index_map > 0, cell_to_index_map - 1, cell_to_index_map)

    return {
        'x_translations': x_translations,
        'y_translations': y_translations,
        'rotations': rotations,
        'cell_to_index_map': cell_to_index_map,
        'centroid_locations_corrected': centroid_locations_corrected
    }

def find_unmatched_rois_pre_conversion(stat_files, aligned_rois, distance_threshold=5):
    """
    Find ROIs that are spatially close to detected ROIs but not included in the pre-conversion aligned ROIs.

    Parameters:
        stat_files (list): List of paths to Suite2P `stat.npy` files for each day.
        aligned_rois (np.ndarray): Pre-conversion aligned ROIs matrix from CellReg.
        distance_threshold (int): Distance threshold to identify potential false negatives.

    Returns:
        dict: Close unmatched ROIs for each day.
    """
    unmatched_rois = {}
    for day_idx, stat_file in enumerate(stat_files):
        stat = np.load(stat_file, allow_pickle=True)
        detected_rois = aligned_rois[:, day_idx][~np.isnan(aligned_rois[:, day_idx])].astype(int)
        detected_coords = np.array([stat[roi]['med'] for roi in detected_rois])

        all_coords = np.array([cell['med'] for cell in stat])
        unmatched_indices = []
        for roi_idx, coords in enumerate(all_coords):
            distances = np.linalg.norm(detected_coords - coords, axis=1)
            if len(distances) > 0 and np.min(distances) < distance_threshold and roi_idx not in detected_rois:
                unmatched_indices.append(roi_idx)

        unmatched_rois[f"Day {day_idx + 1}"] = unmatched_indices
    return unmatched_rois

def reverse_map_cellreg_to_suite2p(cellreg_file, s2p_stat_file):
    """
    Generate a mapping from CellReg's .mat footprints back to Suite2P's stat.npy indices.

    Parameters:
        cellreg_file (str): Path to the .mat file used in CellReg.
        s2p_stat_file (str): Path to Suite2P's stat.npy.

    Returns:
        np.ndarray: Mapping of CellReg indices to Suite2P indices.
    """
    cellreg_data = loadmat(cellreg_file)
    footprints = cellreg_data['footprint']  # Shape: (num_rois, Ly, Lx)
    s2p_stat = np.load(s2p_stat_file, allow_pickle=True)

    # Extract centroids
    s2p_centroids = np.array([roi['med'] for roi in s2p_stat])
    cellreg_centroids = []
    for footprint in footprints:
        y, x = np.where(footprint > 0)
        if len(y) > 0 and len(x) > 0:
            cellreg_centroids.append([np.mean(y), np.mean(x)])
        else:
            cellreg_centroids.append([np.nan, np.nan])
    cellreg_centroids = np.array(cellreg_centroids)

    # Map CellReg centroids to Suite2P
    distances = cdist(cellreg_centroids, s2p_centroids)
    return np.argmin(distances, axis=1)

def update_aligned_rois(cellreg_to_suite2p_map, aligned_rois):
    """
    Update the CellReg aligned_rois matrix to use Suite2P indices.

    Parameters:
        cellreg_to_suite2p_map (np.ndarray): Mapping from CellReg to Suite2P indices.
        aligned_rois (np.ndarray): CellReg's aligned ROIs matrix.

    Returns:
        np.ndarray: Updated aligned_rois matrix with Suite2P indices.
    """
    updated_rois = aligned_rois.copy()
    for cell_idx in range(updated_rois.shape[0]):
        for day_idx in range(updated_rois.shape[1]):
            roi_idx = updated_rois[cell_idx, day_idx]
            if roi_idx >= 0:
                updated_rois[cell_idx, day_idx] = cellreg_to_suite2p_map[int(roi_idx)]
    return updated_rois

def process_rois(neural_data_objects, cellreg_data, dates, pixel_distance_threshold=10):
    """
    Processes ROIs based on CellReg data and generates aligned ROIs matrices.

    Parameters:
        neural_data_objects (list): List of NeuralData objects for all days.
        cellreg_data (dict): Processed CellReg data.
        dates (list): List of dates for the experiment.
        pixel_distance_threshold (int): Threshold for nearest neighbor detection.
    Returns:
        dict: Aligned ROIs matrices and converted ROI information.
    """
    n_days = len(dates)
    aligned_rois = np.full((len(cellreg_data['cell_to_index_map']), n_days), np.nan)
    converted_flags = []
    converted_rois = []

    for cell_idx in range(len(cellreg_data['cell_to_index_map'])):
        for date_idx, date in enumerate(dates):
            roi_idx = cellreg_data['cell_to_index_map'][cell_idx, date_idx]

            if roi_idx >= 0:
                aligned_rois[cell_idx, date_idx] = roi_idx
            else:
                transformed_coords_list = []

                for other_day_idx in range(n_days):
                    if other_day_idx == date_idx:
                        continue
                    other_roi_idx = cellreg_data['cell_to_index_map'][cell_idx, other_day_idx]
                    if other_roi_idx >= 0:
                        centroids_for_day = cellreg_data['centroid_locations_corrected'][other_day_idx][0]
                        ref_coords = centroids_for_day[other_roi_idx]
                        transformed_coords_list.append(ref_coords)

                if transformed_coords_list:
                    avg_transformed_coords = np.mean(transformed_coords_list, axis=0)
                    transformed_coords = apply_transformation(
                        avg_transformed_coords,
                        cellreg_data['rotations'][date_idx],
                        cellreg_data['x_translations'][date_idx],
                        cellreg_data['y_translations'][date_idx]
                    )

                    stat_file = os.path.join(neural_data_objects[date_idx].s2p_dir, 'stat.npy')
                    stat = np.load(stat_file, allow_pickle=True)
                    iscell_file = os.path.join(neural_data_objects[date_idx].s2p_dir, 'iscell.npy')
                    iscell = np.load(iscell_file)[:, 0].astype(bool)
                    stat_centroids = np.array([cell['med'] for idx, cell in enumerate(stat) if not iscell[idx]])

                    distances = cdist([transformed_coords], stat_centroids)
                    nearest_idx = np.argmin(distances)

                    if distances[0][nearest_idx] <= pixel_distance_threshold and not iscell[nearest_idx]:
                        aligned_rois[cell_idx, date_idx] = nearest_idx
                        converted_flags.append((nearest_idx, date_idx))
                        converted_rois.append((date_idx, nearest_idx))
                        iscell[nearest_idx] = 1
                        np.save(os.path.join(neural_data_objects[date_idx].s2p_dir, 'new_iscell.npy'),
                                np.column_stack([iscell, np.zeros_like(iscell)]))

    return aligned_rois, converted_flags

def apply_transformation(coords, rotation, x_translation, y_translation):
    """
    Applies affine transformation to the given coordinates.

    Parameters:
        coords (np.ndarray): original coordinates.
        rotation (float): rotation angle, degrees
        x_translation (float)
        y_translation (float)
    Returns:
        np.ndarray: Transformed coordinates.
    """
    cos_theta = np.cos(np.deg2rad(rotation))
    sin_theta = np.sin(np.deg2rad(rotation))
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return np.dot(rotation_matrix, coords) + np.array([x_translation, y_translation])

def save_results(base_dir, mouse, aligned_rois, converted_flags):
    """
    Saves aligned ROI matrices and converted ROI flags.

    Parameters:
        base_dir (str): Base directory for saving results.
        mouse (str): Mouse identifier.
        aligned_rois (np.ndarray): Aligned ROIs matrix.
        converted_flags (list): List of converted ROI flags.
    """
    mouse_dir_matrix = os.path.join(base_dir, mouse)
    if not os.path.exists(mouse_dir_matrix):
        os.makedirs(mouse_dir_matrix)

    np.save(os.path.join(mouse_dir_matrix, 'aligned_rois.npy'), aligned_rois)
    np.save(os.path.join(mouse_dir_matrix, 'converted_flags.npy'), converted_flags)

def find_close_unmatched_rois(stat_files, aligned_rois, distance_threshold=10):
    """
    Find ROIs that are spatially close to detected ROIs but not included in the aligned ROIs.

    Parameters:
        stat_files (list): List of paths to Suite2P `stat.npy` files for each day.
        aligned_rois (np.ndarray): Aligned ROIs matrix.
        distance_threshold (int): Distance threshold to identify potential false negatives.

    Returns:
        dict: Close unmatched ROIs for each day.
    """
    unmatched_rois = {}
    for day_idx, stat_file in enumerate(stat_files):
        stat = np.load(stat_file, allow_pickle=True)
        detected_rois = aligned_rois[:, day_idx][~np.isnan(aligned_rois[:, day_idx])].astype(int)
        detected_coords = np.array([stat[roi]['med'] for roi in detected_rois])

        all_coords = np.array([cell['med'] for cell in stat])
        unmatched_indices = []
        for roi_idx, coords in enumerate(all_coords):
            distances = np.linalg.norm(detected_coords - coords, axis=1)
            if len(distances) > 0 and np.min(distances) < distance_threshold and roi_idx not in detected_rois:
                unmatched_indices.append(roi_idx)

        unmatched_rois[f"Day {day_idx + 1}"] = unmatched_indices
    return unmatched_rois

def evaluate_conversion(aligned_rois, converted_flags, dates):
    """
    Evaluates the efficacy of ROI conversion by calculating:
    1. Number of converted ROIs for each mouse and date.
    2. Number of ROIs detected across all four days before and after conversion.

    Parameters:
        aligned_rois (np.ndarray): Aligned ROIs matrix.
        converted_flags (list): List of converted ROI flags (ROI index, day index).
        dates (list): List of dates.

    Returns:
        dict: Summary statistics for the conversion process.
    """
    n_days = len(dates)
    n_cells = aligned_rois.shape[0]

    # Create a matrix to simulate the state of ROIs before conversion
    pre_conversion_aligned_rois = aligned_rois.copy()
    for roi_idx, day_idx in converted_flags:
        pre_conversion_aligned_rois[:, day_idx] = np.where(pre_conversion_aligned_rois[:, day_idx] == roi_idx, np.nan, pre_conversion_aligned_rois[:, day_idx])

    # Number of ROIs detected across all four days BEFORE conversion
    detected_4days_before = np.sum(np.sum(~np.isnan(pre_conversion_aligned_rois), axis=1) == n_days)

    # Number of ROIs detected across all four days AFTER conversion
    detected_4days_after = np.sum(np.sum(~np.isnan(aligned_rois), axis=1) == n_days)

    # Number of ROIs detected on exactly 3 days BEFORE conversion
    detected_3days_before = np.sum(np.sum(~np.isnan(pre_conversion_aligned_rois), axis=1) == n_days - 1)

    # Number of ROIs detected on exactly 3 days AFTER conversion
    detected_3days_after = np.sum(np.sum(~np.isnan(aligned_rois), axis=1) == n_days - 1)

    # Count the number of conversions per day
    conversions_per_day = {date: 0 for date in dates}
    for _, day_idx in converted_flags:
        conversions_per_day[dates[day_idx]] += 1

    summary = {
        "total_converted_rois": len(converted_flags),
        "detected_4days_before": detected_4days_before,
        "detected_4days_after": detected_4days_after,
        "detected_3days_before": detected_3days_before,
        "detected_3days_after": detected_3days_after,
        "conversions_per_day": conversions_per_day
    }

    return summary

def visualize_and_save_converted_rois(cell_idx, aligned_rois, neural_data_objects, dates, save_path):
    """
    Visualize the FOV with ROI outlines across all 4 days and highlight the converted ROI, then save the image.
    
    Parameters:
        cell_idx (int): Index of the ROI in the aligned_rois matrix.
        aligned_rois (np.ndarray): Matrix of aligned ROIs.
        neural_data_objects (list): List of NeuralData objects for each day.
        dates (list): List of dates corresponding to the experiment.
        save_path (str): Path to save the generated image.
    """
    fig, axes = plt.subplots(2, len(dates), figsize=(20, 10))
    side_pix = 50  # Size of zoomed-in region around ROI

    for date_idx, date in enumerate(dates):
        roi_idx = aligned_rois[cell_idx, date_idx] - 1

        if roi_idx < 0:
            # If ROI is missing, indicate it in both plots
            axes[0, date_idx].set_title(f"Missing ROI for {date}", color='red')
            axes[0, date_idx].axis('off')
            axes[1, date_idx].set_title(f"Missing ROI for {date}", color='red')
            axes[1, date_idx].axis('off')
            continue

        roi_idx = int(roi_idx) # Ensure ROI index is an integer
        data = neural_data_objects[date_idx]
        stat = np.load(os.path.join(data.s2p_dir, 'stat.npy'), allow_pickle=True)
        mean_img = data.ops['meanImg']
        stat_entry = stat[roi_idx]

        # Get ROI coordinates
        ypix, xpix = stat_entry['ypix'], stat_entry['xpix']
        roi_mask = np.zeros(mean_img.shape)
        roi_mask[ypix, xpix] = 1
        y_center, x_center = stat_entry['med']
        min_y, max_y = max(int(y_center - side_pix), 0), min(int(y_center + side_pix), data.ops['Ly'])
        min_x, max_x = max(int(x_center - side_pix), 0), min(int(x_center + side_pix), data.ops['Lx'])

        # Full FOV Plot
        axes[1, date_idx].imshow(mean_img, cmap='gray')
        for roi in stat:  # Outline all ROIs
            ypix_all, xpix_all = roi['ypix'], roi['xpix']
            roi_mask_all = np.zeros(mean_img.shape)
            roi_mask_all[ypix_all, xpix_all] = 1
            axes[1, date_idx].contour(roi_mask_all, colors='blue', linewidths=0.5)

        # Highlight the converted ROI
        axes[1, date_idx].contour(roi_mask, colors='red', linewidths=1.5)  # Highlight converted ROI
        axes[1, date_idx].set_title(f"Full FOV - {date}", color='red' if date_idx == day_idx else 'black')
        axes[1, date_idx].axis('off')

        # Zoomed-in Plot
        cropped_img = mean_img[min_y:max_y, min_x:max_x]
        cropped_mask = roi_mask[min_y:max_y, min_x:max_x]
        axes[0, date_idx].imshow(cropped_img, cmap='gray')
        axes[0, date_idx].contour(cropped_mask, colors='red', linewidths=1.5)  # Highlight converted ROI
        axes[0, date_idx].set_title(f"Zoomed ROI - {date}", color='red' if date_idx == day_idx else 'black')
        axes[0, date_idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':

    # Define parameters
    mouse = '42R'
    dates = ['072224', '072324', '072424', '072524']
    base_path = '/media/maclean/Storage/42R/'
    cellreg_file = '/home/maclean/CellReg/cellreg42R/091324/cellRegistered_20240913_140427.mat'
    base_dir = '/home/maclean/data_proc/'
    stat_files = [os.path.join(base_path, f'{date}/runs/suite2p/plane0/stat.npy') for date in dates]

    # Step 1: Prepare new folders and copy necessary files
    neural_data_objects = [NeuralData(mouse, date, sure=True) for date in dates]
    for neural_data in neural_data_objects:
        prepare_new_folder(neural_data)

    # Step 2: Load CellReg data
    cellreg_data = load_cellreg_data(cellreg_file)

    # Step 3: Analyze unmatched ROIs BEFORE conversion
    unmatched_before_conversion = find_unmatched_rois_pre_conversion(stat_files, cellreg_data['cell_to_index_map'])
    print("\n=== Potential False Negatives BEFORE Conversion ===")
    for day, rois in unmatched_before_conversion.items():
        print(f"{day}: {len(rois)} unmatched ROIs")

    # Step 4: Process ROIs (convert to Suite2P indices and align)
    cellreg_to_suite2p_maps = [
        reverse_map_cellreg_to_suite2p(
            cellreg_file = os.path.join(base_path, 'cellreg', f'{date}_rois_cellreg.mat'),
            s2p_stat_file=stat_files[date_idx]
        )
        for date_idx, date in enumerate(dates)
    ]

    aligned_rois, converted_flags = process_rois(neural_data_objects, cellreg_data, dates)

    # Update aligned_rois to use Suite2P indices
    for date_idx, cellreg_to_suite2p_map in enumerate(cellreg_to_suite2p_maps):
        aligned_rois[:, date_idx] = update_aligned_rois(cellreg_to_suite2p_map, aligned_rois[:, date_idx])

    save_results(base_dir, mouse, aligned_rois, converted_flags)

    # Step 5: Analyze unmatched ROIs AFTER conversion
    false_negatives_after_conversion = find_close_unmatched_rois(stat_files, aligned_rois)
    print("\n=== Potential False Negatives AFTER Conversion ===")
    for day, rois in false_negatives_after_conversion.items():
        print(f"{day}: {len(rois)} unmatched ROIs")

    # Step 6: Evaluate and print conversion efficacy
    conversion_summary = evaluate_conversion(aligned_rois, converted_flags, dates)
    print("\n=== Conversion Summary ===")
    print(f"Total Converted ROIs: {conversion_summary['total_converted_rois']}")
    print(f"ROIs Detected Across Four Days Before Conversion: {conversion_summary['detected_4days_before']}")
    print(f"ROIs Detected Across Four Days After Conversion: {conversion_summary['detected_4days_after']}")
    print("Conversions Per Day:")
    for date, count in conversion_summary['conversions_per_day'].items():
        print(f"  {date}: {count}")

    # Step 7: Visualize and save converted ROIs
    save_dir = os.path.join(base_dir, mouse, 'converted_roi_visualizations')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for cell_idx, (roi_idx, day_idx) in enumerate(converted_flags):
        save_path = os.path.join(save_dir, f"converted_roi_{cell_idx}.png")
        visualize_and_save_converted_rois(cell_idx, aligned_rois, neural_data_objects, dates, save_path)

    print(f"All ROI visualizations saved in: {save_dir}")