import numpy as np
import matplotlib.pyplot as plt
from neural_data_object import NeuralData
import os 

# Define base paths
base_dir = '/media/maclean/Storage'
output_base_dir = '/home/maclean/data_proc/42R/ROIs'

# Create output directory if it doesn't exist
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Define the mouse, dates, and aligned ROIs matrix
mouse = '42R'
dates = ['072224', '072324', '072424', '072524']

# Load the matrices for different alignment scenarios
aligned_rois_3days = np.load('/home/maclean/data_proc/42R/3aligned_rois.npy', allow_pickle=True)  # Cells detected on 3 days with 1 converted
aligned_rois_4days = np.load('/home/maclean/data_proc/42R/4aligned_rois.npy', allow_pickle=True)  # Cells detected on all 4 days

# Load converted flags
converted_flags = np.load('/home/maclean/data_proc/42R/converted_flags.npy', allow_pickle=True)

# Ensure all aligned matrices have the correct dimensions
if aligned_rois_4days.ndim != 2 or aligned_rois_3days.ndim != 2:
    raise ValueError(f"Aligned ROIs matrices must have 2 dimensions.")

# Subfolders for each aligned matrix
roi_subfolders = {
    '3': aligned_rois_3days,
    #'4': aligned_rois_4days,
}

side_pix = 25  # Margin around the ROI for cropping the image

# added by Hal -- just load in neural data objects once to speed things up
neural_data_objects = [NeuralData(mouse, date) for date in dates]

# Loop through each matrix and generate images
for folder_suffix, aligned_rois in roi_subfolders.items():
    output_dir = os.path.join(output_base_dir, folder_suffix)
    os.makedirs(output_dir, exist_ok=True)

    # For each cell in the aligned_rois matrix, generate a combined image across days
    for cell_idx in range(aligned_rois.shape[0]):
        fig, axes = plt.subplots(2, len(dates), figsize=(20, 10))

        for date_idx, date in enumerate(dates):
            roi_idx = aligned_rois[cell_idx, date_idx]
            
            # Skip plotting if the ROI index is NaN (missing ROI)
            if np.isnan(roi_idx):
                axes[0, date_idx].set_title(f"Missing ROI for {date}")
                axes[0, date_idx].axis('off')
                axes[1, date_idx].set_title(f"Missing ROI for {date}")
                axes[1, date_idx].axis('off')
                continue

            # Load the NeuralData object for the specific day
            # modified by Hal to load these in once at the top, just index them here, to run faster
            neural_data = neural_data_objects[date_idx]
            
            # Load the updated iscell.npy, stat.npy, and ops.npy files from the "new" directory
            # Hal -- this was previously loading in the 'iscell.npy' file in the 'new' folder that's never updated
            #new_iscell_file = os.path.join(neural_data.s2p_dir, 'new', 'iscell.npy')
            new_iscell_file = os.path.join(neural_data.s2p_dir, 'new_iscell.npy')
            new_stat_file = os.path.join(neural_data.s2p_dir, 'new', 'stat.npy')
            new_ops_file = os.path.join(neural_data.s2p_dir, 'new', 'ops.npy')

            if not os.path.exists(new_stat_file) or not os.path.exists(new_ops_file) or not os.path.exists(new_iscell_file):
                print(f"Files not found for {date}. Skipping.")
                axes[0, date_idx].set_title(f"Files not found for {date}")
                axes[0, date_idx].axis('off')
                axes[1, date_idx].set_title(f"Files not found for {date}")
                axes[1, date_idx].axis('off')
                continue
            
            # note from Hal -- loading these in every iteration is also probably slowing things down
            # (mainly the ops file which is fairly large)
            # you probably want to just do that once at the top and index them here
            # like I did for the neural data object
            iscell = np.load(new_iscell_file)[:, 0].astype(bool)
            stat = np.load(new_stat_file, allow_pickle=True)
            ops = np.load(new_ops_file, allow_pickle=True).item()

            try:
                # Get the list of cell ROIs from the iscell.npy file (ROIs labeled as cells)
                cell_indices = np.where(iscell)[0]

                # Convert roi_idx (from cellreg) to the correct index in the stat file
                # Ensure roi_idx is valid in the iscell file, which gives the list of cells
                if int(roi_idx) - 1 < 0 or int(roi_idx) - 1 >= len(cell_indices):
                    raise IndexError(f"ROI {roi_idx} is out of bounds for {date}")

                # Get the correct stat file ROI corresponding to the roi_idx from the iscell list
                corrected_roi_idx = cell_indices[int(roi_idx) - 1]

            except IndexError:
                # If the index is out of bounds, fallback to using the direct stat file ROI
                print(f"Falling back to direct stat file index for ROI {roi_idx} on {date}")
                corrected_roi_idx = int(roi_idx) - 1  # Adjust for Suite2p's 0-based indexing

                # Make sure the fallback index is within bounds of the stat file
                if corrected_roi_idx < 0 or corrected_roi_idx >= len(stat):
                    print(f"Fallback ROI index {corrected_roi_idx} is still out of bounds for {date}")
                    continue

            # Load the mean image (mean correlation or max projection image)
            mean_img = ops['meanImg']

            # Get the coordinates for the ROI from the stat file
            ypix = stat[corrected_roi_idx]['ypix']
            xpix = stat[corrected_roi_idx]['xpix']

            # Create a mask for the ROI
            roi_mask = np.zeros(mean_img.shape)
            roi_mask[ypix, xpix] = 1

            # Zoom around the ROI by selecting a region with side_pix margin
            y_center, x_center = stat[corrected_roi_idx]['med']
            min_y = max(int(y_center - side_pix), 0)
            max_y = min(int(y_center + side_pix), ops['Ly'])
            min_x = max(int(x_center - side_pix), 0)
            max_x = min(int(x_center + side_pix), ops['Lx'])

            # Crop the mean image and ROI mask
            cropped_img = mean_img[min_y:max_y, min_x:max_x]
            cropped_mask = roi_mask[min_y:max_y, min_x:max_x]

            #to check if current ROI is in converted_flags matrix
            is_converted = any ((roi_idx == flag[0] and date_idx == flag[1]) for flag in converted_flags)

            # Plot the cropped (zoomed) image in the upper row
            title_color = 'red' if is_converted else 'black'
            axes[0, date_idx].imshow(cropped_img, cmap='gray')
            axes[0, date_idx].contour(cropped_mask, colors='r')
            axes[0, date_idx].set_title(f"ROI {roi_idx} on {date} (Zoomed)", color=title_color)
            axes[0, date_idx].axis('off')

            # Plot the full FOV in the lower row
            axes[1, date_idx].imshow(mean_img, cmap='gray')
            axes[1, date_idx].contour(roi_mask, colors='r')
            axes[1, date_idx].set_title(f"Full FOV {date}", color='black')
            axes[1, date_idx].axis('off')

        # Save the figure for this cell
        output_path = os.path.join(output_dir, f'cell_{cell_idx}_aligned_rois.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Saved ROI comparison for cell {cell_idx} in folder {folder_suffix} across {len(dates)} days.")