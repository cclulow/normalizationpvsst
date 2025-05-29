"""For every single one of the converted rois, those are rois that were previously marked as “not a cell”. In suite2p, they tend to be liiiiike really weirdly shaped. So, want to iterate over all the rois that were converted and look at the shape

For each day with only 3 detected, if there is a converted roi available…
    Figure out the maximum diameter of the other 3 rois (max distance from centroid, how far it stretches out)
    Take the average of those three

The converted roi should only stretch out the average of the other three days/the diameter shouldn’t exceed that average. If it is less, it’s fine
    If it’s more, redraw the ROI (look at suite2p docs for how to put that in python code

Redrawing the ROI…
    Oval shape, average of the other three days in terms of relative shape, diameter, etc etc
    Then, same thing as the label_ROIs code except in a “Y/N” manner (accept as an ROI or do not)
    Popup window, like labeling the red cells – figure is same as label rois code (fov zoomed in and full fov for all 4 days, with the converted ROI in red and also indication if it was redrawn)
    Pressing Y will accept the converted ROI as an ROI, pressing N will discard it and move to the next (won’t convert, won’t redraw, etc)

At the end, 1) converts ROIs, 2) looks at if the ROI needs to be redrawn, 3) redraws in suite2p (would theoretically update the stat file, but first step 4), outputs same thing as label_rois but in a yes/no style (popup window with figure), 5) if yes, save converted roi to stat file, matrix, and figure
Save in data_proc/42R/ROIs/3/3-new (new folder cuz yeah)"""

"""currently REALLY BAD descriptions of what i am doing (note: plsplspls update later)"""

import os
import shutil
import numpy as np
import mat73
from neural_data_object import NeuralData
from scipy.spatial.distance import cdist
from suite2p import io as s2p_io
import cv2 
import matplotlib.pyplot as plt

#paths (save, mouse, cellreg, etc)
mouse = '42R'
dates = ['072224', '072324', '072424', '072524']
base_path = '/media/maclean/Storage/42R/'
neural_data_objects = [NeuralData(mouse, date, sure=True) for date in dates]
distance_threshold = 10 # pixels, defined here I guess
side_pix = 25

#load cellreg (same as before)
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

#prep neural data
converted_flags = [] # in C's format
flipped_cells = []

#calculate ROI diameter -- NEW FUNCTION
def calculate_roi_diameter(stat):
    y,x = stat['med']
    distances = np.sqrt((stat['ypix']-y) ** 2 + (stat['xpix'] -x) ** 2)
    return distances.max()

#alignment/converting ROIs -- OLD FUNCTION, but also with label_rois dispalying plot and accepting/rejecting etc
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
        centroids = np.array([roi['med'] for roi in stat])
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

    three_day_rois = index_map_copy[flipped_cells]
    save_dir = '/home/maclean/data_proc/42R/ROIs/3/3-new'
    os.makedirs(save_dir, exist_ok = True)
    np.save(os.path.join(save_dir, f'new_iscell_{dates[date_idx]}.npy'), np.column_stack([new_iscell, np.zeros_like(new_iscell)]))

 # Save the matrices
np.save(os.path.join(save_dir, '3aligned_rois.npy'), three_day_rois)
np.save(os.path.join(save_dir, 'converted_flags.npy'), np.array(converted_flags))
# Load the 3-day aligned ROIs matrix and converted flags
aligned_rois_3days = np.load('/home/maclean/data_proc/42R/3aligned_rois.npy', allow_pickle=True)
converted_flags = np.load('/home/maclean/data_proc/42R/converted_flags.npy', allow_pickle=True)

# Subfolders for each aligned matrix
roi_subfolders = {
    '3': aligned_rois_3days}

output_base_dir = '/home/maclean/data_proc/42R/3-new'

#ROI Diameter and keypress loop here rahhhhhhh

    # Loop through each matrix and generate images
for folder_suffix, aligned_rois in roi_subfolders.items():
    output_dir = os.path.join(output_base_dir, folder_suffix)
    os.makedirs(output_dir, exist_ok=True)

    # For each cell in the aligned_rois matrix, generate a combined image across days
    for cell_idx in range(aligned_rois.shape[0]):
        fig, axes = plt.subplots(2, len(dates), figsize=(20, 10))

        for date_idx, date in enumerate(dates):
            roi_idx = aligned_rois[cell_idx, date_idx]

            neural_data = neural_data_objects[date_idx]

            new_iscell_file = os.path.join(neural_data.s2p_dir, 'new_iscell.npy')
            new_stat_file = os.path.join(neural_data.s2p_dir, 'new', 'stat.npy')
            new_ops_file = os.path.join(neural_data.s2p_dir, 'new', 'ops.npy')
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
            ypix = stat[corrected_roi_idx]['ypix']
            xpix = stat[corrected_roi_idx]['xpix']
            y_center, x_center = stat[corrected_roi_idx]['med']

            # Redraw if converted and diameter exceeds average
            if any((roi_idx == flag[0] and date_idx == flag[1]) for flag in converted_flags):
                roi_diameter = calculate_roi_diameter(stat[corrected_roi_idx])
                other_days = np.where(~np.isnan(aligned_rois_3days[cell_idx]))[0]
                other_day_diameters = [calculate_roi_diameter(stat[int(aligned_rois_3days[cell_idx, dt]) - 1])
                                    for dt in other_days if dt != date_idx]
                avg_diameter = np.mean(other_day_diameters) if other_day_diameters else 0

                if roi_diameter > avg_diameter:
                    circle_mask = np.zeros((ops['Ly'], ops['Lx']), dtype=np.uint8)
                    cv2.ellipse(
                        circle_mask,                              # image to draw on
                        (int(x), int(y)),                         # Ccnter of the ellipse
                        (int(avg_diameter / 2), int(avg_diameter / 2)),  # axes lengths (radius for width, height)
                        0,                                        # angle of rotation
                        0, 360,                                   # start and end angle (full ellipse)
                        1,                                        # color/intensity (1 for binary mask)
                        -1    )                                   # thickness (-1 for filled ellipse)
                    #lowkey ... do i have cv2? (used in germany but idk if used here) if not then will get
                    #for this line, cv2.ellipse, parameters are: 
                        #center of ellipse (int(x),int(y)) --> axes (avg diameter div 2) --> angle (0) --> start/end angle (0,360)
                        # --> color (1 idk) --> thickness (-1)
                        #rn just going for a filled, unrotated circular mask centered at x,y -- can get more complicated if needed but idk how
                    stat[corrected_roi_idx]['ypix'], stat[corrected_roi_idx]['xpix'] = np.where(circle_mask)

                print(f"/nprocessing cell {cell} day {date_idx}+1")
                print(f"diameters of other days: {other_day_diameters}")
                print(f"avg diameter is {avg_diameter}")
                print(f"needs redrawing!" if roi_diameter>avg_diameter else "nothing needed")

            roi_mask = np.zeros(mean_img.shape)
            roi_mask[ypix, xpix] = 1
            min_y, max_y = max(int(y_center - side_pix), 0), min(int(y_center + side_pix), ops['Ly'])
            min_x, max_x = max(int(x_center - side_pix), 0), min(int(x_center + side_pix), ops['Lx'])
            cropped_img, cropped_mask = mean_img[min_y:max_y, min_x:max_x], roi_mask[min_y:max_x, min_x:max_x]

            # Plot the cropped (zoomed) image in the upper row
            title_color = 'blue' if cell_idx in flipped_cells else 'black' #need to add converted_flags back in
            axes[0, date_idx].imshow(cropped_img, cmap='gray')
            axes[0, date_idx].contour(cropped_mask, colors='r')
            axes[0, date_idx].set_title(f"ROI {roi_idx} on {date} (Zoomed)", color=title_color)
            axes[0, date_idx].axis('off')

            # Plot the full FOV in the lower row
            axes[1, date_idx].imshow(mean_img, cmap='gray')
            axes[1, date_idx].contour(roi_mask, colors='r')
            axes[1, date_idx].set_title(f"Full FOV {date}", color='black')
            axes[1, date_idx].axis('off')

        plt.tight_layout()
        fig.canvas.draw()
        img_combined = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_combined = img_combined.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # old line -- cv2.imshow(f'Converted ROI {cell}', cv2.resize(img_combined, dsize=(img_combined.shape[1] * 3, img_combined.shape[0] * 3), interpolation=cv2.INTER_LINEAR))
        # Scale down the image by 25% for display
        resized_img = cv2.resize(img_combined, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
        # named window/set the window size
        window_name = f'Converted ROI {cell}'
        cv2.imshow(window_name, resized_img)
        cv2.resizeWindow(window_name, 600, 400)  # Set window to 600x400 pixels

        # keypress logic (user validation)
        inputting, quit_loop = True, False
        while inputting:
            keypress = cv2.waitKey(0)
            if keypress > 0:
                keypress = chr(keypress)
            if keypress == 'y':  # press 'y' to accept the converted ROI
                output_path = os.path.join(output_dir, f'accepted_cell_{cell_idx}.png')
                print(f"Accepted and saved ROI {cell_idx}")
                inputting = False
            elif keypress == 'n':  # press 'n' to reject the ROI
                inputting = False
                print(f"rejected ROI {cell_idx}")
            elif keypress == 'q':  # press 'q' to quit the entire process
                inputting, quit_loop = False, True
        cv2.destroyAllWindows()
        #if we want to quit midway through...
        if quit_loop:
            break
        plt.close()