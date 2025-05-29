"""SCRIPT roi_shapes pt 2, only loading in 3aligned_rois from before"""

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
import numpy as np
import cv2
import matplotlib.pyplot as plt
from neural_data_object import NeuralData


# Define paths
base_dir = '/home/maclean/data_proc/42R/'
aligned_rois_file = os.path.join(base_dir, 'aligned_rois.npy')
output_dir = os.path.join(base_dir, '3-new')
os.makedirs(output_dir, exist_ok=True)

# Load previously aligned ROIs
aligned_rois = np.load(aligned_rois_file, allow_pickle=True)

# Define parameters
mouse = '42R'
dates = ['072224', '072324', '072424', '072524']
neural_data_objects = [NeuralData(mouse, date, sure=True) for date in dates]
side_pix = 25

# Display and validate each ROI
for cell_idx in range(aligned_rois.shape[0]):
    fig, axes = plt.subplots(2, len(dates), figsize=(20, 10))

    for date_idx, date in enumerate(dates):
        roi_idx = aligned_rois[cell_idx, date_idx] - 1  # Adjust for 0-indexing

        if roi_idx < 0:
            axes[0, date_idx].set_title(f"Missing ROI for {date}")
            axes[0, date_idx].axis('off')
            axes[1, date_idx].set_title(f"Missing ROI for {date}")
            axes[1, date_idx].axis('off')
            continue

        # Load data
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
        
        # Crop and display ROI
        cropped_img = mean_img[min_y:max_y, min_x:max_x]
        cropped_mask = roi_mask[min_y:max_y, min_x:max_x]
        axes[0, date_idx].imshow(cropped_img, cmap='gray')
        axes[0, date_idx].contour(cropped_mask, colors='r')
        axes[0, date_idx].set_title(f"ROI {roi_idx + 1} on {date} (Zoomed)", color='black')
        axes[0, date_idx].axis('off')
        axes[1, date_idx].imshow(mean_img, cmap='gray')
        axes[1, date_idx].contour(roi_mask, colors='r')
        axes[1, date_idx].set_title(f"Full FOV {date}", color='black')
        axes[1, date_idx].axis('off')

    # Convert matplotlib plot into an image for OpenCV display
    plt.tight_layout()
    fig.canvas.draw()
    img_combined = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_combined = img_combined.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    resized_img = cv2.resize(img_combined, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
    window_name = f'Converted ROI {cell_idx}'
    cv2.imshow(window_name, resized_img)
    cv2.resizeWindow(window_name, 600, 400)

    # Keypress validation
    inputting = True
    while inputting:
        keypress = cv2.waitKey(0)
        if keypress > 0:
            keypress = chr(keypress)
        if keypress == 'y':  # Press 'y' to accept
            output_path = os.path.join(output_dir, f'roi_{cell_idx}_validated.png')
            plt.savefig(output_path)
            inputting = False
        elif keypress == 'n':  # Press 'n' to reject
            inputting = False
        elif keypress == 'q':  # Press 'q' to quit the loop
            inputting = False
            quit_flag = True
            break

    cv2.destroyAllWindows()
    if quit_flag:
        break
    plt.close(fig)



#old code old code


#old roi diameter code (idk if i still need or not) 
"""if roi_diameter > avg_diameter:
                y,x = stat[min_dist_idx]['med']
                # Initialize circle_mask with the correct dtype
                circle_mask = np.zeros((data.ops['Ly'], data.ops['Lx']), dtype=np.uint8)

                # Draw an ellipse with the correct parameters
                tat[min_dist_idx]['ypix'], stat[min_dist_idx]['xpix'] = np.where(circle_mask)
            
            
                stat_entry = stat[roi_idx]
                mean_img = data.ops['meanImg']
                # Get the coordinates for the ROI from the stat file
                ypix, xpix = stat_entry['ypix'], stat_entry['xpix']
                # Create a mask for the ROI
                roi_mask = np.zeros(mean_img.shape)
                roi_mask[ypix, xpix] = 1
                # Zoom around the ROI by selecting a region with side_pix margin
                y_center, x_center = stat_entry['med']
                min_y = max(int(y_center - side_pix), 0)
                max_y = min(int(y_center + side_pix), data.ops['Ly'])
                min_x = max(int(x_center - side_pix), 0)
                max_x = min(int(x_center + side_pix), data.ops['Lx'])
                # Crop the mean image and ROI mask
                cropped_img = mean_img[min_y:max_y, min_x:max_x]
                cropped_mask = roi_mask[min_y:max_y, min_x:max_x]

                #to check if current ROI is in converted_flags matrix
                #removing for now -- is_converted = any ((roi_idx == flag[0] and date_idx == flag[1]) for flag in converted_flags)

               

            #convert matplotlib plot into image for opencv
            plt.tight_layout()
            fig.canvas.draw()
            

    # Save updated iscell and stat files after each day
    save_dir = '/home/maclean/data_proc/42R/ROIs/3/3-new'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'new_iscell_{dates[date_idx]}.npy'), np.column_stack([new_iscell, np.zeros_like(new_iscell)]))
    np.save(os.path.join(save_dir, f'new_stat_{dates[date_idx]}.npy'), stat)

# Save final converted flags
np.save(os.path.join(save_dir, 'converted_flags.npy'), np.array(converted_flags))
print(f"Processed and saved ROIs for mouse {mouse}")"""



#label red cells file for reference
"""img_combined = cv2.resize(img_combined, dsize=(img_combined.shape[1] * 3, img_combined.shape[0] * 3),
                                        interpolation=cv2.INTER_LINEAR)
            cv2.imshow(f'roi {idx}', img_combined)
            # now get input to label
            inputting = True
            quit = False
            while inputting:
                keypress = cv2.waitKey(0)
                if keypress > 0:
                    keypress = chr(keypress)
                if keypress == 'r':
                    labels[idx] = True
                    inputting = False
                elif keypress == 'g':
                    inputting = False
                elif keypress == 'p':
                    inputting = False
                    quit = True
            cv2.destroyAllWindows()
            if quit:
                break
            if sure:
                fname = data.proc_dir + 'sure_red_labels_tmp.npy'
            else:
                fname = data.proc_dir + 'red_labels_tmp.npy'
            np.save(fname, np.array(labels))
"""




#the part of displaying (use combo of label red cells and label rois code)
#but first making sure 1) everything UP TO cv2.ellipse, and 2) cv2.ellipse works

