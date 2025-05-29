### 050924 semi-manual red cell labeling inspired by adesnik method
### works with an already-existing neural data object
import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage import gaussian_filter
from skimage.io import imsave
from neural_data_object import NeuralData
from red_reg_functions import load_red_series, plot_contours
from suite2p.registration.rigid import phasecorr, phasecorr_reference, shift_frame

# maybe make these cmdline args
mousedates = [('42R', '072324')]
base_name = '/media/maclean/Storage/'

sure = True

# params for labeling
percentile = 70 # what bottom percentile to ignore... 50 is a little on safe/slow side
side_pix = 25 # how much of the ROI to look at

def convert_8bit(image):
    min_val = image.min()
    max_val = (image - min_val).max()
    return (255 * ((image - min_val) / max_val)).astype(np.uint8)

for mouse, date in mousedates:
    # load in data and red channel data
    print('loading in data...')
    base_name = '/media/maclean/Storage/'    
    data = NeuralData(mouse, date, base_name=base_name)
    stat = data.stat
    #ops = data.ops # for convenience
    # hires stuff could be changed, maybe. not sure this is fully compatible with february data
    # have not experimented with s>0 (bleedthru correction) but it could help...
    # should probably have this save an intermediate file... or after normalization
    # assuming these are returned as 1024x1024 mean images
    red, green = load_red_series(data.base_dir, hires=True, s=0, green=True, keep_hires=True,
                                    hires_only=True, unweave_frames=True)

    # process red channel scans
    # first gaussian filter to subtract out rough fluctuations
    print('normalizing images...')
    filt = gaussian_filter(red, sigma=100)
    norm = red - filt
    # then normalize to 0-255 range so cv2's local contrast enhancement will run
    norm = norm - norm.min()
    norm = norm / norm.max()
    norm = (norm * 255).astype(np.uint8)
    green = green - green.min()
    green = green / green.max()
    green = (green * 255).astype(np.uint8)
    # and run local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=5)
    red_norm = clahe.apply(norm).astype(float)
    green_norm = clahe.apply(green).astype(float)

    # get things to right size
    red = red.astype(float) # TODO was norm
    ds_red = cv2.resize(red,dsize=(512,512), interpolation = cv2.INTER_AREA)
    ds_green = cv2.resize(green, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    ds_norm = cv2.resize(red_norm, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    imsave(f'{data.proc_dir}/ds_red.png', convert_8bit(ds_red))
    imsave(f'{data.proc_dir}/ds_green.png', convert_8bit(ds_green))
    imsave(f'{data.proc_dir}/ds_norm.png', convert_8bit(ds_norm))
    ops = {'Lx': 512, 'Ly': 512, 'refImg': ds_green} # for 32L w/ deleted ops

    
    # phasecorr align downsampled images with ref img for recording
    # often they're off by a couple pixels which really can matter
    # this is why you need to hold onto the green image from the red-channel scan
    ref = phasecorr_reference(ops['refImg'], 1.2)
    y_shift, x_shift, _ = phasecorr(ds_green[np.newaxis, :, :], ref, 0.1, 0)
    ds_red = shift_frame(ds_red, y_shift[0], x_shift[0])
    ds_green = shift_frame(ds_green, y_shift[0], x_shift[0])
    ds_norm = shift_frame(ds_norm, y_shift[0], x_shift[0])

    # now, for each cell, extract the relevant part around it
    # will loop thru again for labeling but this first loop collects everything
    # necessary for percentile filtering before manual intervention
    print('collecting ROI stats...')
    green_roi_list = []
    red_roi_list = []
    cell_idxs = []
    ypixs = []
    xpixs = []
    red_vals = []
    for i in range(len(stat)):
        # midpoint of ROI to center surrounding pictures on
        y, x = stat[i]['med']

        # want square around ROI center
        min_y_end = max(0, y-side_pix)
        max_y_end = min(ops['Ly'], y+side_pix)
        min_x_end = max(0, x-side_pix)
        max_x_end = min(ops['Lx'], x+side_pix)
        # these needed to pad resulting picture to square so contours draw right
        y_offset = 2 * side_pix - (max_y_end - min_y_end)
        x_offset = 2 * side_pix - (max_x_end - min_x_end)

    
        # save the ROIs for later display
        red_roi_pic = np.zeros((2 * side_pix, 2 * side_pix))
        green_roi_pic = np.zeros((2 * side_pix, 2 * side_pix))
        red_roi_pic[y_offset:, x_offset:] = ds_norm[min_y_end:max_y_end, min_x_end:max_x_end]
        green_roi_pic[y_offset:, x_offset:] = (ops['refImg'] / ops['refImg'].max())[min_y_end:max_y_end, min_x_end:max_x_end]

        green_roi_list.append(green_roi_pic)
        red_roi_list.append(red_roi_pic)


        # cut down to central parts of ROI
        # empirically this looks nice, but excluding overlapping pixels does not
        idxs_include = stat[i]['soma_crop']
        cell_idxs.append(idxs_include)

        # get red channel avg value in there
        red_vals.append(np.mean(ds_norm[stat[i]['ypix'][idxs_include],
                                    stat[i]['xpix'][idxs_include]]))

        # for plotting contours
        ypixs.append(stat[i]['ypix'][idxs_include] - min_y_end + y_offset)
        xpixs.append(stat[i]['xpix'][idxs_include] - min_x_end + x_offset)
    red_vals = np.array(red_vals) # could init to zero as array

    # now with those collected, can organize red channel values
    red_idxs = np.argsort(-red_vals) # descending order
    red_pctle = np.percentile(red_vals, percentile)

    # and loop thru to label
    print(f'{np.sum(red_vals > red_pctle)} cells to label')
    labels = np.zeros(red_vals.shape, dtype=bool) # just true/false for now... no maybe category
    for i, idx in enumerate(red_idxs):
        val = red_vals[idx]
        if val < red_pctle: # will hit this halfway thru the loop silly
            continue
        else:
            print(f'labelling cell {idx}, number {i}')
            # labelling with cv2
            # need to combine ROIs and contour into nice image
            img_combined = np.zeros((3 * 2 * side_pix + 2, 4 * side_pix + 1))
            # top left panel: ROI over unnormalized green image
            img_combined[:2 * side_pix, :2 * side_pix] = green_roi_list[idx]
            # sometimes (rarely) these are out-of-frame
            pix_idxs = np.logical_and(np.logical_and(ypixs[idx] > 0, ypixs[idx] < 2 * side_pix),
                                        np.logical_and(xpixs[idx] > 0, xpixs[idx] < 2 * side_pix))
            img_combined[ypixs[idx][pix_idxs], xpixs[idx][pix_idxs]] = 1
            # top right panel: ROI over unnormalized red image
            img_combined[:2 * side_pix, 2 * side_pix + 1:] = red_roi_list[idx] / 255.
            img_combined[ypixs[idx][pix_idxs], xpixs[idx][pix_idxs] + 2 * side_pix + 1] = 1
            # middle left panel: unnormalized green image alone
            img_combined[2*side_pix + 1:4 * side_pix + 1, :2 * side_pix] = green_roi_list[idx]
            # middle right panel: unnormalized red image alone
            img_combined[2*side_pix + 1:4 * side_pix + 1, 2 * side_pix + 1:] = red_roi_list[idx] / 255.
            # bottom left panel: normalized green image
            img_combined[4 * side_pix + 2:, :2 * side_pix] = green_roi_list[idx] / green_roi_list[idx].max()
            # bottom right panel: normalized red image
            img_combined[4 * side_pix + 2:, 2 * side_pix + 1:] = red_roi_list[idx] / red_roi_list[idx].max()


            # panel 2: normalized red image alone
            #img_combined[2 * side_pix + 1:4 * side_pix + 1] = red_roi_list[idx] / red_roi_list[idx].max()
            # panel 3: unnormalized green image alone
            #img_combined[4 * side_pix + 2:6 * side_pix + 2] = green_roi_list[idx]
            # panel 4: normalized green image alone
            #img_combined[6 * side_pix + 3:] = green_roi_list[idx] / green_roi_list[idx].max()
            img_combined = cv2.resize(img_combined, dsize=(img_combined.shape[1] * 3, img_combined.shape[0] * 3),
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


    # done looping, process to full list and save
    print(f'{np.sum(labels)} red cells detected')
    if sure:
        fname = data.proc_dir + 'sure_red_labels.npy'
    else:
        fname = data.proc_dir + 'red_labels.npy'
    print(f'done! saving to {fname}')
    np.save(fname, np.array(labels))
