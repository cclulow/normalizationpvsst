import glob
import skimage.io as skio
import cv2
import scipy
from skimage.morphology import remove_small_objects, remove_small_holes, dilation, closing
from scipy.ndimage import label, center_of_mass
from scipy.sparse import csc_matrix
import skimage
from skimage.measure import find_contours
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import math
import cv2
from scipy import stats
import g_utils as utils
import sys

from suite2p.registration.rigid import phasecorr, phasecorr_reference, shift_frame

def load_s2p_results(s2p_fld):
    F = np.load(s2p_fld + 'F.npy', allow_pickle=True)
    Fneu = np.load(s2p_fld + 'Fneu.npy', allow_pickle=True)
    spks = np.load(s2p_fld + 'spks.npy', allow_pickle=True)
    stat = np.load(s2p_fld + 'stat.npy', allow_pickle=True)
    ops =  np.load(s2p_fld + 'ops.npy', allow_pickle=True)
    ops = ops.item()
    iscell = np.load(s2p_fld + 'iscell.npy', allow_pickle=True)
    return F, Fneu, spks, stat, ops, iscell

def load_red_series(day_fld, hires=True, s=0.05, hires_only=True, green=False,
                    keep_hires=False, unweave_frames=False):
    '''
    load in all the red channel data you can
    weight average by dwell time * n_frames or something close to it

    also include green channel to cross-register with suite2p
    '''
    fnames = os.listdir(day_fld)
    red_series = [day_fld + fname for fname in fnames if fname.startswith('1020_series')
            or fname.startswith('1040_series')]
    #red_frames = [skio.imread(series, plugin='tifffile')[:, 1] for
            #series in red_series]
    red_means = []
    red_ts = []
    for series in red_series:
        red_frame = skio.imread(series, plugin='tifffile')#[:, 1]
        red_means.append(red_frame.mean(0))
        red_ts.append(red_frame.shape[0])


    if hires_only:
        # only took hires, at start of recording, don't need to correlate with
        # lowres scan at beginning
        red_means = []
        red_ts = []
        hires_series = [day_fld + fname for fname in fnames
            if 'hires' in fname]
        print(hires_series)
        for series in hires_series:
            red_frame = skio.imread(series, plugin='tifffile')#[:, 1]
            if unweave_frames:
                idxs = np.arange(0, red_frame.shape[0], 2) + 1
                red_only_frame = red_frame[idxs]
                green_only_frame = red_frame[idxs - 1]
                red_frame = np.stack([green_only_frame, red_only_frame], axis=1)
            print(red_frame.shape)
            if not keep_hires:
                red_mean = skimage.transform.rescale(red_frame.mean(0), (1, 0.5, 0.5))
            else:
                red_mean = red_frame.mean(0) # still take mean across time
            # since I take these at the end, need to translate them to lowres means
            # this encodes the assumption I took lowres series as well
           
            red_means.append(red_mean)
            red_ts.append(red_frame.shape[0] * 2) # *2 because each frame double length
    elif hires:    # compute fourier-transformed (or something) image for registering with later series
        lowres_reference = phasecorr_reference(red_means[0][1], 1.2) # 1.2 sigma is fine
        hires_series = [day_fld + fname for fname in fnames
                if fname.startswith('hires_1020_series') or
                fname.startswith('hires_1040_series')]
        for series in hires_series:
            red_frame = skio.imread(series, plugin='tifffile')#[:, 1]
            red_mean = skimage.transform.rescale(red_frame.mean(0), (1, 0.5, 0.5))
            # since I take these at the end, need to translate them to lowres means
            # this encodes the assumption I took lowres series as well
            ymax, xmax, _ = phasecorr(red_mean[np.newaxis, 1], lowres_reference, 0.1, 0)
            red_mean = np.array([shift_frame(channel, ymax[0], xmax[0]) for
                channel in [red_mean[0], red_mean[1]]])

            red_means.append(red_mean)
            red_ts.append(red_frame.shape[0] * 2) # *2 because each frame double length

    # concatenate all together, weighted by time
    print(red_means[0].shape)
    full_mean = np.average(red_means, axis=0, weights=red_ts)
    print(full_mean.shape)
    red_mean = full_mean[1] - s * full_mean[0]# drop green now, maybe don't need it

    # finally do some histogram normalization shit
    # so it more nicely converts to uint8
    # these params don't seem to be super sensitive
    # first scale to uint16 range for clahe
    #red_mean = red_mean + red_mean.min()
    #red_mean = red_mean.astype(np.uint16)
    #clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(40, 40))
    #red_mean = clahe.apply(red_mean)
    #red_mean = red_mean - red_mean.min()
    #red_mean = red_mean / red_mean.max()
    #red_mean = (red_mean * 255).astype(np.uint8)

    # temporary work
    #lowres_mean = np.mean(red_means[:2], 0)
    #hires_mean = np.mean(red_means[2:], 0)
    if not green:
        return red_mean
    else:
        return red_mean, full_mean[0]


def downsample_components_mat(Ain,new_dim,order='C'):
    '''downsamples a matrix of mask (can be square or sparse).
    Inputs are the original masks, either as a square matrix (npixels,npixels,n_neurons) or sparse (npixels*npixels,n_neurons)
    and the desired new dimension as an integer (assumes that it will be square).
    the 'reshape' order ('C' or 'F') is an optional input.
    Returns the downsampled masks with interlinear interpolation'''
    if len(Ain.shape) == 2:
        square_dim = np.sqrt(Ain.shape[0]).astype(int)
        Ain_square = Ain.reshape(square_dim,square_dim,-1,order=order)
    elif len(Ain.shape)==3:
        Ain_square = Ain
    # plt.figure()
    # plt.imshow(Ain_square[:,:,0])
    #print(np.shape(Ain_square))
    #print(np.shape(Ain_square[:,:,0]))
    n_components = Ain_square.shape[2]
    Ain_down = np.zeros((new_dim,new_dim,n_components))
    for i in range(n_components):
        A = cv2.resize(Ain_square[:,:,i].astype(float),dsize=(new_dim,new_dim), interpolation = cv2.INTER_LINEAR)
        A[A>0]=1
        Ain_down[:,:,i] = A.astype(int)
    # plt.figure()
    # plt.imshow(Ain_down[:,:,0])
    # plt.show()
    if len(Ain.shape) == 2:
        Ain_down_sparse = Ain_down.reshape(new_dim**2,np.shape(Ain_down)[2],order='F')
        return Ain_down_sparse.astype(bool)
    elif len(Ain.shape)==3:
        return Ain_down.astype(bool)

# 011024 -- modified for hal
def find_red_rois(day_fld,save_fld=None,stack=False,block_size = 5,min_obj_size=35,area_threshold=10,ecc_th=0.9, background_img=None):
    '''Saves "A_red.npy", "contours_all.pdf", and "all_contours.tiff" 
    Also returns A_red
    "A_red.npy": array of masks with shape (512,512,n_red_cells)
    "contours_all.pdf": just for reference - image with the identified contours
    "all_contours.tiff": mean image with 512 x 512 pixels, overlaid with drawn contours. Used to draw additional masks.'''
    

    '''PARAMETERS (adjust these if results are poor)
    block_size: width of the gaussian in pixels for the adaptive thresholding
    min_obj_size: minimum size (pixel area) of objects selected as possible cells (this will change based on resolution)
    area_threshold: holes smaller than this get removed
    ecc_th: eccentricity threshold for discarding non-circular rois (lower is stricter)'''

    #if needed, save multipage tiff stack of red channel data
    #red_fld = glob.glob(day_fld + '/red/TSeries*')
    #red_fld = glob.glob(day_fld + '1020_series*')
    
    red_fld = [day_fld]
    print('red fld: ', red_fld)
    if stack: 
        #save_multipage_tiffs.save_stack(red_fld[0],L=3000,two_channels=True)
        print('some multipage tiff shit for bruker')
        sys.exit()
    #fname_red = glob.glob(red_fld[0] + '/stack*Ch1.tif')
    fname_red = glob.glob(red_fld[0] + '/hires_1040_series*01.tif')

    if save_fld ==None:
        save_fld = red_fld[0] + '/red_reg_results/'
    if not os.path.exists(save_fld):
        os.mkdir(save_fld)

    '''`
    #load red image
    red_stack = skio.imread(fname_red[0], plugin="tifffile")[:, 1]
    print('red stack shape: ',red_stack.shape)
    mean_img = np.mean(red_stack,axis=0)
    if (mean_img.shape != (512,512)) and (mean_img.shape !=(1024,1024)):
        mean_img = np.mean(red_stack,axis=2)
    print('mean_img shape',mean_img.shape)
    '''
    mean_img = load_red_series(day_fld)
    mean_img = load_red_series(day_fld, hires=True, s=0, hires_only=True, green=False, keep_hires=False)

    fig, ax = plt.subplots(2,4)
    ax[0,0].imshow(mean_img)
    ax[0,0].set_title('Mean Image')

    #Algorithm to extract static ROIs:

    Y = mean_img

    #BLUR
    mR = Y.mean(axis=0) if Y.ndim == 3 else Y
    img = cv2.blur(mR, (block_size, block_size))
    ax[0,1].imshow(img)
    ax[0,1].set_title('blur')
    #NORMALIZE
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.
    ax[0,2].imshow(img)
    ax[0,2].set_title('Norm')
    #BINARIZE
    img = img.astype(np.uint8)
    #DENOISE
    img = cv2.fastNlMeansDenoising(img)
    ax[0,3].imshow(img)
    ax[0,3].set_title('Denoise')
    #ADAPTIVE THRESH
    th = cv2.adaptiveThreshold(img, np.max(img), cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, block_size, 0)
    ax[1,0].imshow(th)
    ax[1,0].set_title('adaptive thresh')
    #REMOVE SMALL HOLES
    th = remove_small_holes(th > 0, area_threshold=area_threshold)
    ax[1,1].imshow(th)
    ax[1,1].set_title('remove small holes')
    #REMOVE SMALL OBJECTS
    th = remove_small_objects(th, min_size=min_obj_size)
    ax[1,2].imshow(th)
    ax[1,2].set_title('remove small objects')
    #plt.show()
    #EXTRACT AREAS
    areas = skimage.measure.label(th)
    #CONVERT TO MASKS
    regions = skimage.measure.regionprops(areas)
    selem = np.ones((3, 3))
    n_cells = len(regions)
    A_sparse = np.zeros((np.prod(th.shape), n_cells), dtype=bool)
    A_square = np.zeros((th.shape[0],th.shape[0], n_cells), dtype=bool)
    expand_method = 'closing'
    for i in range(n_cells):
        temp = (areas == i + 1)
        if expand_method == 'dilation':
            temp = dilation(temp, selem=selem)
        elif expand_method == 'closing':
            temp = closing(temp, selem=selem)
        A_square[:,:,i] = temp
        A_sparse[:, i] = temp.flatten('F')
    #IMPOSE CIRCULARITY CONSTRAINT
    ecc = []
    for prop in regions:
        ecc.append(prop.eccentricity)
    A_square = A_square[:,:,np.array(ecc)<ecc_th]
    A_sparse = A_sparse[:,np.array(ecc)<ecc_th]
    # plt.figure()
    # plt.hist(ecc)
    #DOWNSAMPLE TO 512 X 512
    A_square = downsample_components_mat(A_square,512,order='C')
    A_sparse = downsample_components_mat(A_sparse,512,order='F')
    ax[1,3].imshow(np.sum(A_square,axis=2))
    ax[1,3].set_title('final masks')
    plt.show()
    #SAVE MASK
    np.save(save_fld + '/A_red.npy', A_square) 
    #PLOT CONTOURS
    #import caiman as cm
    fig = plt.figure(figsize=(5.12, 5.12),frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ##need to downsample image too for this plotting to work
    img_down = cv2.resize(mR,dsize=(512,512), interpolation = cv2.INTER_LINEAR)
    if background_img is not None:
        plot_contours(A_sparse, background_img,display_numbers=False)#,cmap='gray')
    else:
        plot_contours(A_sparse, img_down,display_numbers=False,cmap='gray')
    plt.savefig(save_fld + '/contours_all.pdf')
    png1 = io.BytesIO()
    plt.savefig(png1, format="png",dpi=100)
    # Load this image into PIL
    png2 = Image.open(png1)
    # Save as TIFF
    png2.save(save_fld + "/all_contours.tiff")
    png1.close()
    return A_square


def draw_new_masks(day_fld, radius=5,save_fld=None, method='fixed_radius'):
    '''heavily drawn from: https://www.tutorialspoint.com/opencv-python-how-to-draw-circles-using-mouse-events'''

    '''Saves (and returns) "manual_masks.tiff": a tiff image with the newly drawn masks (the input for add_manual_masks)'''

    #METHOD OPTIONS:
    #'fixed_radius': click and a fixed radius circle appears
    #'drag_circle': click and drag to get a circle - radius depends on how far you drag
    #'draw': draw the contours
    #NOTE: I decided I would just use fixed radius, and added more functionality to it (saving the masks individually).
    #this stuff can be added to the other methods fairly easily. 
    #red_fld = glob.glob(day_fld + '/red/TSeries*')
    red_fld = day_fld
    red_contours_path = red_fld + '/red_reg_results/all_contours.tiff'
    if save_fld ==None:
        save_fld = red_fld + '/red_reg_results/'
    if not os.path.exists(save_fld):
        os.mkdir(save_fld)

    if method == 'fixed_radius':
       global img 
       global cache_img
       global cache_mask
       global mask_img
       img = cv2.imread(red_contours_path)
       mask_img = np.zeros((512,512), np.uint8)
       cache_img = img.copy()
       cache_mask = mask_img.copy()
       # define mouse callback function to draw circle
       def draw_circle(event, x, y, flags, param):
             global img 
             global cache_img
             global cache_mask
             global mask_img
             if event == cv2.EVENT_LBUTTONDOWN:
                cache_img = img.copy()
                cache_mask = mask_img.copy()
                cv2.circle(img, (x, y), radius, (0, 255, 255), 2)
                cv2.circle(mask_img,(x,y),radius,255,-1)
             if event == cv2.EVENT_MBUTTONDOWN: #middle click reverts to previous version of img
                img = cache_img.copy()
                mask_img = cache_mask.copy()
       # Create a window
       cv2.namedWindow("Click To Draw Masks. Esc when finished.",cv2.WINDOW_NORMAL) 
       # bind the callback function to the window
       cv2.setMouseCallback("Click To Draw Masks. Esc when finished.", draw_circle)

       # display the image
       while True:
          cv2.imshow("Click To Draw Masks. Esc when finished.", img)
          if cv2.waitKey(20) & 0xFF == 27: #break for esc key
             break
       cv2.destroyAllWindows()

    elif method == 'drag_circle':
       drawing = False # true if mouse is pressed
       ix, iy = -1, -1

       # define mouse callback function to draw circle
       def draw_circle(event, x, y, flags, param):
          global ix, iy, drawing
          if event == cv2.EVENT_LBUTTONDOWN:
             drawing = True
          
             # we take note of where that mouse was located
             ix, iy = x, y
          elif event == cv2.EVENT_MOUSEMOVE:
             drawing == True
          elif event == cv2.EVENT_LBUTTONUP:
             radius = int(math.sqrt(((ix - x) ** 2) + ((iy - y) ** 2)))
             cv2.circle(img, (ix, iy), radius, (255, 0, 255), thickness=2)
             cv2.circle(mask_img,(ix,iy),radius,(255,255,255),-1)
             drawing = False
       # Create a black image
       img = cv2.imread(red_contours_path)
       mask_img = np.zeros((512, 512, 3), np.uint8)

       # Create a window
       cv2.namedWindow('Add Masks', cv2.WINDOW_NORMAL)

       # bind the callback function to above defined window
       cv2.setMouseCallback('Drag Circle Window', draw_circle)

       # display the image
       while True:
          cv2.imshow('Add Masks', img)
          k = cv2.waitKey(1) & 0xFF
          if k == 27:
             break
       cv2.destroyAllWindows()
    elif method == 'draw':
       #this is incomplete right now - mask image has drawn contours, not filled in masks
       drawing = False # true if mouse is pressed
       pt1_x , pt1_y = None , None

       # mouse callback function
       def line_drawing(event,x,y,flags,param):
           global pt1_x,pt1_y,drawing

           if event==cv2.EVENT_LBUTTONDOWN:
               drawing=True
               pt1_x,pt1_y=x,y

           elif event==cv2.EVENT_MOUSEMOVE:
               if drawing==True:
                   cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=1)
                   cv2.line(mask_img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=1)
                   pt1_x,pt1_y=x,y
           elif event==cv2.EVENT_LBUTTONUP:
               drawing=False
               cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=1) 
               cv2.line(mask_img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=1)        

       img = cv2.imread('/mnt/birch/051523/red/TSeries-05152023-0937-967/all_contours.tiff')
       mask_img = np.zeros((512,512,3), np.uint8)

       cv2.namedWindow('draw masks',cv2.WINDOW_NORMAL)
       cv2.setMouseCallback('draw masks',line_drawing)


       while(1):
           cv2.imshow('draw masks',img)
           if cv2.waitKey(1) & 0xFF == 27:
               break
       cv2.destroyAllWindows()
    else:
       print('not a valid method')

    cv2.imwrite(save_fld + '/manual_masks.tiff',mask_img)
    return mask_img

def add_manual_masks(day_fld,save_fld=None):
    '''Assumes "A_red.npy" has been saved. 
    Returns and saves as npy file "A_red_all.npy" including the manually added masks.
    Also plots an overlay.'''
    #red_fld = glob.glob(day_fld + '/red/TSeries*')
    red_fld = day_fld
    A = np.load(red_fld + '/red_reg_results/A_red.npy')
    dim = A.shape[0]
    manual_masks_path = red_fld + '/red_reg_results/manual_masks.tiff'
    manual_masks = cv2.imread(manual_masks_path,cv2.IMREAD_GRAYSCALE)

    if save_fld==None:
        save_fld = red_fld + '/red_reg_results/'
    if not os.path.exists(save_fld):
        os.mkdir(save_fld)

    #extract individual masks from image of all the masks
    #EXTRACT AREAS
    areas = skimage.measure.label(manual_masks)
    #CONVERT TO MASKS
    regions = skimage.measure.regionprops(areas)
    selem = np.ones((3, 3))
    n_cells = len(regions)
    A_manual = np.zeros((dim,dim, n_cells), dtype=bool)
    expand_method = 'dilation'
    for i in range(n_cells):
        temp = (areas == i + 1)
        if expand_method == 'dilation':
            temp = dilation(temp, selem=selem)
        elif expand_method == 'closing':
            temp = closing(temp, selem=selem)
        A_manual[:,:,i] = temp

    #reshape A into square matrix
    A_square = A.reshape(dim,dim,-1,order='F')

    #concatenate masks
    masks_new = np.zeros([dim,dim,A_manual.shape[2]+A_square.shape[2]])
    masks_new[:,:,0:A_square.shape[2]] = A_square
    masks_new[:,:,A_square.shape[2]:] = A_manual 

    #save masks
    np.save(save_fld + '/A_red_all.npy',masks_new)

    #load red mean image for plot
    #red_stack = skio.imread(red_fld + '/1020_scan_0001.tif', plugin="tifffile")
    #red_stack = load_red_series(day_fld)
    #mean_img = np.mean(red_stack,axis=0)
    mean_img = load_red_series(day_fld)

    #plot red masks overlaid on red mean img
    plt.figure()
    plt.imshow(mean_img,cmap='gray')
    plt.imshow(np.sum(masks_new,axis=2),cmap='gray',alpha=0.5)
    plt.title('red masks over red image')
    plt.savefig(save_fld + '/contours_manually_added.tiff')
    plt.show()

    return masks_new

def compute_red_mask_correlation(day_fld,s2p_fld, save_fld = None, thresh=0.5):
    '''Returns the set of indices of cells with an above threshold spatial correlation (pearson)
    between red channel and green channel masks. Assumes s2p has been run and red masks have been saved.
    Indices are returned as a tuple of lists of indices, with the first list being ind for the red masks and
    the second list being indices for the green channel.'''
    #red_fld = glob.glob(day_fld + '/red/TSeries*')
    red_fld = day_fld
    #fname_red = glob.glob(red_fld[0] + '/stack*Ch1.tif')
    fname_red = glob.glob(red_fld + '/hires_1040_series*01.tif')

    if save_fld == None:
        save_fld = red_fld + '/red_reg_results/'
    if not os.path.exists(save_fld):
        os.mkdir(save_fld)

    #load s2p results from functional channel
    F, Fneu, spks, stat, ops, iscell = utils.load_s2p_results(s2p_fld)
    green_img = ops['refImg']
    #exclude cells that have high sensitivity to AR2 threshold
    #exclude_ind = utils.sensitivity_to_thresh(day_fld,s2p_fld,[2,3,4,5])
    stat_cells = stat[iscell[:,0].astype(bool)]
    #stat_cells = np.delete(stat_cells,exclude_ind,0)

    #load red channel masks
    #A_red = np.load(red_fld + '/red_reg_results/A_red_all.npy')
    A_red = np.load(red_fld + '/red_reg_results/A_red.npy')
    ncells_green = len(stat_cells)
    dim = 512
    #get x and y pixels as masks
    green_masks = np.zeros([dim,dim,ncells_green])
    for n in range(0,ncells_green):
        ypix = stat_cells[n]['ypix']
        xpix = stat_cells[n]['xpix']
        green_masks[ypix,xpix,n] = stat_cells[n]['lam'] # was 1

    #reshape A_red to no longer be sparse
    if len(A_red.shape)==2:
        square_dim = np.sqrt(A_red.shape[0]).astype(int)
        red_masks = A_red.reshape(square_dim,square_dim,-1)
    else:
        red_masks = A_red

    #plot red masks to make sure nothing got scrambled in the reshaping
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(np.sum(red_masks,axis=2))
    ax[0].set_title('red masks')
    #plot red masks overlaid on red mean img
    #red_stack = skio.imread(red_fld + '/1020_scan_00001.tif', plugin="tifffile")
    #red_img = np.mean(red_stack,axis=0)
    red_img = load_red_series(day_fld)
    ax[1].imshow(red_img,cmap='gray')
    ax[1].imshow(np.sum(red_masks,axis=2),cmap='gray',alpha=0.5)
    ax[1].set_title('red masks over red image')
    #plot red masks overlaid on green ref img
    ax[2].imshow(green_img,cmap='gray')
    ax[2].imshow(np.sum(red_masks,axis=2),cmap='gray',alpha=0.5)
    ax[2].set_title('red masks over green image')

    #compute correlation between masks 
    # if os.path.exists(red_fld[0] + '/red_reg_results/corr_red.npy'):
    #     corr = np.load(red_fld[0] + '/red_reg_results/corr_red.npy')
    # else:
    #always compute corr because masks might be different
    ncells_red = red_masks.shape[2]
    corr = np.zeros([ncells_red,ncells_green])
    for n in range(ncells_red):
        for m in range(ncells_green):
            pearson = stats.pearsonr(red_masks[:,:,n].flatten(),green_masks[:,:,m].flatten())
            corr[n,m] = pearson[0]
    np.save(save_fld + '/corr_red.npy', corr)

    fig,ax = plt.subplots(2,1)
    ax[0].imshow(corr)
    ax[1].hist(corr[corr>0].flatten())
    ax[0].set_title('mask spatial correlation')
    ax[1].set_title('distribution of nonzero correlations')
    plt.show()

    ind = np.where(corr>thresh)

    #plot an overlay of only the high correlation masks
    A_red_high_corr = red_masks[:,:,ind[0]].reshape(red_masks.shape[0]**2,-1,order='F')
    A_red_all = red_masks.reshape(red_masks.shape[0]**2,-1,order='F')
    red_img_down = cv2.resize(red_img,dsize=(512,512), interpolation = cv2.INTER_LINEAR)
    #import caiman as cm
    plt.figure()
    plot_contours(A_red_high_corr, red_img_down,display_numbers=True,cmap='gray')
    plt.savefig(save_fld + '/red_mask_overlay_high_corr_red.pdf')
    plt.figure()
    plot_contours(A_red_high_corr, green_img,display_numbers=True,cmap='gray')
    plt.savefig(save_fld + '/red_mask_overlay_high_corr_green.pdf')
    plt.figure()
    plot_contours(A_red_high_corr, np.sum(green_masks,axis=2),display_numbers=True,cmap='gray')
    plt.savefig(save_fld + '/red_mask_overlay_high_corr_green_masks.pdf')
    plt.figure()
    plot_contours(A_red_all, np.sum(green_masks,axis=2),display_numbers=True,cmap='gray')
    plt.savefig(save_fld + '/red_mask_overlay_all_green_masks.pdf')

    #sometimes a red mask has an above threshold correlation with more than one green mask
    #in these cases, keep the one with the higher correlation

    unique_ind = ([],[])
    if len(np.unique(ind[0])) < len(ind[0]):
        for n in np.unique(ind[0]):
            if list(ind[0]).count(n) > 1:
                duplicate_ind_green = ind[1][np.where(ind[0]==n)]
                corr_max_ind = duplicate_ind_green[np.argmax(corr[n,duplicate_ind_green])]
                unique_ind[0].append(n)
                unique_ind[1].append(corr_max_ind)
            else:
                unique_ind[0].append(n)
                unique_ind[1].append(ind[1][np.where(ind[0]==n)][0])
    else:
        unique_ind = ind

    #also make sure all green ind are unique, although it's more rare that they're not
    unique_ind_green = ([],[])
    if len(np.unique(unique_ind[1])) < len(unique_ind[1]):
        for n in np.unique(ind[1]):
            if list(ind[1]).count(n) > 1:
                duplicate_ind_red = ind[0][np.where(ind[1]==n)]
                corr_max_ind = duplicate_ind_red[np.argmax(corr[duplicate_ind_red,n])]
                unique_ind_green[1].append(n)
                unique_ind_green[0].append(corr_max_ind)
            else:
                unique_ind_green[1].append(n)
                unique_ind_green[0].append(ind[0][np.where(ind[1]==n)][0])
    else:
        unique_ind_green = unique_ind


    np.save(save_fld + 'red_reg_ind.npy', unique_ind_green)
    print('Found ', len(unique_ind_green[0]), ' colabeled cells.')
    print(unique_ind_green)
    return unique_ind_green

def pipeline(day_fld,s2p_fld, thresh=0.3):
    find_red_rois(day_fld)
    draw_new_masks(day_fld)
    add_manual_masks(day_fld)
    compute_red_mask_correlation(day_fld,s2p_fld, thresh=thresh)

# TODO: conda couldn't figure out caiman install
# so I'm copy-pasting their function
# I think that's illegal...
def plot_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                  cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None, coordinates=None,
                  contour_args={}, number_args={}, **kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates

     Args:
         A:   np.ndarray or sparse matrix
                   Matrix of Spatial components (d x K)
    
         Cn:  np.ndarray (2D)
                   Background image (e.g. mean, correlation)
    
         thr_method: [optional] string
                  Method of thresholding:
                      'max' sets to zero pixels that have value less than a fraction of the max value
                      'nrg' keeps the pixels that contribute up to a specified fraction of the energy
    
         maxthr: [optional] scalar
                    Threshold of max value
    
         nrgthr: [optional] scalar
                    Threshold of energy
    
         thr: scalar between 0 and 1
                   Energy threshold for computing contours (default 0.9)
                   Kept for backwards compatibility. If not None then thr_method = 'nrg', and nrgthr = thr
    
         display_number:     Boolean
                   Display number of ROIs if checked (default True)
    
         max_number:    int
                   Display the number for only the first max_number components (default None, display all numbers)
    
         cmap:     string
                   User specifies the colormap (default None, default colormap)

     Returns:
          coordinates: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
    """

    if swap_dim:
        Cn = Cn.T
        print('Swapping dim')

    if thr is None:
        try:
            thr = {'nrg': nrgthr, 'max': maxthr}[thr_method]
        except KeyError:
            thr = maxthr
    else:
        thr_method = 'nrg'


    for key in ['c', 'colors', 'line_color']:
        if key in kwargs.keys():
            color = kwargs[key]
            kwargs.pop(key)

    ax = plt.gca()
    if vmax is None and vmin is None:
        plt.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=np.percentile(Cn[~np.isnan(Cn)], 1),
                  vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    else:
        plt.imshow(Cn, interpolation=None, cmap=cmap, vmin=vmin, vmax=vmax)

    if coordinates is None:
        if len(np.shape(Cn)) == 3:
            coordinates = get_contours(A, np.shape(Cn)[:2], thr, thr_method, swap_dim)
        else:
            coordinates = get_contours(A, np.shape(Cn), thr, thr_method, swap_dim)
    for c in coordinates:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        plt.plot(*v.T, c=colors, **contour_args)

    if display_numbers:
        d1, d2 = np.shape(Cn)
        d, nr = np.shape(A)
        cm = com(A, d1, d2)
        if max_number is None:
            max_number = A.shape[1]
        for i in range(np.minimum(nr, max_number)):
            if swap_dim:
                ax.text(cm[i, 0], cm[i, 1], str(i + 1), color=colors, **number_args)
            else:
                ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors, **number_args)
    return coordinates

def get_contours(A, dims, thr=0.9, thr_method='nrg', swap_dim=False):
    """Gets contour of spatial components and returns their coordinates

     Args:
         A:   np.ndarray or sparse matrix
                   Matrix of Spatial components (d x K)

             dims: tuple of ints
                   Spatial dimensions of movie (x, y[, z])

             thr: scalar between 0 and 1
                   Energy threshold for computing contours (default 0.9)

             thr_method: [optional] string
                  Method of thresholding:
                      'max' sets to zero pixels that have value less than a fraction of the max value
                      'nrg' keeps the pixels that contribute up to a specified fraction of the energy

     Returns:
         Coor: list of coordinates with center of mass and
                contour plot coordinates (per layer) for each component
    """

    if 'csc_matrix' not in str(type(A)):
        A = csc_matrix(A)
    d, nr = np.shape(A)
    # if we are on a 3D video
    if len(dims) == 3:
        d1, d2, d3 = dims
        x, y = np.mgrid[0:d2:1, 0:d3:1]
    else:
        d1, d2 = dims
        x, y = np.mgrid[0:d1:1, 0:d2:1]

    coordinates = []

    # get the center of mass of neurons( patches )
    cm = com(A, *dims)

    # for each patches
    for i in range(nr):
        pars:Dict = dict()
        # we compute the cumulative sum of the energy of the Ath component that has been ordered from least to highest
        patch_data = A.data[A.indptr[i]:A.indptr[i + 1]]
        indx = np.argsort(patch_data)[::-1]
        if thr_method == 'nrg':
            cumEn = np.cumsum(patch_data[indx]**2)
            if len(cumEn) == 0:
                pars = dict(
                    coordinates=np.array([]),
                    CoM=np.array([np.NaN, np.NaN]),
                    neuron_id=i + 1,
                )
                coordinates.append(pars)
                continue
            else:
                # we work with normalized values
                cumEn /= cumEn[-1]
                Bvec = np.ones(d)
                # we put it in a similar matrix
                Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]][indx]] = cumEn
        else:
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = np.zeros(d)
            Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]]] = patch_data / patch_data.max()

        if swap_dim:
            Bmat = np.reshape(Bvec, dims, order='C')
        else:
            Bmat = np.reshape(Bvec, dims, order='F')
        pars['coordinates'] = []
        # for each dimensions we draw the contour
        for B in (Bmat if len(dims) == 3 else [Bmat]):
            vertices = find_contours(B.T, thr)
            # this fix is necessary for having disjoint figures and borders plotted correctly
            v = np.atleast_2d([np.nan, np.nan])
            for _, vtx in enumerate(vertices):
                num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
                if num_close_coords < 2:
                    if num_close_coords == 0:
                        # case angle
                        newpt = np.round(vtx[-1, :] / [d2, d1]) * [d2, d1]
                        vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)
                    else:
                        # case one is border
                        vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                v = np.concatenate(
                    (v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)

            pars['coordinates'] = v if len(
                dims) == 2 else (pars['coordinates'] + [v])
        pars['CoM'] = np.squeeze(cm[i, :])
        pars['neuron_id'] = i + 1
        coordinates.append(pars)
    return coordinates


def com(A: np.ndarray, d1: int, d2: int, d3 = None) -> np.array:
    """Calculation of the center of mass for spatial components

     Args:
         A:   np.ndarray
              matrix of spatial components (d x K)

         d1:  int
              number of pixels in x-direction

         d2:  int
              number of pixels in y-direction

         d3:  int
              number of pixels in z-direction

     Returns:
         cm:  np.ndarray
              center of mass for spatial components (K x 2 or 3)
    """

    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)

    if d3 is None:
        Coor = np.matrix([np.outer(np.ones(d2), np.arange(d1)).ravel(),
                          np.outer(np.arange(d2), np.ones(d1)).ravel()],
                         dtype=A.dtype)
    else:
        Coor = np.matrix([
            np.outer(np.ones(d3),
                     np.outer(np.ones(d2), np.arange(d1)).ravel()).ravel(),
            np.outer(np.ones(d3),
                     np.outer(np.arange(d2), np.ones(d1)).ravel()).ravel(),
            np.outer(np.arange(d3),
                     np.outer(np.ones(d2), np.ones(d1)).ravel()).ravel()
        ],
                         dtype=A.dtype)

    cm = (Coor * A / A.sum(axis=0)).T
    return np.array(cm)
