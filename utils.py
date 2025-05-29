# timing utility functions
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle as pkl

from scipy.ndimage import gaussian_filter
from scipy.signal import butter, lfilter, freqz
from suite2p.extraction.dcnv import preprocess


def setup_subfolder(main_path, sub='proc'):
    '''ideally just run once, to setup subfolder for processed data'''
    assert os.path.exists(main_path) and os.path.isdir(main_path)
    sub_path = os.path.join(main_path, sub)
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)
        print(f'creating path {sub_path} for processed data')

    return sub_path # why not

def get_traces_multiple_files(base_path, file_list):
    '''
    when running manually, I get multiple timing files, one
    per stimulus block. this basically just concatenates them,
    and returns the photodiode and imaging frame traces.
    '''
    # this is likely a very inefficient way to do this
    # but disk speed is probably the main killer here
    phdiode_trace = np.array([])
    img_trace = np.array([])
    cam_trace = np.array([])

    # make sure reading files in the right order
    print('multiple timing files detected, reading in this order:')
    print(f'{file_list}')
    for fname in file_list:
        fle = h5py.File(os.path.join(base_path, fname))
        # need time series for photodiode and imaging frames
        phdiode = np.array(fle['Screen Photodiode'])
        img = np.array(fle['Acquisition Frames'])
        cam = np.array(fle['Camera Frames'])
        phdiode_trace = np.append(phdiode_trace, phdiode)
        img_trace = np.append(img_trace, img)
        cam_trace = np.append(cam_trace, cam)
        fle.close()

    return phdiode_trace, img_trace, cam_trace

def get_stimuli_multiple_files(base_path, file_list):
    '''
    combine stimulus information across runs

    which orientation and which are plaids
    '''
    orientations = np.array([])
    plaids = np.array([])

    print('reading stimulus files in this order:')
    print(file_list)
    for fname in file_list:
        with open(os.path.join(base_path, fname), 'rb') as fle:
            expt_data = pkl.load(fle)
        oris = np.array(expt_data['oris'])
        plds = np.array(expt_data['plaids']) # to allow for nice indexing
        orientations = np.append(orientations, oris[expt_data['conds']])
        plaids = np.append(plaids, plds[expt_data['conds']])

    print(f'{len(orientations)} trials detected in stimulus files')
    stim_info = np.stack([orientations, plaids])

    return stim_info

### ABOVE: file handling, mostly
### BELOW: neural data handling, mostly
def get_trial_data(resps, trial_times, f_before=40, f_after=60):
    '''
    resps: NxT, activity traces. N is irrelevant to this
    trial_times: vector indexing into T axis for each trial start
    f_before: number of frames before each trial to slice
    f_after: number of frames after each trial (start) to slice

    returns NxTrx(f_before + f_after) trial-by-trial traces
    '''
    # can handle this if I need to...
    #assert np.max(trial_times) + f_after < resps.shape[1]
    # throw out trials if too long/short

    return np.stack([resps[:, time-f_before:time+f_after] for
        time in trial_times], axis=1)

def proc_data_to_trials(paths, threshold=None, f_before=40, f_after=60,
        slow=False):
    '''
    process suite2p outputs + h5 timing files to
    save trial-by-trial Fneu and spiking data to a subfolder

    runs independently on each path in paths

    requires input at least to determine cutoff for timing file
    photodiode signal
    '''
    assert isinstance(paths, list) # TODO: couldn't I pass a tuple? who cares
    for path in paths:
        # save it here
        proc_dir = setup_subfolder(path)

        # find and load timing file(s)
        h5_files = [fname for fname in os.listdir(path) if 'h5' in fname]
        if len(h5_files) > 1:
            # don't want this but happens when I fuck up recordings
            # or run them manually...
            phdiode_trace, img_trace, _ = get_traces_multiple_files(path, sorted(h5_files))
        else:
            timing_file = os.path.join(path, h5_files[0])
            print(f'using timing file: {timing_file}')
            fle = h5py.File(timing_file)
            # need time series for photodiode and imaging frames
            phdiode_trace = np.array(fle['Screen Photodiode'])
            img_trace = np.array(fle['Acquisition Frames'])
            fle.close()

        # using default params from the function here, seem to work ok
        trial_idxs = get_trial_idxs(img_trace, phdiode_trace, threshold=threshold, slow=slow)
        np.save(os.path.join(proc_dir, 'trial_idxs.npy'), trial_idxs) # added 041724

        # find and load suite2p outputs
        suite2p_dir = os.path.join(path, 'runs/suite2p/plane0/')
        assert os.path.exists(suite2p_dir)
        # load in cell classification
        iscell = os.path.join(suite2p_dir, 'iscell.npy')
        iscell = np.load(iscell)[:, 0].astype(bool) # only want binary yes/no, not probs
        # only want traces for cells
        f_raw = os.path.join(suite2p_dir, 'F.npy')
        f_raw = np.load(f_raw)[iscell]
        f_neu = os.path.join(suite2p_dir, 'Fneu.npy')
        f_neu = np.load(f_neu)[iscell]
        spks = os.path.join(suite2p_dir, 'spks.npy')
        spks = np.load(spks)[iscell]
        if os.path.exists(os.path.join(suite2p_dir, 'alt_spks.npy')):
            alt_spks = np.load(os.path.join(suite2p_dir, 'alt_spks.npy'))[iscell]
        else:
            alt_spks = None
        # report some data abt lengths
        print(f'{spks.shape[1]} frames of neural data')

        # have to do correction myself here
        # cause baseline stuff doesn't work w/ trial-to-trial data
        dF = f_raw - 0.7 * f_neu
        # TODO maybe make these args... probably are default suite2p params
        # except sig_baseline is documented confusingly
        dF = preprocess(dF, baseline='maximin', win_baseline=120,
                sig_baseline=10, fs=30.01)

        # now generate the responses to be saved
        # using the trial indices and the response timeseries
        f_dict = {'f_raw': f_raw, 'f_neu': f_neu, 'spks': spks, 'dF' : dF,
                'alt_spks': alt_spks}
        for name, resp in f_dict.items():
            # uses default f_before and f_after
            if resp is not None: # b/c alt spikes not always there
                trial_resps = get_trial_data(resp, trial_idxs, f_before, f_after)
                np.save(os.path.join(proc_dir, name), trial_resps)
        
        # and now do stimulus data too
        # TODO might have expt_data files in diff folder for other sessions
        expt_data_files = [fname for fname in sorted(os.listdir(path))
                if 'expt_data' in fname]
        stim_info = get_stimuli_multiple_files(path, expt_data_files)
        np.save(os.path.join(proc_dir, 'stim_info.npy'), stim_info)

    return

### ABOVE: mainly neural data processing stuff (as of 121123)
### BELOW: mainly timing data stuff, previously in time_utils.py
def get_crossings(trace, threshold, below=True):
    '''
    get indices where a threshold is crossed (from below)
    '''
    # where you flip over the threshold
    crossings = np.diff(np.sign(trace - threshold))

    # sign flips are 2 for below->above (default), -2 otherwise
    flip_val = 2 if below else -2

    # deal with cases where threshold is hit exactly
    # rare with floats but it hasppened to me
    hits = np.where(crossings == 1)[0]
    for i, hit in enumerate(hits):
        # just set where it reaches the threshold, not afterward
        if i % 2 == 0:
            crossings[hit] = flip_val
            # I guess this could happen at the very last timebin
            if hit + 1 < len(crossings):
                crossings[hit+1] = 0
    
    # and return indices of crossings now
    trial_times = np.where(crossings == flip_val)[0]
    return trial_times

def smooth_and_detrend_phdiode(phdiode_trace, subsample=400,
        sigma=5, trend_factor=500, slow=False):
    '''
    perform smoothing and 'detrending' (subtracting out a
    much smoother version for long-timescale trends, since I
    can't get a highpass filter working)

    also subsample since it's way higher resolution than needed
    (which makes smoothing really slow)

    by trial and error found to be pretty good for photodiode trace

    defaults are assuming 400Khz sampling I settled on
    '''
    if slow:
        # slow presentations, use params for those
        # TODO bit of a sloppy way to handle this
        sigma = 50
        trend_factor = 15000
    subsampled = phdiode_trace[::subsample]
    smoothed = gaussian_filter(subsampled, sigma)
    #more_sub = smoothed[::100]
    # TODO make this work for both fast and slow presentations!
    # ideally without a 3000-sigma frame gaussian filter -- slow
    trend = gaussian_filter(smoothed, trend_factor)

    return smoothed - trend

def get_phdiode_crossings(phdiode_trace, threshold=None):
    '''
    get trial times for a single session from the photodiode trace
    '''
    # subsample cause histogram too slow otherwise
    if threshold is None:
        # eyeball it
        plt.figure()
        plt.hist(phdiode_trace, bins=100)
        plt.xlabel('voltage')
        plt.title('histogram of photodiode trace')
        plt.show()
        threshold = float(input('voltage threshold for trials: '))

    return get_crossings(phdiode_trace, threshold)

def get_trial_idxs(imaging_trace, phdiode_trace, threshold=None, subsample=400,
        sigma=5, trend_factor=500, imaging_threshold=3.0, slow=False):
    '''
    combine smooth_and_detrend_phdiode and get_crossings_session
    to get trial start indices from the raw trace. using default args

    also combine with imaging frame time series
    '''
    # photodiode requires some tricky processing
    filtered_trace = smooth_and_detrend_phdiode(phdiode_trace,
            subsample=subsample, sigma=sigma, trend_factor=trend_factor,
            slow=slow)
    # below requires input, unless I can find a hardcoded threshold
    # that works. or mathematically determine one. TODO I guess
    trial_idxs = get_phdiode_crossings(filtered_trace, threshold)
    trial_times = trial_idxs * subsample
    print(f'{len(trial_times)} trials detected')

    # now get imaging crossings, easily
    img_times = get_crossings(imaging_trace, imaging_threshold)

    # and trial indices: start frames for each trial
    # this gets the closest frame to image presentation...
    # might want to always round down...
    trial_idxs = np.array([np.argmin(np.abs(img_times - trial_time)) for
        trial_time in trial_times])
    
    # cut from get_trial_data, want to get correct indices at this stage
    if np.min(trial_idxs) == 0:
        trial_idxs = trial_idxs[1:]
        print(f'first trial time bad, removing for {len(trial_idxs)} trials detected now')

    print(f'first trial at frame {trial_idxs[0]}, last at {trial_idxs[-1]}')

    return trial_idxs

def get_camera_frames(cam_trace, threshold=2.0, flip=4.0):
    '''
    get camera frame times. beginnings or ends?
    threshold of 2.0 is quite reliable
    4.0 is always above the signal, used to invert it
    '''
    # main issue here is dealing with the jumps around ends of recordings...
    # want to grab threshold crossing immediately followed by other crossing
    crossings = get_crossings(cam_trace, threshold)
    inv_crossings = get_crossings((flip - cam_trace), threshold)

    # should be one more inverted crossing than crossing
    # not sure how this will work out in manual mode tho. TODO
    print(f'cam trace: {crossings.shape} crossings, {inv_crossings.shape} inverted')
    diffs = crossings[:-1] - inv_crossings

    # these are very brief pulses -- 17-20 bins in the one I looked at
    good_frames = np.logical_and(diffs > -25, diffs < -10) # some wiggle room
    print(f'{np.sum(good_frames)} frames detected in camera trace')

    # so just return those indexes
    # beginning/end of pulse is irrelevant
    # but whether this is at start or end of frame does matter.
    return inv_crossings[good_frames]

def translate_timestamps(from_ts, to_ts):
    '''
    translate timestamps of one series to another's indices
    does that make sense? #TODO
    '''
    # inspired by similar use of interp1d in facemap utils
    interp_func = interp1d(to_ts, np.arange(len(to_ts)), kind='nearest')
    # this maps timestamps (shared) to indices of target TS
    # so now just use this to map
    # should fill in NANs where it doesn't fit.
    mapped_ts = interp_func(from_ts)

    return mapped_ts



### BELOW: plotting utils?

def process_and_plot(path, mouse, date):
    '''
    super sloppy need to clean this up.
    ''' #TODO
    #proc_data_to_trials([path], threshold=None)

    proc_dir = os.path.join(path, 'proc/')
    r_spks = np.load(os.path.join(proc_dir, 'spks.npy'))#[:, ]#[:, ::4]
    r_neu = np.load(os.path.join(proc_dir, 'f_neu.npy'))#[:, ::2, 40:]#[:, ::4]
    r_raw = np.load(os.path.join(proc_dir, 'f_raw.npy'))#[:, ::2, 40:]#[:, ::4]
    r_dF = np.load(os.path.join(proc_dir, 'dF.npy'))#[:, ::2, 40:]#[:, ::4]

    # zscore everything -- don't trust scipy's function
    def z_score(resps):
        #return resps / resps.max((1, 2)).reshape(-1, 1, 1)
        means = resps.mean((1, 2))
        stds = resps.std((1, 2))
        return (resps - means.reshape(-1, 1, 1)) / stds.reshape(-1, 1, 1)

    def sort(resps):
        #return resps
        return resps[np.argsort(np.argmax(resps, 1))]

    spks = z_score(r_spks).mean(1)
    neu = z_score(r_neu).mean(1)
    raw = z_score(r_raw).mean(1)
    dF = z_score(r_dF).mean(1)

    # try thresholded spikes
    spk_thr_tr = (z_score(r_spks) > 3.0)
    spk_thr = spk_thr_tr.mean(1)

    # set up to save
    fname_base = f'figures/psths/{mouse}_{date}'

    # plot things
    plt.figure()
    plt.imshow(sort(spks))
    plt.colorbar()
    plt.title('spikes')
    plt.savefig(f'{fname_base}_spikes.png', dpi=200)

    plt.figure()
    plt.imshow(sort(neu))
    plt.colorbar()
    plt.title('f_neu')
    plt.savefig(f'{fname_base}_neu.png', dpi=200)

    plt.figure()
    plt.imshow(sort(dF))
    plt.colorbar()
    plt.title('f - 0.7f_neu, baseline corrected')
    plt.savefig(f'{fname_base}_dF.png', dpi=200)

    plt.figure()
    plt.imshow(sort(raw))
    plt.colorbar()
    plt.title('f_raw')
    plt.savefig(f'{fname_base}_raw.png', dpi=200)

    plt.figure()
    plt.imshow(sort(spk_thr))
    plt.colorbar()
    plt.title('thresholded spikes')
    plt.savefig(f'{fname_base}_spk_thr.png', dpi=200)

    return spks, neu, raw, dF

def load_s2p_results(s2p_dir):
    F = np.load(s2p_dir + 'F.npy', allow_pickle=True)
    Fneu = np.load(s2p_dir + 'Fneu.npy', allow_pickle=True)
    spks = np.load(s2p_dir + 'spks.npy', allow_pickle=True)
    stat = np.load(s2p_dir + 'stat.npy', allow_pickle=True)
    ops =  np.load(s2p_dir + 'ops.npy', allow_pickle=True)
    ops = ops.item()
    iscell = np.load(s2p_dir + 'iscell.npy', allow_pickle=True)

    return F, Fneu, spks, stat, ops, iscell

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def low_pass_filt(data,cutoff): # cutoff in Hz
    order = 6
    fs = 30.0
    b, a = butter_lowpass(cutoff, fs, order)
    y = butter_lowpass_filter(data, cutoff, fs, order)
    return y