import numpy as np
from PIL import Image
import os
import seaborn as sb
import matplotlib.pyplot as plt
import glob
from scipy import stats
import skimage.io as skio
import xml.etree.ElementTree as ET
import pandas as pd

def load_s2p_results(s2p_fld):
    F = np.load(s2p_fld + 'F.npy', allow_pickle=True)
    Fneu = np.load(s2p_fld + 'Fneu.npy', allow_pickle=True)
    spks = np.load(s2p_fld + 'spks.npy', allow_pickle=True)
    stat = np.load(s2p_fld + 'stat.npy', allow_pickle=True)
    ops =  np.load(s2p_fld + 'ops.npy', allow_pickle=True)
    ops = ops.item()
    iscell = np.load(s2p_fld + 'iscell.npy', allow_pickle=True)
    return F, Fneu, spks, stat, ops, iscell

def ms2frames(n_ms,fps):
    n_frames = n_ms*(fps/1000)
    return int(np.round(n_frames))

def get_trial_ends(s2p_fld):
    '''For use with calculating entire session FN. Finds the ends of each recording block,
    minus any bad frames that were excluded by s2p.'''
    F, Fneu, spks, stat, ops, iscell = load_s2p_results(s2p_fld)

    bad_frames = ops['badframes']
    file_list = ops['filelist']
    bad_frame_inds = np.where(bad_frames>0)[0]

    frames_so_far = 0
    all_frames_so_far = 0
    tr_ends = np.zeros(len(file_list))
    for i,fn in enumerate(file_list):
        print('Getting trial end for: ',fn)
        img = Image.open(fn)
        frame_rng = np.array(range(all_frames_so_far,all_frames_so_far+img.n_frames))
        n_bad_frames = np.sum(np.isin(frame_rng,bad_frame_inds)) #counts all frames in this range that are also in the bad frames list
        tr_ends[i] = img.n_frames + frames_so_far - n_bad_frames
        frames_so_far  += img.n_frames - n_bad_frames
        all_frames_so_far += img.n_frames
        img.close()
    tr_ends = tr_ends.astype(int)
    print('trial ends are: /n',tr_ends)
    np.save(s2p_fld + 'trial_ends.npy',tr_ends)
    print('saved trial ends array in ' + s2p_fld)
    return tr_ends

def get_frame_times(path_to_xml):
    '''reads the Bruker xml file and extracts frame times'''
    tree = ET.parse(path_to_xml)
    root = tree.getroot()

    start_time = root[2].attrib['time']

    rel_times = []
    abs_times = []
    indices = []
    for frame in root[2].iter('Frame'):
        rel_times.append(frame.attrib['relativeTime'])
        abs_times.append(frame.attrib['absoluteTime'])
        indices.append(frame.attrib['index'])
    frame_times_dict = {'start_time': start_time,
                        'rel_times' : rel_times,
                        'abs_times' : abs_times,
                        'indices'   : indices}
    return frame_times_dict

def get_cam_frame_times_from_voltage_recording(voltage_recording_csv,xml):
    '''reads the voltage recording file recorded on the Bruker DAQ to get the
    times of the pulses sent by the camera'''
    #get day from folder title
    day = voltage_recording_csv.split('/')[-4]

    #load csv file
    voltage_recording = pd.read_csv(voltage_recording_csv)

    #load xml file
    frame_time_dict = get_frame_times(xml)
    start_time = pd.to_datetime(day + frame_time_dict['start_time'], format = '%m%d%y%H:%M:%S.%f')

    #add voltage recording times to start time
    voltage_times = pd.to_timedelta(voltage_recording['Time(ms)']/1000,unit='s')
    voltage_recording['Time Relative to Start'] = start_time + voltage_times

    #get pellet delivery times
    voltage_diff_0 = np.zeros(len(voltage_recording[' Input 0'])) #pad beginning with a zero for indexing purposes
    voltage_diff_0[1:] = np.diff(voltage_recording[' Input 0'])
    pellet_times = voltage_recording['Time Relative to Start'][voltage_diff_0>0.05]

    #get camera frame times
    voltage_diff_1 = np.zeros(len(voltage_recording[' Input 1'])) #pad beginning with a zero for indexing purposes
    voltage_diff_1[1:] = np.diff(voltage_recording[' Input 1'])
    cam_frame_times = voltage_recording['Time Relative to Start'][voltage_diff_1>0.05]
    return cam_frame_times


def threshold_oasis_output(spks_cells,th_multiplier = 4):
    '''applies a threshold to output of the oasis AR2 process; returns a spike raster and list of spike times
    optional input th_multiplier defines how many standard deviations above the mean are used for the threshold.
    Default is 4 std.'''
    spike_times = list()
    spk_raster = np.zeros(spks_cells.shape)
    for n in range(spks_cells.shape[0]):
        stdev = np.std(spks_cells[n])
        thresh = np.mean(spks_cells[n]) + th_multiplier*stdev
        spk_raster[n,:] = spks_cells[n]>thresh
        spike_times.append(np.where(spks_cells[n]>thresh)[0])
    return spk_raster, spike_times

def raster(spike_times,title='spike times'):
    '''plots a raster, given a list of spike times'''
    fig, ax = plt.subplots()
    ax.eventplot(spike_times, linelengths=1, color='black')
    ax.set_title(title)
    plt.show()

def frame2sec(n_frames,fps):
    return n_frames/fps

def load_red_ind(day_fld):
    '''loads the results of the red channel registration'''
    red_fld = glob.glob(day_fld + '/red/TSeries*')
    ind = np.load(red_fld[0] + '/red_reg_results/red_reg_ind.npy')
    return ind

def raster_cell_types(spike_times,ind,title='spike raster',colors=['black','m']):
    '''plots a raster with cell types in different colors, given spike times and the
    red registration results. default colors are exc in black and PV in magenta.'''
    fig, ax = plt.subplots()
    spike_times_ordered = []
    ind_PN = list(set(range(len(spike_times))).difference(ind[1]))
    ind_red = ind[1]
    spike_times_green = [spike_times[i] for i in ind_PN]
    spike_times_red = [spike_times[i] for i in ind_red]
    spike_times_ordered = spike_times_green + spike_times_red
    color_list = [colors[0]]*len(spike_times_green) + [colors[1]]*len(spike_times_red)
    ax.eventplot(spike_times_ordered, linelengths=1, color=color_list)
    ax.set_title(title)

def plot_fr_dist_type_specific(raster,ind,title=''):
    '''plots firing rate distribution over whole raster, separated for cell type, given 
    raster and red reg results'''
    ind_PN = np.array([i for i in range(raster.shape[0]) if i not in ind[1]])
    rates_green = np.sum(raster[ind_PN,:],axis=1)/frame2sec(raster.shape[1],30)
    rates_red = np.sum(raster[ind[1],:],axis=1)/frame2sec(raster.shape[1],30)
    fig, ax = plt.subplots(2,1)
    ax[0].hist(rates_green)
    ax[0].set_title('firing rate dist for putative excitatory'+title)
    ax[0].set_xlim(0,np.max([np.max(rates_green),np.max(rates_red)]))
    ax[1].hist(rates_red)
    ax[1].set_title('firing rate dist for PV'+ title)
    ax[1].set_xlim(0,np.max([np.max(rates_green),np.max(rates_red)]))
    return rates_green,rates_red

def plot_fr_dist(raster,title=''):
    '''plots firing rate distribution given raster'''
    rates = np.sum(raster,axis=1)/frame2sec(raster.shape[1],30)
    plt.figure()
    plt.hist(rates)
    plt.title('firing rate dist '+title)
    return rates

def plot_binned_fr_dist(day_fld, s2p_fld,bin_size):
    '''plots the distribution of peak firing rates for each cell, with red separated'''
    F, Fneu, spks, stat, ops, iscell = load_s2p_results(s2p_fld)
    spks_cells = spks[iscell[:,0].astype(bool)]
    raster, spk_times = threshold_oasis_output(spks_cells,th_multiplier=3)
    ind = load_red_ind(day_fld)
    PV_ind = ind[1]
    PN_ind = np.array([i for i in range(spks_cells.shape[0]) if i not in PV_ind])
    #can't find an equivalent of matlab histcounts in python...
    edges = np.arange(0,raster.shape[1],bin_size)
    counts_PV = np.array([np.sum(raster[PV_ind,i:i+bin_size],axis=1) for i in edges])
    counts_PN = np.array([np.sum(raster[PN_ind,i:i+bin_size],axis=1) for i in edges])
    frPV = counts_PV/bin_size
    frPN = counts_PN/bin_size
    # plt.figure()
    # plt.plot(fr[:,0],'-')
    fig,ax = plt.subplots(2,1)
    ax[0].hist(np.max(frPN,axis=0))
    ax[0].set_title('peak firing rate distribution PN; bin size ' + str(bin_size))
    ax[0].set_xlim(0,np.max([np.max(frPV.flatten()),np.max(frPN.flatten())]))
    ax[1].hist(np.max(frPV,axis=0))
    ax[1].set_title('peak firing rate distribution PV; bin size ' + str(bin_size))
    ax[1].set_xlim(0,np.max([np.max(frPV.flatten()),np.max(frPN.flatten())]))
# day_fld = '/mnt/birch/052623'
# s2p_fld = '/media/macleanlab/HDD/gwf-data/0526/suite2p/plane0/'
# bin_size = 5
# print('bin size: ',bin_size)
# plot_binned_fr_dist(day_fld, s2p_fld,bin_size)
# plt.show()

def plot_raster_cell_types(day_fld,s2p_fld, title='spike raster',colors=['black','m']):
    '''actually plots the raster with cell types colored, given the folders where results are saved'''
    ind = load_red_ind(day_fld)
    F, Fneu, spks, stat, ops, iscell = load_s2p_results(s2p_fld)
    spks_cells = spks[iscell[:,0].astype(bool)]
    raster, spk_times = threshold_oasis_output(spks_cells)
    raster_cell_types(spk_times,ind,title=title,colors=colors)
    #plot_fr_dist(raster,ind)
    # window = ms2frames(1000,30)
    # plot_binned_fr_dist(raster,window)

# day_fld = '/mnt/birch/052623'
# s2p_fld = '/media/macleanlab/HDD/gwf-data/0526/suite2p/plane0/'
# plot_raster_cell_types(day_fld,s2p_fld)
# plt.show()
def sensitivity_to_thresh_cell_type_specific(day_fld,s2p_fld,multiplier_range):
    '''sensitivity to threshold analysis with cell type delineated, in case we want to use a separate
    exclusion threshold for red/compare the PV to each other'''
    F, Fneu, spks, stat, ops, iscell = load_s2p_results(s2p_fld)
    spks_cells = spks[iscell[:,0].astype(bool)]
    F_cells = F[iscell[:,0].astype(bool)]
    ind = load_red_ind(day_fld)
    rates_green = np.zeros([len(spks_cells)-len(ind[1]),len(multiplier_range)])
    rates_red = np.zeros([len(ind[1]),len(multiplier_range)])
    ex_neuron = 16
    fig,ax = plt.subplots(len(multiplier_range),1)
    for i,th_multiplier in enumerate(multiplier_range):
        raster, spk_times = threshold_oasis_output(spks_cells,th_multiplier=th_multiplier)
        rates_green[:,i],rates_red[:,i] = plot_fr_dist_type_specific(raster,ind,title='thresh '+str(th_multiplier) + 'stdev')
        ax[i].plot(F_cells[ex_neuron,:])
        ax[i].vlines(spk_times[ex_neuron],ymin=0,ymax=0.3*np.max(F_cells[ex_neuron,:]),color='black')
        ax[i].set_title('threshold '+str(th_multiplier)+' above mean')
    diffs_green = np.diff(rates_green,axis=1)
    diffs_red = np.diff(rates_red,axis=1)
    diffs_all = np.concatenate([diffs_green,diffs_red])
    fig, ax = plt.subplots(2,1)
    ax[0].hist(diffs_green.flatten())
    ax[0].set_title('diff green')
    ax[0].plot([np.percentile(diffs_green.flatten(),10),np.percentile(diffs_green.flatten(),10)],[0,400],'-')
    ax[1].hist(diffs_red.flatten())
    ax[1].plot([np.percentile(diffs_red.flatten(),10),np.percentile(diffs_red.flatten(),10)],[0,16],'-')
    ax[1].set_title('diff red')
    where_less_green = np.where(diffs_green<np.percentile(diffs_green.flatten(),10))
    print('no. green excluded: ', np.unique(where_less_green[0]).shape)
    where_less_red = np.where(diffs_red<np.percentile(diffs_red.flatten(),10))
    print('no. red excluded: ',np.unique(where_less_red[0]).shape)
    return np.unique(where_less_green[0]), np.unique(where_less_red[0])

def sensitivity_to_thresh(day_fld,s2p_fld,multiplier_range,thresh_percent=10):
    '''calculates the sensitivity of each cell's firing rate to the threshold used. 
    Takes the folders where data are saved, and a range of std above the mean.
    Returns the indices of the cells whose firing rates change the most (top tenth percentile by default)
    when the threshold changes.'''
    F, Fneu, spks, stat, ops, iscell = load_s2p_results(s2p_fld)
    spks_cells = spks[iscell[:,0].astype(bool)]
    F_cells = F[iscell[:,0].astype(bool)]
    rates = np.zeros([len(spks_cells),len(multiplier_range)])
    for i,th_multiplier in enumerate(multiplier_range):
        raster, spk_times = threshold_oasis_output(spks_cells,th_multiplier=th_multiplier)
        rates[:,i] = plot_fr_dist(raster,title='thresh '+str(th_multiplier) + 'stdev')
    diffs = np.diff(rates,axis=1)
    plt.figure()
    plt.hist(diffs.flatten())
    plt.title('diff distribution')
    plt.plot([np.percentile(diffs.flatten(),thresh_percent),np.percentile(diffs.flatten(),thresh_percent)],[0,400],'-')
    where_less = np.where(diffs<np.percentile(diffs.flatten(),thresh_percent))
    print('no. excluded: ', np.unique(where_less[0]).shape)
    return np.unique(where_less[0])

def sensitivity_to_thresh_hal(spks,multiplier_range,thresh_percent=10):
    '''calculates the sensitivity of each cell's firing rate to the threshold used. 
    Takes the folders where data are saved, and a range of std above the mean.
    Returns the indices of the cells whose firing rates change the most (top tenth percentile by default)
    when the threshold changes.

    modified by Hal to work with my data structures'''
    spks_cells = spks # from neura
    rates = np.zeros([len(spks_cells),len(multiplier_range)])
    for i,th_multiplier in enumerate(multiplier_range):
        raster, spk_times = threshold_oasis_output(spks_cells,th_multiplier=th_multiplier)
        rates[:,i] = plot_fr_dist(raster,title='thresh '+str(th_multiplier) + 'stdev')
    diffs = np.diff(rates,axis=1)
    plt.figure()
    plt.hist(diffs.flatten())
    plt.plot([np.percentile(diffs.flatten(),thresh_percent),np.percentile(diffs.flatten(),thresh_percent)],[0,400],'-')
    plt.title('diff distribution')
    plt.show()
    where_less = np.where(diffs<np.percentile(diffs.flatten(),thresh_percent))
    print('no. excluded: ', np.unique(where_less[0]).shape)
    return diffs, np.unique(where_less[0])


# day_fld = '/mnt/birch/052623'
# s2p_fld = '/media/macleanlab/HDD/gwf-data/0526/suite2p/plane0/'
# #plot_raster_cell_types(day_fld,s2p_fld)
# green_exclude_ind, red_exclude_ind = sensitivity_to_thresh(day_fld,s2p_fld,[2,3,4,5])
# plt.show()

def confMI_mat(raster,trial_ends):
    '''from the dev branch of the snn repo. calculates the conf MI matrix from a raster.'''
    neurons = np.shape(raster)[0]
    print(neurons)
    mat = np.zeros([neurons, neurons])
    for pre in range(0, neurons):
        for post in range(0, neurons):
            if pre != post:
                mat[pre, post] = confMI(
                    raster[pre, :], raster[post, :], trial_ends
                )
                print('confMI for ',pre, ' and ', post, ' is ', mat[pre,post])
    return mat


def confMI(train_1, train_2, trial_ends):
    '''from the dev branch of the snn repo'''
    """Comput confluent mutual information between two spike trains.

    `trial_ends` lists the indices where a trial ended (causal
    discontinuities).
    """
    # the trial adjustment will need to be changed for a larger lag
    # since as-is it will only work properly with a lag of 1
    # so set it inside this function
    lag = 1

    # j-hat
    train_2_lagged = train_2[lag:]
    train_2 = train_2[:-lag]
    j_hat = np.logical_or(train_2, train_2_lagged)

    # marginal probs
    p_pre_1 = np.mean(train_1)
    p_post_1 = np.mean(j_hat)
    p_pre_0 = 1 - p_pre_1
    p_post_0 = 1 - p_post_1

    # for the overlaps to work
    train_1 = train_1[:-lag]

    # joint variables -- compute joint probabilities from these
    # after taking out the trial ends
    both_1 = np.logical_and(train_1, j_hat)
    pre_0_post_1 = np.logical_and(np.logical_not(train_1), j_hat)
    pre_1_post_0 = np.logical_and(train_1, np.logical_not(j_hat))
    both_0 = np.logical_and(np.logical_not(train_1), np.logical_not(j_hat))

    # find trial_end indices
    # if one falls in the last lag bin, get rid of it
    # (the end of the last trial is not a transition that needs to be discarded)
    trial_ends = trial_ends[trial_ends < len(train_1)]
    trial_ends_bin = np.ones(len(train_1), dtype=bool)
    trial_ends_bin[trial_ends] = False

    # now get rid of them
    # the times are aligned with train_1, so at 0 indices, the t+1 in j_hat
    # is across the causal discontinuity, this will take them out
    # could update to make [lag] indices up to the trial ends 0 indices, to
    # take all of them out for longer lags
    # (that would involve changing the lines above)
    both_1 = both_1[trial_ends_bin]
    pre_0_post_1 = pre_0_post_1[trial_ends_bin]
    pre_1_post_0 = pre_1_post_0[trial_ends_bin]
    both_0 = both_0[trial_ends_bin]

    # and compute probabilities
    p_both_1 = np.mean(both_1)
    p_pre_0_post_1 = np.mean(pre_0_post_1)
    p_pre_1_post_0 = np.mean(pre_1_post_0)
    p_both_0 = np.mean(both_0)

    # do updates -- check before dividing by zero!
    MI = 0
    if p_both_1 > 0:
        MI += p_both_1 * np.log2(p_both_1 / (p_pre_1 * p_post_1))
    if p_both_0 > 0:
        MI += p_both_0 * np.log2(p_both_0 / (p_pre_0 * p_post_0))
    if p_pre_0_post_1 > 0:
        MI += p_pre_0_post_1 * np.log2(p_pre_0_post_1 / (p_pre_0 * p_post_1))
    if p_pre_1_post_0 > 0:
        MI += p_pre_1_post_0 * np.log2(p_pre_1_post_0 / (p_pre_1 * p_post_0))

    return MI

def compute_session_FN(day_fld,s2p_fld):
    '''given folders where data is saved, computes FN with confMI for entire session.'''
    F, Fneu, spks, stat, ops, iscell = load_s2p_results(s2p_fld)
    spks_cells = spks[iscell[:,0].astype(bool)]
    raster, spk_times = threshold_oasis_output(spks_cells)
    #exclude neurons with high sensitivity to threshold
    green_exclude_ind, red_exclude_ind = sensitivity_to_thresh(day_fld,s2p_fld,[2,3,4,5])
    exclude_ind = np.concatenate([green_exclude_ind,red_exclude_ind])
    raster = np.delete(raster,exclude_ind,0)
    #plot_raster(spk_times)
    if not os.path.isfile(s2p_fld +'/trial_ends.npy'):
        tr_ends = get_trial_ends(s2p_fld)
    else:
        tr_ends = np.load(s2p_fld +'/trial_ends.npy')
    FN = confMI_mat(raster.astype(int),tr_ends)
    plt.figure()
    sb.heatmap(FN,vmin=0, vmax=0.00001)
    plt.figure()
    plt.hist(np.log(nonzero(FN.flatten())))
    plt.xlim([0,0.002])
    plt.show()
    np.save(s2p_fld,'FN_conMI_session.npy')
    return FN

def compute_reach_FN(day_fld, s2p_fld,reach_starts,reach_ends,force_compute=False):
    '''given folders where data is saved, computes FN with confMI for only reaches.
    Takes reach start/ends in terms of calcium frames.'''
    F, Fneu, spks, stat, ops, iscell = load_s2p_results(s2p_fld)
    spks_cells = spks[iscell[:,0].astype(bool)]
    raster, spk_times = threshold_oasis_output(spks_cells)
    #exclude neurons with high sensitivity to threshold
    #IMPORTANT: makes sure this is the 
    exclude_ind = sensitivity_to_thresh(day_fld,s2p_fld,[2,3,4,5])
    raster = np.delete(raster,exclude_ind,0)
    #get one long spike train with all the spikes during reaches
    reach_ind = [np.arange(int(reach_starts[i]),int(reach_ends[i])) for i in range(len(reach_starts))]
    reach_ind = [item for sublist in reach_ind for item in sublist]
    raster_reaches = raster[:,reach_ind]
    if not os.path.exists(s2p_fld + '/FN_conMI_reaches.npy') or force_compute:
        FN = confMI_mat(raster_reaches.astype(int),np.array(reach_ends).astype(int))
        np.save(s2p_fld+'/FN_conMI_reaches.npy',FN)
    else:
        FN = np.load(s2p_fld + '/FN_conMI_reaches.npy')

    plt.figure()
    sb.heatmap(FN)
    # plt.figure()
    # plt.hist(np.log(np.nonzero(FN.flatten())))
    # plt.show()
    return FN


# s2p_fld = '/media/macleanlab/HDD/gwf-data/0515/suite2p/plane0/'
# raster_cell_types(s2p_fld)

