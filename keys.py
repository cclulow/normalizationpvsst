### 030124 some way to organize data for a day.
import os
import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel
from scipy.ndimage import gaussian_filter
from utils import get_trial_idxs, low_pass_filt
from suite2p.extraction.dcnv import preprocess
from er_est import r2er_n2n_split
# don't like importing from there... these should be saved in initial processing

# need to have better way to organize this
std_f_before = 30
std_f_after = 60

class NeuralData():
    def __init__(self, mouse, date, base_name='/home/halrockwell/data_proc/data/',
            alt_spikes=False, thresh_spikes=False, resp_start=30, resp_end=60,
            grey_start=2, grey_end=28, use_l=-1, smooth=0, sure=False):
        '''
        docs
        '''
        # in case you want to reference it later
        self.mouse = mouse
        self.date = date
        self.alt_spikes = alt_spikes
        self.thresh_spikes = thresh_spikes # should this really be an argument here?
        spike_thresh = 1e-8 # effectively 0
        # params for defining start/end of response
        # to allow wiggle room for spike timing errors etc.
        self.resp_start = resp_start
        self.resp_end = resp_end
        self.grey_start = grey_start
        self.grey_end = grey_end

        # locate where all the data is stored/will be stored
        self.base_dir = f'{base_name}{mouse}/{date}/'
        if not os.path.isdir(self.base_dir):
            print(f'base folder {self.base_dir} does not exist, get the damn data')
            # raise some kind of error maybe for these
        self.facecam_dir = f'{self.base_dir}facecam/'
        if not os.path.isdir(self.facecam_dir):
            print(f'facecam folder {self.facecam_dir} not present, facecam files not copied over?')
        self.motion = None
        self.motSVD = None
        self.movSVD = None # maybe load these in if they're saved

        # assuming run_suite2p already run
        self.s2p_dir = f'{self.base_dir}runs/suite2p/plane0/'
        if not os.path.isdir(self.s2p_dir):
            print(f'suite2p folder {self.s2p_dir} does not exist, run suite2p')

        # assuming proc dir already created in process_data or other script
        self.proc_dir = f'{self.base_dir}proc/'
        if not os.path.isdir(self.proc_dir):
            print(f'proc folder {self.proc_dir} does not exist, run processing first')

        # assuming get_red_cells or similar already run
        # handle old method of red registration (G's) or new (mine)
        # this is a bit sloppy, may break on some cases
        self.old_red = True
        if os.path.exists(f'{self.proc_dir}red_labels.npy'):
            self.no_red = False
            if not sure:
                self.red_labels = np.load(f'{self.proc_dir}red_labels.npy')
            else:
                self.red_labels = np.load(f'{self.proc_dir}sure_red_labels.npy')
            self.old_red = False
        #self.red_reg_dir = f'{self.base_dir}red_reg_results/'
        #self.no_red = False
        #if not os.path.isdir(self.red_reg_dir):
        #   print(f'red reg folder {self.red_reg_dir} does not exist, run red channel reg first')
        #    self.no_red = True

        #if not self.no_red and self.old_red:
        #    self.red_reg_ind = np.load(self.red_reg_dir + 'red_reg_ind.npy')[1]
        #    self.red_all = np.load(self.red_reg_dir + 'A_red_all.npy')


        # now start loading in data
        # can't imagine looking at non-cell ROIs
        self.iscell = np.load(self.s2p_dir + 'iscell.npy')[:, 0].astype(bool)
        self.F = np.load(self.s2p_dir + 'F.npy')[self.iscell]
        self.Fneu = np.load(self.s2p_dir + 'Fneu.npy')[self.iscell]
        # 111224 break old alt_spikes functionality I don't use
        # and allow for original s2p spikes missing
        # to use partially-lost processed data
        # also breaks thresh_spikes which I don't use anyway
        spks_file = self.s2p_dir + 'spks.npy'
        if os.path.exists(spks_file):
            self.spks = np.load(spks_file)[self.iscell]
        # if not alt_spikes:
        #     self.spks = np.load(self.s2p_dir + 'spks.npy')[self.iscell]
        # elif alt_spikes:
        #     self.spks = np.load(self.s2p_dir + 'alt_spks.npy')[self.iscell]
        # if self.thresh_spikes:
        #     self.spks[self.spks < spike_thresh] = 0
        #     self.spks[self.spks >= spike_thresh] = 1
        # not sure I need ops but stat contains ROI positions -- important
        # ops is a dict when item is called to take it out
        if os.path.exists(self.s2p_dir + 'ops.npy'):
            self.ops = np.load(self.s2p_dir + 'ops.npy', allow_pickle=True).item()
        else:
            print("ops file doesn't exist... did reg_metrics delete it? :(")
        # stat is an nparray of dicts, one for each ROI
        self.stat = np.load(self.s2p_dir + 'stat.npy', allow_pickle=True)[self.iscell]

        # now, processed data -- trials etc.
        if not alt_spikes:
            self.spks_tr = np.load(self.proc_dir + 'spks.npy')
        elif alt_spikes:
            self.spks_tr = np.load(self.proc_dir + 'alt_spks.npy')
        if self.thresh_spikes:
            self.spks_tr[self.spks_tr < spike_thresh] = 0
            self.spks_tr[self.spks_tr >= spike_thresh] = 1
        self.dF_tr = np.load(self.proc_dir + 'dF.npy') # need to doublecheck this is right
        self.stim_info = np.load(self.proc_dir + 'stim_info.npy')

        # if l0, l1 spikes exist, load them and proc them trial-by-trial
        # also get indices for that loaded just in case
        if os.path.exists(self.proc_dir + 'trial_idxs.npy'):
            self.trial_idxs = np.load(self.proc_dir + 'trial_idxs.npy')
            if os.path.exists(self.proc_dir + 'l0_spks.npy') and use_l == 0:
                print('loading l0 spikes')
                self.l0_spks = np.load(self.proc_dir + 'l0_spks.npy')[self.iscell]
                self.l0_spks_tr = np.array([self.l0_spks[:, t-std_f_before:t+std_f_after] for t in self.trial_idxs])
                self.l0_spks_tr = np.transpose(self.l0_spks_tr, (1, 0 ,2))
            if os.path.exists(self.proc_dir + 'l1_spks.npy') and use_l == 1:
                print('loading l1 spikes')
                self.l1_spks = np.load(self.proc_dir + 'l1_spks.npy')[self.iscell]
                if smooth > 0:
                    self.l1_spks = gaussian_filter(self.l1_spks, sigma=(0, smooth))
                self.l1_spks_tr = np.array([self.l1_spks[:, t-std_f_before:t+std_f_after] for t in self.trial_idxs])
                self.l1_spks_tr = np.transpose(self.l1_spks_tr, (1, 0 ,2))
            if os.path.exists(self.proc_dir + 'fast_l1_spks.npy') and use_l == 'fast': # TODO this is busted actually
                print('loading fast l1 spikes')
                self.f_l1_spks = np.load(self.proc_dir + 'fast_l1_spks.npy')[self.iscell]
                if smooth > 0:
                    self.f_l1_spks = gaussian_filter(self.f_l1_spks, sigma=(0, smooth))

                self.f_l1_spks_tr = np.array([self.f_l1_spks[:, t-std_f_before:t+std_f_after] for t in self.trial_idxs])
                self.f_l1_spks_tr = np.transpose(self.f_l1_spks_tr, (1, 0 ,2))

            if os.path.exists(self.proc_dir + 'g_l1_spks.npy') and use_l == 'g':
                print('loading G l1 spikes')
                self.g_l1_spks = np.load(self.proc_dir + 'g_l1_spks.npy')[self.iscell]
                if smooth > 0:
                    self.g_l1_spks = gaussian_filter(self.g_l1_spks, sigma=(0, smooth))

                self.g_l1_spks_tr = np.array([self.g_l1_spks[:, t-std_f_before:t+std_f_after] for t in self.trial_idxs])
                self.g_l1_spks_tr = np.transpose(self.g_l1_spks_tr, (1, 0 ,2))
            
            # 072924 'new' spikes with faster time constants
            # also 'denoised' traces
            if os.path.exists(self.proc_dir + 'new_spks.npy'): # do this anyway to have denoised traces
                print('loading new spikes')
                if use_l == 'new':
                    self.new_spks = np.load(self.proc_dir + 'new_spks.npy') # only saved for cells
                    if smooth > 0:
                        self.new_spks = gaussian_filter(self.new_spks, sigma=(0, smooth))

                    self.new_spks_tr = np.array([self.new_spks[:, t-std_f_before:t+std_f_after] for t in self.trial_idxs])
                    self.new_spks_tr = np.transpose(self.new_spks_tr, (1, 0, 2))
                # denoised... wanna use this for new SNR function maybe
                self.new_denoised = np.load(self.proc_dir + 'new_denoised.npy') # no need to trial split
            # 103024 cascade spikes. no denoised version. may want to compute and save cascade noise levels...
            if os.path.exists(self.proc_dir + 'cascade_spks.npy') and use_l =='cascade': # do this anyway to have denoised traces
                print('loading cascade spikes')
                self.cascade_spks = np.load(self.proc_dir + 'cascade_spks.npy') # only saved for cells
                if smooth > 0:
                    self.cascade_spks = gaussian_filter(self.cascade_spks, sigma=(0, smooth)) # should never do this

                self.cascade_spks_tr = np.array([self.cascade_spks[:, t-std_f_before:t+std_f_after] for t in self.trial_idxs])
                self.cascade_spks_tr = np.transpose(self.cascade_spks_tr, (1, 0, 2))
            

        # hacky, but allows me to sub in spikes without options elsewhere
        if use_l == 0:
            self.spks = self.l0_spks
            self.spks_tr = self.l0_spks_tr
        elif use_l == 1:
            self.spks = self.l1_spks
            self.spks_tr = self.l1_spks_tr
        elif use_l == 2:
            self.spks = self.f_l1_spks
            self.spks_tr = self.f_l1_spks_tr
        elif use_l == 'g':
            self.spks = self.g_l1_spks
            self.spks_tr = self.g_l1_spks_tr
        elif use_l == 'new':
            self.spks = self.new_spks
            self.spks_tr = self.new_spks_tr
        elif use_l == 'cascade':
            self.spks = self.cascade_spks
            self.spks_tr = self.cascade_spks_tr
    
    def get_dF(self, alpha=1.0, win_baseline=120, sig_baseline=10):
        '''
        return dF/F (compute it from scratch)
        '''
        f_minus = self.F - alpha * self.Fneu
        dF = preprocess(f_minus, baseline='maximin', win_baseline=win_baseline, sig_baseline=sig_baseline,
                        fs=30.01)
        return dF

    def get_snrs(self, alpha=1.0, win_baseline=120, sig_baseline=10,
                 cutoff=5):
        '''
        measure of SNR (normalized peak dF/F after (default) 5Hz lowpass filtering)
        for each cell. recomputes dF/F but it's pretty fast
        '''
        dF = self.get_dF(alpha, win_baseline, sig_baseline)
        dF_filt = np.array([low_pass_filt(dF[neu], cutoff) for neu in range(len(dF))])
        dF_norm = dF_filt / np.std(dF_filt, 1, keepdims=True)
        peaks = np.max(dF_norm, axis=1)

        return peaks
    
    def denoised_snr(self, alpha=1.0, win_baseline=120, sig_baseline=10):
        '''
        073124 alternative SNR metric based on R2 of 'denoised' trace
        needs that to have been loaded with 'new' spikes
        '''
        denoised = self.new_denoised
        dF = self.get_dF(alpha=alpha, win_baseline=win_baseline, sig_baseline=sig_baseline)
        leftover_var = np.mean((dF - denoised) ** 2, axis=1)
        base_var = np.var(dF, axis=1)
        r2s = 1 - (leftover_var / base_var)

        return r2s

    def get_trial_responses(self, dF=False, l=None, only_idxs=None):
        '''
        get stimulus-length list of trial-by-trial spiking responses

        also returns the unique stimulus values
        '''
        unique_stim = np.unique(self.stim_info, axis=1)
        relevant_resps = self.dF_tr if dF else self.l0_spks_tr if l == 0 else self.l1_spks_tr if l == 1 else self.spks_tr
        stim_info = np.copy(self.stim_info)

        if only_idxs is not None:
            # indices of trials to consider
            # pretty sure if this contains none of a particular stimulus
            # it'll still be in the list returned, just empty... which we want
            # tho it would later be dropped out in norm_trial_responses... so bit confusing
            stim_info = stim_info[:, only_idxs]
            unique_stim = np.unique(stim_info, axis=1)
            print(relevant_resps.shape)
            print(only_idxs.shape)
            relevant_resps = relevant_resps[:, only_idxs]
            print(relevant_resps.shape)

        stim_responses = []
        for stim_idx in range(unique_stim.shape[1]):
            # matching orientation (first row) and plaid status (second row)
            idxs = np.logical_and(stim_info[0] == unique_stim[0, stim_idx],
                    stim_info[1] == unique_stim[1, stim_idx])
            resps = relevant_resps[:, idxs]
            stim_responses.append(resps)
        
        #print(unique_stim.T)
        #print(unique_stim.T.shape)

        return stim_responses, unique_stim.T

    def get_norm_trial_responses(self, dF=False, l=None, only_idxs=None):
        '''
        get trial-by-trial responses for the normalization-related stimuli only

        also returns the unique stimulus values
        '''
        responses, stimuli = self.get_trial_responses(dF=dF, l=l, only_idxs=only_idxs)

        # now filter for the main orientations
        #main_oris = [0, 45, 90, 135] # shouldn't be hardcoded
        response_cutoff = 50 # should have >50 trials... hardcoded slightly less stupidly
        if only_idxs is not None:
            # can't trust number of responses anymore...
            # just not doing that on feb data :/
            response_cutoff = 0
        filtered_resps = []
        filtered_stim = []
        for stim_idx in range(stimuli.shape[0]):
            ori, plaid = stimuli[stim_idx]
            # response cutoff thing doesn't work for limited indices
            # was a shitty hack anyway. do things this way instead
            #if len(responses[stim_idx][0]) > response_cutoff:
            if plaid == 1 or 1 in stimuli[stimuli[:, 0] == ori] or 1 in stimuli[stimuli[:, 0] == ori - 90]:
                filtered_resps.append(responses[stim_idx])
                filtered_stim.append(stimuli[stim_idx])

        return filtered_resps, np.array(filtered_stim)
    
    def get_osis(self, dF=False, l=None, only_idxs=None, circvar=True):
        '''
        orientation selectivity index from mazurek et al 2014,
        circular variance based if circvar, else simpler index
        based on max, min responses to gratings, to compare with other papers
        '''
        resps, stims = self.get_trial_responses(dF=dF, l=l, only_idxs=only_idxs)
        # ends up N x C with C conditions, N neurons
        ori_resps = np.array([resp[:, :, self.resp_start:self.resp_end].mean((1, 2)) for \
                     resp, stim in zip(resps, stims) if stim[1] == 0]).T
        oris = stims[stims[:, 1] == 0, 0]
        
        if circvar:
            # compute l_ori as in mazurek 2014
            top = ori_resps.dot(np.exp(2 * 1j * oris)) # 1j is i
            bot = ori_resps.sum(1)
            l_ori = np.abs(top / bot)
        else:
            # OSI from niell 2008 and other papers, (A-B)/(A+B) for pref grating and orthogonal
            _, pref_ori, _, nonpref_ori = self.normalization_indices(only_idxs=only_idxs, return_prefs=True, return_nonpref=True)
            all_resps = np.array([resp[:, :, self.resp_start:self.resp_end].mean((1, 2)) for \
                resp, stim in zip(resps, stims)]) # not just oris as above
            pref_resps = np.diag(all_resps[pref_ori])
            nonpref_resps = np.diag(all_resps[nonpref_ori])
            l_ori = (pref_resps - nonpref_resps) / (pref_resps + nonpref_resps)

        return l_ori

    def signal_corrs(self, vis_resp_only=False, easy=False, dF=False, l=None,
                     use_er_est=False, ori_only=False):
        '''
        signal corrs raw or from pospilis 2022 paper
        '''
        resps, stims = self.get_trial_responses(dF=dF, l=l) # don't want low-trial stimuli
        if vis_resp_only: # compatibility with other funcs I guess
            idxs = self.vis_responsive_cells(easy=easy)
            resps = [resp[idxs] for resp, stim in zip(resps, stims) if stim[1] == 0] # oris only.. why is this here? TODO
        
        if not use_er_est:
            # ends up N x C with C conditions, N neurons
            avg_resps = np.array([resp[:, :, self.resp_start:self.resp_end].mean((1, 2)) for \
                        resp in resps]).T
            if ori_only:
                avg_resps = avg_resps[:, stims[:, 1] == 0] #  leave out plaids -- TODO maybe implement for er_est too
            signal_corrs = np.corrcoef(avg_resps) # that simple?
        else:
            # er_est want n_neurons x n_trials x n_stimuli
            # could experiment doing this with just orientation stimuli
            trial_resps = np.array([resp[:, :, self.resp_start:self.resp_end].mean(2) for resp in resps])
            trial_resps = np.transpose(trial_resps, (1, 2, 0))
            # need to do this pair-by-pair to make it pairwise, it seems
            signal_corrs = np.zeros((len(trial_resps), len(trial_resps)))
            for n in range(len(trial_resps)):
                # can do col-by-col at least
                signal_corrs[n] = r2er_n2n_split(np.tile(trial_resps[n], (len(trial_resps), 1, 1)), trial_resps)
            
            # unbiased estimator so can be <0 or >1, they suggest thresholding
            signal_corrs[signal_corrs > 1] = 1
            signal_corrs[signal_corrs < 0] = 0
            # I guess really these aren't corrs, but r2 values at this point... hm. would like a signed version, might be simple
            # TODO take sqrts and sign based on normal corr sign? even worth taking sqrts?
            # TODO hack this to work with uneven stimulus numbers in february data

        return signal_corrs

    def vis_responsive_cells(self, thresh=0.1, easy=False, only_idxs=None):
        '''
        return boolean index of visually responsive cells

        with mean response at least thresh% higher to some main
        stimulus than preceding grey

        if easy, then also statistical test for overall response
        to stimulus. if not, then needs to pass that for some
        particular plaid
        '''
        #resps, stimuli = self.get_norm_trial_responses()
        resps, stimuli = self.get_trial_responses(only_idxs=only_idxs)
        # not up to 30 to allow for some misplacement of spikes
        greys = [resp[:, :, self.grey_start:self.grey_end].mean(2) for resp in resps]
        stims = [resp[:, :, self.resp_start:self.resp_end].mean(2) for resp in resps]
        avg_greys = np.array([resp.mean(1) for resp in greys])
        avg_stim = np.array([resp.mean(1) for resp in stims])

        # check for at least 10% increase to some stimulus on avg
        # vast majority of cells pass this
        any_increases = []
        for stim in range(avg_greys.shape[0]):
            increases = (avg_stim[stim] - avg_greys[stim]) / avg_greys[stim]
            any_increases.append(increases)

        any_increases = np.any(np.array(any_increases) > thresh, axis=0)

        # test for statistically significant response to some grating
        # fewer cells pass this
        # too few!
        any_sig_mods = []
        n_oris = np.sum(stimuli[1] == 0)
        if not easy:
            for stim in range(stimuli.shape[1]):
                if stimuli[1, stim] == 0:
                    # not a plaid, consider
                    t_res = ttest_rel(stims[stim], greys[stim], alternative='greater', axis=1)
                    any_sig_mods.append(t_res.pvalue * n_oris < 0.05)
            any_sig_mods = np.any(np.array(any_sig_mods), axis=0)
        else:
            # combine all stimuli together
            # many more cells pass this... but should they? dubious
            all_stim_resps = np.concatenate(stims, axis=1)
            all_grey_resps = np.concatenate(greys, axis=1)
            t_res = ttest_rel(all_stim_resps, all_grey_resps, alternative='greater', axis=1)
            # misleadingly named here
            any_sig_mods = t_res.pvalue < 0.05

        return np.logical_and(any_increases, any_sig_mods)

    def noise_correlations(self, vis_resp_only=False, easy=False, logs=False,
                           grand_corr=False, only_idxs=None):
        '''
        return noise correlations between cells, separately for each
        stimulus. also list of corresponding stimulus identities
        '''
        resps, stimuli = self.get_norm_trial_responses(only_idxs=only_idxs)
        if vis_resp_only:
            idxs = self.vis_responsive_cells(easy=easy)
            resps = [resp[idxs] for resp in resps]
            if logs:
                resps = [np.log(resp + 0.01) for resp in resps]

        # compute noise correlations for each stimulus
        noise_corrs = np.empty((len(resps), len(resps[0]), len(resps[0])))
        for i, resp in enumerate(resps):
            # neurons by trials by time
            avg_over_pres = resp[:, :, self.resp_start:self.resp_end].mean(2)
            noise_corrs[i] = np.corrcoef(avg_over_pres)
        
        # separately compute all-stimulus corrs if desired
        # by subtracting mean response to each stimulus... is this done?
        # it could be...
        if grand_corr:
            trial_avgs = [resp[:, :, self.resp_start:self.resp_end].mean((1, 2))
                                   for resp in resps]
            avg_sub_resps = [resp[:, :, self.resp_start:self.resp_end].mean(2) - avg.reshape(-1, 1)
                             for resp, avg in zip(resps, trial_avgs)]
            avg_sub_resps = np.concatenate(avg_sub_resps, axis=1)
            print(avg_sub_resps.shape)
            grand_corrs = np.corrcoef(avg_sub_resps)
            return noise_corrs, grand_corrs
        
        return noise_corrs

    def spatial_locations(self, vis_resp_only=False, easy=False):
        '''
        return y, x positions for each cell
        '''
        pos = np.array([cell['med'] for cell in self.stat])
        if vis_resp_only:
            pos = pos[self.vis_responsive_cells(easy=easy)]

        return pos

    def normalization_indices(self, vis_resp_only=False, return_prefs=False, return_nonpref=False,
                              only_idxs=None, sub_spont=False):
        '''
        return normalization indices for each cell
        '''
        resps, stimuli = self.get_norm_trial_responses(only_idxs=only_idxs)
        #print(stimuli)
        #print(stimuli.shape)
        spont_act = self.spks_tr[:, :, (self.resp_start // 3):(2 * (self.resp_start // 3))].mean((1, 2))
        if vis_resp_only:
            idxs = self.vis_responsive_cells()
            resps = [resp[idxs] for resp in resps]
            spont_act = spont_act[idxs]
        # ends up stimuli x neurons x time
        avg_resps = np.array([resp.mean(1) for resp in resps])
        ori_resps = avg_resps[stimuli[:, 1] == 0]
        plaid_resps = avg_resps[stimuli[:, 1] == 1]

        # for each neuron, get preferred orientation
        # null is assumed opposite
        # plaid corresponds to orientation + null
        indices = np.empty(avg_resps.shape[1:2])
        pref_oris = np.empty(avg_resps.shape[1:2], dtype=int)
        pref_plaids = np.empty(avg_resps.shape[1:2], dtype=int)
        nonpref_oris = np.empty(avg_resps.shape[1:2], dtype=int)
        for neu in range(avg_resps.shape[1]):
            neu_ori_resps = ori_resps[:, neu, self.resp_start:self.resp_end].mean(1)
            pref_ori = np.argmax(neu_ori_resps)
            # this assumes my 4 orientations, 2 plaids setup in order
            #if pref_ori == 0 or pref_ori == 2:
            #    pref_plaid = 0
            #else:
            #    pref_plaid = 1
            pref_plaid = np.where(stimuli[stimuli[:, 1] == 1, 0] == stimuli[stimuli[:, 1] == 0, 0][pref_ori])[0]
            if len(pref_plaid) == 0:
                # for oris past 
                pref_plaid = np.where(stimuli[stimuli[:, 1] == 1, 0] == (stimuli[stimuli[:, 1] == 0, 0][pref_ori] - 90))[0]
            #nonpref_ori = (pref_ori + 2) % 4
            pref_plaid = pref_plaid[0] # .item() essentially
            nonpref_ori = np.argmax(stimuli[stimuli[:, 1] == 0, 0] == (stimuli[stimuli[:, 1] == 0, 0][pref_ori] + 90) % 180)
            #nonpref_ori = (stimuli[pref_ori, 0] + 90) % 180 # correct?

            # now can extract responses, compute NI
            pref_resp = ori_resps[pref_ori, neu, self.resp_start:self.resp_end].mean()
            nonpref_resp = ori_resps[nonpref_ori, neu, self.resp_start:self.resp_end].mean()
            plaid_resp = plaid_resps[pref_plaid, neu, self.resp_start:self.resp_end].mean()
            # subtract spontaneous activity -- grey avged. if required. apparently most papers do this
            if sub_spont:
                #spont_act = np.average([ori_resps[:, neu, self.resp_start // 3:2 * self.resp_start // 3].mean(), 
                #                        plaid_resps[:, neu, self.resp_start // 3:2 * self.resp_start // 3].mean()],
                #                       weights = [2, 1]) # twice as many oris as plaids -- exactly for non-feb data, basically for that
                pref_resp = pref_resp - spont_act[neu]
                nonpref_resp = nonpref_resp - spont_act[neu]
                plaid_resp = plaid_resp - spont_act[neu]

            
            indices[neu] = (pref_resp + nonpref_resp - plaid_resp) / (
                    pref_resp + nonpref_resp + plaid_resp)
            indices[neu] = np.clip(indices[neu], -1, 1) # indices can fall out of -1, 1 if some resps are lower than spontaneous
            # these being the indices into the resp/stim, in order
            # to use them to index right into corrs etc.
            # should probably be its own function... refactor at some point TODO
            pref_oris[neu] = np.where(stimuli[:, 1] == 0)[0][pref_ori]
            pref_plaids[neu] = np.where(stimuli[:, 1] == 1)[0][pref_plaid]
            nonpref_oris[neu] = np.where(stimuli[:, 1] == 0)[0][nonpref_ori]
            #pref_plaids[neu] = pref_plaid # hack for now TODO drop this

        if return_prefs:
            if return_nonpref:
                return indices, pref_oris, pref_plaids, nonpref_oris
            return indices, pref_oris, pref_plaids # might as well since I computed them here
        return indices

    def red_field(self, sigma=60, vis_resp_only=False):
        '''
        compute a spatial field of red cell density... centroids
        of red cells (not just colabeled) convolved with a gaussian
        filter

        return the field and strength of each gcamp cell in it
        '''
        # first compute the centroids from the A_red_all img
        # which is set up as y, x, ROI dimensions
        # TODO fix for new red method
        # tho it doesn't make much sense for new method, which only gets colabeled cells
        y, x, idx = np.where(self.red_all)
        centroids = np.empty((np.max(idx)+1, 2))
        for i in np.unique(idx):
            centroids[i] = [np.median(y[idx == i]), np.median(x[idx == i])]

        # now turn those into an image
        seed_field = np.zeros((512, 512)) # hardcoding resolution!
        for c in centroids.astype(int):
            seed_field[c[0], c[1]] = 1

        # and convolve with Gaussian
        field = gaussian_filter(seed_field, sigma=(sigma, sigma))

        # then compute strengths of each cell
        locs = self.spatial_locations(vis_resp_only)
        strengths = np.array([field[loc[0], loc[1]] for loc in locs])

        return field, strengths

    def get_roi_pics(self, side_pix=30, vis_resp_only=False, given_red=None):
        '''
        return list of pixels around each (cell) ROI, for
        visualizing. little images basically. side_pix each
        side around ROI centroid
        '''
        # idx of cells to get
        if vis_resp_only:
            idxs = np.where(self.vis_responsive_cells())[0]
        else:
            idxs = list(range(len(self.stat)))

        # get images to consider: reference and red channel
        if given_red is None:
            red_img = tifffile.imread(self.red_reg_dir + 'all_contours.tiff')[:,:,0]
            red_img = red_img / (red_img.max() * 1.5) # empirically looks better
        else:
            red_img = given_red / (given_red.max())
        green_img = self.ops['refImg']
        green_img = green_img / (green_img.max() / 1.5)

        # loop thru once to accumulate ROI masks
        masks = np.zeros((512, 512))
        for idx in idxs:
            y, x = self.stat[idx]['med']
            masks[self.stat[idx]['ypix'], self.stat[idx]['xpix']] = self.stat[idx]['lam'] / self.stat[idx]['lam'].max()
        # combine red, green and ROI in blue
        full_img = np.stack([red_img, green_img, masks], axis=-1)

        # now loop again to select area around each ROI
        pic_list = []
        for idx in idxs:
            y, x = self.stat[idx]['med']
            # not always square because of img edges
            min_y_end = max(0, y-side_pix)
            max_y_end = min(512, y+side_pix)
            min_x_end = max(0, x-side_pix)
            max_x_end = min(512, x+side_pix)

            roi_pic = full_img[min_y_end:max_y_end, min_x_end:max_x_end]

            # normalize so it doesn't look like shit
            #roi_pic = (roi_pic * 255).astype(np.uint8)
            #conv_pic = cv2.cvtColor(roi_pic, cv2.COLOR_RGB2YCrCb)
            #conv_pic[:, :, 0] = cv2.equalizeHist(conv_pic[:, :, 0])
            #roi_pic = cv2.cvtColor(conv_pic, cv2.COLOR_YCrCb2RGB)
            roi_pic = roi_pic / roi_pic.max((0, 1), keepdims=True)

            pic_list.append(roi_pic)

        return pic_list

    def characterize_neurons(self, neu_list, save_dir, tagline, given_red=None):
        '''
        make characteristic plots for a list of neurons. these
        include a tuning curve (with normalization idx listed),
        a sample bit of image around the ROI, and an example
        activity trace

        as first done in data_explo.py
        neu_list can be bools or integer idxs
        '''
        # just convert to idxs. this probably works
        if neu_list.dtype.name == 'bool': # really???
            neu_list = np.where(neu_list)[0]

        # get everything needed
        indices = self.normalization_indices()
        rois = self.get_roi_pics(given_red=given_red)
        # response processing a bit hardcoded :/
        resps, stim = self.get_trial_responses()
        avg_resps = np.array([resp[:,:,self.resp_start:self.resp_end].mean((1, 2)) for resp in resps])
        ori_resps = avg_resps[stim[:, 1] == 0]
        p0_resps = avg_resps[np.logical_and(stim[:, 1] == 1, stim[:, 0] == 0)]
        p45_resps = avg_resps[np.logical_and(stim[:, 1] == 1, stim[:, 0] == 45)]
        p22_resps = avg_resps[np.logical_and(stim[:, 1] == 1, stim[:, 0] == 22.5)]
        p67_resps = avg_resps[np.logical_and(stim[:, 1] == 1, stim[:, 0] == 67.5)]

        # look thru neurons in list, make figure for each one
        for neu in neu_list:
            # figsize makes fig nice and large so saves high-res
            # gridspec makes middle panel smaller so left/right are wide enough
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,9),
                    gridspec_kw={'width_ratios':[3, 1, 3]})

            # first plot tuning curve on left
            ax1.plot(stim[stim[:, 1] == 0, 0], ori_resps[:, neu], color='black', label='gratings')
            ax1.hlines(p0_resps[:, neu], 0, 167.5, label='0-90 plaid', color='red')
            ax1.hlines(p45_resps[:, neu], 0, 167.5, label='45-135 plaid', color='green')
            ax1.hlines(p22_resps[:, neu], 0, 167.5, label='22.5-112.5 plaid', color='blue')
            ax1.hlines(p67_resps[:, neu], 0, 167.5, label='67.5-157.5 plaid', color='purple')
            
            ax1.set_xlabel('grating orientation (degrees)')
            ax1.set_ylabel('inferred spiking response (arbitrary units)')
            ax1.legend()
            ax1.set_title(f'tuning curve, NI={indices[neu]:.3f}')

            # then, roi
            ax2.imshow(rois[neu])
            if self.old_red:
                red = neu in self.red_reg_ind
            else:
                red = self.red_labels[neu]
            ax2.set_title(f'ROI in blue, red cell? {red}')

            # annnd example trace around some event
            # fluorescence trace not baseline-corrected but it's fine
            neu_trace = self.F[neu] - self.Fneu[neu] * 0.7
            spk_trace = self.spks[neu]
            high_percentile = np.percentile(neu_trace, 99.5)
            idx_around = np.argmax(neu_trace > high_percentile)
            min_idx = max(0, idx_around - 200)
            max_idx = min(len(neu_trace), idx_around + 200)
            time = np.array(list(range(max_idx - min_idx))) / 30
            ax3.plot(time, neu_trace[min_idx:max_idx], color='blue')
            ax3.set_xlabel('time (s)')
            ax3.set_ylabel('F - Fneu')
            ax3.set_title('example activity trace around 98th pctle')
            ax3.tick_params(axis='y', labelcolor='blue')
            # with spikes on alternative y axis
            ax4 = ax3.twinx()
            ax4.set_ylabel('inferred spikes (arbitrary units)')
            ax4.plot(time, spk_trace[min_idx:max_idx], color='black')
            ax4.tick_params(axis='y', labelcolor='black')

            fig.tight_layout() # improves formatting?

            # finally title and save to specified location
            plt.suptitle(f'{self.mouse} {self.date} neu {neu} ({tagline}) characterizations')
            plt.savefig(save_dir + f'{self.mouse}_{self.date}_{neu}_{tagline}_char.png')
            plt.close()
    
    def get_facecam_traces(self, num_pcs=20):
        '''
        load in and return facemap-processed traces. save them to
        self as well (but don't load in on object loading to avoid slowing that down)
        for now, return face motion, the 500 motion SVDs, and the 500 movie SVDs
        '''
        # kind of stupid basic error handling. need to have facecam copied, facemap run
        self.facemap_file = self.base_dir + 'face_vid_proc.npy' # hardcode this? for now...
        if not os.path.isdir(self.facecam_dir) or not os.path.exists(self.facemap_file):
            print('either facecam dir or facemap file missing, make sure everything is in place')
            return None

        # if this has run before, just use those
        # TODO this won't work if you want to rerun with diff num_pcs
        if self.motion is not None and self.motSVD is not None and self.movSVD is not None:
            return self.motion, self.motSVD, self.movSVD

        # deal with the february sessions with shorter recordings
        if self.date.startswith('02') and self.date.endswith('24'):
            imgs_per_block = 6100
        else:
            imgs_per_block = 6340
        imgs_total = imgs_per_block * 8

        # otherwise load in file, keep around for later use
        self.facemap_data = np.load(self.facemap_file, allow_pickle=True).item() # it's a dict
        # need to keep track of which frames are dropped to align with neural data
        # these were taken care of in convert_vids, just need to parse that
        # TODO maybe don't hardcode all these numbers... do they work for feb data? don't think so...
        dropped_fnames = os.listdir(self.facecam_dir + '/dropped/')
        runs = [int(fname[4:6]) for fname in dropped_fnames]
        frames = [int(fname[-9:-4]) for fname in dropped_fnames]
        frame_idxs = [(run - 1) * imgs_per_block + (frame - 1) for run, frame in zip(runs, frames)] # should be correctly 0-indexed

        # set up correctly-aligned data... let's get face motion, motSVD, and movSVD
        motion = np.full((imgs_total,), np.nan)
        motSVD = np.full((imgs_total, 500), np.nan)
        movSVD = np.full((imgs_total, 500), np.nan) # really going overboard hardcoding here
        # mask for values to keep nan
        mask = np.ones((imgs_total,), dtype=bool)
        mask[frame_idxs] = False
        motion[mask] = self.facemap_data['motion'][1]
        motSVD[mask] = self.facemap_data['motSVD'][1]
        movSVD[mask] = self.facemap_data['movSVD'][1]

        # now these are aligned with the cam_idxs
        # but you want to align them with the imaging frames
        # get_cam_idxs.py generated necessary data
        img_times = np.load(f'{self.proc_dir}img_crossings.npy')
        cam_times = np.load(f'{self.proc_dir}cam_crossings.npy')
        cam_frame_dists = np.load(f'{self.proc_dir}cam_frame_dists.npy') # minimum distances -- if too large, throw out that frame

        # need to figure out sampling rate from these... stupid
        med_img_time = np.median(np.diff(img_times))
        if med_img_time < 1000:
            sample_rate = 20000 # KHz, few recordings where I had this set by accident
        else:
            sample_rate = 20000000
        # use that to determine viable distances
        max_distance = 0.05 * sample_rate # 20 Hz camera frames should never be more than 1/20th second from an imaging frame
        bad_frames = cam_frame_dists > max_distance # use this as index for nans
        # but have to interpolate first

        # can only interp one dimension at a time afaict
        motion = np.interp(img_times, cam_times, motion)
        motSVD = np.array([np.interp(img_times, cam_times, motSVD[:, i]) for i in range(num_pcs)]).T
        movSVD = np.array([np.interp(img_times, cam_times, movSVD[:, i]) for i in range(num_pcs)]).T

        # now... set bad frames to nans
        motion[bad_frames] = np.nan
        motSVD[bad_frames] = np.nan
        movSVD[bad_frames] = np.nan

        # and save to self
        # TODO save to disk too? depends how long this takes...
        self.motion = motion
        self.motSVD = motSVD
        self.movSVD = movSVD

        return self.motion, self.motSVD, self.movSVD

    def get_facecam_traces_norm_trials(self, num_pcs=20):
        '''
        get the facecam traces during normalizing stimulus trials
        parallel to the neural responses
        
        using these for partials, maybe other things too
        '''
        # first just get them
        motion, mot, mov = self.get_facecam_traces(num_pcs)

        # now... need to trial-align
        motion_tr = np.array([motion[t-std_f_before:t+std_f_after] for t in self.trial_idxs])
        motion_tr = motion_tr[np.newaxis, :, :] # give it the 1-length axis
        mot_tr = np.array([mot[t-std_f_before:t+std_f_after] for t in self.trial_idxs])
        mot_tr = np.transpose(mot_tr, (2, 0, 1))
        mov_tr = np.array([mov[t-std_f_before:t+std_f_after] for t in self.trial_idxs])
        mov_tr = np.transpose(mov_tr, (2, 0, 1))
        #print(motion_tr.shape)
        #print(mot_tr.shape)
        #print(mov_tr.shape)

        # now, get it sorted by stimuli like in norm_trial_responses
        # guess I'll just copy-paste from there... shitty
        responses, stimuli = self.get_trial_responses()
        #print(responses[0].shape)

        response_cutoff = 50
        unique_stim = np.unique(self.stim_info, axis=1)
        filtered_traces = [[], [], []]
        trace_types = [motion_tr, mot_tr, mov_tr]
        for stim_idx in range(unique_stim.shape[1]):
            # matching orientation (first row) and plaid status (second row)
            idxs = np.logical_and(self.stim_info[0] == unique_stim[0, stim_idx],
                    self.stim_info[1] == unique_stim[1, stim_idx])
            if np.sum(idxs) > response_cutoff:
                for i in range(3):
                    filtered_traces[i].append(trace_types[i][:, idxs])
        
        return filtered_traces[0], filtered_traces[1], filtered_traces[2]