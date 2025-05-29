import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from neural_data_object import NeuralData

# params for processing
use_l = -1
sigma = 0
easy = True

#mousedates = [('42R2', '072924'),('42R2', '073024'),('42R2', '073124')]
mousedates = [('42R', '072224'),('42R', '072324'),('42R', '072424'),('42R', '072524')]
#plt.style.use('./poster.mplstyle')
#plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})

for mouse, date in mousedates:
    print('loading data')
    base_name = '/media/maclean/Storage/'
    
    data = NeuralData(mouse, date, base_name=base_name, use_l='cascade', smooth=sigma)

    # this largely copy-pasted out of char plots in object
    # which won't work here since rois won't plot

    indices = data.normalization_indices()
    resps, stim = data.get_norm_trial_responses(dF=False)
    
    # computing trial averaged responses to all the stimuli
    avg_resps = np.array([resp[:,:,data.resp_start:data.resp_end].mean((1, 2)) for resp in resps])
    std_resps = np.array([resp[:,:,data.resp_start:data.resp_end].mean(2).std(1) / np.sqrt(resp.shape[1]) for resp in resps])
    ori_resps = avg_resps[stim[:, 1] == 0]
    ori_stds = std_resps[stim[:, 1] == 0]
    # plaid shit -- would like errorbars here too... but not now
    p0_resps = avg_resps[np.logical_and(stim[:, 1] == 1, stim[:, 0] == 0)]
    p45_resps = avg_resps[np.logical_and(stim[:, 1] == 1, stim[:, 0] == 45)]
    p22_resps = avg_resps[np.logical_and(stim[:, 1] == 1, stim[:, 0] == 22.5)]
    p67_resps = avg_resps[np.logical_and(stim[:, 1] == 1, stim[:, 0] == 67.5)]
    plaid_resps = [p0_resps, p22_resps, p45_resps, p67_resps] # need to pick one

    # get pref plaids
    # this dependent on hack in neural_data_object -- bad!
    pref_plaids = data.normalization_indices(return_prefs=True)[2]
    print(pref_plaids)

    # just do hard vis resp neurons for now
    vis_resp = data.vis_responsive_cells(easy=easy)
    neu_list = np.where(data.vis_responsive_cells(easy=easy))[0]
    print(f'{len(neu_list)} vis responsive cells, easy={easy}')

#I ADDED THIS SHIT AND IDK IF IT WILL WORK BUT IS FOR CELLREG
    # Save the list of visually responsive cells
    vis_resp_labels_file = f'{base_name}{mouse}/{date}/runs/suite2p/plane0/vis_resp_labels.npy'
    np.save(vis_resp_labels_file, vis_resp)
    # Save the ROIs of visually responsive cells
    stat = data.stat  # Assuming stat contains the ROI information
    vis_resp_rois = [stat[neu] for neu in neu_list]
    vis_resp_rois_file = f'{base_name}{mouse}/{date}/runs/suite2p/plane0/vis_resp_rois.mat'
    savemat(vis_resp_rois_file, {'rois': vis_resp_rois})

    # check out norm index distribution
    plt.figure()
    plt.hist(indices[data.vis_responsive_cells(easy=easy)], bins='doane')
    plt.xlabel('norm index')
    plt.ylabel('count')
    plt.xlim(-1, 1)
    plt.title(f'norm index distribution for {mouse} {date}')
    plt.show()

    for neu in neu_list:
        pref_plaid = pref_plaids[neu]
        plaid_to_plot = plaid_resps[pref_plaid]
        fig, ax1 = plt.subplots()
        fontname = 'Nimbus Sans' 

        # Plot tuning curve
        ax1.errorbar(stim[stim[:, 1] == 0, 0], ori_resps[:, neu], yerr=ori_stds[:, neu],
                    color='black', label='oriented gratings')
        ax1.hlines(plaid_to_plot[:, neu], 0, 167.5, label='plaid at preferred angle', color='red')

        # === Font size upgrades ===
        ax1.set_xlabel('Grating orientation (degrees)', fontsize=18, fontname=fontname)
        ax1.set_ylabel('Average response (a.u.)', fontsize=18, fontname=fontname)
        ax1.set_title(f'NI = {indices[neu]:.3f}', fontsize=24, fontname=fontname)
        ax1.tick_params(axis='both', labelsize=18)  # x and y axis ticks
        ax1.legend(prop={'size': 18})               # legend font size

        fig.set_tight_layout(True)
        fig.savefig(f'figures/tuning_curves/{date}_n{neu}_l{use_l}_s{sigma}.png', dpi=300)
        plt.close()
