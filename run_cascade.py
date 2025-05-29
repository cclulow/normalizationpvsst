import os, sys
cascade_path = '/home/maclean/cascade'
if cascade_path not in sys.path:
    sys.path.append(cascade_path)

print('Cascade folder added to sys.path')
print('Current working directory: {}'.format(os.getcwd()))
from cascade2p import checks
checks.check_packages()
import numpy as np
import scipy.io as sio
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')

from cascade2p import cascade # local folder
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth
import matplotlib.pyplot as plt
import os
from cascade2p.utils_discrete_spikes import infer_discrete_spikes
from cascade_utils import preprocess_hal



def load_neurons_x_time(s2p_fld, alpha=1.0, win_baseline=60, sig_baseline=10, fs=30.01):
    """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""
    F = np.load(s2p_fld + 'F.npy', allow_pickle=True)
    Fneu = np.load(s2p_fld + 'Fneu.npy', allow_pickle=True)
    iscell = np.load(s2p_fld + 'iscell.npy', allow_pickle=True)
    F_cells = F[iscell[:,0].astype(bool)]
    Fneu_cells = Fneu[iscell[:,0].astype(bool)]
    sub_cells = F_cells - alpha * Fneu_cells
    _, baseline = preprocess_hal(sub_cells, 'maximin', win_baseline, sig_baseline, fs, prctile_baseline=20)
    print(np.median(baseline, 1))
    print(np.median(sub_cells, 1))
    #return sub_cells, baseline
    
    dF_F = (sub_cells - baseline)# / np.abs(baseline)
    return dF_F

base_path = '/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/'
mouseID = '42R2'
dates = {'42R2': ['073024', '073124', '072924']}

"""dates = {'21N': ['022424', '022524'], # finally looking good
         '44N': ['022524', '022624'],
          '42N': ['022524', '022624'],
         '30G': ['082424', '082524'], # no cellreg yet
          '32L': ['042424', '042524'], 
          '30N': ['070124', '070224'],
         '37R': ['070224', '070324'],
          '37L': ['070424', '070524'],
          '37R2': ['070924', '071024'],
          '37L2': ['070924', '071024'],
         '18S': ['071924', '072024'],
         '42R': ['072224', '072324'],
         '42R2': ['072924', '073024'],
         '33R': ['082524', '082624',]
         }"""
models = ['Global_EXC_30Hz_smoothing50ms_high_noise',
        'GC8s_EXC_30Hz_smoothing50ms_high_noise']
# 'Global_EXC_30Hz_smoothing50ms_high_noise',
# 'Global_EXC_30Hz_smoothing50ms_causalkernel',
# 'Global_EXC_30Hz_smoothing25ms_causalkernel',
# 'Global_EXC_30Hz_smoothing50ms',
# 'Global_EXC_30Hz_smoothing50ms_asymmetric_window_1_frame',
# 'GCaMP6f_mouse_30Hz_smoothing200ms']
"""dates = {'32L': ['042624'],
         '32L2': ['053024'],
         '30N': ['062824', '062924', '070124', '070224']}
dates = {'18S': ['071724']}
dates = {'59N': ['111224']}"""
mice = dates.keys()

# G had this, unclear why
np.random.seed(3952)
for mouse in dates.keys():
    if mouse in ['21N', '42N', '44N', '32L', '32L2', '42R', '42R2', '30N']:
        model_name = models[0]
    else:
        model_name = models[1] # 8s or not
    for date in dates[mouse]:
        s2p_fld = f'{base_path}/{mouse}/{date}/runs/suite2p/plane0/'
        proc_fld = f'{base_path}/{mouse}/{date}/proc/'
        #red_labels = np.load(s2p_fld+'red_labels.npy')
        #if os.path.exists(proc_fld + 'cascade_spks.npy'):
        #    continue # in case of rerunning.
        print(f'running {mouse} {date}')


        # G's loading and plotting, but my df/f
        #sub_cells, baseline = load_neurons_x_time(s2p_fld, alpha=0.7)
        #sys.exit()
        traces = load_neurons_x_time(s2p_fld, alpha=1.0)
        frame_rate = 30.01 # in Hz
        #normalize traces btwn 0 and 4 in an effort to improve the noise level estimate and get closer to what they expect? hacky but seems to work...
        mean_F = 0#np.mean(traces,axis=1)[:,None]
        traces = 4*((traces-mean_F)/(np.max(traces,axis=1)[:,None]-np.min(traces,axis=1)[:,None]))
        #traces = traces / 100
        #noise_levels = plot_noise_level_distribution(traces,frame_rate)
        #print(noise_levels)
        #print(np.median(noise_levels))

        print('Number of neurons in dataset:', traces.shape[0])
        print('Number of timepoints in dataset:', traces.shape[1])

        #PLOT EXAMPLE TRACES 
        #this is where I got the idea that traces should be between 0 and 4
        neuron_indices = np.random.randint(traces.shape[0], size=10)
        #plot_dFF_traces(traces,neuron_indices,frame_rate)
        #plt.show()

        cascade.download_model(model_name,verbose = 1)
        num_blocks = 5
        chunk_size = len(traces) // num_blocks
        preds = [cascade.predict(model_name, traces[i*chunk_size:(i+1)*chunk_size]) for i in range(num_blocks)]
        # get last few if not evenly divided
        if num_blocks * chunk_size < len(traces):
            preds.append(cascade.predict(model_name, traces[num_blocks * chunk_size:]))
        #first_prob = cascade.predict(model_name, traces[:len(traces)//3])
        #second_prob = cascade.predict(model_name, traces[len(traces)//3:2*len(traces)//3]) # split in half b/c running out of ram sometimes if I don't
        #third_prob = cascade.predict(model_name, traces[2*len(traces)//3:])
        spike_prob = np.concatenate(preds, axis=0)
        np.save(proc_fld+'cascade_spks.npy',spike_prob)
        del spike_prob

        # G's plotting analysis stuff, mostly untouched
        """
        traces_4th_q = traces[noise_levels>np.percentile(noise_levels,75)]
        traces_3rd_q = traces[(noise_levels>np.percentile(noise_levels,50))&(noise_levels<np.percentile(noise_levels,75))]
        traces_2nd_q = traces[(noise_levels>np.percentile(noise_levels,25))&(noise_levels<np.percentile(noise_levels,50))]
        traces_1st_q = traces[(noise_levels<np.percentile(noise_levels,25))]
        traces_by_noise_level = [traces_1st_q,traces_2nd_q,traces_3rd_q,traces_4th_q]

        for model_name in models: 
            spks = np.load(s2p_fld+model_name+'_spks.npy')

            if not os.path.isdir('figures/cascade/' + model_name):
                os.mkdir('figures/cascade/' + model_name)
            #plot a few examples for each noise level

            spks_4th_q = spks[noise_levels>np.percentile(noise_levels,75)]
            spks_3rd_q = spks[(noise_levels>np.percentile(noise_levels,50))&(noise_levels<np.percentile(noise_levels,75))]
            spks_2nd_q = spks[(noise_levels>np.percentile(noise_levels,25))&(noise_levels<np.percentile(noise_levels,50))]
            spks_1st_q = spks[(noise_levels<np.percentile(noise_levels,25))]
            spks_by_noise_level = [spks_1st_q,spks_2nd_q,spks_3rd_q,spks_4th_q]

            n_plots = 5
            for nl,tr in enumerate(traces_by_noise_level): 
                fig1, axs1 = plt.subplots(n_plots, 1, tight_layout=True, figsize=(30, 20))
                for i,cell in enumerate(np.random.randint(tr.shape[0], size=n_plots)):
                    axs1[i].plot(tr[cell,35000:40000],color='teal')
                    axs1[i].plot(spks_by_noise_level[nl][cell,35000:40000],color='purple')        
                fig1.savefig('figures/cascade/'+model_name+'/nl_'+str(nl)+'_example_traces_zoom'+'.png')
        """