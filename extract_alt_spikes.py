### 031924 run some tests on trial-robustness of noise corrs
import numpy as np

from suite2p.extraction.dcnv import preprocess, oasis_trace_alt, oasis_trace_l1, oasis_trace_l0, fast_l1, fast_l0
from oasis.functions import deconvolve
from neural_data_object import NeuralData

mice = ['42R']
dates = ['072224','072324','072424','072524']
mice = ['42R2']
dates = ['072924','073024','073124']
mousedates = [('42R','072224'),('42R', '072324'),('42R', '072424'),('42R', '072524'),('42R2', '072924'),('42R2', '073024'),('42R2', '073124')]
#mousedates = [('44N', '022624')]
#mousedates = [('32L', '042624'), ('91N', '042624')]
alpha=1.0
win_baseline=120
#mousedates = [('30N', '062724'), ('30N', '062824')]

for bleh in range(1):
    for mouse, date in mousedates:
        # load in data and labels
        if date.startswith('02'):
            base_name = '/media/maclean/Storage/'
        else:
            base_name = '/media/maclean/Storage/'
        try:
            data = NeuralData(mouse, date, base_name=base_name)
        except:
            print(f'skipping {mouse} {date}')
            continue
        f_raw = data.F
        f_neu = data.Fneu
        f = f_raw - alpha * f_neu
        f = preprocess(f, baseline='maximin', win_baseline=win_baseline, sig_baseline=10,
                fs=30.01) # params right?

        iscell = data.iscell
        l1_traces = np.zeros((iscell.shape[0], f.shape[1]))
        #l0_traces = np.zeros((iscell.shape[0], f.shape[1]))
        
        # loop over neurons
        for i, idx in enumerate(np.where(iscell)[0]):
            print(f'running neuron {i}')
            s = oasis_trace_l1(f[i])
            l1_traces[idx] = s
            #s = oasis_trace_l0(f[i])
            #l0_traces[idx] = s
        
            # save
            if i % 10 == 0:
                np.save(data.proc_dir + 'g_l1_spks.npy', l1_traces)
                #np.save(data.proc_dir + 'l0_spks.npy', l0_traces)
        np.save(data.proc_dir + 'g_l1_spks.npy', l1_traces)
        #np.save(data.proc_dir + 'l0_spks.npy', l0_traces)


        # that extracted the spikes, but want them processed trial-by-trial
        # too. not quite sure how to do that...
        # need to rerun process_data after this then
