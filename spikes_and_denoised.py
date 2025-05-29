# 072924 re-infer spikes with faster, better-eyeballed time constants
# determined in SNR work. and saved 'denoised' traces for later SNR computation
import numpy as np
import matplotlib.pyplot as plt

from suite2p.extraction.dcnv import oasis_trace_c, oasis_trace_c_l1
from neural_data_object import NeuralData



# some params
fs = 30.01

mice = [
    ('18S', '071724'), ('18S', '071924'), ('18S', '072024'),
    ('21N', '022424'), ('21N', '022524'),
    ('30G', '082424'), ('30G', '082524'),
    ('30N', '062824'), ('30N', '062924'), ('30N', '070124'), ('30N', '070224'),
    ('32L', '042424'), ('32L', '042524'),
    ('32L2', '053024'),
    ('37L', '070424'), ('37L', '070524'),
    ('37L2', '070924'), ('37L2', '071024'),
    ('37R', '070224'), ('37R', '070324'),
    ('37R2', '070924'), ('37R2', '071024'),
    ('42N', '022524'), ('42N', '022624'),
    ('44N', '022524'), ('44N', '022624')
]

# need to get all this info in a dict or something
#mice = [('42R', '072224'),('42R', '072324'),('42R', '072424'),('42R', '072524')]
        #('42R2', '072924'),('42R2', '073024'),('42R2', '073124'),]
mouse_types = [mice]

param_pairs = [
    (1.6, 0.4),# 6f tau_d and tau_r
    #(1.2, 0.3) # 8s tau_d and tau_r
]

for mousedates, params in zip(mouse_types, param_pairs):
    tau_d, tau_r = params
    for mouse, date in mousedates:
        print(f'running mouse {mouse} {date}, tau_d{tau_d} and tau_r{tau_r}')
        # new default base_name symlinks in data subdirectory
        data = NeuralData(mouse, date)

        dF = data.get_dF() # default args for now -- alpha = 1.0 etc

        # now infer spikes and return c -- should be fast
        denoised = np.zeros(dF.shape)
        spks = np.zeros(dF.shape) # don't need this I think
        for i, neu in enumerate(dF):
            print(f'neu {i} of {len(dF)}')
            c, s = oasis_trace_c(neu, fs, tau_d, tau_r)
            denoised[i] = c
            spks[i] = s
        
        # reshape to fit all rois
        all_spks = np.zeros((len(data.iscell), spks.shape[1]))
        for i, idx in enumerate(np.where(data.iscell)[0]):
            all_spks[idx] = spks[i]

        # how to save? in proc... I guess simplest is same format as l1 etc
        np.save(data.s2p_dir + 'spks.npy', all_spks)
        np.save(data.proc_dir + 'denoised.npy', denoised) # this is truly new