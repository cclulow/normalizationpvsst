### 121123 get some data into processable form
import os
import numpy as np
import matplotlib.pyplot as plt

from oasis.oasis_methods import constrained_oasisAR2
from oasis.functions import estimate_parameters, deconvolve
from suite2p.extraction.dcnv import preprocess
from scipy.stats import ttest_ind
from utils import proc_data_to_trials

micedates = {
    '18S': ['071724','071924','072024'],
    '21N': ['022424','022524'],
    '30G': ['082424','082524'],
    '30N': ['062824','062924','070124','070224'],
    '32L': ['042424','042524'],
    '32L2': ['053024'],
    '37L': ['070424','070524',],
    '37L2': ['070924','071024',],
    '37R': ['070224','070324',],
    '37R2': ['070924','071024',],
    '42N': ['022524','022624'],
    '44N': ['022524','022624'],
}

#'42R': ['072224', '072324', '072424', '072524'],
#'42R2': ['072924', '073024', '073124','080124'],
#mice = ['42R2']
#dates = ['072924','073024','073124']
for mouse, dates in micedates.items():
    for date in dates:
        path = f'/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/{mouse}/{date}/'
        proc_data_to_trials([path], threshold=0.0028, f_before=30)

        proc_dir = os.path.join(path, 'proc/')
        dF = np.load(os.path.join(proc_dir, 'dF.npy'))
        spks = np.load(os.path.join(proc_dir, 'spks.npy'))

        # Check the number of trials
        num_trials_dF = dF.shape[1]
        num_trials_spks = spks.shape[1]

        # Print the number of trials detected in data
        print(f"Trials in dF: {num_trials_dF}, Trials in spks: {num_trials_spks}")

        # Adjust if there's an extra trial
        expected_trials = 1248

        if num_trials_dF > expected_trials:
            # Assume the extra trial is the last one (or could be the first if known)
            dF = dF[:, :expected_trials, :]
            spks = spks[:, :expected_trials, :]

            print("Extra trial detected and removed.")



        # zscore everything -- don't trust scipy's function
        def z_score(resps):
            #return resps / resps.max((1, 2)).reshape(-1, 1, 1)
            means = resps.mean((1, 2))
            stds = resps.std((1, 2))
            return (resps - means.reshape(-1, 1, 1)) / stds.reshape(-1, 1, 1)

        def sort(resps):
            #return resps
            return resps[np.argsort(np.argmax(resps, 1))]

        dF = z_score(dF).mean(1)
        spks = z_score(spks).mean(1)

        # set up to save
        #mouse = 'MS'
        fname_base = f'figures/psths/{mouse}_{date}'

        # plot things
        '''
        plt.figure()
        plt.imshow(sort(dF))
        plt.colorbar()
        plt.title('corrected dF/F')
        plt.savefig(f'{fname_base}_dF.png', dpi=200)
	'''

        plt.figure()
        plt.imshow(sort(spks))
        plt.colorbar()
        plt.title('inferred spikes')
        plt.savefig(f'{fname_base}_spks.png', dpi=200)
