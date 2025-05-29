### 030524 convert suite2p 
### cellreg's builtin function only takes matfiles which I don't bother saving
import os
import numpy as np
from scipy.io import savemat

mice = ['42R']
dates = ['072224','072324','072424','072524']

base_path = '/media/maclean/Storage/'

for mouse in mice:
    mouse_dir = base_path + f'{mouse}/'
    cellreg_dir = mouse_dir + 'cellreg/'
    if not os.path.isdir(cellreg_dir):
        os.mkdir(cellreg_dir)
    for date in dates:
        s2p_path = base_path + f'{mouse}/{date}/runs/suite2p/plane0/'
        ops = np.load(s2p_path + 'ops.npy', allow_pickle=True).item()
        iscell = np.load(s2p_path + 'iscell.npy')[:, 0].astype(bool)
        stat = np.load(s2p_path + 'stat.npy', allow_pickle=True)[iscell]

        # cellreg wants NxYxX footprint matrix, s2p is sparse form
        cellreg_rois = np.empty((len(stat), ops['Ly'], ops['Lx']))
        for i, cell in enumerate(stat):
            # keeping overlapping pixels for now... is that a problem?
            ypix = cell['ypix']
            xpix = cell['xpix']
            cellreg_rois[i, ypix, xpix] = cell['lam']

        # save now
        savemat(cellreg_dir + f'{date}_rois_cellreg.mat', {'footprint': cellreg_rois})
