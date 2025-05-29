import numpy as np
import sys

from suite2p.run_s2p import run_s2p

# dogshit error handling
if len(sys.argv) < 2:
    print('usage: python run_suite2p.py <ops_file> <data_dir1> <data_dir2>...')

# run suite2p where?
ops_file = sys.argv[1]
ops = np.load(ops_file, allow_pickle=True).item()

mousedates = [('18S', '071724')]
for mouse, date in mousedates:
    base_name = f'/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/'
    dir_name = f'{base_name}{mouse}/{date}/runs/'
    db = {'data_path': [dir_name], 'num_workers': 8}
    run_s2p(ops=ops, db=db)


