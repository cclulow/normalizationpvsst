import h5py

file_path = '/media/maclean/Storage/42R2/cellreg/072924_rois_cellreg.mat'

try:
    with h5py.File(file_path, 'r') as f:
        print("File opened successfully.")
        # List all groups
        print("Keys: %s" % f.keys())
        # Get the data
        data = list(f.keys())
        print("Data: %s" % data)
except OSError as e:
    print(f"Error opening file {file_path}: {e}")
