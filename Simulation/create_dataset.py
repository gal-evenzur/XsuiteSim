# %% IMPORTS
from dataset_funcs import *
from params import *
import matplotlib.pyplot as plt
import sys
import os
import time

# Start timing
start_time = time.time() 

try:
    idx = int(sys.argv[1])
    storage_path = sys.argv[2]
    datafile_path = f"{storage_path}/Data_2/h_{idx}.h5"
except:
    idx = int(time.time() % 1e6)
    pydir = os.path.dirname(os.path.abspath(__file__)) # This results ""
    simdir = os.path.dirname(pydir)
    datafile_path = f"{simdir}/Data_2/h_{idx}.h5"


print("saved file will be:", datafile_path)
# Ensure the directory exists
os.makedirs(os.path.dirname(datafile_path), exist_ok=True)

plt.rcParams['image.cmap'] = 'afmhot'
rng = np.random.default_rng(seed=idx)
# Define the magnet settings to test

magnet_settings = [490, 490.1, 490.2]
change_beam = True

n = {
    'train': 10,
    'val': 0,
    'test': 0
}


# | TRAIN | 
shift_train, histogram_train, xedges, yedges = rand_from_scratch_histogram(shifts_template=shifts, shifts_range=shifts_range,
                                                        particles_file=dat_file,
                                                        rng=rng,
                                                        n_batch=n['train'], magnet_settings=magnet_settings,
                                                        verbose=True,
                                                        change_beam=change_beam, normalize=False, std=False, minmax=True)
print("Histogram train shape:", histogram_train.shape)


# VAL 
if n['val'] > 0:
    shift_val, histogram_val, _, _ = rand_from_scratch_histogram(shifts_template=shifts, shifts_range=shifts_range,
                                                            particles_file=dat_file,
                                                            rng=rng,
                                                            n_batch=n['val'], magnet_settings=magnet_settings,
                                                            verbose=True,
                                                            change_beam=change_beam, normalize=False, std=False, minmax=True)
    print("Histogram val shape:", histogram_val.shape)
else:
    shift_val = np.zeros_like(shift_train)
    histogram_val = np.zeros_like(histogram_train)
# < TEST >
if n['test'] > 0:
    # shift_test, histogram_test, _, _ = rand_from_scratch_histogram(shifts_template=shifts, shifts_range=shifts_range,
    #                                                         particles_file=dat_file,
    #                                                         n_batch=n['test' ], magnet_settings=magnet_settings,
    #                                                         verbose=True,
    #                                                         change_beam=change_beam, normalize=True, std=True, minmax=False)
    # For testing purposes, use deterministic shifts
    shift_test = shifts_array_deterministic(shifts, 'q0', 'x', [0, 1e-3], magnet_settings=magnet_settings)
    shift_test = shift_list_to_matrix(shift_test, n=num_shifts)
    histogram_test, xedges, yedges = shifts_to_histogram(shift_test, change_beam=True, rng=rng)
    print("Histogram test shape:", histogram_test.shape)
else:
    shift_test = np.zeros_like(shift_train)
    histogram_test = np.zeros_like(histogram_train)

# Calculate total execution time
total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds")

# Save the data
save_histogarms_hd5(histogram_train, shift_train, magnet_settings, xedges, yedges,
                    dataset="train", write_add='w', filename=datafile_path)

save_histogarms_hd5(histogram_val, shift_val, magnet_settings, xedges, yedges,
                    dataset="val", write_add='a', filename=datafile_path)

save_histogarms_hd5(histogram_test, shift_test, magnet_settings, xedges, yedges,
                    dataset="test", write_add='a', filename=datafile_path)

# Save execution time to HDF5
import h5py
with h5py.File(datafile_path, 'a') as f:
    f.attrs['total_execution_time'] = total_time
