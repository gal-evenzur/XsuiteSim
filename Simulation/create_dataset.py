# %% IMPORTS
from dataset_funcs import *
from params import *
import matplotlib.pyplot as plt


plt.rcParams['image.cmap'] = 'afmhot'
# np.random.seed(1)

# Define the magnet settings to test
name = 'q0'
setting = 'x'  
magnet_settings = [490, 490.1]
change_beam = True
shift_list = shifts_array_random(shifts, shifts_range, 5, magnet_settings=magnet_settings, is_array=is_array)

n = {
    'train': 1,
    'val': 1,
    'test': 1
}


# | TRAIN | 
shift_train, histogram_train, xedges, yedges = rand_from_scratch_histogram(shifts_template=shifts, shifts_range=shifts_range,
                                                         particles_file=dat_file,
                                                         n_batch=n['train'], magnet_settings=magnet_settings,
                                                         verbose=True,
                                                         change_beam=change_beam, normalize=False, std=False, minmax=True)
print("Histogram train shape:", histogram_train.shape)


# VAL 
shift_val, histogram_val, _, _ = rand_from_scratch_histogram(shifts_template=shifts, shifts_range=shifts_range,
                                                         particles_file=dat_file,
                                                         n_batch=n['val'], magnet_settings=magnet_settings,
                                                         verbose=True,
                                                         change_beam=change_beam, normalize=True, std=False, minmax=True)
print("Histogram val shape:", histogram_val.shape)

# < TEST >
# shift_test, histogram_test, _, _ = rand_from_scratch_histogram(shifts_template=shifts, shifts_range=shifts_range,
#                                                          particles_file=dat_file,
#                                                          n_batch=n['test' ], magnet_settings=magnet_settings,
#                                                          verbose=True,
#                                                          change_beam=change_beam, normalize=True, std=True, minmax=False)
# For testing purposes, use deterministic shifts
shift_test = shifts_array_deterministic(shifts, 'q0', 'x', [0], magnet_settings=magnet_settings)
shift_test = shift_list_to_matrix(shift_test, n=num_shifts)
histogram_test, xedges, yedges = shifts_to_histogram(shift_test, filename=dat_file)
print("Histogram test shape:", histogram_test.shape)


# Save the data
# save_histogarms_hd5(histogram_train, shift_train, magnet_settings, xedges, yedges,
#                     dataset="train", write_add='w', filename=histogram_dat)

# save_histogarms_hd5(histogram_val, shift_val, magnet_settings, xedges, yedges,
#                     dataset="val", write_add='a', filename=histogram_dat)

save_histogarms_hd5(histogram_test, shift_test, magnet_settings, xedges, yedges,
                    dataset="test", write_add='w', filename=histogram_dat)
