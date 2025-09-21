# %% IMPORTS
import xobjects as xo
import xtrack as xt

import h5py
from dataset_funcs import *
from params import *
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


from scipy import ndimage
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

plt.rcParams['image.cmap'] = 'afmhot'
# np.random.seed(1)

# Define the magnet settings to test
name = 'q0'
setting = 'x'  
magnet_settings = [490, 490.1]
shift_list = shifts_array_random(shifts, shifts_range, 5, magnet_settings=magnet_settings, is_array=is_array)


# histograms, xedges, yedges = plot_shift_array(shift_list, magnet_settings, name=name, setting=setting, verbose=False,
#                                               normalize=False)
# plt.show()

# histograms, xedges, yedges = shifts_to_histogram(shift_list, filename=dat_file, verbose=True,
#                                                  change_beam=True, normalize=False, std=False, minmax=True)

histograms, xedges, yedges = rand_from_scratch_histogram(shifts_template=shifts, shifts_range=shifts_range,
                                                         particles_file=dat_file,
                                                         n_batch=5, magnet_settings=magnet_settings,
                                                         verbose=True,
                                                         change_beam=False, normalize=False, std=False, minmax=True)

# Save the data
save_histogarms_hd5(histograms, shift_list, magnet_settings, xedges, yedges, verbose=False)
