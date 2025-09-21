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

plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'

ctx = xo.ContextCpu()  # Use xo.ContextCupy() for GPU


xedges, yedges, magnet_settings, histograms, shifts_list = import_histograms_hd5(histogram_dat)

plot_from_file(filename=histogram_dat)

h = histograms.copy()
threshold = 2
h = np.where(h > threshold, h, 0)
plot_from_file(h, shift_list=shifts_list, magnet_settings=magnet_settings, xedges=xedges, yedges=yedges,
               name='beam', setting='fy0')

plt.show()