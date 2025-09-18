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


xedges, yedges, magnet_settings, histograms, shifts_list = import_histogram_data(histogram_dat)

plot_from_file(filename=histogram_dat)

plt.show()