# %% IMPORTS

from dataset_funcs import *
from params import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'

split = 'train'
import os

try:
    idx = int(sys.argv[1])
except:
    idx = 0
pydir = os.path.dirname(os.path.abspath(__file__)) # This results ""
simdir = os.path.dirname(pydir)
datafile_path = f"{simdir}/Data/h_{idx}.h5"

cutoff_path = datafile_path.replace('.h5', f'_cutoff_{split}.pdf')
xedges, yedges, magnet_settings, histograms, shifts_list = import_histograms_hd5(datafile_path, split=split)

plot_from_file(filename=datafile_path, split=split)

h = histograms.copy()
threshold = 2
h = np.where(h > threshold, h, 0)
plot(xedges=xedges, yedges=yedges, magnet_settings=magnet_settings, histograms=h, shift_list=shifts_list,
        pdfname=cutoff_path, split=split,
               name='beam', setting='fy0')

# NEED TO CHECK IF shifts are correctly imported and lead to the same histograms
