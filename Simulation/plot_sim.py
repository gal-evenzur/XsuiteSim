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
    datafile_path = sys.argv[2]
except:
    idx = 0
    pydir = os.path.dirname(os.path.abspath(__file__)) # This results ""
    simdir = os.path.dirname(pydir)
    datafile_path = f"{simdir}/Data/h_{idx}.h5"
print(f"Plotting {datafile_path}")

cutoff_path = datafile_path.replace('.h5', f'_cutoff_{split}.pdf')
xedges, yedges, magnet_settings, histograms, shifts_list = import_histograms_hd5(datafile_path, split=split)

plot_from_file(filename=datafile_path, split=split)

h = histograms.copy()
threshold = 2
h = np.where(h > threshold, h, 0)
plot(xedges=xedges, yedges=yedges, magnet_settings=magnet_settings, histograms=h, shift_list=shifts_list,
        pdfname=cutoff_path, split=split,
               name='beam', setting='fy0')

# Checked lmao
# Flatten shifts_list across magnets and samples to get all parameter values
# Shape: (n_samples, n_params)
all_shifts = shifts_list[0]


p_idx = 1
for param, vals in shifts.items():
    if param == 'magnetSettings':
        continue
    
    n_settings_per_element = len(vals)
    fig, axes = plt.subplots(1, n_settings_per_element, figsize=(4 * n_settings_per_element, 4))

    for i, (setting, val) in enumerate(vals.items()):
        param_values = all_shifts[:, p_idx]
        axes[i].hist(param_values, bins=50, alpha=0.7)
        axes[i].set_title(f'{param} {setting}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].grid(True, alpha=0.3)

        # Get the corresponding range from shifts_range
    
        range_min, range_max = shifts_range[param][setting]
        axes[i].axvline(range_min, color='red', linestyle='--', alpha=0.7, label=f'Range: [{range_min:.2e}, {range_max:.2e}]')
        axes[i].axvline(range_max, color='red', linestyle='--', alpha=0.7)
        axes[i].legend()
        # Set ticks at edges and center with scientific notation
        min_val = np.min(param_values)
        max_val = np.max(param_values)
        center_val = (min_val + max_val) / 2
        axes[i].set_xticks([min_val, center_val, max_val])
        axes[i].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        p_idx += 1

    plt.tight_layout()
    param_hist_path = datafile_path.replace('.h5', f'_{param}_histograms_{split}_.pdf')
    plt.savefig(param_hist_path, dpi=300, bbox_inches='tight')
    print(f"{param} histograms plot saved to {param_hist_path}")
