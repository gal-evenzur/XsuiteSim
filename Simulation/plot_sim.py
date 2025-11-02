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
    pydir = os.path.dirname(os.path.abspath(__file__)) # This results "fresh-start/Simulation"
    simdir = os.path.dirname(pydir) # This results "fresh-start"
    datafile_path = f"{simdir}/merged_data/merged_data.h5" # Default = merged_data.h5
print(f"Plotting {datafile_path}")

cutoff_path = datafile_path.replace('.h5', f'_cutoff_{split}.pdf')
xedges, yedges, magnet_settings, histograms, shifts_list, time_stamps = import_histograms_lightweight(datafile_path)
print("Imported histogram data.")
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
all_shifts = shifts_list[:,0]


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
        if isinstance(shifts_range[param][setting], tuple):
            range_min, range_max = shifts_range[param][setting]
            axes[i].axvline(range_min, color='red', linestyle='--', alpha=0.7, label=f'Range: [{range_min:.2e}, {range_max:.2e}]')
            axes[i].axvline(range_max, color='red', linestyle='--', alpha=0.7)
            axes[i].legend()
        # Set ticks at edges and center with scientific notation
        min_val = np.min(param_values)
        max_val = np.max(param_values)
        center_val = (min_val + max_val) / 2
        
        # Find the 3 maximum areas of the histogram
        hist_counts, hist_bins = np.histogram(param_values, bins=50)
        # Get the indices of the 3 highest bins
        top_3_indices = np.argsort(hist_counts)[-3:]
        # Get the center positions of these bins
        top_3_positions = [(hist_bins[i] + hist_bins[i+1]) / 2 for i in top_3_indices]
        
        # Combine all tick positions
        tick_positions = [min_val, center_val, max_val] + top_3_positions
        
        # Remove overlapping ticks by sorting and filtering
        tick_positions = sorted(set(tick_positions))  # Remove duplicates and sort

        
        # Set ticks with alternating heights to avoid overlapping labels
        axes[i].set_xticks(tick_positions)
        # Create custom tick labels with different vertical positions
        tick_labels = [f'{tick:.2e}' for tick in tick_positions]
        axes[i].set_xticklabels(tick_labels)
        
        # Apply alternating vertical offsets to tick labels
        for j, (tick, label) in enumerate(zip(tick_positions, tick_labels)):
            y_offset = -0.15 if j % 2 == 1 else -0.08  # Alternate between two heights
            axes[i].text(tick, y_offset, label, transform=axes[i].get_xaxis_transform(), 
                ha='center', va='top', fontsize=8, rotation=0)
        
        # Hide the default tick labels since we're using custom ones
        axes[i].set_xticklabels([])

        p_idx += 1

    plt.tight_layout()
    param_hist_path = datafile_path.replace('.h5', f'_{param}_histograms_{split}_.pdf')
    plt.savefig(param_hist_path, dpi=300, bbox_inches='tight')
    print(f"{param} histograms plot saved to {param_hist_path}")

# Plot histogram of time stamps
plt.figure(figsize=(8, 6))
plt.hist(time_stamps/3600, bins=50, alpha=0.7)
plt.title('Time Stamps Distribution')
plt.xlabel('Time Stamp (hours)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

plt.tight_layout()
time_stamps_path = datafile_path.replace('.h5', f'_time_stamps_{split}.pdf')
plt.savefig(time_stamps_path, dpi=300, bbox_inches='tight')
print(f"Time stamps histogram saved to {time_stamps_path}")
plt.close()

# Calculate total sum for each histogram
# histograms shape: (n_magnet_settings, n_samples, height, width)
total_sums = np.sum(histograms, axis=(2, 3))  # Sum over height and width dimensions

# Create histogram plots for each magnet setting
n_magnet_settings = total_sums.shape[0]
fig, axes = plt.subplots(1, n_magnet_settings, figsize=(4 * n_magnet_settings, 4))

# Handle case where there's only one magnet setting
if n_magnet_settings == 1:
    axes = [axes]

for i in range(n_magnet_settings):
    setting_sums = total_sums[i]  # Array of total sums for this magnet setting
    
    axes[i].hist(setting_sums, bins=50, alpha=0.7)
    axes[i].set_title(f'Magnet Setting {i}')
    axes[i].set_xlabel('Total Sum')
    axes[i].set_ylabel('Count')
    axes[i].grid(True, alpha=0.3)
    axes[i].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

plt.tight_layout()
total_sums_path = datafile_path.replace('.h5', f'_total_sums_{split}.pdf')
plt.savefig(total_sums_path, dpi=300, bbox_inches='tight')
print(f"Total sums histograms saved to {total_sums_path}")
plt.close()