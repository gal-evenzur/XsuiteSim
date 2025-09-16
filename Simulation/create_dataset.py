# %% IMPORTS
import xobjects as xo
import xtrack as xt

import h5py
from sim_functions import *
from params import *
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


from scipy import ndimage
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

plt.rcParams['image.cmap'] = 'afmhot'


# Save multiple histograms for a shift list
def shifts_to_histogram(shift_list, ref=ref, filename=None, change_beam=False,
                        normalize=False,
                        verbose=False, monitor_bins=monitor_bins):
    
    if filename is None or change_beam:
        states = generate_secondary_particles(shifts, n_particles, verbose=verbose)
        particles = particles_from_states(states, ref, verbose=verbose)

    else:
        particles = import_particles_from_hdf5(filename, ref, verbose=verbose)

    height = monitor_bins[0]
    width = monitor_bins[1]
    n_magnet_settings = len(shift_list)
    n_changes = len(shift_list[0])
    histograms = np.zeros((n_magnet_settings, n_changes, width, height))

    for magnet_idx in range(n_magnet_settings):
        for i, shift in enumerate(shift_list[magnet_idx]):
            if verbose: print(f"Processing shift {i+1}/{len(shift_list[magnet_idx])}: {shift}")
            line, env, ref = line_init(shifts=shift)
            if change_beam:
                states = generate_secondary_particles(shift, n_particles, verbose=verbose)
                particles = particles_from_states(states, ref, verbose=verbose)
            h, xedges, yedges = track_monitor(line, particles)
            histograms[magnet_idx, i] = h.T  # Transpose to match the orientation

        # After creating the entire batch of histograms for this magnet setting, normalize if needed
        if normalize:
            histograms[magnet_idx] = normalize_batch_hists(histograms[magnet_idx], std=False, minmax=True)
            if verbose: print(f"Normalized histograms for magnet setting {magnet_idx+1}/{n_magnet_settings}")

    return histograms, xedges, yedges


# Define the magnet settings to test
change = np.linspace(-1.5,1.5, 6)  # Example range for y shift in meters
name = 'q0'
setting = 'x'  
magnet_settings = [490, 490.1]
# shift_list = shifts_array_deterministic(shifts, name, setting, change, magnet_settings)
shift_list = shifts_array_random(shifts, shifts_range, 5, magnet_settings=magnet_settings)

def plot_shift_array(shift_list, magnet_settings, name, setting, verbose=False):

    histograms, xedges, yedges = shifts_to_histogram(shift_list, filename=dat_file, change_beam=False,
                                                      verbose=verbose, normalize=True)
    # Plot the histograms

    fig, axs = plt.subplots(len(shift_list), len(shift_list[0]), figsize=(len(magnet_settings)*6, 5), 
                            tight_layout=True, sharex=True, sharey=True)

    for magnet_idx, m in enumerate(magnet_settings):
        for i, shift in enumerate(shift_list[magnet_idx]):
            if verbose: print(f"{m}: Started plotting change ", i+1, " of ", len(shift_list[magnet_idx]))
            ax = axs[magnet_idx, i]
            h = histograms[magnet_idx][i]
            ax.imshow(h, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
            change = shift[name][setting]
            ax.set_title(f'{magnet_settings[magnet_idx]} {name} {setting} = {change*1e3:.2f} mm')

            ax.xaxis.set_minor_locator(AutoMinorLocator(10))
            ax.yaxis.set_minor_locator(AutoMinorLocator(10))
            ax.grid(True,linewidth=0.25,alpha=0.25,which='major')
            if verbose: print(f"{m}: Finished plotting change ", i+1, " of ", len(shift_list))
    return histograms, xedges, yedges

histograms, xedges, yedges = plot_shift_array(shift_list, magnet_settings, name, setting,verbose=False)
plt.show()

# Save histograms and shifts to HDF5 file
def save_to_hdf5(histograms, shift_list, magnet_settings, xedges, yedges, 
                 filename=histogram_dat):
    with h5py.File(filename, 'w') as f:
        # Store edges as global datasets
        f.create_dataset('xedges', data=xedges)
        f.create_dataset('yedges', data=yedges)
        
        # Create a group for each magnet setting
        for magnet_idx, setting in enumerate(magnet_settings):
            # Create main group for this magnet setting
            group_name = f'magnet_{setting}'
            magnet_group = f.create_group(group_name)
            
            # Create subgroup for shifts
            shifts_group = magnet_group.create_group('shifts')
            for change_idx, shift in enumerate(shift_list[magnet_idx]):
                shift_group = shifts_group.create_group(f's_{change_idx}')
                for key, value in shift.items():
                    if isinstance(value, dict):
                        # Create subgroup for nested dictionaries
                        subgroup = shift_group.create_group(key)
                        for subkey, subvalue in value.items():
                            # Store each parameter as a dataset
                            subgroup.create_dataset(
                                subkey, 
                                data=subvalue, 
                            )
                    else:
                        # Store scalar values directly
                        shift_group.create_dataset(
                            key, 
                            data=value, 
                        )

            # Create subgroup for histograms
            histograms_group = magnet_group.create_group('histograms')
            for change_idx in range(len(shift_list[magnet_idx])):
                histograms_group.create_dataset(f'h_{change_idx}', data=histograms[magnet_idx][change_idx])

# Save the data
# save_to_hdf5(histograms, shift_list, magnet_settings, xedges, yedges)
