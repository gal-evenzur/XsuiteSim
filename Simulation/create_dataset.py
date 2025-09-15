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
def shifts_to_histogram(shift_list, ref=ref, filename=None, change_beam=False, verbose=False, monitor_bins=monitor_bins):
    
    if filename is None and change_beam:
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
            print(f"Processing shift {i+1}/{len(shift_list[magnet_idx])}: {shift}")
            line, env, ref = line_init(shifts=shift)
            if change_beam:
                states = generate_secondary_particles(shift, n_particles, verbose=verbose)
                particles = particles_from_states(states, ref, verbose=verbose)
            h, xedges, yedges = track_monitor(line, particles)
            histograms[magnet_idx, i] = h.T  # Transpose to match the orientation

    return histograms, xedges, yedges


# Define the magnet settings to test
change = np.linspace(-1e-3, 1e-3, 4)  # Example range for y shift in meters
name = 'q0'
setting = 'x'  
magnet_settings = [490, 490.1]
shift_list = shifts_array(shifts, name, setting, change, magnet_settings)


histograms, xedges, yedges = shifts_to_histogram(shift_list, filename=dat_file, change_beam=False, verbose=True)


# Plot the histograms

fig, axs = plt.subplots(len(magnet_settings), len(change), figsize=(len(magnet_settings)*6, 5), 
                        tight_layout=True, sharex=True, sharey=True)

for magnet_idx in range(len(magnet_settings)):
    for i, shift in enumerate(shift_list[magnet_idx]):
        print("Started plotting change ", i+1, " of ", len(shift_list[magnet_idx]))
        ax = axs[magnet_idx, i]
        h = histograms[magnet_idx][i]
        ax.imshow(h, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
        ax.set_title(f'{magnet_settings[magnet_idx]} {name} {setting} = {change[i]*1e3:.2f} mm')
        # ax.locator_params(axis='x', nbins=10)
        # ax.locator_params(axis='y', nbins=10)
        # ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        # ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        # ax.grid(True,linewidth=0.25,alpha=0.25,which='major')
        # ax.set_xlabel('x [m]')
        # ax.set_ylabel('y [m]')
        # plt.colorbar(ax=ax, label='Counts per bin')
        print("Finished plotting change ", i+1, " of ", len(shift_list))
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


def save_shifts_to_hdf5(shifts_list, filename, compression='None'):
    """
    Save list of shift dictionaries to HDF5 with hierarchical structure.
    Optimized for the specific shift structure with nested dicts.
    
    Args:
        shifts_list: List of shift dictionaries
        filename: Path to HDF5 file
        compression: Compression method ('gzip', 'lzf', 'szip', or None)
    """
    with h5py.File(filename, 'w') as f:
        # Create main group for all shifts
        shifts_group = f.create_group('shifts')
        
        for i, shift in enumerate(shifts_list):
            # Create group for this shift
            shift_group = shifts_group.create_group(f'shift_{i:06d}')
            
            for key, value in shift.items():
                if isinstance(value, dict):
                    # Create subgroup for nested dictionaries
                    subgroup = shift_group.create_group(key)
                    for subkey, subvalue in value.items():
                        # Store each parameter as a dataset
                        subgroup.create_dataset(
                            subkey, 
                            data=subvalue, 
                            compression=compression
                        )
                else:
                    # Store scalar values directly
                    shift_group.create_dataset(
                        key, 
                        data=value, 
                        compression=compression
                    )
# Save the data
save_to_hdf5(histograms, shift_list, magnet_settings, xedges, yedges)
print("Data saved successfully to histogram_data.h5")


# Print the structure of the HDF5 file
def print_hdf5_structure(filename=histogram_dat):
    def print_attrs(name, obj):
        print(f"{name}")
        for key, val in obj.attrs.items():
            print(f"    Attribute: {key}: {val}")

    def print_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name} - Shape: {obj.shape}, Type: {obj.dtype}")
        
    print(f"\nHDF5 file structure for: {filename}")
    print("="*50)
    
    with h5py.File(filename, 'r') as f:
        # Print all groups and datasets
        print("Groups and datasets:")
        f.visit(print_attrs)
        
        # Print datasets with shape information
        print("\nDataset details:")
        f.visititems(print_datasets)
        
        # Show the main structure
        print("\nHierarchical structure:")
        def print_hierarchy(name, obj):
            indent = "    " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}Dataset: {name}, Shape: {obj.shape}")
        
        f.visititems(print_hierarchy)

# Execute the function to print the structure
with h5py.File(histogram_dat, 'r') as f:
    # Print all groups and datasets
    print()