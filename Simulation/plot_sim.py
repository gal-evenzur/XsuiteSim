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

plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'

ctx = xo.ContextCpu()  # Use xo.ContextCupy() for GPU
plotting = False
plot_first = True
# Define the magnet settings to test
change = np.linspace(0, 0, 1)  # Example range for y shift in meters
name = 'dd'
setting = 'ang_y'  
magnet_settings = [490, 490.1]
shift_list = shifts_array(shifts, name, setting, change)

if plotting:
    fig, axs = plt.subplots(len(magnet_settings), len(change), figsize=(len(magnet_settings)*6, 5), 
                            tight_layout=True, sharex=True, sharey=True)
    
    for i, shift in enumerate(shift_list):
        plot_multiple_magnet_settings(shift, magnet_settings, axs=axs[:,i])
        for j, ax in enumerate(axs[:,i]):
            ax.set_title(f'{magnet_settings[j]} {name} {setting} = {change[i]*1e3:.2f} mm')
            print(f"{name} {setting} shift = {change[i]*1e3:.2f} mm:")

if plot_first:

    # shifts[name][setting] = change.min()
    line, env, ref = line_init(shifts=shifts)

    particles = import_particles_from_hdf5(dat_file, ref)
    # temp_particles = generate_secondary_particles(shifts, n_particles, verbose=False)
    # save_particles_to_hdf5(temp_particles, dat_file)

    # particles = import_particles_from_hdf5(line, dat_file, p0c=ref['p'])

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    h, xedges, yedges = track_monitor(line, particles)
    plt.imshow(h.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
    ax.locator_params(axis='x', nbins=10)
    ax.locator_params(axis='y', nbins=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.grid(True,linewidth=0.25,alpha=0.25,which='major')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.title('Monitor at the end of the line')
    plt.colorbar(label='Counts per bin')
    plt.tight_layout()


    particle_dir = track_line(line, particles)

    plot_trajectories(particle_dir, line, n_plot=100, show_dead=show_dead)
    # xy_plot_line(line, particle_dir, ele_str='q', elementNames='Quadrupoles')
    # xy_plot_line(line, particle_dir, ele_str='dd', elementNames='Dipoles')

plt.show()