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

def shifts_array(shifts, element, setting, range_vals):
    """ create a list of shifts for a given element and setting """
    shift_list = []
    shifts_copy = deepcopy(shifts)
    for val in range_vals:
        shifts_copy[element][setting] = val
        shift_list.append(deepcopy(shifts_copy))
    return shift_list

# Define the magnet settings to test
magnet_settings = [490, 490.1]
change = np.linspace(-1.5, 1.5, 4)  # Example range for x shift in meters
shift_list = shifts_array(shifts, 'q0', 'ang_x', change)

fig, axs = plt.subplots(len(change), len(magnet_settings), figsize=(len(magnet_settings)*6, 5), 
                        tight_layout=True, sharex=True, sharey=True)
for i, shift in enumerate(shift_list):

    plot_multiple_magnet_settings(shift, magnet_settings, axs=axs[i,:])
    for j, ax in enumerate(axs[i,:]):
        ax.set_title(f'{magnet_settings[j]} q0 y-shift = {change[i]*1e3:.2f} mm')


line, env, ref = line_init(shifts=shifts)

particles = import_particles_from_hdf5(line, 'Data/secondary_particles.h5', p0c=ref['p'])


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

print(shifts)
# # plt.show()

# particle_dir = track_line(line, particles)

# plot_trajectories(particle_dir, line, n_plot=100, show_dead=show_dead)
# # xy_plot_line(line, particle_dir, ele_str='q', elementNames='Quadrupoles')
# # xy_plot_line(line, particle_dir, ele_str='dd', elementNames='Dipoles')

plt.show()