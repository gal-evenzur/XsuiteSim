# %% IMPORTS
import xobjects as xo
import xtrack as xt

import h5py
from sim_functions import *
from params import *

plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'
# 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

ctx = xo.ContextCpu()  # Use xo.ContextCupy() for GPU

line, env, ref = line_init(shifts=shifts)

particles = import_particles_from_hdf5(line, 'Data/secondary_particles.h5', p0c=ref['p'])


# %% [] Varius plots []
# First, plot the initial distribution
# plot_divergence(particles.x, particles.px, particles.y, particles.py, title="Initial distribution")

# particle_list, s_values = track_line(line, particles)
print("Tracked line.")


# phase_plot_line(line, particle_list)
# xy_plot_line(line, particle_list, ele_str='q', elementNames='Quadrupoles', n_bin=300)
# xy_plot_line(line, particle_list, ele_str='dd', elementNames='Dipoles', n_bin=300)
print("Finished creating plots of phase planes.")
# plt.show()

print("Plotted phase planes.")


# plot_trajectories(particle_list, line, s_values, n_plot=190, show_dead=show_dead)

# %% Monitors!!
line_monitor = line.copy()
h, xedges, yedges = track_monitor(line_monitor, particles)
plt.imshow(h.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
plt.grid()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Monitor at the end of the line')
plt.colorbar(label='Counts per bin')
plt.tight_layout()

plt.show()
# test_integration_models(line, particles)

particle_list, s_values = track_line(line, particles)
phase_plot_line(line, particle_list)
xy_plot_line(line, particle_list, ele_str='q', elementNames='Quadrupoles', n_bin=300)
xy_plot_line(line, particle_list, ele_str='dd', elementNames='Dipoles', n_bin=300)
print("Finished creating plots of phase planes.")   

test_integration_models(line, particles)
plt.show()