# %% IMPORTS
import xobjects as xo
import xtrack as xt

import h5py
from sim_functions import *
from params import *

plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'

ctx = xo.ContextCpu()  # Use xo.ContextCupy() for GPU

line, env, ref = line_init(shifts=shifts)
tt = line.get_table()

tt.show()

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

# plt.show()

particle_dir = track_line(line, particles)

plot_trajectories(particle_dir, line, n_plot=100, show_dead=show_dead)
# xy_plot_line(line, particle_dir, ele_str='q', elementNames='Quadrupoles')
# xy_plot_line(line, particle_dir, ele_str='dd', elementNames='Dipoles')

plt.show()