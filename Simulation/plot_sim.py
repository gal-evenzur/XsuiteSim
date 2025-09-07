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
tt = line.get_table()

tt.show()

particles = import_particles_from_hdf5(line, 'Data/secondary_particles.h5', p0c=ref['p'])

h, xedges, yedges = track_monitor(line, particles)
plt.imshow(h.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
plt.grid()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Monitor at the end of the line')
plt.colorbar(label='Counts per bin')
plt.tight_layout()

plt.show()

particle_list, s_values = track_line(line, particles)

plot_trajectories(particle_list, line, s_values, n_plot=100, show_dead=show_dead)

plt.show()