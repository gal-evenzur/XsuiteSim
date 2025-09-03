# %% IMPORTS
import xobjects as xo
import xtrack as xt
import xpart as xp

import h5py
from line_functions import *

plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'
# 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

ctx = xo.ContextCpu()  # Use xo.ContextCupy() for GPU

# %% ENV AND UNITS
ref = { # All in natural units
    'q': 1,
    'p': p_from_E(3e9, u['rest_e']),  # E = 3 GeV, p is in eV/c
    'betx_0': 1.0,
    'alfx_0': 0.0,
    'bety_0': 1.0,
    'alfy_0': 0.0,
}


sizes = { # min_x, max_x, min_y, max_y in m, start z, stop z, length in m
    'dr0': [3.6733336],
    'q0': [-0.01, 0.01, -0.005, 0.005, 4.646664-3.6733336],
    'dr0.1': [5.903336-4.646664],
    'q1': [-0.024610, 0.024610, -0.024610, 0.024610, 6.876664-5.903336],
    'dr1.2': [8.123336-6.876664],
    'q2': [-0.024610, 0.024610, -0.024610, 0.024610, 9.096664-8.123336],
    'dr2.corr': [10.1115-9.096664],
    'corr': [-0.1795, 0.1795, -0.047, 0.047, 10.1115 - 9.87779],
    'drcorr.d': [12.6034-10.1115],
    'dd': [-0.022352, 0.02352, -0.063752, 0.031752, 13.5178-12.6034],
    'm0': [detector_x_center_m-chipYm/2., detector_x_center_m+chipYm/2.,
           detector_y_center_m-chipXm/2.,detector_y_center_m+chipXm/2.,
           detector_z_base_m]
}


print(f"ref['p'] = {ref['p']:.5e} eV/c")

env = xt.Environment()
env['kq_p'] = grad_kG_to_k(Grad1, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  # k1 in 1/m^2
env['kq_n'] = grad_kG_to_k(Grad2, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  
env['kd'] = B_T_to_k(B_dd, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  # k0 in 1/m , Bx --> +yhat
env['kd_corr'] = B_T_to_k(B_dd_xcorr, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e']) # By --> -xhat

# %% -*-*--*-*-Line creation
env.new('a_q0', xt.LimitRect, min_x=sizes['q0'][0], max_x=sizes['q0'][1], min_y=sizes['q0'][2], max_y=sizes['q0'][3]),
env.new('a_q1', xt.LimitRect, min_x=sizes['q1'][0], max_x=sizes['q1'][1], min_y=sizes['q1'][2], max_y=sizes['q1'][3]),
env.new('a_q2', xt.LimitRect, min_x=sizes['q2'][0], max_x=sizes['q2'][1], min_y=sizes['q2'][2], max_y=sizes['q2'][3]),
env.new('a_dd_corr', xt.LimitRect, min_x=sizes['corr'][0], max_x=sizes['corr'][1], min_y=sizes['corr'][2], max_y=sizes['corr'][3]),
env.new('a_dd', xt.LimitRect, min_x=sizes['dd'][0], max_x=sizes['dd'][1], min_y=sizes['dd'][2], max_y=sizes['dd'][3]),

env.new('beampipe', xt.LimitEllipse, a=0.02, b=0.02) #beampipe of 2 cm

# Monitor at the end
env.new('a_m0', xt.LimitRect, min_x=sizes['m0'][0], max_x=sizes['m0'][1], min_y=sizes['m0'][2], max_y=sizes['m0'][3]),
env.elements['m0'] = xt.ParticlesMonitor(num_particles=n_particles,
                                start_at_turn=0, stop_at_turn=1,
                                auto_to_numpy=True)


# Creating Line 
# Order: drift - beampipe - quadrupole - aperture
'''
line = env.new_line(components=[
    env.new('dr0', xt.Drift, length=sizes['dr0'][0]),
    env.place('beampipe'),
    env.new('q0', xt.Quadrupole, length=sizes['q0'][-1], k1='kq_p'),
    # env.place('a_q0'),
    env.new('dr0.1', xt.Drift, length=sizes['dr0.1'][0]),
    env.place('beampipe'),
    env.new('q1', xt.Quadrupole, length=sizes['q1'][-1], k1='kq_n'),
    # env.place('a_q1'),
    env.new('dr1.2', xt.Drift, length=sizes['dr1.2'][0]),
    env.place('beampipe'),
    env.new('q2', xt.Quadrupole, length=sizes['q2'][-1], k1='kq_p'),
    # env.place('a_q2'),
    env.new('dr2.corr', xt.Drift, length=sizes['dr2.corr'][0]),
    env.place('beampipe'),
    env.new('dd_corr', xt.Bend, length=sizes['corr'][-1],k0 ='kd_corr'), # creates By field
    # env.place('a_dd_corr'),
    env.new('drcorr.d', xt.Drift, length=sizes['drcorr.d'][0]),
    env.place('beampipe'),
    env.new('dd', xt.Bend, length=sizes['dd'][-1], rot_s_rad=-np.pi/2, k0='kd'), # Bx field
    # env.place('a_dd'),
    # env.place('a_m0', at=sizes['m0'][-1]),
    env.place('m0', at=sizes['m0'][-1]),
])
'''

line = env.new_line(components=[

    env.new('dr0', xt.Drift, length=sizes['dr0'][0]),
    # env.place('beampipe'),
    # env.new('q0', xt.Quadrupole, length=sizes['q0'][-1], k1='kq_p'),
    env.place('a_q0'),
    env.new('dr0.1', xt.Drift, length=sizes['dr0.1'][0]),
    # env.place('beampipe'),
    # env.new('q1', xt.Quadrupole, length=sizes['q1'][-1], k1='kq_n'),
    # env.place('a_q1'),
    env.new('dr1.2', xt.Drift, length=sizes['dr1.2'][0]),
    # env.place('beampipe'),
    # env.new('q2', xt.Quadrupole, length=sizes['q2'][-1], k1='kq_p'),
    # env.place('a_q2'),
    # env.new('dr2.corr', xt.Drift, length=sizes['dr2.corr'][0]),
])


# Need to input in natural units
line.particle_ref = xt.Particles( 
    p0c=ref['p'],
    mass0=xt.ELECTRON_MASS_EV,
    q0=ref['q'],
)

line.build_tracker()


# %% !!!! Particles and tracking !!!!

# Function to import particles from HDF5 file
def import_particles_from_hdf5(filename, p0c):
    """
    Import particles from an HDF5 file created by particle_generation.py
    
    Args:
        filename: Path to the HDF5 file
    
    Returns:
        xpart.Particles: Particle object for tracking
    """
    print(f"Loading particles from {filename}")
    with h5py.File(filename, 'r') as f:
        # Extract the 6D phase space coordinates
        x_coords = f['x'][:] # [m]
        y_coords = f['y'][:] # [m]
        z_coords = f['z'][:] # [m]
        px_coords = f['px'][:] # [GeV/c]
        py_coords = f['py'][:] # [GeV/c]
        pz_coords = f['pz'][:] # [GeV/c]

        px_eV = px_coords * u['GeV_to_eV']
        py_eV = py_coords * u['GeV_to_eV']
        pz_eV = pz_coords * u['GeV_to_eV']

        p = np.sqrt(px_eV**2 + py_eV**2 + pz_eV**2)

        px = px_eV / p0c # dimensionless
        py = py_eV / p0c # dimensionless

        delta = (p - p0c) / p0c  # dimensionless
        

        # Get number of particles
        num_particles = f.attrs['num_particles']
        print(f"Loaded {num_particles} particles")
        
        # Print min/max values to verify data
        print(f"x range: [{np.min(x_coords):.6f}, {np.max(x_coords):.6f}] m")
        print(f"y range: [{np.min(y_coords):.6f}, {np.max(y_coords):.6f}] m")
        print(f"z range: [{np.min(z_coords):.6f}, {np.max(z_coords):.6f}] m")
        print(f"px range: [{np.min(px):.6f}, {np.max(px):.6f}]")
        print(f"py range: [{np.min(py):.6f}, {np.max(py):.6f}]")
        print(f"delta range: [{np.min(delta):.6f}, {np.max(delta):.6f}]")

        # Create the particle object for tracking
        particles = xp.Particles(
            x=x_coords,
            px=px,
            y=y_coords,
            py=py,
            zeta=z_coords,
            delta=delta,  # delta = (pz [eV/c] - p0 [eV/c]) / p0
            _context=ctx,
        )
        
        return particles

# particles = import_particles_from_hdf5('Data/secondary_particles.h5', ref['p'])
particles = line.build_particles(x=[0, 0.01],y=[0.01, 0],px=[1e-4, -1e-4], py=[1e-4, 0])
# particles = line.build_particles(x=[0],y=[0.01],px=[1e-4], py=[1e-4])


tt = line.get_table()
print(tt)

def track_line(line, particles):
    # Track particles through each element and plot the divergence
    tt = line.get_table()
    elements_names = [el for el in line.element_names]
    print(f"Elements in the line: {elements_names}")

    # Create a copy of the particles to track
    tracked_particles = particles.copy()
    
    # Initialize data structures to store particle coordinates
    s_values = np.zeros((len(elements_names)+1, 1))
    x_values = np.zeros((len(elements_names)+1, len(tracked_particles.x)))
    y_values = np.zeros((len(elements_names)+1, len(tracked_particles.x)))
    
    # Store initial positions
    x_values[0, :] = tracked_particles.x
    y_values[0, :] = tracked_particles.y
    
    particle_list = [tracked_particles.copy()]

    # Track through each element individually
    for i, element_name in enumerate(elements_names):
        s_start = tt.rows[i].s
        s_start = s_start[0]
        s_stop = tt.rows[i+1].s
        s_stop = s_stop[0]
        print(f"ELEMENT {i}: {element_name} || s={s_start:.3f}:{s_stop:.3f} m")

        s_values[i+1] = s_stop

        # Track through this single element
        line.track(tracked_particles, ele_start=element_name, num_elements=1)
        
        # Store particle positions after this element
        x_values[i+1, :] = tracked_particles.x
        y_values[i+1, :] = tracked_particles.y

        par_to_list = tracked_particles.copy()
        par_to_list.sort(interleave_lost_particles=True)
        particle_list.append(par_to_list)

        # Print particle IDs before and after tracking
        print(f"Before tracking element {element_name}: Particle IDs = {tracked_particles.particle_id}")
        line.track(tracked_particles, ele_start=element_name, num_elements=1)
        print(f"After tracking element {element_name}: Particle IDs = {tracked_particles.particle_id}")

    return particle_list, s_values, x_values, y_values


# %% [] Varius plots []

particle_list, s_values, x_values, y_values = track_line(line, particles)
print("Tracked line.")

def plot_tracks(line, x_values, y_values, s_values, n_plot=100):
    tt = line.get_table()
    # Create a figure for particle trajectories
    fig, axes = plt.subplots(1, 2, figsize=u['fig_size'])

    # Select a subset of particles for better readability (max 100 particles)
    num_to_plot = min(n_plot, particles.x.size)
    particle_indices = np.random.choice(particles.x.size, num_to_plot, replace=False)


    # Plot trajectories with truncation at loss point
    for idx in particle_indices:
        idx = int(idx)

        if particle_list[-1].state[idx] > 0:
            # If particle survived, use the original blue and red
            axes[0].plot(s_values, x_values[:, idx], 'b-', alpha=0.3, linewidth=0.5)
            axes[1].plot(s_values, y_values[:, idx], 'r-', alpha=0.3, linewidth=0.5)
        else:
            # If particle died, use purple for x and yellow for y
            axes[0].plot(s_values[:], x_values[:, idx], 'purple', alpha=0.3, linewidth=0.5)
            axes[1].plot(s_values[:], y_values[:, idx], 'magenta', alpha=0.3, linewidth=0.5)
            # Mark the loss point with a scatter point
            # axes[0].scatter(s_values[loss_step-1], x_values[loss_step-1, idx], color='k', s=9, alpha=0.7)
            # axes[1].scatter(s_values[loss_step-1], y_values[loss_step-1, idx], color='k', s=9, alpha=0.7)
    # Add titles to the plots
    axes[0].set_title("Particle Trajectories in X")
    axes[1].set_title("Particle Trajectories in Y")

    # Add element markers
    for name, s_pos in zip(tt.name, tt.s):
        for ax in axes:
            ylim = ax.get_ylim()
            if 'a_' in name.lower():  # Apertures - black lines
                ax.axvline(x=s_pos, color='k', alpha=0.5, linestyle='--')
                ax.text(s_pos, ylim[1]*0.9, name, rotation=90, verticalalignment='top', fontsize=8)
            elif name.startswith('dd'):  # Dipoles - pink lines
                ax.axvline(x=s_pos, color='magenta', alpha=0.5, linestyle='--')
                ax.text(s_pos, ylim[1]*0.8, name, rotation=90, verticalalignment='top', fontsize=8)
            elif name.startswith('q'):  # Quadrupoles - green lines
                ax.axvline(x=s_pos, color='green', alpha=0.5, linestyle='--')
                ax.text(s_pos, ylim[1]*0.7, name, rotation=90, verticalalignment='top', fontsize=8)
            ax.set_ylim(ylim)




def plot_trajectories(particle_list, s_values, n_plot=100):
    x_values = [p.x for p in particle_list]
    y_values = [p.y for p in particle_list]  # shape = (num_elements+1, num_particles)

    s_values = np.array(s_values)
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Create a figure for particle trajectories
    fig, axes = plt.subplots(1, 2, figsize=u['fig_size'])

    # Select a subset of particles for better readability (max 100 particles)
    num_to_plot = min(n_plot, particles.x.size)
    particle_indices = np.random.choice(particles.x.size, num_to_plot, replace=False)

    particle_lost_at = particle_lost_at_step(particle_list)
    # Get the final state to know which particles survived
    final_alive = particle_list[-1].state > 0

    # Plot trajectories with truncation at loss point
    for idx in particle_indices:
        idx = int(idx)
        loss_step = particle_lost_at[idx]
        loss_step = int(loss_step)

        if particle_list[-1].state[idx] > 0:
            # If particle survived, use the original blue and red
            axes[0].plot(s_values, x_values[:, idx], 'b-', alpha=0.3, linewidth=2 
                         )
            axes[1].plot(s_values, y_values[:, idx], 'r-', alpha=0.3, linewidth=2)
        else:
            # If particle died, use purple for x and yellow for y
            axes[0].plot(s_values[:], x_values[:, idx], 'purple', alpha=0.3, linewidth=2)
            axes[1].plot(s_values[:], y_values[:, idx], 'magenta', alpha=0.3, linewidth=2)
            # Mark the loss point with a scatter point
            axes[0].scatter(s_values[loss_step-1], x_values[loss_step-1, idx], color='k', s=9, alpha=0.7)
            axes[1].scatter(s_values[loss_step-1], y_values[loss_step-1, idx], color='k', s=9, alpha=0.7)

    alive_particles = []
    for i, p in enumerate(particle_list):
        alive_particles.append(p.filter(final_alive))
        print(len(alive_particles[i].x), end=' ')

    print()
    living_particles = []
    for i, p in enumerate(particle_list):
        living_particles.append(p.filter(p.state > 0))
        print(len(living_particles[i].x), end=' ')


    x_alive = [p.x for p in alive_particles]
    y_alive = [p.y for p in alive_particles]  # shape = (num_elements+1, num_particles_alive)
    x_alive = np.array(x_alive)
    y_alive = np.array(y_alive)

    mean_x = np.mean(x_alive, axis=1)
    mean_y = np.mean(y_alive, axis=1)
    # axes[0].plot(s_values, mean_x, 'b-', linewidth=2, label='Mean x')
    # axes[1].plot(s_values, mean_y, 'r-', linewidth=2, label='Mean y')
    axes[0].legend()
    axes[1].legend()
    # Add element positions
    # Add element markers
    for name, s_pos in zip(tt.name, tt.s):
        for ax in axes:
            ylim = ax.get_ylim()
            if 'a_' in name.lower():  # Apertures - black lines
                ax.axvline(x=s_pos, color='k', alpha=0.5, linestyle='--')
                ax.text(s_pos, ylim[1]*0.9, name, rotation=90, verticalalignment='top', fontsize=8)
            elif name.startswith('dd'):  # Dipoles - pink lines
                ax.axvline(x=s_pos, color='magenta', alpha=0.5, linestyle='--')
                ax.text(s_pos, ylim[1]*0.8, name, rotation=90, verticalalignment='top', fontsize=8)
            elif name.startswith('q'):  # Quadrupoles - green lines
                ax.axvline(x=s_pos, color='green', alpha=0.5, linestyle='--')
                ax.text(s_pos, ylim[1]*0.7, name, rotation=90, verticalalignment='top', fontsize=8)
            ax.set_ylim(ylim)


plot_tracks(line, x_values, y_values, s_values, n_plot=10)
# plot_trajectories(particle_list, s_values, n_plot=190)
plt.show()

