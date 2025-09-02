# %% IMPORTS
import xobjects as xo
import xtrack as xt
import xpart as xp

import h5py
import json
from line_functions import *

plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'


ctx = xo.ContextCpu()  # Use xo.ContextCupy() for GPU

# %% ENV AND LINE
u = {
    'c': 299792458,
    'c2': 299792458**2,
    'e': 1.602176634e-19, # elementary charge in C
    'rest_e': xt.ELECTRON_MASS_EV,
    'rest_p': xt.PROTON_MASS_EV,
    'm_to_cm': 1e2,
    'm_to_mm': 1e3,
    'm_to_um': 1e6,
    'cm_to_mm': 1e1,
    'cm_to_um': 1e4,
    'cm_to_m': 1e-2,
    'mm_to_m': 1e-3,
    'mm_to_cm': 1e-1,
    'mm_to_um': 1e3,
    'um_to_mm': 1e-3,
    'um_to_cm': 1e-4,
    'um_to_m': 1e-6,
    'kG_to_T': 0.1,
    'GeV_to_eV': 1e9,
    'GeV_to_kgms': 5.39e-19,
    'eV_to_kgms': 5.34e-28,
    'GeV_to_kg': 1.8e-27,
    'GeV_to_kgm2s2': 1.6e-10,
    'fig_size': (13, 8)
}


ref = { # All in natural units
    'q': 1,
    'p': p_from_E(3e9, u['rest_e']),  # E = 3 GeV, p is in eV/c
    'betx_0': 1.0,
    'alfx_0': 0.0,
    'bety_0': 1.0,
    'alfy_0': 0.0,
}

print(f"ref['p'] = {ref['p']:.5e} eV/c")

env = xt.Environment()

env['kq_p'] = grad_kG_to_k(-6.66, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  # k1 in 1/m^2
env['kq_n'] = grad_kG_to_k(28.86, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  
env['kd'] = B_T_to_k(0.219, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  # k0 in 1/m , By
env['kd_corr'] = B_T_to_k(0.026107, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e']) # Bx


sizes = { # min_x, max_x, min_y, max_y in m, start z, stop z, length in m
    'dr0': [3.6733336],
    'q0': [-0.024610, 0.024610, -0.024610, 0.024610, (4.646664-3.6733336)],
    'dr0.1': [5.903336-4.646664],
    'q1': [-0.024610, 0.024610, -0.024610, 0.024610, (6.876664-5.903336)],
    'dr1.2': [8.123336-6.876664],
    'q2': [-0.024610, 0.024610, -0.024610, 0.024610, (9.096664-8.123336)],
    'dr2.corr': [10.1115-9.096664],
    'corr': [-0.1795, 0.1795, -0.047, 0.047, 10.1115 - 9.87779],
    'drcorr.d': [12.6034-10.1115],
    'dd': [-0.022352, 0.02352, -0.063752, 0.031752, (13.5178-12.6034)],
}


env.new('a_q0', xt.LimitRect, min_x=sizes['q0'][0], max_x=sizes['q0'][1], min_y=sizes['q0'][2], max_y=sizes['q0'][3]),
env.new('a_q1', xt.LimitRect, min_x=sizes['q1'][0], max_x=sizes['q1'][1], min_y=sizes['q1'][2], max_y=sizes['q1'][3]),
env.new('a_q2', xt.LimitRect, min_x=sizes['q2'][0], max_x=sizes['q2'][1], min_y=sizes['q2'][2], max_y=sizes['q2'][3]),
env.new('a_dd_corr', xt.LimitRect, min_x=sizes['corr'][0], max_x=sizes['corr'][1], min_y=sizes['corr'][2], max_y=sizes['corr'][3]),
env.new('a_dd', xt.LimitRect, min_x=sizes['dd'][0], max_x=sizes['dd'][1], min_y=sizes['dd'][2], max_y=sizes['dd'][3]),

env.new('beampipe', xt.LimitEllipse, a=0.02, b=0.02) #beampipe of 2 cm

# Creating Line 
line = env.new_line(components=[
    env.new('dr0', xt.Drift, length=sizes['dr0'][0]),
    env.place('beampipe'),
    env.new('q0', xt.Quadrupole, length=sizes['q0'][-1], k1='kq_p'),
    env.place('a_q0'),
    env.new('dr0.1', xt.Drift, length=sizes['dr0.1'][0]),
    env.place('beampipe'),
    env.new('q1', xt.Quadrupole, length=sizes['q1'][-1], k1='kq_n'),
    env.place('a_q1'),
    env.new('dr1.2', xt.Drift, length=sizes['dr1.2'][0]),
    env.place('beampipe'),
    env.new('q2', xt.Quadrupole, length=sizes['q2'][-1], k1='kq_p'),
    env.place('a_q2'),
    env.new('dr2.corr', xt.Drift, length=sizes['dr2.corr'][0]),
    env.place('beampipe'),
    env.new('dd_corr', xt.Bend, length=sizes['corr'][-1],k0 ='kd_corr'), # creates By field
    env.place('a_dd_corr'),
    env.new('drcorr.d', xt.Drift, length=sizes['drcorr.d'][0]),
    env.place('beampipe'),
    env.new('dd', xt.Bend, length=sizes['dd'][-1], rot_s_rad=-np.pi/2, k0='kd'), # Bx field
    env.place('a_dd'),
    env.new('dr_end', xt.Drift, length=1.0),
])


# Need to input in natural units
line.particle_ref = xt.Particles( 
    p0c=ref['p'],
    mass0=xt.ELECTRON_MASS_EV,
    q0=ref['q'],
)

line.build_tracker()

# Twiss
init = xt.TwissInit(betx=ref['betx_0'], alfx=ref['alfx_0'], bety=ref['bety_0'], alfy=ref['alfy_0'])  # example values

tw = line.twiss(
    method='4d',
    init=init,
    end='_end_point',
)

# Beam size investigation
def plot_beam_size():
    # Transverse normalized emittances
    nemitt_x = 2.5e-6
    nemitt_y = 2.5e-6

    # Longitudinal emittance from energy spread
    sigma_pzeta = 2e-4
    gemitt_zeta = sigma_pzeta**2 * 1.0
    # similarly, if the bunch length is known, the emittance can be computed as
    # gemitt_zeta = sigma_zeta**2 / tw.bets0

    tt = line.get_table()

    # Compute beam sizes
    beam_sizes = tw.get_beam_covariance(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                        gemitt_zeta=gemitt_zeta)

    # Inspect beam sizes (table can be accessed similarly to twiss tables)
    beam_sizes.show()

    sv = line.survey()
    sv.plot()

    # Plot
    fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
    spbet = plt.subplot(3,1,1)
    spdisp = plt.subplot(3,1,2, sharex=spbet)
    spbsz = plt.subplot(3,1,3, sharex=spbet)

    spbet.plot(tw.s, tw.betx, 'b-', label=r'$\beta_x$')
    spbet.plot(tw.s, tw.bety, 'r-', label=r'$\beta_y$')
    spbet.set_ylabel(r'$\beta_{x,y}$ [m]')
    spbet.legend(loc='best')
    spbet.grid(True)
    spbet.set_title('Optical Functions')

    spdisp.plot(tw.s, tw.dx, 'b-', label=r'$D_x$')
    spdisp.plot(tw.s, tw.dy, 'r-', label=r'$D_y$')
    spdisp.set_ylabel(r'$D_{x,y}$ [m]')
    spdisp.legend(loc='best')
    spdisp.grid(True)

    spbsz.plot(beam_sizes.s, beam_sizes.sigma_x, 'b-', label=r'$\sigma_x$')
    spbsz.plot(beam_sizes.s, beam_sizes.sigma_y, 'r-', label=r'$\sigma_y$')
    spbsz.set_ylabel(r'$\sigma_{x,y}$ [m]')
    spbsz.set_xlabel('s [m]')
    spbsz.legend(loc='best')
    spbsz.grid(True)

    # Add element markers
    for ax in [spbet, spdisp, spbsz]:
        ylim = ax.get_ylim()
        for name, s_pos in zip(tt.name, tt.s):
            if 'q' in name.lower():
                ax.axvline(x=s_pos, color='g', alpha=0.3, linestyle='--')
            elif 'dd' in name.lower() and len(name) > 1:  # Avoid drift elements
                ax.axvline(x=s_pos, color='m', alpha=0.3, linestyle='--')
        ax.set_ylim(ylim)

    fig1.subplots_adjust(left=.15, right=.92, hspace=.27)

# plot_beam_size()

# %% 

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

particles = import_particles_from_hdf5('Data/secondary_particles.h5', ref['p'])
pt = particles.get_table()

tt = line.get_table()
print(tt)

def track_line(line, particles, plot=True):
    # Track particles through each element and plot the divergence
    tt = line.get_table()
    elements_names = [el for el in line.element_names]
    print(f"Elements in the line: {elements_names}")

    # Create a copy of the particles to track
    tracked_particles = particles.copy()
    # Initialize data structures to store particle coordinates

    s_values = np.zeros((len(elements_names)+1, 1))

    # First, plot the initial distribution
    if plot:
        plot_divergence(particles.x, particles.px, particles.y, particles.py, title="Initial distribution")

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
        particle_list.append(tracked_particles.copy())


    return particle_list, s_values

particle_list, s_values = track_line(line, particles, plot=True)
print("Tracked line.")


def phase_plot_line(line, particle_list):
    """
    Generate phase plane plots for each drift section in the beam line.
    
    For each drift (except the last one), create a figure with 6 subplots:
    - Phase plane histogram for the drift itself (x-px and y-py)
    - Phase plane histogram for the adjacent aperture (x-px and y-py)
    - Phase plane histogram for the adjacent magnet (x-px and y-py)
    
    For the last drift, create a figure with just 2 subplots showing its phase plane histogram.
    """
    # Get all element names in the line
    element_names = line.element_names
    
    # Identify all drifts in the line
    drift_elements = [name for name in element_names if name.startswith('dr')]
    print("...Plotting phase planes...")
    
    alive_particles = []
    for p in particle_list:
        alive_particles.append(p.filter(p.state > 0))
        print(len(alive_particles[-1].x), end=' ')

    print()

    
    # Iterate through each drift except the last one
    for i, drift_name in enumerate(drift_elements[:-1]):
        # Find the drift index in the element_names list
        drift_idx = element_names.index(drift_name)
        
        # Get adjacent elements: aperture and magnet
        # Typically drift -> beampipe -> magnet pattern -> magnet apr
        beampipe_idx = drift_idx + 1
        magnet_idx = drift_idx + 3
        
        # Make sure indices are valid
        if magnet_idx >= len(element_names):
            continue

        # +1 because I want _after_ the element    
        drift_particles = alive_particles[drift_idx+1]
        aperture_particles = alive_particles[beampipe_idx+1]
        magnet_particles = alive_particles[magnet_idx+1]
        
        # Create a figure with 6 subplots: 2 rows (x and y) and 3 columns (drift, aperture, magnet)
        fig, axs = plt.subplots(2, 3, figsize=u['fig_size'],
                                sharex='col', sharey='row', tight_layout=True)
        fig.suptitle(f"Phase Plane Histograms for {drift_name} and Adjacent Elements", fontsize=16)
        
        # Column titles
        col_titles = [drift_name, element_names[beampipe_idx], element_names[magnet_idx]]
        for j, title in enumerate(col_titles):
            axs[0, j].set_title(f"after {title}")
        
        # Row labels
        axs[0, 0].set_ylabel("p_x / p_0")
        axs[1, 0].set_ylabel("p_y / p_0")

        # X-PX histograms (top row)
        for j, particles in enumerate([drift_particles, aperture_particles, magnet_particles]):
            h, _, _, im = axs[0, j].hist2d(particles.x, particles.px, bins=(100, 100), rasterized=True)
            axs[0, j].set_xlabel('$x$ [m]')
            axs[0, j].grid(True, linewidth=0.25, alpha=0.25)
            fig.colorbar(im, ax=axs[0, j])
        
        # Y-PY histograms (bottom row)
        for j, particles in enumerate([drift_particles, aperture_particles, magnet_particles]):
            h, _, _, im = axs[1, j].hist2d(particles.y, particles.py, bins=(100, 100), rasterized=True)
            axs[1, j].set_xlabel('$y$ [m]')
            axs[1, j].grid(True, linewidth=0.25, alpha=0.25)
            fig.colorbar(im, ax=axs[1, j])
        
        plt.subplots_adjust(top=0.9)


        print(f"Finished {drift_name}")

    print("Plotting last element..")
    # Special handling for the last drift
    last_drift = drift_elements[-1]
    last_drift_idx = element_names.index(last_drift)
    last_drift_particles = alive_particles[last_drift_idx]
    
    # Create a figure with 2 subplots just for the last drift
    fig, axs = plt.subplots(1, 2, figsize=u['fig_size'])
    fig.suptitle(f"Phase Plane Histograms for {last_drift}", fontsize=16)
    
    # X-PX histogram
    h, _, _, im = axs[0].hist2d(last_drift_particles.x, last_drift_particles.px, bins=(100, 100), rasterized=True)
    axs[0].set_xlabel('$x$ [m]')
    axs[0].set_ylabel('p_x/p_0')
    axs[0].grid(True, linewidth=0.25, alpha=0.25)
    fig.colorbar(im, ax=axs[0], label='Counts')
    
    # Y-PY histogram
    h, _, _, im = axs[1].hist2d(last_drift_particles.y, last_drift_particles.py, bins=(100, 100), rasterized=True)
    axs[1].set_xlabel('$y$ [m]')
    axs[1].set_ylabel('p_y/p_0')
    axs[1].grid(True, linewidth=0.25, alpha=0.25)
    fig.colorbar(im, ax=axs[1], label='Counts')
    


def xy_plot_line(line, particle_list, ele_str, elementNames, n_bin=100):
    """
    Generate XY plots for quadrupoles in the beam line.
    For each quadrupole, create a figure showing the XY distribution 
    at both the entrance (before quad) and exit (after quad).
    """
    # Get all element names in the line
    element_names = line.element_names
    
    # Identify all quadrupoles in the line
    plot_elements = [name for name in element_names if name.startswith(ele_str)]
    print("...Plotting XY pictures for plotted elements...")
    
    alive_particles = []
    for p in particle_list:
        alive_particles.append(p.filter(p.state > 0))
    
    # Create one figure with 3x2 subplots (3 quads, entrance and exit)
    fig, axs = plt.subplots(2, len(plot_elements), figsize=u['fig_size'],
                             sharex=True, sharey=True, tight_layout=True)
    fig.suptitle(f"XY Distribution at {elementNames} Entrances and Exits", fontsize=16)
    
    for i, ele_names in enumerate(plot_elements):

        # Find the quadrupole index in the element_names list
        ele_idx = element_names.index(ele_names)
        
        # Get particles at entrance (element before ele) and exit (after ele)
        entrance_particles = alive_particles[ele_idx]  # Before the ele
        exit_particles = alive_particles[ele_idx+2]    # After the aperture of the ele

        # Plot entrance distribution (top row)
        h, _, _, im = axs[0, i].hist2d(entrance_particles.x, entrance_particles.y, 
                                     bins=(n_bin, n_bin), rasterized=True)
        axs[0, i].set_title(f"{ele_names} entrance")
        axs[0, i].set_xlabel('x [m]')
        axs[0, i].set_ylabel('y [m]')
        # axs[0, i].grid(True, linewidth=0.25, alpha=0.25)
        fig.colorbar(im, ax=axs[0, i])
        
        # Plot exit distribution (bottom row)
        h, _, _, im = axs[1, i].hist2d(exit_particles.x, exit_particles.y, 
                                     bins=(n_bin, n_bin), rasterized=True)
        axs[1, i].set_title(f"{ele_names} exit")
        axs[1, i].set_xlabel('x [m]')
        axs[1, i].set_ylabel('y [m]')
        # axs[1, i].grid(True, linewidth=0.25, alpha=0.25)
        fig.colorbar(im, ax=axs[1, i])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    print(f"Finished plotting {elementNames} XY distributions.")



# phase_plot_line(line, particle_list)
xy_plot_line(line, particle_list, ele_str='q', elementNames='Quadrupoles', n_bin=300)
xy_plot_line(line, particle_list, ele_str='dd', elementNames='Dipoles', n_bin=300)
print("Finished creating plots of phase planes.")
# plt.show()

print("Plotted phase planes.")

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

        if loss_step == x_values.shape[0]:
            # If particle survived, use the original blue and red
            axes[0].plot(s_values, x_values[:, idx], 'b-', alpha=0.3, linewidth=0.5)
            axes[1].plot(s_values, y_values[:, idx], 'r-', alpha=0.3, linewidth=0.5)
        else:
            # If particle died, use purple for x and yellow for y
            axes[0].plot(s_values[:loss_step], x_values[:loss_step, idx], 'purple', alpha=0.3, linewidth=0.5)
            axes[1].plot(s_values[:loss_step], y_values[:loss_step, idx], 'yellow', alpha=0.3, linewidth=0.5)
            # Mark the loss point with a scatter point
            axes[0].scatter(s_values[loss_step-1], x_values[loss_step-1, idx], color='k', s=9, alpha=0.7)
            axes[1].scatter(s_values[loss_step-1], y_values[loss_step-1, idx], color='k', s=9, alpha=0.7)

    alive_particles = []
    for p in particle_list:
        alive_particles.append(p.filter(final_alive))
        print(len(alive_particles[-1].x), end=' ')

    x_alive = [p.x for p in alive_particles]
    y_alive = [p.y for p in alive_particles]  # shape = (num_elements+1, num_particles_alive)
    x_alive = np.array(x_alive)
    y_alive = np.array(y_alive)

    mean_x = np.mean(x_alive, axis=1)
    mean_y = np.mean(y_alive, axis=1)
    axes[0].plot(s_values, mean_x, 'b-', linewidth=2, label='Mean x')
    axes[1].plot(s_values, mean_y, 'r-', linewidth=2, label='Mean y')
    axes[0].legend()
    axes[1].legend()
    # Add element positions
    for name, s_pos in zip(tt.name, tt.s):
        for ax in axes:
            if 'a_q' in name.lower() and not name.startswith('d'):
                ax.axvline(x=s_pos, color='g', alpha=0.5, linestyle='--', label=name if 'q' in locals() else '_')
                ax.text(s_pos, ax.get_ylim()[1]*0.9, name, rotation=90, verticalalignment='top')
            elif 'a_dd' in name.lower():
                ax.axvline(x=s_pos, color='m', alpha=0.5, linestyle='--', label=name if 'dd' in locals() else '_')
                ax.text(s_pos, ax.get_ylim()[1]*0.9, name, rotation=90, verticalalignment='top')



plot_trajectories(particle_list, s_values, n_plot=190)


plt.show()
