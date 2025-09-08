from matplotlib.ticker import AutoMinorLocator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import xtrack as xt
from params import *
plt.rcParams['image.cmap'] = 'afmhot'
# %% +++++++++Monitor sizes 
npix_x = 1024
npix_y = 512
pix_x  = 0.02924
pix_y  = 0.02688  
chipXmm = npix_x*pix_x
chipYmm = npix_y*pix_y
chipXcm = chipXmm*u['mm_to_cm']
chipYcm = chipYmm*u['mm_to_cm']
chipXm  = chipXmm*u['mm_to_m']
chipYm  = chipYmm*u['mm_to_m']
dy_det  = 0 # cm

particle_start_pos = Z0 # m
# Define detector x range
detector_x_center_cm = -1.0 # cm
# detector_x_center_cm = 0. # cm
detector_x_center_m = detector_x_center_cm*u['cm_to_m']

# Define detector y range
detector_y_center_cm = 5.165 + 0.1525 + 3.685 + dy_det # cm
detector_y_center_m  = detector_y_center_cm*u['cm_to_m']

# Calculate detector z position
detector_z_base_cm = 1363 + 303.2155 + 11.43 + 1.05 - particle_start_pos*u['m_to_cm'] # cm
detector_z_base_m  = detector_z_base_cm*u['cm_to_m']
detector_z_base_mm = detector_z_base_cm*u['cm_to_mm']


sizes = { # min_x, max_x, min_y, max_y in m, start z, stop z, length in m
    'dr0': [3.6733336-particle_start_pos],
    'q0': [-0.024610, 0.024610, -0.024610, 0.024610, 4.646664-3.6733336],
    'dr0.1': [5.903336-4.646664],
    'q1': [-0.024610, 0.024610, -0.024610, 0.024610, 6.876664-5.903336],
    'dr1.2': [8.123336-6.876664],
    'q2': [-0.024610, 0.024610, -0.024610, 0.024610, 9.096664-8.123336],
    'dr2.corr': [9.87779-9.096664],
    'corr': [-0.1795, 0.1795, -0.047, 0.047, 10.1115 - 9.87779],
    'drcorr.d': [12.6034-10.1115],
    'dd': [-0.022352, 0.02352, -0.063752, 0.031752, 13.5178-12.6034],
    'm0': [detector_x_center_m-chipYm/2., detector_x_center_m+chipYm/2.,
           detector_y_center_m-chipXm/2.,detector_y_center_m+chipXm/2.,
           detector_z_base_m],
    'pipe': 0.02 # radius in m
}


# %% INITIALIZATION

def p_from_E(E, E_rest):
    # m is in eV / c2
    # E_rest = m * c2
    # E is in eV
    # p is in eV / c
    p = (E**2 - (E_rest)**2)**0.5 #p is in eV/c
    return p

ref = { # All in natural units
    'q': 1,
    'p': p_from_E(Eavg, u['rest_e']),  # E = 3 GeV, p is in eV/c
    'betx_0': 1.0,
    'alfx_0': 0.0,
    'bety_0': 1.0,
    'alfy_0': 0.0,
}

def grad_kG_to_k(grad_kG, p_mks, q_mks):
    kG_to_T = 0.1
    grad_T = grad_kG * kG_to_T  # grad in T/m 
    k = q_mks * grad_T / p_mks  # k in 1/m
    return k

def B_T_to_k(B_T, p_mks, q_mks):
    k = q_mks * B_T / p_mks  # k in 1/m
    return k

# %% RUNNIN LINE---------------------

def quadElement(env, spacer, name, k1, length, max_x, max_y, r_pipe, 
                dx=0, dy=0, ang_z=0, ang_x=0, ang_y=0):
    env.new(f'a_{name}', xt.LimitRectEllipse,
             max_x=max_x, max_y=max_y, a=r_pipe, b=r_pipe)
    
    qElement = env.new_line(components=[
        env.new(f"rots_{name}", xt.SRotation, angle=ang_z),
        spacer,
        env.new(f"rotx_{name}", xt.XRotation, angle=ang_x),
        spacer,
        env.new(f"roty_{name}", xt.YRotation, angle=ang_y),
        spacer,
        env.new(f'xy_{name}', xt.XYShift, dx=-dx, dy=-dy),
        spacer,
        f'a_{name}',
        env.new(name, xt.Quadrupole, length=length, k1=k1),
        f'a_{name}',
        spacer,
        env.new(f'xy_restore_{name}', xt.XYShift, dx=dx, dy=dy),
        spacer,
        env.new(f"roty_restore_{name}", xt.YRotation, angle=-ang_y),
        spacer,
        env.new(f"rotx_restore_{name}", xt.XRotation, angle=-ang_x),
        spacer,
        env.new(f"rots_restore_{name}", xt.SRotation, angle=-ang_z),
    ])

    return qElement

def dipoleElement(env, spacer, name, k0, length, max_x, max_y, r_pipe,
                  min_x=0, min_y=0, 
                  dx=0, dy=0, ang_z=0, ang_x=0, ang_y=0):
    env.new(f'a_{name}', xt.LimitRectEllipse,
             max_x=max_x, max_y=max_y, a=r_pipe, b=r_pipe)
    env.new(f'a_{name}_out', xt.LimitRect, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y),

    def dip(a_in, a_out):
        dElement = env.new_line(components=[
            env.new(f"rots_{name}", xt.SRotation, angle=ang_z),
            spacer,
            env.new(f"rotx_{name}", xt.XRotation, angle=ang_x),
            spacer,
            env.new(f"roty_{name}", xt.YRotation, angle=ang_y),
            spacer,
            env.new(f'xy_{name}', xt.XYShift, dx=-dx, dy=-dy),
            spacer,
            a_in,
            env.new(name, xt.Bend, length=length, k0=k0),
            a_out,
            spacer,
            env.new(f'xy_restore_{name}', xt.XYShift, dx=dx, dy=dy),
            spacer,
            env.new(f"roty_restore_{name}", xt.YRotation, angle=-ang_y),
            spacer,
            env.new(f"rotx_restore_{name}", xt.XRotation, angle=-ang_x),
            spacer,
            env.new(f"rots_restore_{name}", xt.SRotation, angle=-ang_z),
            ])
        
        return dElement


    if name=='dd': 
        dElement = dip(f'a_{name}', f'a_{name}_out')
        dElement[name].rot_s_rad = -np.pi/2  # to have Bx field
    
    else:
        dElement = dip(f'a_{name}', f'a_{name}')

    
    
    return dElement

def line_init(shifts):

    env = xt.Environment()
    env['kq_p'] = grad_kG_to_k(Grad1, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  # k1 in 1/m^2
    env['kq_n'] = grad_kG_to_k(Grad2, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  
    env['kd'] = B_T_to_k(B_dd, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  # k0 in 1/m , Bx --> +yhat
    env['kd_corr'] = B_T_to_k(B_dd_xcorr, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e']) # By --> -xhat


    # Monitor at the end
    env.new('a_m0', xt.LimitRect, min_x=sizes['m0'][0], max_x=sizes['m0'][1], min_y=sizes['m0'][2], max_y=sizes['m0'][3]),
    env.elements['m0'] = xt.ParticlesMonitor(num_particles=int(n_particles),
                                    start_at_turn=0, stop_at_turn=1,
                                    auto_to_numpy=True)

    spacer = env.new('spacer', xt.Drift, length=1e-6)
    # Creating Line 
    # Order: drift - beampipe - quadrupole - aperture
    line = env.new_line(components=[
        env.new('dr0', xt.Drift, length=sizes['dr0'][0]),
        quadElement(env, spacer, 'q0', k1='kq_p', length=sizes['q0'][-1],
                     max_x=sizes['q0'][1], max_y=sizes['q0'][3], r_pipe=sizes['pipe'],
                     dx=shifts['q0']['x'], dy=shifts['q0']['y'],
                     ang_z=shifts['q0']['ang_z'], ang_x=shifts['q0']['ang_x'], ang_y=shifts['q0']['ang_y']),

        env.new('dr0.1', xt.Drift, length=sizes['dr0.1'][0]),
        quadElement(env, spacer, 'q1', k1='kq_n', length=sizes['q1'][-1],
                     max_x=sizes['q1'][1], max_y=sizes['q1'][3], r_pipe=sizes['pipe'],
                     dx=shifts['q1']['x'], dy=shifts['q1']['y'],
                     ang_z=shifts['q1']['ang_z'], ang_x=shifts['q1']['ang_x'], ang_y=shifts['q1']['ang_y']),

        env.new('dr1.2', xt.Drift, length=sizes['dr1.2'][0]),
        quadElement(env, spacer, 'q2', k1='kq_p', length=sizes['q2'][-1],
                     max_x=sizes['q2'][1], max_y=sizes['q2'][3], r_pipe=sizes['pipe'],
                     dx=shifts['q2']['x'], dy=shifts['q2']['y'],
                     ang_z=shifts['q2']['ang_z'], ang_x=shifts['q2']['ang_x'], ang_y=shifts['q2']['ang_y']),

        env.new('dr2.corr', xt.Drift, length=sizes['dr2.corr'][0]),
        dipoleElement(env, spacer, 'dd_corr', k0='kd_corr', length=sizes['corr'][-1],
                     max_x=sizes['corr'][1], max_y=sizes['corr'][3], r_pipe=sizes['pipe'],
                     dx=shifts['dd_corr']['x'], dy=shifts['dd_corr']['y'],
                     ang_z=shifts['dd_corr']['ang_z'], ang_x=shifts['dd_corr']['ang_x'], ang_y=shifts['dd_corr']['ang_y']),
        
        env.new('drcorr.d', xt.Drift, length=sizes['drcorr.d'][0]),
        dipoleElement(env, spacer, 'dd', k0='kd', length=sizes['dd'][-1],
                     min_x=sizes['dd'][0], min_y=sizes['dd'][2],
                     max_x=sizes['dd'][1], max_y=sizes['dd'][3], r_pipe=sizes['pipe'],
                     dx=shifts['dd']['x'], dy=shifts['dd']['y'],
                     ang_z=shifts['dd']['ang_z'], ang_x=shifts['dd']['ang_x'], ang_y=shifts['dd']['ang_y']),
        
        env.place('a_m0', at=sizes['m0'][-1]),
        env.place('m0', at=sizes['m0'][-1]),
    ])

    if use_integration:
        model = 'mat-kick-mat'
        # Go through all elements in the line and update the model attribute if it exists
        for name in line.element_names:
            element = line[name]
            if hasattr(element, 'model'):
                print(f"Updating model for element {name} from {element.model} to {model}")
                element.model = model

    # Need to input in natural units
    line.particle_ref = xt.Particles( 
        p0c=ref['p'],
        mass0=xt.ELECTRON_MASS_EV,
        q0=ref['q'],
    )

    line.build_tracker()

    return line, env, ref


# Function to import particles from HDF5 file
def import_particles_from_hdf5(line, filename, p0c):
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
        num_particles = f.attrs['num_particles']

    px_eV = px_coords * u['GeV_to_eV']
    py_eV = py_coords * u['GeV_to_eV']
    pz_eV = pz_coords * u['GeV_to_eV']

    p = np.sqrt(px_eV**2 + py_eV**2 + pz_eV**2)

    px = px_eV / p0c # dimensionless
    py = py_eV / p0c # dimensionless

    delta = (p - p0c) / p0c  # dimensionless
    

    # Get number of particles
    print(f"Loaded {num_particles} particles")
    
    # Create the particle object for tracking
    particles = line.build_particles(
        x=x_coords,
        px=px,
        y=y_coords,
        py=py,
        zeta=0,
        delta=delta,  # delta = (pz [eV/c] - p0 [eV/c]) / p0
    )
        
    return particles


def histogram_monitors(line, verbose=True):

    m = [el for el in line.elements if isinstance(el, xt.ParticlesMonitor)]
    for i, mon in enumerate(m):
        x, y = np.squeeze(mon.x), np.squeeze(mon.y)
        px, py = np.squeeze(mon.px), np.squeeze(mon.py)
    
        # Filter out dead particles (those with x=y=px=py=0)
        mask = ~((x == 0) & (y == 0) & (px == 0) & (py == 0))
        x_clean = x[mask]
        y_clean = y[mask]

        if verbose: print(f"Monitor {i}: {len(x_clean)}/{len(x)} particles alive")

        if len(x_clean) > 0:  # Only plot if there are particles
            # XY spatial plot
            h, xedges, yedges = np.histogram2d(x_clean, y_clean, bins=monitor_bins)
        else:
            if verbose: print(f"No particles alive at monitor {i}")
    
    return h, xedges, yedges

def track_monitor(line, particles):
    line.track(particles.copy())

    h, xedges, yedges = histogram_monitors(line)

    return h, xedges, yedges

def track_line(line, particles):
    # Track particles through each element and plot the divergence
    tt = line.get_table()
    elements_names = [el for el in line.element_names]

    # Create a copy of the particles to track
    tracked_particles = particles.copy()
    # Initialize data structures to store particle coordinates

    s_values = np.zeros((len(elements_names)+1, 1))

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
        p_to_list = tracked_particles.copy()
        p_to_list.sort(interleave_lost_particles=True)
        particle_list.append(p_to_list)


    return particle_list, s_values

def test_integration_models(line, particles):
    """
    Test the effect of different integration models on particle tracking.
    Compare results using 'adaptive' integration model vs simple matrix model.
    """
    line_integration = line.copy()

    model = 'mat-kick-mat'
    # Go through all elements in the line and update the model attribute if it exists
    for name in line.element_names:
        element = line[name]
        if hasattr(element, 'model'):
            element.model = model
    line_integration.build_tracker()
    line_integration.track(particles.copy())

    line_no_integ = line.copy()
    model = 'adaptive'
    # Go through all elements in the line and update the model attribute if it exists
    for name in line_no_integ.element_names:
        element = line_no_integ[name]
        if hasattr(element, 'model'):
            element.model = model
        

    line_no_integ.build_tracker()
    line_no_integ.track(particles.copy())
    
    fig, axes = plt.subplots(1, 2, figsize=u['fig_size'], sharey=True, sharex=True)

    h1, xedges1, yedges1 = histogram_monitors(line, verbose=True)
    axes[0].pcolormesh(xedges1, yedges1, h1.T)
    
    h2, xedges2, yedges2 = histogram_monitors(line_no_integ, verbose=True)
    axes[1].pcolormesh(xedges2, yedges2, h2.T)
    
    axes[0].set_title("With integration model")
    axes[1].set_title("With simple matrix model")




# %% {PLotting {} FUNCTIONS}
def twiss_plot(line, ref):
    init = xt.TwissInit(betx=ref['betx_0'], alfx=ref['alfx_0'], bety=ref['bety_0'], alfy=ref['alfy_0'])  # example values

    if 'a_m0' not in line.element_names:
        tw = line.twiss(
            method='4d',
            init=init,
            end='_end_point',
        )
    else:
        print("Skipping Twiss calculation because monitor 'a_m0' is in the line.")
        return

    # plot_beam_size
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



def track_line(line, particles):
    # Track particles through each element and plot the divergence
    tt = line.get_table()
    elements_names = [el for el in line.element_names]

    # Create a copy of the particles to track
    tracked_particles = particles.copy()
    # Initialize data structures to store particle coordinates

    s_values = [0.0]

    particle_list = [tracked_particles.copy()]

    # Track through each element individually
    for i, element_name in enumerate(elements_names):
        s_start = tt.rows[i].s
        s_start = s_start[0]
        s_stop = tt.rows[i+1].s
        s_stop = s_stop[0]
        print(f"ELEMENT {i}: {element_name} || s={s_start:.3f}:{s_stop:.3f} m")


        # Track through this single element
        line.track(tracked_particles, ele_start=element_name, num_elements=1)
        if "a_" in element_name or "m" in element_name:
            s_values.append(s_stop)
            p_to_list = tracked_particles.copy()
            p_to_list.sort(interleave_lost_particles=True)
            particle_list.append(p_to_list)


    return particle_list, s_values


def plot_histogram(x, y, bins, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist2d(x, y, bins=bins, cmap='inferno', norm=LogNorm())
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_title(title)
    plt.colorbar(label='Counts')
    plt.show()


def plot_divergence(XX, PX, YY, PY, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
           
    # hdivx = axs[0].hist2d(XX, PX, bins=(100,100), range=[[-6e-4,+6e-4],[-3e-3,+3e-3]], rasterized=True)
    hdivx, _, _, im = axs[0].hist2d(XX, PX, bins=(100,100), rasterized=True)
    axs[0].set_xlabel(r'$x$ [m]')
    axs[0].set_ylabel(r'$p_x/p_0$')
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25)
    # Draw colorbar with subplot
    fig.colorbar(im, ax=axs[0], label='Counts')


    # hdivy = axs[1].hist2d(YY, PY, bins=(100,100), range=[[-6e-4,+6e-4],[-3e-3,+3e-3]], rasterized=True)
    hdivy, _, _, im = axs[1].hist2d(YY, PY, bins=(100,100), rasterized=True)
    axs[1].set_xlabel(r'$y$ [m]')
    axs[1].set_ylabel(r'$p_y/p_0$')
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25)
    fig.colorbar(im, ax=axs[1], label='Counts')

    fig.suptitle(title, fontsize=16) # Add overall title

    print(f"XX: min={min(XX)}, max={max(XX)}, mean={np.mean(XX)}")
    print(f"PX: min={min(PX)}, max={max(PX)}, mean={np.mean(PX)}")
    # plt.show()


def particle_lost_at_step(particle_list):
    # First, determine at which step each particle was lost
    particle_lost_at = np.full(particle_list[0].x.size, len(particle_list))  # Default: particle survives all elements
    for step in range(1, len(particle_list)):
        prev_state = particle_list[step-1].state
        curr_state = particle_list[step].state
        lost_at_this_step = (prev_state > 0) & (curr_state <= 0)
        # Update particle_lost_at for particles that got lost at this step
        particle_lost_at[lost_at_this_step] = step
    return particle_lost_at


def plot_trajectories(particle_list, line, s_values, n_plot=100, show_dead=False, limit_line_width=2, limit_line_length=0.1):
    particles = particle_list[0]
    tt = line.get_table()

    x_values = [p.x for p in particle_list]
    y_values = [p.y for p in particle_list]  # shape = (num_elements+1, num_particles)

    s_values = np.array(s_values)
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Create a figure for particle trajectories
    fig, axes = plt.subplots(1, 2, figsize=u['fig_size'],
                             sharex=True, sharey=True)

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
        elif show_dead:
            # If particle died, use purple for x and yellow for y
            axes[0].plot(s_values[:loss_step+1], x_values[:loss_step+1, idx], 'purple', alpha=0.3, linewidth=0.5)
            axes[1].plot(s_values[:loss_step+1], y_values[:loss_step+1, idx], 'magenta', alpha=0.3, linewidth=0.5)
            # Mark the loss point with a scatter point
            axes[0].scatter(s_values[loss_step], x_values[loss_step, idx], color='k', s=9, alpha=0.7)
            axes[1].scatter(s_values[loss_step], y_values[loss_step, idx], color='k', s=9, alpha=0.7)

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
    # Add element markers
    for name, s_pos in zip(tt.name, tt.s):
        for ax in axes:
            ylim = ax.get_ylim()
            if 'a_' in name.lower():  # Apertures - black lines
                ax.axvline(x=s_pos, color='k', alpha=0.5, linestyle='--')
                ax.text(s_pos, ylim[1]*0.9, name, rotation=90, verticalalignment='top', fontsize=8)
                
                # Add element limits as small horizontal lines with proper x/y limits
                if 'q0' in name:
                    # X limits in ax[0], Y limits in ax[1]
                    if ax == axes[0]:  # X axis plot
                        ax.hlines(y=sizes['q0'][0], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q0'][1], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                    else:  # Y axis plot
                        ax.hlines(y=sizes['q0'][2], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q0'][3], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                elif 'q1' in name:
                    if ax == axes[0]:
                        ax.hlines(y=sizes['q1'][0], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q1'][1], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                    else:
                        ax.hlines(y=sizes['q1'][2], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q1'][3], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                elif 'q2' in name:
                    if ax == axes[0]:
                        ax.hlines(y=sizes['q2'][0], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q2'][1], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                    else:
                        ax.hlines(y=sizes['q2'][2], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q2'][3], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                elif 'corr' in name:
                    if ax == axes[0]:
                        ax.hlines(y=sizes['corr'][0], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['corr'][1], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                    else:
                        ax.hlines(y=sizes['corr'][2], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['corr'][3], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                elif 'dd' in name:
                    if ax == axes[0]:
                        ax.hlines(y=sizes['dd'][0], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['dd'][1], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                    else:
                        ax.hlines(y=sizes['dd'][2], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['dd'][3], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                elif 'm0' in name:
                    if ax == axes[0]:
                        ax.hlines(y=sizes['m0'][0], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['m0'][1], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                    else:
                        ax.hlines(y=sizes['m0'][2], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)
                        ax.hlines(y=sizes['m0'][3], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='k', linewidth=limit_line_width)

            elif name.startswith('dd'):  # Dipoles - pink lines
                ax.axvline(x=s_pos, color='magenta', alpha=0.5, linestyle='--')
                ax.text(s_pos, ylim[1]*0.8, name, rotation=90, verticalalignment='top', fontsize=8)
                
                # Add element limits with proper x/y limits
                if 'corr' in name:
                    if ax == axes[0]:
                        ax.hlines(y=sizes['corr'][0], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='magenta', linewidth=limit_line_width)
                        ax.hlines(y=sizes['corr'][1], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='magenta', linewidth=limit_line_width)
                    else:
                        ax.hlines(y=sizes['corr'][2], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='magenta', linewidth=limit_line_width)
                        ax.hlines(y=sizes['corr'][3], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='magenta', linewidth=limit_line_width)
                else:
                    if ax == axes[0]:
                        ax.hlines(y=sizes['dd'][0], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='magenta', linewidth=limit_line_width)
                        ax.hlines(y=sizes['dd'][1], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='magenta', linewidth=limit_line_width)
                    else:
                        ax.hlines(y=sizes['dd'][2], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='magenta', linewidth=limit_line_width)
                        ax.hlines(y=sizes['dd'][3], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='magenta', linewidth=limit_line_width)
                
            elif name.startswith('q'):  # Quadrupoles - green lines
                ax.axvline(x=s_pos, color='green', alpha=0.5, linestyle='--')
                ax.text(s_pos, ylim[1]*0.7, name, rotation=90, verticalalignment='top', fontsize=8)
                
                # Add element limits with proper x/y limits
                if '0' in name:
                    if ax == axes[0]:
                        ax.hlines(y=sizes['q0'][0], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q0'][1], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
                    else:
                        ax.hlines(y=sizes['q0'][2], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q0'][3], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
                elif '1' in name:
                    if ax == axes[0]:
                        ax.hlines(y=sizes['q1'][0], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q1'][1], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
                    else:
                        ax.hlines(y=sizes['q1'][2], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q1'][3], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
                elif '2' in name:
                    if ax == axes[0]:
                        ax.hlines(y=sizes['q2'][0], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q2'][1], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
                    else:
                        ax.hlines(y=sizes['q2'][2], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
                        ax.hlines(y=sizes['q2'][3], xmin=s_pos-limit_line_length, xmax=s_pos+limit_line_length, color='green', linewidth=limit_line_width)
            
            ax.set_ylim(ylim)



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

