from sim_functions import *
import matplotlib.pyplot as plt

Eavg = 1e9


ref = { # All in natural units
    'q': -1,
    'p': p_from_E(Eavg, u['rest_e']),  # E = 3 GeV, p is in eV/c
    'm': u['rest_e'],  # electron mass in eV/c2
    'betx_0': 1.0,
    'alfx_0': 0.0,
    'bety_0': 1.0,
    'alfy_0': 0.0,
}

def dipoleElement(env, spacer, name, k0, length, max_x, max_y,
                  min_x=0, min_y=0, 
                  dx=0, dy=0, ang_z=0, ang_x=0, ang_y=0):
    """
    Creates a Dipole (Bend) element with apertures and alignment shifts.

    Similar to quadElement, this constructs a dipole assembly with misalignments and apertures.
    
    Args:
        env (xtrack.Environment): The environment to add the element to.
        spacer (str): Name of the spacer element.
        name (str): Name of the dipole.
        k0 (str or float): Normalized dipole strength (curvature) [m^-1].
        length (float): Length of the dipole [m].
        max_x (float): Max horizontal aperture [m].
        max_y (float): Max vertical aperture [m].
        r_pipe (float): Beam pipe radius [m].
        min_x (float): Min horizontal aperture [m].
        min_y (float): Min vertical aperture [m].
        dx (float): Horizontal shift [m].
        dy (float): Vertical shift [m].
        ang_z (float): Rotation around z-axis [rad].
        ang_x (float): Rotation around x-axis [rad].
        ang_y (float): Rotation around y-axis [rad].

    Returns:
        list: A list of components forming the dipole assembly.
    """
    # Define apertures
    env.new(f'a_{name}', xt.LimitRect,
             max_x=max_x, max_y=max_y, min_x=min_x, min_y=min_y),

    # Helper function to build the sandwich
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
            a_in, # Aperture in
            env.new(name, xt.Bend, length=length, k0=k0), # The Magnet
            a_out, # Aperture out
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

    dElement = dip(f'a_{name}', f'a_{name}')
    # Special rotation for 'dd' dipole to make it bend in the vertical plane (Bx field)
    dElement[name].rot_s_rad = -np.pi/2  # to have Bx field
    
    
    
    return dElement


def line_init(shifts, verbose=False):
    """
    Initializes the beamline lattice using xtrack.

    Constructs the beamline elements (drifts, quads, dipoles) based on the 
    defined sizes and magnet settings. Applies alignment shifts.

    A subtle note on the different xsuite element models:
    xt.LimitRect and xt.LimitRectEllipse are used to define apertures.
    xt.Quadrupole and xt.Bend define the magnetic elements.
    xt.ParticlesMonitor is used to monitor particles at specific locations - can be used as a detector.
    xt.SRotation, xt.XRotation, xt.YRotation, xt.XYShift are used to apply alignment shifts and rotations.
    "spacer" elements are needed between two "thin" elements to avoid issues in the tracking (it solves a xsuite bug).

    Args:
        shifts (dict): Dictionary containing magnet settings and alignment shifts.
        verbose (bool): If True, prints initialization details.

    Returns:
        tuple: (line, env, ref)
            - line (xtrack.Line): The assembled beamline.
            - env (xtrack.Environment): The environment object.
            - ref (dict): Reference particle properties.
    """
    # Extract magnet settings (gradients)
    m = round(float(shifts['magnetSettings']), 1)
    Grad1 = magsetvals[m][0]
    Grad2 = magsetvals[m][1]

    # Create the Xtrack Environment
    # The environment stores all the elements and their parameters.
    env = xt.Environment()
    
    # Define global parameters (magnet strengths) in the environment
    # This allows us to refer to them by name ('kq_p', 'kq_n') in the element definitions
    env['kq_p'] = grad_kG_to_k(Grad1, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  # k1 in 1/m^2
    env['kq_n'] = grad_kG_to_k(Grad2, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  
    env['kd'] = B_T_to_k(B_dd, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  # k0 in 1/m , Bx --> +yhat
    env['kd_corr'] = B_T_to_k(B_dd_xcorr, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e']) # By --> -xhat
    env['kdip'] = B_T_to_k(1, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  # k0 in 1/m , 1 T dipole for reference

    # Monitor at the end (Detector)
    env.new('a_m0', xt.LimitRect, min_x=-0.03, max_x=0.03, min_y=-0.1, max_y=0),
    env.elements['m0'] = xt.ParticlesMonitor(num_particles=int(n_particles),
                                    start_at_turn=0, stop_at_turn=1,
                                    auto_to_numpy=True)

    # Spacer element (drift of negligible length) to separate thin elements
    spacer = env.new('spacer', xt.Drift, length=1e-6)
    
    # Creating the Line 
    # The line is a sequence of components. We use `env.new_line` to create it.
    # We interleave drifts, quadrupoles (via quadElement), and dipoles (via dipoleElement).
    line = env.new_line(components=[
        env.new('dr0', xt.Drift, length=2),

        dipoleElement(env, spacer, 'dip', k0='kdip', length=0.3,
                max_x=0.015, max_y=0.035, min_x=-0.015, min_y=-0.035),

        env.new('drcorr.d', xt.Drift, length=2.7),


        env.place('a_m0'),
        env.place('m0'),

        env.new('dlast', xt.Drift, length=1.0),
    ])

    # Set tracking model (optional, for more accurate integration)
    if use_integration:
        model = 'mat-kick-mat'
        # Go through all elements in the line and update the model attribute if it exists
        for name in line.element_names:
            element = line[name]
            if hasattr(element, 'model'):
                if verbose: print(f"Updating model for element {name} from {element.model} to {model}")
                element.model = model

    # Define the reference particle for the line
    # This sets the reference momentum/energy for the lattice
    line.particle_ref = xt.Particles( 
        p0c=ref['p'],
        mass0=xt.ELECTRON_MASS_EV,
        q0=ref['q'],
    )

    # Build the tracker (compiles the line for tracking)
    line.build_tracker()

    return line, env, ref

def Energy_sample_return_deltas(E_min, E_max, ref, n_pars):
    """
    Samples particle energies uniformly between E_min and E_max,
    and returns the corresponding delta values relative to the reference energy.

    Args:
        E_min (float): Minimum energy in eV.
        E_max (float): Maximum energy in eV.
        ref (dict): Reference particle properties.
        n_pars (int): Number of particles to sample.

    Returns:
        np.ndarray: Array of delta values for the sampled energies.
    """
    E_samples = np.random.uniform(E_min, E_max, n_pars)
    p0 = ref['p']
    m0 = ref['m']

    # Calculate P from E and then delta
    p = np.sqrt(E_samples**2 - m0**2)


    delta_values = np.array([((p_i - p0)/p0) for p_i in p])

    return delta_values

print("Tracking particles...")
# Track particles through the line
# Returns:
# - particle_list: List of Particle objects at each element
# - s_values: Longitudinal positions of elements

line, env, ref = line_init(shifts, verbose=True)


n_particles = 1000
lst = np.zeros(n_particles)
deltas = Energy_sample_return_deltas(1e9, 5e9, ref, n_particles)
particles = xt.Particles(x=lst, px=lst, y=lst, py=lst,
        zeta=lst, delta=deltas)

particle_list, s_values = track_line(line, particles)

# Check dimensions
print(f"Number of tracking steps: {len(particle_list)}")
print(f"Number of s-positions: {len(s_values)}")

# Get the final particle distribution
final_particles = particle_list[-1]

# Identify surviving particles (state > 0)
alive_mask = final_particles.state > 0
n_alive = np.sum(alive_mask)
print(f"Particles surviving to the end: {n_alive} ({n_alive/n_particles*100:.2f}%)")


# Plot trajectories of a subset of surviving particles
n_plot = 50
indices = np.where(alive_mask)[0][:n_plot]
plt.figure(figsize=(12, 6))

# Calculate energies for the particles we're plotting
p0 = ref['p']
m0 = ref['m']
energies = []
for idx in indices:
    P = particles.delta[idx] * p0 + p0
    E = np.sqrt(P**2 + m0**2)
    energies.append(E / 1e9)  # Convert to GeV

# Create a colormap based on energy
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(energies), vmax=max(energies))

for i, idx in enumerate(indices):
    # Extract y position for this particle across all steps
    y_traj = [p_step.y[idx] for p_step in particle_list]
    plt.plot(s_values, y_traj, alpha=0.5, color=cmap(norm(energies[i])))

# Add colorbar as legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Energy [GeV]')

plt.xlabel('s [m]')
plt.ylabel('y [m]')
plt.title(f'Trajectories of {n_plot} particles')
plt.grid(True, alpha=0.3)
plt.savefig('particle_trajectories.png', dpi=300)
plt.show()

p0 = ref['p']
m0 = ref['m']
for p in range(n_particles):
    if final_particles.state[p] <= 0:
        continue
    # d = (p - p0)/p0
    # d * p0 + p0
    P = final_particles.delta[p] * p0 + p0
    E = np.sqrt(P**2 + m0**2)
    
    print(f"Final particle: x={final_particles.x[p]}, y={final_particles.y[p]}, \
          E={E/1e9}, delta={final_particles.delta[p]}, zeta={final_particles.zeta[p]}")