from matplotlib.ticker import AutoMinorLocator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


import xobjects as xo
import xtrack as xt
import xpart as xp

import h5py
import json

plt.close('all')

plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'
plt.rcParams['text.usetex'] = True


ctx = xo.ContextCpu()  # Use xo.ContextCupy() for GPU

# Physical constants
c   = 299792458  # speed of light in m/s
c2  = c*c
e   = 1.602176634e-19  # elementary charge in C
m_e = 9.1093837015e-31  # electron/positron mass in kg
m_p = 1.67262192e-27 # proton/antiproton mass in kg


u = {
    'c': 299792458,
    'c2': 299792458**2,
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
    'GeV_to_kgms': 5.39e-19,
    'GeV_to_kg': 1.8e-27,
    'GeV_to_kgm2s2': 1.6e-10,
}

def p_from_E(E, E_rest):
    # m is in eV / c2
    # E_rest = m * c2
    print("E rest = ", E_rest)
    # E is in eV
    # p is in eV / c
    pc = (E**2 - (E_rest)**2)**0.5
    return pc / u['c']

ref = {
    'q': 1,
    'p': p_from_E(3e9, u['rest_e']),  # E = 3 GeV, p is in mks
    'betx_0': 1.0,
    'alfx_0': 0.0,
    'bety_0': 1.0,
    'alfy_0': 0.0,
    'num_particles': 100  # This will be overridden by the actual number in the HDF5 file
}


def grad_kG_to_k(grad_kG):
    grad_T = grad_kG * u['kG_to_T']
    k = grad_T / ref['p']
    return k

def B_T_to_k(B_T):
    k = B_T / ref['p']
    return k

print(ref['p'])

env = xt.Environment()

env['kq_p'] = grad_kG_to_k(-6.66)
env['kq_n'] = grad_kG_to_k(28.86)
env['kd'] = B_T_to_k(0.219)


env['qL'] = 1

# We'll define the number of particles after loading the file
# Use a default value for now and update after loading
num_monitor_particles = 200000

env.elements['m0'] = xt.ParticlesMonitor(num_particles=num_monitor_particles,
                                start_at_turn=0, stop_at_turn=1,
                                auto_to_numpy=True)
env.elements['m1'] = xt.ParticlesMonitor(num_particles=num_monitor_particles,
                                start_at_turn=0, stop_at_turn=1,
                                auto_to_numpy=True)
env.elements['m2'] = xt.ParticlesMonitor(num_particles=num_monitor_particles,
                                start_at_turn=0, stop_at_turn=1,
                                auto_to_numpy=True)
env.elements['md'] = xt.ParticlesMonitor(num_particles=num_monitor_particles,
                                start_at_turn=0, stop_at_turn=1,
                                auto_to_numpy=True)

# Creating Line 
line = env.new_line(components=[
    env.new('d0', xt.Drift, length=3.6),
])


line.particle_ref = xt.Particles(
    p0c=ref['p'] * u['c'],  # in eV
    mass0=xt.ELECTRON_MASS_EV,
    q0=1,
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
    plt.show()

# Function to import particles from HDF5 file
def import_particles_from_hdf5(filename):
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
        x_coords = f['x'][:]
        y_coords = f['y'][:]
        z_coords = f['z'][:]
        px_coords = f['px'][:]
        py_coords = f['py'][:]
        pz_coords = f['pz'][:]
        
        # Get number of particles
        num_particles = f.attrs['num_particles']
        print(f"Loaded {num_particles} particles")
        
        # Print min/max values to verify data
        print(f"x range: [{np.min(x_coords):.6f}, {np.max(x_coords):.6f}] m")
        print(f"y range: [{np.min(y_coords):.6f}, {np.max(y_coords):.6f}] m")
        print(f"z range: [{np.min(z_coords):.6f}, {np.max(z_coords):.6f}] m")
        print(f"px range: [{np.min(px_coords):.6f}, {np.max(px_coords):.6f}] GeV")
        print(f"py range: [{np.min(py_coords):.6f}, {np.max(py_coords):.6f}] GeV")
        print(f"pz range: [{np.min(pz_coords):.6f}, {np.max(pz_coords):.6f}] GeV")
        
        # Create the particle object for tracking
        particles = xp.Particles(
            x=x_coords,
            px=px_coords,
            y=y_coords,
            py=py_coords,
            _context=ctx,
        )
        
        return particles

# Create particle object using either method
# Uncomment one of the following options:

# Option 1: Generate Gaussian beam as before
# particles = xp.Particles(
#     x=x,
#     px=px / pz,  # Normalized to reference momentum
#     y=y,
#     py=py / pz,  # Normalized to reference momentum
#     delta=(pz / ref['p'] - 1),  # (p - p0)/p0
#     zeta=np.zeros(n_particles),  # Assuming all particles start at same longitudinal position
#     _context=ctx,
# )

# Option 2: Import particles from HDF5 file
particles = import_particles_from_hdf5('Data/secondary_particles.h5')
pt = particles.get_table()
print(pt)

# Update the number of particles for the monitors
num_monitor_particles = particles.x.size
for monitor_name in ['m0', 'm1', 'm2', 'md']:
    env.elements[monitor_name].num_particles = num_monitor_particles

# Print the number of particles
print(f"Number of particles: {env.elements['m0'].num_particles}")
tt = line.get_table()
print(tt)


def plot_divergence(XX, PX, YY, PY, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
           
    # hdivx = axs[0].hist2d(XX, PX, bins=(100,100), range=[[-6e-4,+6e-4],[-3e-3,+3e-3]], rasterized=True)
    hdivx, _, _, im = axs[0].hist2d(XX, PX, bins=(100,100), rasterized=True, norm=LogNorm())
    axs[0].set_xlabel(r'$x$ [m]')
    axs[0].set_ylabel(r'$p_x$ [GeV]')
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25)
    # Draw colorbar with subplot
    fig.colorbar(im, ax=axs[0], label='Counts')


    # hdivy = axs[1].hist2d(YY, PY, bins=(100,100), range=[[-6e-4,+6e-4],[-3e-3,+3e-3]], rasterized=True)
    hdivy, _, _, im = axs[1].hist2d(XX, PX, bins=(100,100), rasterized=True)
    axs[1].set_xlabel(r'$x$ [m]')
    axs[1].set_ylabel(r'$p_x$ [GeV]')
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25)
    fig.colorbar(im, ax=axs[1], label='Counts')

    fig.suptitle(title, fontsize=16) # Add overall title

    plt.tight_layout()
    print(f"XX: min={min(XX)}, max={max(XX)}, mean={np.mean(XX)}")
    print(f"PX: min={min(PX)}, max={max(PX)}, mean={np.mean(PX)}")
    # plt.show()

plot_divergence(particles.x, particles.px, particles.y, particles.py, title="Initial Distribution")
plt.show()

line.track(particles)

plot_divergence(particles.x, particles.px, particles.y, particles.py, title="After drift")
plt.show()
