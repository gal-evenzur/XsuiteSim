from matplotlib.ticker import AutoMinorLocator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xtrack as xt
plt.rcParams['image.cmap'] = 'afmhot'

n_particles = 50000
MagnetSettings = 502

# %% CONSTS________
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



### magnets
magsetvals = {502:[-7.637,28.55,-7.637], 490.0:[-30.68,46.42,-30.68], 490.1:[-27.99,44.98,-27.99], 490.2:[-20.38,40.42,-20.38], 490.3:[-11.56,30.05,-11.56], 490.4:[-3.37,26.72,-3.37], 490.5:[-6.66,28.86,-6.66] }
magsetdelt = {"quad0":[0,0], "quad1":[0,0], "quad2":[0,0], "xcorr":[0,0], "dipole":[0,0]} ### cm
Grad1 = magsetvals[MagnetSettings][0]
Grad2 = magsetvals[MagnetSettings][1]

B_dd_xcorr = 0.026107 # Bx 
B_dd = 0.219 # By

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
dy_det  = +0.35 # cm

# Define detector x range
detector_x_center_cm = 0 # cm
# detector_x_center_cm = 0. # cm
detector_x_center_m = detector_x_center_cm*u['cm_to_m']

# Define detector y range
detector_y_center_cm = 5.165 + 0.1525 + 3.685 + dy_det # cm
detector_y_center_m  = detector_y_center_cm*u['cm_to_m']

# Calculate detector z position
detector_z_base_cm = 1363 + 303.2155 + 11.43 + 1.05  # cm
detector_z_base_m  = detector_z_base_cm*u['cm_to_m']
detector_z_base_mm = detector_z_base_cm*u['cm_to_mm']


sizes = { # min_x, max_x, min_y, max_y in m, start z, stop z, length in m
    'dr0': [3.6733336],
    'q0': [-0.024610, 0.024610, -0.024610, 0.024610, 4.646664-3.6733336],
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


# %% (--) _FUNCTIONS_ (--)

def p_from_E(E, E_rest):
    # m is in eV / c2
    # E_rest = m * c2
    print("E rest = ", E_rest)
    # E is in eV
    # p is in eV / c
    p = (E**2 - (E_rest)**2)**0.5 #p is in eV/c
    return p


def grad_kG_to_k(grad_kG, p_mks, q_mks):
    kG_to_T = 0.1
    grad_T = grad_kG * kG_to_T  # grad in T/m 
    k = q_mks * grad_T / p_mks  # k in 1/m
    return k

def B_T_to_k(B_T, p_mks, q_mks):
    k = q_mks * B_T / p_mks  # k in 1/m
    return k

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

def plot_tracks(particle_list, element_names=None, track_vars=['x', 'y'], fig_size=(12, 8), n_cols=2, title="Particle Tracks"):
    """
    Plot multiple particle tracks in a subplot arrangement.
    
    Args:
        particle_list: List of particle states at different elements
        element_names: Optional list of element names corresponding to particle_list
        track_vars: Variables to plot, default ['x', 'y']
        fig_size: Figure size as tuple (width, height)
        n_cols: Number of columns in the subplot grid
        title: Main title for the figure
    
    Returns:
        List of figures created
    """
    if element_names is None:
        element_names = [f"Element {i}" for i in range(len(particle_list))]
    
    # Identify drift elements (can be customized based on naming convention)
    drift_indices = [i for i, name in enumerate(element_names) if 'drift' in name.lower()]
    
    # If no drifts found, treat the entire beamline as one segment
    if not drift_indices:
        drift_indices = [0, len(particle_list)-1]
    
    figures = []
    
    # Create figures for each segment (between drifts, inclusive)
    for i in range(len(drift_indices)-1):
        start_idx = drift_indices[i]
        end_idx = drift_indices[i+1] # Include the ending drift
        
        # Get elements in this segment
        segment_particles = particle_list[start_idx:end_idx]
        segment_names = element_names[start_idx:end_idx]
        
        # Calculate subplot grid
        n_plots = len(segment_particles)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create figure and subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size, tight_layout=True)
        axs = axs.flatten() if n_plots > 1 else [axs]
        
        # Plot each element in the segment
        for j, (particle, elem_name) in enumerate(zip(segment_particles, segment_names)):
            if j < len(axs):
                ax = axs[j]
                
                # Plot the specified track variables
                for var in track_vars:
                    if hasattr(particle, var):
                        values = getattr(particle, var)
                        ax.plot(values, label=var)
                
                ax.set_title(elem_name)
                ax.grid(True, linewidth=0.25, alpha=0.25)
                ax.legend()
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        
        # Hide unused subplots
        for j in range(n_plots, len(axs)):
            axs[j].set_visible(False)
        
        segment_title = f"{title} - Segment {i+1} ({element_names[start_idx]} to {element_names[end_idx-1]})"
        fig.suptitle(segment_title, fontsize=16)
        
        figures.append(fig)
    
    return figures
