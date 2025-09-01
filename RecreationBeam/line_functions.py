from matplotlib.ticker import AutoMinorLocator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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
