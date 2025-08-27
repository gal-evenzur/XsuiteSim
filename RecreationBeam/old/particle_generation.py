import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import time
import pickle
import multiprocessing
from matplotlib.animation import FuncAnimation, PillowWriter


import argparse
parser = argparse.ArgumentParser(description='serial_analyzer.py...')
parser.add_argument('-gif', metavar='do gif?', required=False,  help='do gif?')
argus = parser.parse_args()
dogif = True if(argus.gif is not None and argus.gif=="1") else False


plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'
plt.rcParams['text.usetex'] = True


# Convert units
m_to_cm  = 1e2
m_to_mm  = 1e3
m_to_um  = 1e6
cm_to_mm = 1e1
cm_to_um = 1e4
cm_to_m  = 1e-2
mm_to_m  = 1e-3
mm_to_cm = 1e-1
mm_to_um = 1e3
um_to_mm = 1e-3
um_to_cm = 1e-4
um_to_m  = 1e-6
kG_to_T  = 0.1
GeV_to_kgms   = 5.39e-19
GeV_to_kg     = 1.8e-27
GeV_to_kgm2s2 = 1.6e-10

# Physical constants
c   = 299792458  # speed of light in m/s
c2  = c*c
e   = 1.602176634e-19  # elementary charge in C
m_e = 9.1093837015e-31  # electron/positron mass in kg
m_p = 1.67262192e-27 # proton/antiproton mass in kg


########################################################################
########################################################################
def GenerateGaussianBeam(E_GeV,mass_GeV,charge,mks=False):
    ### These variables assumed to be class members
    fx0         = -5*mm_to_m ### TODO???
    fy0         = 0 ### TODO???
    fz0         = -200*cm_to_m
    fbeamfocus  = 0
    fsigmax     = 50*um_to_m
    fsigmay     = 50*um_to_m
    fsigmaz     = 150*um_to_m
    lf          = E_GeV/mass_GeV
    femittancex = 50e-3*mm_to_m/lf ### mm-rad
    femittancey = 50e-3*mm_to_m/lf ### mm-rad
    fbetax      = (fsigmax**2)/femittancex
    fbetay      = (fsigmay**2)/femittancey
    
    ### z
    z0     = np.random.normal(fz0,fsigmaz)
    zdrift = z0 - fbeamfocus ### correct drift distance for x, y distribution.
    ### x
    sigmax  = fsigmax * np.sqrt(1.0 + (zdrift/fbetax)**2)
    x0      = np.random.normal(fx0, sigmax)
    meandx  = x0*zdrift / (zdrift**2 + fbetax**2)
    sigmadx = np.sqrt( femittancex*fbetax / (zdrift**2 + fbetax**2) )
    dx0     = np.random.normal(meandx, sigmadx)
    ### y
    sigmay  = fsigmay * np.sqrt(1.0 + (zdrift/fbetay)**2)
    y0      = np.random.normal(fy0, sigmay)
    meandy  = y0*zdrift / (zdrift**2 + fbetay**2)
    sigmady = np.sqrt( femittancey*fbetay / (zdrift**2 + fbetay**2) )
    dy0     = np.random.normal(meandy, sigmady)
    ### p
    pz = np.sqrt( (E_GeV**2 - mass_GeV**2)/ (dx0**2 + dy0**2 + 1.0) )
    px = dx0*pz
    py = dy0*pz
    pz0 = pz*GeV_to_kgms # kg*m/s
    px0 = px*GeV_to_kgms # kg*m/s
    py0 = py*GeV_to_kgms # kg*m/s
    mass_kg = mass_GeV*GeV_to_kgm2s2/c2 # kg
    ### state
    state_mks = [x0,y0,z0, px0,py0,pz0, mass_kg,charge] ### [x[m],y[m],z[m], px[kg*m/s],py[kg*m/s],pz[kg*m/s], m[kg],q[unit]]
    state_nat = [x0,y0,z0, px,py,pz, mass_GeV,charge]   ### [x[m],y[m],z[m], px[GeV],py[GeV],pz[GeV], m[GeV],q[unit]]
    return state_mks if(mks) else state_nat


def propagate_state_in_vacuum_to_z(state, z):
    if(z==state[2]): return state
    x0 = state[0]
    y0 = state[1]
    z0 = state[2]
    px = state[3]
    py = state[4]
    pz = state[5]
    m  = state[6]
    q  = state[7]
    pxz = np.sqrt(px**2 + pz**2)
    pyz = np.sqrt(py**2 + pz**2)
    thetax = np.arcsin(px/pxz)
    thetay = np.arcsin(py/pyz)
    x = x0 + np.tan(thetax)*(z-z0)
    y = y0 + np.tan(thetay)*(z-z0)
    state_at_z = [x,y,z, px,py,pz, m,q]
    return state_at_z


def truncated_exp_NK(a,b,how_many):
    a = -np.log(a)
    b = -np.log(b)
    rands = np.exp(-(np.random.rand(how_many)*(b-a) + a))
    return rands[0] if(how_many==1) else rands


def simulate_secondary_production(primary_state,q=+1,Emin=0.5,Emax=5,smear_T=False,smear_pT=False):
    x      = primary_state[0]
    y      = primary_state[1]
    z      = primary_state[2]
    px     = primary_state[3]
    py     = primary_state[4]
    pz     = primary_state[5]
    mass   = primary_state[6]
    charge = primary_state[7]
    
    ### smear trasverse position
    if(smear_pT):
        x = x + np.random.normal(0,0.3*um_to_m)
        y = y + np.random.normal(0,0.3*um_to_m)
    ### smear trasverse momenta
    if(smear_pT):
        smear_sigmax = 1.5e-3 ### GeV
        smear_sigmay = 1.5e-3 ### GeV
        px = px + np.random.normal(0,smear_sigmax) 
        py = py + np.random.normal(0,smear_sigmay)
    ### sample energy from exponential
    E = truncated_exp_NK(Emin,Emax,1) # GeV
    ### assume the x-y momemnta staty the same and correct the z momentum
    pz = np.sqrt( E**2 - mass**2 - px**2 - py**2 ) # GeV
    secondary_state = [x,y,z, px,py,pz, mass, q]
    
    return secondary_state



def plot_divergence(states, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    XX = []
    PX = []
    YY = []
    PY = []
    for pid,state in enumerate(states):
        xx = state[0]
        px = state[3]
        yy = state[1]
        py = state[4]
        XX.append( xx )
        PX.append( px )
        YY.append( yy )
        PY.append( py )
           
    # hdivx = axs[0].hist2d(XX, PX, bins=(100,100), range=[[-6e-4,+6e-4],[-3e-3,+3e-3]], rasterized=True)
    hdivx = axs[0].hist2d(XX, PX, bins=(100,100), rasterized=True)
    axs[0].set_xlabel(r'$x$ [m]')
    axs[0].set_ylabel(r'$p_x$ [GeV]')
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25)

    # hdivy = axs[1].hist2d(YY, PY, bins=(100,100), range=[[-6e-4,+6e-4],[-3e-3,+3e-3]], rasterized=True)
    hdivy = axs[1].hist2d(YY, PY, bins=(100,100), rasterized=True)
    axs[1].set_xlabel(r'$y$ [m]')
    axs[1].set_ylabel(r'$p_y$ [GeV]')
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25)

    fig.suptitle(title, fontsize=16) # Add overall title

    plt.tight_layout()
    # plt.show()



def plot_divergence_gif(states,fig,axs,z_pos):
    # Clear the axes
    for ax in axs: ax.clear()
    
    XX = []
    PX = []
    YY = []
    PY = []
    for pid,state in enumerate(states):
        ### state = [XX,YY,ZZ, PX,PY,PZ, MM,QQ]
        xx = state[0]
        px = state[3]
        yy = state[1]
        py = state[4]
        XX.append( xx )
        PX.append( px )
        YY.append( yy )
        PY.append( py )
           
    hdivx = axs[0].hist2d(XX, PX, bins=(200,200), range=[[-7.5e-3,+1.1e-2],[2.1e-2,+2.6e-2]], rasterized=True)
    # hdivx = axs[0].hist2d(XX, PX, bins=(200,200), rasterized=True)
    axs[0].set_xlabel(r'$x$ [m]')
    axs[0].set_ylabel(r'$p_x$ [GeV]')
    # axs[0].set_title(f'z = {z_pos*100:.1f} cm')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25)

    hdivy = axs[1].hist2d(YY, PY, bins=(200,200), range=[[-1e-3,+1e-3],[-3e-3,+3e-3]], rasterized=True)
    # hdivy = axs[1].hist2d(YY, PY, bins=(200,200), rasterized=True)
    axs[1].set_xlabel(r'$y$ [m]')
    axs[1].set_ylabel(r'$p_y$ [GeV]')
    # axs[1].set_title(f'z = {z_pos*100:.1f} cm')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25)

    plt.tight_layout()
    # plt.show()
    

def animate_beam_propagation(states, z_positions, output_filename='beam_propagation.gif'):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) # Create figure and axes
    
    def animate(frame):
        z_pos = z_positions[frame]
        states_at_z = []
        for state in states: states_at_z.append(propagate_state_in_vacuum_to_z(state, z_pos)) # Propagate all states to current z position
        plot_divergence_gif(states_at_z, fig, axs, z_pos) # Plot the phase space
        fig.suptitle(f'Beam Propagation: z = {z_pos*100:.1f} [cm]', fontsize=16) # Add overall title
        return axs
    
    print(f"Creating animation with {len(z_positions)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(z_positions), interval=200, blit=False, repeat=True)
    
    print(f"Saving animation to {output_filename}...")
    writer = PillowWriter(fps=5)
    anim.save(output_filename, writer=writer)
    
    print(f"Animation saved as {output_filename}")
    plt.close(fig)
    
    return anim



def plot_2h(states1,states2):
    x1 = np.array([state[0] for state in states1])
    y1 = np.array([state[1] for state in states1])
    z1 = np.array([state[2] for state in states1])
    x2 = np.array([state[0] for state in states2])
    y2 = np.array([state[1] for state in states2])
    z2 = np.array([state[2] for state in states2])
    
    px1 = np.array([state[3] for state in states1])
    py1 = np.array([state[4] for state in states1])
    pz1 = np.array([state[5] for state in states1])
    px2 = np.array([state[3] for state in states2])
    py2 = np.array([state[4] for state in states2])
    pz2 = np.array([state[5] for state in states2])
    
    xmin = min(min(x1),min(x2))
    xmax = max(max(x1),max(x2))
    ymin = min(min(y1),min(y2))
    ymax = max(max(y1),max(y2))
    zmin = min(min(z1),min(z2))
    zmax = max(max(z1),max(z2))

    xmin *= 1.2 if(xmin<0) else 0.8
    xmax *= 1.2
    ymin *= 1.2 if(ymin<0) else 0.8
    ymax *= 1.2
    zmin *= 1.2 if(zmin<0) else 0.8
    zmax *= 1.2

    pxmin = min(min(px1),min(px2))
    pxmax = max(max(px1),max(px2))
    pymin = min(min(py1),min(py2))
    pymax = max(max(py1),max(py2))
    pzmin = 0
    pzmax = max(max(pz1),max(pz2))*1.1
    
    pxmin *= 1.2 if(pxmin<0) else 0.8
    pxmax *= 1.2
    pymin *= 1.2 if(pymin<0) else 0.8
    pymax *= 1.2
    
    if(xmin==xmax):
        xmin=xmin*(1.-0.8)
        xmax=xmax*(1.+0.8)
    if(ymin==ymax):
        ymin=ymin*(1.-0.8)
        ymax=ymax*(1.+0.8)
    if(zmin==zmax):
        zmin=zmin*(1.-0.8)
        zmax=zmax*(1.+0.8)
        
    if(pxmin==pxmax):
        pxmin=pxmin*(1.-0.8)
        pxmax=pxmax*(1.+0.8)
    if(pymin==pymax):
        pymin=pymin*(1.-0.8)
        pymax=pymax*(1.+0.8)
    if(pzmin==pzmax):
        pzmin=pzmin*(1.-0.8)
        pzmax=pzmax*(1.+0.8)
        
    print(f"x[{xmin:.3f},{xmax:.3f}], y[{ymin:.3f},{ymax:.3f}], z[{zmin:.3f},{zmax:.3f}]")
    print(f"px[{pxmin:.3f},{pxmax:.3f}], py[{pymin:.3f},{pymax:.3f}], pz[{pzmin:.3f},{pzmax:.3f}]")
    
    fig, axs = plt.subplots(2, 3, figsize=(12,5), tight_layout=True)
    h1x = axs[0][0].hist(x1, bins=100, range=(xmin,xmax), alpha=0.5, label='Primary', color='blue', rasterized=True)
    h1y = axs[0][1].hist(y1, bins=100, range=(ymin,ymax), alpha=0.5, label='Primary', color='blue', rasterized=True)
    h1z = axs[0][2].hist(z1, bins=100, range=(zmin,zmax), alpha=0.5, label='Primary', color='blue', rasterized=True)
    h2x = axs[0][0].hist(x2, bins=100, range=(xmin,xmax), alpha=0.5, label='Secondary', color='red',  rasterized=True)
    h2y = axs[0][1].hist(y2, bins=100, range=(ymin,ymax), alpha=0.5, label='Secondary', color='red',  rasterized=True)
    h2z = axs[0][2].hist(z2, bins=100, range=(zmin,zmax), alpha=0.5, label='Secondary', color='red',  rasterized=True)
    
    h1px = axs[1][0].hist(px1, bins=100, range=(pxmin,pxmax), alpha=0.5, label='Primary', color='blue', rasterized=True)
    h1py = axs[1][1].hist(py1, bins=100, range=(pymin,pymax), alpha=0.5, label='Primary', color='blue', rasterized=True)
    h1pz = axs[1][2].hist(pz1, bins=100, range=(pzmin,pzmax), alpha=0.5, label='Primary', color='blue', rasterized=True)
    h2px = axs[1][0].hist(px2, bins=100, range=(pxmin,pxmax), alpha=0.5, label='Secondary', color='red', rasterized=True)
    h2py = axs[1][1].hist(py2, bins=100, range=(pymin,pymax), alpha=0.5, label='Secondary', color='red', rasterized=True)
    h2pz = axs[1][2].hist(pz2, bins=100, range=(pzmin,pzmax), alpha=0.5, label='Secondary', color='red', rasterized=True)

    axs[0][0].set_xlim(xmin,xmax)
    axs[0][0].set_xlabel(r"$x$ [m]")
    axs[0][0].set_ylabel('Particles')
    plt.locator_params(axis='x', nbins=10)
    axs[0][0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0][0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0][0].grid(True,linewidth=0.25,alpha=0.25)
    
    axs[0][1].set_xlim(ymin,ymax)
    axs[0][1].set_xlabel(r"$y$ [m]")
    axs[0][1].set_ylabel('Particles')
    plt.locator_params(axis='x', nbins=10)
    axs[0][1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0][1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0][1].grid(True,linewidth=0.25,alpha=0.25)
    
    axs[0][2].set_xlim(zmin,zmax)
    axs[0][2].set_xlabel(r"$z$ [m]")
    axs[0][2].set_ylabel('Particles')
    plt.locator_params(axis='x', nbins=10)
    axs[0][2].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0][2].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0][2].grid(True,linewidth=0.25,alpha=0.25)
    
    
    axs[1][0].set_xlim(pxmin,pxmax)
    axs[1][0].set_xlabel(r"$p_x$ [GeV]")
    axs[1][0].set_ylabel('Particles')
    plt.locator_params(axis='x', nbins=10)
    axs[1][0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1][0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1][0].grid(True,linewidth=0.25,alpha=0.25)
    
    axs[1][1].set_xlim(pymin,pymax)
    axs[1][1].set_xlabel(r"$p_y$ [GeV]")
    axs[1][1].set_ylabel('Particles')
    plt.locator_params(axis='x', nbins=10)
    axs[1][1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1][1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1][1].grid(True,linewidth=0.25,alpha=0.25)
    
    axs[1][2].set_xlim(pzmin,pzmax)
    axs[1][2].set_xlabel(r"$p_z$ [GeV]")
    axs[1][2].set_ylabel('Particles')
    plt.locator_params(axis='x', nbins=10)
    axs[1][2].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1][2].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1][2].grid(True,linewidth=0.25,alpha=0.25)

    plt.legend()
    plt.tight_layout()
    # plt.savefig(f"{pdfname}_energy.pdf")
    # plt.show()



###############################################################
###############################################################
###############################################################


states = []
for i in range(100000):
    ### particle species
    MM = m_e ## kg, positron
    QQ = +1  ## unit charge, positron
    mass_GeV = (MM*c2)/GeV_to_kgm2s2 ## GeV
    E_GeV = 10 # GeV
    state = GenerateGaussianBeam(E_GeV,mass_GeV,QQ)
    # print(state)
    states.append(state)
# plot_divergence_gif(states)
print(f"Generated {len(states)} particle states")



if(dogif):
    # Define z positions for animation (from -2m to +0.3m)
    z_start  = -2.0  # meters
    z_end    = 3.67   # meters
    n_frames = 50    # number of frames in animation
    z_positions = np.linspace(z_start, z_end, n_frames)
    print(f"Animation will show propagation from z = {z_start*100:.1f} cm to z = {z_end*100:.1f} cm")
    anim = animate_beam_propagation(states, z_positions, 'beam_propagation.gif')



### show the primary beam a few key points along z in meters
###                #shoot  Be-window  IP     Al-foil   Q0-entrance
z_key_positions = [-2.0,   -0.84,     0.0,   0.3,      3.67]
for z in z_key_positions:
    print(f"Plotting at z = {z*100:.1f} cm")
    states_at_z = []
    for state in states:
        states_at_z.append(propagate_state_in_vacuum_to_z(state,z))
    plot_divergence(states_at_z, f"Primaries at z = {z*100:.1f} cm")



### plot the "positrons"
print(f"Plotting positrons at z=30 cm")
primary_states_at_foil = []
secondary_states_at_foil = []
for state in states:
    primary_state_at_foil = propagate_state_in_vacuum_to_z(state,0.3)
    primary_states_at_foil.append(primary_state_at_foil)
    secondary_state_at_foil = simulate_secondary_production(primary_state_at_foil,q=+1,Emin=0.5,Emax=5,smear_T=True,smear_pT=True)
    secondary_states_at_foil.append(secondary_state_at_foil)
plot_divergence(secondary_states_at_foil,f"Secondaries at z = 30 cm")



### plot the differences in x,y,px,py
plot_2h(primary_states_at_foil,secondary_states_at_foil)