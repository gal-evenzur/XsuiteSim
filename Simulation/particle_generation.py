import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from matplotlib.animation import FuncAnimation, PillowWriter
import h5py
from params import n_particles, shifts, dat_file
dogif = False

plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'
plt.rcParams['text.usetex'] = True

doshw  = True
dogif = False
fullacc = False

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


##################################
######### configurations #########
##################################
fsigmax = 50*um_to_m ## beam sigma
fsigmay = 50*um_to_m ## beam sigma
fsigmaz = 150*um_to_m ## beam sigma
MM      = m_e ## kg, positron
QQ      = +1  ## unit charge, positron
mGeV    = (MM*c2)/GeV_to_kgm2s2 ## GeV
E_GeV   = 10 # GeV, energy of primary partticles
Emin    = 1 ## GeV
Emax    = 6 ## GeV
smearT  = True
smearP  = True
smear_sigma_T_um  = 0.3 ## um
smear_sigma_P_GeV = 1.5e-3 ## GeV
ZMAX    = 18 ## METERES
tmax    = ZMAX / (0.99 * c) ### time range for propagation (seconds): approximate time to travel 18 meters (last detector is at ~18 meters, relativistic particles going ~c)
t_span  = (0, tmax)
max_dt  = 1e-9
##################################
##################################
##################################



########################################################################
########################################################################
def GenerateGaussianBeam(E_GeV,mass_GeV,charge,shifts, mks=False):
    fx0     = shifts['beam']['fx0']
    fy0     = shifts['beam']['fy0']
    fz0     = shifts['beam']['fz0']
    fbeamfocus  = shifts['beam']['fbeamfocus']

    lf          = E_GeV/mass_GeV
    femittancex = 50e-3*mm_to_m/lf ### mm-rad
    femittancey = 50e-3*mm_to_m/lf ### mm-rad
    fbetax      = (fsigmax**2)/femittancex
    fbetay      = (fsigmay**2)/femittancey
    ### z
    z0     = np.random.normal(fz0,fsigmaz)
    zdrift = z0 - fbeamfocus ### correct drift distance for x, y distribution. Forces the beam to pass through the IP (i.e. focuesd at z=0)
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
    ### smear trasverse position
    if(smear_T):
        x = x + np.random.normal(0,smear_sigma_T_um*um_to_m)
        y = y + np.random.normal(0,smear_sigma_T_um*um_to_m)
    ### smear trasverse momenta
    if(smear_pT):
        px = px + np.random.normal(0,smear_sigma_P_GeV) 
        py = py + np.random.normal(0,smear_sigma_P_GeV)
    ### sample energy from exponential
    E = truncated_exp_NK(Emin,Emax,1) if(Emax>Emin) else Emin # GeV
    ### assume the x-y momemnta staty the same and correct the z momentum
    pz = np.sqrt( E**2 - mass**2 - px**2 - py**2 ) # GeV
    secondary_state = [x,y,z, px,py,pz, mass, q]
    return secondary_state


def state_GeV_to_kgms(state):
    state_mks = [0]*len(state)
    state_mks[0] = state[0]
    state_mks[1] = state[1]
    state_mks[2] = state[2]
    state_mks[3] = state[3]*GeV_to_kgms # kg*m/s
    state_mks[4] = state[4]*GeV_to_kgms # kg*m/s
    state_mks[5] = state[5]*GeV_to_kgms # kg*m/s
    state_mks[6] = state[6]*GeV_to_kgm2s2/c2 # kg
    state_mks[7] = state[7]
    return state_mks


def save_particles_to_hdf5(states, filename):
    """
    Save particle 6D phase space coordinates (x, y, z, px, py, pz) to an HDF5 file

    
    Args:
        states: List of particle states [x, y, z, px, py, pz, mass, charge]
        filename: Output HDF5 filename
    """
    # Extract the 6D phase space coordinates
    x_coords = np.array([state[0] for state in states]) # m
    y_coords = np.array([state[1] for state in states]) # m
    z_coords = np.array([state[2] for state in states]) # m
    px_coords = np.array([state[3] for state in states]) # GeV/c
    py_coords = np.array([state[4] for state in states]) # GeV/c
    pz_coords = np.array([state[5] for state in states]) # GeV/c
    mass = np.array([state[6] for state in states]) # GeV/c^2
    charge = np.array([state[7] for state in states]) # e

    # Create HDF5 file
    with h5py.File(filename, 'w') as f:
        # Create dataset for each coordinate
        f.create_dataset('x', data=x_coords)
        f.create_dataset('y', data=y_coords)
        f.create_dataset('z', data=z_coords)
        f.create_dataset('px', data=px_coords)
        f.create_dataset('py', data=py_coords)
        f.create_dataset('pz', data=pz_coords)
        f.create_dataset('mass', data=mass)
        f.create_dataset('charge', data=charge)
        
        # Add metadata
        f.attrs['num_particles'] = len(states)
        f.attrs['description'] = 'Particle 6D phase space coordinates (x, y, z, px, py, pz, mass, charge)'
        
        # Create a compound dataset with all coordinates together
        dt = np.dtype([('x', np.float64), ('y', np.float64), ('z', np.float64),
                       ('px', np.float64), ('py', np.float64), ('pz', np.float64),
                       ('mass', np.float64), ('charge', np.float64)])
        phase_space = np.zeros(len(states), dtype=dt)
        phase_space['x'] = x_coords
        phase_space['y'] = y_coords
        phase_space['z'] = z_coords
        phase_space['px'] = px_coords
        phase_space['py'] = py_coords
        phase_space['pz'] = pz_coords
        phase_space['mass'] = mass
        phase_space['charge'] = charge
        
        f.create_dataset('phase_space', data=phase_space)
    
    print(f"Saved {len(states)} particles to {filename}")



###############################################################
###############################################################
###############################################################

def generate_secondary_particles(shifts, n_particles, verbose=True):
    states = []
    for i in range(int(n_particles)):
        ### particle species
        MM = m_e ## kg, positron
        QQ = +1  ## unit charge, positron
        mass_GeV = (MM*c2)/GeV_to_kgm2s2 ## GeV
        E_GeV = 10 # GeV
        state = GenerateGaussianBeam(E_GeV,mass_GeV,QQ, shifts)
        states.append(state)
    if verbose: print("Finised creating beam")
    zAL     = +30 ### the aluminum foil, cm
    zBe     = -84 ### the beryllium window, cm
    Z0      = zBe if(shifts['magnetSettings']==502) else zAL
    Z0_m    = Z0*cm_to_m


    ### plot the "positrons"
    primary_states_at_foil = []
    secondary_states_at_foil = []
    for i, state in enumerate(states):
        primary_state_at_foil = propagate_state_in_vacuum_to_z(state,Z0_m)
        primary_states_at_foil.append(primary_state_at_foil)
        secondary_state_at_foil = simulate_secondary_production(primary_state_at_foil,q=+1,Emin=0.5,Emax=5,smear_T=True,smear_pT=True)
        secondary_states_at_foil.append(secondary_state_at_foil)
        if verbose and i%10000 == 0:
            print(f"created {i} particles")


    return secondary_states_at_foil

secondary_states_at_foil = generate_secondary_particles(shifts, n_particles)
# Save secondary particles at foil
save_particles_to_hdf5(secondary_states_at_foil, dat_file)
