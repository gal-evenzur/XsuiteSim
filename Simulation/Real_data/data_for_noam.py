from matplotlib.ticker import AutoMinorLocator
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import xtrack as xt
import xpart as xp
import sys
import os
pyPath = os.path.dirname(os.path.abspath(__file__))
dirPath = os.path.dirname(pyPath)
sys.path.append(dirPath)
import bremss as br
from params import *
from copy import deepcopy
import time
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
dy_det  = 0.35 # cm

particle_start_pos = Z0 # m
# Define detector x range
detector_x_center_cm = 0 # cm
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
    'm': u['rest_e'],  # electron mass in eV/c2
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

# %% * * * * * P A R T I C L E S * * * * 

def generate_secondary_particles(shifts, n_particles, verbose=True, rng=default_rng()):
    states = []
    for i in range(int(n_particles)):
        ### particle species
        QQ = +1  ## unit charge, positron
        mass_GeV = (MM*u['c2'])/u['GeV_to_kgm2s2'] ## GeV
        E_GeV = 10 # GeV
        state = GenerateGaussianBeam(E_GeV,mass_GeV,QQ, shifts, rng=rng)
        states.append(state)
    if verbose: print("Finised creating beam")
    zAL     = +30 ### the aluminum foil, cm
    zBe     = -84 ### the beryllium window, cm
    Z0      = zBe if(shifts['magnetSettings']==502) else zAL
    Z0_m    = Z0*u['cm_to_m']


    ### plot the "positrons"
    primary_states_at_foil = []
    secondary_states_at_foil = []
    for i, state in enumerate(states):
        primary_state_at_foil = propagate_state_in_vacuum_to_z(state,Z0_m)
        primary_states_at_foil.append(primary_state_at_foil)
        secondary_state_at_foil = simulate_secondary_production(primary_state_at_foil,rng=rng, q=+1,Emin=Emin,Emax=Emax,smear_T=smear_T,smear_pT=smear_pT)
        secondary_states_at_foil.append(secondary_state_at_foil)
        if verbose and i%10000 == 0:
            print(f"created {i} particles")


    return secondary_states_at_foil

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

def particles_from_states(states, ref, verbose=False):
    """
    Create a particle object from a list of particle states
    Args:
        states: List of particle states [x, y, z, px, py, pz, mass, charge]
    Returns:
        particles: Particle object
    """
    x_coords = np.array([state[0] for state in states]) # m
    y_coords = np.array([state[1] for state in states]) # m
    z_coords = np.array([state[2] for state in states]) # m
    px_coords = np.array([state[3] for state in states]) # GeV/c
    py_coords = np.array([state[4] for state in states]) # GeV/c
    pz_coords = np.array([state[5] for state in states]) # GeV/c
    num_particles = len(states)

    p0c=ref['p']
    mass0=ref['m']
    q0=ref['q']

    
    px_eV = px_coords * u['GeV_to_eV']
    py_eV = py_coords * u['GeV_to_eV']
    pz_eV = pz_coords * u['GeV_to_eV']

    p = np.sqrt(px_eV**2 + py_eV**2 + pz_eV**2)

    px = px_eV / p0c # dimensionless
    py = py_eV / p0c # dimensionless

    delta = (p - p0c) / p0c  # dimensionless
    

    # Get number of particles
    if verbose: print(f"Loaded {num_particles} particles")
    
    # Create the particle object for tracking
    particles = xp.Particles(
        p0c=p0c,
        mass0=mass0,
        q0=q0,
        x=x_coords,
        px=px,
        y=y_coords,
        py=py,
        zeta=0,
        delta=delta,  # delta = (pz [eV/c] - p0 [eV/c]) / p0
    )
        
    return particles

def GenerateGaussianBeam(E_GeV,mass_GeV,charge,shifts, mks=False, rng=default_rng()):
    fx0     = shifts['beam']['fx0']
    fy0     = shifts['beam']['fy0']
    fz0     = shifts['beam']['fz0']
    fbeamfocus  = shifts['beam']['fbeamfocus']

    lf          = E_GeV/mass_GeV
    femittancex = 50e-3*u['mm_to_m']/lf ### mm-rad
    femittancey = 50e-3*u['mm_to_m']/lf ### mm-rad
    fbetax      = (fsigmax**2)/femittancex
    fbetay      = (fsigmay**2)/femittancey
    ### z
    z0     = rng.normal(fz0,fsigmaz)
    zdrift = z0 - fbeamfocus ### correct drift distance for x, y distribution. Forces the beam to pass through the IP (i.e. focuesd at z=0)
    ### x
    sigmax  = fsigmax * np.sqrt(1.0 + (zdrift/fbetax)**2)
    x0      = rng.normal(fx0, sigmax)
    meandx  = x0*zdrift / (zdrift**2 + fbetax**2)
    sigmadx = np.sqrt( femittancex*fbetax / (zdrift**2 + fbetax**2) )
    dx0     = rng.normal(meandx, sigmadx)
    ### y
    sigmay  = fsigmay * np.sqrt(1.0 + (zdrift/fbetay)**2)
    y0      = rng.normal(fy0, sigmay)
    meandy  = y0*zdrift / (zdrift**2 + fbetay**2)
    sigmady = np.sqrt( femittancey*fbetay / (zdrift**2 + fbetay**2) )
    dy0     = rng.normal(meandy, sigmady)
    ### p
    pz = np.sqrt( (E_GeV**2 - mass_GeV**2)/ (dx0**2 + dy0**2 + 1.0) )
    px = dx0*pz
    py = dy0*pz
    pz0 = pz*u['GeV_to_kgms'] # kg*m/s
    px0 = px*u['GeV_to_kgms'] # kg*m/s
    py0 = py*u['GeV_to_kgms'] # kg*m/s
    mass_kg = mass_GeV*u['GeV_to_kgm2s2']/u['c2'] # kg
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


def truncated_exp_NK(a,b,how_many, rng=default_rng()):
    a = -np.log(a)
    b = -np.log(b)
    rands = np.exp(-(rng.random(how_many)*(b-a) + a))
    return rands[0] if(how_many==1) else rands


def simulate_secondary_production(primary_state, rng=default_rng(), q=+1,Emin=0.5,Emax=5,smear_T=False,smear_pT=False):    
    x      = primary_state[0]
    y      = primary_state[1]
    z      = primary_state[2]
    px     = primary_state[3]
    py     = primary_state[4]
    pz     = primary_state[5]
    mass   = primary_state[6]
    ### smear trasverse position
    if(smear_T):
        x = x + rng.normal(0,smear_sigma_T_um*u['um_to_m'])
        y = y + rng.normal(0,smear_sigma_T_um*u['um_to_m'])
    ### smear trasverse momenta
    if(smear_pT):
        px = px + rng.normal(0,smear_sigma_P_GeV) 
        py = py + rng.normal(0,smear_sigma_P_GeV)
    
    ### sample energy like in bremss

    # E = truncated_exp_NK(Emin,Emax, 1) if(Emax>Emin) else Emin # GeV
    E = br.sample_from_pdf_on_bins(E_vals, eplus, nsamples=1, rng=rng)
    while(E[0]<Emin or E[0]>Emax): E = br.sample_from_pdf_on_bins(E_vals, eplus, nsamples=1, rng=rng)
    E = E[0]


    ### assume the x-y momemnta staty the same and correct the z momentum
    pz = np.sqrt( E**2 - mass**2 - px**2 - py**2 ) # GeV
    secondary_state = [x,y,z, px,py,pz, mass, q]
    return secondary_state


def state_GeV_to_kgms(state):
    state_mks = [0]*len(state)
    state_mks[0] = state[0]
    state_mks[1] = state[1]
    state_mks[2] = state[2]
    state_mks[3] = state[3]*u['GeV_to_kgms'] # kg*m/s
    state_mks[4] = state[4]*u['GeV_to_kgms'] # kg*m/s
    state_mks[5] = state[5]*u['GeV_to_kgms'] # kg*m/s
    state_mks[6] = state[6]*u['GeV_to_kgm2s2']/u['c2'] # kg
    state_mks[7] = state[7]
    return state_mks



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

def line_init(shifts, verbose=False):
    m = round(float(shifts['magnetSettings']), 1)
    Grad1 = magsetvals[m][0]
    Grad2 = magsetvals[m][1]

    env = xt.Environment()
    env['kq_p'] = grad_kG_to_k(Grad1, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  # k1 in 1/m^2
    env['kq_n'] = grad_kG_to_k(Grad2, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  
    env['kd'] = B_T_to_k(B_dd, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e'])  # k0 in 1/m , Bx --> +yhat
    env['kd_corr'] = B_T_to_k(B_dd_xcorr, ref['p'] * u['eV_to_kgms'], ref['q'] * u['e']) # By --> -xhat


    # Monitor at the end
    env.new('a_m0', xt.LimitRect, min_x=sizes['m0'][0], max_x=sizes['m0'][1], min_y=sizes['m0'][2], max_y=sizes['m0'][3]),
    env.new('a_m1', xt.LimitRect, min_x=sizes['m0'][0], max_x=sizes['m0'][1], min_y=sizes['m0'][2], max_y=sizes['m0'][3]),
    env.new('a_m2', xt.LimitRect, min_x=sizes['m0'][0], max_x=sizes['m0'][1], min_y=sizes['m0'][2], max_y=sizes['m0'][3]),
    env.new('a_m3', xt.LimitRect, min_x=sizes['m0'][0], max_x=sizes['m0'][1], min_y=sizes['m0'][2], max_y=sizes['m0'][3]),
    env.new('a_m4', xt.LimitRect, min_x=sizes['m0'][0], max_x=sizes['m0'][1], min_y=sizes['m0'][2], max_y=sizes['m0'][3]),
    env.elements['m0'] = xt.ParticlesMonitor(num_particles=int(n_particles),
                                    start_at_turn=0, stop_at_turn=1,
                                    auto_to_numpy=True)
    env.elements['m1'] = xt.ParticlesMonitor(num_particles=int(n_particles),
                                    start_at_turn=0, stop_at_turn=1,
                                    auto_to_numpy=True)
    env.elements['m2'] = xt.ParticlesMonitor(num_particles=int(n_particles),
                                    start_at_turn=0, stop_at_turn=1,
                                    auto_to_numpy=True)
    env.elements['m3'] = xt.ParticlesMonitor(num_particles=int(n_particles),
                                    start_at_turn=0, stop_at_turn=1,
                                    auto_to_numpy=True)
    env.elements['m4'] = xt.ParticlesMonitor(num_particles=int(n_particles),
                                    start_at_turn=0, stop_at_turn=1,
                                    auto_to_numpy=True)

    drift_monitor = env.new('dr_m', xt.Drift, length=0.01) # 1 cm

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
        env.new('dr_m0', xt.Drift, length=0.01),
        env.place('a_m1'),
        env.place('m1'),
        env.new('dr_m1', xt.Drift, length=0.01),
        env.place('a_m2'),
        env.place('m2'),
        env.new('dr_m2', xt.Drift, length=0.01),
        env.place('a_m3'),
        env.place('m3'),
        env.new('dr_m3', xt.Drift, length=0.01),
        env.place('a_m4'),
        env.place('m4'),
    
    ])

    if use_integration:
        model = 'mat-kick-mat'
        # Go through all elements in the line and update the model attribute if it exists
        for name in line.element_names:
            element = line[name]
            if hasattr(element, 'model'):
                if verbose: print(f"Updating model for element {name} from {element.model} to {model}")
                element.model = model

    # Need to input in natural units
    line.particle_ref = xt.Particles( 
        p0c=ref['p'],
        mass0=xt.ELECTRON_MASS_EV,
        q0=ref['q'],
    )

    line.build_tracker()

    return line, env, ref

# Go element by element and track particles
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
        if "a_dd_out" in element_name or element_name.startswith("m"):
            s_values.append(s_stop)
            p_to_list = tracked_particles.copy()
            p_to_list.sort(interleave_lost_particles=True)
            particle_list.append(p_to_list)



    return particle_list, s_values

def histogram_monitors(line, verbose=False):

    m = [el for el in line.elements if isinstance(el, xt.ParticlesMonitor)]
    h = np.zeros((len(m), monitor_bins[0], monitor_bins[1]))
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
            
            
            h[i], xedges, yedges = np.histogram2d(x_clean, y_clean, bins=monitor_bins)
        else:
            if verbose: print(f"No particles alive at monitor {i}")
            h[i], xedges, yedges = np.histogram2d([],[], bins=monitor_bins)

    return h, xedges, yedges




def track_monitor(line, particles):
    line.track(particles.copy())

    h, xedges, yedges = histogram_monitors(line)

    return h, xedges, yedges

# Function to import particles from HDF5 file
def import_particles_from_hdf5(filename, ref, verbose=False):
    """
    Import particles from an HDF5 file created by particle_generation.py
    
    Args:
        filename: Path to the HDF5 file
    
    Returns:
        xpart.Particles: Particle object for tracking
    """
    if verbose: print(f"Loading particles from {filename}")
    with h5py.File(filename, 'r') as f:
        # Extract the 6D phase space coordinates
        x_coords = f['x'][:] # [m]
        y_coords = f['y'][:] # [m]
        z_coords = f['z'][:] # [m]
        px_coords = f['px'][:] # [GeV/c]
        py_coords = f['py'][:] # [GeV/c]
        pz_coords = f['pz'][:] # [GeV/c]
        num_particles = f.attrs['num_particles']
    p0c=ref['p']
    mass0=ref['m']
    q0=ref['q']

    
    px_eV = px_coords * u['GeV_to_eV']
    py_eV = py_coords * u['GeV_to_eV']
    pz_eV = pz_coords * u['GeV_to_eV']

    p = np.sqrt(px_eV**2 + py_eV**2 + pz_eV**2)

    px = px_eV / p0c # dimensionless
    py = py_eV / p0c # dimensionless

    delta = (p - p0c) / p0c  # dimensionless
    

    # Get number of particles
    if verbose: print(f"Loaded {num_particles} particles")
    
    # Create the particle object for tracking
    particles = xp.Particles(
        p0c=p0c,
        mass0=mass0,
        q0=q0,
        x=x_coords,
        px=px,
        y=y_coords,
        py=py,
        zeta=0,
        delta=delta,  # delta = (pz [eV/c] - p0 [eV/c]) / p0
    )
        
    return particles


# Retarted functions:

# Simple shifts array 
def shifts_array_deterministic(shifts, element, setting, range_vals, magnet_settings=[490]):
    """ 
    Create a list of shifts for a given element and setting.
    If multiple magnet_settings are provided, returns a matrix where:
    - Each row corresponds to a different magnet setting
    - Each column corresponds to a different value in range_vals
    
    Returns:
    - shift_list: A list of dictionaries (if single magnet setting) or
                    A 2D list where shift_list[i][j] has magnetSettings=magnet_settings[i]
                    and element[setting]=range_vals[j]
    """
    shift_matrix = []
    for mag_setting in magnet_settings:
        row = []
        shifts_copy = deepcopy(shifts)
        shifts_copy['magnetSettings'] = mag_setting
        for val in range_vals:
            shifts_copy[element][setting] = val
            row.append(deepcopy(shifts_copy))
        shift_matrix.append(row)
    return shift_matrix

def histogram_mean_std(h, xedges, yedges, ax=None, threshold=3, point_threshold=30):
    mask = h > threshold
    h = np.where(mask, h, 0)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)
    weights = h.T.flatten()
    mask = weights > 0

    # Check if we have enough relevant points above the threshold
    if np.sum(mask) <= threshold:
        print("Error: Not enough data points above threshold")
        return None, None, None, None

    if np.any(mask):
        mean_x = np.average(x_mesh.flatten()[mask], weights=weights[mask])
        mean_y = np.average(y_mesh.flatten()[mask], weights=weights[mask])
        std_x = np.sqrt(np.average((x_mesh.flatten()[mask] - mean_x)**2, weights=weights[mask]))
        std_y = np.sqrt(np.average((y_mesh.flatten()[mask] - mean_y)**2, weights=weights[mask]))
        
        if ax is not None:
            # Plot the mean point
            ax.plot(mean_x, mean_y, 'wo', markersize=8)
            ax.plot(mean_x, mean_y, 'ko', markersize=5)

            # Plot horizontal line for x variance
            ax.plot([mean_x - std_x, mean_x + std_x], [mean_y, mean_y], 'w-', linewidth=2)

            # Plot vertical line for y variance
            ax.plot([mean_x, mean_x], [mean_y - std_y, mean_y + std_y], 'w-', linewidth=2)

            # Add text with mean and std values
            ax.text(0.05, 0.95, f'μx={mean_x:.2e}, σx={std_x:.2e}\nμy={mean_y:.2e}, σy={std_y:.2e}', 
                transform=ax.transAxes, color='white', fontsize=8,
                verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))
            
        
        return mean_x, std_x, mean_y, std_y

def plot_multiple_magnet_settings(shifts_orig, mag_settings, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, len(mag_settings), figsize=(len(mag_settings)*6, 5), tight_layout=True)
    shifts = deepcopy(shifts_orig)  # To avoid modifying the original shifts dictionary

    for idx, setting in enumerate(mag_settings):
        print(f"Magnet setting: {setting}")
        shifts['magnetSettings'] = setting  # Set the magnet setting
        
        # Initialize line with new settings
        line, env, ref = line_init(shifts=shifts)
        
        # Import particles
        particles = import_particles_from_hdf5(line, 'Data/secondary_particles.h5', p0c=ref['p'])
        
        # Track particles and get histogram data
        h, xedges, yedges = track_monitor(line, particles)

        thersh= 4
        # mask = h > thersh
        # h = np.where(mask, h, 0)

        
        # Plot the histogram
        im = axs[idx].imshow(h.T, origin='lower', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    aspect='auto')

        histogram_mean_std(h, xedges, yedges, ax=axs[idx], threshold=thersh)
        # Apply a threshold to ignore bins with small counts (less than 10)

        # axs[idx].set_xlabel('x [m]')
        # axs[idx].set_ylabel('y [m]')
        # axs[idx].set_title(f'magnetSettings = {setting}')
        
        # plt.colorbar(im, ax=axs[idx], label='Counts per bin')

    plt.tight_layout()


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


def plot_trajectories(particle_dir, line, n_plot=100, show_dead=False, limit_line_width=2, limit_line_length=0.1):
    particle_list = particle_dir['p']
    s_values = particle_dir['s']
    plot_names = particle_dir['names']
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

    # Set titles, grid and auto minor locators
    axes[0].set_title('X Trajectories')
    axes[1].set_title('Y Trajectories')
    
    for ax in axes:
        ax.set_xlabel('s [m]')
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    
    axes[0].set_ylabel('x [m]')
    axes[1].set_ylabel('y [m]')
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
    


def xy_plot_line(line, particle_dir, ele_str, elementNames, n_bin=100):
    """
    Generate XY plots for quadrupoles in the beam line.
    For each quadrupole, create a figure showing the XY distribution 
    at both the entrance (before quad) and exit (after quad).
    """
    # Get all element names in the line
    particle_list = particle_dir['p']
    s_values = particle_dir['s']
    tracked_elements = particle_dir['names']
    
    element_names = line.element_names
    # Identify all quadrupoles in the line
    plot_elements = [name for name in element_names if name.startswith(ele_str)]
    print(f"...Plotting XY pictures for {elementNames}...")
    
    alive_particles = []
    for p in particle_list:
        alive_particles.append(p.filter(p.state > 0))
    
    # Create one figure with 3x2 subplots (3 quads, entrance and exit)
    fig, axs = plt.subplots(2, len(plot_elements), figsize=u['fig_size'],
                             sharex=True, sharey=True, tight_layout=True)
    fig.suptitle(f"XY Distribution at {elementNames} Entrances and Exits", fontsize=16)
    
    for i, ele_name in enumerate(plot_elements):
        # Find the entrance and exit indices from tracked_elements
        entrance_idx = None
        exit_idx = None
        
        # Look for the "a_" + ele_str elements in tracked_elements
        for j, tracked_name in enumerate(tracked_elements):
            if f'a_{ele_name}' == tracked_name or f'a_{ele_name}_out' == tracked_name:
                if entrance_idx is None:
                    entrance_idx = j
                else:
                    exit_idx = j
                    break
        
        if entrance_idx is None or exit_idx is None:
            print(f"Warning: Could not find both entrance and exit for {ele_name}")
            continue
            
        # Get particles at entrance and exit
        entrance_particles = alive_particles[entrance_idx]
        exit_particles = alive_particles[exit_idx]
        
        # Plot entrance distribution (top row)
        h, _, _, im = axs[0, i].hist2d(entrance_particles.x, entrance_particles.y, 
                                     bins=(n_bin, n_bin), rasterized=True)
        axs[0, i].set_title(f"{ele_name} entrance\n elem: {tracked_elements[entrance_idx]}")
        axs[0, i].set_xlabel('x [m]')
        axs[0, i].set_ylabel('y [m]')
        axs[0, i].grid(True, linewidth=0.5, alpha=0.5)
        axs[0, i].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs[0, i].yaxis.set_minor_locator(AutoMinorLocator(10))
        fig.colorbar(im, ax=axs[0, i])
        
        # Plot exit distribution (bottom row)
        h, _, _, im = axs[1, i].hist2d(exit_particles.x, exit_particles.y, 
                         bins=(n_bin, n_bin), rasterized=True)
        axs[1, i].set_title(f"{ele_name} exit\n elem: {tracked_elements[exit_idx]}")
        axs[1, i].set_xlabel('x [m]')
        axs[1, i].set_ylabel('y [m]')
        axs[1, i].grid(True, linewidth=0.5, alpha=0.5)
        axs[1, i].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs[1, i].yaxis.set_minor_locator(AutoMinorLocator(10))
        fig.colorbar(im, ax=axs[1, i])
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    print(f"Finished plotting {elementNames} XY distributions.")


# %% main settings 
print("Importing Finished")
verbose = True
plot = True
rng = np.random.default_rng()

magnet_settings = [490, 490.1, 490.2, 490.5]
n_particles = 1e5


#%% ####### THIS PART SHOWS ALIVE PARTICLES AT EACH MONITOR (N_PARS CHANGES PER MONITOR) ########
histograms = np.empty((len(magnet_settings), 5, monitor_bins[0], monitor_bins[1]))

states = generate_secondary_particles(shifts, n_particles, verbose=False, rng=rng)
particles = particles_from_states(states, ref, verbose=verbose)

for m_idx, m in enumerate(magnet_settings):
    shifts['magnetSettings'] = m
    print(f"Magnet setting: {m}")
    line, env, ref = line_init(shifts=shifts, verbose=verbose)

    histograms[m_idx], xedges, yedges = track_monitor(line, particles)


print(f"Shape of h: {histograms.shape}")

h = histograms[3]  # Use the first magnet setting for plotting
# Plot the monitor histograms
monitor_bins = 100  # Define bins if not already defined

if plot:
    fig, axs = plt.subplots(1, 5, figsize=(20, 4), tight_layout=True)

    for i in range(5):
        im = axs[i].imshow(h[i].T, origin='lower', 
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                        aspect='auto')
        axs[i].set_xlabel('x [m]')
        axs[i].set_ylabel('y [m]')
        p_alive = np.sum(h[i])
        axs[i].set_title(f'Monitor {i}\nAlive={p_alive}')

        plt.colorbar(im, ax=axs[i], label='Counts')


#%% !!!!!!!!!! THIS PART SHOWS AT MONITORS + DIPOLE, WITH FINAL ALIVE PARTICLES ONLY (N_PARS IS CONSTANT) !!!!!!!!!! ########

particle_matrix = {}
alive_matrix = {}

places_matrix = {}
for m_idx, m in enumerate(magnet_settings):
    particle_list, s_values = track_line(line, particles)

    particle_matrix[m] = particle_list

    # particle_list = a list with 7 elements, that tracks the particles whole object at 7 different locations:
    # 0: start, 1: dipole exit, 2: after m0, 3: after m1, ... 6: after m4 (end)

    final_alive = particle_list[-1].state > 0
    alive_particles = []
    for p in particle_list:
        alive_particles.append(p.filter(final_alive))

    alive_matrix[m] = alive_particles

    places_matrix[m] = []
    for p in alive_particles:
        x = p.x
        y = p.y

        places_matrix[m].append((x, y))

# Plot the XY distributions at each monitor for the last magnet setting
fig, axs = plt.subplots(1, 6, figsize=(21, 4), tight_layout=True)

for idx in range(1, 7):
    x, y = places_matrix[magnet_settings[2 ]][idx]
    # Print sizes of x,y
    print(f"Monitor {idx}: x shape: {x.shape}, y shape: {y.shape}")
    h, xedges, yedges = np.histogram2d(x, y, bins=monitor_bins)
    im = axs[idx-1].imshow(h.T, origin='lower', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    aspect='auto')
    axs[idx-1].set_xlabel('x [m]')
    axs[idx-1].set_ylabel('y [m]')
    p_alive = len(x)
    axs[idx-1].set_title(f'After elem {idx}\nAlive={p_alive}')

    plt.colorbar(im, ax=axs[idx-1], label='Counts')



pz = {}
for m_idx, m in enumerate(magnet_settings):
    alive_particles = alive_matrix[m]
    pz[m] = []
    for p in alive_particles:
        p_tot = p.delta * ref['p'] + ref['p'] # in eV/c
        px = p.px * ref['p'] # in eV/c
        py = p.py * ref['p'] # in eV/c
        pz[m].append(np.sqrt(p_tot**2 - px**2 - py**2) * 1e-9) # in GeV/c

        print(f"pz: min={min(pz[m][-1])}, max={max(pz[m][-1])}, mean={np.mean(pz[m][-1])} in GeV/c, len = {len(pz[m][-1])}")
    print(f"----------- FINISHED MAGNET SETTING = {m} -----------")

plt.show()