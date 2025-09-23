Eavg = 3e9 # eV, for ref particle


shifts = {
    'magnetSettings': 490,
    'q0': {'x': 0.0, 'y': 0, 'ang_x': 0, 'ang_y': 0, 'ang_z': 0}, # in m, shift in x and y, ang in deg
    'q1': {'x': 0.0, 'y': 0.0, 'ang_x': 0, 'ang_y': 0, 'ang_z': 0},
    'q2': {'x': 0.0, 'y': 0.0, 'ang_x': 0, 'ang_y': 0, 'ang_z': 0},
    'beam': {'fx0': 0.0, 'fy0': 0.0, 'fz0': -2, 'fbeamfocus':0.0},
    'dd_corr': {'x': 0.0, 'y': 0.0, 'ang_x': 0, 'ang_y': 0, 'ang_z': 0},
    'dd': {'x': 0.0, 'y': 0.0, 'ang_x': 0, 'ang_y': 0, 'ang_z': 0},
}

shifts_range = {
    'q0': {'x': (-0.6e-3, 4.2e-3), 'y': (-1e-3, 1e-3), 'ang_x': (-0.5,0.5), 'ang_y': (-0.5,0.5), 'ang_z': (-0.5,0.5)}, # in m, shift in x and y, ang in deg
    'q1': {'x': (-1e-3, 1e-3), 'y': (-1e-3, 1e-3), 'ang_x': (-0.5,0.5), 'ang_y': (-0.5,0.5), 'ang_z': (-0.5,0.5)},
    'q2': {'x': (-1e-3, 1e-3), 'y': (-1e-3, 1e-3), 'ang_x': (-0.5,0.5), 'ang_y': (-0.5,0.5), 'ang_z': (-0.5,0.5)},
    'beam': {'fx0': (-1e-5, 1e-5), 'fy0': (-1e-4, 1e-4), 'fz0': (-2, -3), 'fbeamfocus': (-1e-4, 1e-4)},
    'dd_corr': {'x': 0.0, 'y': 0.0, 'ang_x': 0, 'ang_y': 0, 'ang_z': 0},
    'dd': {'x': 0.0, 'y': 0.0, 'ang_x': 0, 'ang_y': 0, 'ang_z': 0},
}

# Calculate the number of integers in shifts
def count_numeric_values(data):
    if isinstance(data, (int, float)):
        return 1
    elif isinstance(data, dict):
        return sum(count_numeric_values(value) for value in data.values())
    elif isinstance(data, (list, tuple)):
        return 1
    else:
        return 0

num_shifts = count_numeric_values(shifts)
num_shifts_range = count_numeric_values(shifts_range)


MagnetSettings = shifts['magnetSettings']
n_particles = 1e5
show_dead = True
use_integration = True
is_array = True
dat_file = 'Data/secondary_particles.h5'
histogram_dat = 'Data/histogram_data.h5'


# %% CONSTS________
u = {
    'c': 299792458,
    'c2': 8.98755179e16,
    'e': 1.602176634e-19, # elementary charge in C
    'rest_e': 0.510998950693e6, #xt.ELECTRON_MASS_EV,
    'rest_p': 938.27208943e6, #xt.PROTON_MASS_EV,
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



# %% ((((magnets))))
magsetvals = {502:[-7.637,28.55,-7.637], 490.0:[-30.68,46.42,-30.68], 490.1:[-27.99,44.98,-27.99], 490.2:[-20.38,40.42,-20.38], 490.3:[-11.56,30.05,-11.56], 490.4:[-3.37,26.72,-3.37], 490.5:[-6.66,28.86,-6.66] }
magsetdelt = {"quad0":[0,0], "quad1":[0,0], "quad2":[0,0], "xcorr":[0,0], "dipole":[0,0]} ### cm

B_dd_xcorr = 0.026107 # By
B_dd = 0.219 # Bx

zAL     = 0.30 ### the aluminum foil, cm
zBe     = -0.84 ### the beryllium window, cm
Z0      = zBe if(shifts['magnetSettings']==502) else zAL

monitor_bins = (128, 256)


# %% Partilces

fsigmax = 50*u['um_to_m'] ## beam sigma
fsigmay = 50*u['um_to_m'] ## beam sigma
fsigmaz = 150*u['um_to_m'] ## beam sigma
MM      = 9.109e-31 ## kg, positron
QQ      = +1  ## unit charge, positron
mGeV    = (MM*u['c2'])/u['GeV_to_kgm2s2'] ## GeV
E_GeV   = 10 # GeV, energy of primary partticles
Emin    = 1e-2 ## GeV
Emax    = 10 ## GeV
smear_T  = False
smear_pT  = True
smear_sigma_T_um  = 0 ## um
smear_sigma_P_GeV = 1e-4 ## GeV (=100 keV)
ZMAX    = 18 ## METERES
tmax    = ZMAX / (0.99 * u['c']) ### time range for propagation (seconds): approximate time to travel 18 meters (last detector is at ~18 meters, relativistic particles going ~c)
t_span  = (0, tmax)
max_dt  = 1e-9

import bremss as br
X0   = 35.3  if(shifts['magnetSettings']==502) else 8.897 # cm for Beryllium of Aluminum
t_cm = 0.005 if(shifts['magnetSettings']==502) else 0.01 # cm (50 or 100 µm)
E_vals, photons, eplus = br.build_pdfs(0.01,E_GeV,X0,t_cm)

