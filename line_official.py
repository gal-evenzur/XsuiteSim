import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt


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
    # m is in eV
    # E is in eV
    # p is in eV / c
    pc = (E**2 - (E_rest)**2)**0.5
    return pc / u['c']

ref = {
    'q': 1,
    'p': p_from_E(1e9, u['rest_e']),  # E = 1 MeV
}

print(ref['p'])

env = xt.Environment()
def grad_kG_to_k(grad_kG):
    grad_T = grad_kG * u['kG_to_T']
    k = grad_T / ref['p']
    return k

def B_T_to_k(B_T):
    k = B_T / ref['p']
    return k

env['kq_p'] = grad_kG_to_k(-6.66)
env['kq_n'] = grad_kG_to_k(28.86)
env['kd'] = B_T_to_k(0.219)


env['qL'] = 0.3

# env.new('q0', xt.Quadrupole, length=1.0, k1='kq_p')
# env.new('q1', xt.Quadrupole, length='qL', k1s='kq_n')
# env.new('q2', xt.Quadrupole, length=1.0, k1='kq_p')
# env.new('dd', xt.Bend, length=0.5, k0='kd')

line = env.new_line(components=[
    env.new('d0', xt.Drift, length=0.5),
    env.new('q0', xt.Quadrupole, length=1.0, k1='kq_p'),
    env.new('d0.1', xt.Drift, length=1.2),
    env.new('q1', xt.Quadrupole, length='qL', k1s='kq_n'),
    env.new('d1.2', xt.Drift, length=1.2),
    env.new('q2', xt.Quadrupole, length=1.0, k1='kq_p'),
    env.new('d2.2', xt.Drift, length=1.2),
    env.new('dd', xt.Bend, length=0.5, k0='kd'),

])


line.particle_ref = xt.Particles(
    p0c=ref['p'] * u['c'],  # in eV
    mass0=xt.ELECTRON_MASS_EV,
    q0=1,
)

line.build_tracker()



init = xt.TwissInit(betx=10, alfx=0, bety=10, alfy=0)  # example values

tw = line.twiss(
    method='4d',
    init=init,
    start='d0',
    end='_end_point',
)


import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
spbet = plt.subplot(3,1,1)
spco = plt.subplot(3,1,2, sharex=spbet)
spdisp = plt.subplot(3,1,3, sharex=spbet)

spbet.plot(tw.s, tw.betx)
spbet.plot(tw.s, tw.bety)
spbet.set_ylabel(r'$\beta_{x,y}$ [m]')

spco.plot(tw.s, tw.x)
spco.plot(tw.s, tw.y)
spco.set_ylabel(r'(Closed orbit)$_{x,y}$ [m]')

spdisp.plot(tw.s, tw.dx)
spdisp.plot(tw.s, tw.dy)
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.set_xlabel('s [m]')

fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
plt.show()