import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
import xpart as xp



# Beam parameters
lambda_0 = 632.8e-9  # Wavelength in meters (e.g., HeNe laser)
w_0 = 0.5e-3        # Beam waist radius at z=0 in meters
n_part = 500       # Number of particles to track

# Calculate derived beam parameters
z_R = np.pi * w_0**2 / lambda_0  # Rayleigh range
theta_0 = lambda_0 / (np.pi * w_0) # Far-field divergence
context = xo.ContextCpu()

p0c = 1e6 # MeV
q = 1

def BtoK(strength, p0c, q):
    # Works for B and for G
    return strength * q / p0c

env = xt.Environment()
env['kquad1'] = BtoK(0, p0c, q)
env['kquad2'] = BtoK(0, p0c, q)
env['kdipole'] = BtoK(0, p0c, q)


env.new('q0', xt.Quadrupole, length=1, k1='kquad1')
env.new('q1', xt.Quadrupole, length=1, k1s='kquad2')
env.new('q2', xt.Quadrupole, length=1, k1='kquad1')
env.new('dd', xt.Bend, length=0.5, k0='kdipole')

line = env.new_line(components=[
    env.place('q0', at=3),
    env.place('q1', at=5),
    env.place('q2', at=7),
    env.place('dd', at=12)
])


line.build_tracker(_context=context)

tt = line.get_table()
tt.show()


# Twiss

tw = line.twiss(
    method='4d',
    init=xt.TwissInit(
        x=0, px = 0, y = 0, py = 0,
    ),
    end='_end_point',
)



fig1 = plt.figure(figsize=(6.4, 4.8*1.5))
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

# fig1.suptitle(
#     r'$q_x$ = ' f'{tw.qx:.5f}' r' $q_y$ = ' f'{tw.qy:.5f}' '\n'
#     r"$Q'_x$ = " f'{tw.dqx:.2f}' r" $Q'_y$ = " f'{tw.dqy:.2f}'
#     r' $\gamma_{tr}$ = '  f'{1/np.sqrt(tw.momentum_compaction_factor):.2f}'
# )

fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
plt.show()

# --- Generate particles at the waist (z=0) ---
# At the waist, the phase space distribution is upright.

# Horizontal plane: generate gaussian distribution in normalized coordinates
x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(n_part)

# Vertical plane: generate pencil distribution in normalized coordinates
pencil_cut_sigmas = 6.
pencil_dr_sigmas = 0.7
nemitt_x = 2.5e-6
nemitt_y = 3e-6
y_in_sigmas, py_in_sigmas, r_points, theta_points = xp.generate_2D_pencil(
                             num_particles=n_part,
                             pos_cut_sigmas=pencil_cut_sigmas,
                             dr_sigmas=pencil_dr_sigmas,
                             side='+-')

# Longitudinal plane: generate gaussian distribution matched to bucket 
zeta, delta = xp.generate_longitudinal_coordinates(
        num_particles=n_part, distribution='gaussian',
        sigma_z=10e-2, line=line)

# Build particles:
#    - scale with given emittances
#    - transform to physical coordinates (using 1-turn matrix)
#    - handle dispersion
#    - center around the closed orbit
particles = line.build_particles(
            zeta=zeta, delta=delta,
            x_norm=x_in_sigmas, px_norm=px_in_sigmas,
            y_norm=y_in_sigmas, py_norm=py_in_sigmas,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y)

# --- Plot initial phase space at z=0 ---


# Create figure and axes properly
fig = plt.figure(figsize=(10, 5))
fig.suptitle("Phase space at z = 0 m")
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(particles.x * 1e3, particles.px * 1e3, '.', markersize=1)
ax1.set_xlabel("x [mm]")
ax1.set_ylabel("px [mrad]")
ax1.axis('equal')
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(particles.y * 1e3, particles.py * 1e3, '.', markersize=1)
ax2.set_xlabel("y [mm]")
ax2.set_ylabel("py [mrad]")
ax2.axis('equal')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# --- Propagate to z=0.3 m and plot ---
line.track(particles, ele_start=0, ele_stop=1)


plt.figure(figsize=(10, 5))
plt.suptitle("Phase space at z = 0.3 m")
plt.subplot(1, 2, 1)
plt.plot(particles.x * 1e3, particles.px * 1e3, '.', markersize=1)
plt.xlabel("x [mm]")
plt.ylabel("px [mrad]")
plt.axis('equal')
plt.subplot(1, 2, 2)
plt.plot(particles.y * 1e3, particles.py * 1e3, '.', markersize=1)
plt.xlabel("y [mm]")
plt.ylabel("py [mrad]")
plt.axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# --- Propagate to z=0.6 m and plot ---
elenames = line.element_names
print(elenames)
stop = len(elenames)
line.track(particles, ele_start=1, ele_stop='dd')

plt.figure(figsize=(10, 5))
s_pos = line.get_s_position(at_elements=elenames[stop-1], mode='upstream')
plt.suptitle(f"Phase space at s={s_pos}\nat element: {elenames[stop-1]}")
plt.subplot(1, 2, 1)
plt.plot(particles.x * 1e3, particles.px * 1e3, '.', markersize=1)
plt.xlabel("x [mm]")
plt.ylabel("px [mrad]")
plt.axis('equal')
plt.subplot(1, 2, 2)
plt.plot(particles.y * 1e3, particles.py * 1e3, '.', markersize=1)
plt.xlabel("y [mm]")
plt.ylabel("py [mrad]")
plt.axis('equal')
plt.tight_layout()
# plt.show()

