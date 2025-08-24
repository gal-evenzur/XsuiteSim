import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt



# Beam parameters
lambda_0 = 632.8e-9  # Wavelength in meters (e.g., HeNe laser)
w_0 = 0.5e-3        # Beam waist radius at z=0 in meters
n_part = 5000       # Number of particles to track

# Calculate derived beam parameters
z_R = np.pi * w_0**2 / lambda_0  # Rayleigh range
theta_0 = lambda_0 / (np.pi * w_0) # Far-field divergence
context = xo.ContextCpu()

env = xt.Environment()
env['kquad'] = 0.1
env['kdipole'] = 0.3


env.new('q0', xt.Quadrupole, length=0.5, k1='kquad')
env.new('q1', xt.Quadrupole, length=0.5, k1='-kquad')
env.new('q2', xt.Quadrupole, length=0.5, k1='kquad')
env.new('dd', xt.Bend, length=0.5, k0='kdipole')

line = env.new_line(components=[
    env.place('q0', at=3),
    env.place('q1', at=5),
    env.place('q2', at=7),
    env.place('dd', at=12)
])


line.build_tracker(_context=context)


line.particle_ref = xt.Particles(p0c=1e6, #Energy = 1MeV
                                  q0=1, #Charge
                                    mass0=xt.ELECTRON_MASS_EV)

tt = line.get_table()
tt.show()

# --- Generate particles at the waist (z=0) ---
# At the waist, the phase space distribution is upright.
particles = xt.Particles(
    _context=context,
    p0c=xt.ELECTRON_MASS_EV,  # Reference momentum (not critical for this simulation)
    x=np.random.normal(0, w_0, n_part),
    px=np.random.normal(0, theta_0, n_part),
    y=np.random.normal(0, w_0, n_part),
    py=np.random.normal(0, theta_0, n_part),
)

# --- Plot initial phase space at z=0 ---
plt.figure(figsize=(10, 5))
plt.suptitle("Phase space at z = 0 m")
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
plt.show()

