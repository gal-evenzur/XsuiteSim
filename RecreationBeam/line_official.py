import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
import xpart as xp

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
    # m is in eV
    # E is in eV
    # p is in eV / c
    pc = (E**2 - (E_rest)**2)**0.5
    return pc / u['c']

ref = {
    'q': 1,
    'p': p_from_E(1e9, u['rest_e']),  # E = 1 MeV
    'betx_0': 1.0,
    'alfx_0': 0.0,
    'bety_0': 1.0,
    'alfy_0': 0.0,
    'num_particles': 300
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

env.elements['m0'] = xt.ParticlesMonitor(num_particles=ref['num_particles'],
                                start_at_turn=0, stop_at_turn=1,
                                auto_to_numpy=True)
env.elements['m1'] = xt.ParticlesMonitor(num_particles=ref['num_particles'],
                                start_at_turn=0, stop_at_turn=1,
                                auto_to_numpy=True)
env.elements['m2'] = xt.ParticlesMonitor(num_particles=ref['num_particles'],
                                start_at_turn=0, stop_at_turn=1,
                                auto_to_numpy=True)
env.elements['md'] = xt.ParticlesMonitor(num_particles=ref['num_particles'],
                                start_at_turn=0, stop_at_turn=1,
                                auto_to_numpy=True)

# Creating Line 
line = env.new_line(components=[
    env.new('d0', xt.Drift, length=3.6),
    env.place('m0'),
    env.new('q0', xt.Quadrupole, length='qL', k1='kq_p'),
    env.new('d0.1', xt.Drift, length=1.3),
    env.place('m1'),
    env.new('q1', xt.Quadrupole, length='qL', k1s='kq_n'),
    env.new('d1.2', xt.Drift, length=1.3),
    env.place('m2'),
    env.new('q2', xt.Quadrupole, length='qL', k1='kq_p'),
    env.new('d2.2', xt.Drift, length=1.2),
    env.place('md'),
    env.new('dd', xt.Bend, length=0.5, k0='kd'),
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
    import matplotlib.pyplot as plt
    plt.close('all')
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

fx0     = +0*u['um_to_m'] ### TODO??? ### this is where the beam is shot from
fy0     = -1000*u['um_to_m'] ### TODO??? ### this is where the beam is shot from
fz0     = -200*u['cm_to_m'] ### fixed, just has to be before the Be window ### this is where the beam is shot from
fsigmax = 50*u['um_to_m'] ## beam sigma
fsigmay = 50*u['um_to_m'] ## beam sigma
fsigmaz = 150*u['um_to_m'] ## beam sigma


def GenerateGaussianBeam(E_GeV,mass_GeV,charge,mks=False):
    fbeamfocus  = 0
    lf          = E_GeV/mass_GeV
    femittancex = 50e-3*u['mm_to_m']/lf ### mm-rad
    femittancey = 50e-3*u['mm_to_m']/lf ### mm-rad
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
    pz0 = pz*u['GeV_to_kgms'] # kg*m/s
    px0 = px*u['GeV_to_kgms'] # kg*m/s
    py0 = py*u['GeV_to_kgms'] # kg*m/s
    mass_kg = mass_GeV*u['GeV_to_kgm2s2']/u['c2'] # kg
    ### state
    state_mks = [x0,y0,z0, px0,py0,pz0, mass_kg,charge] ### [x[m],y[m],z[m], px[kg*m/s],py[kg*m/s],pz[kg*m/s], m[kg],q[unit]]
    state_nat = [x0,y0,z0, px,py,pz, mass_GeV,charge]   ### [x[m],y[m],z[m], px[GeV],py[GeV],pz[GeV], m[GeV],q[unit]]
    return state_mks if(mks) else state_nat

# Generate a batch of 300 particles with Gaussian distributions
n_particles = ref['num_particles']

# Create arrays to store particle parameters
x = np.random.normal(fx0, fsigmax, n_particles)
y = np.random.normal(fy0, fsigmay, n_particles)
z = np.random.normal(fz0, fsigmaz, n_particles)

# Energy and momentum calculations
E_GeV = 1.0  # 1 GeV electron beam
mass_GeV = u['rest_e'] / 1e9  # Electron rest mass in GeV
lf = E_GeV / mass_GeV  # Lorentz factor

# Compute emittances and beam parameters
femittancex = 50e-3 * u['mm_to_m'] / lf  # mm-rad
femittancey = 50e-3 * u['mm_to_m'] / lf  # mm-rad
fbetax = (fsigmax**2) / femittancex
fbetay = (fsigmay**2) / femittancey


# Initialize momentum arrays
px = np.zeros(n_particles)
py = np.zeros(n_particles)
pz = np.zeros(n_particles)

# Generate momentum distribution for each particle
for i in range(n_particles):
    zdrift = z[i] - 0  # Assuming beam focus is at z=0
    
    # x-plane momentum
    meandx = x[i] * zdrift / (zdrift**2 + fbetax**2)
    sigmadx = np.sqrt(femittancex * fbetax / (zdrift**2 + fbetax**2))
    dx = np.random.normal(meandx, sigmadx)
    
    # y-plane momentum
    meandy = y[i] * zdrift / (zdrift**2 + fbetay**2)
    sigmady = np.sqrt(femittancey * fbetay / (zdrift**2 + fbetay**2))
    dy = np.random.normal(meandy, sigmady)
    
    # Momentum components
    pz[i] = np.sqrt((E_GeV**2 - mass_GeV**2) / (dx**2 + dy**2 + 1.0))
    px[i] = dx * pz[i]
    py[i] = dy * pz[i]

# Create particle object for tracking
particles = xp.Particles(
    x=x,
    px=px / pz,  # Normalized to reference momentum
    y=y,
    py=py / pz,  # Normalized to reference momentum
    delta=(pz / ref['p'] - 1),  # (p - p0)/p0
    zeta=np.zeros(n_particles),  # Assuming all particles start at same longitudinal position
    _context=ctx,
)
# Print the number of particles
print(f"Number of particles: {particles.x.size}")
tt = line.get_table()
print(tt)


line.track(particles)


def plot_monitors():
    m = [env[f'm{i}'] for i in range(3)] + [env['md']]


    # Plot the particle trajectory at monitor points
    plt.figure(figsize=(10, 4))
    s_values = np.squeeze([data.s for data in m])
    x_values = np.squeeze([data.x for data in m])
    y_values = np.squeeze([data.y for data in m])

    print(env['m1'].s)

    # Create individual phase space plots for each monitor
    for i, mon in enumerate(m):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Monitor m{i} at s = {np.squeeze(mon.s[0]):.2f} m')
        
        # x-px phase space
        axs[0].scatter(mon.x, mon.px, color='blue')
        axs[0].set_xlabel('x [m]')
        axs[0].set_ylabel('px []')
        axs[0].set_title('x-px Phase Space')
        axs[0].grid(True)
        
        # y-py phase space
        axs[1].scatter(mon.y, mon.py, color='red')
        axs[1].set_xlabel('y [m]')
        axs[1].set_ylabel('py []')
        axs[1].set_title('y-py Phase Space')
        axs[1].grid(True)
        
        # zeta-delta phase space
        axs[2].scatter(mon.zeta, mon.delta, color='green')
        axs[2].set_xlabel('zeta [m]')
        axs[2].set_ylabel('delta []')
        axs[2].set_title('zeta-delta Phase Space')
        axs[2].grid(True)
        
        plt.tight_layout()
    
    # Plot particle trajectory as s progresses
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, x_values, 'b-o', label='x')
    plt.plot(s_values, y_values, 'r-o', label='y')
    
    # Add markers for element positions
    for name, s_pos in zip(tt.name, tt.s):
        if 'q' in name.lower():
            plt.axvline(x=s_pos, color='g', alpha=0.3, linestyle='--', label='_')
        elif 'dd' in name.lower() and len(name) > 1:
            plt.axvline(x=s_pos, color='m', alpha=0.3, linestyle='--', label='_')
    
    plt.xlabel('s [m]')
    plt.ylabel('Position [m]')
    plt.title('Particle Trajectory')
    plt.legend()
    plt.grid(True)

    # Plotting the last step
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Last Image, after dipole')

    # x-px phase space
    axs[0].scatter(particles.x, particles.px, color='blue')
    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('px []')
    axs[0].set_title('x-px Phase Space')
    axs[0].grid(True)

    # y-py phase space
    axs[1].scatter(particles.y, particles.py, color='red')
    axs[1].set_xlabel('y [m]')
    axs[1].set_ylabel('py []')
    axs[1].set_title('y-py Phase Space')
    axs[1].grid(True)

    # zeta-delta phase space
    axs[2].scatter(particles.zeta, particles.delta, color='green')
    axs[2].set_xlabel('zeta [m]')
    axs[2].set_ylabel('delta []')
    axs[2].set_title('zeta-delta Phase Space')
    axs[2].grid(True)

    plt.tight_layout()


plot_monitors()
plt.show()
