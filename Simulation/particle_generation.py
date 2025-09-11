from sim_functions import *

secondary_states_at_foil = generate_secondary_particles(shifts, n_particles)
# Save secondary particles at foil
save_particles_to_hdf5(secondary_states_at_foil, dat_file)
