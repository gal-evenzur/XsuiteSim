# XSuite Simulation and Data Creation Documentation

This document explains the implementation of the simulation and data creation pipeline using `xsuite` (xtrack, xpart). The code is primarily located in the `Simulation/` directory.

## 1. The Line Itself

The beamline is defined in `Simulation/sim_functions.py`. The main function responsible for building the line is `line_init`.

### Line Construction
The `line_init` function initializes an `xtrack.Environment` and constructs the line element by element.
- **Elements**: The line is composed of drifts (`xt.Drift`), quadrupoles (`xt.Quadrupole`), bends(=dipole magnets) (`xt.Bend`), and apertures (`xt.LimitRect`, `xt.LimitRectEllipse`).
> Do note: in regular xsuite, the line itself is bending, defined by parameter h. Here, we use a straight line, so no need for that. 
- **Helper Functions**: Custom helper functions `quadElement` and `dipoleElement` are used to wrap the creation of magnets. These functions handle:
    - **Apertures**: Defining the physical limits of the beam pipe at the magnet.
    - **Misalignments**: Applying shifts (`xt.XYShift`) and rotations (`xt.SRotation`, `xt.XRotation`, `xt.YRotation`) to the magnet and its aperture.
    - **Restoration**: Ensuring the coordinate system is restored after the misaligned element so the rest of the line remains on the reference orbit.

### Physics
Magnet strengths are calculated from physical parameters and use normalized parameters. Note that $P_0$ is the total momentom of the refrence particle, and is defined arbitrarily. 
- **Quadrupoles**: The focusing strength $k_1$ is calculated from the gradient (kG/cm) using `grad_kG_to_k`. k1 is defined $k_1 = \frac{Grad \cdot q}{P_0}$ 
- **Dipoles**: The bending strength $k_0$ is calculated from the magnetic field (T) using `B_T_to_k`. Defined by: $k_0 = \frac{B_0 q}{P_0}$

## 2. Particle Generation

Particle generation is handled in `Simulation/sim_functions.py`, specifically in the `GenerateGaussianBeam` and `generate_secondary_particles` functions.

### Beam Definition
The `GenerateGaussianBeam` function creates the initial state of a single particle based on a Gaussian distribution.
- **Transverse Position ($x, y$)**: Sampled from a normal distribution defined by the beta functions ($\beta_x, \beta_y$) and emittance.
- **Longitudinal Position ($z$)**: Sampled from a normal distribution.
- **Momentum**:
    - $p_z$ is calculated to be consistent with the total energy and transverse momenta.
    - $p_x, p_y$ are derived from the angular divergence, which is also sampled normally.
- **Focusing**: The beam is constructed to be focused at a specific point (interaction point), and the drift to the start of the simulation is accounted for.

### Secondary Particles
The `generate_secondary_particles` function iterates $N$ times to create a list of particle states. It calls `GenerateGaussianBeam` for each particle. It can also simulate secondary physics processes (like Bremsstrahlung) by sampling energies from a probability density function (PDF) using the `bremss` module.

## 3. Randomization

Randomization is a critical part of the simulation for generating diverse datasets. It is implemented using the `numpy.random.Generator` API to ensure reproducibility.

### Seeding
- A random number generator (`rng`) is initialized with a seed.
- In `create_dataset.py`, the seed is either passed as a command-line argument or derived from the current time (`int(time.time() % 1e6)`).
- This `rng` instance is passed down to all functions that require randomness.

### Usage
- **Particle Coordinates**: `GenerateGaussianBeam` uses `rng.normal` to sample $x, y, z, x', y'$.
- **Physics**: `simulate_secondary_production` uses `rng` to sample particle energies.
- **Misalignments**: The magnet shifts and rotations are randomized using `rng` in the dataset generation functions.

## 4. Dataset Creation

The creation of randomized datasets is orchestrated by `Simulation/create_dataset.py` and `Simulation/dataset_funcs.py`.

### The `create_dataset.py` Script
This script is the entry point for generating data. It:
1.  **Initializes**: Sets up the random seed and output paths.
2.  **Splits**: Defines the number of batches for Training, Validation, and Testing.
3.  **Generates**: Calls `rand_from_scratch_histogram` to generate the data.
4.  **Saves**: Writes the histograms, shift labels, and metadata to an HDF5 file.

### `rand_from_scratch_histogram`
Located in `Simulation/dataset_funcs.py`, this function generates a batch of data from scratch.
- **Loop**: It iterates `n_batch` times.
- **Random Shifts**: For each batch, it calls `shifts_array_random` to generate a random configuration of magnet misalignments (shifts and rotations) based on a template and specified ranges.
- **Particle Tracking**:
    1.  It initializes a new line with the specific random shifts using `line_init(shifts=shift)`.
    2.  It tracks particles through this line using `track_monitor`.
    3.  It generates a 2D histogram (image) of the particle distribution at the monitor.
- **Validation**: It checks if the generated histogram is valid (e.g., has enough hits) using `is_valid`.
- **Output**: It returns the matrix of applied shifts (labels) and the corresponding histograms (input data).

This process ensures that each data point in the dataset corresponds to a unique, randomly misaligned beamline, providing the neural network with the variety needed to learn the relationship between beam images and magnet misalignments.
