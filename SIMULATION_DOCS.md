# XSuite Simulation and Data Creation Documentation

## Code Reference / Function Dictionary

| Function | File | Description |
| :--- | :--- | :--- |
| [`line_init`](Simulation/sim_functions.py#L414) | `sim_functions.py` | Initializes the beamline, constructing elements and applying misalignments. |
| [`quadElement`](Simulation/sim_functions.py#L341) | `sim_functions.py` | Helper to create a quadrupole with apertures and shifts. |
| [`dipoleElement`](Simulation/sim_functions.py#L370) | `sim_functions.py` | Helper to create a dipole with apertures and shifts. |
| [`grad_kG_to_k`](Simulation/sim_functions.py#L81) | `sim_functions.py` | Converts quadrupole gradient (kG/cm) to normalized strength $k_1$. |
| [`B_T_to_k`](Simulation/sim_functions.py#L87) | `sim_functions.py` | Converts dipole field (T) to normalized bending strength $k_0$. |
| [`GenerateGaussianBeam`](Simulation/sim_functions.py#L227) | `sim_functions.py` | Generates initial 6D coordinates for a single particle. |
| [`generate_secondary_particles`](Simulation/sim_functions.py#L93) | `sim_functions.py` | Generates a batch of particles, handling secondary physics. |
| [`track_monitor`](Simulation/sim_functions.py#L550) | `sim_functions.py` | Tracks particles through the line and returns the monitor image. |
| [`rand_from_scratch_histogram`](Simulation/dataset_funcs.py#L7) | `dataset_funcs.py` | Main loop for generating a batch of randomized data. |
| [`shifts_array_random`](Simulation/dataset_funcs.py#L236) | `dataset_funcs.py` | Generates random misalignment configurations from ranges. |
| [`is_valid`](Simulation/dataset_funcs.py#L295) | `dataset_funcs.py` | Validates if a generated histogram has sufficient hits. |
| [`save_histogarms_hd5`](Simulation/dataset_funcs.py#L130) | `dataset_funcs.py` | Saves generated data batches to HDF5 format. |
| [`merge_hdf5_files`](Simulation/file_merger.py#L54) | `file_merger.py` | Merges multiple partial HDF5 datasets into one file. |

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

### Coordinate System
Xsuite uses a normalized coordinate system where transverse momenta $p_x, p_y$ are normalized by the reference momentum $P_0$ (i.e., $p_x = P_x/P_0$). The longitudinal coordinate $\zeta$ represents the time lag distance, which is zero for our purpose, and $\delta$ is the relative momentum deviation $(P-P_0)/P_0$. Note that $P$ is the total momentum. 
In our simulation, physical coordinates (in GeV/c) are explicitly converted to these normalized units before initializing the `xpart.Particles` object.

### Secondary Particles
The `generate_secondary_particles` function iterates $N$ times to create a list of particle states. It calls `GenerateGaussianBeam` for each particle. It can also simulate secondary physics processes (like Bremsstrahlung) by sampling energies from a probability density function (PDF) using the `bremss` module.

## 3. Parameters and Configuration

### The `shifts` Dictionary
The `shifts` dictionary serves as a **template** for the state of the machine. It defines the hierarchical structure of all possible misalignments and settings.
- **Structure**: It contains keys for each element (e.g., `q0`, `q1`, `q2` for quadrupoles, `beam` for initial beam parameters).
- **Parameters**: For each element, it defines:
    - `x`, `y`: Transverse shifts [m].
    - `ang_x`, `ang_y`, `ang_z`: Rotations [degrees].
- **Magnet Settings**: The `magnetSettings` key controls the current setting of the main dipole (e.g., 490, 502), which affects the reference trajectory.

### The `shifts_range` Dictionary
The `shifts_range` dictionary mirrors the structure of `shifts` but defines the **sampling space** for randomization.
- **Ranges**: Instead of single values, it contains tuples `(min, max)`.
- **Fixed Values**: If a value is 0 or a single number, it remains fixed.
- **Usage**: This dictionary tells the random number generator the valid bounds for each parameter. For example, `shifts_range['q0']['x'] = (-2e-2, 2e-2)` means the x-shift of q0 will be sampled uniformly between -2cm and +2cm.
- **Why those values?**: I don't want to generate a parameter which will kill all my particles. Using trail and error, I've determined that for bigger ranges than the ones chosen, there are very few instances which can create a surviving particle configuration. 


## 4. Randomization

Randomization is a critical part of the simulation for generating diverse datasets. It is implemented using the `numpy.random.Generator` API to ensure reproducibility.

## 5. Dataset Creation
The final HDF5 file produced by the simulation (after merging) contains datasets with the following dimensions:
- **Histograms**: `(N_samples, N_magnet_settings, Width, Height)`
  - `Width` = 256, `Height` = 128 (defined by `monitor_bins`)
  - `N_magnet_settings` depends on the configuration (e.g., 3 or 7).
  - `N_samples` is the total number of generated beamlines (batches × jobs).
- **Labels (Shifts)**: `(N_samples, N_magnet_settings, N_parameters)`
  - `N_parameters` is the flattened number of shift parameters (calculated from `shifts` dictionary).
The simulation configuration is centralized in `Simulation/params.py`. This file defines the physical constants, beam parameters, and most importantly, the misalignment structure and ranges.

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

### Data Merging
The simulation is typically run in many parallel jobs, each producing a small HDF5 file. The `Simulation/file_merger.py` script is used to combine these into a single dataset.
- **Operation**: It scans a specified directory for `.h5` files.
- **Merging**: It concatenates the `train`, `val`, and `test` datasets from all found files along the sample dimension.
- **Output**: It produces a single HDF5 file containing the aggregated data, ready for training.
- **Usage**: `python Simulation/file_merger.py [directory_path]`


## 6. Generating Random Data from Ranges
The randomization logic is encapsulated in `shifts_array_random` (in `Simulation/dataset_funcs.py`).
1.  **Iteration**: The function iterates through every key in the `shifts_range` dictionary.
2.  **Sampling**: For each parameter, if a tuple `(min, max)` is found, it uses `rng.uniform(min, max)` to generate a random value within that interval.
3.  **Construction**: These random values are populated into a copy of the `shifts` template, creating a unique `shift` configuration dictionary.
4.  **Magnet Settings**: This process is repeated for each requested magnet setting (e.g., 490, 502), ensuring the same random misalignments are tested across different operational modes if needed.

This approach allows for flexible control over which parameters are varied and by how much, simply by editing `Simulation/params.py`.
