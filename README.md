# Xsuite Installation
For more details, refer here: [xsuite installation](https://xsuite.readthedocs.io/en/latest/installation.html).
To use xsuite, you need to have python, a linux-like terminal.
This means that a Mac/Linux computer is fine, while for a windows computer you'll have to first download [wsl](https://learn.microsoft.com/en-us/windows/wsl/install)
If you are using wsl, remember to download in wsl all the necessities you need (for example: git)

## Miniforge
First, install python in your terminal.
Now, you need to create a python environment. Xsuite recommends downloading miniforge. 
For linux/wsl terminal, run:
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```
mac terminal:
```
curl -OL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$(uname -m).sh
bash Miniforge3-MacOSX-$(uname -m).sh
```
If you are using a wsl machine, you need to download miniforge _in the wsl command line_, and _not_ in the cmd.

**Small explanation:**
A python environment is a clean python installation which is seperate from the global python in your computer. 
The seperation is important in case one project needs a specific version of a library, while another one needs a different version. 
This seperation is possible using different python environments for each project.

## Environment
After downloading miniforge, restart the terminal, and then create an environment. you can do that with:
```
conda create -n xsuite-env
conda activate xsuite-env
```
Now, you've created an environment! And also the env is "activated", which means that when you install libraries through the terminal, this libraries will only be downloaded to this environment. 
You can now install all the python libraries you need using pip. 

## Using environment in vscode
Now, I"ll explain how to use that environment using vscode.
For all operating systems, start by creating a new directory. 

_for windows_
In your normal (not wsl) vscode installation, install the wsl extension. 
Next, press ctrl+shift+p, write `WSL: Connect to WSL`
Now you are using vscode in WSL! To use the environment you picked earlier, press ctrl+shift+p, write `Python: Select interpreter`, and choose the created environment. 
And that's it folks. 

_for linux/mac_
You don't need to download the wsl extension, so your life is easier. In vscode, press ctrl+shift+p, write `Python: Select interpreter`, and choose the created environment. 
And that's it folks.

## Simulation Folder Structure
The `Simulation/` folder contains all the necessary code to run the particle tracking simulations using Xsuite. 

### Key Files
*   **`sim_functions.py`**: This is the core library. It contains functions to:
    *   Define the beamline lattice (`line_init`, `quadElement`, `dipoleElement`).
    *   Generate particles (`GenerateGaussianBeam`, `generate_secondary_particles`).
    *   Track particles through the line (`track_line`, `track_monitor`).
    *   Plot results (`plot_trajectories`, `twiss_plot`).
    *   *Check the docstrings in this file for detailed explanations of each function.*
*   **`bremss.py`**: Handles the physics of secondary particle production (Bremsstrahlung and Pair Production). It calculates energy spectra and samples new particle energies.
*   **`params.py`**: Contains configuration parameters, such as magnet settings, alignment shifts, and beam parameters.
*   **`basic_usage.ipynb`**: A Jupyter Notebook tutorial that demonstrates how to set up a simple simulation, track particles, and visualize the results. **Start here!**
*   **`SIMULATION_DOCS.md`**: A comprehensive documentation file that explains the code structure, physics models, and implementation details in depth.

### Where to find explanations?
1.  **For a quick start**: Open `basic_usage.ipynb` and run through the cells to see the simulation in action.
2.  **For understanding the code**: Read `SIMULATION_DOCS.md`. It provides a high-level overview and a function dictionary.
3.  **For detailed function logic**: Open `sim_functions.py` or `bremss.py`. Every function has a detailed docstring explaining its inputs, outputs, and physical significance.

## Dataset Generation
To train the Neural Network, you first need to generate a dataset of particle tracks. This is a two-step process: generating many small data files, and then merging them into one large dataset.

### Step 1: Generate Data Files
The script `Simulation/create_dataset.py` runs the simulation and saves the results.
**Usage:**
```bash
python Simulation/create_dataset.py <index> <storage_path>
```
*   `<index>`: An integer ID for this specific run (e.g., 1, 2, 3...). Used to seed the random number generator.
*   `<storage_path>`: The directory where the data will be saved. The script creates a `Data_2` folder inside this path.

**Example:**
```bash
python Simulation/create_dataset.py 1 ./my_data
```
This will create `./my_data/Data_2/h_1.h5`.

**Batch Generation (Cluster):**
To generate a large dataset, you typically run this script hundreds or thousands of times in parallel on a cluster. The folder `Simulation/batchScripts/` contains scripts (`qsubCreate.sh`, `pyRun.sh`) for submitting these jobs to a scheduler (like PBS).

### Step 2: Merge Data Files
After generating many `h_*.h5` files, you need to merge them into a single file for efficient training.
**Usage:**
```bash
python Simulation/file_merger.py <storage_path>
```
*   `<storage_path>`: The same path you used in Step 1. The script looks for files in `<storage_path>/Data_2/`.

This will produce a merged file (e.g., `merged_data.h5`) in the storage directory.

## Neural Network Training
The Neural Network is trained using the `Regression_Eff.py` script.

### Configuration
Before running, open `Regression_Eff.py` and check the following:
1.  **Data Path**: Find the line `data_path = ...` (around line 80). **You must change this** to point to your merged .h5 file from the previous step.
2.  **Hyperparameters**: The `hyperVar` dictionary (around line 40) controls the training:
    *   `batch_size`: Number of samples per training step.
    *   `n_epochs`: Total number of training passes.
    *   `h_lr`, `b_lr`: Learning rates for the head and body of the network.

### Running the Training
**Local Run:**
```bash
python Regression_Eff.py
```
This will start the training, print progress, and save checkpoints to a `checkpoints/` folder.

**Cluster Run:**
The folder `NN_batchScripts/` contains scripts for running the training on a cluster.
*   `pyTrain.sh`: The execution script.
*   `qsubTrain.sh`: The submission script.

## File Dictionary & Necessity
Here is a breakdown of the key files and why they are needed:

### Simulation Files (`Simulation/`)
*   **`create_dataset.py`**: The main script to generate training data. It runs the physics simulation with randomized parameters.
*   **`file_merger.py`**: Combines thousands of small simulation files into one big dataset for the NN.
*   **`dataset_funcs.py`**: Helper functions for data generation (e.g., randomizing beam parameters).
*   **`batchScripts/`**: Scripts to automate running `create_dataset.py` on a computing cluster.

### Neural Network Files
*   **`Regression_Eff.py`**: The main script for training the Neural Network. It defines the model architecture (EfficientNet), the training loop, and evaluation metrics.
*   **`NNfunctions.py`**: Contains utility functions for the NN, including:
    *   `SignalDataset`: A PyTorch Dataset class that loads and processes the .h5 data.
    *   `scale_tensor`: Normalizes the input images.
    *   `perc_error_per_parameter`: Custom metric for evaluation.
*   **`NN_batchScripts/`**: Scripts to submit the training job to a cluster. Useful for long training runs on powerful GPUs.
 
