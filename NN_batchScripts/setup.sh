#!/bin/bash

# -------------------------------
# Setup script for xsuite project
# -------------------------------

# Virtual environment folder name
# STORAGEPATH="/storage/agrp/galeven/"
STORAGEPATH=~/fresh-start/NN_batchScripts/
cd ${STORAGEPATH}

# Load ATLAS / LCG environment
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_108_cuda x86_64-el9-gcc13-opt"


VENV_DIR="pytorch-env"

# 1. Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR ..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# 2. Activate the venv
echo "Activating virtual environment..."
source ${STORAGEPATH}/${VENV_DIR}/bin/activate
echo "Virtual environment '$VENV_DIR' activated."


# 3. List of required packages
REQUIRED_PKGS=("numpy" "matplotlib" "torch" "h5py" "torchvision")

# 4. Check and install missing packages
for pkg in "${REQUIRED_PKGS[@]}"; do
    if pip show $pkg &> /dev/null; then
        echo "Package '$pkg' already installed."
    else
        echo "Installing package '$pkg'..."
        pip install $pkg
    fi
done

# 6. Check and install pytorch-ignite
if pip show pytorch-ignite &> /dev/null; then
    echo "Package 'pytorch-ignite' already installed."
else
    echo "Installing package 'pytorch-ignite'..."
    pip install pytorch-ignite
fi

echo "Setup complete. Virtual environment is active."
