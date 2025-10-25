#!/bin/bash

# -------------------------------
# Setup script for xsuite project
# -------------------------------

# Load ATLAS / LCG environment
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_108 x86_64-el9-gcc13-opt"

# Virtual environment folder name
VENV_DIR="xsuite-env"

# 1. Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR ..."
    python3 -m venv --system-site-packages $VENV_DIR
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# 2. Activate the venv
echo "Activating virtual environment..."
source ${VENV_DIR}/bin/activate

# 3. List of required packages
REQUIRED_PKGS=("xsuite" "numpy" "matplotlib" "h5py")

# 4. Check and install missing packages
for pkg in "${REQUIRED_PKGS[@]}"; do
    if python -c "import $pkg" &> /dev/null; then
        echo "Package '$pkg' already installed."
    else
        echo "Installing package '$pkg'..."
        pip install $pkg
    fi
done

echo "Setup complete. Virtual environment is active."