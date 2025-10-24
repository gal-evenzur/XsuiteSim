#!/bin/bash

# -------------------------------
# Setup script for xsuite project
# -------------------------------

# Virtual environment folder name
VENV_DIR="xsuite-env"

# 1. Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR ..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# 2. Activate the venv
echo "Activating virtual environment..."
source ${VENV_DIR}/bin/activate


# 6. Check and install xsuite
if python -c "import xtrack" &> /dev/null; then
    echo "Package 'xtrack' already installed."
else
    echo "Installing package 'xtrack'..."
    pip install xtrack==0.55.0
fi


# 6. Check and install xsuite
if python -c "import xpart" &> /dev/null; then
    echo "Package 'xpart' already installed."
else
    echo "Installing package 'xpart'..."
    pip install xpart==0.19.0
fi


# 3. List of required packages
REQUIRED_PKGS=("numpy" "matplotlib" "torch" "h5py" "torchvision" "tkinter")

# 4. Check and install missing packages
for pkg in "${REQUIRED_PKGS[@]}"; do
    if python -c "import $pkg" &> /dev/null; then
        echo "Package '$pkg' already installed."
    else
        echo "Installing package '$pkg'..."
        pip install $pkg
    fi
done


# 6. Check and install pytorch-ignite
if python -c "import ignite" &> /dev/null; then
    echo "Package 'pytorch-ignite' already installed."
else
    echo "Installing package 'pytorch-ignite'..."
    pip install pytorch-ignite
fi

echo "Setup complete. Virtual environment is active."
