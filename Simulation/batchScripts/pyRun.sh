#!/usr/bin/env bash

### PBS Job Specifications ###
### ====================== ###

#PBS -m n
#PBS -l select=1:ncpus=1:mem=2gb -l walltime=10:00:00 -l io=1

export IOTHROTTLE_VERBOSE=1

### Setting Up the Environment ###q
### ========================== ###

cd ${BASEPATH}
source setup.sh
### Running the Job ###
### =============== ###

cd Simulation/
echo "Python initiate"
python3 create_dataset.py ${idx}
