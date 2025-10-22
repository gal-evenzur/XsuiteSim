#!/usr/bin/env bash

### PBS Job Specifications ###
### ====================== ###

#PBS -m n
#PBS -l select=1:ncpus=1:mem=16gb -l walltime=02:00:00 -l io=1

export IOTHROTTLE_VERBOSE=1

### Setting Up the Environment ###
### ========================== ###

cd ${BASEPATH}
source setup.sh
### Running the Job ###
### =============== ###

cd Simulation/Real_data
echo "Python initiate"
python3 data_for_noam.py 