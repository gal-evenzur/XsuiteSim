#!/usr/bin/env bash

### PBS Job Specifications ###
### ====================== ###

#PBS -m n
#PBS -l select=1:ncpus=1:mem=200gb -l walltime=10:00:00 -l io=100

export IOTHROTTLE_VERBOSE=1

### Setting Up the Environment ###
### ========================== ###

cd ${BASEPATH}
cd Simulation/batchScripts/
source setup.sh
### Running the Job ###
### =============== ###

cd ${BASEPATH}
cd Simulation/
echo "Python initiate"
python3 file_merger.py ${STORAGEPATH}