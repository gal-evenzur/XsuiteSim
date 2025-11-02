#!/usr/bin/env bash

### PBS Job Specifications ###
### ====================== ###

#PBS -m n
#PBS -l select=1:ncpus=1:mem=100gb -l walltime=02:00:00 -l io=10

export IOTHROTTLE_VERBOSE=1

### Setting Up the Environment ###
### ========================== ###

cd ${BASEPATH}
cd Simulation/batchScripts/
source setup.sh
### Running the Job ###
### =============== ###

### Running the Job ###
### =============== ###
DATAPATH=${BASEPATH}/merged_data/merged_data_2.h5
# DATAPATH=${BASEPATH}/Data/h_${idx}.h5

cd ${BASEPATH}
cd Simulation/
echo "Python initiate"
python3 plot_sim.py ${idx} ${DATAPATH}
