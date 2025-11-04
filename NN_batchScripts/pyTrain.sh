#!/usr/bin/env bash

### PBS Job Specifications ###
### ====================== ###

#PBS -m n
#PBS -l select=1:ncpus=8:ngpus=1:gputype=A5000:mem=2gb -l walltime=10:00:00 -l io=1

export IOTHROTTLE_VERBOSE=1

### Setting Up the Environment ###
### ========================== ###

cd ${BASEPATH}
cd NN_batchScripts/
source setup.sh
### Running the Job ###
### =============== ###
cd ${BASEPATH}
echo "Python initiate"
python3 Regression_Eff.py ${CKPT_FLAG}