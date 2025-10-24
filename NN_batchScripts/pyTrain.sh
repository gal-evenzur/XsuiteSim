#!/usr/bin/env bash

### PBS Job Specifications ###
### ====================== ###

#PBS -m n
#PBS -l select=1:ngpus=1:mem=20gb -l walltime=02:00:00 -l io=50

export IOTHROTTLE_VERBOSE=1

### Setting Up the Environment ###
### ========================== ###

cd ${BASEPATH}
source setup.sh
### Running the Job ###
### =============== ###

echo "Python initiate"
python3 Regression_Eff.py ${CKPT_FLAG}