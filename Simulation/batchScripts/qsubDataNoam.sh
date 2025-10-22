#!/usr/bin/env bash

### Parameters Defining Locations ###
### ============================= ###

export BASEPATH="/srv01/agrp/galeven/fresh-start"
export LOGDIR="${BASEPATH}/logs"
mkdir -p ${LOGDIR}


### ======================================== ###

cd ${BASEPATH}
cd Simulation/batchScripts
### User Input ###
### ========== ###

echo "starting merge..."
qsub -q N -v BASEPATH="${BASEPATH}" -o ${LOGDIR} -e ${LOGDIR} pyDataNoam.sh
