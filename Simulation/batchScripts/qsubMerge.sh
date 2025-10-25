#!/usr/bin/env bash

### Parameters Defining Locations ###
### ============================= ###

export BASEPATH="/srv01/agrp/galeven/fresh-start"
export LOGDIR="${BASEPATH}/logs"
mkdir -p ${LOGDIR}
STORAGEPATH="/storage/agrp/galeven/"


### Parameters Defining the LumiCalcProd Job ###
### ======================================== ###

cd ${BASEPATH}
cd Simulation/batchScripts
### User Input ###
### ========== ###

echo "starting merge..."
qsub -q N -v BASEPATH="${BASEPATH}",idx="${i},STORAGEPATH=${STORAGEPATH}" -o ${LOGDIR} -e ${LOGDIR} pyMerge.sh
