#!/usr/bin/env bash

### Parameters Defining Locations ###
### ============================= ###

export BASEPATH="/srv01/agrp/galeven/fresh-start/"
export LOGDIR="${BASEPATH}/logs"
mkdir -p ${LOGDIR}


### Parameters Defining the LumiCalcProd Job ###
### ======================================== ###

cd ${BASEPATH}
cd Simulation/batchScripts
### User Input ###
### ========== ###
for i in {0..2}
do
    echo "Submitting job ${i}..."
    qsub -q N -v BASEPATH="${BASEPATH}",idx="${i}" -o ${LOGDIR} -e ${LOGDIR} pyPlot.sh
done

