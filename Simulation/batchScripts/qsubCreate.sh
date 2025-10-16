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

base_time=$(date +%s)
### User Input ###
### ========== ###
for i in {0..1000}
do
    # Numeric ID = timestamp * 10000 + counter
    # This ensures IDs are sortable and unique
    unique_id=$((base_time * 10000 + i))
    echo "Submitting job ${unique_id}..."
    qsub -q N -v BASEPATH="${BASEPATH}",idx="${unique_id}" -o ${LOGDIR} -e ${LOGDIR} pyRun.sh

done
