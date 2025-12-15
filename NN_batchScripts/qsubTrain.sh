#!/usr/bin/env bash

### Parameters Defining Locations ###
### ============================= ###

export BASEPATH="/srv01/agrp/galeven/fresh-start"
export LOGDIR="${BASEPATH}/logs_NN"
mkdir -p ${LOGDIR}


### Parameters Defining the LumiCalcProd Job ###
### ======================================== ###

cd ${BASEPATH}
cd NN_batchScripts
### User Input ###
### ========== ###
CKPT_FLAG=0 # 0: start from scratch, 1: from checkpoint 1, 2: from checkpoint 2...
echo "starting training..."
for CKPT_FLAG in {1..11}
do
    qsub -q N -v BASEPATH="${BASEPATH}",CKPT_FLAG="${CKPT_FLAG}" -o ${LOGDIR} -e ${LOGDIR} pyTrain.sh
done
