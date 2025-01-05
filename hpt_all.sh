#!/bin/bash

N_TRIALS=1
LOG_FILE="hpt_${N_TRIALS}_trials.log"
DATASETS=("smile" "speech" "finger_tapping")
METHODS=("mcdropout" "bnn" "dec")

if [ -f ${LOG_FILE} ]; then
    rm ${LOG_FILE}
fi

for dataset in "${DATASETS[@]}"
do
    for method in "${METHODS[@]}"
    do
        echo "Started hyperparameter tuning  on ${dataset} for ${method}."
        echo "--------------------------------------------------"
        python -m ${method}.hpt_${dataset} --n_trials ${N_TRIALS} --log_file ${LOG_FILE}
        echo "Ended hyperparameter tuning  on ${dataset} for ${method}."
        echo "--------------------------------------------------"
    done
done
