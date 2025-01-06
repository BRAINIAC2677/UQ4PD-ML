#!/bin/bash

DATASETS=("smile" "speech" "finger_tapping")
METHODS=("mcdropout" "bnn" "dec")

declare -A HYPERPARAMETERS

HYPERPARAMETERS["smile_mcdropout"]="--model ann --seed 914 --lr 0.005636638313326733 --max_epochs 53 --drop_prob 0.23801571998298293 --num_estimators 700 --corr_thr 0.9446033986181408 --scaler standard --optimizer adamw --weight_decay 0.0631540840367034 --beta1 0.843677246295737 --beta2 0.9202703944120154"
HYPERPARAMETERS["smile_bnn"]="--model shallow_bnn --seed 424 --lr 0.0006967700379803089 --max_epochs 76 --corr_thr 0.8006589883426534 --scaler standard --optimizer sgd --momentum 0.6033083488968148 --weight_decay 0.0417783578129261 --kl_weight 0.0012841899092900643 --num_samples 5"
HYPERPARAMETERS["smile_dec"]="--model shallow_ann --seed 618 --lr 0.040976923137632744 --max_epochs 27 --drop_prob 0.3456346210388861 --corr_thr 0.7663593323700479 --scaler standard --optimizer adamw --weight_decay 0.05077018899451539 --beta1 0.9814062450747135 --beta2 0.9052609237298104 --reg_weight 0.0005335847978785615"

HYPERPARAMETERS["speech_mcdropout"]="--model shallow_ann --seed 828 --lr 0.0007981029786428762 --max_epochs 49 --drop_prob 0.36153215020397433 --num_estimators 200 --corr_thr 0.6404653201694775 --scaler standard --optimizer adamw --weight_decay 0.07393361837303444 --beta1 0.9165804558296943 --beta2 0.9206889124723837"
HYPERPARAMETERS["speech_bnn"]="--model bnn --seed 102 --lr 0.00018920045062896081 --max_epochs 92 --corr_thr 0.7126370945673131 --scaler standard --optimizer adamw --weight_decay 0.0021809348042378662 --beta1 0.9476378278149027 --beta2 0.9770192705077213 --kl_weight 0.003186199035151547 --num_samples 8"
HYPERPARAMETERS["speech_dec"]="--model shallow_ann --seed 750 --lr 0.04501385819298678 --max_epochs 97 --drop_prob 0.47571850843328095 --corr_thr 0.0771574017336602 --scaler minmax --optimizer adamw --weight_decay 0.09997433321146701 --beta1 0.8372292131602858 --beta2 0.990077644400955 --reg_weight 0.0008742443723505313"

HYPERPARAMETERS["finger_tapping_mcdropout"]="--model shallow_ann --seed 561 --lr 0.0030217847079087932 --max_epochs 58 --drop_prob 0.30532033796497754 --num_estimators 700 --corr_thr 0.9850899042796892 --scaler standard --optimizer adamw --weight_decay 0.04728221652857254 --beta1 0.8516213400684327 --beta2 0.9439773075230373"
HYPERPARAMETERS["finger_tapping_bnn"]="--model shallow_bnn --seed 5 --lr 0.020045583767965038 --max_epochs 9 --corr_thr 0.834015383298273 --scaler minmax --optimizer sgd --momentum 0.9205390411878682 --weight_decay 0.0025649065111398054 --kl_weight 0.00010439231807535055 --num_samples 6"
HYPERPARAMETERS["finger_tapping_dec"]="--model ann --seed 985 --lr 0.0005094585189359994 --max_epochs 56 --drop_prob 0.3036340788920021 --corr_thr 0.12234239316043549 --scaler standard --optimizer sgd --momentum 0.6342621398945811 --weight_decay 0.07334347008344907 --reg_weight 2.7294108606293365e-05"

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        hyperparameters_key="${dataset}_${method}"
        hyperparameters="${HYPERPARAMETERS[$hyperparameters_key]}"
        echo "Started training on ${dataset} for ${method}."
        echo "--------------------------------------------------"
        python -m ${method}.train_${dataset} ${hyperparameters}
        echo "Ended training on ${dataset} for ${method}."
        echo "--------------------------------------------------"
    done
done
