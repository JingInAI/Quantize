#!/bin/bash

ROOT='results/Compress'
CFG=$1      # config name
MODEL=$2    # resnet18
W_BITS=$3   # 8
GPU=$4      # 4

for SEED in 1 2 3
do
    DIR=${ROOT}/results/adaround/bias_correct/${CFG}/${MODEL}/W${W_BITS}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python -u main.py \
        --cfg configs/runners/adaround/bias_correct/${CFG}.yaml \
        --opts \
        model.name=${MODEL} \
        quant.default.weight.n_bits=${W_BITS} \
        gpus=${GPU} \
        seed=${SEED} \
        output_dir=${DIR}
    fi
done
