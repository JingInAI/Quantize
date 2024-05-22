#!/bin/bash

ROOT='results/Compress'
MODEL=$1    # resnet18
W_BITS=$2   # 8
GPU=$3      # 4

for SEED in 1 2 3
do
    DIR=${ROOT}/results/adaround/awq/${MODEL}/W${W_BITS}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python -u main.py \
        --cfg configs/runners/adaround/awq/base.yaml \
        --opts \
        model.name=${MODEL} \
        quant.default.weight.n_bits=${W_BITS} \
        gpus=${GPU} \
        seed=${SEED} \
        output_dir=${DIR}
    fi
done
