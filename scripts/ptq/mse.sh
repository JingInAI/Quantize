#!/bin/bash

ROOT='results/Compress'
MODEL=$1    # resnet18
W_BITS=$2   # 8
A_BITS=$3   # 8
GPU=$4      # 4

if [ ${A_BITS} -ge 6 ]; then
    A_PERCENTILE=0.0
else
    A_PERCENTILE='1e-3'
fi

for SEED in 1 2 3
do
    DIR=${ROOT}/results/ptq/mse/ptq_${MODEL}_w${W_BITS}a${A_BITS}_bnf_sym_chan_in1k_16shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python -u main.py \
        --cfg configs/runners/ptq/mse/ptq_rn18_w8a8_bnf_sym_chan_in1k_16shots.yaml \
        --opts \
        model.name=${MODEL} \
        quant.default.weight.n_bits=${W_BITS} \
        quant.default.activation.n_bits=${A_BITS} \
        quant.default.activation.range.percentile=${A_PERCENTILE} \
        gpus=${GPU} \
        seed=${SEED} \
        output_dir=${DIR}
    fi
done
