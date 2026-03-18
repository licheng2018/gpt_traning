#!/bin/bash
set -euo pipefail

source scripts/env.sh
mkdir -p logs outputs

STRATEGY=(ddp fsdp)
CKPT=(0 1)

MB=(1 2 4 8)
ACC=(16 8 4 2)

SEQ=1024
STEPS=200
WARMUP=30

for strat in "${STRATEGY[@]}"
do
  for ckpt in "${CKPT[@]}"
  do
    for i in "${!MB[@]}"
    do
      mb=${MB[$i]}
      acc=${ACC[$i]}

      echo "Running strategy=$strat checkpoint=$ckpt seq=$SEQ mb=$mb acc=$acc"

      EXTRA_ARGS=""
      if [ "$ckpt" -eq 1 ]; then
        EXTRA_ARGS="--checkpoint"
      fi

      torchrun --nproc_per_node=4 train_check_point.py \
        --strategy "$strat" \
        --dataset synthetic \
        --seq_len "$SEQ" \
        --microbatch "$mb" \
        --grad_accum "$acc" \
        --steps "$STEPS" \
        --warmup_steps "$WARMUP" \
        --fp16 \
        --log_every 20 \
        --out_dir "./outputs/day5_${strat}_ckpt${ckpt}_seq${SEQ}_mb${mb}_acc${acc}" \
        $EXTRA_ARGS \
        2>&1 | tee "./logs/day5_${strat}_ckpt${ckpt}_seq${SEQ}_mb${mb}_acc${acc}.log"

    done
  done
done