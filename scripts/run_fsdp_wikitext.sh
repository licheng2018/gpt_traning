#!/bin/bash
set -euo pipefail

source scripts/env.sh
mkdir -p logs outputs

SEQ=(512 1024)
MB=(1 2 4 8)
ACC=(16 8 4 2)

for seq in "${SEQ[@]}"
do
  for i in "${!MB[@]}"
  do
    mb=${MB[$i]}
    acc=${ACC[$i]}

    echo "Running seq=$seq mb=$mb acc=$acc"

    torchrun --nproc_per_node=4 train.py \
      --strategy ddp \
      --dataset wikitext \
      --seq_len "$seq" \
      --microbatch "$mb" \
      --grad_accum "$acc" \
      --steps 200 \
      --warmup_steps 30 \
      --fp16 \
      --log_every 20 \
      --out_dir "./outputs/sweep_seq${seq}_mb${mb}_acc${acc}_4xh100" \
      2>&1 | tee "logs/sweep_seq${seq}_mb${mb}_acc${acc}_4xh100.log"
  done
done