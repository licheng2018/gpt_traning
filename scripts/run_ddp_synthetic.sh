#!/bin/bash
set -euo pipefail

source scripts/env.sh
mkdir -p logs outputs

torchrun --nproc_per_node=4 train.py \
  --strategy ddp \
  --dataset synthetic \
  --seq_len 512 \
  --microbatch 2 \
  --grad_accum 8 \
  --steps 50 \
  --warmup_steps 10 \
  --fp16 \
  --log_every 5 \
  --out_dir ./outputs/ddp_synth_4xh100 \
  2>&1 | tee logs/ddp_synth_4xh100.log