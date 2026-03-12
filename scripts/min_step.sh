#!/bin/bash
set -euo pipefail

source scripts/env.sh
mkdir -p logs outputs

torchrun --nproc_per_node=4 train.py \
  --strategy ddp \
  --dataset synthetic \
  --seq_len 512 \
  --microbatch 1 \
  --grad_accum 2 \
  --steps 10 \
  --warmup_steps 2 \
  --fp16 \
  --log_every 1