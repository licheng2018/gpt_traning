#!/bin/bash
set -euo pipefail

source scripts/env.sh
mkdir -p logs outputs

torchrun --nproc_per_node=4 train.py \
  --strategy ddp \
  --dataset wikitext \
  --seq_len 1024 \
  --microbatch 2 \
  --grad_accum 8 \
  --steps 300 \
  --warmup_steps 50 \
  --fp16 \
  --log_every 10 \
  --out_dir ./outputs/ddp_wikitext_seq1024_mb2_acc8_4xh100 \
  2>&1 | tee logs/ddp_wikitext_seq1024_mb2_acc8_4xh100.log