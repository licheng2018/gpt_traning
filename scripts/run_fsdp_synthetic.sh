#!/bin/bash

source scripts/env.sh

mkdir -p logs

torchrun --nproc_per_node=4 train.py \
  --strategy fsdp \
  --dataset synthetic \
  --seq_len 512 \
  --microbatch 1 \
  --grad_accum 8 \
  --steps 50 \
  --warmup_steps 10 \
  --fp16 \
  2>&1 | tee logs/fsdp_synthetic.log