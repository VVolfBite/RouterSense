#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/RouterSense"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3

python experiment/poc2/single_node_runner.py \
  --model olmoe \
  --model-path /root/autodl-tmp/models/OLMoE-1B-7B-0924 \
  --world-size 4 \
  --precision bf16 \
  --service-backend synthetic-matmul \
  --output-dir outputs/poc2_single_node_real_olmoe

python experiment/poc2/single_node_runner.py \
  --model qwen_moe \
  --model-path /root/autodl-tmp/models/Qwen1.5-MoE-A2.7B \
  --world-size 4 \
  --precision bf16 \
  --service-backend synthetic-token-linear \
  --output-dir outputs/poc2_single_node_real_qwen
