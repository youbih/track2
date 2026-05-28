#!/bin/bash
# Ablation 1: Only fix frozen_llm=False, keep everything else original
# Run on GPU 1

cd /work/2025/liusiyu/track2_rebuild
export CUDA_VISIBLE_DEVICES=5

python train.py \
    --cfg-path train_configs/ablation_baseline_fix.yaml \
    --options run.device=cuda run.world_size=1
