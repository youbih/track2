#!/bin/bash
# Ablation 2: Only add wheel label filtering, keep frozen_llm=True
# Run on GPU 2

cd /work/2025/liusiyu/track2_rebuild
export CUDA_VISIBLE_DEVICES=2

python train.py \
    --cfg-path train_configs/ablation_filter_only.yaml \
    --options run.device=cuda run.world_size=1
