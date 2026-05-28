#!/bin/bash
# Ablation 3: Full config except hierarchical_supervision=false
# Run on GPU 3

cd /work/2025/liusiyu/track2_rebuild
export CUDA_VISIBLE_DEVICES=3

python train.py \
    --cfg-path train_configs/ablation_no_hierarchical.yaml \
    --options run.device=cuda run.world_size=1
