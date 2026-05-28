#!/bin/bash
# P2b: filter_only with ratio=1.0
# GPU 7

cd /work/2025/liusiyu/track2_rebuild
export CUDA_VISIBLE_DEVICES=7

python train.py \
    --cfg-path train_configs/filter_only_ratio_1.0.yaml \
    --options run.device=cuda run.world_size=1
