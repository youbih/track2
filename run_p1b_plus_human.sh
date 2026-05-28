#!/bin/bash
# P1b: filter_only + Human dataset
# GPU 5

cd /work/2025/liusiyu/track2_rebuild
export CUDA_VISIBLE_DEVICES=5

python train.py \
    --cfg-path train_configs/filter_only_plus_human.yaml \
    --options run.device=cuda run.world_size=1
