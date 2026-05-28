#!/bin/bash
# P2a: filter_only with cmef_num_layers=2
# GPU 6

cd /work/2025/liusiyu/track2_rebuild
export CUDA_VISIBLE_DEVICES=6

python train.py \
    --cfg-path train_configs/filter_only_cmef_layers_2.yaml \
    --options run.device=cuda run.world_size=1
