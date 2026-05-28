#!/bin/bash
# P1a: filter_only with frozen_llm=false
# GPU 4

cd /work/2025/liusiyu/track2_rebuild
export CUDA_VISIBLE_DEVICES=4

python train.py \
    --cfg-path train_configs/filter_only_frozen_llm_false.yaml \
    --options run.device=cuda run.world_size=1
