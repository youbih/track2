#!/bin/bash
# P0: Debug stage1_joint_pretrain - hierarchical_supervision=false
# GPU 3

cd /work/2025/liusiyu/track2_rebuild
export CUDA_VISIBLE_DEVICES=3

python train.py \
    --cfg-path train_configs/stage1_joint_pretrain_hier_false.yaml \
    --options run.device=cuda run.world_size=1
