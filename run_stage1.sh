#!/bin/bash
# Stage 1: Joint Pre-training on MERCaptionPlus + Human
# Key fixes: unfrozen LLM (LoRA), multiframe mode, wheel label filtering, hierarchical supervision

cd /work/2025/liusiyu/track2_rebuild

python train.py \
    --cfg-path train_configs/stage1_joint_pretrain.yaml \
    --options run.device=cuda run.world_size=1
