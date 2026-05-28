#!/bin/bash
# Stage 2: Fine-tuning on Human dataset
# Load best checkpoint from stage1, lower LR

cd /work/2025/liusiyu/track2_rebuild

# IMPORTANT: Update ckpt path after stage1 completes
CKPT_PATH="output/stage1_joint_pretrain/stage1_joint_pretrain_*/checkpoint_best.pth"

python train.py \
    --cfg-path train_configs/stage2_human_finetune.yaml \
    --options model.ckpt="${CKPT_PATH}" run.device=cuda run.world_size=1
