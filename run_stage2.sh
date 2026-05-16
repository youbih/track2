#!/bin/bash
# Stage 2 Full Training (use after Stage 1 completes)
# Update CKPT_PATH with your Stage 1 best checkpoint path
CKPT_PATH="output/mercaptionplus_cmef_tpa_stage1/YOUR_STAGE1_DIR/checkpoints/best_val.pth"

cd /work/2025/liusiyu/gitcode/MERTools/MER2026/MER2026_Track2
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --cfg-path train_configs/mercaptionplus_cmef_tpa.yaml --options model.ckpt=$CKPT_PATH
