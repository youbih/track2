#!/bin/bash
# Method A Stage 2: Fine-tune LLM on Human data only
# Load Stage 1 best checkpoint, freeze CMEF, unfreeze LLM
# IMPORTANT: Update CKPT_PATH to the actual Stage 1 best checkpoint before running

source /work/2025/liusiyu/anaconda3/etc/profile.d/conda.sh
conda activate vllm3

cd /work/2025/liusiyu/track2
export CUDA_VISIBLE_DEVICES=2

# Set this to the actual Stage 1 best checkpoint path after Stage 1 finishes
CKPT_PATH="output/method_a_stage1/method_a_stage1_*/checkpoint_best.pth"

python train.py \
    --cfg-path train_configs/method_a_stage2.yaml \
    --options run.device=cuda run.world_size=1 run.resume_ckpt_path="${CKPT_PATH}" \
    2>&1 | tee output/method_a_stage2_terminal.log
