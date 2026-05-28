#!/bin/bash
# Method A Stage 1: CMEF + Quality Gating
# Train fusion module on MERCaptionPlus + Human data
# LLM frozen

source /work/2025/liusiyu/anaconda3/etc/profile.d/conda.sh
conda activate vllm3

cd /work/2025/liusiyu/track2_quality_gate
export CUDA_VISIBLE_DEVICES=2

python train.py \
    --cfg-path train_configs/method_a_stage1.yaml \
    --options run.device=cuda run.world_size=1 2>&1 | tee output/method_a_stage1_terminal.log
