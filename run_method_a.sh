#!/bin/bash
# Method A: Sample Quality Gating in CMEF
# Single GPU

source /work/2025/liusiyu/anaconda3/etc/profile.d/conda.sh
conda activate vllm3

cd /work/2025/liusiyu/track2_quality_gate
export CUDA_VISIBLE_DEVICES=2

python train.py \
    --cfg-path train_configs/method_a_sample_quality_gate.yaml \
    --options run.device=cuda run.world_size=1 2>&1 | tee output/method_a_terminal.log
