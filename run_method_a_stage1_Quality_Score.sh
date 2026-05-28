#!/bin/bash
# Method A Stage 1 with Quality Score Regularization
# Same as baseline stage1 but adds quality_reg_weight=0.05
# This prevents the quality estimator from collapsing to all-1.0 scores
# Expected behavior: quality scores should center around 0.5 instead of 1.0

source /work/2025/liusiyu/anaconda3/etc/profile.d/conda.sh
conda activate vllm3

cd /work/2025/liusiyu/track2_quality_gate
export CUDA_VISIBLE_DEVICES=3

python train.py \
    --cfg-path train_configs/method_a_stage1_quality_reg.yaml \
    --options run.device=cuda run.world_size=1 2>&1 | tee output/method_a_stage1_quality_reg_terminal.log
