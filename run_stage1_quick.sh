#!/bin/bash
# Stage 1 Quick Test
cd /work/2025/liusiyu/gitcode/MERTools/MER2026/MER2026_Track2
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --cfg-path train_configs/mercaptionplus_cmef_tpa_stage1.yaml --options datasets.mercaptionplus.ratio=0.3 run.iters_per_epoch=500 run.max_epoch=3 run.warmup_steps=200
