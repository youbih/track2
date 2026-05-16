#!/bin/bash
# Stage 1 Full Training
cd /work/2025/liusiyu/gitcode/MERTools/MER2026/MER2026_Track2
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --cfg-path train_configs/mercaptionplus_cmef_tpa_stage1.yaml
