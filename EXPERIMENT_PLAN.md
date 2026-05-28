# Method A: Two-Stage Training with Sample Quality Gating

## Current Status: ALL CODE CHANGES IMPLEMENTED

All modifications described in this plan are already in place in `/work/2025/liusiyu/track2`.
The current running experiments (`method_a_stage1`, `method_a_stage1_qr`) are using this code.

---

## 1. Motivation

### Problem
- `MERCaptionPlus` (machine-annotated): ~31K samples, large quantity but noisy labels
- `Human` (human-annotated): ~1.5K samples, high quality but small quantity
- Official MER2026 test set uses human annotation style

### Solution
Two-stage training with sample quality gating in CMEF:
- **Stage 1**: Learn robust fusion representations on mixed data with quality-aware weighting
- **Stage 2**: Fine-tune LLM generation on human-only data

---

## 2. Code Changes (Already Implemented)

### 2.1 `my_affectgpt/models/fusion_modules.py`

**Change**: Added `quality_estimator` in `CrossModalEmotionFusion.__init__` (line 481-487)

```python
# Sample quality estimator: per-sample audio/video quality gating
self.quality_estimator = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 2),
)
```

**Change**: Modified `CrossModalEmotionFusion.forward()` (line 583-595)
- Returns `(fused, audio_proj, video_proj, quality_reg)` instead of just `fused`
- Quality gating suppresses low-quality samples:
  - High-quality -> `quality_scale` near 1.0
  - Low-quality -> `quality_scale` near 0.0
- Added `quality_reg` to prevent collapse to 1.0

```python
audio_quality = torch.sigmoid(self.quality_estimator(
    torch.cat([audio.mean(dim=1), video.mean(dim=1)], dim=-1)
))  # [b, 2]
sample_quality = audio_quality.mean(dim=-1, keepdim=True)  # [b, 1]
quality_scale = sample_quality.unsqueeze(1)  # [b, 1, 1]
fused = fused * quality_scale

quality_reg = ((quality_scale - 0.5) ** 2).mean()
return fused, audio_proj, video_proj, quality_reg
```

### 2.2 `my_affectgpt/models/affectgpt.py`

**Changes**:
1. Added `frozen_cmef` parameter to `__init__` (line 96)
2. Added CMEF freezing logic (line 331-334):
   ```python
   if frozen_cmef:
       self._set_grad(True, [self.cmef_module], name="CMEF module")
   else:
       self._set_grad(False, [self.cmef_module], name="CMEF module")
   ```
3. Added `quality_reg_weight` parameter (line 119, 359)
4. Modified `encode_multi_merge()` to unpack 4 return values (line 451-465)
5. Added quality_reg loss in `forward()` (line 575-576):
   ```python
   if quality_reg is not None and self.quality_reg_weight > 0:
       loss = loss + self.quality_reg_weight * quality_reg
   ```
6. Added `quality_reg_weight` to `from_config()` (line 630, 671)

### 2.3 `my_affectgpt/datasets/builders/image_text_pair_builder.py`

**Change**: Added `split_prefix` support (line 124-126)

```python
split_prefix = builder.dataset_cfg.get("split_prefix", "")
if split_prefix:
    datasets[f"{split_prefix}val"] = val_dataset
```

This creates prefixed validation splits like `machine_val` and `human_val`.

### 2.4 Config Files

| File | Purpose |
|------|---------|
| `train_configs/method_a_stage1.yaml` | Stage 1: mixed data, frozen LLM, train CMEF |
| `train_configs/method_a_stage2.yaml` | Stage 2: human only, frozen CMEF, train LLM |

### 2.5 Run Scripts

| File | Purpose |
|------|---------|
| `run_method_a_stage1.sh` | Stage 1 runner |
| `run_method_a_stage2.sh` | Stage 2 runner |

**Note**: Run scripts currently reference `track2_quality_gate` path. Update to `track2` before running if needed.

---

## 3. Stage 1 Configuration Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `frozen_llm` | `True` | Don't train LLM yet |
| `frozen_cmef` | `False` | Train quality gating module |
| `multi_fusion_type` | `cmef` | Core innovation |
| `cmef_num_layers` | `2` | p2a validated 2 layers suffice |
| `contrastive_weight` | `0.1` | Learn emotion prototypes |
| `lora_r` | `8` | Lower rank for stage 1 |
| `init_lr` | `5e-5` | Standard pretraining LR |
| `max_epoch` | `12` | Full convergence |
| `ratio` (mercaptionplus) | `1.0` | p2b proved full data is better |
| `ratio` (human) | `1.0` | All human samples |

**Validation sets**: `val`, `machine_val`, `human_val`

**What gets trained**:
- CMEF module (with quality_estimator)
- Multi/Audio/Video projection layers
- LoRA adapters (rank 8)

**What gets frozen**:
- LLM (Qwen2.5-7B)
- Audio/Video encoders (HuBERT, CLIP)

---

## 4. Stage 2 Configuration Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `frozen_llm` | `False` | Fine-tune LLM |
| `frozen_cmef` | `True` | Keep Stage 1 fusion frozen |
| `contrastive_weight` | `0.0` | No contrastive, focus on generation |
| `lora_r` | `16` | Higher rank for LLM adaptation |
| `init_lr` | `1e-5` | Lower LR for fine-tuning |
| `max_epoch` | `8` | Prevent overfitting |

**Data**: Human only (~1.5K)

**Validation sets**: `val`, `human_val`

**Checkpoint loading**:
```bash
CKPT_PATH="output/method_a_stage1/method_a_stage1_*/checkpoint_best.pth"
```

---

## 5. Execution Steps

```bash
# Step 1: Fix run script path if needed
sed -i 's/track2_quality_gate/track2/g' run_method_a_stage1.sh run_method_a_stage2.sh

# Step 2: Run Stage 1
bash run_method_a_stage1.sh

# Step 3: Monitor three validation metrics in output/method_a_stage1/*/log.txt
#   - val_wheel_f1 (combined)
#   - machine_val_wheel_f1 (machine distribution)
#   - human_val_wheel_f1 (human distribution - PRIMARY)

# Step 4: Identify best Stage 1 checkpoint (saved as checkpoint_best.pth)

# Step 5: Update Stage 2 script with actual checkpoint path
vim run_method_a_stage2.sh  # Update CKPT_PATH

# Step 6: Run Stage 2
bash run_method_a_stage2.sh
```

---

## 6. Key Metrics to Monitor

| Stage | Primary Metric | Secondary Metrics |
|-------|---------------|-------------------|
| Stage 1 | `human_val_wheel_f1` | `machine_val_wheel_f1`, `val_wheel_f1`, quality score distribution |
| Stage 2 | `human_val_wheel_f1` | `val_wheel_f1`, overfitting signs |

---

## 7. Current Running Experiments

| Experiment | Status | Config | GPU |
|------------|--------|--------|-----|
| `method_a_stage1` | Running | `method_a_stage1.yaml` | CUDA 2 |
| `method_a_stage1_qr` | Running | Quality reg variant | CUDA 2 |

Both are using the code changes described above.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Stage 1 OOM | `accum_grad_iters=16`, `batch_size=1` |
| Stage 2 overfitting | `early_stop_patience=6`, `max_epoch=8` |
| Quality gating collapses | Monitor quality score distribution |
| LoRA rank mismatch | `strict=False` loading handles 8->16 transition |

---

## 9. Modified Files Checklist

- [x] `my_affectgpt/models/fusion_modules.py` - quality_estimator + gating logic
- [x] `my_affectgpt/models/affectgpt.py` - frozen_cmef + quality_reg loss
- [x] `my_affectgpt/datasets/builders/image_text_pair_builder.py` - split_prefix
- [x] `train_configs/method_a_stage1.yaml`
- [x] `train_configs/method_a_stage2.yaml`
- [x] `run_method_a_stage1.sh`
- [x] `run_method_a_stage2.sh`
