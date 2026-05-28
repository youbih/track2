# Method A: Two-Stage Experiment Plan
## Sample Quality Gating in CMEF for Multimodal Emotion Recognition

---

## 1. Motivation

### Problem
- MERCaptionPlus (machine-annotated) has large quantity (~27K) but noisy labels
- Human (human-annotated) has high quality but small quantity (~1.5K)
- Official MER2026 test set uses human annotation style

### Solution
Two-stage training with sample quality gating:
- **Stage 1**: Learn robust fusion representations on mixed data with quality-aware weighting
- **Stage 2**: Fine-tune LLM generation on human-only data

---

## 2. Innovation: Sample Quality Gating

### Location
`my_affectgpt/models/fusion_modules.py` - `CrossModalEmotionFusion`

### Design
```python
self.quality_estimator = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 2),
)
```

### Forward Logic
```python
audio_quality = torch.sigmoid(self.quality_estimator(
    torch.cat([audio.mean(dim=1), video.mean(dim=1)], dim=-1)
))  # [b, 2]
sample_quality = audio_quality.mean(dim=-1, keepdim=True)
quality_scale = sample_quality.unsqueeze(1)
fused = fused * quality_scale
```

### Expected Behavior
- High-quality samples (clear emotion signals) -> quality_scale close to 1.0
- Low-quality samples (noisy or ambiguous) -> quality_scale close to 0.0
- Human samples should get higher quality scores than machine-captioned samples

---

## 3. Three-Way Validation Strategy

### Rationale
Previous experiments showed that adding human data hurts val performance on MERCaptionPlus val (p1b: 0.5753 vs baseline: 0.6473). However, the official test set is human-annotated. We need separate validation sets to monitor performance on different distributions.

### Validation Sets
| Split | Data Source | Purpose |
|-------|------------|---------|
| `machine_val` | MERCaptionPlus only | Monitor machine annotation distribution |
| `human_val` | Human only | Monitor human annotation distribution (target) |
| `val` | Combined (machine + human) | Overall performance |

### Implementation
Modified `image_text_pair_builder.py` to support `split_prefix`:
```yaml
mercaptionplus:
  split_prefix: "machine_"  # Creates machine_val
human:
  split_prefix: "human_"    # Creates human_val
```

Both datasets also create standard `val` split which is automatically combined.

---

## 4. Stage 1: Fusion Learning

### Goal
Learn good audio-visual fusion representations using mixed data (MERCaptionPlus + Human).

### Data
| Dataset | Size | Annotation | ratio | val_ratio |
|---------|------|------------|-------|-----------|
| MERCaptionPlus | ~27K | Machine | 1.0 | 0.1 (~2.7K val) |
| Human | ~1.5K | Human | 1.0 | 0.15 (~230 val) |

**Key change**: ratio=1.0 (p2b proved full data is better than 0.7)

### Model Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `frozen_llm` | True | Don't train LLM yet |
| `frozen_cmef` | False | Train quality gating module |
| `multi_fusion_type` | cmef | Core innovation |
| `cmef_num_layers` | 2 | p2a validated 2 layers suffice |
| `contrastive_weight` | 0.1 | Learn emotion prototypes |
| `lora_r` | 8 | Lower rank for stage 1 |
| `init_lr` | 5e-5 | Standard pretraining LR |
| `max_epoch` | 12 | Full convergence |

### What Gets Trained
- CMEF module (with quality_estimator)
- Multi Q-Former LLaMA projection
- Audio/Video Q-Former projections
- LoRA adapters (rank 8)

### What Gets Frozen
- LLM (Qwen2.5-7B)
- Audio/Video encoders (HuBERT, CLIP)

### Key Metrics to Monitor
1. `val_wheel_f1` - Combined validation
2. `machine_val_wheel_f1` - Machine annotation distribution
3. `human_val_wheel_f1` - Human annotation distribution (most important)
4. Quality score distribution (human vs machine-caption)

### Checkpoint
- Save best checkpoint based on best across all valid_splits
- Path: `output/method_a_stage1/*/checkpoint_best.pth`

---

## 5. Stage 2: LLM Generation Alignment

### Goal
Fine-tune LLM to generate human-style emotion labels.

### Data
| Dataset | Size | Annotation | Usage |
|---------|------|------------|-------|
| Human | ~1.5K | Human | Only data source |

### Model Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `frozen_llm` | False | Fine-tune LLM |
| `frozen_cmef` | True | Keep Stage 1 fusion frozen |
| `multi_fusion_type` | cmef | Reuse learned fusion |
| `contrastive_weight` | 0.0 | No contrastive, focus on generation |
| `lora_r` | 16 | Higher rank for LLM adaptation |
| `init_lr` | 1e-5 | Lower LR for fine-tuning |
| `max_epoch` | 8 | Prevent overfitting on small data |

### What Gets Trained
- LLM with LoRA (rank 16, expanded from rank 8)
- Multi Q-Former LLaMA projection

### What Gets Frozen
- CMEF module (quality gating preserved from Stage 1)
- Audio/Video encoders
- Audio/Video Q-Formers

### Why Keep CMEF Frozen?
- Stage 1 already learned good fusion representations
- Freezing CMEF acts as a "feature extractor" for Stage 2
- Reduces trainable parameters, preventing overfitting on 1.5K data

### Checkpoint Loading
```bash
# Load Stage 1 best checkpoint
CKPT_PATH="output/method_a_stage1/method_a_stage1_*/checkpoint_best.pth"
```
Note: `strict=False` loading allows LoRA rank change (8 -> 16).

---

## 6. Two-Stage Comparison

| Dimension | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Data | MERCaptionPlus + Human | Human only |
| Goal | Learn fusion | Align generation |
| LLM | Frozen | Unfrozen (LoRA) |
| CMEF | Trained | Frozen |
| LoRA rank | 8 | 16 |
| Contrastive | 0.1 | 0.0 |
| LR | 5e-5 | 1e-5 |
| Epochs | 12 | 8 |
| Validation | machine_val, human_val, val | human_val, val |

---

## 7. Code Changes

### 1. `my_affectgpt/models/fusion_modules.py`
- Added `quality_estimator` in `CrossModalEmotionFusion.__init__`
- Added quality gating in `forward`

### 2. `my_affectgpt/models/affectgpt.py`
- Added `frozen_cmef` parameter
- Added freezing logic for CMEF module

### 3. `my_affectgpt/datasets/builders/image_text_pair_builder.py`
- Added `split_prefix` support
- Creates prefixed val splits (e.g., `machine_val`, `human_val`)

### Config Files
- `train_configs/method_a_stage1.yaml` - Stage 1 with three validation sets
- `train_configs/method_a_stage2.yaml` - Stage 2 with human validation

### Run Scripts
- `run_method_a_stage1.sh` - Stage 1 runner
- `run_method_a_stage2.sh` - Stage 2 runner

---

## 8. Execution Order

```bash
# Step 1: Run Stage 1
bash run_method_a_stage1.sh

# Step 2: Monitor three validation sets
# Check: output/method_a_stage1/*/log.txt

# Step 3: Identify best Stage 1 checkpoint
# Best is saved automatically as checkpoint_best.pth

# Step 4: Update Stage 2 script with checkpoint path
vim run_method_a_stage2.sh  # Update CKPT_PATH

# Step 5: Run Stage 2
bash run_method_a_stage2.sh
```

---

## 9. Expected Results

### Stage 1
- `human_val_wheel_f1` should be the primary metric (targets human distribution)
- `machine_val_wheel_f1` may be lower than baseline due to human data mixing
- Quality scores should distinguish human vs machine-caption
- Loss should converge smoothly

### Stage 2
- Should improve `human_val_wheel_f1` over Stage 1
- LLM output should match human annotation style better
- Risk: Overfitting on 1.5K data (monitor early stopping)

---

## 10. Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| Stage 1 OOM | accum_grad_iters=16, batch_size=1 |
| Stage 2 overfitting | early_stop_patience=6, max_epoch=8 |
| Quality gating collapses | Monitor quality score distribution |
| Human data too small | Stage 1 already exposes model to human |
| LoRA rank mismatch | strict=False loading handles this |
| Multiple val sets confusion | human_val is primary for checkpoint |

---

## 11. Files

```
track2_quality_gate/
├── my_affectgpt/
│   ├── models/
│   │   ├── fusion_modules.py      # Modified: quality_estimator
│   │   └── affectgpt.py           # Modified: frozen_cmef
│   └── datasets/builders/
│       └── image_text_pair_builder.py  # Modified: split_prefix
├── train_configs/
│   ├── method_a_stage1.yaml       # Stage 1: three val sets
│   └── method_a_stage2.yaml       # Stage 2: human only
├── run_method_a_stage1.sh         # Stage 1 script
├── run_method_a_stage2.sh         # Stage 2 script
└── EXPERIMENT_PLAN.md             # This document
```

---

## 12. Next Steps

1. [ ] Run Stage 1 experiment
2. [ ] Monitor three validation metrics (machine_val, human_val, val)
3. [ ] Analyze quality score distribution
4. [ ] Identify best Stage 1 checkpoint
5. [ ] Update Stage 2 script with checkpoint path
6. [ ] Run Stage 2 experiment
7. [ ] Compare Stage 2 human_val with baselines
