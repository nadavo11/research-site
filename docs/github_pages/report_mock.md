# Texture Reports: 2026-03-09_edge_rwtd_diffedge_lorac_bsdsft

Short purpose: Fine-tune DiffusionEdge on RWTD with LoRA preset C to improve edge continuity on real textures.

Status: `reviewed`

## 1. Title Block
- Experiment: `2026-03-09_edge_rwtd_diffedge_lorac_bsdsft`
- One-line purpose: Improve RWTD val ODS F1 with preset C + explicit LR/rank overrides.
- Status tag: `reviewed`

## 2. Run Identity / Provenance
- Date: `2026-03-09`
- Git commit: `abcdef123456`
- Author: `nadav`
- Project/task: `texture-research / edge_detection.fine_tuning`
- Datasets: `RWTD (val)`
- Model/checkpoint: `DiffusionEdge / my60k.pt`

## 3. Configuration
- Preset: `C`
- Overrides: `lr=1e-4`, `rank=16`, `alpha=16`
- Seed: `0`
- Key hyperparameters: `batch_size=8`, `epochs=20`, `img_size=512`
- Repro links:
  - `repro/config.yaml`
  - `repro/command.txt`
  - `repro/env.txt`

## 4. Headline Metrics
| Metric | Value | Baseline | Delta |
|---|---:|---:|---:|
| ODS F1 | 0.742 | 0.721 | +0.021 |
| OIS F1 | 0.761 | 0.744 | +0.017 |
| AP | 0.705 | 0.697 | +0.008 |

Primary metric: `ODS F1`

## 5. Visual Comparison Gallery
Fixed sample list: `rwtd_val_report_v1` (8 samples, fixed order)

Sample template per row:
- `sample_001` | Input | Prediction | GT | Error map (optional)
- caption: `id=sample_001, split=val, preset=C, seed=0`

## 6. QC / Diagnostics
- Preview grid: `assets/qc/preview_grid.png`
- Diagnostics:
  - `assets/qc/edge_density_hist.png`
  - `assets/qc/confidence_distribution.png`
- Failure cases:
  - `assets/qc/fail_case_01.png`
  - `assets/qc/fail_case_02.png`
- Warnings/anomalies:
  - mild over-thinning on high-frequency textures

## 7. Interpretation
- ODS and OIS improved over baseline.
- Failure cases show reduced recall on thin curved boundaries.
- Confidence calibration appears overconfident in repetitive patterns.
- Next: test preset C with stronger augmentation and compare to preset B.

## 8. Artifact Links
- Full outputs: `<external/full_outputs_uri>`
- Checkpoint: `<external/checkpoint_uri>`
- Raw logs: `<external/logs_uri>`
- Dataset shard: `<external/dataset_shard_uri>`
- HF / notebook / related run: `<optional_links>`
