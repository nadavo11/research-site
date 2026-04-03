# Frozen SAM GlaS Mask Head

## Overview

This document explains the dense-supervised frozen-SAM GlaS method implemented in this repository and gives exact replication instructions for the saved runs.

Scientific role:

- task: binary gland foreground segmentation on GlaS
- backbone: SAM 3, frozen for the full run
- trainable component: a tiny convolutional mask head over frozen SAM `backbone_fpn` features
- supervision: dense gland-vs-background masks from the repo-native GlaS adapter
- primary metrics: `direct_foreground_iou`, `direct_foreground_dice`
- auxiliary metrics: `eval_miou`, `eval_ari`

This is a supervision-matched frozen-feature baseline. It is not AutoSAM, does not learn prompts, and does not unfreeze the SAM backbone.

Primary code paths:

- model: [`src/rwtd_sam3/models/sam3_frozen_multiscale_mask_head.py`](../../src/rwtd_sam3/models/sam3_frozen_multiscale_mask_head.py)
- train/eval runner: [`src/rwtd_sam3/eval/glas_frozen_feature_mask_head.py`](../../src/rwtd_sam3/eval/glas_frozen_feature_mask_head.py)
- dataset adapter: [`src/rwtd_sam3/data/glas_binary.py`](../../src/rwtd_sam3/data/glas_binary.py)
- CLI entrypoints: [`src/rwtd_sam3/cli.py`](../../src/rwtd_sam3/cli.py)

Related result summaries:

- phase 1 baseline: [`outputs/glas_supervised_frozen_sam_phase1/README.md`](../glas_supervised_frozen_sam_phase1/README.md)
- phase 2 ablations: [`outputs/glas_supervised_frozen_sam_phase2/README.md`](../glas_supervised_frozen_sam_phase2/README.md)

## Method

### Problem setup

The method answers one narrow question:

> If SAM stays frozen, can a very small dense supervised readout already segment glands well on GlaS?

The implementation is intentionally minimal so that failures remain interpretable:

- no prompt generator
- no prompt encoder changes
- no transformer decoder
- no attention blocks
- no U-Net-scale decoder
- no SAM fine-tuning

### Dataset contract

The method uses the existing repo-native GlaS binary loader in [`src/rwtd_sam3/data/glas_binary.py`](../../src/rwtd_sam3/data/glas_binary.py).

Accepted local layout:

```text
datasets/GlaS/
  Grade.csv                  # optional
  train_1.bmp
  train_1_anno.bmp
  ...
  testA_1.bmp
  testA_1_anno.bmp
  testB_1.bmp
  testB_1_anno.bmp
```

Resolved split semantics:

- `train`: files named `train_*`
- `test`: files named `testA_*` and `testB_*`
- `all`: `train_*`, `testA_*`, and `testB_*`

Mask polarity is fixed by code:

- pixels with mask value `> 0` are gland foreground
- pixels with mask value `== 0` are background
- gland foreground becomes `texture_a_mask`
- background becomes `texture_b_mask`

Hard loader checks:

- dataset root must resolve to a directory containing matched image and `*_anno.bmp` pairs
- image and mask sizes must match exactly
- gland foreground must be non-empty
- background must be non-empty

### Frozen feature extraction path

Frozen features come from `Sam3CoarseVsFineScaleFeatureExtractor` in [`src/rwtd_sam3/models/sam3_coarse_vs_fine_scale_probe.py`](../../src/rwtd_sam3/models/sam3_coarse_vs_fine_scale_probe.py).

The runner calls:

- `extract_sam_pyramid(sample.image)`

That extractor returns:

- `image_size = (image.height, image.width)`
- `pyramid = {level_name: np.ndarray[C, H_l, W_l]}`

On the recorded GlaS runs, the discovered main SAM levels were:

- `fpn_2`
- `fpn_1`
- `fpn_0`

Observed reference shapes from saved runs:

- `fpn_2`: `(256, 72, 72)`
- `fpn_1`: `(256, 144, 144)`
- `fpn_0`: `(256, 288, 288)`

The training/eval runner stores cached feature arrays as `float16` NumPy arrays on CPU, then materializes one sample at a time onto the torch device as `float32` tensors with shape `[1, C, H_l, W_l]`.

The feature source is fixed by settings:

- `feature_source = backbone_fpn`

### Scale-selection variants

The dense mask-head family supports these level subsets:

| Variant | Selected levels |
| --- | --- |
| `all_scales` | all discovered levels, currently `fpn_2`, `fpn_1`, `fpn_0` |
| `mid_plus_fine` | `fpn_1`, `fpn_0` |
| `coarse_plus_next_finer` | first two discovered levels, currently `fpn_2`, `fpn_1` |
| `fpn_2_only` | `fpn_2` |
| `fpn_1_only` | `fpn_1` |
| `fpn_0_only` | `fpn_0` |

This document covers only the dense mask-head variants above. The same CLI family also exposes smaller probe variants such as `linear_fpn2` and `tiny_multiscale`, but those are separate model families and not the main supervised mask-head method.

### Head architecture

The dense head is implemented by `FrozenSamMultiscaleMaskHead`.

For selected levels `L = {l_1, ..., l_k}` with raw tensors `x_l in R^[1,C_l,H_l,W_l]`:

1. Project each level with a learned `1x1` convolution:
   - `p_l = Conv1x1_l(x_l)`
   - output shape: `[1, projection_dim, H_l, W_l]`
2. Resolve the finest selected grid:
   - the selected level with the largest spatial resolution
3. Bilinearly upsample every projected map to the finest selected grid
4. Concatenate projected maps channel-wise:
   - `fused = cat([p_l], dim=1)`
   - shape: `[1, k * projection_dim, H_f, W_f]`
5. Decode with two conv blocks:
   - `Conv3x3(k * projection_dim -> decoder_dim) -> GroupNorm(8) -> GELU`
   - `Conv3x3(decoder_dim -> decoder_dim) -> GroupNorm(8) -> GELU`
6. Bilinearly upsample the decoded tensor to image resolution `(H_img, W_img)`
7. Apply a final `1x1` classifier:
   - output logits shape: `[1, 1, H_img, W_img]`

Invariant implementation choices:

- normalization is `GroupNorm`, not `BatchNorm`
- fusion is concatenation, not summation
- interpolation mode is bilinear with `align_corners=False`
- output is one foreground logit channel
- batch size is effectively `1` throughout this path

Observed real-sample tensor path from the saved smoke checks:

- image: `522 x 775`
- raw `fpn_2`: `[1, 256, 72, 72]`
- raw `fpn_1`: `[1, 256, 144, 144]`
- raw `fpn_0`: `[1, 256, 288, 288]`
- projected all-scales tensors: `[1, 64, 72, 72]`, `[1, 64, 144, 144]`, `[1, 64, 288, 288]`
- fused all-scales tensor: `[1, 192, 288, 288]`
- final logits: `[1, 1, 522, 775]`

### Training objective

Training uses `AdamW` on the head parameters only. All SAM parameters remain frozen.

Loss is defined in `compute_bce_dice_loss()`:

- `BCE = binary_cross_entropy_with_logits(logits, target)`
- `prob = sigmoid(logits)`
- `DiceScore = (2 * sum(prob * target) + smooth) / (sum(prob) + sum(target) + smooth)`
- `DiceLoss = 1 - DiceScore`
- `TotalLoss = bce_weight * BCE + dice_weight * DiceLoss`

Current dense-head defaults:

| Setting | Default |
| --- | --- |
| `projection_dim` | `64` |
| `decoder_dim` | `64` |
| `group_norm_groups` | `8` |
| `learning_rate` | `1e-3` |
| `weight_decay` | `1e-4` |
| `num_epochs` | `20` |
| `bce_weight` | `1.0` |
| `dice_weight` | `1.0` |
| `foreground_threshold` | `0.5` |
| `smooth` in Dice | `1.0` |
| `seed` | `0` |

Optimization behavior:

- one sample is processed at a time
- the training order is shuffled once per epoch with a NumPy RNG seeded from `--seed`
- the optimizer only receives parameters with `requires_grad=True`
- no validation split is created by this path

### Evaluation contract

At eval time the runner:

1. computes `sigmoid(logits)`
2. thresholds probabilities at `--foreground-threshold`
3. interprets the thresholded mask as gland foreground
4. interprets the complement as background

Primary reporting metrics:

| Metric | Meaning | Range | Better |
| --- | --- | --- | --- |
| `direct_foreground_iou` | Direct IoU between predicted foreground and gland mask. No label swapping. | `[0, 1]` | Higher |
| `direct_foreground_dice` | Direct Dice / F1 between predicted foreground and gland mask. | `[0, 1]` | Higher |

Auxiliary repo-native diagnostics from the same prediction route:

| Metric | Meaning | Range | Better |
| --- | --- | --- | --- |
| `eval_miou` | Partition-invariant mean IoU under the repo canonical binary evaluation contract. | `[0, 1]` | Higher |
| `eval_ari` | Partition-invariant ARI under the same contract. | `[-1, 1]` | Higher |

Important metric note:

- for AutoSAM-style comparison, use `direct_foreground_iou` and `direct_foreground_dice`
- keep `eval_miou` and `eval_ari` as auxiliary diagnostics, not the headline claim

## CLI Reference

### `train-glas-frozen-sam-mask-head`

| Name | Default | Type | Description |
| --- | --- | --- | --- |
| `--dataset-root` | required | `path` | Local GlaS root or an archive root that contains a flat Warwick-style export. |
| `--model-id` | `facebook/sam3` | `str` | SAM model identifier for frozen feature extraction. |
| `--device` | `auto` | `auto\|cpu\|cuda` | Device for frozen feature extraction and head training/eval. |
| `--hf-token` | unset | `str` | Optional Hugging Face token. If omitted, the code tries `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`. |
| `--official-checkpoint-path` | unset | `path` | Optional local Meta checkpoint passed into the SAM extractor backend. |
| `--variant` | `all_scales` | dense-head variant choice | Scale subset used by the dense mask head. |
| `--projection-dim` | `64` | `int` | Output width of each per-level `1x1` projection. |
| `--decoder-dim` | `64` | `int` | Decoder width. Must be divisible by `8`. |
| `--train-split` | `train` | `train\|test\|all` | GlaS split used for optimization. |
| `--eval-split` | `test` | `train\|test\|all` | GlaS split evaluated immediately after training. |
| `--train-limit` | unset | `int` | Optional cap on loaded training samples. |
| `--eval-limit` | unset | `int` | Optional cap on loaded evaluation samples. |
| `--learning-rate` | `1e-3` | `float` | AdamW learning rate. Must be positive. |
| `--weight-decay` | `1e-4` | `float` | AdamW weight decay. Must be non-negative. |
| `--num-epochs` | `20` | `int` | Number of epochs. Must be at least `1`. |
| `--bce-weight` | `1.0` | `float` | Weight on `BCEWithLogitsLoss`. Must be non-negative. |
| `--dice-weight` | `1.0` | `float` | Weight on Dice loss. Must be non-negative. |
| `--foreground-threshold` | `0.5` | `float` | Probability threshold used during the post-training evaluation stage. Must satisfy `0 <= threshold <= 1`. |
| `--output-dir` | auto | `path` | Default pattern: `outputs/glas_binary/frozen_sam_mask_head/train_<variant>_<eval_split>/`. |
| `--save-visuals` / `--no-save-visuals` | save | flag pair | Save or skip post-training evaluation panels. |

Inherited global flags:

| Name | Default | Type | Description |
| --- | --- | --- | --- |
| `--log-level` | `INFO` | `str` | Python logging level. |
| `--seed` | `0` | `int` | Reproducibility seed used by the trainer. |

### `eval-glas-frozen-sam-mask-head`

| Name | Default | Type | Description |
| --- | --- | --- | --- |
| `--dataset-root` | required | `path` | Local GlaS root or archive root. |
| `--split` | `test` | `train\|test\|all` | GlaS split used for evaluation. |
| `--model-id` | `facebook/sam3` | `str` | SAM model identifier for frozen feature extraction. |
| `--device` | `auto` | `auto\|cpu\|cuda` | Device for frozen feature extraction and head inference. |
| `--hf-token` | unset | `str` | Optional Hugging Face token. |
| `--official-checkpoint-path` | unset | `path` | Optional local Meta checkpoint used by the extractor backend. |
| `--checkpoint-path` | required | `path` | Checkpoint produced by `train-glas-frozen-sam-mask-head`. |
| `--variant` | `all_scales` | dense-head variant choice | Placeholder only. The checkpoint variant overrides this at runtime. |
| `--limit` | unset | `int` | Optional cap on evaluated rows. |
| `--dataset-partition` | unset | `K/N` | Optional deterministic evaluation partition. Partitioning is resolved before `--limit`. |
| `--foreground-threshold` | `0.5` | `float` | Eval threshold. If the checkpoint stores a threshold and the CLI remains at `0.5`, the checkpoint value is reused. |
| `--output-dir` | auto | `path` | Default pattern: `outputs/glas_binary/frozen_sam_mask_head/eval_<checkpoint_variant>_<split>/`. |
| `--save-visuals` / `--no-save-visuals` | save | flag pair | Save or skip visualization panels. |

## Outputs

Each training run writes a standard artifact directory:

```text
outputs/glas_binary/frozen_sam_mask_head/<run_name>/
  checkpoint.pt
  config.json
  experiment_terms.md
  per_sample_metrics.csv
  per_sample_metrics.jsonl
  summary.csv
  summary.json
  summary.md
  train_history.csv
  train_set_summary.json
  visuals_manifest.jsonl
  masks/
    <sample_index>.npz
  sample_rows/
    <sample_index>.json
  visuals/                   # only when visuals are enabled
    <sample_index>.png
```

Important artifact semantics:

- `config.json` stores the exact command, selected levels, settings, seed, and package versions
- `summary.json` stores the aggregated metrics and should be the canonical file for result tables
- `train_history.csv` stores `epoch`, `mean_train_loss`, `mean_bce_loss`, `mean_dice_loss`, and `mean_predicted_positive_fraction`
- `masks/<index>.npz` stores `foreground_prediction`, `background_prediction`, `target_foreground`, `target_background`, and `foreground_probability`
- `visuals_manifest.jsonl` is always written; if visuals are disabled it stays empty and `visuals/` is omitted

## Replication

### Environment

The recorded runs in this workspace used:

- Python interpreter: `./.venv/bin/python`
- shell prefix: `PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
- seed: `0`
- device: `cuda`
- model id: `facebook/sam3`

Recorded package versions from the best current run:

| Package | Version |
| --- | --- |
| `Pillow` | `12.1.1` |
| `datasets` | `4.7.0` |
| `numpy` | `1.26.4` |
| `torch` | `2.10.0` |
| `transformers` | `5.3.0` |
| `wandb` | `null` |

### Quick replication ladder

#### 1. Smoke test the full path

```bash
PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./.venv/bin/python main.py train-glas-frozen-sam-mask-head \
  --dataset-root datasets/GlaS \
  --device cuda \
  --variant all_scales \
  --train-limit 1 \
  --eval-limit 1 \
  --num-epochs 1 \
  --output-dir outputs/glas_binary/frozen_sam_mask_head/smoke_all_scales_train_to_test_1x1 \
  --no-save-visuals
```

Expected output directory:

- `outputs/glas_binary/frozen_sam_mask_head/smoke_all_scales_train_to_test_1x1`

#### 2. Run the tiny overfit sanity check

```bash
PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./.venv/bin/python main.py train-glas-frozen-sam-mask-head \
  --dataset-root datasets/GlaS \
  --device cuda \
  --variant all_scales \
  --train-limit 2 \
  --eval-split train \
  --eval-limit 2 \
  --num-epochs 20 \
  --output-dir outputs/glas_binary/frozen_sam_mask_head/overfit_all_scales_train2_evaltrain2_e20 \
  --no-save-visuals
```

Expected behavior:

- loss should drop sharply
- predictions should not remain all-zero or all-one
- the output directory should contain `checkpoint.pt`, `train_history.csv`, and summary files

#### 3. Reproduce the first full 20-epoch all-scales baseline

```bash
PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./.venv/bin/python main.py train-glas-frozen-sam-mask-head \
  --dataset-root datasets/GlaS \
  --device cuda \
  --variant all_scales \
  --num-epochs 20 \
  --output-dir outputs/glas_binary/frozen_sam_mask_head/train_all_scales_test_e20_streamed \
  --no-save-visuals
```

Recorded metrics from the saved run:

- `direct_foreground_iou = 0.7962518203`
- `direct_foreground_dice = 0.8822305733`
- `eval_miou = 0.7798676951`
- `eval_ari = 0.6001156537`
- `materialization_mode = streamed_reextract`

#### 4. Reproduce the 40-epoch all-scales reference used in Phase 2

```bash
PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./.venv/bin/python main.py train-glas-frozen-sam-mask-head \
  --dataset-root datasets/GlaS \
  --device cuda \
  --variant all_scales \
  --num-epochs 40 \
  --output-dir outputs/glas_binary/frozen_sam_mask_head/train_all_scales_test_e20_visuals
```

Recorded metrics from the saved run:

- `direct_foreground_iou = 0.8031169369`
- `direct_foreground_dice = 0.8859184015`
- `eval_miou = 0.7932073208`
- `eval_ari = 0.6245927585`

Important caveat:

- the directory suffix says `e20`, but the saved `config.json` and `summary.json` both record `num_epochs = 40`
- for audit and replication, trust `config.json` and `summary.json`, not the legacy folder suffix

#### 5. Reproduce the current best coarse-only width-128 run

```bash
PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./.venv/bin/python main.py train-glas-frozen-sam-mask-head \
  --dataset-root datasets/GlaS \
  --device cuda \
  --variant fpn_2_only \
  --projection-dim 128 \
  --decoder-dim 128 \
  --num-epochs 40 \
  --output-dir outputs/glas_binary/frozen_sam_mask_head/train_fpn_2_only_d128_test_e40
```

Recorded metrics from the saved run:

- `direct_foreground_iou = 0.8243115542`
- `direct_foreground_dice = 0.8994357114`
- `eval_miou = 0.8135546960`
- `eval_ari = 0.6596848701`
- `trainable_parameter_count = 328705`

#### 6. Re-evaluate a checkpoint and save visualization panels

```bash
PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./.venv/bin/python main.py eval-glas-frozen-sam-mask-head \
  --dataset-root datasets/GlaS \
  --device cuda \
  --checkpoint-path outputs/glas_binary/frozen_sam_mask_head/train_fpn_2_only_d128_test_e40/checkpoint.pt \
  --split test \
  --output-dir outputs/glas_binary/frozen_sam_mask_head/eval_fpn_2_only_d128_test_e40_visuals
```

### Phase 2 exact ablation matrix

| Run | Command tail | Output dir |
| --- | --- | --- |
| `fine_only` | `--variant fpn_0_only --num-epochs 40` | `outputs/glas_binary/frozen_sam_mask_head/train_fpn_0_only_test_e40` |
| `mid_plus_fine` | `--variant mid_plus_fine --num-epochs 40` | `outputs/glas_binary/frozen_sam_mask_head/train_mid_plus_fine_test_e40` |
| `coarse_only` | `--variant fpn_2_only --num-epochs 40` | `outputs/glas_binary/frozen_sam_mask_head/train_fpn_2_only_test_e40` |
| `coarse_only_d32` | `--variant fpn_2_only --projection-dim 32 --decoder-dim 32 --num-epochs 40` | `outputs/glas_binary/frozen_sam_mask_head/train_fpn_2_only_d32_test_e40` |
| `coarse_only_d128` | `--variant fpn_2_only --projection-dim 128 --decoder-dim 128 --num-epochs 40` | `outputs/glas_binary/frozen_sam_mask_head/train_fpn_2_only_d128_test_e40` |

All commands above use the shared prefix:

```bash
PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./.venv/bin/python main.py train-glas-frozen-sam-mask-head \
  --dataset-root datasets/GlaS \
  --device cuda
```

### Replication checks

After a run finishes, verify these files first:

- `config.json`
- `summary.json`
- `train_history.csv`

Minimum fields to check in `summary.json`:

- `variant`
- `selected_level_names`
- `projection_dim`
- `decoder_dim`
- `num_epochs`
- `trainable_parameter_count`
- `mean_metrics.direct_foreground_iou`
- `mean_metrics.direct_foreground_dice`
- `mean_metrics.eval_miou`
- `mean_metrics.eval_ari`
- `materialization_mode`

## Current Recorded Results

| Variant | Levels | Dim | Params | Epochs | fg_iou | dice | eval_miou | eval_ari | Output dir |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `all_scales` | `fpn_2,fpn_1,fpn_0` | 64 | 197249 | 20 | 0.796252 | 0.882231 | 0.779868 | 0.600116 | `outputs/glas_binary/frozen_sam_mask_head/train_all_scales_test_e20_streamed` |
| `all_scales` | `fpn_2,fpn_1,fpn_0` | 64 | 197249 | 40 | 0.803117 | 0.885918 | 0.793207 | 0.624593 | `outputs/glas_binary/frozen_sam_mask_head/train_all_scales_test_e20_visuals` |
| `fpn_0_only` | `fpn_0` | 64 | 90625 | 40 | 0.729788 | 0.838083 | 0.715166 | 0.479865 | `outputs/glas_binary/frozen_sam_mask_head/train_fpn_0_only_test_e40` |
| `mid_plus_fine` | `fpn_1,fpn_0` | 64 | 143937 | 40 | 0.786244 | 0.876425 | 0.771574 | 0.582750 | `outputs/glas_binary/frozen_sam_mask_head/train_mid_plus_fine_test_e40` |
| `fpn_2_only` | `fpn_2` | 64 | 90625 | 40 | 0.813221 | 0.892288 | 0.803947 | 0.644130 | `outputs/glas_binary/frozen_sam_mask_head/train_fpn_2_only_test_e40` |
| `fpn_2_only` | `fpn_2` | 32 | 26881 | 40 | 0.808445 | 0.889356 | 0.798616 | 0.633705 | `outputs/glas_binary/frozen_sam_mask_head/train_fpn_2_only_d32_test_e40` |
| `fpn_2_only` | `fpn_2` | 128 | 328705 | 40 | 0.824312 | 0.899436 | 0.813555 | 0.659685 | `outputs/glas_binary/frozen_sam_mask_head/train_fpn_2_only_d128_test_e40` |

Current takeaways from the saved results:

- fine-only is not enough in this model family
- the coarsest SAM level currently carries the strongest supervised gland signal
- widening the head helps, but scale choice matters more than width
- direct foreground metrics are already available, so metric routing is not the main unresolved comparison issue

## Failure Modes And Caveats

### Hard failures from code

- missing GlaS root or unmatched `image + *_anno.bmp` pairs
- empty resolved split after partitioning or `--limit`
- requested `cuda` when CUDA is unavailable
- selected SAM pyramid level not discovered in the extractor output
- inconsistent channel counts across samples
- target mask shape different from predicted logit shape
- non-finite projected features, logits, or losses
- `--decoder-dim` not divisible by `8`

### Important caveats

- there is no validation split in this training path
- there is no repo-native `224x224` resize knob in this method today
- eval uses the checkpoint variant even if `--variant` is passed differently
- large runs may switch automatically from `in_memory` caching to `streamed_reextract` when the estimated feature cache exceeds the internal `5 GiB` budget
- the recorded directory name `train_all_scales_test_e20_visuals` is legacy naming; the saved metadata records `40` epochs

### What this method does not test

- learned prompt generators
- prompt encoder training
- SAM backbone fine-tuning
- heavy decoder architectures
- protocol-matched `224x224` resizing

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Could not resolve a GlaS dataset root` | `--dataset-root` does not point to a valid local export | Point to the flat Warwick-style directory or an archive root that contains it. |
| `No matched GlaS image/mask pairs were found` | image and `*_anno.bmp` files are missing or misnamed | Fix the local dataset layout. |
| `CUDA was requested but is not available` | `--device cuda` on a host without CUDA | Use `--device cpu` or fix CUDA. |
| `decoder_dim must be divisible by 8` | incompatible width for fixed GroupNorm setting | Use a decoder width divisible by `8`, such as `32`, `64`, or `128`. |
| `Target mask shape does not match the predicted logit map` | image-size assumption broke somewhere in the path | Do not insert ad hoc resizing; fix the upstream preprocessing or feature extraction path. |
| strict `224x224` replication is needed | current trainer has no resize flag | Add a single explicit image-and-mask resize knob as a separate controlled protocol change, not inside an existing result claim. |
