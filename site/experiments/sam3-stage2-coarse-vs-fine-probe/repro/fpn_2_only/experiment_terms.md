# Coarse-vs-Fine SAM Scale Study: Stage 2 Experiment Terms

This file is the run-local explanation for the Stage-2 coarse-vs-fine SAM study.
It records the exact hypothesis, method, split separation, architecture, evaluation path, outputs, and hard-failure rules for this specific results directory.

## Source References

- `experiments/coarse_vs_fine_sam_scales/experiment_contract.md`
- `src/rwtd_sam3/models/sam3_coarse_vs_fine_scale_probe.py`
- `src/rwtd_sam3/eval/coarse_vs_fine_sam_scales.py`
- `src/rwtd_sam3/eval/experiment_terms.py`

## Experiment Scope

This document describes only the experiment that produced this results directory. It is intentionally experiment-specific and should be sufficient for a reviewer to understand the run without reading unrelated repository docs.

- Command family: `train-coarse-vs-fine-sam-probe`.
- Run kind: `train`.
- Variant: `fpn_2_only`.
- Variant summary: Single-scale Stage-2 control using only the coarsest SAM pyramid level `fpn_2`.
- Dataset: `aviadcohz/RWTD`.
- Model: `facebook/sam3` on device request `auto`.
- Official checkpoint override for SAM feature extraction: `None`.
- Saved visuals: `True`.
- Repository-wide metric contract: `architexture_binary_v1` with primary metrics `eval_miou` and `eval_ari`.

## Hypothesis

- Coarse SAM feature scales should carry most of the useful signal for binary texture-region partitioning.
- A tiny learned multi-scale adapter should therefore place most useful mass on coarse scales or show only weak gains from finer scales.
- Finer scales may still help local boundary placement, but they are not expected to dominate global partition quality on RWTD.

## Scientific Idea

Stage 1 compared non-learned clustering baselines over frozen SAM scales. Stage 2 keeps the same frozen feature source and the same clustering-based evaluation semantics, but adds the smallest interpretable learned component: per-level `1x1` projections and one global gate per selected scale.

The scientific question is not whether a large supervised decoder can solve RWTD. The question is whether a tiny learned probe, when forced to work through clustering on frozen features, still prefers coarse scales.

## Data And Split Separation

- Training split: `train` with limit `None`; loaded `253` sample(s).
- Post-training evaluation split: `test` with limit `None`; loaded `227` sample(s).
- Validation split: none. This command family uses one train split and one post-training evaluation split instead of a separate validation loop.

## Feature Extraction And Alignment

- Frozen SAM feature source: `backbone_fpn` from the main image pyramid (`backbone_fpn`).
- All discovered main pyramid levels for this run: `fpn_2`, `fpn_1`, `fpn_0`.
- Selected levels for the active variant: `fpn_2`.
- Coarsest selected level: `fpn_2` with aligned grid `(72, 72)`.
- Feature alignment policy: resize every selected level to the coarsest selected grid with `bilinear` interpolation before learning or clustering.
- Per-level normalization: `l2` across channels at every spatial location.
- Post-fusion normalization: `l2` on the final embedding map.
- Clustering never sees image-resolution features. Only the final binary label map is upsampled back to image size for permutation-invariant evaluation and visualization.
- Raw level shape `fpn_2`: `(256, 72, 72)`.
- Raw level shape `fpn_1`: `(256, 144, 144)`.
- Raw level shape `fpn_0`: `(256, 288, 288)`.
- Aligned level shape `fpn_2`: `(256, 72, 72)`.

## Probe Architecture

The Stage-2 model is a tiny interpretable probe on top of frozen SAM features. For each selected level `l`, the aligned normalized feature map `F_l` is projected with one learnable `1x1` convolution `P_l: C_l -> d`. Global scalar gate logits `g_l` are converted to scale weights `alpha = softmax(g)`, and the fused representation is `Z = sum_l alpha_l * P_l(F_l)`. A tiny output head maps `Z` from `d` to `d_emb` when needed, and the final per-pixel embedding is L2-normalized before clustering.

The probe stays deliberately low-capacity: no decoder, no UNet, no transformer blocks, no image-resolution feature fusion, and no direct supervised segmentation head. The interpretation target is the learned scale weights themselves.

- Projection dim `d`: `64`.
- Embedding dim `d_emb`: `32`.
- Learnable gates: `False`.
- Gate behavior: fixed uniform weights over the selected levels; logits are not trainable.

## Training Objective

- SAM stays frozen. Only the tiny scale-gated probe parameters are trainable.
- Ground-truth binary partitions are downsampled to the coarsest aligned grid with nearest-neighbor interpolation.
- If nearest-neighbor GT alignment collapses one of the two texture regions on the coarsest grid, the run hard-fails instead of silently dropping the sample.
- Affinity supervision samples `512` same/different pixel pairs per image on the coarsest grid.
- Affinity similarity uses temperature-scaled cosine logits with temperature `0.1` and a BCE-style objective.
- Optimizer: `AdamW(lr=0.001, wd=0.0001)` for `5` epoch(s).
- Random seed: `0`.

## Evaluation Protocol

- Evaluation path: frozen SAM -> aligned selected levels -> probe embedding -> deterministic k-means with `k=2` -> nearest-neighbor upsample to image size -> permutation-invariant binary assignment -> `eval_miou` and `eval_ari`.
- Clustering backend: deterministic k-means with metric `cosine`, max iterations `25`, tolerance `0.0001`.
- Probe outputs are never thresholded directly as the primary evaluation path. The main result always comes from clustering the learned embedding.

## Runtime Settings

- Variant policy: `single_level_by_name`.
- Projection dim argument: `64`.
- Embedding dim argument: `32`.
- Pair samples per image argument: `512`.
- Affinity temperature argument: `0.1`.
- K-means arguments: metric=`cosine`, `k=2`, `max_iterations=25`, `tolerance=0.0001`.

## Outputs In This Results Directory

- Common run files: `config.json`, `experiment_terms.md`, `summary.json`, `summary.csv`, `summary.md`, `per_sample_metrics.csv`, `per_sample_metrics.jsonl`, and `visuals_manifest.jsonl` under `outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2`.
- Per-sample artifacts: `sample_rows/<index>.json`, `masks/<index>.npz`, and `visuals/<index>.png` when `--save-visuals` is enabled.
- Mask bundles store `prediction_a`, `prediction_b`, `target_a`, `target_b`, and the coarsest-grid GT labels used for the affinity objective.
- Training runs additionally write `checkpoint.pt`, `gate_weights.json`, and `train_history.csv`.

## Failure Semantics

- Hard-fail on a missing experiment contract file, missing Stage-2 checkpoint path, or an unknown Stage-2 variant.
- Hard-fail on missing feature levels, inconsistent tensor ranks, inconsistent per-level channel counts across samples, or invalid coarsest-grid sizes.
- Hard-fail on NaN/Inf features, embeddings, similarities, or clustering inputs.
- Hard-fail when pair sampling yields no valid same/different supervision pairs.
- Hard-fail when k-means does not return a binary label map or when metric computation cannot assign a valid binary partition.
