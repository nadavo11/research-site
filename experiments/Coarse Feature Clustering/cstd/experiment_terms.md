# CSTD Benchmark Terms

## Run Scope

- Command family: `eval-cstd-binary`
- Dataset id: `cstd`
- Requested dataset root: `datasets/CSTD`
- Expected local root form: `<root>/images`, `<root>/regions`, and `<root>/edges`.
- Executed variant: `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only`
- Variant summary: Flip-average coarsest features across identity/hflip/vflip/hvflip, then run pooled coarse-only.
- Registered supported datasets: `rwtd`, `stld`, `caid`, `detexture_ade20k`, `cstd`
- Model: `facebook/sam3`
- Device request: `cuda`
- Hardware compatibility standard: `cstd_streaming_eval_v1`
- Sample loading mode: `streamed_iter` (the adapter does not preload all images into memory).
- Save visuals: `True`
- Dataset partition: `1/20` -> indices `0:500` (500 selected sample(s) from 500 partition items, full dataset size 10000).

## Dataset Semantics

- Each sample is one image from `images/`, one binary region mask from `regions/`, and one binary edge map from `edges/` with the same natural-sorted stem.
- The adapter uses the provided `regions/` mask as texture A, its complement as texture B, and preserves the provided `edges/` mask as the boundary visualization input.
- The task is scored as a two-region binary partition with permutation-invariant evaluation.
- Cross-dataset experiment contract: every current-method experiment promoted into this adapter must register itself in `src/rwtd_sam3/eval/experiment_registry.py`, declare support for all registered binary datasets, and be documented in the root README before merge.

## Protocol Standard

1. Run `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only` exactly as registered in the cross-dataset experiment registry.
2. The dataset adapter reuses the registered runner unchanged and applies only CSTD-specific GT decoding, permutation-invariant assignment, and output-directory handling.
3. Preserve disconnected same-texture regions unless the registered experiment explicitly changes that behavior.
4. Shared pooled-init settings available to the registered current-method family: feature source=`backbone_fpn`, pooled init kernel=`3`, pooled init stride=`3`, k-means max iterations=`25`, k-means convergence tolerance=`0.0001`.

## Metric Standard

- `eval_miou` / `eval_ari` are the repo-wide default metrics here and are identical to partition-invariant `miou` / `ari`.
- Both cluster-to-region assignments are scored and the better one is reported.
- `miou_agg` is still exported for consistency with the repo-wide artifact contract, but it is not the primary dataset metric here.

## Output Contract

- `prediction.json` / `prediction.png` for `predict-cstd-binary`.
- `per_sample_metrics.csv`, `summary.json`, `summary.md`, and `visuals_manifest.jsonl` for `eval-cstd-binary`.
- The preview contract comes from the registry entry for `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only` and is currently `coarse_only_three_panel`.

## Failure Semantics

- There is no silent fallback to another dataset or refinement stage.
- Missing dataset roots, unmatched image/region/edge triples, empty regions, shape mismatches, or missing official Meta `sam3` features are explicit failures.
- `failure_policy=skip` applies only to evaluation runs and records failed samples in the summary instead of aborting immediately.
