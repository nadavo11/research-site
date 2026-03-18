# ArchiTexture Binary Benchmark Terms

## Run Scope

- Command family: `eval-architexture-binary`
- Route: `stld`
- Dataset id: `architexture:stld`
- Requested benchmark root: `datasets/STLD`
- Expected local root form: `<root>/benchmark` or the experiment root containing it
- Executed variant: `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only`
- Variant summary: Flip-average coarsest features across identity/hflip/vflip/hvflip, then run pooled coarse-only.
- Registered supported datasets: `rwtd`, `stld`, `caid`
- Model: `facebook/sam3`
- Device request: `cuda`
- Save visuals: `True`

## Route Fidelity Notes

- This command family reuses the selected registered repository SAM-feature experiment, not the original ArchiTexture learned Stage-A model.
- The benchmark roots and primary metric views are matched to the public ArchiTexture paper documentation as closely as this repository allows.
- STLD is scored in the direct-foreground view used by the paper.
- CAID is scored in the partition-invariant view used by the paper.
- Cross-dataset experiment contract: every current-method experiment promoted into this adapter must register itself in `src/rwtd_sam3/eval/experiment_registry.py`, declare support for `rwtd`, `stld`, and `caid`, and be documented in the root README before merge.

## Protocol Standard

1. Run `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only` exactly as registered in the cross-dataset experiment registry.
2. The route adapter reuses the registered runner unchanged and applies only route-specific GT assignment and output-directory handling.
3. Preserve disconnected same-texture regions unless the registered experiment explicitly changes that behavior.
4. Shared pooled-init settings available to the registered current-method family: feature source=`backbone_fpn`, pooled init kernel=`3`, pooled init stride=`3`, k-means max iterations=`25`, k-means convergence tolerance=`0.0001`.

## Metric Standard

- `eval_miou` / `eval_ari` are the repo-wide default metrics here and are identical to route `miou` / `ari` in this adapter.
- STLD direct foreground: the predicted cluster with higher IoU to the canonical GT foreground is treated as foreground before computing the final direct `mIoU` / `ARI`.
- CAID partition invariant: both cluster-to-region assignments are scored and the better one is reported.
- `miou_agg` is still exported for consistency with the repo-wide artifact contract, but it is not the primary route metric here.

## Output Contract

- `prediction.json` / `prediction.png` for `predict-architexture-binary`.
- `per_sample_metrics.csv`, `summary.json`, `summary.md`, and `visuals_manifest.jsonl` for `eval-architexture-binary`.
- The preview contract comes from the registry entry for `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only` and is currently `coarse_only_three_panel`.

## Failure Semantics

- There is no silent fallback to another route, dataset, or refinement stage.
- Missing benchmark roots, unmatched image/label files, invalid binary labels, empty pooled partitions, or missing official Meta `sam3` features are explicit failures.
- `failure_policy=skip` applies only to evaluation runs and records failed samples in the summary instead of aborting immediately.
