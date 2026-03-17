# ArchiTexture Binary Benchmark Terms

## Run Scope

- Command family: `eval-architexture-binary`
- Route: `stld`
- Dataset id: `architexture:stld`
- Requested benchmark root: `datasets/STLD`
- Expected local root form: `<root>/benchmark` or the experiment root containing it
- Executed variant: `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only`
- Model: `facebook/sam3`
- Device request: `cuda`
- Save visuals: `True`

## Route Fidelity Notes

- This command family reuses the current repository SAM-feature baseline, not the original ArchiTexture learned Stage-A model.
- The benchmark roots and primary metric views are matched to the public ArchiTexture paper documentation as closely as this repository allows.
- STLD is scored in the direct-foreground view used by the paper.
- CAID is scored in the partition-invariant view used by the paper.

## Protocol Standard

1. Run `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only` exactly: extract the official Meta `sam3` multiscale `backbone_fpn` features, average-pool the coarsest feature level, cluster that pooled grid globally into 2 unlabeled groups, upsample the pooled label map back to the native coarsest resolution, then upsample directly to image resolution.
2. Do not run finer-level boundary-band refinement.
3. Do not feed the coarse masks back into the official Meta SAM mask-prompt refinement path.
4. Treat the pooled coarsest partition itself as the final 2-mask prediction set.
5. Preserve disconnected same-texture regions; there is no connected-component cleanup or proposal filtering.
6. Current pooled-init settings: feature source=`backbone_fpn`, pooled init kernel=`3`, pooled init stride=`3`, k-means max iterations=`25`, k-means convergence tolerance=`0.0001`.

## Metric Standard

- `miou` and `ari` are the primary route-comparison metrics here.
- STLD direct foreground: the predicted cluster with higher IoU to the canonical GT foreground is treated as foreground before computing the final direct `mIoU` / `ARI`.
- CAID partition invariant: both cluster-to-region assignments are scored and the better one is reported.
- `miou_agg` is still exported for consistency with the repo-wide artifact contract, but it is not the primary route metric here.

## Output Contract

- `prediction.json` / `prediction.png` for `predict-architexture-binary`.
- `per_sample_metrics.csv`, `summary.json`, `summary.md`, and `visuals_manifest.jsonl` for `eval-architexture-binary`.
- The preview panel is the same slim three-panel coarse-only view already used by this ablation: `Input | GT | Coarsest partition`.

## Failure Semantics

- There is no silent fallback to another route, dataset, or refinement stage.
- Missing benchmark roots, unmatched image/label files, invalid binary labels, empty pooled partitions, or missing official Meta `sam3` features are explicit failures.
- `failure_policy=skip` applies only to evaluation runs and records failed samples in the summary instead of aborting immediately.
