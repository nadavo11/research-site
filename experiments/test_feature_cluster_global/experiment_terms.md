# SAM-3 Automatic Mask Comparison Definitions

- Dataset: `aviadcohz/RWTD`
- Split: `test`
- Requested variant setting: `feature_cluster_global`
- Executed variant(s): `feature_cluster_global`
- Model: `facebook/sam3`

## Variant Definitions

- `default`: SAM-3 automatic mask generation with `points_per_crop=32` and `stability_score_thresh=0.95`.
- `dense`: SAM-3 automatic mask generation with `points_per_crop=64` and `stability_score_thresh=0.2`.
- `feature_mask`: uses the `dense` proposals, prefers the best adjacent dense-mask pair when available, falls back to a single dense mask plus a synthetic local complement when needed, builds a local coarse feature prior with official SAM dense image features, and re-runs official SAM with that mask prompt.

- `feature_cluster_global`: extracts official SAM dense image features, runs a global 2-way clustering over normalized features without spatial regularization or proposals, refines both cluster masks with official SAM mask prompts, and evaluates them with permutation-invariant assignment.

## Metric Definitions

- `all` dataset view: concatenates the public RWTD `train` and `test` splits in that order.
- `miou`: non-aggregated mean IoU using one-to-one IoU matching between raw predicted masks and ground-truth regions.
- `ari`: non-aggregated Adjusted Rand Index between ground-truth region labels and the pixelwise partition induced by raw predicted-mask memberships.
- `miou_agg`: aggregated mean IoU after unioning all raw masks with non-empty overlap against each ground-truth region.
- `texture_a_overlap_mask_count` / `texture_b_overlap_mask_count`: number of raw masks contributing to each aggregated region.

## Evaluation Notes

- This is a supplementary SAM-3 comparison track, not the TextureSAM paper baseline.
- This repository treats `miou_agg` as the primary automatic-mask comparison metric.
- Aggregation is used only for `miou_agg`.
- ARI and non-aggregated `miou` are computed from the raw mask set.
- Raw ARI uses the full predicted-mask membership partition, so overlapping and nested masks still penalize fragmentation.
- Saved visuals include a raw categorical-mask panel with one color per visible proposal plus a separate aggregated A/B panel.
- `feature_mask` saves an extended panel with dense proposals, the selected support regions, the coarse feature prior, and the final refined output.
- `feature_mask` records `feature_mask_mode=pair` when both support regions come from dense proposals and `feature_mask_mode=single_mask_complement` when the second region is synthesized locally.
- `feature_mask` fails explicitly only if it cannot build a valid foreground/background prototype pair, coarse prompt, or refined mask.
- `feature_cluster_global` is intentionally spatially agnostic: it does not use proposals, points, text, boxes, adjacency, smoothing, or connected-component cleanup.
- `feature_cluster_global` records the chosen GT-aligned assignment in `assignment_used` and treats cluster labels as permutation-invariant.
