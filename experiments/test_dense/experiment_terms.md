# SAM-3 Automatic Mask Comparison Definitions

- Dataset: `aviadcohz/RWTD`
- Split: `test`
- Requested variant setting: `dense`
- Executed variant(s): `dense`
- Model: `facebook/sam3`

## Variant Definitions

- `default`: SAM-3 automatic mask generation with `points_per_crop=32` and `stability_score_thresh=0.95`.
- `dense`: SAM-3 automatic mask generation with `points_per_crop=64` and `stability_score_thresh=0.2`.

## Metric Definitions

- `all` dataset view: concatenates the public RWTD `train` and `test` splits in that order.
- `miou`: non-aggregated mean IoU using the best raw predicted mask per ground-truth region.
- `ari`: non-aggregated Adjusted Rand Index between pixel labels from ground-truth regions and raw predicted masks.
- `miou_agg`: aggregated mean IoU after unioning all raw masks with non-empty overlap against each ground-truth region.
- `texture_a_overlap_mask_count` / `texture_b_overlap_mask_count`: number of raw masks contributing to each aggregated region.

## Evaluation Notes

- This is a supplementary SAM-3 comparison track, not the TextureSAM paper baseline.
- This repository treats `miou_agg` as the primary automatic-mask comparison metric.
- Aggregation is used only for `miou_agg`.
- ARI and non-aggregated `miou` are computed from the raw mask set.
- Predicted masks are ordered by descending mask score; earlier masks take precedence when assigning a single predicted label per pixel for ARI.
- Saved visuals include a raw categorical-mask panel with one color per visible proposal plus a separate aggregated A/B panel.
