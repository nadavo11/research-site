# DeTexture ADE20K Summary

- Dataset: `detexture_ade20k`
- Dataset root: `/home/nada/PycharmProjects/texture representations/datasets/detexture_ADE20K`
- Evaluation view: `partition_invariant`
- Variant: `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only`
- Registered supported datasets: `rwtd`, `stld`, `caid`, `detexture_ade20k`
- Model: `facebook/sam3`
- Hardware compatibility standard: `detexture_streaming_eval_v1`
- Evaluated samples: `2201` / `2201`
- Failed samples: `0`
- Default comparison view: `eval_miou` / `eval_ari` = `0.626535` / `0.406370`

## Mean Metrics

| Metric | Mean | Median |
| --- | ---: | ---: |
| `eval_miou` | 0.626535 | 0.638943 |
| `eval_ari` | 0.406370 | 0.354642 |
| `miou` | 0.626535 | 0.638943 |
| `ari` | 0.406370 | 0.354642 |
| `miou_agg` | 0.490175 | 0.490201 |
| `texture_a_best_iou` | 0.632776 | 0.660969 |
| `texture_b_best_iou` | 0.620294 | 0.652882 |
| `texture_a_agg_iou` | 0.502515 | 0.490015 |
| `texture_b_agg_iou` | 0.477834 | 0.484332 |
| `num_predicted_masks` | 2.000000 | 2.000000 |
| `texture_a_overlap_mask_count` | 1.976829 | 2.000000 |
| `texture_b_overlap_mask_count` | 1.980918 | 2.000000 |
| `mask_score_mean` | 0.000000 | 0.000000 |
| `mask_score_median` | 0.000000 | 0.000000 |
