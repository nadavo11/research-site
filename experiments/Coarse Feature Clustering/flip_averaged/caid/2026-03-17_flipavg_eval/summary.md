# ArchiTexture Binary Benchmark Summary

- Route: `caid`
- Dataset: `architexture:caid`
- Benchmark root: `/home/nada/PycharmProjects/texture representations/datasets/architexture/caid_test`
- Evaluation view: `partition_invariant`
- Variant: `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only`
- Registered supported datasets: `rwtd`, `stld`, `caid`
- Model: `facebook/sam3`
- Hardware compatibility standard: `architexture_streaming_eval_v1`
- Evaluated samples: `3104` / `3104`
- Failed samples: `0`
- Default comparison view: `eval_miou` / `eval_ari` = `0.658828` / `0.450670`

## Mean Metrics

| Metric | Mean | Median |
| --- | ---: | ---: |
| `eval_miou` | 0.658828 | 0.689241 |
| `eval_ari` | 0.450670 | 0.470281 |
| `miou` | 0.658828 | 0.689241 |
| `ari` | 0.450670 | 0.470281 |
| `miou_agg` | 0.531660 | 0.500000 |
| `texture_a_best_iou` | 0.665195 | 0.746285 |
| `texture_b_best_iou` | 0.652461 | 0.808609 |
| `texture_a_agg_iou` | 0.531505 | 0.546980 |
| `texture_b_agg_iou` | 0.531814 | 0.551622 |
| `num_predicted_masks` | 2.000000 | 2.000000 |
| `texture_a_overlap_mask_count` | 1.866946 | 2.000000 |
| `texture_b_overlap_mask_count` | 1.870812 | 2.000000 |
| `mask_score_mean` | 0.000000 | 0.000000 |
| `mask_score_median` | 0.000000 | 0.000000 |
