# RWTD SAM 3 Evaluation Summary

- Dataset: `aviadcohz/RWTD`
- Split: `test`
- Protocol: `oracle_points`
- Model: `facebook/sam3`
- Evaluated samples: `227` / `227`
- Failed samples: `0`
- Empty texture predictions: `5`

## Mean Metrics

| Metric | Mean | Median |
| --- | ---: | ---: |
| `texture_a_iou` | 0.512714 | 0.679670 |
| `texture_a_dice` | 0.561300 | 0.809290 |
| `texture_a_precision` | 0.962572 | 0.999980 |
| `texture_a_recall` | 0.515460 | 0.679670 |
| `texture_b_iou` | 0.519953 | 0.700897 |
| `texture_b_dice` | 0.573401 | 0.824150 |
| `texture_b_precision` | 0.934451 | 0.999339 |
| `texture_b_recall` | 0.527699 | 0.718438 |
| `sample_macro_iou` | 0.516333 | 0.484611 |
| `sample_miou` | 0.516333 | 0.484611 |
| `sample_macro_dice` | 0.567350 | 0.516301 |
| `sample_macro_precision` | 0.948512 | 0.997375 |
| `sample_macro_recall` | 0.521579 | 0.491324 |
| `sample_ari` | 0.614436 | 0.774900 |
| `boundary_iou` | 0.006563 | 0.000000 |
| `boundary_dice` | 0.012046 | 0.000000 |
| `boundary_precision` | 0.169939 | 0.000000 |
| `boundary_recall` | 0.026854 | 0.000000 |
