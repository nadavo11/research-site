# RWTD SAM 3 Evaluation Summary

- Dataset: `aviadcohz/RWTD`
- Split: `test`
- Protocol: `text`
- Model: `facebook/sam3`
- Evaluated samples: `227` / `227`
- Failed samples: `0`
- Empty texture predictions: `133`

## Mean Metrics

| Metric | Mean | Median |
| --- | ---: | ---: |
| `texture_a_iou` | 0.568840 | 0.852976 |
| `texture_a_dice` | 0.599573 | 0.920655 |
| `texture_a_precision` | 0.630572 | 0.992386 |
| `texture_a_recall` | 0.589765 | 0.917247 |
| `texture_b_iou` | 0.527165 | 0.795648 |
| `texture_b_dice` | 0.555392 | 0.886196 |
| `texture_b_precision` | 0.577606 | 0.972298 |
| `texture_b_recall` | 0.549400 | 0.852570 |
| `sample_macro_iou` | 0.548003 | 0.492691 |
| `sample_miou` | 0.548003 | 0.492691 |
| `sample_macro_dice` | 0.577482 | 0.496373 |
| `sample_macro_precision` | 0.604089 | 0.500000 |
| `sample_macro_recall` | 0.569582 | 0.495099 |
| `sample_ari` | 0.685940 | 0.865928 |
| `boundary_iou` | 0.010057 | 0.000000 |
| `boundary_dice` | 0.015642 | 0.000000 |
| `boundary_precision` | 0.177200 | 0.000000 |
| `boundary_recall` | 0.075263 | 0.000000 |
