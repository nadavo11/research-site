# Experiment Definitions and Terms

- Dataset: `aviadcohz/RWTD`
- Split: `test`
- Requested protocol setting: `both`
- Executed protocol(s): `text`, `oracle_points`
- Model: `facebook/sam3`

## Core Terms

- RWTD crop: one image crop with two annotated texture regions and one boundary mask.
- `texture_a` / `texture_b`: natural-language texture descriptions stored in RWTD.
- `original_texture_a` / `original_texture_b`: shorter source labels stored in RWTD.
- `oracle_points_a` / `oracle_points_b`: positive point prompts in pixel coordinates.
- `text` protocol: predicts texture A from `texture_a` text and texture B from `texture_b` text.
- `oracle_points` protocol: predicts texture A from `oracle_points_a` and texture B from `oracle_points_b`.
- `both`: runs both protocols and writes separate protocol subdirectories.
- `sample_macro_*`: arithmetic mean of the texture-A and texture-B metric for one sample.
- `sample_miou`: explicit mean IoU over texture A and texture B for one sample.
- `sample_ari`: Adjusted Rand Index over the per-pixel partition labels `{background, A-only, B-only, overlap}`.
- `boundary_*`: metrics computed on a derived boundary mask from the predicted A/B regions.
- `boundary_tolerance_px`: boundary matching radius in pixels, set to `2` for this run.
- `empty_texture_predictions`: number of empty texture masks for a sample or aggregated run.

## Visual Artifacts

- `visuals/*.png`: input, ground-truth, and prediction panels with wrapped footer captions.
- `visuals_manifest.jsonl`: one JSON record per saved visual with the image path, full caption text, prompts, labels, oracle-point counts, and per-sample metrics.

## Prompt Semantics

- SAM 3 is always conditioned on a prompt in this repository.
- `text` uses RWTD texture descriptions as prompts.
- `oracle_points` uses RWTD positive oracle points as prompts.
