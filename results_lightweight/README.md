# Lightweight Results Bundle

This folder contains a compact, report-ready subset of the RWTD LoRA Stage-1 run outputs.
It intentionally excludes heavy artifacts such as full checkpoints and full prediction dumps.

## Structure

- `tables/`
  - `master_results.csv|json|md`: copied from `summary/`
  - `core_results.csv|json`: compact table with the key report fields
- `per_run/<run_id>/`
  - `metrics.json` / `eval_results.json` / `performance_summary.json` (if available)
  - `training_data.json` (Parsed tensorboard logs and eval metrics for learning curves)
  - `edgeeval/eval_bdry*.txt` (PR/threshold/image-level curve data)
- `plots/`
  - `ap_vs_shots.png`
  - `f1_vs_shots.png`
  - `delta_vs_zero_shot.png`
  - `status_overview.png`
  - `pr_curve_<domain>.png`
- `qualitative/`
  - `<domain>_preview_grid.png`: representative side-by-side qualitative comparisons
  - `selected_preview_ids.json`: chosen sample IDs and included runs
  - `previews/<run_id>/<id>_preview.png`: only selected preview images used in the report
  - `all_previews/<run_id>/*.png`: complete exhaustive set of evaluation previews for the dedicated gallery
- `report.md`: short report index and best-run summary
- `manifest.json`: machine-readable inventory (generated timestamp, included files, bundle size)
- `scripts/build_lightweight_results.py`: regeneration script

## Regenerate

From the run directory:

```bash
python results_lightweight/scripts/build_lightweight_results.py
```

Optional:

```bash
python results_lightweight/scripts/build_lightweight_results.py --previews-per-domain 6
```

