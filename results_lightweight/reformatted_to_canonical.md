# Canonical Reformat Mapping

This lightweight bundle has been reformatted into a canonical experiment page package at:

`site/experiments/2026-03-10_edge_rwtd_diffedge_lorac_64shot/`

Selected run:
- `synthetic_diffedge_rwtd_loraC_64shot`

Mapping used:
- `per_run/<run_id>/metrics.json` -> `site/experiments/<run_name>/metrics.json`
- `per_run/<run_id>/eval_results.json` -> `site/experiments/<run_name>/repro/eval_results.json`
- `per_run/<run_id>/performance_summary.json` -> `site/experiments/<run_name>/repro/performance_summary.json`
- `qualitative/previews/<run_id>/*_preview.png` -> `site/experiments/<run_name>/assets/gallery/`
- `qualitative/<domain>_preview_grid.png` -> `site/experiments/<run_name>/assets/qc/preview_grid.png`
- `plots/*.png` -> `site/experiments/<run_name>/assets/plots/` and selected QC diagnostics
- `tables/core_results.csv` -> `site/experiments/<run_name>/assets/tables/core_results.csv`

Agent contract for future runs:
- `results_lightweight/agent_result_format_manifest.yaml`
- `docs/agents/specs/result_bundle_page_contract.yaml`
