# Canonical Unit Validation Checklist

## Human + Agent Checklist
- run folder exists at `experiments/<run_name>/`
- folder name follows `YYYY-MM-DD_<task>_<dataset>_<model>_<preset>[_shorttag]`
- `manifest.yaml` exists and validates against schema
- `manifest.yaml.run_name` equals folder name
- `summary.md` exists and is non-empty
- `metrics.json` exists and is valid JSON
- `links.json` exists and is valid JSON
- `assets/thumbs/` exists with at least 4 files
- `assets/qc/` exists with at least 1 file
- `assets/plots/` exists with at least 1 file
- `assets/tables/` exists (may be empty)
- `repro/config.yaml` exists
- `repro/command.txt` exists and contains launch command
- `repro/env.txt` exists and contains environment snapshot
- `status.stage` is in allowed enum
- `status.verdict` is in allowed enum
- publish decision is derived only from `status.publish` plus hard gate checks

## Publish Gate Outcome
- `publishable` only if all required checks pass and `status.publish: true`
- otherwise `not_publishable` with explicit reason list
