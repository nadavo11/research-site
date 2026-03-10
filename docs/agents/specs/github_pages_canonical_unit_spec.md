# GitHub Pages Canonical Unit Spec

## Purpose
Define strict operational rules for creating, validating, and publishing experiment folders.

## Inputs
- `run_name` (string)
- run metadata fields required by schema
- summary text
- metrics payload
- links payload
- asset files for `thumbs`, `qc`, `plots`, optional `tables`
- reproducibility files (`config.yaml`, `command.txt`, `env.txt`)

## Output
- one folder at `experiments/<run_name>/` that passes schema and structure checks
- publishability decision: `publishable` or `not_publishable`

## Required Files and Folders
The unit is valid only if all required entries exist:

```text
experiments/<run_name>/
  manifest.yaml
  summary.md
  metrics.json
  links.json
  assets/thumbs/
  assets/qc/
  assets/plots/
  assets/tables/
  repro/config.yaml
  repro/command.txt
  repro/env.txt
```

Optional:
- `notes.md`
- `status.json`

## Validation Rules
1. `run_name` must match naming policy.
2. `manifest.yaml` must validate against `schemas/experiment_manifest.schema.json`.
3. `manifest.yaml.run_name` must equal folder name.
4. `evaluation.metrics_file` must point to `metrics.json`.
5. `status.publish` controls publish eligibility.
6. reproducibility files are mandatory regardless of `publish`.
7. minimum visual evidence for publish:
- at least 4 files in `assets/thumbs/`
- at least 1 file in `assets/qc/`
- at least 1 file in `assets/plots/`

## Failure Conditions
Return `not_publishable` if any condition fails. Report explicit reasons, including:
- missing required files/folders
- schema violations with field-level path
- invalid run name format
- missing reproducibility pointers
- missing minimum visual evidence
- `status.publish` is `false`

## Publishing Contract
Only publish when all pass:
1. folder structure valid
2. schema valid
3. asset minimums satisfied
4. `status.publish: true`

Rendered pages must be generated from canonical source units, never hand-edited as source of truth.
