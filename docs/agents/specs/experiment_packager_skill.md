# Skill Note: Experiment Packager

## Purpose
Turn completed run outputs into a valid canonical experiment folder.

## Inputs
- run metadata
- config used
- metrics outputs
- asset file paths
- optional notes

## Outputs
- populated `experiments/<run_name>/` structure
- generated `manifest.yaml` and required companion files

## Required Checks
- naming convention compliance
- schema-required metadata fields present
- required folders/files created
- repro files captured

## Failure Conditions
- missing commit hash
- missing metrics file
- no visual evidence assets
- invalid `run_name`
- ambiguous preset/override description
