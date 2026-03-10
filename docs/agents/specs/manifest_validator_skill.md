# Skill Note: Manifest Validator

## Purpose
Validate `manifest.yaml` against the canonical schema and policy checks.

## Inputs
- `experiments/<run_name>/manifest.yaml`
- `schemas/experiment_manifest.schema.json`

## Outputs
- pass/fail result
- list of field-level violations
- suggested corrections

## Good Integration Points
- pre-commit hook
- CI validation job
- pre-publish gate
