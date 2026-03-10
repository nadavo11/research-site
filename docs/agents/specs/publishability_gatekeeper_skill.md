# Skill Note: Publishability Gatekeeper

## Purpose
Decide whether a canonical experiment folder is ready for GitHub Pages publication.

## Inputs
- canonical experiment folder path
- schema validation result
- publish policy checks

## Outputs
- `publishable` or `not_publishable`
- explicit failure reasons
- privacy/sensitivity warnings if links or assets indicate risk

## Gate Rules
- require structure + schema pass
- require reproducibility files
- require minimum visual evidence
- require `status.publish: true`
