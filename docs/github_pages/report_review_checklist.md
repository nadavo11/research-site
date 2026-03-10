# Report Review Checklist

Use this checklist to approve or reject one experiment page before publication.

## Scope
- page represents exactly one experiment/run
- page is understandable without local code or notebook context

## Structure
- section order matches the canonical 8-section order
- each required section contains mandatory content

## Required Identity
- `experiment_id` present and valid
- `git_commit` present and valid hash format
- `preset` and explicit `overrides` present
- date, author, task, dataset, model, checkpoint present

## Metrics
- compact metrics table exists
- primary metric is explicit
- baseline and delta shown when available
- no empty metrics block

## Visuals
- gallery exists with fixed example count
- sample order follows named fixed sample list
- captions include sample metadata
- no cherry-picked-only presentation

## QC
- preview grid exists
- at least one diagnostic distribution plot exists
- failure cases are shown
- warnings/anomalies section exists (`none` allowed)

## Links and Artifacts
- links to full outputs, checkpoint, and raw logs exist
- heavy raw artifacts are linked externally, not copied into site repo

## Reject Conditions
- manual edits were applied to generated page content
- missing commit hash
- missing preset/overrides
- missing metrics
- missing gallery visuals
- missing QC diagnostics

## Decision
- `approve` only if every required check passes
- otherwise `reject` with explicit reasons
