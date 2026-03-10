# Step 3: Canonical Experiment Report Page Template

## Purpose
One report page represents one experiment/run.

The page is:
- human-inspection first
- generated from structured metadata
- understandable without opening local notebooks or code

## Source of Truth
The page must be generated from the canonical experiment folder defined in Step 2:
- `manifest.yaml`
- `summary.md`
- `metrics.json`
- `links.json`
- `assets/*`
- `repro/*`

No hand-maintained page metadata is allowed outside this source unit.

## Required Section Order
All pages must use this exact order:
1. Title block
2. Run identity / provenance
3. Configuration
4. Headline metrics
5. Visual comparison gallery
6. QC / diagnostics
7. Interpretation
8. Artifact links

## Section Contract
### 1) Title block
Must contain:
- experiment name (`experiment_id`)
- one-line purpose (`short_description`)
- status tag (`draft`, `reviewed`, `final`)

### 2) Run identity / provenance
Must contain:
- date
- git commit
- author
- project/task
- dataset list
- model and checkpoint

### 3) Configuration
Must contain:
- preset
- overrides
- seed
- key hyperparameters
- links to reproducibility config (`repro/config.yaml`, `repro/command.txt`, `repro/env.txt`)

### 4) Headline metrics
Must contain:
- compact metrics table
- baseline value (if available)
- delta vs baseline (if available)
- highlighted primary metric

### 5) Visual comparison gallery
Must contain:
- fixed number of examples per report
- fixed sample ordering from named sample list
- per-sample views: input, prediction, ground truth, optional error map
- captions with sample id and split

### 6) QC / diagnostics
Must contain:
- preview grid
- at least one distribution-style diagnostic (histogram, density, or equivalent)
- failure-case panel
- warnings/anomalies list (or explicit `none`)

### 7) Interpretation
Must contain 3-6 short bullets:
- what improved
- what regressed
- what is suspicious
- what happens next

### 8) Artifact links
Must contain links to:
- full outputs
- checkpoint
- raw logs
- external references (dataset shard / HF / notebook / related run as applicable)

## Metadata Model
Manifest for page rendering is defined by:
- schema: `docs/github_pages/report_manifest_schema.yaml`

Field classes:
- required fields: page cannot render without them
- optional fields: render if present, omit section detail if absent
- derived fields: computed by generator; never manually authored in generated page

## Visual Standard
- every report must use the same `example_count`
- gallery samples must come from a fixed named sample list
- cherry-picking best-looking-only examples is forbidden
- QC assets are mandatory

## Naming Conventions
- `experiment_id` should match Step 2 run naming convention
- asset filenames should be deterministic and sortable
- metric keys should be stable across comparable runs

## Not Allowed
- manual post-generation edits to experiment pages
- page without git commit hash
- page without preset/overrides
- page without metrics
- page without visual examples
- page without QC diagnostics
- copying raw heavy artifacts into the site repo

## Definition Of Done
A report page is valid only if:
- section order exactly matches this spec
- all required manifest fields validate
- required visual and QC assets exist
- metrics block is populated
- provenance fields (commit, preset, overrides) are explicit
- artifact links are present and resolvable
