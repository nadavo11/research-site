# Step 2: Canonical Experiment/Report Unit

## Decision
Every meaningful experiment is represented by one canonical source folder with fixed metadata, fixed assets, fixed reproducibility pointers, and fixed publishability status. GitHub Pages pages are generated from these folders, not authored manually.

## Why
- keeps experiment tracking consistent for humans and agents
- enables deterministic validation and generation
- separates source-of-truth experiment records from rendered pages

## Canonical Source Unit
One experiment maps to exactly one folder:

`experiments/<run_name>/`

Required structure:

```text
experiments/<run_name>/
  manifest.yaml
  summary.md
  metrics.json
  links.json
  assets/
    thumbs/
    qc/
    plots/
    tables/
  repro/
    config.yaml
    command.txt
    env.txt
```

Optional:

```text
  notes.md
  status.json
```

## Run Name Convention
Required format:

`YYYY-MM-DD_<task>_<dataset>_<model>_<preset>[_shorttag]`

Rules:
- lowercase only
- underscores only
- no spaces
- no vague names (`test2`, `final_final`)
- optional `shorttag` only when informative

## Publish Rule
A run is publishable only when:

`status.publish: true`

Runs may still exist as canonical source units even when not publishable.

## Controlled Status Values
`status.stage` must be one of:
- `draft`
- `running`
- `completed`
- `failed`
- `archived`

`status.verdict` must be one of:
- `keep`
- `reject`
- `inconclusive`
- `baseline`
- `milestone`

## Step 2 Deliverables
- decision doc (this file)
- strict agent operational spec
- machine schema for `manifest.yaml`
- template folder scaffold
- validation checklist and skill notes
