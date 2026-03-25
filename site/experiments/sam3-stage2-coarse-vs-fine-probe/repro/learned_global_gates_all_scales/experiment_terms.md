# Coarse vs Fine SAM Scales: Experiment Contract

## Status

This file is the lock on the experiment question.
Do not change the hypothesis, metrics, feature sources, or output contract mid-study without editing this file first and recording the reason.

## Claim

Coarse SAM features contain most of the signal for texture-region partitioning.
Finer SAM features contribute less to global partitioning and may mainly help boundary localization.

## What This Experiment Is Allowed To Prove

- It can compare how useful different SAM feature scales are for binary texture partitioning on this benchmark.
- It can measure whether coarse-only, fine-only, or cumulative coarse-to-fine feature usage is more stable under the current clustering pipeline.
- It can test whether adding finer scales improves boundary-local behavior without materially improving global partition consistency.

## What This Experiment Is Not Allowed To Prove

- It cannot yet explain why SAM learned this internal scale behavior.
- It cannot yet claim that coarse is universally better for every downstream task.
- It cannot yet claim that the result will transfer unchanged to non-texture, multi-object, or prompt-driven segmentation settings.

## Primary Metrics

- `eval_miou`
- `eval_ari`

These are the invariant binary-partition metrics already used by the repository evaluation contract.

## Secondary Metrics

- `boundary_f1`, if already available with negligible extra plumbing
- `trimap_iou`, if already available with negligible extra plumbing

Secondary metrics are optional in the first pass. They must not delay the scale-comparison implementation.

## Main Evaluation Settings

- single-scale only
- cumulative coarse-to-fine
- later: learned gated probe

The first pass is diagnostic, not a learned system.
No training is part of the initial comparison.

## Expected Signatures If The Hypothesis Is True

- coarse-only performs best or near-best on partition metrics
- adding fine scales gives small gains or unstable gains
- fine-only underperforms on partition consistency
- if boundary metrics are included, finer scales may help there more than they help `eval_miou` / `eval_ari`

## Dataset Contract

- dataset: `aviadcohz/RWTD`
- default split for the study: `test`
- optional later extension: `all`, but only after the `test`-split comparison is stable

If a run uses anything other than `split=test`, that change must be stated explicitly in the run config and summary.

## Exact Feature Levels

The experiment uses the official Meta SAM-3 image backend and its multiscale `backbone_fpn` features.

Current expected ordering in this repository:

1. `fpn_2`: coarsest
2. `fpn_1`: middle
3. `fpn_0`: finest

Important constraints:

- level ordering must be determined by actual spatial resolution, not name alone
- every run must record the level names and native `(height, width)` resolutions it used
- comparisons must remain explicit about whether they are:
  - coarse-only
  - fine-only
  - cumulative coarse-to-fine

## Exact Clustering Backend

The clustering backend is fixed for this study unless this file is revised first.

- binary clustering only: `k=2`
- feature vectors are L2-normalized before clustering
- no spatial coordinates are appended
- no adjacency graph, superpixels, CRF, or connected-component cleanup
- no learned probe in the initial phase

Current reference implementation target:

- deterministic cosine-style 2-cluster assignment over normalized SAM feature vectors
- same backend across all scale comparisons unless explicitly stated otherwise

If a later study uses learned gating or a different clustering backend, that is a separate phase and must not silently replace this one.

## Comparison Conditions

The initial coarse-vs-fine study is allowed to compare:

- single-scale coarse only
- single-scale middle only
- single-scale fine only
- cumulative coarse-to-fine

The single most important invariant is:

- the comparison must isolate scale usage, not bundle in unrelated postprocessing changes

## Output Folder Structure

All runs for this study should live under a dedicated namespace:

```text
outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/
  <run_name>/
    config.json
    experiment_terms.md
    per_sample_metrics.csv
    summary.json
    summary.md
    visuals_manifest.jsonl
    visuals/
      <crop_name>.png
```

Recommended run naming:

- `test_single_scale_coarse`
- `test_single_scale_middle`
- `test_single_scale_fine`
- `test_cumulative_coarse_to_fine`

Each run must save enough artifacts to recover:

- which feature levels were used
- whether the run was single-scale or cumulative
- the exact clustering backend
- the final invariant partition metrics

## Visualization Contract

Per-sample visuals should make the scale question obvious.

Minimum useful panels:

- input
- ground truth
- active cluster map or partition from the tested scale setting
- final chosen binary output

If cumulative coarse-to-fine is used, intermediate per-level maps may be shown, but the panel must still clearly indicate the tested condition.

## Non-Goals For This Phase

- no training
- no learned gates
- no architectural interpretation claims
- no broad generalization claims beyond this benchmark and evaluator

## Decision Rule

Do not change the experiment question while implementing convenience fixes.

If a code change would alter:

- the feature source
- the scale ordering
- the clustering backend
- the evaluation split
- the primary metrics

then update this contract first and explain why.

## Stage 2 Runtime Settings

- Run kind: `train`
- Variant: `learned_global_gates_all_scales`
- Model: `facebook/sam3`
- Device: `auto`
- Official checkpoint override: `None`
- Projection dim: `64`
- Embedding dim: `32`
- Pair samples per image: `512`
- Affinity temperature: `0.1`
- Optimizer: `AdamW(lr=0.001, wd=0.0001)`
- Epochs: `5`
- K-means: `metric=cosine`, `k=2`, `max_iterations=25`, `tolerance=0.0001`
- Primary metrics: `eval_miou`, `eval_ari`
