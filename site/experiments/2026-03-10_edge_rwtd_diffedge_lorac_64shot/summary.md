# Summary

## What was tested
Synthetic-to-RWTD edge detection performance for `synthetic_diffedge_rwtd_loraC_64shot`.

## Why
This run is the best completed synthetic-domain AP in the lightweight bundle and is suitable as a canonical reporting example.

## What changed
LoRA preset C with 64-shot adaptation, compared against the synthetic zero-shot baseline in the same bundle.

## Result
- AP: `0.484539` (delta vs synthetic zero-shot: `+0.071225`)
- ODS F1: `0.654000`
- OIS F1: `0.654737`

## Verdict
Milestone for synthetic-domain AP, but QC still flags difficult preview cases with fragmented thin boundaries.
