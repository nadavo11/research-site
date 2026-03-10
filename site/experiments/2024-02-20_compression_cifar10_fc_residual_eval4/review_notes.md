# Step 4 Pilot Review Notes

## What worked well
- Section order was easy to scan in under a minute.
- Fixed sample list (`prova_1..prova_4`) prevented cherry-picking.
- QC block made trust issues obvious (edge loss and explicit failure case).
- Manifest-first structure maps directly to Step 3 schema and future automation.

## What was awkward or missing
- Legacy artifacts do not preserve an exact checkpoint-to-output mapping.
- Raw training logs are not available as dedicated files.
- Baseline comparator is synthetic (mean-color), useful for sanity but weak for benchmarking.

## Changes needed before Step 5 automation
- Require a strict run-id + checkpoint-id link in generated outputs.
- Require machine-readable eval logs per run (json/csv) in the source run folder.
- Add an explicit baseline registry so baseline choice is consistent across reports.
- Add a small validation script that blocks publish when provenance links are ambiguous.
