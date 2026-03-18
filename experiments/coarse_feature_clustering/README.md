# Coarse Feature Clustering Navigation

This folder is the navigation index for the SAM 3 Coarse Feature Clustering experiments.

Why it exists:
- The original run roots under `experiments/Coarse Feature Clustering/` are still the canonical artifact locations used by older scripts.
- The historical layout mixed flat dataset folders with nested flavor-specific runs, which made browsing awkward and easy to misread.
- This index documents two stable browsing views without rewriting the original run roots.

Recommended views:
- `by_dataset/<dataset>/<flavor>` groups all flavors for one dataset.
- `by_flavor/<flavor>/<dataset>` groups all datasets for one flavor.

Current flavors:
- `coarse_only`
- `flip_averaged`

Current datasets:
- `rwtd`
- `caid`
- `stld`

Canonical artifact roots:
- `experiments/Coarse Feature Clustering/<dataset>` for the historical `coarse_only` runs.
- `experiments/Coarse Feature Clustering/flip_averaged/<dataset>/<run_stamp>` for the new `flip_averaged` runs.

Build note:
- `build_multi_dataset_report.py` resolves the raw run roots directly from the canonical locations above.
- This index is for human navigation and documentation first; the old paths remain valid and unchanged.
