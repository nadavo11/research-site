# Coarse-vs-Fine SAM Scales — Stage 2 Probe

- Dataset: `aviadcohz/RWTD` test split
- Best run: `learned_global_gates_coarse_plus_next_finer` with mean `eval_miou=0.912400` and `eval_ari=0.845200`
- Best single scale: `fpn_2_only` with mean `eval_miou=0.903166` and `eval_ari=0.828592`
- Main conclusion: coarse-dominant, not coarse-only. `fpn_1` is weaker alone but improves the best run when added to `fpn_2`.
- Fine control: `fpn_0_only_native` drops `-0.078088` mIoU and `-0.125536` ARI versus aligned `fpn_0_only`.
- Missing next control: `fixed_uniform_gates_coarse_plus_next_finer`
