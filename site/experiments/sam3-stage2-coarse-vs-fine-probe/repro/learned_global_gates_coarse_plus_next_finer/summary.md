# Coarse-vs-Fine SAM Scales Stage 2 Summary

- Run kind: `train`
- Variant: `learned_global_gates_coarse_plus_next_finer`
- Dataset: `aviadcohz/RWTD`
- Split: `test`
- Selected levels: `fpn_2`, `fpn_1`
- Coarsest level: `fpn_2`
- Probe dims: `d=64`, `d_emb=32`
- Learned gates: `True`
- Mean `eval_miou`: `0.912400`
- Mean `eval_ari`: `0.845200`
- Final gate weights: `{"fpn_1": 0.4235217273235321, "fpn_2": 0.5764782428741455}`
- Checkpoint: `outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_learned_global_gates_coarse_plus_next_finer_test/checkpoint.pt`

