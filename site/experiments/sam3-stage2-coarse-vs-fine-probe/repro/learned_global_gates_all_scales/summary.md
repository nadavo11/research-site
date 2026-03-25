# Coarse-vs-Fine SAM Scales Stage 2 Summary

- Run kind: `train`
- Variant: `learned_global_gates_all_scales`
- Dataset: `aviadcohz/RWTD`
- Split: `test`
- Selected levels: `fpn_2`, `fpn_1`, `fpn_0`
- Coarsest level: `fpn_2`
- Probe dims: `d=64`, `d_emb=32`
- Learned gates: `True`
- Mean `eval_miou`: `0.904164`
- Mean `eval_ari`: `0.830345`
- Final gate weights: `{"fpn_0": 0.25559553503990173, "fpn_1": 0.3147805631160736, "fpn_2": 0.42962387204170227}`
- Checkpoint: `outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/smoke_learned_global_gates_all_scales/checkpoint.pt`

