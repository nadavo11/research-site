# Coarse-vs-Fine SAM Scales Stage 2 Summary

- Run kind: `train`
- Variant: `fixed_uniform_gates_all_scales`
- Dataset: `aviadcohz/RWTD`
- Split: `test`
- Selected levels: `fpn_2`, `fpn_1`, `fpn_0`
- Coarsest level: `fpn_2`
- Probe dims: `d=64`, `d_emb=32`
- Learned gates: `False`
- Mean `eval_miou`: `0.908272`
- Mean `eval_ari`: `0.838149`
- Final gate weights: `{"fpn_0": 0.3333333432674408, "fpn_1": 0.3333333432674408, "fpn_2": 0.3333333432674408}`
- Checkpoint: `outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fixed_uniform_gates_all_scales_test/checkpoint.pt`

