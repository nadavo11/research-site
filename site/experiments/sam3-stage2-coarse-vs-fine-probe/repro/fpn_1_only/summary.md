# Coarse-vs-Fine SAM Scales Stage 2 Summary

- Run kind: `train`
- Variant: `fpn_1_only`
- Dataset: `aviadcohz/RWTD`
- Split: `test`
- Selected levels: `fpn_1`
- Coarsest level: `fpn_1`
- Probe dims: `d=64`, `d_emb=32`
- Learned gates: `False`
- Mean `eval_miou`: `0.821202`
- Mean `eval_ari`: `0.677888`
- Final gate weights: `{"fpn_1": 1.0}`
- Checkpoint: `outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fpn_1_only_test/checkpoint.pt`

