# Coarse-vs-Fine SAM Scales Stage 2 Summary

- Run kind: `train`
- Variant: `fpn_0_only`
- Dataset: `aviadcohz/RWTD`
- Split: `test`
- Selected levels: `fpn_0`
- Coarsest level: `fpn_2`
- Probe dims: `d=64`, `d_emb=32`
- Learned gates: `False`
- Mean `eval_miou`: `0.732633`
- Mean `eval_ari`: `0.531057`
- Final gate weights: `{"fpn_0": 1.0}`
- Checkpoint: `outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fpn_0_only_test/checkpoint.pt`

