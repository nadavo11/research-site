# SAM2 vs SAM3 Frozen Feature Clustering for Binary Texture Partitioning

- Date: `2026-03-25`
- Scope: RWTD and STLD frozen-feature automatic binary partitioning
- RWTD: `rwtd_sam2_coarse` = `eval_miou=0.856065`, `eval_ari=0.747178` vs `rwtd_sam3_flip` = `eval_miou=0.839748`, `eval_ari=0.725812`
- STLD: `stld_sam2_coarse` = `eval_miou=0.790776`, `eval_ari=0.686396`
- STLD: `stld_sam2_flip` = `eval_miou=0.806408`, `eval_ari=0.708896`
- STLD SAM3 baseline: `stld_sam3_flip` = `eval_miou=0.753395`, `eval_ari=0.619408`
- Key deltas:
  - RWTD SAM2 coarse vs SAM3 flip: `Δ mIoU=0.016317`, `Δ ARI=0.021366`
  - STLD SAM2 coarse vs SAM3 flip: `Δ mIoU=0.037381`, `Δ ARI=0.066988`
  - STLD SAM2 flip vs SAM3 flip: `Δ mIoU=0.053013`, `Δ ARI=0.089487`
- Caution: STLD `miou_agg` stays near `0.5`, so the page treats `eval_miou` and `eval_ari` as the main interpretable signal.
