# SAM2 vs SAM3 Frozen Feature Clustering for Binary Texture Partitioning

- Date: `2026-03-26`
- Scope: RWTD, CAID, and STLD same-flavor frozen-feature binary partitioning
- Coarse-only lead split: `SAM2 3/3`
- Flip-avg lead split: `SAM2 2/3`
- Coarse-only rows:
  - RWTD: `SAM2 0.856065/0.747178` vs `SAM3 0.823512/0.698426`
  - CAID: `SAM2 0.726335/0.567273` vs `SAM3 0.674265/0.495585`
  - STLD: `SAM2 0.790776/0.686396` vs `SAM3 0.468205/0.185865`
- Flip-avg rows:
  - RWTD: `SAM2 0.827775/0.701311` vs `SAM3 0.839748/0.725812`
  - CAID: `SAM2 0.662040/0.460517` vs `SAM3 0.658828/0.450670`
  - STLD: `SAM2 0.806408/0.708896` vs `SAM3 0.753395/0.619408`
- Caution: STLD `miou_agg` remains nearly flat across both models and both flavors.
