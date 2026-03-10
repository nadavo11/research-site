# Lightweight Report Pack

- Generated (UTC): 2026-03-10T12:56:42.166789+00:00
- Source run root: `/home/nada/PycharmProjects/DiffusionEdge/runs/diffedge_rwtd_lora_stage1`
- Total runs in master table: 12
- Runs with AP metric: 10

## Best AP per domain

- `wildbsds_diffedge_rwtd_loraC_16shot` (Wild-BSDS): AP=0.565426, shots=16, status=running
- `synthetic_diffedge_rwtd_loraC_64shot` (Synthetic): AP=0.484539, shots=64, status=done

## Included plots

- `plots/ap_vs_shots.png`
- `plots/delta_vs_zero_shot.png`
- `plots/f1_vs_shots.png`
- `plots/pr_curve_bsds_wild.png`
- `plots/pr_curve_synthetic.png`
- `plots/status_overview.png`

## Qualitative selections

- Wild-BSDS:
  - Preview ids: `207, 387, 175, 205, 216`
  - Runs: `wildbsds_diffedge_rwtd_zs, wildbsds_diffedge_rwtd_loraC_1shot, wildbsds_diffedge_rwtd_loraC_4shot, wildbsds_diffedge_rwtd_loraC_16shot`
  - Grid: `qualitative/bsds_wild_preview_grid.png`
- Synthetic:
  - Preview ids: `85, 441, 96, 244, 181`
  - Runs: `synthetic_diffedge_rwtd_zs, synthetic_diffedge_rwtd_loraC_1shot, synthetic_diffedge_rwtd_loraC_4shot, synthetic_diffedge_rwtd_loraC_16shot, synthetic_diffedge_rwtd_loraC_32shot, synthetic_diffedge_rwtd_loraC_64shot`
  - Grid: `qualitative/synthetic_preview_grid.png`
