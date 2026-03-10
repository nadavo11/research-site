# DiffusionEdge RWTD LoRA Stage-1 Results

Primary metric: `edge/AP` (or run-specific `primary_metric`).

## synthetic_diffedge

| run_id | lora_shots | status | primary_metric_value | delta_from_zero_shot | checkpoint_out | error_message |
|---|---:|---|---:|---:|---|---|
| synthetic_diffedge_rwtd_zs | 0 | done | 0.413314 | 0.000000 |  |  |
| synthetic_diffedge_rwtd_loraC_1shot | 1 | done | 0.421151 | 0.007837 | /home/nada/PycharmProjects/DiffusionEdge/runs/diffedge_rwtd_lora_stage1/synthetic_diffedge_rwtd_loraC_1shot/checkpoints/lora-best_step1700.pt |  |
| synthetic_diffedge_rwtd_loraC_4shot | 4 | done | 0.449453 | 0.036139 | /home/nada/PycharmProjects/DiffusionEdge/runs/diffedge_rwtd_lora_stage1/synthetic_diffedge_rwtd_loraC_4shot/checkpoints/lora-best_step1500.pt |  |
| synthetic_diffedge_rwtd_loraC_16shot | 16 | done | 0.477241 | 0.063927 | /home/nada/PycharmProjects/DiffusionEdge/runs/diffedge_rwtd_lora_stage1/synthetic_diffedge_rwtd_loraC_16shot/checkpoints/lora-best_step3400.pt |  |
| synthetic_diffedge_rwtd_loraC_32shot | 32 | done | 0.472831 | 0.059517 | /home/nada/PycharmProjects/DiffusionEdge/runs/diffedge_rwtd_lora_stage1/synthetic_diffedge_rwtd_loraC_32shot/checkpoints/lora-best_step3800.pt |  |
| synthetic_diffedge_rwtd_loraC_64shot | 64 | done | 0.484539 | 0.071225 | /home/nada/PycharmProjects/DiffusionEdge/runs/diffedge_rwtd_lora_stage1/synthetic_diffedge_rwtd_loraC_64shot/checkpoints/lora-best_step3800.pt |  |

## wild_bsds_diffedge

| run_id | lora_shots | status | primary_metric_value | delta_from_zero_shot | checkpoint_out | error_message |
|---|---:|---|---:|---:|---|---|
| wildbsds_diffedge_rwtd_zs | 0 | done | 0.565264 | 0.000000 |  |  |
| wildbsds_diffedge_rwtd_loraC_1shot | 1 | done | 0.561164 | -0.004100 | /home/nada/PycharmProjects/DiffusionEdge/runs/diffedge_rwtd_lora_stage1/wildbsds_diffedge_rwtd_loraC_1shot/checkpoints/lora-best_step100.pt |  |
| wildbsds_diffedge_rwtd_loraC_4shot | 4 | done | 0.556727 | -0.008537 | /home/nada/PycharmProjects/DiffusionEdge/runs/diffedge_rwtd_lora_stage1/wildbsds_diffedge_rwtd_loraC_4shot/checkpoints/lora-best_step100.pt |  |
| wildbsds_diffedge_rwtd_loraC_16shot | 16 | running | 0.565426 | 0.000162 |  |  |
| wildbsds_diffedge_rwtd_loraC_32shot | 32 | pending |  |  |  | No metrics artifact found |
| wildbsds_diffedge_rwtd_loraC_64shot | 64 | pending |  |  |  | No metrics artifact found |

