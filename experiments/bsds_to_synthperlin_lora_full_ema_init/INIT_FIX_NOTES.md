# BSDS init fix (LoRA on synthetic perlin)

## What was wrong
- Original run loaded `checkpoints/bsds.pt` via `model` key only.
- In this checkpoint, `model` contains only `scale_factor`; full weights are under `ema` (`ema_model.*`).
- Result in original run: `Restored from .../checkpoints/bsds.pt with 1394 missing and 0 unexpected keys`.

## What was prepared
- Converted checkpoint created at:
  - `checkpoints/bsds_ema_unwrapped.pt`
- It unwraps `ema_model.* -> *` and stores under `model` key.
- Verified with dry-run:
  - `Restored from .../bsds_ema_unwrapped.pt with 0 missing and 0 unexpected keys`.

## Configs wired to corrected base
- `resolved_train_cfg.yaml`
  - `model.ckpt_path` points to `checkpoints/bsds_ema_unwrapped.pt`
  - `lora.base_ckpt_ref` points to `checkpoints/bsds_ema_unwrapped.pt`
- `resolved_eval_cfg.yaml`
  - `sampler.ckpt_path` points to `checkpoints/bsds_ema_unwrapped.pt`

## Relaunch
/home/nada/PycharmProjects/DiffusionEdge/.venv/bin/accelerate launch train_cond_ldm.py \
  --cfg /home/nada/PycharmProjects/DiffusionEdge/runs/diffedge_rwtd_lora_stage1/bsds_to_synthperlin_lora_full_ema_init/resolved_train_cfg.yaml \
  --lora_preset relation_qkvo_plus_initconv --lora_only --assert_lora_used

## Eval runtime note
- Per eval event runs 64 val batches at batch_size=16 => 1024 exported val predictions
  + RWTD 8 batches at batch_size=8.
- Edge metric pass over 1024 files is expected to be long.
