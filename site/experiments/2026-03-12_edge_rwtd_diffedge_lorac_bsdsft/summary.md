# Summary

## What was tested
RWTD edge detection using DiffusionEdge with LoRA preset C, trained on synthetic Perlin-stitched textures from a BSDS-pretrained checkpoint with EMA initialization.

## Why
Full-data LoRA training (no few-shot restriction) from an EMA-initialized BSDS checkpoint, targeting improved edge quality on real-world textures via synthetic pretraining.

## What changed
- Source checkpoint: BSDS EMA unwrapped (`bsds_ema_unwrapped.pt`)
- LoRA config: rank=128, alpha=128, dropout=0.05, preset `relation_qkvo_plus_initconv`
- Training: 4000 steps on synthetic Perlin-stitched data

## Result
- AP: `0.5878` (delta vs 64-shot synthetic: `+0.1033`)
- ODS F1: `0.6630` (delta vs 64-shot: `+0.0090`)
- OIS F1: `0.6920` (delta vs 64-shot: `+0.0373`)

## Verdict
Milestone — highest AP achieved in this comparison group. Full-data training with EMA init substantially improves AP over the 64-shot variant.
