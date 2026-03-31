# GlaS vs AutoSAM: Frozen SAM3 Feature Readout Study

- Rendered: `2026-03-31`
- Question: does weak training-free GlaS CFC imply that frozen SAM3 lacks gland information, or is the readout the main bottleneck?
- Current answer: dense supervision on frozen SAM3 features is already strong, and the best current frozen-feature run is unexpectedly coarse-only.

## Headline Numbers

- Training-free CFC baseline: `fg_iou=0.559870` `dice=0.703093` `eval_miou=0.576410` `eval_ari=0.254778`
- Frozen mask head all_scales d64: `fg_iou=0.803117` `dice=0.885918` `eval_miou=0.793207` `eval_ari=0.624593`
- Best current frozen-feature run (`fpn_2_only d128`): `fg_iou=0.824312` `dice=0.899436` `eval_miou=0.813555` `eval_ari=0.659685`
- AutoSAM reported paper reference: `fg_iou=0.870800` `dice=0.928200`

## Caveats

- AutoSAM values are reported paper references, not reproduced runs from this repo.
- A strict `224x224` protocol-matched rerun is still pending.
- The default-width coarse-only run is labeled explicitly as `fpn_2_only d64` on this page.
