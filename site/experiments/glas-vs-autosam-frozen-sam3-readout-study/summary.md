# GlaS + MoNuSeg vs AutoSAM: Frozen SAM3 Readout Study

- Rendered: `2026-04-04`
- Question: across GlaS and MoNuSeg, do frozen SAM3 features already carry the right histology signal for an AutoSAM-style comparison, or is the main bottleneck the readout?
- Current answer: the frozen features look much stronger than the weakest readout suggested. GlaS still favors coarse-only, while MoNuSeg now improves sharply when a tiny mid-scale residual branch is added.

## Headline Numbers

- Training-free CFC baseline: `fg_iou=0.559870` `dice=0.703093` `eval_miou=0.576410` `eval_ari=0.254778`
- Frozen mask head all_scales d64: `fg_iou=0.803117` `dice=0.885918` `eval_miou=0.793207` `eval_ari=0.624593`
- Best GlaS readout-study run (`fpn_2_only d128`): `fg_iou=0.824312` `dice=0.899436` `eval_miou=0.813555` `eval_ari=0.659685`
- Closest current clean GlaS endpoint (`fpn_2_only d128 @ 224 + autosam aug`): `fg_iou=0.835445` `dice=0.905842`
- Strict MoNuSeg anchor (`fpn_2_only d128 @ 512x512`): `fg_iou=0.635201` `dice=0.776317`
- Best current MoNuSeg run (`fpn_2 + fpn_1` residual refine): `fg_iou=0.687613` `dice=0.814392` `eval_miou=0.789239` `eval_ari=0.653307`
- AutoSAM reported paper references: `GlaS fg_iou=0.870800` `dice=0.928200` | `MoNuSeg fg_iou=0.701700` `dice=0.824300`

## Caveats

- AutoSAM values are reported paper references, not reproduced runs from this repo.
- The strongest MoNuSeg run on this page is native-resolution, so the strict `512x512` anchor remains the clean protocol reference.
- The default-width coarse-only run is labeled explicitly as `fpn_2_only d64` on this page.
