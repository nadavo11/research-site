# GlaS Frozen SAM Mask Head Summary

- Run kind: `train`
- Variant: `fpn_2_only d64`
- Variant summary: Single-scale frozen mask head using only the coarsest SAM level `fpn_2` at the default projection/decoder width `d=64`.
- Split: `test`
- Selected levels: `fpn_2`
- Model family: `multiscale_mask_head`
- Head kind: `multiscale_mask_head`
- Trainable params: `90625`
- Mean direct foreground IoU: `0.813221`
- Mean Dice: `0.892288`
- Mean auxiliary partition mIoU: `0.803947`
- Mean auxiliary partition ARI: `0.644130`
- Mean train-set foreground IoU: `0.896835`
- Mean train-set Dice: `0.944962`
- Validation set metrics: `None`
- Checkpoint: `outputs/glas_binary/frozen_sam_mask_head/train_fpn_2_only_test_e40/checkpoint.pt`
