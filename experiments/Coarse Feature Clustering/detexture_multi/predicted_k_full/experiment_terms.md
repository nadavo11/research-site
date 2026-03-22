# DeTexture Multi-Region Benchmark Terms

## Run Scope

- Command family: `eval-detexture-multi`
- Dataset id: `detexture_ade20k_multi`
- Requested dataset root: `datasets/detexture_ADE20K`
- Expected local root form: either the directory containing `detecture_data/images` and `detecture_data/masks`, or `detecture_data/` itself.
- Executed variant: `feature_cluster_coarse_global_pooled_predicted_k`
- Variant summary: Predicted-K prompt-free pooled coarsest clustering on frozen SAM features using deterministic silhouette selection over a small K range.
- Model: `facebook/sam3`
- Device request: `cuda`
- Hardware compatibility standard: `detexture_multi_streaming_eval_v1`
- Sample loading mode: `streamed_iter` (the adapter does not preload all crops into memory).
- Dataset partition: full dataset order (no partitioning).
- Save visuals: `True`

## Dataset Semantics

- Each sample is one RGB crop image from `detecture_data/images` and one JPEG-compressed multi-region color mask from `detecture_data/masks`.
- GT is decoded into a single integer label map by deterministic weighted color clustering over the compressed RGB mask values.
- GT decode settings: criterion=`bic`, metric=`euclidean`, max clusters=`8`.
- Oracle K is the number of decoded GT regions in that sample.
- Valid-pixel evaluation currently includes all pixels in the decoded mask. This route does not silently drop pixels; any future exclusions must be reported explicitly in row metadata.

## Method Standard

1. Extract frozen SAM dense features using the same official coarsest FPN level used by the current binary pooled coarse-only method.
2. L2-normalize coarsest features per spatial location.
3. Average-pool the coarsest feature map with kernel=3 and stride=3.
4. Flatten pooled features to `[N, C]` and cluster them deterministically in cosine space.
5. Upsample the pooled label map back to native coarsest resolution, then to image resolution, with nearest-neighbor only.
6. Do not add prompts, proposal banks, graph reasoning, component cleanup, CRFs, appended coordinates, or learned refinement.

## Variant Definitions

- `feature_cluster_coarse_global_pooled_oracle_k`: use oracle K from the decoded GT region count; cluster raw coarsest pooled features directly.
- `feature_cluster_coarse_global_pooled_oracle_k_flipavg`: same oracle-K route, but flip-average coarsest features across `identity`, `hflip`, `vflip`, and `hvflip` before pooling/clustering.
- `feature_cluster_coarse_global_pooled_predicted_k`: choose K deterministically by silhouette score over a small candidate range on the pooled normalized feature vectors. This is a secondary ablation, not the main result route.

## Metric Standard

- `eval_miou`: Hungarian-matched mean IoU across predicted and GT labels on valid pixels. When `K_pred != K_gt`, zero-IoU padded assignment slots penalize count mismatch.
- `eval_ari`: pixel ARI on valid pixels.
- `eval_nmi`: normalized mutual information on valid pixels.
- `count_accuracy`: exact region-count match indicator, macro-averaged in summaries.
- `coverage`: valid-pixel fraction used in evaluation.

## Output Contract

- `prediction.json`, `prediction.png`, `predicted_label_map.npy`, `gt_label_map.npy`, and `valid_pixel_mask.npy` for `predict-detexture-multi`.
- `per_sample_metrics.csv`, `summary.json`, `summary.md`, `visuals_manifest.jsonl`, `visuals/*.png`, and `label_maps/*.npy` for `eval-detexture-multi`.
- Visual audit contract: `input|gt_labelmap|pred_labelmap|pred_boundaries`.

## Failure Semantics

- There is no silent fallback to the binary DeTexture route or any prompt-refined path.
- Missing dataset roots, missing image/mask pairs, shape mismatches, GT decode failure, degenerate clustering, or missing official SAM features are explicit failures.
- `failure_policy=skip` applies only to evaluation runs and records failed samples in the summary instead of aborting immediately.
