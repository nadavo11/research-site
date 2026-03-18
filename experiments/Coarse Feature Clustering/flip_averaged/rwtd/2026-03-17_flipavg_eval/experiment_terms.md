# SAM-3 Automatic Mask Comparison Terms

## Run Scope

- Command family: `eval-sam3-auto`
- Dataset: `aviadcohz/RWTD`
- Split: `test`
- Requested variant setting: `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only`
- Executed variant(s): `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only`
- Model: `facebook/sam3`
- Device request: `cuda`
- Save visuals: `True`
- Limit: `None`
- Failure policy: `abort`
- Official SAM checkpoint override: `None`

## Core Terms

- RWTD crop: one RGB image crop with two annotated texture regions (`texture_a_mask`, `texture_b_mask`) and one derived or stored boundary mask.
- `all` dataset view: concatenates the public RWTD `train` and `test` splits in that order.
- Raw predicted mask set: the unordered conceptual set of binary masks emitted by the active variant before region-to-GT assignment.
- Rough mask: the pre-SAM-refinement binary mask produced directly by feature clustering or feature-margin thresholding.
- Refined mask: the binary mask selected from the official Meta `sam3` mask-prompt path after feeding a rough mask as `input_masks` / mask prompt.
- Permutation-invariant assignment: for unlabeled 2-cluster variants, both `(A->texture_a, B->texture_b)` and `(B->texture_a, A->texture_b)` are scored and the better one is reported.
- Supplementary comparison track: these SAM-3 automatic-mask runs are not the TextureSAM paper baseline and should be read as a separate ablation track.
- Default evaluation contract: `architexture_binary_v1`. Every variant now exports `eval_miou` / `eval_ari` as the shared repo-wide comparison view.

## Standard Per-Sample Row Fields

- Common fields written for every successful SAM-3 auto sample: `variant`, `split`, `sample_index`, `crop_name`, `evaluation_view`, `eval_miou`, `eval_ari`, `num_predicted_masks`, `miou`, `ari`, `miou_agg`, `texture_a_best_iou`, `texture_b_best_iou`, `texture_a_agg_iou`, `texture_b_agg_iou`, `texture_a_overlap_mask_count`, `texture_b_overlap_mask_count`, `mask_score_mean`, and `mask_score_median`.
- `mask_score_mean` / `mask_score_median`: arithmetic summaries of the retained SAM mask scores for the final raw mask set produced by the active variant.
- Feature variants append additional protocol-specific fields such as selected support ids, assignment metadata, feature-level names, coarse-mask sizes, refined overlap diagnostics, and explicit status flags.

## Variant Standard

### `default`

1. Use the Transformers `facebook/sam3` automatic mask-generation pipeline with `points_per_crop=32` and `stability_score_thresh=0.95`.
2. Keep the returned raw mask set and score it directly with the repository automatic-mask metrics.
3. Do not use text, points, boxes, proposals from other methods, or official Meta refinement.

### `dense`

1. Use the same Transformers automatic mask-generation path with `points_per_crop=64` and `stability_score_thresh=0.2`.
2. Keep the denser raw mask set and score it directly with the repository automatic-mask metrics.
3. Do not apply any additional feature-based refinement in this baseline.

### `feature_mask`

1. Run the existing `dense` automatic-mask generator once and cache its raw proposals.
2. Extract official Meta `sam3` dense image features from the current image state.
3. If at least two usable dense masks exist, score adjacent support pairs using feature distance plus boundary contact.
4. If only one usable dense mask exists, synthesize a local complement region instead of silently falling back to `dense`.
5. Mean-pool feature prototypes over eroded support regions, build a local feature-similarity margin, and threshold it into coarse mask prompts.
6. Feed the coarse prompt(s) into the official Meta mask-prompt path and keep the final raw 2-mask partition for evaluation.
7. Current settings: base variant=`dense`, prototype erosion radius=`2`, local dilation radius=`8`, single-mask outer dilation radius=`24`, min region pixels=`128`, preferred region pixels=`1024`, min prototype pixels=`32`, min boundary contact=`16`, margin threshold=`0.05`, refine confidence threshold=`0.0`.

### `feature_cluster_global`

1. Extract official Meta `sam3` dense image features from `backbone_fpn` and upsample the highest-resolution useful level to image resolution.
2. L2-normalize the per-pixel feature vectors and run a global 2-way clustering over the full image with no appended spatial coordinates.
3. Convert the binary label map into two disconnected-allowed rough masks over the whole image.
4. Feed each rough mask independently into the official Meta mask-prompt path.
5. Collect all candidate masks returned for prompt A and prompt B, then choose the final A/B pair jointly so prompt agreement is rewarded and A/B overlap is penalized.
6. Evaluate the final refined pair permutation-invariantly against the two GT regions.
7. Current settings: feature source=`highest_resolution_backbone_fpn`, k-means max iterations=`25`, k-means convergence tolerance=`0.0001`, apply refinement=`True`, refine confidence threshold=`0.0`.

### `feature_cluster_coarse_to_fine_global`

1. Extract the full official Meta `sam3` multiscale `backbone_fpn` feature pyramid and keep every level at its native spatial resolution.
2. Order feature levels from coarsest to finest by spatial area.
3. On the coarsest level only, flatten the feature grid, L2-normalize the vectors, and run a global 2-way clustering with no spatial coordinates.
4. Upsample the current label map to the next finer level with nearest-neighbor interpolation.
5. Build a local uncertainty band from 4-neighborhood label changes, dilate that band by a small radius, and freeze labels outside the band.
6. Recompute cluster prototypes from the current labels and update only the uncertain pixels using finer-level cosine similarity to the two prototypes.
7. Repeat the boundary-band refinement for every finer level, then convert the final label map into two rough masks.
8. Feed both final rough masks into the official Meta mask-prompt path, collect prompt-aligned candidates, and choose the final refined A/B pair jointly with an overlap penalty.
9. Evaluate the final refined pair permutation-invariantly against the two GT regions.
10. Current settings: feature source=`backbone_fpn`, k-means max iterations=`25`, k-means convergence tolerance=`0.0001`, boundary band radius=`2`, refinement iterations per level=`2`, refine confidence threshold=`0.0`.

### `feature_cluster_coarse_to_fine_global_pooled_init`

1. Extract the same official Meta `sam3` multiscale `backbone_fpn` feature pyramid used by `feature_cluster_coarse_to_fine_global` and sort it from coarsest to finest by spatial area.
2. Keep the original coarsest feature map only for diagnostics, then spatially average-pool that coarsest level before any clustering.
3. L2-normalize the pooled coarsest features and run the same global 2-way clustering on that pooled grid with no spatial coordinates.
4. Upsample the pooled label map back to the native coarsest feature resolution with nearest-neighbor interpolation.
5. Reuse the unchanged boundary-band refinement logic on every finer feature level.
6. Feed the final pooled-init rough masks into the official Meta mask-prompt path and choose the final refined A/B pair jointly with the same prompt-alignment and overlap-penalty logic.
7. Evaluate the pooled-init refined pair permutation-invariantly against the two GT regions.
8. Current settings: feature source=`backbone_fpn`, pooled init kernel=`3`, pooled init stride=`3`, k-means max iterations=`25`, k-means convergence tolerance=`0.0001`, boundary band radius=`2`, refinement iterations per level=`2`, refine confidence threshold=`0.0`.

### `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only`

1. Extract the same official Meta `sam3` `backbone_fpn` pyramid and sort feature levels from coarsest to finest by spatial area.
2. Average-pool only the coarsest feature level before the initial global 2-way clustering.
3. L2-normalize the pooled coarsest features, cluster the pooled grid globally into 2 unlabeled groups, and upsample that pooled label map back to the native coarsest resolution.
4. Upsample the pooled coarsest partition directly to image resolution and treat those two coarse masks as the final raw prediction set.
5. Do not run finer-level boundary-band refinement and do not send the coarse masks back through the official Meta SAM mask-prompt path in this ablation.
6. Evaluate the unlabeled 2-mask partition permutation-invariantly against the two GT regions.
7. Current settings: feature source=`backbone_fpn`, pooled init kernel=`3`, pooled init stride=`3`, k-means max iterations=`25`, k-means convergence tolerance=`0.0001`.

### `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only`

1. Extract the raw coarsest official Meta `sam3` feature map and build four content-preserving views: `identity`, `hflip`, `vflip`, and `hvflip`.
2. Undo each flipped coarsest feature map back into the original image coordinates and average the four restored coarsest feature tensors.
3. L2-normalize that flip-averaged coarsest feature map, average-pool it on the coarsest level, and run the same global 2-way clustering on the pooled grid.
4. Upsample the pooled label map back to the native coarsest resolution, then upsample that partition directly to image resolution.
5. Treat the two image-space coarse masks as the final raw prediction set.
6. Do not run finer-level boundary-band refinement and do not send the coarse masks through the official Meta SAM mask-prompt path in this ablation.
7. Evaluate the unlabeled 2-mask partition permutation-invariantly against the two GT regions.
8. Current settings: feature source=`backbone_fpn`, pooled init kernel=`3`, pooled init stride=`3`, k-means max iterations=`25`, k-means convergence tolerance=`0.0001`.

### `feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only`

1. Extract the raw coarsest official Meta `sam3` feature map before any spatial pooling and keep it as the diagnosis surface.
2. Quantify raw label positionality by fitting coordinate-only separators to the raw 2-cluster map: vertical threshold, horizontal threshold, affine half-plane, and quadratic basis.
3. Quantify raw feature positionality by regressing the flattened coarsest feature vectors on the normalized coordinate basis `[1, x, y, x^2, xy, y^2]` and reporting mean/max per-channel `R^2`.
4. Run flip tests on the raw coarsest feature map using `identity`, `hflip`, `vflip`, and `hvflip`; unflip the resulting features back into the original image coordinates before comparing them.
5. Run null-image controls on a constant gray image, a weak-noise image, and a strongly blurred image of the same size.
6. Remove the coordinate subspace before clustering by projecting the raw coarsest feature vectors onto the coordinate basis and subtracting that projection, then L2-normalize the debiased features.
7. Average-pool the debiased coarsest features, run the same global 2-way clustering, upsample back to the native coarsest resolution, and then upsample directly to image resolution.
8. Do not run finer-level refinement and do not send the resulting masks into the official Meta SAM mask-prompt path in this debiasing ablation.
9. Also run two comparison branches: the baseline pooled-init coarse-only branch and a flip-averaged feature branch built from `mean(untransform(E(T(I))))` over `{id, hflip, vflip, hvflip}`.
10. Use the projection-removed branch as the active final prediction set, but export per-image and dataset-level comparisons for the baseline and flip-averaged branches as diagnostics.
11. Current settings: feature source=`backbone_fpn`, pooled init kernel=`3`, pooled init stride=`3`, k-means max iterations=`25`, k-means convergence tolerance=`0.0001`, null noise seed=`0`, null noise std=`4.0`, null blur radius=`12.0`.

### `feature_cluster_coarse_to_fine_global_pooled_init_direct`

1. Extract the same official Meta `sam3` `backbone_fpn` pyramid and sort feature levels from coarsest to finest by spatial area.
2. Average-pool only the coarsest feature level before the initial global 2-way clustering.
3. L2-normalize the pooled features, cluster the pooled grid globally into 2 unlabeled groups, and upsample that pooled label map back to the native coarsest resolution.
4. Upsample the coarsest label map directly to image resolution and use that rough 2-mask partition as the SAM mask prompt input.
5. Do not run any finer-level boundary-band refinement in this ablation.
6. Still run the official Meta mask-prompt refinement path once per coarsest cluster mask and choose the final refined A/B pair jointly.
7. Evaluate the final unlabeled 2-mask partition permutation-invariantly against the two GT regions.
8. Current settings: feature source=`backbone_fpn`, pooled init kernel=`3`, pooled init stride=`3`, k-means max iterations=`25`, k-means convergence tolerance=`0.0001`.

### `mask_prompt_invariance_control`

1. This is a designated side experiment, not a mainline benchmark baseline.
2. Extract the same coarsest `backbone_fpn` feature level used by the pooled-init ablations.
3. Build three prompt branches that all go through the same official Meta SAM mask-prompt refinement path: a dirty raw coarsest prompt, a cleaner pooled coarsest prompt, and a deterministic random prompt control with the same label balance as the pooled prompt.
4. Do not run finer-level feature refinement in this side experiment; the point is to isolate prompt cleanliness, not multiscale updating.
5. Refine all three prompt partitions through SAM, align raw/random final partitions to the pooled branch label order for comparison, and keep the pooled branch as the canonical branch for the standard automatic-mask fields.
6. Report the side-experiment comparisons that matter: dirty-vs-clean prompt similarity, dirty-vs-clean final similarity, random-vs-clean prompt similarity, random-vs-clean final similarity, and per-branch GT scores.
7. Save visuals that show the three prompt partitions directly beside the three final SAM outputs so prompt insensitivity is visible at a glance.
8. Current settings: feature source=`backbone_fpn`, pooled init kernel=`3`, pooled init stride=`3`, k-means max iterations=`25`, k-means convergence tolerance=`0.0001`.

### `both`

1. `both` is intentionally narrow in this repository.
2. It executes only `default` and `dense` and writes separate subdirectories for those two legacy automatic-mask baselines.
3. It does not automatically include `feature_mask`, `feature_cluster_global`, `feature_cluster_coarse_to_fine_global`, `feature_cluster_coarse_to_fine_global_pooled_init`, `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only`, `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only`, `feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only`, `feature_cluster_coarse_to_fine_global_pooled_init_direct`, or `mask_prompt_invariance_control`.

## Metric Standard

- `eval_miou` / `eval_ari`: the repo-wide default ArchiTexture-style evaluator. Raw-mask variants first collapse into a GT-overlap aggregated binary partition; unlabeled 2-mask variants use permutation-invariant assignment and then score that binary partition.
- `miou`: non-aggregated mean IoU from one-to-one IoU matching between the raw predicted mask set and the two GT texture regions. Higher is better. Range `[0, 1]`.
- `ari`: Adjusted Rand Index between GT region labels and the pixelwise partition induced by raw predicted-mask memberships. Higher is better. Range `[-1, 1]`, with practical values usually in `[0, 1]` here.
- `miou_agg`: aggregated mean IoU after unioning every raw predicted mask with non-empty overlap against each GT region. Higher is better. Range `[0, 1]`.
- `texture_a_overlap_mask_count` / `texture_b_overlap_mask_count`: number of raw masks contributing to the aggregated texture-A / texture-B region.
- `assignment_direct_*` and `assignment_swapped_*`: unlabeled 2-cluster diagnostics reported for the feature-cluster variants before the final permutation-invariant choice is made.
- `refined_pair_overlap_iou`: IoU overlap between the final selected refined A and refined B masks for feature-cluster variants. Lower is better; values near `1` indicate collapse onto the same region.

## Visual Artifact Standard

- `prediction.json`: one per-sample JSON object for `predict-sam3-auto` containing the exact per-sample row written by the variant.
- `prediction.png`: one per-sample visual panel for `predict-sam3-auto`.
- `per_sample_metrics.csv`: one flattened row per evaluated sample for `eval-sam3-auto`.
- `summary.json` / `summary.md`: aggregate metrics, failure counts, and primary metric selection for the current run.
- `visuals_manifest.jsonl`: one machine-readable record per saved PNG with the full wrapped footer text and per-sample metric fields.
- Default / dense visuals: input, GT, raw categorical proposals, and aggregated A/B view.
- `feature_mask` visuals: input, GT, dense proposals, selected support pair, coarse feature prior, and final raw A/B partition.
- `feature_cluster_global` visuals: input, GT, global 2-cluster map, rough masks, refined masks, and chosen permutation-aligned result.
- `feature_cluster_coarse_to_fine_global` visuals: input, GT, coarsest cluster map, one panel per refined feature level, final rough partition before SAM, refined masks, and chosen permutation-aligned result.
- `feature_cluster_coarse_to_fine_global_pooled_init` visuals: input, GT, pooled coarsest cluster map, one panel per refined feature level, final rough partition before SAM, refined cluster A/B masks, and the chosen permutation-aligned result.
- `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only` visuals: input, GT, and the pooled coarsest partition only; the coarsest partition is already the final output in this ablation.
- `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only` visuals: input, GT, and the flip-averaged pooled coarsest partition only; that coarsest partition is already the final output in this ablation.
- `feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only` visuals: input, GT, raw coarsest cluster map, baseline pooled partition, projection-removed pooled partition, flip-averaged pooled partition, and the three null-image control maps.
- `feature_cluster_coarse_to_fine_global_pooled_init_direct` visuals: input, GT, pooled coarsest cluster map, final rough partition before SAM, SAM-refined cluster A/B masks, and the chosen permutation-aligned result.
- `mask_prompt_invariance_control` visuals: input, GT, dirty raw prompt partition, cleaner pooled prompt partition, deterministic random prompt control, and the three corresponding final SAM outputs with raw/random finals aligned to the pooled label order for comparison.

## Failure Semantics

- No SAM-feature variant silently falls back to another protocol when a required intermediate state is invalid.
- `feature_mask` raises explicit failures when it cannot build a valid support pair or complement, coarse prompt, or refined mask; on `predict-sam3-auto` it may still save dense-only diagnostics when the dense proposals exist.
- `feature_cluster_global` raises explicit failures when feature extraction, global clustering, prompt refinement, or joint A/B selection fails.
- `feature_cluster_coarse_to_fine_global` raises explicit failures when multiscale features are missing, coarsest clustering collapses, finer-level prototype refinement cannot proceed, or final SAM refinement fails.
- `feature_cluster_coarse_to_fine_global_pooled_init` raises explicit failures when coarsest pooling is invalid, pooled coarsest clustering collapses, the reused finer-level refinement cannot proceed, or final SAM refinement fails.
- `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only` raises explicit failures when coarsest pooling is invalid or the pooled coarsest partition collapses into an empty binary output.
- `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only` raises explicit failures when flipped coarsest feature extraction returns incompatible shapes, pooled clustering collapses after flip averaging, or the final coarse binary partition becomes empty.
- `feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only` raises explicit failures when raw coarsest features are missing, coordinate projection removal becomes degenerate, any diagnosis branch collapses into an empty binary output, or the null/flip control extractions return incompatible coarsest shapes.
- `feature_cluster_coarse_to_fine_global_pooled_init_direct` raises explicit failures when coarsest pooling is invalid or the pooled coarsest prompt cannot produce a valid prompt-refined binary output.
- `mask_prompt_invariance_control` raises explicit failures when any of the three prompt branches cannot be constructed or any of the shared SAM prompt-refinement passes fails; there is no fallback that drops the random control or replaces a failed branch.
- `failure_policy=skip` applies only to evaluation commands and records the failed sample in the run summary instead of aborting immediately.

## Backend And Dependency Assumptions

- `default` and `dense` use the Transformers `facebook/sam3` automatic mask-generation path.
- The SAM-feature variants rely on the official Meta `sam3` image backend because this repository needs access to the official multiscale image features and, for the prompt-based variants, the mask-prompt refinement hook exposed there.
- Missing official `sam3` imports, missing gated model access, invalid checkpoints, or unexpected tensor shapes are treated as explicit runtime failures, not hidden fallbacks.
