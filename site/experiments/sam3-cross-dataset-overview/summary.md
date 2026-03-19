This page compares the published Coarse Feature Clustering flavors under the shared
        `architexture_binary_v1` evaluator (`mIoU`, `ARI`).

        Across the paired datasets, `flip_averaged` now leads RWTD (`0.840` mIoU, `0.726` ARI)
        and STLD (`0.753` mIoU, `0.619` ARI), with the largest
        gain on STLD (`+0.285` mIoU, `+0.434` ARI).
        CAID still favors `coarse_only` (`0.674` mIoU, `0.496` ARI).

CSTD is currently published as `flip_averaged` only (`0.745` mIoU, `0.578` ARI), so it expands coverage without a coarse-only delta yet.
