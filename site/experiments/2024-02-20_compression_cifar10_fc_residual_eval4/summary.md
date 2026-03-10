# Pilot Summary

This pilot page documents one completed legacy compression run slice from the `Variable-Rate-Image-Inference-RNN` project using existing `test_imgs` outputs (`prova_1..prova_4`).

What was tested:
- residual fully connected patch model outputs on fixed CIFAR-10 test samples.

What changed:
- no retraining in this step; this page standardizes reporting around existing outputs.

Result:
- average PSNR is below a simple mean-color baseline on this fixed slice.
- visual review shows persistent blur and edge attenuation.

Verdict:
- report format works, but this run is not a performance milestone and should be treated as a baseline-quality reference.
