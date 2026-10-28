Our GLM-5.2/ROCm PD audit found an independent instance of the batch-major padded-extend output bug fixed here. We will follow this PR instead of submitting our overlapping D23 patch.

In the archived DP2×DCP2 `mixrun R13`, padding `[1, rows, heads, dim]` on the batch dimension instead of the token dimension caused the 49-to-50-row path to fail. Our local fix (`b17da53725`) pads the token dimension and removed that crash. It also carried CPU regression coverage for the tensor/tuple result forms; please retain coverage of an auxiliary LSE result where that contract applies.

This is historical crash evidence, not a fresh GPU test of your head. Other DP2×DCP2 graph outputs remained garbled and unbisected, so eliminating this shape failure does not certify the whole DCP stack. Historical performance used simulated MTP acceptance; no stack throughput claim is made here. This generic padding fix should be evaluated independently of the ROCm DCP enablement in #44322.
