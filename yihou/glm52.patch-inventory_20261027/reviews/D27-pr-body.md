## Problem and root cause

An eager IDLE DP-attention rank has padding rows but no valid request/page-table rows for the DSA indexer. Main `8a455f1` returns invalid top-k rows only on CUDA, so HIP continues into paged MQA/top-k with idle metadata. Our historical DP2×DCP2 NextN run `mixrun R18` faulted in `_topk_transform_v2_paged` (localized with `AMD_SERIALIZE_KERNEL=3`). Extending the idle guard to HIP removed that fault in the archived branch (`da6e383513`).

## Fix

Apply the existing eager-IDLE shortcut on HIP as well as CUDA. Preserve the physical padded shape, invalid indices, top-k broadcast/capture handling, and the normal path during graph capture or active forwards. This is independent of ROCm DCP enablement in #44322 and distinct from active draft-attention-backend initialization in #45156.

## Validation

```bash
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_indexer_idle.py
pre-commit run --hook-stage pre-commit --files python/sglang/srt/layers/attention/dsa/dsa_indexer.py test/registered/unit/layers/attention/test_dsa_indexer_idle.py
```

CPU: **4 passed, 20 subtests passed**. The regression executes the actual forward entry point on padded idle inputs for both mocked platform flags; the additional matrix checks active/capture controls. These are CPU dispatch/contract tests, not GPU-kernel or distributed-collective validation.

## Draft blockers

- Fresh AMD distributed DP/NextN eager-IDLE and graph-capture regression on the final main-based revision; upstream CI/review.
- The archived DP2×DCP2 graph stack still had unrelated, unbisected garbled outputs. Historical throughput used simulated MTP acceptance, so no stack throughput or overall model-correctness claim is made here.
