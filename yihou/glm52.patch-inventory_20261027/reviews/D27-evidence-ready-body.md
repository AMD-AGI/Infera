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

## Historical evidence and review readiness

The production added/removed lines are identical to archived commit `da6e383513` (cherry-picked from `fdf73b6fb2`); the current port adds CPU coverage of active and capture controls. The CUDA behavior is unchanged.

The original archive's `working_process.md`, Phase 6, localizes the idle NextN failure in R16/R17. With this fix, R18 passed the 5-case check and 40-case gate. A later 4×1500 request run exposed a separate MQA row-count failure, traced in R19/R23 to in-place host-list padding; R24/R25 passed after that separate fix, now covered upstream by #45796. The R18 evidence is for the idle-indexer fault, not proof that every issue in that stack was fixed.

Ready for review based on that hardware investigation, an unchanged production delta, and the current-main CPU contract matrix. A new full distributed model run is not a default prerequisite for this narrow guard port. The historical branch's remaining graph-output issue is disclosed and is not claimed resolved here.

Maintainer-enabled CI and review remain merge requirements. No fresh GPU result at this exact PR head, performance gain, or complete DCP-stack correctness is claimed.


<!-- pr-states:start -->
---
### CI States

Latest PR Test (Base): <!-- slot:pr-test:start -->:x: [Run #40492640946](https://github.com/sgl-project/sglang/actions/runs/40492640946)<!-- slot:pr-test:end -->
Latest PR Test (Extra): <!-- slot:pr-test-extra:start -->:x: [Run #40492640651](https://github.com/sgl-project/sglang/actions/runs/40492640651)<!-- slot:pr-test-extra:end -->
Latest PR Test (AMD ROCm 10): <!-- slot:pr-test-amd:start -->:x: [Run #40492640626](https://github.com/sgl-project/sglang/actions/runs/40492640626)<!-- slot:pr-test-amd:end -->
<!-- pr-states:end -->