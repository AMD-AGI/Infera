## Problem and fix

Current DSA DCP configuration rejects every PD role. A DCP1 prefill feeding a ROCm DCP decode becomes valid once the ROCm backend and sharded-draft relayout are available. Permit only that decode role on HIP; continue rejecting a DCP prefill and non-HIP PD DCP. Preserve HiSparse/HiCache and branching-speculation restrictions.

This is the narrow D11 configuration change (`228263d116`) rebased onto main `8a455f1`. It does not import the backend implementation or change backend defaults. It belongs in SGLang, not an Infera example launcher.

## Validation

```bash
PYTHONPATH=python python -m pytest -q test/registered/unit/test_dsa_pd_dcp_config.py
pre-commit run --hook-stage pre-commit --files python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py test/registered/unit/test_dsa_pd_dcp_config.py
```

**3 passed, 9 subtests passed**; all applicable lint passed. CPU tests cover HIP/non-HIP, each role, DCP1 prefill and preserved incompatible-feature guards. They explicitly select Triton and test the configuration gate, not backend execution.

Historical combined-branch evidence: `pdBe-r1` passed two 29-case math checks and 8K/34K/96K passkeys. This is not a fresh result for this port. No stack-throughput claim: those throughput runs used simulated MTP acceptance.

## Draft blockers / merge order

1. ROCm DSA DCP backend/default selection from #44322 (coordinate broader #42618). Main still defaults to TRT-LLM here; this PR alone is not a working ROCm DCP implementation.
2. Sharded draft PD transfer contract in #46824 must land before this guard is lifted.
3. Revalidate the combined final stack on GPU with MTP, partial/cached prefixes and both DCP4 and DP2×DCP2. The archived DP2×DCP2 graph outputs remained garbled and unbisected. PP ordering also needs the appropriate #45524 fix.
4. Maintainer-enabled CI and review.
