## Problem and fix

Current DSA DCP configuration rejects every PD role. A DCP1 prefill feeding a ROCm DCP decode becomes valid once the ROCm backend and sharded-draft relayout are available. Permit only that decode role on HIP; continue rejecting a DCP prefill and non-HIP PD DCP. Preserve HiSparse/HiCache and branching-speculation restrictions.

This is the narrow D11 configuration change (`228263d116`) rebased onto main `acd6f5570` on 2026-11-02. It does not import the backend implementation or change backend defaults. It belongs in SGLang, not an Infera example launcher.

## Validation

```bash
PYTHONPATH=python python -m pytest -q test/registered/unit/test_dsa_pd_dcp_config.py test/registered/unit/test_dsa_dcp_backend_defaults.py
pre-commit run --hook-stage pre-commit --files python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py test/registered/unit/test_dsa_pd_dcp_config.py
```

**12 passed, 13 subtests passed** across the PD policy tests and the merged backend-default/constraint suite; all applicable lint passed. New tests check implicit Triton selection and the a2a Q-rep default for ROCm PD decode, in addition to HIP/non-HIP, each role, DCP1 prefill and preserved incompatible-feature guards. These are configuration tests, not backend or transport execution. Head `21f8e688f`; the production delta is unchanged.

Historical combined-branch evidence: `pdBe-r1` passed two 29-case math checks and 8K/34K/96K passkeys. This is not a fresh result for this port. No stack-throughput claim: those throughput runs used simulated MTP acceptance.

## Draft blockers / merge order (updated 2026-11-02)

1. #44322 is merged. Its ROCm Triton backend default is now in this base and covered by the new PD-specific CPU tests; the old TRT-LLM default blocker is resolved.
2. Sharded draft PD transfer contract #46824 must land before this guard is lifted. Both PRs can be reviewed in parallel; enabling this configuration without its transport is not safe.
3. Resolve or explicitly exclude the historical DP2×DCP2 graph-correctness gap and map `pdBe-r1` evidence onto the combined production contract. Supplement only missing MTP/prefix/PP cases. No blanket fresh full-model rerun is asserted. PP producer ordering remains the separate #45524; #43299 is merged but does not replace it.
4. Maintainer-enabled CI and review.

<!-- pr-states:start -->
---
### CI States

Latest PR Test (Base): <!-- slot:pr-test:start -->:x: [Run #40493630200](https://github.com/sgl-project/sglang/actions/runs/40493630200)<!-- slot:pr-test:end -->
Latest PR Test (Extra): <!-- slot:pr-test-extra:start -->:x: [Run #40493630263](https://github.com/sgl-project/sglang/actions/runs/40493630263)<!-- slot:pr-test-extra:end -->
Latest PR Test (AMD ROCm 10): <!-- slot:pr-test-amd:start -->:x: [Run #40493630173](https://github.com/sgl-project/sglang/actions/runs/40493630173)<!-- slot:pr-test-amd:end -->
<!-- pr-states:end -->
