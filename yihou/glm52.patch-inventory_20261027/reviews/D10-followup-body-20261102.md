## Problem and root cause

The DCP1-prefill to DCP-N-decode transfer path treats draft KV as replicated over widened pages. The ROCm DSA/MTP backend merged in #44322 stripes its draft KV like the target instead. Using the replicated-draft geometry either rejects the draft entry or sends the wrong rows/head slice.

Our archived D10 (`a6bdc7894d`) fixed this on a branch containing #44322. Historical `pdBe-r1` passed two 29-case math checks and 8K/34K/96K passkey checks. Those results are not validation of this current-main port. Historical stack throughput used simulated MTP acceptance and is deliberately not cited as patch performance evidence.

## Fix

- Let the backend's draft-layout policy select target-style draft geometry for Mooncake DCP relayout.
- Send sharded draft entries with the target's owned token rows and pack; preserve existing replicated-draft behavior.
- Validate sharded draft item widths and reject unsupported NIXL/Mori sharded-draft transfers explicitly.
- Include the related D13/D16/D29 host-byte regressions for DP-attention/PP prefill fan-out, layer mapping, partial pages/chunks and replicated index-K pairing. The independent PP wait-event fix is tracked in #45524, not included here.

Rebased onto main `acd6f5570` on 2026-11-02, including merged #44322 and #43299. The production delta is unchanged. The canonical `dsa_draft_shards_under_dcp(config, dcp_size)` policy is now in main; the adapter delegates to it, and an additional CPU matrix exercises the real merged predicate for HIP/non-HIP, DCP1/DCP4 and DSA/non-DSA configs. The fallback for a missing policy remains conservative. This PR still does not lift the PD configuration guard.

## Validation

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/disaggregation/test_dsa_pd_dcp.py \
  test/registered/unit/disaggregation/test_dcp_pack.py \
  test/registered/unit/disaggregation/test_mooncake_transfer_batching.py
pre-commit run --hook-stage pre-commit --files \
  python/sglang/srt/disaggregation/common/conn.py \
  python/sglang/srt/disaggregation/mooncake/conn.py \
  python/sglang/srt/disaggregation/mori/conn.py \
  python/sglang/srt/disaggregation/nixl/conn.py \
  test/registered/unit/disaggregation/test_dsa_pd_dcp.py
```

Local CPU result: **43 passed, 61 subtests passed** at `1bcfc5214`. Transfers execute the actual relayout/send planner over host byte arrays; transport and GPU execution are not claimed.

## Draft blockers / dependencies (updated 2026-11-02)

1. #44322 merged on October 28. Its canonical policy and our adapter agree on the current base; this is no longer an external merge blocker.
2. #46826 must not enable ROCm PD DCP before this transfer contract is available. That is merge order, not a need to wait before reviewing either patch.
3. Resolve or explicitly exclude the archived DP2×DCP2 CUDA-graph garbling. Match historical `pdBe-r1` coverage to the current production contract and supplement only genuinely uncovered MTP/prefix/PP cases. No blanket rerun of every historical experiment is required for review. Separate PP producer ordering remains #45524; merged #43299 covers a distinct synchronization path.
4. Maintainer CI and review. Upstream functional tests remain gated; the host-byte tests do not execute a GPU transport or establish combined-stack model accuracy.

<!-- pr-states:start -->
---
### CI States

Latest PR Test (Base): <!-- slot:pr-test:start -->:x: [Run #40493344568](https://github.com/sgl-project/sglang/actions/runs/40493344568)<!-- slot:pr-test:end -->
Latest PR Test (Extra): <!-- slot:pr-test-extra:start -->:x: [Run #40493344138](https://github.com/sgl-project/sglang/actions/runs/40493344138)<!-- slot:pr-test-extra:end -->
Latest PR Test (AMD ROCm 10): <!-- slot:pr-test-amd:start -->:x: [Run #40493344191](https://github.com/sgl-project/sglang/actions/runs/40493344191)<!-- slot:pr-test-amd:end -->
<!-- pr-states:end -->

All applicable local pre-commit hooks and the new-head upstream lint passed. Upstream test execution is still gated by the maintainer-controlled `run-ci` label; latest PR title check passed.
