## Problem and root cause

The DCP1-prefill to DCP-N-decode transfer path treats draft KV as replicated over widened pages. The ROCm DSA/MTP backend proposed in #44322 stripes its draft KV like the target instead. Using the replicated-draft geometry either rejects the draft entry or sends the wrong rows/head slice.

Our archived D10 (`a6bdc7894d`) fixed this on a branch containing #44322. Historical `pdBe-r1` passed two 29-case math checks and 8K/34K/96K passkey checks. Those results are not validation of this current-main port. Historical stack throughput used simulated MTP acceptance and is deliberately not cited as patch performance evidence.

## Fix

- Let the backend's draft-layout policy select target-style draft geometry for Mooncake DCP relayout.
- Send sharded draft entries with the target's owned token rows and pack; preserve existing replicated-draft behavior.
- Validate sharded draft item widths and reject unsupported NIXL/Mori sharded-draft transfers explicitly.
- Include the related D13/D16/D29 host-byte regressions for DP-attention/PP prefill fan-out, layer mapping, partial pages/chunks and replicated index-K pairing. The independent PP wait-event fix is tracked in #45524, not included here.

Rebased onto main `8a455f1`. That main lacks #44322's `dsa_draft_shards_under_dcp` policy: the adapter returns False when the policy is absent, preserving main's existing layout contract. It delegates to the canonical backend policy once present. This PR does not independently enable ROCm DCP or lift the PD configuration guard.

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

Local CPU result: **40 passed, 53 subtests passed**. Transfers execute the actual relayout/send planner over host byte arrays; transport and GPU execution are not claimed.

## Draft blockers / dependencies

1. Land or reconcile ROCm DSA DCP/MTP backend support (#44322; overlapping broader #42618). Verify the final shared layout policy, including the same-model/same-platform peer assumption; this PR must not be enabled against an incompatible decode layout.
2. Land the separate, constrained ROCm DCP PD configuration enablement (D11) only after this transfer contract is available.
3. Run real Mooncake DCP1-to-DCP4 and DP2×DCP2, with/without MTP, cached prefixes/chunks and PP layer ranges, on the final dependency stack. The archived DP2×DCP2 graph output remained garbled and unbisected; this PR does not clear it. PP producer ordering must include #45524 as appropriate.
4. Maintainer CI and review. Related merged #37709 handles replicated DSPARK drafts; it does not cover this sharded-draft layout. No competing implementation was found in the transport files of #42618.

<!-- pr-states:start -->
---
### CI States

Latest PR Test (Base): <!-- slot:pr-test:start -->:x: [Run #40493344568](https://github.com/sgl-project/sglang/actions/runs/40493344568)<!-- slot:pr-test:end -->
Latest PR Test (Extra): <!-- slot:pr-test-extra:start -->:x: [Run #40493344138](https://github.com/sgl-project/sglang/actions/runs/40493344138)<!-- slot:pr-test-extra:end -->
Latest PR Test (AMD ROCm 10): <!-- slot:pr-test-amd:start -->:x: [Run #40493344191](https://github.com/sgl-project/sglang/actions/runs/40493344191)<!-- slot:pr-test-amd:end -->
<!-- pr-states:end -->

All applicable local pre-commit hooks and the new-head upstream lint passed. Upstream test execution is still gated by the maintainer-controlled `run-ci` label; latest PR title check passed.
