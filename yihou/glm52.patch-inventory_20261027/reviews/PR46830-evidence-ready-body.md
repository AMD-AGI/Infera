## Problem and root cause

DSA declares `needs_cpu_seq_lens=False`, but its eager fallback still reads GPU lengths on the host when the host mirror is absent. Under PD + DP-attention + EAGLE/MTP, busy and idle ranks can take different host-mirror branches. A blocking `.max().item()` or `.cpu()` on only some ranks can prevent them from issuing matching collectives and hang the group.

The two offending paths remain on main `8a455f1`: page-table width selection without a host mirror, and DRAFT_EXTEND_V2 host-mirror materialization. The original report observed a first-routed-request deadlock; the fresh evidence below is a CPU contract regression, not a new distributed deadlock reproduction.

## Fix

Use the static request-table width when a host length mirror is unavailable. Keep actual GPU sequence lengths for per-row bounds, and retain the narrower width when a host mirror is already present. DRAFT_EXTEND_V2 uses its supplied extend-length metadata without constructing unused host mirrors. Existing graph behavior is unchanged.

## Validation

```bash
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_eager_metadata_no_host_sync.py
pre-commit run --hook-stage pre-commit --files python/sglang/srt/layers/attention/dsa_backend.py test/registered/unit/layers/attention/test_dsa_eager_metadata_no_host_sync.py
```

Rebased CPU result: **3 passed, 2 subtests passed**. Unpatched main: **2 expected failures, 1 passed, 2 subtests passed**. Tests run the actual metadata entry point, reject host reads of the device-length inputs, inspect decode/draft page tables, and retain a host-mirror control. GPU helper kernels use CPU substitutes; these are dispatch/data-contract tests. All applicable lint passed.

## Historical hardware evidence and port comparison

The production added/removed lines are identical to the original #33973 diff; this port changes placement to match main and adds registered CPU regressions. The original investigation reported that the first-request deadlock cleared on 8×MI355X with GLM-5.2-FP8, PD + DP attention + EAGLE MTP. It also recorded MI300X narrow-versus-static-width top-k checks: real selections and scores matched with per-row masking; surplus -inf slots for sequences shorter than top-k were present at both widths. See the [original verification and its limitations](https://github.com/sgl-project/sglang/pull/33973).

Rechecked the current source contracts: DRAFT_EXTEND_V2 still uses the supplied extend lengths; prefix checks/ragged dispatch remain restricted to ordinary EXTEND, and the additional automatic prefill-selection use of seq_lens_sum is guarded by ForwardMode.EXTEND. The added CPU regression exercises the current metadata entry point without host reads and retains the host-mirror control. Identical patch lines do not establish identical whole-stack behavior, so the historical deployment and current CPU results are reported separately.

## Review scope and remaining coverage

Ready for review using the historical hardware evidence, unchanged production delta, current consumer audit and current-main regression coverage. A full fresh PD/DP/MTP hardware run at this head has not been performed. Broader AMD/CUDA regression, long-context memory qualification and maintainer-enabled CI remain follow-up coverage; no new throughput or whole-stack accuracy result is claimed.

This replaces closed stale draft #33973. GitHub refused reopening that PR; its discussion links here. Required upstream CI and maintainer review remain merge gates.


<!-- pr-states:start -->
---
### CI States

Latest PR Test (Base): <!-- slot:pr-test:start -->:x: [Run #40494799941](https://github.com/sgl-project/sglang/actions/runs/40494799941)<!-- slot:pr-test:end -->
Latest PR Test (Extra): <!-- slot:pr-test-extra:start -->:x: [Run #40494799911](https://github.com/sgl-project/sglang/actions/runs/40494799911)<!-- slot:pr-test-extra:end -->
Latest PR Test (AMD ROCm 10): <!-- slot:pr-test-amd:start -->:x: [Run #40494799953](https://github.com/sgl-project/sglang/actions/runs/40494799953)<!-- slot:pr-test-amd:end -->
<!-- pr-states:end -->