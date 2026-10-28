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

## Draft blockers

- Fresh distributed AMD/CUDA PD + DP-attention + EAGLE/MTP regression with a busy rank and an idle peer, including over-capture-size eager fallback. CPU tests cannot prove collective progress.
- Measure the temporary page-table memory cost of using static width for long contexts; check every consumer's masking contract on the final revision.
- Maintainer-enabled upstream CI and review. No simulated-acceptance throughput or combined-stack accuracy is used as patch evidence.

This replaces closed stale draft #33973 with a current-main port and registered regressions. GitHub rejects reopening that PR because its branch was force-pushed/recreated; #33973 remains closed and will point to this replacement. No competing open PR was found.
