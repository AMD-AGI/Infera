# Teammate poll log

First sighting of a problem is RECORDED only. Intervention happens on the next
poll if it is still unresolved.

## 2026-09-18 10:40 UTC — `image-hicache` — DONE, PASS

Built `infera-sglang:v0519-yihou-0917-nextnfix-hicache` on both nodes; four
runtime checks pass on each; the base's 7 DSA `.orig` files unchanged.
Independently spot-checked by the leader: the tag exists on both nodes
(`fd7220a57b7d` on 135, `972d8fd952e9` on 138 — different ids are expected,
each is a local layer over that node's own locally-built base, never pushed).

## 2026-09-18 10:50 UTC — `bench-preflight` — FIRST SIGHTING, recorded, no intervention

Leader's own read of the in-flight artifacts, not a teammate report:
`preflight/mooncakeperf/20260918T103940-1655187/0_1_rdma-gpu{0..7}/result.json`
all show

```json
{"gb_s": null, "gib": 0.0, "loc": "gpu", "gpu": N,
 "verified": null, "reason": "transfer_failed", "dev": "", "operation": "write"}
```

The empty `"dev"` is the interesting part: no HCA was selected at all, which is
not the same failure as the cross-rail mismatch documented in
`yihou/glm52-1p1d-samerail-c32-c40.packup_20260918` (there, a device *was*
chosen — just a different one at each end).

Two candidate explanations, neither verified yet:

1. `preflight.sh` enumerates all 8 GPUs, but our `RDMA_DEVICE` JSON is keyed on
   **local visible-device indices 0-3** (physical 2,3,4,5 under
   `HIP_VISIBLE_DEVICES=2,3,4,5`). If the preflight container does not set that
   mask, keys 0-3 mean physical GPUs 0-3 and 4-7 have no entry — but that would
   not explain GPUs 0-3 failing too.
2. The JSON form is not reaching the probe at all, so the transport engine has
   an empty device list.

The C32/C40 packup's preflight evidence came from a **pinned-NIC probe script**,
not necessarily from a bare `preflight.sh` with the JSON map — so a bare run
failing here may be a harness mismatch rather than a fabric problem.

Left with the teammate. Next poll: if it has not diagnosed this, the leader
joins the analysis.

## 2026-09-18 ~11:10 UTC — poll 2 — RESOLVED by the teammate, no intervention needed

The teammate diagnosed it before this poll and is already re-running. Cause:
**`preflight.sh` mounts the host GPUDirect provider only if `HOST_RDMA_LIB` is
already exported**, and it sources **no config file at all** — verified by the
leader by reading the script end to end: it takes only `IMAGE=` plus node names
(`preflight.sh:10`, and the mount at `:62-63`). So `config.sh:104`'s default
never reaches it, the container kept its stock `libionic`, and the transport
engine came up with no usable device — which is exactly why `"dev"` was empty
rather than wrong.

**The leader's own brief was the defect here**: it told the teammate to pass
`CONFIG=` to `preflight.sh`, which has no such parameter. The teammate read the
script first, as briefed, and worked around the bad instruction.

First-hand confirmation that run2 differs, from its rank logs — a line absent
from run1:

```
infera-inject-host-ionic: replaced /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187 with host build
```

Run2 (`preflight-libionic/`, started 10:51:35) is in the mooncake stage; per-GPU
directories not yet created. CPU-memory RDMA was already healthy in run1
(`rdma-default` 7.28 / 7.25 GB/s both directions), consistent with a
GPUDirect-only fault.

Leader supplied one piece of missing information rather than analysis: run2 must
also export `RDMA_DEVICE` and `MC_TE_FILTERS`, or Mooncake falls back to
auto-discovery and reproduces the isolated-rail trap from the C32/C40 packup
(§1) — GPUs 4-7 failing symmetrically in a way that looks like dead hardware.
Also flagged that the JSON keys are local visible-device indices, so if preflight
enumerates all 8 physical GPUs, 0/1/6/7 having no mapping is **expected** and the
GO/NO-GO must be judged on GPUs 2,3,4,5 alone.

## 2026-09-18 ~11:10 UTC — `image-hicache` — idle, task complete

Nothing outstanding.

## 2026-09-18 ~12:15 UTC — poll 3 — no live teammates

`image-hicache` and `bench-preflight` are both idle with their deliverables
written (`image/build-verification.md`, `preflight-report.md`). Deliberately
**not** messaged: waking a finished agent to ask for a status line costs tokens
and returns nothing. Nothing outstanding on either.

Work from here is leader-driven: the T2 sweep runs unattended via
`sweep.yihou.sh`, and the T1 pack-up is being written in parallel.

Sweep state at this poll: point 1 (CONC=40) started 12:11:09Z, in dataset
reconstruction (16 workers, 393 traces) — the 4-14 min setup phase that
`benchmark_lib.sh` documents, not a stall.
