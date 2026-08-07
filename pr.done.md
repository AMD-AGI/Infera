# sglang GLM-5.2 patches — upstream PRs opened

Companion to `work.todo.md`, which inventories all seven sglang GLM-5.2 patches
and establishes that exactly three needed a new upstream PR. This file records
what was opened, on what evidence, and what is still outstanding.

Date: 2026-08-07. Upstream base: `sgl-project/sglang` `main` @ `7395ee833e`.
All three are **drafts** — see "Deferred validation" for why.

## Opened

| PR | Title | Patch it upstreams | State |
|----|-------|--------------------|-------|
| [#33968](https://github.com/sgl-project/sglang/pull/33968) | `[ROCm] Fix HiCache host-pool allocator: hipHostRegister's device pointer is not the host VA` | `sglang_rocm/patch_hicache_rocm_host_alloc.py` | draft |
| [#33970](https://github.com/sgl-project/sglang/pull/33970) | `[PD] Make the mooncake KV transfer wait on the prefill forward that wrote the pages` | `sglang_disagg/patch_mooncake_early_send_wait_event.py` | draft |
| [#33973](https://github.com/sgl-project/sglang/pull/33973) | `[DSA] Remove the device-to-host syncs on the decode DP-divergent branch` | `sglang_dsa/dsa_backend_dp_sync_and_page_table_rows.diff` — **2a half only** | draft |

Branches live on `dorado269/sglang`. Working folder with the adapted diffs,
validation scripts and PR bodies: `work/upstream-glm52-sglang-prs/`.

## What each PR changes, and what was found while adapting it

### #33968 — HiCache ROCm allocator

`ALLOC_MEMORY_FUNCS` defaults to `alloc_with_host_register` (anonymous `mmap` +
`hipHostRegister`), but the host pools hand host `data_ptr()`s to GPU kernels
through a device-side pointer table. On ROCm those addresses are not the same,
so the first write-back faults at the host VA.

**The record described this as a one-line override in the shape of the merged
MUSA PR #23361. It cannot be**, for two reasons established first-hand:

1. `get_device()` returns `"cuda"` on ROCm — measured in the container — so HIP
   has no key of its own to add, unlike `"npu"`/`"musa"`.
2. `memory_pool_host.py:768` and `:1257` key the table with a `torch.device`
   **object**. Measured: `torch.device("cuda:0") == "cuda"` is `False` and the
   hashes differ, so those two pools *always* resolve through the `defaultdict`
   default. Adding only a `"cuda"` key would silently miss them.

So the default has to change too. That widens behaviour on a ROCm host to any
other device string (e.g. `"xpu"`); raised with the user, who chose to keep
equivalence with the proven fix rather than ship an unvalidated narrower one.

### #33970 — mooncake early-send wait event

Silent correctness bug: chunked prefill over mooncake RDMA-reads KV pages while
the forward writing them is still running, so long prompts come back partially
wrong with no crash and no log line.

Confirmed on `main`: `mooncake/conn.py` contains `wait_event` **0 times** while
`mori/conn.py` contains it **6 times**, and the overlap non-final-chunk send in
`prefill.py` records no event at all.

**Cross-transport effect worth knowing:** `prefill.py` is transport-agnostic, so
recording the event there also closes the same gap for `mori`, which reads the
field but was never handed one on that path. Checked that `nixl`, `ascend`,
`fake` and `base` do not read `_early_send_wait_event` and that no sender uses
`__slots__`, so setting the attribute is inert for them.

### #33973 — DSA decode DP host-sync (2a only)

`seq_lens.max().item()` is a D2H sync on a branch only some DP ranks enter, so
the group deadlocks on the first routed request.

The strongest upstream argument turned out to be one the local record did not
use: the backend **already declares** `needs_cpu_seq_lens = False`, with a
comment saying it opts out of the D2H sync — so the eager fallback contradicts
its own contract. Also corrected a citation: the CPU mirror this path relies on
comes from `eagle_worker_common.prepare_for_draft_extend` (line 105), not
`base_spec_worker` as the record said.

**Scope deliberately narrowed.** The local diff bundles a second, independent
defect (2b, the per-request vs per-token page-table row mismatch). It is **not**
in this PR: it overlaps #32209, and porting that PR's trimming approach here
fails reproducibly at concurrency 32 with the cause unidentified. One fix, one
PR; 2b stays local until that is understood.

## Validation

Everything below was run **this session**, on 8× MI300X (gfx942), ROCm 7.2.0,
torch 2.9.1, inside `lmsysorg/sglang-rocm:v0.5.15-rocm720-mi30x-20260713` with
the adapted upstream tree on `PYTHONPATH`. Scripts are in the working folder.

| Script | What it establishes | Result |
|--------|--------------------|--------|
| `probe_host_devptr.py` | host VA vs device pointer per allocation strategy | PASS — all four strategies **equal** on MI300X, reproducing the documented negative control |
| `validate_A.py` | #33968 equivalence + scope vs the proven local fix | PASS |
| `validate_B.py` | #33970 plumbing + that the barrier is real, against live CUDA events | PASS |
| `validate_C.py` | #33973 — the sync is real; widening does not change top-k; the removed mirrors are unreachable | PASS |

Two findings from those runs that changed the PRs rather than just confirming
them:

- `.max().item()` measurably blocks — **~0.5–3 ms** run to run, behind 30 queued
  4096² matmuls — versus **~3 µs** for `.shape[1]`. First-hand evidence that the
  branch really does desynchronize ranks, quoted in #33973. The number moved
  during cleanup: an earlier revision measured 44 ms because the timed call was
  also the first one, so it carried lazy-init cost. Warmed, it is 0.5–3 ms. The
  PR was corrected; what matters is that it blocks at all, not the magnitude.
- `validate_C.py` initially **failed** an assertion I had written. Diagnosis: a
  row with `seq_len < topk` legitimately returns indices past its own `seq_len`,
  because top-k must return `topk` entries; those slots carry `-inf` and occur
  identically at both widths. My test's model was wrong, not the fix — but the
  correct property (every *real* selection is bit-identical narrow vs wide) is
  now asserted, and stated in the PR.

## Deferred validation — why all three are drafts

The original faults need hardware this session does not have. The vultr MI355X
cluster is **unreachable**: `149.28.124.225` answers neither ping (100% loss)
nor `:22`; `chi2865` / `chi2866` the same. The local box is gfx942, which is the
*negative control* for #33968 and cannot reproduce any of the three faults.

So this session established **equivalence and scope** against the proven local
fixes, and nothing more. Each PR body says so explicitly and marks which evidence
is historical.

**TODO when the cluster returns** — re-run against the exact PR diffs, then flip
each draft to ready:

| PR | Needs | Run |
|----|-------|-----|
| #33968 | 1× gfx950 | HiCache write-back repro: stock must fault at the host VA, patched must not. Re-run `probe_host_devptr.py` there — expect `same=False`, the positive control. |
| #33970 | 2 nodes with RDMA | GLM-5.2-FP8 1P1D over mooncake, `--chunked-prefill-size 131072`, overlap on. Needle retrieval, expect 5/9 → 9/9. **Plus** the added `synchronize()`'s cost on prefill throughput — flagged as unmeasured in the PR, and the first thing a reviewer will ask. |
| #33973 | 1× gfx950, PD + DP-attention + MTP | Group must not deadlock on the first routed request; `py-spy` should show no rank inside `dsa_backend`. |

## CI

All three show `pr-gate` red. Cause verified via the job steps API: the failing
step is **"Block draft PR"** — a repository policy that fails the gate for any
draft, and the `*-finish` jobs just aggregate it. Not a lint, format, or compile
failure; `pre-commit` passes clean locally on every changed file. These go green
when the drafts are marked ready.

## Not upstreamed, and why

From `work.todo.md`, four of the seven sglang patches correctly get no PR:

- `patch_glm52_nextn_quark_exclude.py` — #30265 already **merged**; ours is a
  backport onto a frozen release base.
- `patch_dsa_indexer_hip_dp_padded_rows.py` — already ours as **#33059** (open).
- `draft_cuda_graph_dp_vote.diff` — **#32209** carries the same fix with the same
  strategy; a competing PR would be noise. Add a ROCm datapoint to that thread
  and to #32527 instead.
- `patch_hicache_rocm_staged_write_back.py` — **#30350** is the upstream repair
  and is better than ours (covers the DeepSeekV4 pools too). It is
  `CHANGES_REQUESTED` with the conflict long cleared; the action is a re-review
  nudge. Note the inversion: the *merged* #28534 is what introduced the defect.
- `dsa_backend_dp_sync_and_page_table_rows.diff` **2b half** — see #33973's scope
  note above.
