# [DRAFT, not submitted] HiCache: optional minimum eviction batch for write_back (`SGLANG_EVICT_HEADROOM_TOKENS`)

Diff: `evict-headroom.diff` (against the v0.5.22 `unified_radix_cache.py`; must be rebased onto upstream main and the env var moved into
`sglang.srt.environ` before submission).

## Problem
With HiCache `write_back`, every device eviction has to wait for the forward that is running on the device (the write fence) before it can copy
its victims to the host pool. `evict_for_alloc` frees only what the allocation is short of, so a busy prefill server evicts many small batches
and pays that wait each time. In a PP4 prefill (GLM-5.2, 8K-token chunks, HiCache host pool 250 GB per rank) one scheduler spent ~35% of its time
in `writing_check`, 79% of it waiting for the forward to end, and the GPU idled ~46% between steps.

## Change
New knob `SGLANG_EVICT_HEADROOM_TOKENS` (default 0 = unchanged). When the policy is `write_back` and an eviction asks for fewer tokens than the knob,
the request is raised to the knob, so one call frees a larger batch. The extra victims are demoted to the host pool (they stay as prefix cache).

## Evidence (one configuration; MI355X, GLM-5.2-MXFP4, 1 prefill TP1xPP4 + 1 decode TP4/DP4, AgentX trace replay, C104, 1 h, knob = 524288)
| | knob unset | knob 524288 |
|---|---|---|
| tok/s/GPU | 48,360 | 49,807 / 49,631 (+2.8% mean) |
| evictions per prefill step | 1.16 | 0.11 |
| PP0 time in eviction (1 h) | 1,686 s | 525 s |
| median full-chunk step | 380 ms | 328 ms |
| prefix-cache hit (host + device) | same | same |

Correctness: capped-pool gate (device pool limited to 300K tokens per rank so that two 16k x 128 passes evict continuously and the second pass loads the
host copies back): GSM8K probe 0.975 (control 0.965), garbled 4 (control 4), long-context needle 64/64 on both passes. The default (production-size) pool
never evicts in the normal correctness suites, so they cannot exercise this path.

## Known limits (state these in the PR)
- The extra victims are kept only if the host pool has room; when it is full, write_back drops them (the existing "KV subtree dropped without backup"
  path). In our runs the dropped counter stayed 0 with a 250 GB host pool.
- The diff adds the `is_write_back` condition; the tested build used the env var only, and write_back was always on, so the gate is a no-op there but the
  gated build itself has not been run.
- Measured on one model and one hardware/topology; no NVIDIA, no non-PP setup. Gains depend on how often evictions happen.
- A deeper fix would remove the write fence wait (we tried a background pre-write-back; it made evictions ~free but gave no end-to-end gain in this
  setup, so it is not proposed here).
