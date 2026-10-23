# OPTIMIZATION — run the DCP4 decode leg at mem-fraction-static 0.88-0.92

For a session that wants to fold this into its own config. Everything below is measured on the run named in
brackets (files under `results/`), unless marked *code read* (file:line on sglang `ab7a898b77`) or *inference*.

## 1. The change (what to apply)

**Release the torch caching allocator's free blocks on every decode rank when device memory runs low.**
Patch `patches/03-ec2-allrank.decode.diff` (`python/sglang/srt/disaggregation/decode.py`, both disagg-decode event
loops, right after `self.process_decode_queue()`; lines 3074 and 3121 on `ab7a898b77`):

```python
            self.process_decode_queue()
            # yihou gmu: release cached allocator blocks on every rank when device free < 6 GiB.
            self._yihou_it = getattr(self, "_yihou_it", 0) + 1
            if self._yihou_it % 8 == 0 and torch.cuda.mem_get_info()[0] < 6 * 2**30:
                torch.cuda.empty_cache()
```

then raise the decode `--mem-fraction-static` (0.85 in the S2 config -> 0.88 or 0.92). Nothing else changes.
The measured builds (S2n, S2m, S2o) also carried patch 01 (memory print on the TP0 Decode-batch line) and patch 02
(the same empty_cache rule at that TP0-only point); 03 alone was not run. This is an experiment-grade patch (hard-coded 6 GiB threshold, every 8 iterations, `getattr` counter); see §5.

## 2. Result

Same-node P4D4 on MI355X, GLM-5.2-MXFP4, llying v0522 PD config with DCP4 decode (TP4 EP4 DCP4, EAGLE 5/1/6,
custom AR on, index_share true, `SGLANG_SIMULATE_ACC_LEN=3.61`), AgentX C80, one 1800 s run each:

| Run | Node | decode mem | patch | tok/s/chip | TTFT mean | TPOT mean | decode running / transfer-in / admission | outcome |
|---|---|---|---|---|---|---|---|---|
| S2 | OAI 039 | 0.85 | no | 25,941 | 17.69 s | 20.32 ms | 30.8 / 22.0 / 12.8 | completed |
| S2r | OAI 227 | 0.85 | no | 25,433 | 18.71 s | 20.54 ms | - | completed |
| S2e, S2h, S2i, S2k | OAI 190 | 0.88 | no / allocator envs / TP0-only | - | - | - | - | **aborted** |
| S2j | OAI 227 | 0.87 | no | - | - | - | - | **aborted** |
| S2g092, S2a, S2b | OAI 227/190 | 0.92 | no (+ graph trims) | - | - | - | - | **aborted** |
| **S2n** | OAI 190 | **0.88** | **yes** | **27,685** | **13.16 s** | 21.10 ms | 36.0 / 17.7 / 8.0 | completed |
| S2o | burst v3-019, SDMA=0 | 0.92 | yes | 24,114 (not comparable: burst) | 20.93 s | 20.66 ms | 30.2 / 29.1 / 7.2 | completed |
| **S2m** | OAI 190 | **0.92** | **yes** | **22,540** | 24.18 s | 20.59 ms | 25.0 / 37.7 / 6.9 | completed |

- Decode KV capacity (logical tokens, TP0 log): 0.85 -> 8,869,120; 0.88 -> 9,460,736 (+6.7 %); 0.92 -> 10,249,216
  (+15.6 %).
- S2n vs the two 0.85 runs: +6.7 % / +8.9 % tok/s/chip, TTFT mean -26..-30 %, decode admission queue 12.8 -> 8.0.
  **One run each**; run-to-run spread was not measured. S2n is on a different node than S2/S2r.
- **0.92 + patch is stable but slower than 0.88 + patch** on the same node (S2m 22,540 vs S2n 27,685; TTFT 24.2 vs
  13.2 s; decode running 25.0 vs 36.0, transfer-in 37.7 vs 17.7). Cause not measured. Measurements that
  would settle it: count `empty_cache()` firings and their duration on every rank, and repeat S2m/S2n to get the
  run-to-run spread. Recommendation from the data in hand: **0.88 + patch**; 0.92 needs a cost measurement first.
- Correctness is not measured (simulated acceptance).

## 3. Why it works (evidence)

- Without the patch, every run above 0.85 aborted under load with
  `rocdevice.cpp:3582 ... HSA_STATUS_ERROR_OUT_OF_RESOURCES` / `Fatal Python error: Aborted`, never a torch OOM
  (`results/runs/*/decode.abort_excerpt.txt`).
- The memlog overlay (patch 01) prints `mem_get_info` free and torch allocated/reserved on each TP0 "Decode batch"
  line. In S2e (0.88, no fix): allocated moved +0.8 GiB over the run while reserved moved +15.7 GiB; device free
  fell 16.36 -> 0.92 GiB; abort 2 s later (`results/memlog/S2e-*.TP0.memlog.txt`). So the cached-but-free blocks of
  the caching allocator, not live tensors, fill the device. The link "device full -> HIP-side OUT_OF_RESOURCES" is
  inferred from timing, not traced.
- The 0.85 runs survive because the same growth (~+16.7 GiB decode VRAM in S2r, rocm-smi) still fits.
- With the patch (S2n), TP0 reserved peaked at 262.29 GiB and was released back to ~251-256; all four decode GPUs
  (rocm-smi) stayed <= ~280 GiB (`results/vram/vram.S2n-190.v2.txt`).
- The patch must run on **every** rank: S2k ran it only on TP0 (inside the TP0-only log path); TP0 recovered to
  10.7 GiB free while GPUs 5-7 stayed at 287.4-287.9 GiB and ranks 1-2 aborted.

## 4. What did not work (do not repeat)

| Tried | Result | Evidence |
|---|---|---|
| `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (S2h, S2f) | same reserve growth (gap 16.6 GiB), aborted | memlog S2h |
| `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9` (S2i) | same (gap 15.4 GiB), aborted | memlog S2i |
| Trimming CUDA graphs at 0.92 (no draft-extend graph S2a; + max bs 64 S2b) | +3..6 GiB start-up headroom, still aborted | `results/RESULTS.md` |
| `--context-length 262144` (graph page tables 11.07 -> 5.24 GiB) | start-up OK, but 11.7 % of AgentX inputs exceed 256K (max 754,580) | e4c, ISL histogram |
| Lower gmu 0.87 (S2j) | aborted | memlog S2j |

## 5. Before upstreaming / combining

1. Make it a switch (env or server arg), threshold and period configurable; drop the `getattr` counter (repo rules:
   `.claude/rules/no-getattr-defensive.md`, `env-var-conventions` skill).
2. Measure the cost of `empty_cache()` (it re-allocates later; in S2n TPOT mean was 21.10 ms vs 20.32/20.54 at 0.85
   — single runs, not attributable).
3. Unknown and worth measuring: which tensor shapes churn the allocator (likely varying batch/sequence shapes in
   eager parts of the decode step; not traced), and what non-torch allocation fails (S2f aborted with 2.9 GiB still
   free 3 s before). `AMD_LOG_LEVEL` / HIP memory tracing would answer it.
4. Larger levers left untouched (from `results/memory_map.md`, code read): custom AR costs 14.03 GiB/GPU before
   weights load in DCP mode (e7: `Load weight begin` 284.16 vs 270.13 GiB) — off-limits here because the user asked
   for custom AR ON; DSA index-K is replicated x4 per rank under DCP (~17 GiB of the pool,
   `mem_cache/memory_pool.py:5346-5352`); the draft KV is over-charged x4 in pool sizing (3.57 GiB,
   `model_executor/pool_configurator.py:303-304`).

## References

- Evidence: `results/RESULTS.md` (all runs, chronological), `results/memlog/`, `results/vram/`, `results/runs/<RUN_ID>/`.
- Code map (workflow code read, file:line on `ab7a898b77`): `results/memory_map.md`.
- Process narrative and dead ends: `PROCESS.md`. Reproduction: `REPRODUCE.md`. Environment: `environment.md`.
- Base config and its pack-up: `../glm52.llying-dcp-s1s3.packup_20261022/` (S1-S3, kit, image recipe).
- sglang code paths: `python/sglang/srt/disaggregation/decode.py:3058,3100` (event loops),
  `python/sglang/srt/managers/scheduler_components/metrics_reporter.py:969` (Decode batch line),
  `python/sglang/srt/mem_cache/kv_cache_configurator.py:2834` (static budget `rest = avail - pre*(1-mfs)`),
  `python/sglang/srt/model_executor/model_runner.py:1118-1157` (`_prepare_replicated_q_proj`, ~5.3 GiB after pool
  sizing), `python/sglang/srt/layers/attention/dsa_backend.py:1598` (wide page table kept on HIP).
- PyTorch caching allocator and `PYTORCH_*_ALLOC_CONF` options: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
  (HIP uses the same allocator; `PYTORCH_HIP_ALLOC_CONF` mirrors `PYTORCH_CUDA_ALLOC_CONF`).
