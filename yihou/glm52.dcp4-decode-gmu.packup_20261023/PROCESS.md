# PROCESS — how the decode gmu limit was found and lifted (2026-10-22 15:20Z -> 10-23 02:45Z, UTC)

Goal (user, 10-22 ~15:20Z): raise the DCP4 decode leg's `--mem-fraction-static` ("gmu") from 0.85 to 0.92 —
"important"; first find who uses memory outside the static budget; extreme start-up experiments allowed; no prefill
needed at first. Constraints kept throughout: custom AR stays ON (user directive), one 30-min run per config, one
variable per run vs a named reference, simulated acceptance (no correctness claims).

Each step: what was done -> what was measured -> what it changed. File references are inside this pack-up.

## Step 0 — Read the existing logs (S1/S2 runs)
- S2 (0.85) decode, TP0: pool end 43.56 GiB free -> `dcp_replicate_q_proj` -5.3 -> CUDA graphs verify 5.75 + draft
  decode 1.92 + draft extend 3.40 -> `available_gpu_mem=27.20`; S1 (DPA4, 0.92) ended at 10.53.
- Pool bytes/token 52.2 KB (DCP4) vs 44.4 KB (DPA4); DCP4 had 7.8 GiB less free before weights load.

## Step 1 — Code map (4 parallel readers + synthesis) -> `results/memory_map.md`
- Static budget `rest = avail_now - pre*(1-mfs)` (`kv_cache_configurator.py:2834`); what runs after it:
  q-rep all-gather (~5.3-5.9 GiB, `model_runner.py:1118-1157`), CUDA graphs (~6 GiB of them are wide page tables
  sized by context_len 1M, `dsa_backend.py:1598-1622`).
- Explained 52.2 vs 44.4 KB: DSA index-K replicated x4 per rank under DCP (`memory_pool.py:5346-5352`).
- Found the draft-pool over-charge x4 (3.57 GiB, `pool_configurator.py:303-304`).
- Projection for plain 0.92: ~8.3-9 GiB left after capture (inference).

## Step 2 — Decode-only start-up probes (no prefill) -> `results/startup/*.mem.txt`
Script `scripts/gmu/decode_only.yihou.sh` (S2 decode argv via `sglang.launch_server`, two slots per node).
| Probe | change at 0.92 | available after capture (GiB) |
|---|---|---|
| e2 | graph bs=[96] only | 11.12 |
| e3 | no draft-extend graph | 12.34 |
| e5 | graph max bs 64 | 11.98 |
| e4c | `--context-length 262144` | 15.06 |
| e6c | no CUDA graphs | 19.94 |
| e7 | `--disable-custom-all-reduce` | 9.13, but KV +9.2 % (custom AR = 14.03 GiB before weights) |
- Everything started at 0.92. Then: the AgentX ISL distribution (S2 run) has 11.7 % of inputs > 256K, max 754,580,
  so the context-length knob is unusable (`scripts/gmu/isl_hist.yihou.py`).
- Pitfalls hit: two slots starting together -> RCCL init `HIP failure: invalid argument` on slot a (fixed with a
  30k-PID burn on slot b, cause not confirmed); editing a running bash script on NFS broke its monitor (use frozen
  `*.v2.*` copies); a stale-tag check refused reruns (new tags).

## Step 3 — Full PD load runs at higher gmu (all aborted)
- S2g092 (0.92 plain), S2a (0.92, no draft-extend graph), S2b (S2a + max bs 64), S2d (0.88), S2j (0.87): each
  aborted under load with `HSA_STATUS_ERROR_OUT_OF_RESOURCES` (3 of 4 ranks typically), never a torch OOM.
- rocm-smi sampling (`scripts/gmu/vram_sampler.v2.yihou.sh`) showed decode VRAM climbing 4-16 GiB after the switch
  from warm-up to profiling until ~287.7-288.0 of 288 GiB. S2a aborted at only 17 % KV usage -> not "KV full".
- Reference S2r (0.85) completed: 25,433 tok/s/chip (S2: 25,941); its decode VRAM also rose +16.7 GiB but fit.

## Step 4 — Instrument: who grows? (S2e, memlog overlay = patch 01)
- TP0 per Decode-batch line: `mem_get_info` free + torch allocated / reserved / peaks.
- S2e (0.88): allocated 248.93 -> 249.73 GiB, reserved 250.59 -> 266.25, free 16.36 -> 0.92, abort 2 s later
  (`results/memlog/S2e-*.TP0.memlog.txt`). Conclusion: caching-allocator reserve, not live tensors.

## Step 5 — Allocator knobs (no effect)
- S2h `expandable_segments:True`: reserve gap 16.6 GiB, aborted. S2i `garbage_collection_threshold:0.9`: gap 15.4,
  aborted. S2f (burst, 0.92 + expandable_segments): aborted with 2.9 GiB still free (unexplained).

## Step 6 — Explicit `empty_cache()` (the fix)
- S2k: `empty_cache()` when free < 6 GiB, but placed in the TP0-only log path (patch 02). TP0 recovered
  (reserved 260.85 -> ~254, free 10.7) while GPUs 5-7 stayed at 287.9 and ranks 1-2 aborted. -> must be all ranks.
- Patch 03 (all ranks, decode event loops, every 8 iterations): S2n (0.88) completed, 27,685 tok/s/chip;
  S2o (0.92, burst) completed, 0 aborts; S2m (0.92, OAI 190) completed, 0 aborts, but 22,540 tok/s/chip
  (< S2n 27,685 and < the 0.85 runs; cause not measured).
- Node loss: OAI 227 NODE_FAIL ~01:00Z killed the first S2m mid-run (no abort seen before).

## Open questions (measurement needed, not concluded)
- Which shapes churn the allocator; what non-torch allocation fails (HIP log level / tracing).
- Cost of `empty_cache()` on step time (need A/B at the same gmu with and without the patch).
- Run-to-run spread of AgentX C80 30-min runs on this config (single runs only so far).
