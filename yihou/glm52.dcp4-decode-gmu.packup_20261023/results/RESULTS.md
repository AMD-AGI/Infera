# gmu results — DCP4 decode-only start-up at mem-fraction-static 0.92 (TP0, GiB as logged)

Flags = S2 decode argv (TP4 EP4 DCP4, EAGLE 5/1/6, graph max bs 96, max running 96, ctx 1M) + one change each.
Reference row S2 = the S2 PD run at 0.85 (039, 12:22Z). Probe script `decode_only.yihou.sh` (v2 frozen copy).

| Run | Node | Change | KV logical tokens | post-pool avail | pool end -> capture begin | verify graph | draft decode graph | draft extend graph | available_gpu_mem |
|---|---|---|---|---|---|---|---|---|---|
| S2 (ref) | OAI 039 | mem 0.85 | 8,869,120 | 43.56 | 5.29 | 5.75 | 1.92 | 3.40 | 27.20 |
| e2 | OAI 227 | 0.92 + `--cuda-graph-bs-decode 96` | 10,249,216 | 25.32 | 5.30 | 5.00 | 1.58 | 2.32 | **11.12** |
| e3 | OAI 227 | 0.92 + `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1` | 10,249,216 | 25.32 | 5.30 | 5.75 | 1.92 | (off) | **12.34** |
| e5 | v3-012 (SDMA=0) | 0.92 + `--cuda-graph-max-bs-decode 64` | 10,249,472 | 25.32 | 5.30 | 4.54 | 1.26 | 2.24 | **11.98** |
| e4c | OAI 227 | 0.92 + `--context-length 262144` | 10,249,216 | 25.70 | 5.40 | 3.83 | 0.62 | 0.79 | **15.06** |
| e6c | v3-012 (SDMA=0) | 0.92 + `--cuda-graph-backend-decode disabled` | 10,249,472 | 25.32 | 5.38 | - | - | - | **19.94** |
| e7 | OAI 227 | 0.92 + `--disable-custom-all-reduce` | **11,191,040** | 26.78 | 6.92 | 5.40 | 1.92 | 3.41 | **9.13** |

- e7: "Load weight begin" avail 284.16 vs 270.13 with custom AR on: the custom-AR buffers cost 14.03 GiB per GPU before
  weights load (charged to the KV budget, not to post-pool headroom). KV logical tokens +9.2 % vs custom AR on at the
  same 0.92. (S1 DPA4, custom AR on: 277.89.) Whether custom AR's speed is worth 14 GiB is not measured.
- e4c: ctx 256K cuts the three graphs from 11.07 to 5.24 GiB (page tables scale with context_len), the cheapest knob.

- All three started ("The server is fired up") at 0.92. Logical KV +15.6 % vs S2.
- First attempts e4/e6 died at RCCL init when started together with the other slot
  (`NCCL error: unhandled cuda error`, `HIP failure: 'invalid argument'`); rerun in series 2 with a PID burn on slot b.
- Unmeasured: plain 0.92 (e1 skipped by the user); runtime peak under load.

## Workload check: `--context-length 262144` is not usable for AgentX
S2 run's input_sequence_length (4240 profiled requests): max 754,580, p99 674,062, p50 72,380; share above
128K 29.0 %, 192K 18.1 %, 256K 11.7 %, 384K 6.3 %, 512K 3.3 %, 768K 0. A 256K cap would reject 11.7 % of requests.

## Load run S2g092 (227, launched ~17:20Z): S2 PD config + decode `--mem-fraction-static 0.92` only
VRAM sampled every 15 s: /shared_nfs/yihou_glm52/dcp_pd_20261021/gmu/vram.S2g092-227.txt

Plain 0.92 (S2g092 start-up, 227): pool end -> graphs same as S2; available_gpu_mem **8.93** GiB (projection was 8.95). VRAM total 288.0 GiB per GPU.

### S2g092 result: decode crashed under load (plain 0.92)
- Warm-up 17:33Z fine; profiling started ~17:55Z. At 18:02:56Z three decode ranks aborted with
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Fatal Python error: Aborted`; TP3 then failed in a broadcast. No torch OOM line.
- rocm-smi on decode GPUs: 283.4 GiB at warm-up start -> 287.70 of 288.0 GiB at 18:02:56Z, just before the abort.
- Last TP0 decode line (18:02:52Z): running 27, token usage 0.98-1.00 (10.2M logical tokens), prealloc 11-13,
  transfer 41, max seq 559,773.
- Whether the abort is device-memory exhaustion is not established (HSA error, not a torch OOM); the VRAM sample is
  the only memory evidence. Logs: work/S2-227/runs/S2-20261022T172214Z/logs-live/. Stack docker-stopped 18:13Z.

### Next (launched ~18:20Z, one 30-min C80 each)
- S2a on OAI 227: 0.92 + `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1` (start-up headroom 12.34 in e3).
- S2b on OAI 190: S2a + `--cuda-graph-max-bs-decode 64`.

### S2a / S2b result: both decode legs aborted the same way as S2g092
| Run | Abort (decode.log) | Last TP0 decode line before abort | Decode VRAM (rocm-smi, GiB of 288.0) |
|---|---|---|---|
| S2g092 (227) | 18:02:56Z | running 27, token usage 0.98, max seq 559,773 | 283.4 at warm-up -> 287.70 just before |
| S2a (227) | 18:59:00Z | running 20-24, **token usage 0.17**, max seq 339,482 | flat 280.2 from 18:42 to 18:56:23 -> 286.9 at 18:57:38 -> 287.7 at 18:58:23 -> abort |
| S2b (190) | 18:59:04Z | running 23-27, token usage 0.99, max seq 692,672 | flat 278.4 -> 284.0 at 18:54:23 -> 281.3 -> abort |
- Error on 3 of 4 decode ranks: `rocdevice.cpp:3582 Callback: Queue ... Aborting with error: HSA_STATUS_ERROR_OUT_OF_RESOURCES`
  then `Fatal Python error: Aborted`. No torch OOM exception in any of the three.
- S2a aborted at 17 % KV usage, so the abort is not tied to a full KV pool.
- S2a and S2b aborted 4 s apart on two different nodes (227, 190). Whether that coincidence means anything is unknown.
- Every run gains 4-7 GiB of decode VRAM within ~1-2 min under load right before aborting; what allocates it is unknown.
  The S2 run at 0.85 (27.2 GiB post-capture headroom) completed 30 min without abort (no VRAM sampling then).
- Logs: work/S2a-227/runs/S2a-20261022T181529Z/logs-live, work/S2b-190/runs/S2b-20261022T181431Z/logs-live.
  Stacks docker-stopped ~19:10Z.
- S2a timeline (decode.log, TP0): 18:57:06 running 1 (warm-up -> profiling switch), 18:57:39 running 19, 18:58:55
  running 24 / max seq 339K, abort 18:59:00. At 18:57:14 all 4 ranks log first-time `_merge_candidates_kernel` Triton
  compile and `[aiter] fused_moe using 2stage default for (gfx950, 256, ...)` (a MoE shape not seen in warm-up).
  The VRAM jump (280.2 -> 286.9, between the 18:56:23 and 18:57:38 samples) falls in the same window. Correlation
  only; which allocation it is is not measured.

### S2d (0.88, 190) also aborted; S2c (0.90, 227) stopped before profiling
- S2d: warm-up decode VRAM 272.9 GiB; in profiling 286.0 (20:12Z), 287.95 (20:15:02Z); abort on one rank 20:15:11Z,
  same `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. Before that: kv_usage 97 %, running 44-51, out 2,370-2,647 tok/s.
- S2c (0.90) was still in warm-up (797/884) and was docker-stopped at ~20:25Z without a profiling phase.
- So 0.92, 0.90 (untested under load), 0.88 all fill decode VRAM to ~288 GiB under load; 0.88 aborted after ~10 min.
- Reference rerun launched: S2 at 0.85 on 227 with VRAM sampling (vram.S2ref085-227.v2.txt), to measure the runtime
  peak of the configuration that completed before.

### S2e (0.88 + memlog, 190): the growth is torch caching-allocator reserve, not live tensors
TP0 "Decode batch" memlog (GiB): profiling start 21:07:22 dev_free 16.36 / alloc 248.93 / rsv 250.59 ->
21:12:42 13.22 / 249.05 / 253.65 -> 21:13:16 4.95 / 249.39 / 261.89 -> 21:16:31 1.40 / 249.26 / 265.42 ->
21:17:42 **0.92 / 249.73 / 266.25** (peak_alloc 249.74) -> abort 21:17:44 (HSA_STATUS_ERROR_OUT_OF_RESOURCES).
- Allocated (live tensors) moved +0.8 GiB over the run; reserved moved +15.7 GiB; device free went to 0.92 GiB.
- So the caching allocator kept ~16.5 GiB of freed blocks until device memory ran out; the abort followed 2 s later
  from the HIP runtime (non-torch allocation). The link between the two is inferred from timing, not traced.
- Next, one variable vs S2e: S2h = S2e + `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (190, launched 21:25Z).
  S2f (v3-019, burst, SDMA=0): 0.92 + expandable_segments + memlog.

### S2r (0.85 reference rerun, 227): completed; S2h and S2f aborted
- S2r = S2 unchanged: **25,433 tok/s/chip** (total 203,468; S2 on 039 was 25,941), TTFT mean 18.71 s, TPOT 20.54 ms,
  3267 ok. Decode VRAM 265.1 GiB after warm-up -> 281.8 in profiling (+16.7, rocm-smi; no memlog in this run).
- S2h (190) = S2e + `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`: same pattern as S2e. Last memlog 22:09:03
  dev_free 0.99 / alloc 249.58 / rsv 266.19 (gap 16.6 GiB); abort 22:09:07. The env made no visible difference.
- S2f (v3-019, burst, SDMA=0) = 0.92 + expandable_segments + memlog: last memlog 22:06:06 dev_free 2.90 / alloc
  259.76 / rsv 263.97 (gap 4.2 GiB); abort 22:06:09. Here the device still had 2.9 GiB free 3 s before the abort.
- Reading: at 0.85 the same +16-17 GiB growth fits in the headroom; at 0.88 it does not. Where the HIP-side
  OUT_OF_RESOURCES allocation comes from (S2f aborted with 2.9 GiB free) is not measured.
- S2i (190) = S2e + `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9`: aborted the same way. Last memlog 23:01:42 dev_free 1.89 / alloc 249.58 / rsv 264.94 (gap 15.4), running 55, token usage 0.94. GC threshold made no visible difference.
- S2j (227) = S2e at 0.87: aborted. Last memlog 23:25:10 dev_free 3.18 / alloc 246.91 / rsv 263.64 (gap 16.7), running 57, token usage 0.98. So under this load: 0.85 completes (S2, S2r), 0.87 / 0.88 / 0.90 (untested under load) / 0.92 abort.
- S2k (190) = S2e + empty_cache() when device free < 6 GiB, but only at the TP0-only Decode-batch log point:
  TP0 fired once (rsv 260.85 -> ~254, dev_free back to 10.7-13 GiB) while TP1-3 (GPUs 5-7, rocm-smi) stayed at
  287.4-287.9 GiB; ranks 1-2 aborted 23:55:13Z. So releasing the cache works on the rank that does it; the patch
  must run on every rank.
- S2m (227, 0.92) / S2n (190, 0.88): memlog-ec2 overlay = the same rule on every rank, every 8 iterations of both
  disagg-decode event loops (`disaggregation/decode.py`, after process_decode_queue). Launched ~00:05Z 10-23.
  S2l (0.92 with the TP0-only patch) was stopped during start-up.

### S2n (190, 0.88 + all-rank empty_cache, memlog-ec2): COMPLETED, first stable run above 0.85
| Run | decode mem | tok/s/chip | TTFT mean (s) | TPOT mean (ms) | decode running / transfer-in / admission (mean) | ok |
|---|---|---|---|---|---|---|
| S2 (039) | 0.85 | 25,941 | 17.69 | 20.32 | 30.8 / 22.0 / 12.8 | 3362 |
| S2r (227) | 0.85 | 25,433 | 18.71 | 20.54 | n/a | 3267 |
| **S2n (190)** | **0.88 + patch** | **27,685** | **13.16** | 21.10 | 36.0 / 17.7 / 8.0 | 3592 |
- KV logical tokens at 0.88: 9,460,736 (S2: 8,869,120, +6.7 %). Peak torch reserve 262.29 GiB was released back
  (TP0 rsv 251-256 later); rocm-smi GPUs 4-7 stayed <= ~280 GiB. One run each (protocol), so +6.7..8.9 % vs the two
  0.85 runs is a single-sample difference.
- S2m (0.92 + patch, 227) was lost with node 227 (NODE_FAIL ~01:00Z) mid-profiling; S2o reruns it on burst v3-019.
- S2o (burst v3-019, SDMA=0, 0.92 + all-rank empty_cache): COMPLETED, 0 aborts. 24,114 tok/s/chip (total 192,914), TTFT mean 20.93 s, TPOT 20.66 ms, decode running 30.2 / transfer-in 29.1 / admission 7.2, 3110 ok. Burst node with SDMA=0: not comparable to the OAI runs.
- S2m (OAI 190, 0.92 + all-rank empty_cache): COMPLETED, 0 aborts. **22,540 tok/s/chip** (total 180,318), TTFT mean 24.18 s, TPOT 20.59 ms, decode running 25.0 / transfer-in 37.7 / admission 6.9, 2865 ok; KV 10,249,216 logical. Lower than S2n (0.88, 27,685) and the 0.85 runs; cause not measured (single run; empty_cache firing frequency not logged on all ranks).
