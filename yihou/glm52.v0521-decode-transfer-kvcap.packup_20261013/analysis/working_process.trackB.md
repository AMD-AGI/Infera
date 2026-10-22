## X1 (2026-10-12 06:47-08:09 UTC, 102): AgentX C44 on the live warm e2j stack (3rd run; e2j R2 C40 warm = 21,681)
- Driver `$WS/harness/r3_e2j44.yihou.sh`, env `env.e2j44-102.yihou.sh` (= e2j, CONC 44). /metrics sampler 5 s:
  `/shared_nfs/yihou_glm52/v0521cap_20261012/runs/x1-e2j44/metrics.tsv`.
- Result [M] `runs/e2j/agentx-c44-R3/agentx_conc44.json`: 4373 ok / 4 err-dropped, **21,773 tok/s/chip** (+0.4 % vs
  same-stack C40 21,681; −1.5 % vs e2g A2 22,114), TTFT mean/p95 9.02 / 30.6 s (C40: 5.21 / 16.57), TPOT 11.7 ms.
- Decode KV [M from metrics, I for the split]: token_usage max 0.71/0.82/0.87/0.72 per DP; prealloc queue always 0;
  retracted 0; transfer queue up to 13-15 reqs/DP. Split used−decode_sum_seq_lens ("held by transfer") p50 0.06-0.09,
  p90 0.33-0.45, max 0.59-0.71 of the pool; in 28-35 % of busy samples running reqs held < half of used KV. At the DP0
  peak: used 0.71, running 0.06 (1 req), transfer 0.65 (7 reqs). Semantics of decode_sum_seq_lens to be confirmed (R1).
- Reading (suspended): C44 does not raise throughput on this stack; TTFT grows. Transfer reqs hold a large share of
  decode KV at peaks, but nothing blocked or retracted at C44.

## TB1 (2026-10-12 08:4x): Track B patch prepared (not run yet)
- Research R1 (`research/R1-decode-admission-code.yihou.md`): no config expresses "running keeps >= 50 %"; in-flight
  (transfer) KV is not retractable; `prefill_complete` policy needs prefill write_back; host receive excluded for MLA/DSA.
- Mount `$WS/harness/mounts-v0521/src/decode.yihou-runshare.py` (from image decode.py md5 bd1e5fe7; +import os;
  in `pop_preallocated`: cap = (1-SHARE)*max_total_num_tokens, inflight starts at `num_tokens_pre_allocated` (same
  accounting as the log's "pre-allocated usage"), block (FIFO break) when inflight>0 and inflight+required_alloc >
  cap; rebootstrap exempt; rate-limited log `YIHOU-RUNSHARE block`). Overlay `config.yihou.runshare.sh`
  (RUNSHARE_BASE_CONFIG, RUNNING_KV_SHARE default 0.5), TSV `mounts.yihou.decode-runshare.tsv` (decode only).
  Dryrun vs e2g: only the decode mount + env differ [M]. Verify live: log "pre-allocated usage" <= 0.5 (+ page slack),
  YIHOU-RUNSHARE lines, #prealloc-req, TTFT.

## TB2 (2026-10-12 ~09:05): patch rewritten to the user's rule
User: "(要申请的量 + 已经占有的量) − running req 占用的量 < 50 % 才可以 … 为 running 和预备 running 的请求留足空间";
patch + debug on a separate machine (burst/oai) until C40 runs with no perf regression; max_running 64→96 important.
- `decode.yihou-runshare.py` (diff `decode.yihou-runshare.diff`, 54 lines): at the top of `pop_preallocated`,
  used = pool − allocator.available_size(); running = Σ running_batch.reqs kv.kv_allocated_len (maintained per step:
  mem_cache/allocation.py:670, 809 [M]); nonrunning = used − running (in-flight + arrived-not-batched + deferred releases
  + slack — closes verify gap G1). Admit only if nonrunning + required_alloc < (1−SHARE)·pool; nonrunning += admitted.
  Rebootstrap exempt; progress escape when transfer_queue and waiting_queue are empty. Log `YIHOU-RUNSHARE block`.
- Burst hold submitted: job 209081 (`yihou-burst-patch`, amd-burst-qos, -x 102) PENDING (Resources); oai 208970 PENDING.
- Debug plan on that node: (1) e2g stack + runshare with a forced small cap (SHARE 0.9) + a burst of long prompts:
  blocks logged, no deadlock, no errors; (2) SHARE 0.5 + max_running 96, AgentX C40 warm protocol vs 22,114.

## TB3 (2026-10-12 09:19): round pc40 launched on burst node 174 (job 209081)
- Image loaded on 174 keeps ID sha256:9d755dd6 [M] (load_image.174.log). New `config.yihou.mc4340n.sh` (adds 174),
  env `env.pc40-174.yihou.sh` = e2g recipe + runshare 0.5 (C40; max_running stays 64 → single-variable vs e2g).
  Dryrun vs e2g: only node IP/paths + decode mount/env [M]. JIT caches seeded from 102 (cp -a, 831 MB).
- Chain `chain_pc40.yihou.sh`: up → census → smoke → audit → long-prompt burst (64 × 60k words, must show
  YIHOU-RUNSHARE blocks and all ok) → A1 C40 → flush → A2 C40 (scored vs 22,114; note node differs: 174 vs 102).

## TB4 (2026-10-12 10:38): burst nodes unusable for 3 h rounds -> patch validation moved to 102 after k1
- Burst job 209081 (174) and 209186 (029) both ended at exactly 00:20:00 elapsed (PREEMPTED / CANCELLED) during prefill
  bring-up [M sacct]. amd-burst-qos: Priority 110, PreemptMode **cancel** [M] -> burst jobs are cancelled once
  higher-priority work needs the node (after ~20 min). Third burst try (205) had a foreign GPU drain pool.
- pc40c armed on 102: `chain_pc40c.yihou.sh` waits for "k1 chain done", stops k1, then up → burst smoke → C40 A1 →
  flush → A2 (e2g + runshare 0.5; same node as the 22,114 baseline -> clean single-variable check). ETA A2 ≈ 15:00 UTC.
- Q1 (max_running 96 + prefill 0.88, C44) follows pc40c on 102 unless the 2nd oai node (208970) arrives first.

## TB5 (2026-10-12 12:17): pc40c burst smoke did not reach the cap
- Patch active [M]: decode container env YIHOU_DECODE_RUNNING_KV_SHARE=0.5, mounted decode.py has YIHOU-RUNSHARE.
- Burst 64 × 60k words = ~69.4k tokens each: 64/64 ok (max 178 s), 0 blocks. /metrics (2 s): kv_used max 0.25-0.30 of
  pool, transfer queue max 9-11/DP, prealloc queue 0 → in-flight never approached 0.5, so the cap was not exercised.
- Armed `post_pc40c_smoke.yihou.sh`: after A2, on the same stack, 64 × 200k words (~230k tok) → ~0.75 in flight without
  cap → must log blocks, complete all, no faults. C40 A1 started 12:16 (A2 ≈ 15:00).

## pc40c A1 (102, 12:16-13:37): e2g + runshare 0.5, C40, FRESH (unscored)
- [M] 4091 ok / 3 err, **20,361/chip**, qps 1.127, ISL 143,478, TTFT mean/p95 5.15 / 17.0 s, TPOT 14.5 ms.
- vs fresh C40 target runs on 102: e2g A1 19,640 (TTFT 7.61), e2i R1 20,371, e2j R1 19,335 → top of the fresh band, no
  regression signal. Scored A2 ≈ 15:00.

## pc40c A2 (102, 13:37-14:57): e2g + runshare v1 (0.5), C40, WARM (scored) — likely REGRESSION
- [M] 4208 ok / 0 err, **21,271/chip** (e2g A2 22,114 −3.8 %; lowest warm replicate on 102 21,681 −1.9 %), qps 1.160
  (e2g 1.181, −1.8 %), ISL 145,655 (148,711), TTFT mean/p95 6.17 / 20.93 s (e2g 5.30 / 19.21), TPOT 12.2 ms (12.1).
- Cap engaged in the profiling window: ~71-83k blocked checks on DP0-2 (5.7k DP3), blocks present in 4-10 of 60
  minutes per DP [M log]. Mechanism (inferred): under `early`, a blocked request cannot start prefill; FIFO `break`
  holds everything behind a ~0.7 M-token prompt → TTFT +0.9 s → closed-loop qps −1.8 %. Not yet an A/B (warm spread ±1 %).
- Action: q3 (Q1 + v1) disarmed. Patch **v2** `decode.yihou-runshare2.py`: same user rule, but a non-fitting request is
  SKIPPED (scan continues to smaller requests) unless the oldest blocked request has waited > YIHOU_RUNSHARE_MAX_WAIT_S
  (5 s) → then FIFO break (anti-starvation). Overlay `config.yihou.runshare2.sh`; round **pc40d** armed on 102 behind
  pc40c smoke2 (stop pc40c → up → 230k-token burst smoke → C40 A1 → A2).

## TB6 (15:08): runshare v1 functional smoke PASS (pc40c smoke2)
- 64 × ~231k-token prompts: 64/64 ok (max latency 616 s), block log lines 49,943 → 74,474 (each = 100 blocks), decode
  log "pre-allocated usage" max **0.50** (= cap 0.5 of pool), 0 faults [M]. The cap holds in-flight KV at the user's
  limit without deadlock. Throughput side: v1 regressed C40 (21,271) → v2 (pc40d) pending.

## TB7 (15:56): runshare v2 functional smoke (pc40d, 64 × ~231k tok): SUMMARY n=64 ok=64 failed=0 max_latency=616.1s
- block log lines 26,799; decode "pre-allocated usage" max 0.42 (≤ 0.5 cap); anti-starvation "stop" engaged (waited
  78-152 s for equal-size prompts); faults 0 [M]. C40 A1 started 15:56 (A2 ≈ 18:40).

