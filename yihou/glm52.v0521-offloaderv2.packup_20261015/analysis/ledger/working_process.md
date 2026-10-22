# Working process — task v0521cap (C44/C48, > 23k)

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

## k1 A1 (102, 2026-10-12 08:57-10:20): e2g + HiCache 360, C44, FRESH (unscored)
- [M] `runs/k1/agentx-c44/agentx_conc44.json`: 4218 ok / 5 err, **20,729/chip**, TTFT mean/p95 7.40 / 27.03 s, TPOT 15.1 ms.
  Fresh C44 comparators: e2c C44 (HiCache 450) 20,524, e2b4 20,347; fresh C40 target 19,335-20,371. Host pool 360 full on
  3/4 DPs by ~35 min. A2 (warm, scored) started 10:22.
- pc40b (Track B patch, burst node 029, job 209186): image loaded (ID 9d755dd6 kept), chain started 10:22 (etcd ok).

## TB4 (2026-10-12 10:38): burst nodes unusable for 3 h rounds -> patch validation moved to 102 after k1
- Burst job 209081 (174) and 209186 (029) both ended at exactly 00:20:00 elapsed (PREEMPTED / CANCELLED) during prefill
  bring-up [M sacct]. amd-burst-qos: Priority 110, PreemptMode **cancel** [M] -> burst jobs are cancelled once
  higher-priority work needs the node (after ~20 min). Third burst try (205) had a foreign GPU drain pool.
- pc40c armed on 102: `chain_pc40c.yihou.sh` waits for "k1 chain done", stops k1, then up → burst smoke → C40 A1 →
  flush → A2 (e2g + runshare 0.5; same node as the 22,114 baseline -> clean single-variable check). ETA A2 ≈ 15:00 UTC.
- Q1 (max_running 96 + prefill 0.88, C44) follows pc40c on 102 unless the 2nd oai node (208970) arrives first.

## Q1 launched on 137 (2026-10-12 10:54)
- env `env.q1-137.yihou.sh` (CONFIG mc4340n incl. 137 ID e8e2838f; e2g + DECODE_MAX_RUNNING 96 + PREFILL_MEM_FRACTION 0.88,
  C44); dryrun vs q1-102: node/IP/paths only [M]. JIT caches seeded node-local. Chain `chain_q1.yihou.sh` (pipeline A1 →
  flush → A2; login-host /metrics samplers -> v0521cap/runs/q1/). gpuidle ok 10:55, etcd ok.
- 102: k1 A2 running; pc40c (Track B patch, C40) armed behind it.

## k1 A2 (102, 10:22-11:45): e2g + HiCache 360, C44, WARM (scored)
- [M] 4304 ok / 2 err, **21,342/chip**, TTFT mean/p95 8.55 / 28.87 s, TPOT 12.4 ms.
- Rate/ISL decomposition [M from JSON qps.mean, tokens.input.mean]:
  | run | tok/s/chip | qps | mean ISL | TTFT |
  | e2g A2 C40 (450) | 22,114 | 1.181 | 148,711 | 5.30 |
  | e2j R2 C40 (450) | 21,681 | 1.171 | 147,086 | 5.21 |
  | X1 C44 (450, e2j stack 3rd run) | 21,773 | 1.207 | 143,345 | 9.02 |
  | k1 A1 C44 (360, fresh) | 20,729 | 1.163 | 141,612 | 7.40 |
  | k1 A2 C44 (360, warm) | 21,342 | 1.186 | 142,958 | 8.55 |
- Reading (suspended): HiCache 360 vs 450 at C44 warm: qps −1.7 % at equal ISL (X1 confounded by e2j extras/3rd run).
  C40→C44 raises qps only +0.4…+2 % while mean ISL falls ~3.5 % and TTFT grows +3 s → the C44 score is held down by
  TTFT (closed loop) and by the ISL mix. No evidence 360 helps; keep 450.

## TB5 (2026-10-12 12:17): pc40c burst smoke did not reach the cap
- Patch active [M]: decode container env YIHOU_DECODE_RUNNING_KV_SHARE=0.5, mounted decode.py has YIHOU-RUNSHARE.
- Burst 64 × 60k words = ~69.4k tokens each: 64/64 ok (max 178 s), 0 blocks. /metrics (2 s): kv_used max 0.25-0.30 of
  pool, transfer queue max 9-11/DP, prealloc queue 0 → in-flight never approached 0.5, so the cap was not exercised.
- Armed `post_pc40c_smoke.yihou.sh`: after A2, on the same stack, 64 × 200k words (~230k tok) → ~0.75 in flight without
  cap → must log blocks, complete all, no faults. C40 A1 started 12:16 (A2 ≈ 15:00).

## Q1 A1 (137, 11:21-12:43): e2g + decode max_running 96 + prefill mem 0.88, C44, FRESH (unscored)
- [M] 4337 ok / 2 err, **21,433/chip**, qps 1.195, ISL 142,474, TTFT mean/p95 **5.91 / 19.05 s**, TPOT 15.4 ms.
- vs fresh C44 on 102: k1 A1 20,729 (TTFT 7.40), e2c 20,524 (8.03), e2b4 20,347 (9.01) → +3.4…+5.3 % and −1.5…−3 s TTFT;
  best fresh C44 so far. Node differs (137 vs 102) — suspended until A2 (≈14:05) and a same-node check.
- pc40c profiling-phase blocks show running>0 (e.g. DP2 running=104,640 / 263,168, nonrunning 1.18-1.32 M, xfer 8-9)
  → the running-KV accounting works in decode [M].

## pc40c A1 (102, 12:16-13:37): e2g + runshare 0.5, C40, FRESH (unscored)
- [M] 4091 ok / 3 err, **20,361/chip**, qps 1.127, ISL 143,478, TTFT mean/p95 5.15 / 17.0 s, TPOT 14.5 ms.
- vs fresh C40 target runs on 102: e2g A1 19,640 (TTFT 7.61), e2i R1 20,371, e2j R1 19,335 → top of the fresh band, no
  regression signal. Scored A2 ≈ 15:00.

## Q1 A2 (137, 12:43-14:04): max_running 96 + prefill 0.88, C44, WARM (scored)
- [M] 4383 ok / 3 err, **21,862/chip**, qps 1.208, ISL 143,725, TTFT mean/p95 7.79 / 28.22 s, TPOT 12.6 ms.
- vs C44 warm on 102: k1 A2 21,342 (qps 1.186, HiCache 360), X1 21,773 (qps 1.207, e2j stack). Q1 ≈ X1 on qps; the A1
  TTFT gain (5.91 s) did not persist warm (7.79 s). Below e2g C40 22,114. Node differs (137) — same-node Q1 check = q3.
- Reading (suspended): at C44 warm the score sits at ~21.3-21.9k regardless of max_running 96 / prefill 0.88 / HiCache;
  the limiter is not the running cap. Next evidence: C48 on this stack (A3, started 14:04, ≈15:25).

## N1 (14:2x): decode radix cache would cut KV send, but is blocked by EAGLE [M code]
- Every AgentX turn ships the WHOLE context KV prefill→decode (decode has no radix cache): ~143k tok × 55 KB ≈ 7.9 GB
  per request; KV send residence 5.25 s at C44 (R2) is the largest TTFT phase [I/M].
- v0.5.21 has `--disaggregation-decode-enable-radix-cache` (decode prefix match + decode_prefix_len protocol), but
  `pd_disaggregation_hook.py:97-113` raises unless speculative_algorithm in (None, "DSPARK") → incompatible with our EAGLE
  MTP target. Not a lever without dropping MTP (user target keeps MTP). Remaining KV-send levers: mc4340prio copy-stream
  priority (R043), xferfix (weak evidence).
- Prepared `config.yihou.mc4340prio.sh` (102 only): image mc4340prio (sha256:aacc9aeb) + MC_HIP_STREAM_PRIORITY=high on
  both legs; chains via PRIO_BASE_CONFIG (runshare can use it as RUNSHARE_BASE_CONFIG). Dryrun vs q1-102: only image tag +
  the env [M].

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

## Q1 A3 C48 (137, 14:04-15:24, 3rd run on the warm q1 stack): **22,306/chip — new best, > 22,114**
- [M] 4425 ok / 2 err, qps 1.220, ISL 145,303, TTFT mean/p95 11.97 / 38.19 s, TPOT 12.1 ms.
- Decode /metrics (5 s): token_usage max 0.77 / **0.99** / 0.92 / 0.91 per DP; running max 10/14/10/18 (DP3 > old 16
  cap → max_running 96 needed at C48); transfer q max 13-18; prealloc 0; retract 0. → **decode KV capacity binds at C48**
  (DP1 0.99) — Track B (running keeps ≥ 50 %) and decode mem 0.96/0.97 become relevant.
- Trend on this stack (137): C44 21,862 → C48 22,306 (+2.0 %). Launched A4 = C52 on the same stack (4th run).

## TB7 (15:56): runshare v2 functional smoke (pc40d, 64 × ~231k tok): SUMMARY n=64 ok=64 failed=0 max_latency=616.1s
- block log lines 26,799; decode "pre-allocated usage" max 0.42 (≤ 0.5 cap); anti-starvation "stop" engaged (waited
  78-152 s for equal-size prompts); faults 0 [M]. C40 A1 started 15:56 (A2 ≈ 18:40).

## Q1 A4 C52 (137, 15:28-16:50, 4th run on the q1 stack): 21,418/chip (below C48)
- [M] 4552 ok / 2 err, qps **1.256** (C48 1.220, +3 %), ISL **135,507** (C48 145,303, −6.7 %), TTFT 15.82 / 51.97 s,
  TPOT 12.7 ms. Decode token_usage max 0.93/0.84/0.98/0.89; prealloc queue reached 2 on DP2 (first admission block);
  0 retractions.
- Score ≈ qps × ISL: more lanes raise qps but lower the mean ISL (fewer deep turns per lane within the hour) and TTFT
  grows ~4 s/step of +4 lanes → **C48 is the sweet spot** on this config (C44 21,862 / C48 22,306 / C52 21,418).
- q4 (decode 0.96, C48) now starts on 137 (q1 stack stopped by chain_q4).

## 2026-10-13 02:1x-02:55 (resumed on slog-003; previous login host lost ~17:18 on 10-12)
- Lost with the old login host: pc40d A2 driver (never started) and q4 chain (stopped after q1 stack stop). Job 207495 (102)
  TIMEOUT 20:01 10-12; oai job 208970 RUNNING on crsuse2-m2m-256 since 20:01 (idle). 137 now used by another user (llying).
- pc40d A1 (runshare v2, C40 fresh) [M]: 4122 ok / 0 err, 20,680/chip, qps 1.137, TTFT 5.26 / 18.29 s, TPOT 14.2 ms.
- q1 Little's-law residence C44/C48/C52 [M] (tools/residence.yihou.py): prefill queue 2.0/3.6/5.2 s, KV send 4.5/6.5/8.2 s,
  decode running ~12-13 s flat → the prefill side (queue + KV send) is the limiter; HiCache was 450 GB in all three.
- Pack-up: /home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-cap-c44c48.packup_20261013 (30 MB, no file > 4 MB).
- Session export: /home/yihou/claude-session-export.yihou.20261013.tar.zst (30 MB, chmod 600) + .RESTORE.md.

## 2026-10-13 session 2 (new machine, driven over ssh; drivers ON the compute node)
- Holds (amd-oai, 24 h, 8x MI355X verified): 211169 crsuse2-m2m-085 (10.245.153.145), 211170 crsuse2-m2m-026
  (10.245.153.227), 211171 crsuse2-m2m-274 (10.245.157.34). Image tar loaded on all three: ID sha256:9d755dd6 (= 102).
  JIT caches seeded from cache/102 (aiter 9d755dd6 + triton). 085/274 carry guanchen's orphan container crusoe-sgl-q36
  (pool_mtp.sh loop; his burst jobs PREEMPTED) — user asked whether to docker stop it; not touched.
- New harness files (no existing file edited): config.yihou.mc4340n2.sh (+026/085/274 image ID), config.yihou.expert8leg.sh
  (E8_LEG=decode|prefill: role-only mounts of an overlay config.json over $MODEL/config.json + yihou-expert8-nextn.patch
  applied to the v0.5.21 image sources [patch base sha256 == image files, verified]; that leg EP4,
  --disable-shared-experts-fusion, --max-total-tokens 3670016 = the 26k runs' cap), mounts.yihou.expert8.tsv,
  tools/audit_live.v3.yihou.py (+{ROLE}_EXPERT8/{ROLE}_MAX_TOTAL_TOKENS), run.v3 (audit v3), pipeline.v3 (no spur shim
  on the node), chain_cap2.yihou.sh (generic: A1 fresh C1 -> flush -> A2 C2 -> flush -> A3 C3, runs ON the node with the
  local ssh shim $N/shimbin/ssh, N=/shared_nfs/yihou_glm52/v0521cap2_20261013), env.{x8d-026,x8p-274,tb56-085}.
  Dryrun token-diff vs q1-137: x8d/x8p differ only by the 5 expert8 mounts, EP 1->4, the two flags on that leg (+ node
  paths/IP); tb56 only by no --hicache-size (ratio 1.5 = 154 GB) + runshare2 mount/env (+ node paths/IP).
- Research (workflow, adversarially verified): the "26k" = v0.5.19 r001-hc274 26,462 / r001-hc450chunk512 26,071 at C56,
  8 experts on BOTH legs, max-total-tokens 3670016 both legs; same-model HiCache 154 GB arm A 23,892. Old sweep with C56:
  t2f (v0.5.19 cross-node, HiCache 154 GB, max_running 128) C40 19,798 / C56 14,675 (-25.9 %) / C72 8,763 / C96 3,028.
- Rounds: x8d (026, decode-only expert8, q1 config, C48 A1 -> C48 A2 -> C52 A3) launched 06:28.
  x8p (274, prefill-only expert8, same) and tb56 (085, q1 config + runshare v2 0.5 + HiCache ratio 1.5, C56 A1 -> C56 A2
  -> C64 A3) prepared; waiting on the guanchen-container decision for 085/274.

## 2026-10-13 08:2x prefill_complete (user ~08:00: "decode allocates after prefill finished; HiCache 370, decode gmu 0.96, C48, new node")
- Research workflow wf_837ba919-06f (verified): --disaggregation-decode-allocation-policy prefill_complete (PR #40703, in
  image). Needed on BOTH legs (mismatch -> every request fails "policy mismatch"). REQUIRES prefill HiCache write_back
  (pd_disaggregation_hook.py:231-247 rejects write_through) -> forced second variable. sglang sets prefill
  optimistic_prefill_attempts=1 itself. Prefill holds finished requests' device KV (parked in inflight queue) until decode
  allocates; transfer no longer overlaps chunked prefill. EAGLE/DPA not rejected; KV pool must be DSA-only (expected).
  Mounted conn.yihou-regbase.py keeps the image's defer logic (only diagnostic hunks).
- Files: config.yihou.pcomplete.sh (overlay), env template $N/env.pc48-NODE.template (pending a node). Dryrun vs q1-137:
  prefill write_through->write_back, hicache 450->370, decode mem 0.95->0.96, policy flag on both legs. Nothing else.
- Node: 4th oai job 211594 PENDING (QOSGrpNodeLimit). 135-139 all busy (llying 3p1d on 135/137/138; limou vLLM on 136 at
  100 % util / 96 % VRAM, started 07:03 -> real utilisation, not killed; 139 no dockerd).

## 2026-10-13 ~16:05 two inserted experiments (user ~15:55)
- hc50 (user: "HiCache 50 GB at C40 on the best v0.5.21 config (e2g), everything else unchanged" — HiCache sensitivity):
  crsuse2-m2m-136 (free: limou's vLLM gone; 135/137 = llying 3p1d, untouched), plain ssh, image = 136's mc4340 build 9872bbda
  (engine.so identical to 102's per config notes). env.hc50-136 = e2g overrides + PREFILL_HICACHE_SIZE=50; dryrun vs e2g-102:
  only 450 -> 50 (+ node ro/rw mounts). Driver chain_cap2l.yihou.sh on login host spur006 (136 NFS is ro). Started 15:59.
- x128p (user: "prefill experts halved, 256 -> 128"): new truncation patch (NOT expert8): yihou_expertn.py + loader.py hook
  (load experts 0..127, slice gate + e_score_correction_bias to 128), overlay config n_routed_experts=128 +
  yihou_synthetic_keep_experts=128; own sigmoid top-8 routing; EP1; --disable-shared-experts-fusion (sglang allows fusion only
  for 256/384 experts). config.yihou.expertnleg.sh, mounts.yihou.expert128.tsv, env.x128p-085. Dryrun vs q1-137: only the 3
  prefill mounts + the fusion flag. Armed on 085 behind tb56n (wait_then -> stop_then_chain), C48 A1 -> A2.
- User 16:0x: use oai nodes (up to 4), 135-137 only if not llying's. 4th oai hold 213048 PENDING (QOSGrpNodeLimit).

## 2026-10-14 summary table written: /shared_nfs/yihou_glm52/v0521cap2_20261013/summary.v0521cap2.yihou.md (all session runs, score/qps/ISL/TTFT/TPOT/GPU+CPU hit/errors).

## 2026-10-14 ~02:30 packups (4, cold-read verified PASS): /home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-{prefill-kv-bottleneck,runshare-highconc,prefill-complete,hicache-sensitivity}.packup_20261014 (raw server logs indexed, not copied).

## 2026-10-14 ~08:30 OffloaderV2 feasibility (workflow wf_d6a67068-1d1) + H2D gate
- Assessment saved locally /tmp/yihou/offv2_assess.md (+ impl/upstream/cost reports). OffloaderV2 (PR #8034) in image == upstream main;
  cpu mode works at TP4; blocking patch needed: is_shuffled attr lost in create_device_tensor (aiter kernel choice). Per MoE layer per GPU
  1.21 GB; 75 layers 84.7 GiB; 50 % offload ~40 GiB freed -> ~3.22 M KV tok; needs >= ~40 GB/s concurrent H2D to hide at p50.
- H2D microbench (089 GPUs 0-3, image torch, pinned 1.21 GB, 8 reps; /offload/h2d_bench.089.json): alone 45.2-46.0 GB/s per GPU;
  4 concurrent 34.6/36.4/37.3/46.3; 4 concurrent H2D+D2H 17.8-19.0 per GPU per direction; 4 concurrent with 64 GiB extra pinned 29.9-31.2.
  -> gate band 25-40 GB/s (H2D only) and <25 with concurrent D2H (HiCache write-through) -> per plan cap at 25 % offload; S0 (1/8) first.
- Packup day2 done: glm52.v0521-prefill-mem-hicache-policy.packup_20261014 (402 files, verified PASS) + glm52.v0521-CONCLUSIONS.yihou.md.

## 2026-10-14 08:18 OffloaderV2 S0 (user "开始")
- Patch YIHOU-OFFLOAD-ATTRS: offloader.py _CpuParamOffloader.create_device_tensor copies is_shuffled from the host param to the device copy
  (post-load restore keeps the Parameter object -> host param keeps the flag; .to() result loses it; aiter fused_moe.py:364/1150/1256/1316
  picks kernels by getattr(w,"is_shuffled") — verified in the 136 image; 136 offloader.py sha == image-9d755dd6 dd22b593…).
  Files: /offload/{offloader.yihou-attrs.py, offloader.py.orig, build_offpatch.yihou.py}, /mounts.yihou.offload-attrs.tsv (prefill only),
  /config.yihou.offload.sh (OFF_G/OFF_K/OFF_STEP, --offload-mode cpu, prefill leg only).
- S0 = G8 K1 step1 on the pf90 config (prefill mem 0.90). ofs0c (sim OFF) correctness vs cx90; ofs0 (sim 3.61) perf C48 A1->A2 vs pf90.
  Dryrun diffs = only the intended deltas. Scheduled on 136 after wts64 (after_wts64_s0.yihou.sh, login host); pf94 moved after S0.
- H2D gate: concurrent 4-GPU H2D 34.6-46.3 GB/s; with simultaneous D2H 17.8-19.0 -> 50 % offload ruled out; S0 then <= 25 %.
