# Poll ledger — task v0521cap

## Poll 2026-10-12 08:11 UTC
- Workflow wf_f78a24f2-89b (research, relaunched 08:0x after session loss): only R1 + R2 started so far (agent
  concurrency apparently 2 on this login host); R3-R5 queued; no research files yet.
- Job 207495 RUNNING (21:15 left). 102: e2j stack up (idle after X1). `yihou-msampler-x1` still sampling (self-exits
  ~08:38). FIRST SIGHTING: new CPU-only `yihou-hostsample-e2jR2-…-e2j-router` created 08:05:08 (thread_cpu on router,
  3600 s, output dir from the 04:26 e2jR2 run) — origin unclear, no host process found; harmless (no devices). Next poll:
  stop it if still up and before any benchmark.
- 137: 8 GPUs 0 VRAM; foreign containers dev_xb (sleep) + crusoe-vector only; image not loaded yet (R4 queued).

## Poll 2026-10-12 08:31 UTC
- User 08:15: no 13x nodes; two amd-oai nodes. 137: image removed, no containers of ours (not watched further).
- Research wf_f78a24f2-89b (resumed w/o the 137 track): R1/R2/R3/R5 reports written; verify R1 done, R2/R3 verifying.
- Job 207495 RUNNING (20:55 left); 2nd oai hold 208970 PENDING (QOSGrpNodeLimit).
- 102: k1 pipeline in gpuidle gate since 08:16 (VRAM max 273 GB draining after e2j stop — expected 28-35 min). The
  stray e2jR2 router hostsample + x1 sampler were removed before k1 (resolved).

## Poll 2026-10-12 08:51 UTC
- Research workflow done (8/8). Job 207495 RUNNING (20:35 left); 208970 PENDING (QOSGrpNodeLimit).
- 102: k1 gpuidle ok 08:45:50, etcd ok, prefill bring-up in progress (expected ~7 min + decode ~6 min). q1 env +
  dryrun ready (`env.q1-102.yihou.sh`). 137 not used (user directive).

## Poll 2026-10-12 09:11 UTC
- Burst job 209081 RUNNING on crsuse2-m2m-174 (10.245.152.55, 8 GPUs idle, NFS rw, 2751 GB RAM) -> patch/debug node.
  docker load of the mc4340 image started 09:11 (log v0521cap/logs/load_image.174.log). 208970 (oai) still PENDING.
- 102: k1 A1 warmup since 08:58, metrics sampler running, 0 fault/Traceback lines in decode log.

## Poll 2026-10-12 09:31 UTC
- PROBLEM (acted on immediately, node lost): burst job 209081 PREEMPTED at 09:30:43 after 20 min, during pc40 prefill
  bring-up on 174. spur exec refused (job CANCELLED); pc40 prefill log stopped growing at 09:30:33 -> containers gone
  with the job (cannot verify directly). Killed the pc40 chain + its run.v2 up (PID-specific; k1 bench untouched).
- Resubmitted burst hold: job 209141 (yihou-burst-patch2). Burst QOS is preemptible: expect repeats.
- 102: k1 A1 profiling since 09:15, 0 fault lines. 208970 (oai) still PENDING (QOSGrpNodeLimit).

## Poll 2026-10-12 09:51 UTC
- k1 A1 (C44, HiCache 360) healthy, 0 fault lines; profiling 09:15 -> ~10:16. /metrics so far: decode token_usage
  max 0.64-0.85, running max 16 (DP0, cap hit) / 15 / 13 / 13, transfer q max 9-13, retract 0; prefill host pool 360 GB
  (8.01 M tok/rank) already full on 3 of 4 DPs by ~35 min.
- 209141 (burst) PENDING (Resources); 208970 (oai) PENDING (QOSGrpNodeLimit).

## Poll 2026-10-12 10:11 UTC
- Burst 209141 got crsuse2-m2m-205, but 205 runs a FOREIGN GPU container `crusoe-sgl-q36` (since 09:16, /dev/kfd+dri):
  a cross-node "drain pool" (pool.sh) that keeps booting sglang servers on GPUs 0-5 in VRAM gaps. Not usable without
  disturbing another user -> stopped my image load there, scancel 209141, resubmitted burst job 209186 excluding
  102/174/205. Rule for any new node: check foreign GPU containers before loading/launching.
- k1 A1 profiling ends ~10:16; 0 fault lines. 208970 (oai) PENDING (QOSGrpNodeLimit).

## Poll 2026-10-12 10:31 UTC
- 029 (burst 209186): pc40b prefill bring-up since ~10:23 (only our containers on the node), 0 faults.
- 102: k1 A2 (warm, scored) running since 10:22, 0 faults. 208970 PENDING.

## Poll-note 2026-10-12 10:38 UTC
- Burst 209186 (029) CANCELLED at 20:00 elapsed (burst QOS PreemptMode=cancel). pc40b driver stopped. pc40c armed on
  102 behind k1 (waits for k1 chain done). No further burst requests.

## Poll 2026-10-12 10:56 UTC
- 137: q1 prefill bring-up since 10:55, 0 faults; foreign containers: dev_xb (sleep, no GPU use) + crusoe-vector only.
- 102: k1 A2 running, 0 faults; pc40c armed behind k1. 208970 PENDING (QOSGrpNodeLimit).

## Poll 2026-10-12 11:11 UTC
- 137 q1: prefill ok 11:09:38 (14 min, cold-ish host alloc), decode bring-up next; 0 faults; no foreign GPU containers.
- 102 k1 A2 running, 0 faults. 208970 PENDING.

## Poll 2026-10-12 11:31 UTC
- 137 q1 A1 bench since 11:21 (max_running 24/DP, extra_slots 48, prefill mem 0.88 verified in logs), 0 faults, no
  foreign GPU containers. 102 k1 A2 profiling since 10:38 (ends ~11:38, JSON ~11:45), 0 faults. 208970 PENDING.

## Poll 2026-10-12 11:51 UTC
- 102: k1 done (A2 21,342); pc40c in gpuidle gate (VRAM draining, expected ~30 min). 137: q1 A1 running, 0 faults, no
  foreign GPU containers. 208970 PENDING.

## Poll 2026-10-12 12:11 UTC
- 102 pc40c: gpuidle 12:01, prefill ok 12:07, decode bring-up. 137 q1 A1 running, 0 faults, no foreign GPU containers.
  208970 PENDING.

## Poll 2026-10-12 12:31 UTC
- FIRST SIGHTING (pc40c, 102): runshare cap fires heavily during AgentX warmup (max_tokens=1, decode idle):
  26,144 log lines (every 100th block) ≈ 2.6 M blocked admission checks in 14 min ≈ one per scheduler loop. Examples:
  DP1 nonrunning=1,125,888 cand=688,430 cap=1,381,088 running=0 xfer=9; DP1 nonrunning=705,216 cand=704,353 xfer=1.
  i.e. a few 0.69-0.70 M-token prompts are held behind in-flight KV (FIFO head-of-line, verify gap G3). Warmup still
  progresses (385/444 at 810 s, 0 errors); 0 faults. /metrics decode gauges stay 0 in warmup (only refreshed by decode
  batches) → not usable to cross-check here. Next poll: check profiling-phase blocks (running>0 must appear), TTFT.
- 137 q1 A1 running, 0 faults, no foreign GPU containers. 208970 PENDING.

## Poll 2026-10-12 12:51 UTC
- 12:31 first sighting (runshare warmup blocking) resolved: profiling phase blocks are sparse and show running>0; no
  stall. pc40c A1 profiling, 0 faults. 137: q1 A2 since 12:43, 0 faults, no foreign GPU containers. 208970 PENDING.

## Poll 2026-10-12 13:11 UTC
- 102 pc40c A1 profiling (ends ~13:33), 0 faults; 137 q1 A2 running, 0 faults, no foreign GPU containers.
  Armed: a3_q1c48 (137), chain_q3 (102). 208970 PENDING.

## Poll 2026-10-12 13:31 UTC
- 102 pc40c A1 finishing (~13:38 JSON), 0 faults; 137 q1 A2 running, 0 faults, no foreign GPU containers. 208970 PENDING.

## Poll 2026-10-12 13:51 UTC
- 102 pc40c A2 since 13:37 (≈15:00), 0 faults; 137 q1 A2 (≈14:05), 0 faults, no foreign GPU containers. 208970 PENDING.

## Poll 2026-10-12 14:11 UTC
- 137: q1 A2 done (21,862); A3 C48 running since 14:04, 0 faults, no foreign GPU containers. 102 pc40c A2 running,
  0 faults. 208970 PENDING.

## Poll 2026-10-12 14:31 UTC
- 102 pc40c A2 running (≈15:00), 0 faults; 137 q1 A3 C48 running (≈15:25), 0 faults, no foreign GPU containers. 208970 PENDING.

## Poll 2026-10-12 14:51 UTC
- 102 pc40c A2 finishing (≈15:00), 0 faults; 137 q1 A3 C48 running, 0 faults, no foreign GPU containers. 208970 PENDING.

## Poll 2026-10-12 15:11 UTC
- 102: pc40c smoke2 PASS (cap holds 0.50, 64/64 ok, 0 faults); pc40d in gpuidle gate. 137: q1 A3 C48 finishing, 0 faults,
  no foreign GPU containers. 208970 PENDING.

## Poll 2026-10-12 15:31 UTC
- 137: q1 A3 C48 = 22,306 (new best); A4 C52 running since 15:28, 0 faults/retracts, no foreign GPU containers.
- 102: pc40d etcd ok 15:29, prefill bring-up. 208970 PENDING.

## Poll 2026-10-12 15:51 UTC
- 102 pc40d: v2 active (container env + mounted file verified), burst smoke running: blocks with waited=78-152 s "stop"
  (anti-starvation engaged on equal-size 231k prompts — expected in this synthetic burst). 137 q1 A4 C52 running, 0
  faults/retracts, no foreign GPU containers. 208970 PENDING.

## Poll 2026-10-12 16:11 UTC
- 102 pc40d A1 running, 0 faults. 137 q1 A4 C52 running, 0 faults/retracts, no foreign GPU containers. q4 armed. 208970 PENDING.

## Poll 2026-10-12 16:31 UTC
- 102 pc40d A1, 0 faults; 137 q1 A4 C52 (≈16:50), 0 faults/retracts, no foreign GPU containers. 208970 PENDING.

## Poll 2026-10-12 16:51 UTC
- 137: q1 A4 C52 = 21,418 (below C48 22,306); q4 chain proceeding (stop q1 → drain → up). 102 pc40d A1, 0 faults.
  208970 PENDING.

## Poll 2026-10-13 (slog-003)
- 207495 (102) ended (TIMEOUT 10-12 20:01). 208970 holds crsuse2-m2m-256 (idle). 137 taken by another user. No runs, no
  drivers (paused for the user's session migration; login host is memory-monitored).
- CORRECTION 03:12: 208970 also ended (COMPLETED 2026-10-13 02:23:35; hold script ended — cause unknown, possibly the
  login-host session loss). We hold NO node now.

## 2026-10-13T06:50:54Z poll (session 2)
- x8d (026): up in progress, prefill not yet healthy at ~18 min (HiCache 450 registration; q1 took 15 min). prefill max_total_num_tokens 2,433,088 (256-expert leg).
- x8p (274): up in progress; prefill = expert8 confirmed (304 YIHOU_EXPERT8 lines, routed=8 ep=4 local=2), prefill max_total_num_tokens 3,670,016 (cap; vs 2,433,088 on a 256-expert prefill at 0.88).
- tb56 (085): prefill ok 06:46 (29x20s), decode a1 starting; no fault lines.
- First sighting: squeue from spur006 failed once "failed to connect to spurctld" (controller flaky; chains run on the nodes, unaffected).

## 2026-10-13T07:10Z poll
- All three in bench A1, AUDIT PASS, 0 Traceback/MAF lines. tb56 A1 C56 started 06:53, x8d A1 C48 07:01, x8p A1 C48 07:03.
- x8d decode = expert8 confirmed incl. MTP draft (312 YIHOU_EXPERT8 lines, 8 role=draft); max_total_num_tokens decode 3,670,016 (256-expert decode 0.95: 2,762,176, +32.9 %), prefill 2,433,088.
- x8p prefill = expert8 (304 lines), prefill 3,670,016 (+50.8 % vs 2,433,088), decode 2,762,176.
- tb56 prefill 2,433,088 / decode 2,762,176 (HiCache ratio 1.5).
- squeue via spur006 refused again (spurctld); resolved by retry — controller flakiness, chains unaffected.

## 2026-10-13T08:30Z poll — A1 (fresh, unscored) results; all A2 running, no faults
- x8d A1 C48 (decode expert8, 026): 20,230/chip, qps 1.146, ISL 140,330, TTFT 14.03/46.28 s, TPOT 14.0 ms.
- x8p A1 C48 (prefill expert8, 274): 25,347/chip, qps 1.392, ISL 144,802, TTFT 5.03/16.05 s, TPOT 14.5 ms.
- tb56 A1 C56 (runshare v2 + HiCache 154 GB, 085): 17,545/chip, qps 1.038, ISL 134,256, TTFT 28.22/85.95 s, TPOT 15.0 ms.
- refs (137): q1 A1 C44 fresh 21,433 (TTFT 5.91); q1 A3 C48 warm 22,306 (TTFT 11.97). Fresh vs warm and node differ.

## 2026-10-13T08:50Z poll — x8d A2 died: HARDWARE (026 GPU3 HBM ECC)
- x8d A2 (C48) bench FAIL 08:41; both legs logged "Memory access fault by GPU node-5 ... Reason: DRAM ECC failure / SRAM ECC failure" at 08:39:57-58, then Fatal Python errors.
- amd-smi metric --ecc on 026: GPU3 (KFD node 5, 0002:00:04.0, a PREFILL GPU) UMC correctable 13,198,148, deferred 2; GPUs 0-2,4-7 all 0. -> x8d A1 (20,230) ran on this GPU too: treat as suspect.
- Actions (own jobs only): x8d containers stopped (0 left); released 211170 (026); cancelled pending 211594 and resubmitted 211763 with -x crsuse2-m2m-026 (amd-oai, 24 h). 211763: 4 JobLaunchFailure restarts ("gpu/resource allocation mismatch"), now PENDING (Resources), SchedNodeList m2m-147. Watch next poll.
- x8p (274) A2 and tb56 (085) A2 running, 0 faults.

## 2026-10-13T10:10Z poll
- SCORED: x8p A2 C48 (prefill expert8, 274) 25,905/chip, qps 1.409, ISL 146,053, TTFT 5.45/17.02 s, TPOT 13.2 ms (A1 25,347).
- SCORED: tb56 A2 C56 (runshare v2 + HiCache 154 GB, 085) 17,611/chip, qps 1.053, ISL 132,674, TTFT 31.08/98.68 s, TPOT 12.8 ms (A1 17,545). A3 C64 running.
- x8d2 (206): A1 C48 running since 09:50; decode expert8 incl. draft (312 / 8 lines), decode 3,670,016 / prefill 2,433,088.
- pc48 (274) died at prefill startup: ValueError "prefill_complete does not yet support attention tensor parallelism greater than one".
  Cause [code]: parallel_hook.handle_deprecated_dp_attention (runs first) rewrites --dp-size 4 --enable-dp-attention to
  attn_dp_size=4, dp_size=1, enable_dp_attention=False; _validate_prefill_complete (pd_disaggregation_hook.py:213) still
  computes cfg.dp_size if cfg.enable_dp_attention else 1 = 1 != tp 4. Upstream main still has the same line (checked via gh 10:0x).
  Fix: mount pd_disaggregation_hook.yihou-attndp.py (base sha == image ad3a2f7a; one line: use cfg.attn_dp_size) on both legs
  -> round pc48b launched 10:11; prefill passed validation (no ValueError), HiCache 370 GB host pool allocated.

## 2026-10-13T10:30Z poll
- pc48b pipeline STOP at audit 10:24 = known false negative of mine: same hook TSV in PREFILL_ and DECODE_MOUNTS_ADD trips the
  role-only rule (2 FAILs, 93 PASS). Stack healthy (smoke ok). server-info: allocation_policy=prefill_complete on BOTH legs;
  prefill optimistic_prefill_attempts=1, hicache write_back 370; decode mem 0.96 -> max_total_num_tokens 2,816,896 (0.95: 2,762,176).
  Continued on the live stack with cont_bench.yihou.sh: A1 C48 started 10:30, then flush -> A2 C48. (Next time: put
  both-leg mounts in YIHOU_MOUNTS, not *_MOUNTS_ADD.)
- x8d2 A1 C48 running (since 09:50), tb56 A3 C64 running (since 09:52); no faults.

## 2026-10-13T11:29Z poll
- x8d2 A1 C48 fresh (decode expert8, 206): 22,511/chip, qps 1.231, ISL 145,265, TTFT 9.95/37.04 s, TPOT 14.3 ms. A2 started 11:16.
- tb56 A3 C64 still running at 11:29 (A1/A2 took 89-90 min -> due ~11:25); pc48b A1 running. No faults.

## 2026-10-13T11:49Z poll
- tb56 DONE 11:33. A3 C64 warm: 15,070/chip, qps 0.926, ISL 129,104, TTFT 51.22/135.36 s, TPOT 11.1 ms.
  tb56 series (runshare v2 0.5 + HiCache 154 GB, q1 config): C56 A1 17,545 / A2 17,611 / C64 A3 15,070 (-14.4 % vs C56 A2).
  t2f (v0.5.19 cross-node, no patch): C40 19,798 / C56 14,675 / C72 8,763. No same-config no-patch control yet. 085 stack left running, idle.
- x8d2 A2 and pc48b A1 running; no faults.

## 2026-10-13T12:15Z poll
- pc48b A1 C48 fresh (prefill_complete + write_back + HiCache 370 + decode 0.96): 18,225/chip, qps 1.040, ISL 139,272, TTFT 17.42/68.59 s, TPOT 15.3 ms. A2 started 11:54.
- x8d2 A2 running (since 11:16); tb56n up in progress (prefill ok 12:11). No faults.

## 2026-10-13T12:55Z poll
- SCORED: x8d2 A2 C48 warm (decode expert8, 206): 23,591/chip, qps 1.286, ISL 145,686, TTFT 10.43/33.44 s, TPOT 11.9 ms.
  Bottleneck pair (C48 warm A2): prefill-only expert8 x8p 25,905 (274) vs decode-only expert8 x8d2 23,591 (206) vs q1 22,306 (137, A3).
  Different nodes; expert8 changes both KV capacity and MoE compute on that leg.
- x8d2 A3 C52 started 12:39; pc48b A2 and tb56n A1 running. No faults.

## 2026-10-13T13:35Z poll
- SCORED: pc48b A2 C48 warm (prefill_complete [+attndp fix] + HiCache 370 write_back + decode 0.96; 274): 20,002/chip, qps 1.112,
  ISL 142,959, TTFT 16.57/52.70 s, TPOT 12.2 ms (A1 18,225). vs q1 C48 A3 22,306 (-10.3 %). 4 variables changed vs q1; no attribution.
  274 now idle (pc48b stack left running).
- x8d2 A3 C52 and tb56n A1 running; no faults.

## 2026-10-13T13:55Z poll
- tb56n A1 C56 fresh (NO-patch control, 085): 17,791/chip, qps 1.057, ISL 133,693, TTFT 27.76/87.55 s, TPOT 14.9 ms.
  vs tb56 A1 (runshare v2, same node, same config otherwise) 17,545 / qps 1.038 / TTFT 28.22. A2 C56 started 13:45.
- x8d2 A3 C52 running; 274 idle (pc48b stack up), awaiting the user choice. No faults.

## 2026-10-13T14:15Z poll
- x8d2 DONE 14:02. A3 C52 warm: 21,713/chip, qps 1.268, ISL 136,044, TTFT 16.12/52.11 s, TPOT 11.3 ms (C48 A2 23,591; q1 C52 A4 21,418).
- 206 and 274 idle (stacks up); tb56n A2 C56 running. No faults.

## 2026-10-13T15:15Z
- SCORED: tb56n A2 C56 warm (NO patch control, 085): 18,016/chip, qps 1.041, ISL 137,357, TTFT 30.09/92.78 s, TPOT 12.5 ms.
  vs tb56 A2 (runshare v2) 17,611 / qps 1.053 / ISL 132,674 / TTFT 31.08. tb56n A3 C64 started 15:11.
- Launched (user "跑"): pc48e on 274 (early + write_back + HiCache 370 + decode 0.96, control of pc48b) and x8pc on 206
  (prefill expert8 with --max-total-tokens 2433088, split of x8p). Dryrun diffs = only the intended deltas. Both C48 A1 -> A2.

## 2026-10-13T16:56Z poll
- tb56n DONE 16:48. A3 C64 warm (NO patch): 15,108/chip, qps 0.930, ISL 128,977, TTFT 50.84/134.50 s, TPOT 11.5 ms.
  Track B same-node A/B (085): C56 A2 runshare 17,611 vs none 18,016; C64 A3 runshare 15,070 vs none 15,108.
- x128p started on 085 16:49 (tb56n stopped cleanly); prefill log shows YIHOU_EXPERTN keep lines.
- pc48e/x8pc/hc50 A1 running; no faults.

## 2026-10-13T17:16Z poll
- x8pc A1 C48 fresh (prefill expert8, KV capped 2,433,088; 206): 21,591/chip, qps 1.202, ISL 142,780, TTFT 11.41/38.23 s, TPOT 14.0 ms.
  vs x8p A1 (uncapped 3,670,016; 274) 25,347 / qps 1.392 / TTFT 5.03. A2 started 16:59.
- x128p up OK: prefill max_total_num_tokens 3,318,400 (+36 % vs 2,433,088), A1 C48 started 17:16.
- pc48e A1 and hc50 A1 running; no faults.

## 2026-10-13T17:36Z poll
- pc48e A1 C48 fresh (early + write_back + HiCache 370 + decode 0.96; 274): 19,313/chip, qps 1.107, ISL 138,615, TTFT 15.18/52.97 s, TPOT 14.9 ms.
  vs pc48b A1 (prefill_complete, same node/config otherwise) 18,225 / qps 1.040 / TTFT 17.42. A2 started 17:26.
- x8pc A2, x128p A1, hc50 A1 running; no faults.

## 2026-10-13T17:56Z poll
- hc50 A1 C40 fresh (e2g + HiCache 50 GB; 136): 17,733/chip, qps 1.044, ISL 134,925, TTFT 9.26/33.71 s, TPOT 14.2 ms.
  vs e2g A1 C40 fresh (HiCache 450; 102) 19,640 / qps 1.100 / ISL 141,791 / TTFT 7.61. A2 started 17:52. Node + image build differ (136/9872bbda).
- pc48e A2, x8pc A2, x128p A1 running; no faults.

## 2026-10-13T18:36Z poll
- SCORED: x8pc A2 C48 warm (prefill expert8, KV capped 2,433,088; 206): 22,489/chip, qps 1.234, ISL 144,759, TTFT 12.28/43.83 s, TPOT 12.4 ms.
  Split of x8p: uncapped (3,670,016) 25,905 vs capped 22,489 vs q1 (real 256 experts, 2,433,088) 22,306. Nodes differ (274/206/137).
  206 idle now (x8pc stack up).
- pc48e A2, hc50 A2, x128p A1 running; no faults.

## 2026-10-13T18:56Z poll
- SCORED: pc48e A2 C48 warm (EARLY + write_back + HiCache 370 + decode 0.96; 274): 19,158/chip, qps 1.097, ISL 138,769, TTFT 17.79/60.45 s, TPOT 12.1 ms.
  Same-node A/B (274): prefill_complete pc48b 20,002 vs early pc48e 19,158 (A2); A1 18,225 vs 19,313. Both far below q1 22,306
  (write_through + 450 + decode 0.95; 137). 274 idle now.
- x128p A1 C48 fresh (prefill 128 experts, KV 3,318,400; 085): 24,316/chip, qps 1.346, ISL 143,619, TTFT 5.64/16.74 s, TPOT 15.1 ms. A2 started 18:40.
- hc50 A2 running; no faults.

## 2026-10-13T19:12Z
- SCORED: hc50 A2 C40 warm (e2g + HiCache 50 GB; 136, image 9872bbda): 20,112/chip, qps 1.119, ISL 142,664, TTFT 8.41/30.92 s, TPOT 11.9 ms.
  vs e2g A2 (HiCache 450; 102, 9d755dd6) 22,114 / qps 1.181 / ISL 148,711 / TTFT 5.30. -9.1 %. Node/build differ. 136 stack left up.

## 2026-10-13T20:16Z poll
- SCORED: x128p A2 C48 warm (prefill 128 experts, prefill KV 3,318,400; 085): 25,297/chip, qps 1.387, ISL 145,013, TTFT 6.23/19.47 s, TPOT 13.3 ms.
  Prefill-KV series at C48 A2: 2,433,088 -> q1 22,306 (256 exp) / x8pc 22,489 (8 exp); 3,318,400 -> x128p 25,297 (128 exp); 3,670,016 -> x8p 25,905 (8 exp).
- All rounds done. 085/136/206/274/089 idle (stacks up on 085/136/206/274). Awaiting the user for the next experiments.

## 2026-10-14T03:25Z poll — NODE FAILURE 274
- Job 211171 (crsuse2-m2m-274) ended JobState=NODE_FAIL Reason=NodeDown (EndTime 03:10:51; sacct COMPLETE -1:0). pf88 server logs stop at
  02:32:36 (16 min into A1 C48) -> pf88 lost, no result. Second node lost this session after 206 (211763, JobHoldMaxRequeue ~01:30).
- Replacement hold 214829: PENDING JobLaunchFailure ("1 gpu/resource allocation mismatch"), retrying. Plan: rerun pf88 there when it starts.
- pf90 (089) and pf92 (085) A1 running; no errors.

## 2026-10-14T03:40Z
- User ~03:30: new round wts48 = q1 config + PREFILL_HICACHE_WRITE_POLICY=write_through_selective (threshold 2 vs 1); keep 4 nodes;
  fallback 135-139 below llying, not 136. Prepared env.wts48-NODE.template (dryrun vs q1: only the write policy).
- hc50 stack on 136 stopped (0 containers). 135/137/138 = llying 3p1d; 139 no dockerd (inactive) -> unusable.
- sbatch failing: "refusing to re-forward an already-forwarded request", then "raft propose failed: ... snapshot not found" (x2).
  Holding: 085 (pf92), 089 (pf90); 214829 PENDING (QOSGrpNodeLimit / JobLaunchFailure).

## 2026-10-14T03:37Z
- User "使用136": wts48 (q1 config + write_through_selective) launched on 136 at 03:35 via chain_cap2l on spur006 (image 9872bbda).
  Launch redirect slip: chain stdout went to /dev/null (wts48.chain.log empty) -> monitor wts48/status.tsv, wts48.pipeline.log, JSONs.

## 2026-10-14T03:55Z poll: pf90 A1 C48 fresh 21,859 (qps 1.219, ISL 142,517, TTFT 9.57/31.32, TPOT 15.7); A2 started 03:52. pf92 A1 running; wts48 prefill ok 03:46. 214829 JobLaunchFailure retries.

## 2026-10-14T04:16Z poll: pf92 A1 C48 fresh 23,003 (qps 1.267, ISL 144,228, TTFT 7.49/23.90, TPOT 15.6); pf92 A2 03:57, pf90 A2 03:52, wts48 A1 03:58 running; no errors.

## 2026-10-14T04:20Z re-plan (user ~04:20: pc64 = pc48b at C64; stop runshare work — none running)
- No free node (214829 JobLaunchFailure; sbatch raft errors). pf90 (089) and pf92 (085) chain drivers killed (A2 benches keep
  running, no A3). Handovers armed (after_a2_then.yihou.sh): 089 -> pc64 (config.yihou.pcomplete2.sh = pcomplete + attndp hook via
  YIHOU_MOUNTS; C64 A1 -> A2), 085 -> pf88b (q1 config 0.88 rerun; C48 A1 -> A2 -> C52 A3). Dryruns: pc64 == pc48b argv; pf88b == q1 argv.

## 2026-10-14T05:11Z poll — 085 LOST
- Job 211169 (085) ended 05:07:15 JobState=NODE_FAIL Reason=NodeDown, RunTime 1-00:00:24, TimeLimit shown 1-00:00:00
  (the 36 h extension did not hold; QOS MaxWall 24 h). pf92 A2 lost ~70 min in (logs end 05:07:19); pf88b handover never started.
- Remaining: 089 (pf90 A2 -> pc64), 136 (wts48). 214829 still JobLaunchFailure. Submitted 2 more oai holds (-x 026,206,274,085).

## 2026-10-14T05:31Z poll
- SCORED: pf90 A2 C48 (real 256-expert model, q1 config + prefill mem 0.90 -> prefill KV 2,541,376; 089): 23,127/chip, qps 1.276,
  ISL 143,964, TTFT 9.30/28.76 s, TPOT 13.1 ms, GPU hit 0.860, CPU hit 0.071, err 3. vs q1 A3 C48 22,306 (137; GPU 0.839 CPU 0.088): +3.7 %.
- pc64 started on 089 (pf90 stopped 05:16). wts48 A1 still finishing. 3 oai holds pending.
- wts48 (136): login-host driver + bench client terminated 04:47 (agentx-c48-A1.log ends "Terminated", ~50 min into A1; no driver
  processes left on spur006; cause NOT established). Stack stayed up. Restarted with cont_benchl.yihou.sh (login-host variant): flush ->
  A2 C48 started 05:31 (the partial A1 serves as warm-up; protocol deviation recorded). Log runs/wts48.cont.log.

## 2026-10-14T05:55Z
- pc64 (089): pipeline STOP at smoke 05:42 — first smoke request hung 300 s then aborted ("Decode handshake failed ... Aborted by
  AbortReq"); prefill startup identical to pc48b (warm-up #inflight-req 1). Manual smoke 05:52 OK in 1 s; audit PASS (policy
  prefill_complete on all DP ranks both legs). Cause of the first-request hang NOT established. Benches continued with cont_bench:
  A1 C64 -> A2 C64.
- Cleanup: phase-2 approved after review (293 lines, 0 audit problems, dry-run 292/292 PASS); running.

## 2026-10-14T07:11Z poll
- SCORED: wts48 A2 C48 (136, q1 config + HiCache write_through_selective; A2 after an interrupted A1 + flush): 21,226/chip, qps 1.169,
  ISL 144,218, TTFT 13.63/42.03 s, TPOT 12.5 ms, GPU hit 0.868, CPU hit 0.039, err 1. vs q1 A3 (write_through, 137) 22,306 (-4.8 %;
  GPU 0.839, CPU 0.088). Node + image build (136 9872bbda) differ; protocol deviation (A1 cut at ~50 min). 136 stack left up, idle.
- pc64 A1 C64 running (089). 3 oai holds pending.

## 2026-10-14T07:35Z
- wts64 (136, C64 on the live wts48 stack): the 07:16 launch (shown to the user as rejected) DID run: flushed and started bench C64 at
  ~07:17 (cont_benchl PID 1464883). My 07:22 relaunch then saw decode flush 400 x60 ("pending requests" = the running bench) and exited.
  One wts64 bench is running; cont_benchl2 (decode-flush tolerant) written but not used.
- Prefill memory accounting (DP0): avail after weights 158.58 GiB (weights 119.84); KV 0.88/0.90/0.92 = 125.14/130.71/136.28 GiB
  (2,433,088/2,541,376/2,649,600 tok); post-pool free = 33.18/27.70/22.21 GiB = 278.42 x (1-frac). 089 pc64 (0.88) 4-min VRAM peak:
  prefill GPUs 274.8 of 287.98 GiB -> ~20 GiB runtime use over the pool, ~13 GiB unused. Full-run sampler running (/runs/pc64/vram.089.tsv).

## 2026-10-14T07:34Z poll
- pc64 A1 C64 fresh (089, prefill_complete + write_back + HiCache 370 + decode 0.96): 10,480/chip, qps 0.664, ISL 125,286, TTFT 82.77/262.94 s,
  TPOT 13.9 ms, GPU hit 0.493, CPU hit 0.341, err 2. vs pc48b A1 C48 18,225. A2 C64 started 07:28.
- 089 VRAM peak (57 samples, 5 s, during pc64 A1 tail + A2 start): prefill GPUs 277.36/276.92/274.89/276.20, decode 285.0 GiB of 287.98.
- wts64 (136) bench running since 07:17. No errors.

## 2026-10-14T07:5xZ
- User: abandon the prefill_complete path. pc64 A2 aborted (my first stop attempt self-killed via a pgrep pattern matching its own
  command line; second attempt: run.v3 stop -> 0 pc64 containers). pc64 A1 C64 10,480 is the only pc64 result.
- cx90 (089): correctness round = pf90 config with simulated acceptance OFF (dryrun diff = only the 3 SIMULATE envs); chain_cx:
  up -> census -> smoke -> audit -> correctness_tests.yihou.py (smoke greedy+sampled, needle 8k..256k x 3 depths, 5-fact 100k) -> stop.

## 2026-10-14T08:0xZ
- User: decode --tokenizer-worker-num 4 on the current best config. Code check (v0.5.21): rejected only with --grpc-port, Elastic EP,
  (skip_tokenizer_init forces detok=1); detokenizer_worker_num left at 1 (one variable). tw90 = pf90 config + decode tokw 4
  (config.yihou.tokw.sh; dryrun diff vs pf90 = only the flag). Armed on 089 after cx90 (wait_log_then -> chain_cap2 48 48); same node as pf90.

## 2026-10-14T09:08Z poll — cx90 correctness DONE (089, pf90 config, sim OFF, real EAGLE acceptance)
- 22/23 PASS: smoke arith/primes/capital greedy+sampled 6/6; needle 8k/32k/64k/128k/256k x d10/50/90 sampled 14/15; 5-fact 100k greedy+sampled 5/5 each.
- FAIL needle_128k_d10: finish=length, content empty; reasoning_tokens 4101 = max_tokens 4096 exhausted while the reasoning re-quoted the
  haystack (no answer emitted) -> budget-limited, not a wrong answer. needle_256k_d10 took 470 s (first 256k prefill on a cold cache).
- Results: /runs/cx90/{correctness.log,results.jsonl}. Stack stopped 09:06; tw90 handover now waiting on 089 VRAM drain.
- wts64 (136) still running at 09:08 (C64 run longer than 85 min; client log alive). ofs0 handover waiting.

## 2026-10-14T09:28Z poll
- wts64 (136): AgentX warmup at C64 took 5,354 s (707 requests, 07:17 -> 08:46); profiling 3600 s started 08:46:39 -> result ~09:50.
  Live srv stats 09:27: prefix_cache_hit 75 %, kv_usage 100 %, cpu_kv_usage 54 %, queue 3r/20w.
- tw90 (089) started 09:07 (VRAM gate). ofs0 handover still waiting for wts64. No errors.

## 2026-10-14T10:10Z poll
- SCORED: wts64 (136, write_through_selective, C64, 3rd run on the wts48 stack): 8,681/chip, qps 0.572, ISL 120,479, TTFT 107.15/307.53 s,
  TPOT 9.2 ms, GPU hit 0.428, CPU hit 0.185, err 3. Warmup alone 5,354 s at C64. vs wts48 A2 C48 21,226; pc64 A1 C64 10,480.
- tw90 (089) A1 C48 running since 09:40. 136 -> ofs0c handover.

## 2026-10-14T11:0xZ — 136 lost to limou; spur006 hung
- spur006 stopped answering ~10:50 (spur007 OK, NFS OK). The 136 handover (after_wts64_s0 on spur006) had been stuck since ~09:55 in run.v3
  stop copy-back. Meanwhile limou started vllm-megamoe-ep4-dspark-c64-limou (~10:37) and vllm-tp4-aiter-dspark-c64-limou on 136 (all 8 GPUs).
  Not touched. If the spur006 driver resumes, the chain_cxl KFD check stops it.
- S0 moved to 089 after tw90: env.ofs0c-089 / env.ofs0-089 (base mc4340n4); dryrun diffs vs pf90/cx90 on 089 = only offload mount+flags.
  Driver after_tw90_s0.yihou.sh ON 089 (log runs/ofs0.handover089.log). pf94 not scheduled.

## 2026-10-14T11:15Z poll: tw90 A1 C48 fresh (089, pf90 + decode tokenizer-worker-num 4): 22,231/chip, qps 1.235, ISL 143,101, TTFT 8.63/29.11, TPOT 15.4, GPU 0.914 CPU 0.035, err 4. vs pf90 A1 (same node) 21,859 (+1.7 %). A2 started 11:11. spur006 still unreachable.

## 2026-10-14T12:55Z poll: SCORED tw90 A2 C48 (089, pf90 + decode tokenizer-worker-num 4): 22,597/chip, qps 1.243, ISL 144,451, TTFT 10.94/34.84, TPOT 12.5, GPU 0.871 CPU 0.056, err 4. vs pf90 A2 same node 23,127 (-2.3 %); A1 22,231 vs 21,859 (+1.7 %) -> no consistent effect. ofs0c started 12:42 on 089.

## 2026-10-14T13:16Z — ofs0c (OffloaderV2 S0 correctness, 089) DONE: 23/23 PASS (cx90 22/23; needle_128k_d10 now answered in 14.9 s).
- Offload active: 9 FusedMoE modules (index 7,15,...,71) offloaded, w13+w2 per layer 1.10 GiB (memory_allocated step 11,809,982,464 B
  per layer across... per log), weights 119.84 -> 110.77 GiB/GPU (-9.07), avail after weights 158.58 -> 167.65 GiB.
- Prefill KV at mem 0.90: 2,541,376 -> 2,717,696 tokens (+176,320 = +6.9 %); post-pool avail 27.63 GiB (pf90 27.70).
- Only error line = uvicorn shutdown Traceback at stop (13:15). ofs0 perf (C48 A1->A2) started 13:15:57.

## 2026-10-14 15:18 UTC poll
- ofs0 (089, job 213048): A1 C48 running since 14:02:10Z; 0 Traceback/MAF/ValueError/OOM/HIP error in last 30 MB of prefill-0/decode-0.a1 logs; decode DP1 15:18 token usage 0.47, accept 3.55. A1 ETA ~15:25, A2 ~16:50.
- oai holds 214829/215368/215369 still PENDING (QOSGrpNodeLimit).
- spur006: probe hung again; stale after_wts64_s0 driver not touched.

## 2026-10-14 15:35 UTC poll
- ofs0 A1 C48 finished 15:24:58Z: score 22,380 qps 1.228 ISL 144,840 TTFT 9.19/27.12 TPOT 15.2 gpu-hit 0.931 cpu-hit 0.024 err 2 (vs pf90 A1 21,859, +2.4%). A2 started 15:24:58Z, ETA ~16:50.
- 0 errors in last 30 MB of ofs0 server logs. oai holds still PENDING. spur006 not probed.

## 2026-10-14 15:55 UTC poll
- ofs0 A2 C48 running (start 15:24:58Z, ETA ~16:50); 0 errors in last 30 MB of server logs. oai holds still PENDING.

## 2026-10-14 16:15 UTC poll
- ofs0 A2 C48 still running (ETA ~16:50); 0 errors in server logs. oai holds still PENDING.

## 2026-10-14 16:35 UTC poll
- ofs0 A2 C48 still running (ETA ~16:50); 0 errors. oai holds still PENDING.

## 2026-10-14 16:55 UTC poll
- ofs0 A2 C48 DONE 16:47:30Z: score 22,825 qps 1.267 ISL 143,116 TTFT 9.75/30.51 TPOT 12.9 gpu-hit 0.887 cpu-hit 0.049 err 4. vs pf90 A2 23,127: -1.3% (qps -0.7%). A1 was +2.4% vs pf90 A1. 0 server-log errors. Chain DONE, stack left running on 089 (job ends 19:27Z).
- S0 gate: loss <=2% -> propose 25% offload (G4K1); awaiting user decision.

## 2026-10-14 18:15 UTC poll
- oai holds STARTED: 214829 m2m-252 (10.245.149.156), 215368 m2m-266 (10.245.157.132), 215369 m2m-288 (10.245.152.60). Verified: ROCR 0-7, 8x MI355X, ECC uncorrectable 0 on all (288: one GPU 8 correctable), no running containers, docker root /mnt/m2m_nobackup (8.6-17 TB free).
- 18:17Z: started zstd|docker load of 9d755dd6 tar + seed aiter/triton caches from cache/102 into cache/{252,266,288} (logs $N/logs/prep.<n>.yihou.log).
- 089: idle stack, job ends 19:27Z.

## 2026-10-14 18:35 UTC poll
- 252/266/288 prep done 18:22Z: docker load rc=0, image ID sha256:9d755dd6a7bf... on all three, SEED_OK, 0 containers. Idle, awaiting user choice of rounds (G4K1 / pf94-93 / pf92 A2).
- 089 idle, job ends 19:27Z.

## 2026-10-14 19:35 UTC poll
- 089 job 213048 gone from squeue (time limit 19:27Z). ofs0 stack containers were left running at chain end; their fate after job end not checked (no spur exec possible).
- 252/266/288 idle, ready, awaiting user choice.

## 2026-10-14 22:15 UTC poll
- After the 22:00 window: 214829/215368/215369 still RUNNING on 252/266/288 (idle, awaiting user). 089 expired 19:27Z.

## 2026-10-15 01:45 UTC - OffloaderV2 ratio sweep launched (user: raise offload ratio, push conc, check correctness; 3 oai nodes in parallel)
- New files (harness, nothing existing edited): config.yihou.mc4340n5.sh (n4 + 252/266/288), env.ofg{4,3,2}{,c}-{252,266,288}.yihou.sh, ofs_seq.yihou.sh (chain_cx correctness -> chain_cap2 C1 C2 C3).
- K=1 everywhere (offloaded modules are index % G >= G-K; K>1 puts offloaded layers back to back with prefetch_step 1, so ratio is raised via G).
- 252 job 214829: ofg4c -> ofg4  G4K1 (1/4), C48/C48/C56.
- 266 job 215368: ofg3c -> ofg3  G3K1 (1/3), C48/C48/C56.
- 288 job 215369: ofg2c -> ofg2  G2K1 (1/2), C48/C48/C56.
- Dryrun tokdiff vs ofs0-089 / ofs0c-089: only node paths/IP and --offload-group-size 8->4/3/2. Host RAM 2751 GiB each, 0 KFD, 0 containers at launch. Logs runs/ofg*.seq.log, runs/ofg*c.chain.log, runs/ofg*.chain.log.

## 2026-10-15 01:52 UTC poll
- ofg4c/ofg3c/ofg2c: stacks loading (waiting for prefill HTTP, 300 s). Offloaded MoE modules per prefill rank (DP0 TP0): G4 19, G3 26, G2 39 (ofs0 G8: 9). 0 errors in prefill logs. Load-weight-end / KV not logged yet.

## 2026-10-15 02:12 UTC poll
- Prefill per rank: G4 weights 99.40 GB, KV 2,938,752, avail 27.72 GB | G3 91.50 GB, KV 3,092,288, avail 27.70 | G2 76.78 GB, KV 3,378,496, avail 27.79 (pf90 2,541,376; ofs0 G8 2,717,696). 0 errors.
- ofg3c smoke ok 02:09:29, correctness tests running; ofg2c smoke ok 02:11:16, tests running; ofg4c prefill HTTP still not up at 1502 s (02:11).

## 2026-10-15 user: free node 136 for colleague Huang Zhen
- Checked 136 (via spur007 ssh): no yihou containers (present: sikl_xb, dev_xb, crusoe-* agents - not ours, untouched), no yihou processes, 0 KFD GPU processes. Login hosts spur005/006/007: no yihou driver targeting 136. spur006 reachable again; stale after_wts64_s0 driver no longer present (nothing killed). Nothing to release.

## 2026-10-15 02:35 UTC poll
- ofg3c DONE 02:14:40: 23/23 PASS. ofg2c DONE 02:20:23: 23/23 PASS (same test list as ofs0c 23/23; cx90 22/23 with needle_128k_d10 FAIL). ofg3 perf chain started 02:14:40, ofg2 02:20:23 (C48/C48/C56).
- ofg4c (252) ANOMALY: prefill Engine startup scheduler_e2e=2063.85 s (ofg3c 894, ofg2c 685), load_weight 389 s; Uvicorn up 02:21:23, "Start of pd disaggregation warmup" 02:21:24 with no "End of disaggregation warmup" by 02:31 (ofg3c: warmup completed in 60 s). /health 503 since; waiting-for-HTTP at 2734 s. rocm-smi GPU use 0% on GPU0-4. 0 Traceback/MAF/ValueError/OOM/HIP error. Only prefill + etcd containers up (decode not started). Not touched; asked user.

## 2026-10-15 02:45 UTC user: request a new node, release 266 to admins
- Submitted oai hold 217052 (HOLD_SECONDS=86400, -t 24h; first submit 217051 cancelled because it lacked HOLD_SECONDS) -> PENDING (QOSGrpNodeLimit).
- 266 NOT released yet: it runs the healthy ofg3 (G3K1) perf chain; the stuck round is ofg4c on 252. Asked user to confirm which node to release.

## 2026-10-15 02:50-03:10 UTC
- 266 released per user ("管理员指定266"): killed ofg3 driver PIDs 650/4319/4320/4328 by number, run.v3 stop (rc=1 "cleanup completed with errors", but 0 containers running, 0 KFD), docker rm of our 7 named exited containers (ofg3-*, ofg3c-*), scancel 215368. ofg3 perf ABORTED in bring-up.
- 252 ofg4c: NOT restarted - before acting, found the PD warm-up had completed on its own at 02:42:09 (20 min 45 s; ofg3c 60 s) and decode was coming up; evidence copied to runs/ofg4c.hang1/. ofg4c smoke/audit PASS, correctness tests started 02:57:09.
- ofg2 (288): A1 C48 started 02:57:20.
- Replacement hold 217052 PENDING (QOSGrpNodeLimit). Job 217099 (yihou-dsp-burst2, amd-primus/burst) is NOT from this session; untouched.
- Packups (copy-only builders in $N): glm52.v0521-offloaderv2.packup_20261015 (16M, 339 files) and glm52.v0521-c48-followups.packup_20261015 (16M, 268 files) under ~/dev/git/infera.glm52.pd/yihou; no file >4MB, secret grep clean, SHA256SUMS ok. CONCLUSIONS index row "(not yet packed)" replaced by the two packups.

## 2026-10-15 03:14 UTC poll
- ofg4c (252, G4K1) DONE 03:08:16: 22/23. FAIL needle_128k_d50 (sampled): finish=length, content empty, reasoning_tokens 4098 vs max_tokens 4096, reasoning re-quoting the haystack (same pattern as cx90 needle_128k_d10). The 1 Traceback in prefill-0.log is at 03:07:06 during stop (SIGTERM -> lifespan CancelledError), not during tests.
- ofg4 perf chain started 03:08:16 (C48/C48/C56). ofg2 A1 C48 running since 02:57:20. 0 errors otherwise. 217052 PENDING.

## 2026-10-15 ~03:40 UTC - new partition (user: "spur新加了amd之外的新的partition…申请两个burst节点")
- New partition amd-spur-v2: 28 nodes crsuse2-m2m-v3-*, 8x MI355X each (gres gpu:mi355x:1 x8; --gres=gpu:8 accepted), 236 CPUs, ~2.8 TB RAM, MaxTime 3 d, PreemptMode=CANCEL, AllowAccounts same as amd-spur (incl. amd-burst/amd-oai), AllowQos=ALL. At ~03:39: 22 alloc, 6 mix, 0 idle. Also amd-di (amd-di only) and amd-aifw-sglang (amd-aifw-sglang only) - not usable by yihou.
- Submitted 217125, 217126: HOLD_SECONDS=86400 sbatch --export=ALL -A amd-burst -p amd-spur-v2 -q amd-burst-qos -N1 --gres=gpu:8 -c 160 -t 24:00:00 -J yihou-newpart-burst ~/spur-hold/hold_node.sh -> PENDING (Resources). amd-burst-qos MaxSubmitPU=4.

## 2026-10-15 04:34 UTC poll
- ofg2 A1 C48 (288, G2K1 1/2, KV 3,378,496) DONE 04:23:45: 21,157/chip, qps 1.171, ISL 143,593, TTFT 12.32/29.13, TPOT 14.5, GPU hit 0.960, CPU hit 0.003, err 1. vs pf90 A1 21,859 (-3.2%), ofs0 A1 22,380 (-5.5%) (both 089; cross-node). A2 started 04:23:45.
- ofg4 A1 C48 running since 03:39. 0 errors. Holds 217052/217125/217126 PENDING; 217187 yihou-dsp-ws not ours.

## 2026-10-15 04:4x UTC - user: cancel pending oai job -> scancel 217052 (yihou-oai-hold, amd-oai). Other pending jobs untouched.

## 2026-10-15 ~04:50 UTC - user: no further oai node requests (pending oai 217052 already cancelled). Running: 214829 (252), 215369 (288). Burst 217125/217126 on amd-spur-v2 still pending (kept; requested earlier by user).

## 2026-10-15 05:00 UTC - user: stop experiments, release oai nodes (TL gave nodes to another team)
- 252: killed ofg4 driver PIDs 765/6938/11717/12037 by number, run.v3 stop (rc=1 "cleanup completed with errors"; 0 containers running, 0 KFD), docker rm of our 9 exited containers (glm52-v0521-yihou-252-ofg4*-). ofg4 A1 C48 ABORTED ~76 min into 3600 s bench (no JSON).
- 288: killed PIDs 660/10323/10326, run.v3 stop (rc=1, 0 running, 0 KFD), docker rm of our 10 exited containers. ofg2 A2 C48 ABORTED ~31 min in (no JSON). ofg2 A1 21,157 is the only ofg2 score.
- scancel 214829 215369 -> gone from squeue. Remaining: burst 217125/217126 PENDING (amd-spur-v2), 217187 not ours.
- User: use burst nodes if they start. Node 136 check: containers sikl_xb (created 10-15 02:12, mounts /home/xiaobche/SIKL) and dev_xb (created 09-29) - owner xiaobche by mount path; logged in: limou102 (since 10-15 02:44), jiejizha (since 09-04); 0 KFD, GPU use 0% on all 8.

## 2026-10-15 05:19 UTC - user: "用136继续"
- 136 pre-check: our image sha256:9872bbda present; VRAM 284 MiB used per GPU, 0 KFD, GPU use 0%; host RAM 2702 GiB available; /mnt/m2m_nobackup 7.4 TB free. Foreign containers sikl_xb/dev_xb (xiaobche) left untouched.
- New: env.ofg4b-136 / env.ofg2b-136 (from env.ofs0-136, OFF_G 4/2), ofs_seq136l.yihou.sh (login-host sequencer: chain_cap2l per round -> run.v3 stop -> wait 0 KFD). Dryrun tokdiff vs ofs0-136: only --offload-group-size 8 -> 4/2.
- Launched on spur007 05:19:40 (setsid): ofg4b 48 48 56 then ofg2b 48 48 56. Log runs/ofs136.seq.log, runs/ofg4b.chain.log; server logs node-local on 136.

## 2026-10-15 05:45 UTC poll
- 136 ofg4b: prefill up 05:33:55, decode coming up; server logs /mnt/m2m_nobackup/yihou_v0521dbg/runs/ofg4b/server-logs (node-local). rocm-smi --showpids on 136 lists PID 17976 gpuagent (crusoe exporter, 0 VRAM) - ofs_seq136l kfd gate counts it; watch at the ofg4b->ofg2b handover.
- Burst holds STARTED: 217125 crsuse2-m2m-v3-020 (10.245.150.105), 217126 crsuse2-m2m-v3-026 (10.245.152.176). Both: ROCR 0-7, 8 GPUs, ECC 0/0, RAM 2.7 TB, docker root /mnt/m2m_nobackup (27 TB free), no infera image.
  - v3-020: 8 running containers glm52x_*_chuschen (lmsysorg/sglang-rocm:v0.5.20-rocm720, created 04:02, mount /shared_nfs/chuschen/agent-kernel-arena), 0 KFD, GPU0 1.8 GB used.
  - v3-026: guanchen container crusoe-sgl-q36 (sleep infinity, created 10-14 05:11), 2 KFD PIDs (0 VRAM).
  Foreign containers not touched; no image load yet; asked user.

## 2026-10-15 06:15 UTC
- 136: ofs_seq136l KFD gate always sees crusoe gpuagent PID 17976 -> would never hand over to ofg2b. Killed only the sequencer parent PID 1843976 (chain_cap2l 1844010 for ofg4b keeps running); started after_ofg4b136l.yihou.sh on spur007 (VRAM<1GiB gate; log runs/ofs136.after.log). ofg4b A1 C48 bench since 05:47:15; prefill 19 offloaded, KV 2,938,752, avail 27.81 GB; 0 errors.

## 2026-10-15 07:13 UTC: ofg4b A1 C48 (136, G4K1) DONE 07:07:29; A2 started 07:07:29.

## 2026-10-15 07:15-07:23 UTC - burst nodes (user: "burst节点你自己规划就好，持有节点以后杀死所有block我们的GPU非系统supervisor进程")
- docker stop (not rm) of foreign GPU containers: v3-020 chuschen glm52x_*_chuschen (5 running at stop time, sglang-rocm v0.5.20 kernel tests); v3-026 guanchen crusoe-sgl-q36 (sleep infinity, 2 KFD PIDs). After: 0 containers, 0 KFD, ~284 MiB VRAM/GPU. No crusoe system agents run on these v3 nodes.
- prep_node.yihou.sh (new, $N): docker load 9d755dd6 rc=0 on both, image ID sha256:9d755dd6...; cache key 020 seeded from 102; key 026 already seeded (old node m2m-026, same image). NODE_KEY collision note: v3-026 -> key 026 (reuses cache/026, rewrites topology.yihou.026.tsv via base config).
- New: config.yihou.mc4340n6.sh (n5 + v3-020/v3-026), env.ofg3b-v3020 (G3K1, rerun of aborted ofg3), env.pf90b-v3026 (pf90, no offload, same-partition reference). Dryrun tokdiff vs ofg3-266 / pf90-089: only IP and node paths.
- Launched 07:22 on-node chain_cap2 48 48 56: ofg3b (v3-020, job 217125), pf90b (v3-026, job 217126).
- 136: ofg4b A1 C48 = 22,506 (07:07); A2 running.

## 2026-10-15 07:45 UTC poll
- ofg3b (v3-020): prefill up 07:41:35; 26 offloaded/rank, weights 91.50 GB, KV 3,092,288, avail 27.70 GB (= ofg3c on 266). pf90b (v3-026): weights 119.84 GB, KV 2,541,376, avail 27.70 GB (= pf90 089), HiCache allocating. ofg4b (136) A2 C48 running since 07:07. 0 errors in all server logs.

## 2026-10-15 07:50 UTC - user: "没必要…我们是要测offload在高conc下能不能拿到更多的吞吐，baseline早就有了"
- pf90b (v3-026) STOPPED in bring-up (killed PIDs 959/968/970/977, run.v3 stop rc=1, 0 containers, 0 KFD). No pf90b result.
- v3-026 -> ofg2h: G2K1 at C56 (A1 fresh) / C56 (A2 scored) / C64 (A3), env.ofg2h-v3026 (tokdiff vs ofg2-288: engine argv identical; only CONC differs). Launched 07:50:49.
- 136 follow-up changed: killed waiter after_ofg4b136l (PID 2003006, was waiting); started after_ofg4b136l2 -> after ofg4b: ofg4h = G4K1 at C56/C56/C64 (env.ofg4h-136; argv identical to ofg4b). ofg2b on 136 dropped (G2 high conc runs on v3-026).
- v3-020 ofg3b unchanged (C48/C48/C56).

## 2026-10-15 08:41 UTC poll
- ofg4b (136, G4K1) A2 C48 DONE 08:28:55: 23,771/chip, qps 1.307, ISL 144,584, TTFT 9.71/28.77, TPOT 12.1, GPU hit 0.925, CPU hit 0.018, err 3. vs pf90 A2 23,127 (089, +2.8%), ofs0 A2 22,825 (089). A3 C56 started 08:28:56.
- ofg2h (v3-026) FAILED in bring-up: prefill PD warm-up started 07:57:04, no end; 08:27:05 "Initialization failed. warmup error ... asyncio.TimeoutError" (_send_disaggregation_warmup_requests) -> prefill exited, decode never started. load_weight 85 s, scheduler_e2e 272 s. Evidence copied to runs/ofg2h.fail1/. Stopped (killed PIDs 2446/2455/2456/2464, run.v3 stop rc=1, 0 KFD).
- Relaunched as ofg2h2 (same argv; tokdiff empty) at 08:43:05, chain_cap2 56 56 64, plus warmup_pyspy.yihou.sh (new): if warm-up not ended within 300 s, two py-spy dumps of all python procs in the prefill container -> runs/ofg2h2/pyspy/. (A first relaunch attempt as ofg2h stopped at up: exited containers with that prefix exist.)
- ofg3b (v3-020) A1 C48 running since 07:59. 0 errors.

## 2026-10-15 ~08:55 UTC - RULE (user): all rounds from now: decode --tokenizer-worker-num 4 --detokenizer-worker-num 4 ("decode侧cpu解码都是用4个worker"; clarified: tokenizer 4 + detokenizer 4; running chains ofg4b/ofg3b/ofg2h2 finish as-is, new rounds get it).
- New overlay config.yihou.tokw2.sh (chain TOKW2_BASE_CONFIG; v0.5.21 forces detok=1 only with --skip-tokenizer-init, not used). Verified by dryrun: decode argv gains exactly those 2 flags.
- env.ofg4h-136 (queued on 136 after ofg4b, not started) switched to tokw2; tokdiff vs ofg4b-136 = the 2 decode flags only.

## 2026-10-15 09:20 UTC - ofg2h2 failure, root cause (first-hand) and fix
- ofg2h2 (v3-026) PD warm-up started 08:52:37 and hung. py-spy (runs/ofg2h2/pyspy/dump3.sched.091626.txt): DP1/DP2/DP3 schedulers blocked in aiter file_baton.wait <- mp_lock <- build_module (forward_hip layernorm, dsa_indexer _get_k_bf16) on /aiter-jit/build/lock_module_norm; DP0 in an idle forward (flydsl gemm compile). The lock file was dated 07:47:27, content "1417 / crsuse2-m2m-v3-026" = left by the pf90b prefill that was killed in bring-up ~07:48.
- Cache key 026 (inherited from old node m2m-026, marked SEED_SKIP at prep) had only 4 aiter .so in prefill vs 16 in cache/102,020,252,288,089 -> modules were being JIT-built during warm-up (also the likely reason ofg2h warm-up exceeded 30 min - not verified).
- Removed the single stale lock file (named path in our cache) at 09:17:02 -> DP1 raised ModuleNotFoundError: module_norm (build never completed) -> prefill exited. Stopped ofg2h2 (killed chain/pipeline/run.v3/watcher by PID pattern of our harness scripts, run.v3 stop, 0 KFD).
- Re-seeded cache/026 from cache/102 with cp -an (no clobber): prefill .so 16 (set identical to 102), decode 14 (= 102), triton merged. No locks left.
- Relaunched as ofg2h3 at 09:19:55 (G2K1, C56/C56/C64, + decode tokenizer 4 + detokenizer 4 per the new rule; tokdiff vs ofg2h2 = those 2 flags only) with warmup_pyspy watcher.
- Lesson: never mark a reused cache key as seeded without counting .so vs cache/102; killing a stack during JIT build can leave an aiter lock that blocks the next run.

## 2026-10-15 09:46 UTC poll
- ofg3b A1 C48 (v3-020, G3K1, KV 3,092,288) DONE 09:27:10: 22,484/chip, qps 1.237, ISL 144,488, TTFT 9.27/23.86, TPOT 15.0, GPU hit 0.954, CPU hit 0.008, err 2. A2 C48 started 09:27:10. (A1 pf90 21,859 / ofs0 22,380 on 089, ofg4b 22,506 on 136, ofg2 21,157 on 288; all cross-node, no tokw2).
- ofg2h3 (v3-026, G2K1 + tokw2): prefill weight load 249.6 s, PD warm-up 61 s (09:28:21 -> 09:29:21) after cache re-seed; prefill ok 09:29:38. 0 errors.
- ofg4b A3 C56 (136) running since 08:28:56.

## 2026-10-15 10:16-10:45 UTC
- ofg4b A3 C56 (136, G4K1, no tokw2) DONE 09:53:45: 21,744/chip, qps 1.238, ISL 139,513, TTFT 20.04/51.99, TPOT 12.4, GPU hit 0.857, CPU hit 0.067, err 0. The only C56 no-offload refs (tb56n 18,016 / tb56 17,611, 085) used PREFILL_HICACHE_RATIO=1.5 (~154 GB, not 450 GB) and prefill mem 0.88 -> NOT a like-for-like baseline.
- ofg2h3 (v3-026) A1 bench FAIL 09:46:48 rc=2 in client setup: uv "Python installation is missing a _sysconfigdata_ file", client-home missing - client caches under reused key cache/026. Moved cache/026/{agentx,uv-python,client-home,client-pycache} aside to *.bad1-20261015 (mv, nothing deleted); moved failed out dir to agentx-c56-A1.fail1. Continued on the LIVE fresh stack via cont2_v3026.yihou.sh (cont_bench 56 56, then cont_bench - 64 -> out agentx-c64-A2); A1 C56 started 10:31:29, client install OK.
- 136: after_ofg4b136l2 was stuck 50 min inside "run.v3 stop" (ofg4b) in its sync_back copy of node-local logs over ssh; containers already gone, VRAM 284 MiB. Killed only run.v3 stop PID 3035584 -> driver proceeded: ofg4h (G4K1 + tokw2) 56 56 64 started 10:44:09. ofg4b server logs remain node-local on 136 (/mnt/m2m_nobackup/yihou_v0521dbg/runs/ofg4b).
- ofg3b (v3-020) A2 C48 running since 09:27. spur007 slow (60 s timeouts); spur005 OK.

## 2026-10-15 10:49 UTC: ofg3b A2 C48 (v3-020, G3K1, no tokw2) DONE: 23,200/chip, qps 1.250, ISL 147,540, TTFT 10.78/29.35, TPOT 12.0, GPU hit 0.936, CPU hit 0.010. A3 C56 next. Packup builder packup_copy_highconc.yihou.sh started on spur005 (log $N/logs/packup_highconc.yihou.log).

## 2026-10-15 ~11:40 UTC - packup glm52.v0521-offloaderv2-highconc.packup_20261015 (31M, 472 files, no file >4MB, secret scan clean, SHA256SUMS ok); CONCLUSIONS index row added. ofg4h2 relaunched on 136 from spur005 11:24 (driver run136_ofg4h2, VRAM gate; old spur007 driver had vanished after ofg4h prefill up 10:54).

## 2026-10-15 11:46 UTC poll
- Burst holds 217125 (v3-020) and 217126 (v3-026) CANCELLED at 11:35:12-13 (sacct State=CANCELLED, ExitCode -1:0; not cancelled by this session; amd-burst-qos PreemptMode=CANCEL). Bench logs stop with "Terminated" at 11:35: ofg3b A3 C56 (started 10:57:57) and ofg2h3 A1 C56 (started 10:31:29) LOST, no JSON. Our docker containers on v3-020/v3-026 may still be running - nodes not reachable without a job.
- 136: ofg4h2 driver still in VRAM gate. 11:4x VRAM per GPU: GPU0 8.6 GB, GPU1 8.7, GPU2 282.0, GPU3 281.9, GPU4-6 0.6, GPU7 1.7 GB; rocm-smi --showpids lists none; no new containers (sikl_xb/dev_xb + crusoe agents). At 11:23 GPU0-3 were 176-285 GB. Values change between samples; owner not identified.

## 2026-10-15 12:14 UTC poll: 136 VRAM drained by 11:49:57 (max 284 MiB) -> ofg4h2 (G4K1 + tok4/detok4) chain_cap2l 56 56 64 started; router ok 12:12:35, A1 C56 next. No yihou-* jobs of ours in squeue.

## 2026-10-15 13:03 UTC - ofg4h2 (136) STOP at census
- Stack was up (bothlegs 12:12:30, router 12:12:35). pipeline stage census hung 50 min on login host spur005: run.v3 census -> sync_back (ssh tar | base64 into a bash variable, then base64 -d | tar -x into $R/ofg4h2) ; the bash process sat in D state, wchan mem_cgroup_handle_over_high (login-host memory cgroup throttling); $R/ofg4h2/server-logs stayed empty (node-side decode-0.a1.log 22.9 MB, prefill 0.27 MB).
- Killed the census PIDs (237518/786177) at 13:02 -> pipeline: STOP at census (scheduler rank PIDs overlap/not found - census output missing). chain stops; run136 driver then runs run.v3 stop (600 s cap).
- Same mechanism as the 50-min ofg4b stop hang on spur007 (sync_back). No A1 C56 result.

## 2026-10-15 ~13:30 UTC
- User chose option 1 (continue on 136 with streaming sync_back). Created run.v3s/pipeline.v3s/chain_cap2ls (new files; diff = sync_back body + call sites). Not launched yet: 136 not reachable by ssh from spur006 (~13:20); spur005/spur007 timing out at banner exchange; spur003 refuses remote commands.
- User: login hosts kill/evict above ~4 GB per user.
- Packup glm52.v0521-offloaderv2-highconc refreshed with delta (packup_delta2): burst cancellation, ofg4h2 census failure, login-host root cause, streaming copies; 33M / 490 files, SHA256SUMS ok.

## 2026-10-15 18:18 UTC: 135/136/139 still TCP-unreachable from spur006. run136_ofg4h3 exited 18:04:53 (STOP: unreachable 4 h); relaunched the same waiter (user-approved).
- [burstkeeper] 2026-10-16T02:30:41Z submitted 219097 (amd-spur) and 219098 (amd-spur-v2), both PENDING (Resources)
- [watch136] 2026-10-16T02:31Z 135/136/139 ssh timeout, 137/138 ok; waiter was dead (last log 21:11 unreachable) -> relaunched pid 3358067 on spur006
- [burstkeeper] 2026-10-16T02:51:10Z 219098 RUNNING on crsuse2-m2m-v3-031 (10.245.145.118), 8 GPU ECC 0; docker-stopped mm_run, sglang_chuschen; prep rc=0 SEED_OK image 9d755dd6; cache 031 new, so 16/14, locks 0. 219097 PENDING

## 2026-10-16 ~02:55 UTC - ofg3h launched on burst v3-031 (job 219098, amd-spur-v2; prepped by burstkeeper, fresh cache key 031, .so 16/14, 0 locks): G3K1 + decode tok4/detok4, chain_cap2 56 56 64 ON the node + warmup_pyspy watcher. New config.yihou.mc4340n7.sh (n6 + crsuse2-m2m-v3-* pattern), env.ofg3h-v3031. tokdiff vs ofg3b-v3020 = the 2 decode flags only.
- [burstkeeper] 2026-10-16T03:09:29Z 219098 (v3-031) CANCELLED 02:55:39 (not by us); resubmitted 219127 -> RUNNING crsuse2-m2m-v3-010 (10.245.145.114), ECC 0, nothing stopped; prep rc=0 SEED_OK 9d755dd6; cache 010 new so 16/14 locks 0. 219097 PENDING

## 2026-10-16 03:1x UTC - 219098 (v3-031) CANCELLED 02:55:39 after 20 min (not by us); ofg3h lost in bring-up. ofg3h2 (same argv; tokdiff vs ofg3h = node only) launched on v3-010 (job 219127) chain_cap2 56 56 64 + pyspy watcher.
- [watch136] 2026-10-16T03:30Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3358067 alive, still "unreachable, waiting"
- [burstkeeper] 2026-10-16T03:58:43Z heartbeat: 219127 RUNNING v3-010; 219097 (amd-spur) PENDING (Resources)
- [watch136] 2026-10-16T04:30Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3358067 alive, waiting
- [burstkeeper] 2026-10-16T04:57:38Z heartbeat: 219127 RUNNING v3-010; 219097 (amd-spur) PENDING (Resources)
- [watch136] 2026-10-16T05:28Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3358067 alive, waiting

## 2026-10-16 05:24 UTC: ofg3h2 A1 C56 (v3-010, G3K1 + tok4/detok4, fresh) = 20,559/chip, qps 1.210, ISL 135,056, TTFT 18.28/59.98, TPOT 16.0, GPU 0.918, CPU 0.027, err 0. vs ofg4b A3 C56 21,744 (136, G4, warm 3rd, no tok4/detok4). A2 C56 started 05:24:49.
- [burstkeeper] 2026-10-16T05:56:27Z heartbeat: 219127 RUNNING v3-010; 219097 (amd-spur) PENDING (Resources)
- [watch136] 2026-10-16T06:26Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3358067 alive, waiting (4h limit ~06:31Z)
- [burstkeeper] 2026-10-16T06:55:17Z heartbeat: 219127 RUNNING v3-010; 219097 (amd-spur) PENDING (Priority)

## 2026-10-16 06:50 UTC: ofg3h2 A2 C56 (v3-010, G3K1 + tok4/detok4, warm, scored) = 21,310/chip, qps 1.217, ISL 139,084, TTFT 20.27/58.47, TPOT 13.2, GPU 0.882, CPU 0.048, err 5. A1 20,559. vs ofg4b A3 C56 21,744 (136, G4, no tok4/detok4, 3rd run). A3 C64 started 06:50:23.

## 2026-10-16 07:12 UTC: 219127 (v3-010) CANCELLED (not by us) -> ofg3h2 A3 C64 lost (started 06:50). burstkeeper got 219504 on v3-025 (stopped mm_run, hl-egress-proxy). Next: ofg2h4 (G2K1 + tok4/detok4, 56 56 64) on v3-025; env.ofg2h4-v3025, tokdiff vs ofg2h3-v3026 = node only.
- [watch136] 2026-10-16T07:20Z 135/136/139 ssh timeout, 137/138 ok; waiter hit "STOP: 136 unreachable for 4 h" at 07:11:26Z -> relaunched pid 2605651
- [burstkeeper] 2026-10-16T07:23:53Z 219127 (v3-010) CANCELLED 07:12:04 (not by us); resubmitted 219504 -> RUNNING crsuse2-m2m-v3-025 (10.245.156.213), ECC 0; docker-stopped mm_run, hl-egress-proxy; prep rc=0 SEED_OK 9d755dd6; cache 025 new so 16/14 locks 0. 219097 PENDING

## 2026-10-16 07:2x UTC: ofg2h4 launched on v3-025 (job 219504) chain_cap2 56 56 64 + pyspy watcher.
- [burstkeeper] 2026-10-16T08:03:14Z heartbeat: 219504 RUNNING v3-025; 219097 (amd-spur) PENDING (Priority)
- [watch136] 2026-10-16T08:18Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2605651 alive, waiting
- [burstkeeper] 2026-10-16T08:32:46Z 219504 (v3-025) PREEMPTED 08:32:18; resubmitted 219668 PENDING. 219097 PENDING

## 2026-10-16 08:32 UTC: 219504 (v3-025) PREEMPTED (Start 07:14:53) -> ofg2h4 A1 C56 (from 08:02:30) lost. burstkeeper resubmitted 219668 (PENDING); 219097 PENDING. No node held. Next node: rerun G2 as ofg2h5 (then G3 C64).

## 2026-10-16 08:4x UTC - user approved SHORT rounds (A1+A2 at one conc per burst node). Queue: ofg2h5 G2 C56; ofg3h3 G3 C64; ofg2h6 G2 C64; ofg4h4 G4 C64 (unless 136 gets it). All decode tok4/detok4. Lost rounds go back to the queue head.
- [burstkeeper] 2026-10-16T08:43:40Z 219668 RUNNING crsuse2-m2m-v3-008 (10.245.144.47), ECC 0; docker-stopped glm52x_mla_proj_q_batched_gemm_chuschen, glm52x_bf16_o_proj_chuschen, ixk3-attempt5, sikl-dev-zhuang12; prep rc=0 SEED_OK 9d755dd6; cache 008 new so 16/14 locks 0. 219097 PENDING

## 2026-10-16 08:4x UTC: ofg2h5 (G2K1 + tok4/detok4, C56 C56) launched on v3-008 (job 219668) + pyspy watcher.
- [watch136] 2026-10-16T09:17Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2605651 alive, waiting
- [burstkeeper] 2026-10-16T09:32:48Z heartbeat: 219668 RUNNING v3-008; 219097 (amd-spur) PENDING (Resources)
- [watch136] 2026-10-16T10:16Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2605651 alive, waiting
- [burstkeeper] 2026-10-16T10:31:40Z heartbeat: 219668 RUNNING v3-008; 219097 (amd-spur) PENDING (Priority)

## 2026-10-16 10:46 UTC: ofg2h5 A1 C56 (v3-008, G2K1 + tok4/detok4, fresh) = 21,766/chip, qps 1.241, ISL 139,447, TTFT 17.52/44.01, TPOT 16.0, GPU 0.944, CPU 0.013, err 2. vs ofg3h2 A1 C56 20,559 (v3-010). A2 C56 started 10:46:12.
- [watch136] 2026-10-16T11:15Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2605651 alive, waiting
- [burstkeeper] 2026-10-16T11:30:29Z heartbeat: 219668 RUNNING v3-008; 219097 (amd-spur) PENDING (Priority)

## 2026-10-16 11:50 UTC: 219668 (v3-008) CANCELLED 11:50:23 (not by us) -> ofg2h5 A2 C56 lost ~64 min in (from 10:46). Kept: ofg2h5 A1 C56 21,766. Queue head now: G2 C56 A2 (rerun as ofg2h7 C56 C56), then ofg3h3 G3 C64, ofg2h6 G2 C64, ofg4h4 G4 C64. burstkeeper should resubmit v2.
- [burstkeeper] 2026-10-16T12:00:03Z 219668 (v3-008) CANCELLED 11:50:23 (not by us); resubmitted 219888 PENDING. 219097 PENDING
- [watch136] 2026-10-16T12:15Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 12:00:13Z -> relaunched
- [burstkeeper] 2026-10-16T12:48:54Z heartbeat: 219888 (v2) and 219097 (spur) both PENDING (Resources); no node held
- [watch136] 2026-10-16T13:13Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1527716 alive, waiting
- [burstkeeper] 2026-10-16T13:48:22Z heartbeat: 219888 (v2) and 219097 (spur) both PENDING (Resources); no node held
- [watch136] 2026-10-16T14:12Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1527716 alive, waiting
- [burstkeeper] 2026-10-16T14:26:34Z 219888 RUNNING crsuse2-m2m-v3-033 (10.245.149.241), ECC 0, nothing stopped; prep rc=0 SEED_OK 9d755dd6; cache 033 new so 16/14 locks 0. 219097 PENDING

## 2026-10-16 14:2x UTC: ofg2h7 (G2K1 + tok4/detok4, C56 C56) launched on v3-033 (job 219888) + pyspy watcher.
- [watch136] 2026-10-16T15:15Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1527716 alive, waiting
- [burstkeeper] 2026-10-16T15:15:56Z heartbeat: 219888 RUNNING v3-033; 219097 (amd-spur) PENDING (Resources)
- [burstkeeper] 2026-10-16T15:31:22Z 219888 (v3-033) ended 15:25:08 PREEMPTED/CANCELLED (not by us); resubmitted 219999 PENDING. 219097 PENDING

## 2026-10-16 15:25 UTC: 219888 (v3-033) PREEMPTED/CANCELLED 15:25:08 (start 14:13:46) -> ofg2h7 lost 4 min into A1 C56 (bring-up 14:26->15:21). burstkeeper resubmitted 219999. v2 node lifetimes today: 20 min, 4h11m, 1h17m, 3h18m, 1h11m; our bring-up (prep ~6 min + stack ~55 min) + one 3600 s bench pair needs ~3.5 h.
- [watch136] 2026-10-16T16:12Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1527716 alive, waiting
- [burstkeeper] 2026-10-16T16:21:07Z 219999 RUNNING crsuse2-m2m-v3-008 (10.245.144.47), ECC 0, nothing stopped; prep rc=0 SEED_OK 9d755dd6; cache 008 REUSED (agentx/client-*/uv-python present) so 16/14 locks 0. 219097 PENDING

## 2026-10-16 16:2x UTC: ofg2h8 (G2K1 + tok4/detok4, C56 C56) launched on v3-008 again (job 219999; reused cache 008, .so 16/14) + pyspy watcher.
- [burstkeeper] 2026-10-16T16:31:16Z 219999 (v3-008) CANCELLED 16:27:54 (not by us); resubmitted 220063 PENDING. 219097 PENDING

## 2026-10-16 16:27 UTC: 219999 (v3-008) CANCELLED 16:27:54 after 22 min (not by us) -> ofg2h8 lost in bring-up. burstkeeper resubmitted 220063. v2 lifetimes today: 20m, 4h11m, 1h17m, 3h18m, 1h11m, 22m.
- [burstkeeper] 2026-10-16T16:58:10Z 220063 RUNNING crsuse2-m2m-v3-008, ECC 0; docker-stopped rw-220038-ab-v25abval-F25-demo_clean-v1g1; prep rc=0 SEED_OK 9d755dd6; cache 008 reused so 16/14 locks 0. 219097 PENDING

## 2026-10-16 16:5x UTC: ofg2h9 (G2K1 + tok4/detok4, C56 C56) launched on v3-008 (job 220063, 3rd allocation of v3-008) + pyspy watcher.
- [watch136] 2026-10-16T17:10Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 16:55:30Z -> relaunched pid 3795695
- [burstkeeper] 2026-10-16T17:27:53Z 220063 (v3-008) PREEMPTED 17:27:28; resubmitted 220110 PENDING. 219097 PENDING

## 2026-10-16 17:27 UTC: 220063 (v3-008) PREEMPTED 17:27:28 after 37 min -> ofg2h9 lost 6 min into A1 C56. burstkeeper resubmitted 220110. v2 lifetimes today: 20m, 4h11m, 1h17m, 3h18m, 1h11m, 22m, 37m.
- [burstkeeper] 2026-10-16T17:37:49Z 220110 RUNNING crsuse2-m2m-v3-008, ECC 0; our ofg2h9 containers already up there, nothing stopped, prep skipped (seed would cp into live cache 008); image 9d755dd6 present, so 16/14 locks 0. 219097 PENDING

## 2026-10-16 17:4x UTC: v3-008 re-allocated (220110); ofg2h9 stack (etcd/prefill/decode/router) survived the 220063 preemption, router /health 200. Continued on the live stack via cont_bench 56 56 (A1 C56 -> flush -> A2 C56); partial A1 moved to agentx-c56-A1.preempt1. Note: stack already served ~6 min of the interrupted A1; node-side docker-logs writers likely died with the old job (server-logs may stop updating). burstkeeper told not to run prep.

## 2026-10-16 17:53-18:0x UTC: ofg2h9 live-stack continuation FAILED: prefill :29101 / decode :29357 /health no response; prefill log "Health check failed ... detokenizer ... last_heartbeat 17:27:56" (= preemption time of 220063); GPU use 0 all; containers still Up. So the preemption killed the engine processes inside the containers while docker kept them Up -> a surviving stack after preemption is NOT reusable. Started run.v3 stop; then 220110 itself got CANCELLED (~18:0x) mid-stop. Our ofg2h9 containers on v3-008 may remain (not verifiable without a job).
- correction: 220110 ended PREEMPTED at 17:53:49 (start 17:28:20), during the stop.
- [burstkeeper] 2026-10-16T17:57:44Z 220110 (v3-008) CANCELLED 17:53:49 (not by us); resubmitted 220137 PENDING. 219097 PENDING
- [watch136] 2026-10-16T18:07Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3795695 alive, waiting
- [burstkeeper] 2026-10-16T18:17:26Z 220137 PENDING reason "dispatch confirmation failed (0/1 confirmed): 1 prolog failure", NodeList null. 219097 PENDING
- [burstkeeper] 2026-10-16T18:56:42Z heartbeat: 220137 (v2) PENDING (prolog-failure reason persists), 219097 (spur) PENDING; no node held
- [watch136] 2026-10-16T19:05Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3795695 alive, waiting

## 2026-10-16 19:5x UTC: main scancelled 220137 (ours; stuck in "1 prolog failure" since 18:17; burstkeeper had not acted on 2 requests) and resubmitted yihou-ofs-burst-v2.
- [burstkeeper] 2026-10-16T19:55:40Z 220137 CANCELLED 19:53:19 (never started, not by me); 220247 yihou-ofs-burst-v2 PENDING appeared (not submitted by me), adopted as v2 hold. 219097 PENDING
- [watch136] 2026-10-16T20:02Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3795695 alive, waiting
- [burstkeeper] 2026-10-16T20:23:23Z 220247 RUNNING crsuse2-m2m-v3-001 (10.245.148.54), ECC 0, nothing stopped; prep rc=0 SEED_OK 9d755dd6; cache 001 new so 16/14 locks 0. 219097 PENDING

## 2026-10-16 20:2x UTC: ofg2h10 (G2K1 + tok4/detok4, C56 C56) launched on v3-001 (job 220247; prep verified rc=0, .so 16/14, 0 locks) + pyspy watcher.
- [burstkeeper] 2026-10-16T20:33:27Z 220247 (v3-001) CANCELLED 20:28:18 (not by me); resubmitted 220281 PENDING. 219097 PENDING

## 2026-10-16 20:28 UTC: 220247 (v3-001) CANCELLED 20:28:18 after 21 min (not by us) -> ofg2h10 lost in bring-up. burstkeeper resubmitted 220281. v2 allocations today: 9 (lifetimes 20m, 4h11m, 1h17m, 3h18m, 1h11m, 22m, 37m, 25m, 21m); 1 scored point obtained (ofg3h2 A2).
- [watch136] 2026-10-16T20:59Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3795695 alive, waiting
- [burstkeeper] 2026-10-16T21:20:50Z 220281 RUNNING crsuse2-m2m-v3-010 (10.245.145.114), ECC 0, nothing stopped; prep rc=0 SEED_OK 9d755dd6; cache 010 reused so 16/14 locks 0. 219097 PENDING

## 2026-10-16 21:2x UTC: ofg2h11 (G2K1 + tok4/detok4, C56 C56) launched on v3-010 (job 220281; reused cache 010, .so 16/14, 0 locks) + pyspy watcher.

## 2026-10-16 21:23 UTC: 220281 (v3-010) PREEMPTED 21:23:06 after 20 min -> ofg2h11 lost in bring-up. 10th v2 allocation today; lifetimes 20m,4h11m,1h17m,3h18m,1h11m,22m,37m,25m,21m,20m.
- [burstkeeper] 2026-10-16T21:30:55Z 220281 (v3-010) CANCELLED 21:23:06 (not by me); resubmitted 220330 PENDING. 219097 PENDING
- [watch136] 2026-10-16T21:57Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 21:50:24Z -> relaunched
- [burstkeeper] 2026-10-16T22:19:47Z heartbeat: 220330 (v2) and 219097 (spur) PENDING; no node held
- [watch136] 2026-10-16T22:54Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2366430 alive, waiting
- [burstkeeper] 2026-10-16T23:18:39Z heartbeat: 220330 (v2) and 219097 (spur) PENDING; no node held
- [watch136] 2026-10-16T23:52Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2366430 alive, waiting
- [burstkeeper] 2026-10-17T00:17:51Z 220330 held JobHoldMaxRequeue (Restarts=5), left in place; submitted 220461 (v2) PENDING. 219097 PENDING
- [watch136] 2026-10-17T00:49Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2366430 alive, waiting

## 2026-10-17 00:5x UTC: main scancelled 220330 (ours, JobHoldMaxRequeue since 00:17; burstkeeper had not acted). Holds: 219097 (spur), 220461 (v2), both PENDING.
- [burstkeeper] 2026-10-17T00:56:45Z 220330 now CANCELLED 00:53:18 (not by me). 220461 (v2) and 219097 (spur) PENDING

## 2026-10-17 01:2x UTC: ofg2h12 (G2K1 + tok4/detok4, C56 C56) launched on v3-007 (job 220461; prep verified, .so 16/14, 0 locks) + pyspy watcher.
- [burstkeeper] 2026-10-17T01:25:11Z 220461 RUNNING crsuse2-m2m-v3-007 (10.245.158.136), ECC 0; docker-stopped nope_s2_q100_rope, nope_eval, mm_run; prep rc=0 SEED_OK 9d755dd6; cache 007 new so 16/14 locks 0. 219097 PENDING
- [watch136] 2026-10-17T01:46Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2366430 alive, waiting
- [burstkeeper] 2026-10-17T02:14:45Z 220461 (v3-007) CANCELLED 02:08:36 (not by me); resubmitted 220507 PENDING. 219097 PENDING

## 2026-10-17 02:08 UTC: 220461 (v3-007) CANCELLED 02:08:36 after 59 min -> ofg2h12 lost before/at A1 start. burstkeeper resubmitted 220507. 11 v2 allocations since 10-16 02:35; 1 scored point.
- [watch136] 2026-10-17T02:43Z 135/136/139 ssh timeout, 137/138 ok; waiter pid 2366430 gone, no STOP line (last log 02:27:59Z unreachable) -> relaunched
- [burstkeeper] 2026-10-17T03:03:37Z heartbeat: 220507 (v2) and 219097 (spur) PENDING; no node held
- [watch136] 2026-10-17T03:41Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 552674 alive, waiting
- [burstkeeper] 2026-10-17T04:12:18Z heartbeat: 220507 (v2) and 219097 (spur) PENDING; no node held
- [burstkeeper] 2026-10-17T04:32:06Z 220507 RUNNING crsuse2-m2m-v3-007, ECC 0; our ofg2h12 prefill-0/etcd already up (3h), nothing stopped, prep skipped; image present, so 16/14 locks 0. 219097 PENDING

## 2026-10-17 04:3x UTC: v3-007 re-allocated (220507). Our dead ofg2h12 prefill/etcd containers (from cancelled 220461) docker-stopped by main (burstkeeper skipped). ofg2h13 (G2K1 + tok4/detok4, C56 C56) launched + pyspy watcher.
- [watch136] 2026-10-17T04:38Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 552674 alive, waiting
- [burstkeeper] 2026-10-17T05:00:10Z 219097 RUNNING crsuse2-m2m-179 (10.245.156.35), ECC 0, nothing stopped, nobackup 80% used; prep rc=0 SEED_OK 9d755dd6; cache 179 new so 16/14 locks 0. 220507 RUNNING v3-007

## 2026-10-17 04:5x UTC: amd-spur hold 219097 RUNNING on crsuse2-m2m-179 (after ~26 h pending); prepped (image 9d755dd6, .so 16/14, 0 locks). New config.yihou.mc4340n8.sh (n7 + m2m-179). ofg3h3 (G3K1 + tok4/detok4, C64 C64) launched + pyspy watcher; tokdiff vs ofg3h2 = none in engine argv (conc only differs).
- [burstkeeper] 2026-10-17T05:10:14Z 219097 (m2m-179) CANCELLED 05:02:58 (not by me); resubmitted 220567 PENDING. 220507 RUNNING v3-007

## 2026-10-17 05:02 UTC: 219097 (m2m-179, amd-spur) CANCELLED 05:02:58 after 20 min (not by us) -> ofg3h3 lost at bring-up. burstkeeper resubmitted 220567. v3-007 (220507) ofg2h13 still running.
- [watch136] 2026-10-17T05:35Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 552674 alive, waiting
- [burstkeeper] 2026-10-17T05:39:42Z 220507 (v3-007) CANCELLED 05:29:37 (not by me); resubmitted 220579 PENDING. 220567 PENDING

## 2026-10-17 05:29 UTC: 220507 (v3-007) CANCELLED 05:29:37 after 1h02m -> ofg2h13 A1 C56 lost 25 min in (from 05:04). burstkeeper resubmitted 220579; 220567 (spur) PENDING. 05:28 sinfo: amd-spur-v2 0 idle (7 alloc, 6 mix, 5 down, 10 drain); amd-spur 0 idle (57 alloc, 17 mix, 67 down, 13 drain). 135/136/139 still TCP-unreachable; 137 (xiaobche) and 138 (llying + our yihou-expert8-tools-138) reachable.

## 2026-10-17 05:52 UTC: BLOCKER - /shared_nfs 100% full cluster-wide (df 420T/420T, avail 0); scp/dryrun writes fail "Disk quota exceeded". Holds 220567 (m2m-232) and 220579 (v3-010) RUNNING but nothing launchable. Our biggest items: img274/v0519...tar 63 GB (old v0519 image tar), v0521cap 23 GB (contains the v0521 9d755dd6 image tar - needed), agentx44 7.9 GB. Asked user; nothing deleted.

## 2026-10-17 05:5x UTC: /shared_nfs back to 274G free at 05:55. v3-010 (220579): main docker-stopped stale ofg2h11-etcd; image 9d755dd6 present, cache 010 .so 16/14, 0 locks; ofg2h14 (G2K1 + tok4/detok4, C56 C56) launched + pyspy. m2m-232 (220567): config mc4340n9 + env.ofg3h4-232 (G3 C64 C64) ready, waiting for burstkeeper prep.
- [burstkeeper] 2026-10-17T06:05:50Z 220567 RUNNING crsuse2-m2m-232 (10.245.153.80) ECC 0, prep rc=0 SEED_OK, cache 232 new so 16/14 locks 0. 220579 (v3-010) CANCELLED 06:00:47 (not by me) during prep (log only start lines); resubmitted 220634 PENDING

## 2026-10-17 06:0x UTC: 220579 (v3-010) CANCELLED 06:00:47 after 20 min -> ofg2h14 lost in bring-up. burstkeeper (late msg) had started prep on v3-010 at 05:58 (seed cp into cache/010 interrupted). m2m-232 prepped 06:02; ofg3h4 (G3K1 + tok4/detok4, C64 C64) launched + pyspy. burstkeeper resubmitted v2 as 220634.
- [burstkeeper] 2026-10-17T06:15:32Z 220567 (m2m-232) CANCELLED 06:08:02 (20 min after start, not by me); resubmitted 220644 PENDING. 220634 PENDING

## 2026-10-17 06:1x UTC: 220567 (m2m-232) CANCELLED 06:08:02 exactly 20 min after start -> ofg3h4 lost. burstkeeper noticed several holds ended at ~20 min. Config read (first-hand): a partition has PreemptMode=CANCEL PreemptExemptTime=1200 (=20 min) - burst jobs are exempt from preemption only for their first 20 min. Resubmitted 220644 (spur), 220634 (v2) pending.
- [watch136] 2026-10-17T06:33Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 552674 alive, waiting
- [burstkeeper] 2026-10-17T07:04:26Z heartbeat: 220644 (spur) and 220634 (v2) PENDING; no node held
- [burstkeeper] 2026-10-17T07:14:26Z 220634 held JobHoldMaxRequeue (Restarts=5), left in place; submitted 220670 (v2) PENDING. 220644 PENDING

## 2026-10-17 07:1x UTC: 220634 (v2) JobHoldMaxRequeue -> main scancelled; burstkeeper replacement 220670 pending; 220644 (spur) pending.
- [watch136] 2026-10-17T07:30Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 07:23:59Z -> relaunched
- [burstkeeper] 2026-10-17T08:03:15Z heartbeat: 220644 (spur) PENDING; 220670 (v2) PENDING reason prolog failure; no node held

## 2026-10-17 08:2x UTC: 220644 (amd-spur) on m2m-179 again; main docker-stopped stale ofg3h3-etcd; prep 08:14:58 SEED_OK; ofg3h5 (G3K1 + tok4/detok4, C64 C64) launched + pyspy. NFS free 1.1 TB at 07:53.
- [burstkeeper] 2026-10-17T08:20:19Z 220644 RUNNING crsuse2-m2m-179, ECC 0, our ofg3h3-etcd up, nothing stopped; prep rc=0 SEED_OK; cache 179 reused so 16/14 locks 0. 220670 PENDING (prolog failure)
- [watch136] 2026-10-17T08:28Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2336619 alive, waiting
- [burstkeeper] 2026-10-17T08:30:24Z 220644 (m2m-179) CANCELLED 08:24:15 (20 min, not by me); resubmitted 220733 PENDING. 220670 PENDING

## 2026-10-17 08:24 UTC: 220644 (m2m-179) CANCELLED 08:24:15, exactly 20 min after start -> ofg3h5 lost in bring-up. Resubmitted 220733. Pattern: 6 of the last 7 burst holds ended at ~20:00 (PreemptExemptTime=1200).

## 2026-10-17 08:5x UTC: main scancelled 220670 (v2, prolog failure since ~07:50) and resubmitted v2.

## 2026-10-17 08:5x UTC: 220773 (v2) RUNNING immediately on v3-007. docker-stopped foreign crusoe-sgl-q36-tree (non-system, guanchen-style; stop only). Image 9d755dd6 present, cache 007 identical to 102, 0 locks. ofg2h15 (G2K1 + tok4/detok4, C56 C56) launched + pyspy.
- [burstkeeper] 2026-10-17T08:59:48Z 220670 CANCELLED 08:53:17 (not by me); 220773 (not submitted by me) RUNNING v3-007 with our ofg2h15 containers up, adopted as v2 hold, prep skipped. 220733 PENDING
- [watch136] 2026-10-17T09:25Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2336619 alive, waiting
- [burstkeeper] 2026-10-17T09:29:09Z 220773 (v3-007) PREEMPTED 09:28:33; resubmitted 220851. 220733 PENDING JobLaunchFailure (agent rejected launch)

## 2026-10-17 09:28 UTC: 220773 (v3-007) PREEMPTED 09:28:33 after 35 min -> ofg2h15 lost at bring-up end (prefill up 09:09). Resubmitted 220851. 220733 (spur) PENDING "JobLaunchFailure ... 1 agent rejected launch".
- [burstkeeper] 2026-10-17T09:39:02Z 220851 RUNNING crsuse2-m2m-v3-019 (10.245.159.51), GPU0 correctable ECC 85 (others 0, uncorr 0), nothing stopped; prep rc=0 SEED_OK; cache 019 new so 16/14 locks 0. 220733 PENDING

## 2026-10-17 09:4x UTC: ofg2h16 (G2K1 + tok4/detok4, C56 C56) launched on v3-019 (job 220851; prep 09:38 OK, .so 16/14, 0 locks; GPU0 correctable ECC 85) + pyspy.
- [burstkeeper] 2026-10-17T09:49:07Z 220733 held JobHoldMaxRequeue (Restarts=5), left in place; submitted 220910 (spur) PENDING. 220851 RUNNING v3-019

## 2026-10-17 10:06 UTC: 220851 (v3-019) CANCELLED after 36 min -> ofg2h16 lost in bring-up. 220965 RUNNING on v3-006; env.ofg2h17-v3006 ready, waiting for prep.
- [burstkeeper] 2026-10-17T10:15:44Z 220851 (v3-019) CANCELLED 10:06:16 (not by me); 220965 RUNNING crsuse2-m2m-v3-006 (10.245.144.8) ECC 0, nothing stopped; prep rc=0 SEED_OK; cache 006 new so 16/14 locks 0. 220910 PENDING

## 2026-10-17 10:1x UTC: ofg2h17 (G2K1 + tok4/detok4, C56 C56) launched on v3-006 (job 220965; prep 10:15 OK) + pyspy.
- [watch136] 2026-10-17T10:22Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2336619 alive, waiting
- [burstkeeper] 2026-10-17T10:35:47Z 220910 held JobHoldMaxRequeue (Restarts=5, earlier JobLaunchFailure agent rejected), left in place; submitted 221042 (spur). 220965 RUNNING v3-006
- [watch136] 2026-10-17T11:19Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2336619 alive, waiting
- [burstkeeper] 2026-10-17T11:34:32Z 221042 held JobHoldMaxRequeue (Restarts=5), left in place; submitted 221260 (spur). 220965 RUNNING v3-006

## 2026-10-17 11:3x UTC: main scancelled 221042 (amd-spur JobHoldMaxRequeue; 3rd in a row: 220733, 220910, 221042). Holds: 221260 (spur) pending, 220965 (v2, v3-006) running ofg2h17. burstkeeper being restarted with the cancel/stale-container rules in its brief.
- [burstkeeper2] 2026-10-17T11:35:17Z start: spur hold 221260 PENDING (Resources); v2 hold 220965 RUNNING crsuse2-m2m-v3-006 (main experiment, untouched)
- [watch136] 2026-10-17T12:17Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 12:10:44Z -> relaunched
- [burstkeeper2] 2026-10-17T12:25:15Z 221260 RUNNING crsuse2-m2m-219 (10.245.152.72), ECC 0/0; docker stopped: qwen-ci-219 dsv4-atom-mono-e2e primus_v26.8_dsv3; prep 219 seed launched
- [burstkeeper2] 2026-10-17T12:34:30Z prep 219: rc=0 SEED_OK image 9d755dd6; cache P=16 D=14 locks 0/0. heartbeat: spur 221260 RUNNING 219, v2 220965 RUNNING v3-006

## 2026-10-17 12:3x UTC: amd-spur 221260 on m2m-219 (burstkeeper2 stopped qwen-ci-219, dsv4-atom-mono-e2e, primus_v26.8_dsv3; prep OK). New config.yihou.mc4340n10.sh (+m2m-219). ofg3h6 (G3K1 + tok4/detok4, C64 C64) launched + pyspy.
- [burstkeeper2] 2026-10-17T12:44:33Z 221260 PREEMPTED (219, 12:24:17-12:44:17); replacement amd-spur hold 221521 PENDING

## 2026-10-17 12:44 UTC: 221260 (m2m-219) PREEMPTED at exactly 20 min -> ofg3h6 lost in bring-up. Replacement 221521 pending.
- 12:35 UTC: ofg2h17 A1 C56 (v3-006, G2K1 + tok4/detok4, fresh) = 19,746/chip, qps 1.151, ISL 136,376, TTFT 21.76/52.26, TPOT 15.2, GPU 0.934, CPU 0.011, err 3. Same config as ofg2h5 A1 (v3-008) 21,766 -> -9.3% on a different node. A2 C56 started 12:35:58.
- [burstkeeper2] 2026-10-17T12:54:49Z 221521 RUNNING crsuse2-m2m-274 (10.245.157.34), ECC 0/0, no containers; prep 274 seed launched
- [burstkeeper2] 2026-10-17T13:02:00Z prep 274: rc=0 SEED_OK image 9d755dd6; cache P=16 D=14 locks 0/0

## 2026-10-17 13:0x UTC: amd-spur 221521 on m2m-274 (prep OK 13:00). ofg3h7 (G3K1 + tok4/detok4, C64 C64) launched + pyspy.
- [watch136] 2026-10-17T13:15Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 4178662 alive, waiting
- [burstkeeper2] 2026-10-17T13:21:51Z 221521 CANCELLED (274, 12:53:18-13:13:19, not by keeper); replacement amd-spur hold 221627 PENDING

## 2026-10-17 13:13 UTC: 221521 (m2m-274) CANCELLED at exactly 20 min -> ofg3h7 lost. amd-spur holds today: 219097, 220567, 220644, 221260, 221521 all ended at 20:00 after start; 220733/220910/221042 JobHoldMaxRequeue. Replacement 221627 pending.
- [burstkeeper2] 2026-10-17T13:41:32Z heartbeat: spur 221627 PENDING (Resources); v2 220965 RUNNING v3-006

## 2026-10-17 14:00 UTC: 220965 (v3-006) PREEMPTED 14:00:02 after 3h52m -> ofg2h17 A2 C56 lost ~5 min before the end of the 3600 s window (no JSON). Kept: ofg2h17 A1 19,746. Replacement v2 221701 pending; amd-spur 221627 RUNNING on m2m-228 (burstkeeper2 prepping).
- [burstkeeper2] 2026-10-17T14:01:37Z 220965 PREEMPTED (v3-006, 10:08:25-14:00:02); replacement v2 hold 221701 PENDING. 221627 RUNNING crsuse2-m2m-228 (10.245.150.103), ECC uncorr 0, GPU6 correctable=16814 (others 0), no containers; prep 228 seed launched
- [burstkeeper2] 2026-10-17T14:08:55Z prep 228: rc=0 SEED_OK, P=16 D=14 locks 0/0. 221701 RUNNING crsuse2-m2m-v3-031 (10.245.145.118), ECC 0/0, no containers; prep 031 seed launched

## 2026-10-17 14:0x UTC: amd-spur 221627 on m2m-228 (GPU6 correctable ECC 16814, uncorrectable 0; prep OK). New config mc4340n11 (+m2m-228). ofg3h8 (G3K1 + tok4/detok4, C64 C64) launched + pyspy. v2 221701 RUNNING on v3-031 (burstkeeper2 prepping).
- [watch136] 2026-10-17T14:12Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 4178662 alive, waiting
- [burstkeeper2] 2026-10-17T14:15:46Z prep 031: rc=0 SEED_OK, P=16 D=14 locks 0/0. heartbeat: spur 221627@228, v2 221701@v3-031 both RUNNING+prepped

## 2026-10-17 14:1x UTC: v2 221701 on v3-031 prepped 14:14; ofg2h18 (G2K1 + tok4/detok4, C56 C56) launched + pyspy.
- [burstkeeper2] 2026-10-17T14:25:50Z 221627 CANCELLED (228, 13:56:17-14:16:18); 221701 PREEMPTED (v3-031, 14:01:32-14:24:13); replacements spur 221725, v2 221726 PENDING

## 2026-10-17 14:24 UTC: 221627 (m2m-228) CANCELLED 14:16:18 (20 min) -> ofg3h8 lost; 221701 (v3-031) PREEMPTED 14:24:13 (23 min) -> ofg2h18 lost. Replacements 221725 (spur), 221726 (v2) pending.
- [burstkeeper2] 2026-10-17T15:05:08Z heartbeat: spur 221725 PENDING, v2 221726 PENDING (Resources)
- [watch136] 2026-10-17T15:09Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 4178662 alive, waiting
- [burstkeeper2] 2026-10-17T15:24:41Z 221725 RUNNING crsuse2-m2m-011 (10.245.151.9), ECC 0/0, no containers; prep 011 seed launched
- [burstkeeper2] 2026-10-17T15:31:32Z prep 011: rc=0 SEED_OK, P=16 D=14 locks 0/0

## 2026-10-17 15:3x UTC: amd-spur 221725 on m2m-011 (prep OK 15:29). Config mc4340n12 (+m2m-011). ofg3h9 (G3K1 + tok4/detok4, C64 C64) launched + pyspy.
- [burstkeeper2] 2026-10-17T16:04:06Z 221725 CANCELLED (011, 15:23:23-15:54:53); replacement spur hold 221768 PENDING. 221726 RUNNING crsuse2-m2m-v3-027 (10.245.155.20), ECC 0/0, no containers; prep 027 seed launched

- 2026-10-17 16:05Z: ofg3h9 (m2m-011, hold 221725, G3 C64) LOST - job CANCELLED 15:54:53 after ~31 min (not by us), during A1; no score. First amd-spur hold to exceed 20 min. Replacement amd-spur hold 221768 PENDING. v2 hold 221726 RUNNING on crsuse2-m2m-v3-027 (10.245.155.20); prep in progress; env.ofg2h19-v3027 written (G2 C56 C56, diff vs ofg2h18 = node/IP/job/name only).
- [watch136] 2026-10-17T16:06Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 4178662 alive, waiting
- [burstkeeper2] 2026-10-17T16:10:57Z prep 027: rc=0 SEED_OK, P=16 D=14 locks 0/0. heartbeat: spur 221768 PENDING, v2 221726@v3-027 RUNNING

- 2026-10-17 16:11 UTC: v2 221726 on v3-027 prep OK 16:10. ofg2h19 (G2K1 + tok4/detok4, C56 C56) launched + pyspy; tokdiff vs ofg2h18 empty.

- 2026-10-17 16:56Z: ofg2h19 PD warm-up 16:28:53-16:30:50 (~2 min), smoke ok, A1 bench C56 started 16:51:25; logs still growing 16:55:36; 0 error hits. spurctld unreachable 16:53-16:55 (squeue omitted 221726 once at 16:53:06, then tcp connect errors; sacct pool timeout).
- [watch136] 2026-10-17T17:04Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 16:57:33Z -> relaunched
- [burstkeeper2] 2026-10-17T17:21:47Z spur controller unstable since ~16:53Z (spurctld connect timeouts, sbatch "not the Raft leader"); v2 221726 (v3-027) gone from squeue at 16:54Z, end state unknown; v2 replacement sbatch retrying with squeue guard; spur 221768 PENDING at last good read

- 2026-10-17 17:23Z: squeue answers again; lists only 221768 PENDING. 221726 absent and squeue -w v3-027 empty; sacct still pool-timeout. Yet ofg2h19 A1 logs keep growing (decode-0.a1.log 17:22:40, bench log 17:22:24); 0 error hits. Job state unknown.

- 2026-10-17 17:55Z: ofg2h19 LOST - A1 bench client and both engine logs stop at 17:43:40 (bench log ends "Terminated", client.shell-exit-code empty), ~52 of 60 min into A1; no score. Controller still unreachable, so 221726 end state not confirmed. Last aiperf interim: prefix_cache_hit 95.6%, kv_usage 85.0%, cpu_kv_usage 84.8%. G2 C56 back to queue head (next name ofg2h20).
- [watch136] 2026-10-17T18:01Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1740554 alive, waiting
- [burstkeeper2] 2026-10-17T18:12:41Z outage 16:53-18:12Z; 221726 sacct COMPLETED 15:59:27-16:43:45 (v3-027); v2 replacement 221869 PENDING (sbatch via spur005; spur006/007 sbatch refused "re-forward"); spur 221768 PENDING

- 2026-10-17 18:15Z (burstkeeper2): controller outage 16:53-18:12Z. sacct 221726 = COMPLETED 15:59:27-16:43:45 (as printed by sacct). Note: our files show the node alive later - launch 16:11:25Z, last log write 17:43:40Z, i.e. sacct End is 59m55s before our last write; time base of sacct output not verified. v2 replacement 221869 PENDING (sbatch via spur005); amd-spur 221768 PENDING.
- [burstkeeper2] 2026-10-17T18:57:34Z heartbeat: spur 221768 PENDING (Priority); v2 221869 PENDING Reason=None since 18:12Z (only pending job in amd-spur-v2)
- [watch136] 2026-10-17T18:59Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1740554 alive, waiting

- 2026-10-17 19:35Z: v2 hold 221869 PENDING Reason=None since 18:12Z (submitted at tail of controller outage) while sinfo shows 18 idle amd-spur-v2 nodes; told burstkeeper2 to scancel it (ours) and resubmit one. 221768 PENDING (Priority).
- [watch136] 2026-10-17T19:56Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1740554 alive, waiting
- [burstkeeper2] 2026-10-17T19:59:32Z heartbeat: spur 221768 PENDING (Priority); v2 221869 PENDING Reason=None (asked main whether to resubmit)

- 2026-10-17 20:55Z: burstkeeper2 did not act on approved cancel+resubmit for 80 min; main scancel 221869 (ours, PENDING) and resubmitted one v2 hold = 221956 (PENDING). 221768 PENDING.
- [watch136] 2026-10-17T20:53Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1740554 alive, waiting
- [burstkeeper2] 2026-10-17T21:01:27Z 221869 CANCELLED 20:53:16 (not by keeper2), v2 221956 submitted 20:53:27 by another actor, adopted as v2 hold; spur 221768 PENDING
- [watch136] 2026-10-17T21:52Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 21:44:29Z -> relaunched

- 2026-10-17 21:54Z: resubmitted v2 hold 221956 PENDING Reason=None for 60 min; sinfo amd-spur-v2: 18 idle, 7 down, 3 drain; it is the only pending job in the partition. Not resubmitting again (same symptom after resubmit). 221768 PENDING (Priority).
- [burstkeeper2] 2026-10-17T21:54:56Z heartbeat: spur 221768 PENDING (Priority); v2 221956 PENDING (None)
- [watch136] 2026-10-17T22:50Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3585068 alive, waiting
- [burstkeeper2] 2026-10-17T22:56:45Z heartbeat: spur 221768 PENDING (Priority); v2 221956 PENDING (None)
- [watch136] 2026-10-17T23:47Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3585068 alive, waiting
- [burstkeeper2] 2026-10-17T23:58:34Z heartbeat: spur 221768 PENDING (Priority); v2 221956 PENDING (None)
- [watch136] 2026-10-18T00:44Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3585068 alive, waiting
- [burstkeeper2] 2026-10-18T01:00:36Z heartbeat: spur 221768 PENDING (Priority); v2 221956 PENDING (None)
- [watch136] 2026-10-18T01:41Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3585068 alive, waiting
- [burstkeeper2] 2026-10-18T02:00:11Z heartbeat: spur 221768 PENDING (Priority); v2 221956 PENDING (None)
- [watch136] 2026-10-18T02:39Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 02:32:42Z -> relaunched
- [burstkeeper2] 2026-10-18T03:00:53Z heartbeat: spur 221768 PENDING (None); v2 221956 PENDING (None)

- 2026-10-18 03:10Z (burstkeeper2): both holds PENDING Reason=None - 221768 (amd-spur, since ~02:30Z, was Priority) and 221956 (v2, since 20:53Z). Cluster-wide only these 2 pending jobs; sinfo amd-spur-v2 18 idle / 7 down / 3 drain; amd-spur shows 0 nodes in sinfo.
- [watch136] 2026-10-18T03:36Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1327833 alive, waiting
- [burstkeeper2] 2026-10-18T04:01:40Z heartbeat: spur 221768 PENDING (None); v2 221956 PENDING (None)
- [burstkeeper2] 2026-10-18T04:13:35Z 221768 RUNNING crsuse2-m2m-062 (10.245.146.152) ECC 0/0, only crusoe agents, nothing stopped; 221956 RUNNING crsuse2-m2m-v3-012 (10.245.151.129) ECC 0/0, docker stopped: sglang_chuschen dev_xb glm53-madvise-owner yzhou_bmm mm_run; prep 062 + 012 seed launched
- [burstkeeper2] 2026-10-18T04:20:47Z prep 062 + 012: rc=0 SEED_OK image 9d755dd6, P=16 D=14 locks 0/0 both (docker stop on v3-012 done via background, verified empty)

- 2026-10-18 04:2xZ: holds RUNNING ~04:05Z (reasons None -> AccountingUnavailable 04:01Z -> RUNNING). v2 221956 = v3-012 (10.245.151.129; burstkeeper2 docker-stopped 5 others containers, no rm), prep OK 04:20:35 -> ofg2h20 (G2K1+tok4, C56 C56) launched 04:21:00, tokdiff vs ofg2h19 empty. amd-spur 221768 = m2m-062 (10.245.146.152), prep OK 04:19:38 -> new config mc4340n13 (n12 + m2m-062) -> ofg3h10 (G3K1+tok4, C64 C64) launched 2026-10-18T04:21:18, tokdiff vs ofg3h9 empty. /shared_nfs 99% (4.7T avail).
- [burstkeeper2] 2026-10-18T04:30:05Z 221768 CANCELLED (062, 04:03:37-04:23:38); replacement spur hold 222214 PENDING

- 2026-10-18 04:25Z: ofg3h10 LOST - amd-spur 221768 (m2m-062) CANCELLED 04:03:37-04:23:38 (20 min, not by us), during prefill bring-up; no score. G3 C64 back to queue head (next ofg3h11). Replacement amd-spur hold 222214 PENDING. ofg2h20 on v2 v3-012 continues.
- [watch136] 2026-10-18T04:34Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1327833 alive, waiting
- [burstkeeper2] 2026-10-18T04:40:49Z 222214 RUNNING crsuse2-m2m-200 (10.245.159.37) ECC 0/0; docker stopped: buildx_buildkit_aic-cache0 hyperloom-local dev_xb; prep 200 seed launched
- [burstkeeper2] 2026-10-18T04:41:30Z 222214 RUNNING crsuse2-m2m-200 (10.245.159.37) ECC 0/0; docker stopped: buildx_buildkit_aic-cache0 hyperloom-local dev_xb; prep 200 seed launched
- [burstkeeper2] 2026-10-18T04:48:28Z prep 200: rc=0 SEED_OK, P=16 D=14 locks 0/0

- 2026-10-18 2026-10-18T04:48:37: amd-spur 222214 on m2m-200 (10.245.159.37) RUNNING ~04:35Z, prep OK 04:46:12. Config mc4340n14 (n13 + m2m-200). ofg3h11 (G3K1+tok4, C64 C64) launched; tokdiff vs ofg3h10 run below.
  (ofg3h11 tokdiff vs ofg3h10: empty; node had img 9d755dd6, kfd 0, 0 containers at launch 04:48:37Z.)
- [burstkeeper2] 2026-10-18T04:50:43Z 222214 CANCELLED (200, 04:30:01-04:50:03); replacement spur hold 222290 PENDING. note: amd-spur holds repeatedly end at ~20:00 runtime

- 2026-10-18 04:52Z: ofg3h11 LOST - amd-spur 222214 (m2m-200) CANCELLED 04:30:01-04:50:03 (20 min), ~1.5 min after launch; no score. G3 C64 back to queue head (next ofg3h12). Replacement 222290 PENDING. amd-spur lifetimes since 10-17 (burstkeeper2): 20:00 (PREEMPTED), 20:01, 20:01, 31:30, 20:01, 20:02. v2: 3h52, 22:41, 44:18 (COMPLETED), 221956 running 45+ min.
- [burstkeeper2] 2026-10-18T04:59:55Z 222290 PENDING JobHoldMaxRequeue -> scancel by keeper2; replacement spur hold 222317 PENDING
- [burstkeeper2] 2026-10-18T05:17:12Z 222317 JobHoldMaxRequeue (Restarts=5) -> scancel; 222352 submitted, 1 JobLaunchFailure (gpu/resource allocation mismatch) then RUNNING crsuse2-m2m-241 (10.245.155.113) ECC 0/0, no containers; prep 241 seed launched
- [burstkeeper2] 2026-10-18T05:17:40Z 222317 JobHoldMaxRequeue (Restarts=5) -> scancel; 222352 RUNNING crsuse2-m2m-241 (10.245.155.113) after 1 JobLaunchFailure, ECC 0/0, no containers; prep 241 seed launched
- [burstkeeper2] 2026-10-18T05:23:48Z prep 241: rc=0 SEED_OK, P=16 D=14 locks 0/0

- 2026-10-18 2026-10-18T05:23:58: amd-spur 222352 on m2m-241 (10.245.155.113) RUNNING ~05:10Z after 222290/222317 JobHoldMaxRequeue (cancelled by burstkeeper2) and one JobLaunchFailure; prep OK 05:22:02. Config mc4340n15 (n14 + m2m-241). ofg3h12 (G3K1+tok4, C64 C64) launched. ofg2h20 A1 bench started 05:02:53 on v3-012.
- [watch136] 2026-10-18T05:31Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1327833 alive, waiting
- [burstkeeper2] 2026-10-18T05:37:53Z 222352 CANCELLED (241, 05:10:15-05:30:16); replacement spur hold 222428 PENDING

- 2026-10-18 05:32Z: ofg3h12 LOST - amd-spur 222352 (m2m-241) CANCELLED 05:10:15-05:30:16 (20:01), ~6 min after launch; no score. Next G3 C64 name ofg3h13. Replacement 222428 PENDING. ofg2h20 continues on v2 v3-012.
- [burstkeeper2] 2026-10-18T05:47:00Z 221956 CANCELLED (v3-012, 04:03:53-05:39:38); replacement v2 hold 222453 PENDING; spur 222428 PENDING JobLaunchFailure (since ~05:4x)

- 2026-10-18 05:42Z: ofg2h20 LOST - v2 221956 (v3-012) CANCELLED 04:03:53-05:39:38 (1h36, not by us), during A1 (started 05:02:53); no score. G2 C56 back to queue head (next ofg2h21). Replacement v2 222453 PENDING; amd-spur 222428 PENDING (JobLaunchFailure).
- [burstkeeper2] 2026-10-18T05:56:13Z 222428 JobHoldMaxRequeue -> scancel, replacement spur hold 222476 PENDING. 222453 RUNNING crsuse2-m2m-v3-006 (10.245.144.8) ECC 0/0, no containers; prep 006 seed launched
- [burstkeeper2] 2026-10-18T05:56:46Z 222428 JobHoldMaxRequeue -> scancel, replacement spur hold 222476 PENDING. 222453 RUNNING crsuse2-m2m-v3-006 (10.245.144.8) ECC 0/0, no containers; prep 006 seed launched
- [burstkeeper2] 2026-10-18T06:02:56Z prep 006: rc=0 SEED_OK, P=16 D=14 locks 0/0

- 2026-10-18 06:05Z: v2 222453 on v3-006 (10.245.144.8) RUNNING ~05:50Z, prep OK 06:02:04, no containers. ofg2h21 (G2K1+tok4, C56 C56) launched 06:04:46 (first spur exec failed on controller connect error, retry via spur007 ok); tokdiff vs ofg2h20 empty. amd-spur 222428 JobHoldMaxRequeue -> cancelled by burstkeeper2; replacement 222476 PENDING.
- [burstkeeper2] 2026-10-18T06:06:30Z 222476 JobHoldMaxRequeue (Restarts=5) -> scancel; replacement spur hold 222494 PENDING
- [burstkeeper2] 2026-10-18T06:10:26Z 222494 JobHoldMaxRequeue (launch: gpu/resource mismatch, agent rejected launch) -> scancel; replacement spur hold 222500 PENDING
- [burstkeeper2] 2026-10-18T06:19:30Z 222500 JobHoldMaxRequeue -> scancel; resubmitted
- [watch136] 2026-10-18T06:28Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1327833 alive, waiting
- [burstkeeper2] 2026-10-18T06:28:36Z 222528 JobHoldMaxRequeue -> scancel; resubmitted (amd-spur sinfo: 5 idle 217,147,150,242,274)
- [burstkeeper2] 2026-10-18T06:37:40Z 222546 JobHoldMaxRequeue -> scancel; resubmitted
- [burstkeeper2] 2026-10-18T06:47:11Z 222575 RUNNING crsuse2-m2m-250 (10.245.158.155) ECC 0/0, no containers; prep 250 seed launched
- [burstkeeper2] 2026-10-18T06:53:19Z prep 250: rc=0 SEED_OK, P=16 D=14 locks 0/0. both holds RUNNING (250, v3-006)

- 2026-10-18 06:54Z: amd-spur 222575 on m2m-250 (10.245.158.155) RUNNING ~06:38Z after 222500/222528/222546 JobHoldMaxRequeue loops; prep OK 06:51:37. Config mc4340n16 (n15 + m2m-250). ofg3h13 (G3K1+tok4, C64 C64) launched 06:53:28; tokdiff vs ofg3h12 empty. ofg2h21 A1 bench started 06:34:56 on v3-006.
- [burstkeeper2] 2026-10-18T07:04:55Z 222575 CANCELLED (250, 06:38:01-06:58:04); replacement spur hold 222652 PENDING

- 2026-10-18 07:00Z: ofg3h13 LOST - amd-spur 222575 (m2m-250) CANCELLED 06:38:01-06:58:04 (20:03), ~5 min after launch; no score. Next G3 C64 name ofg3h14. Replacement 222652 PENDING.
- [burstkeeper2] 2026-10-18T07:13:59Z 222652 JobHoldMaxRequeue -> scancel; resubmitted
- [burstkeeper2] 2026-10-18T07:23:03Z 222662 JobHoldMaxRequeue -> scancel; resubmitted
- [watch136] 2026-10-18T07:26Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 07:19:23Z -> relaunched

- 2026-10-18 08:25Z: ofg2h21 A1 C56 done 08:03:04 (bench rc=0, ~88 min wall): score 19,650 qps 1.153 ISL 135,411 TTFT 21.62/56.23 TPOT 15.9 gpu 0.936 cpu 0.014 err 4. Same config/node earlier: ofg2h17 A1 19,746 (v3-006); ofg2h5 A1 21,766 (v3-008). A2 C56 started 08:03:04.
- [watch136] 2026-10-18T08:23Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2817117 alive, waiting
- [watch136] 2026-10-18T09:21Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2817117 alive, waiting

- 2026-10-18 2026-10-18T09:55:14: ofg2h21 A2 C56 DONE 09:30:44: score 21,780 qps 1.220 ISL 141,883 TTFT 21.13/54.37 TPOT 13.2 gpu 0.918 cpu 0.019 err 4 (A1 19,650). Stack stopped (run.v3s stop x2; rc=1 "cleanup completed with errors", spur exec transport errors on 1st try; after 2nd: 0 containers, kfd 0). ofg3h14 (G3K1+tok4, C64 C64) launched on same node v3-006 (222453), tokdiff vs ofg3h13 empty.
- [watch136] 2026-10-18T10:18Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2817117 alive, waiting
- [watch136] 2026-10-18T11:15Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 2817117 alive, waiting
- [watch136] 2026-10-18T12:13Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 12:06:14Z -> relaunched

- 2026-10-18 12:15Z: ofg3h14 A1 C64 done 11:54:55 (rc=0): score 17,669 qps 1.080 ISL 129,956 TTFT 33.74/93.13 TPOT 16.4 gpu 0.857 cpu 0.064 err 1. A2 C64 started 11:54:56. 136 still unreachable (~71 h).
- [watch136] 2026-10-18T13:10Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 4126121 alive, waiting

- 2026-10-18 13:25Z: ofg3h14 A2 C64 json written: score 18,333 qps 1.144 ISL 127,166 TTFT 35.57/88.68 TPOT 13.2 gpu 0.773 cpu 0.131 err 7 (A1 17,669). First C64 offload result (G3).

- 2026-10-18 13:28Z: ofg3h14 chain DONE 13:23:52; stack stopped (run.v3s stop rc=1, controller connect error; node then 0 containers, kfd 0). ofg2h22 (G2K1+tok4, C64 C64) launched 13:27:56 on v3-006 (222453); env = ofg2h21 with CONC 56->64; tokdiff vs ofg2h21 empty, dryrun differs only in CONC=64.
- [watch136] 2026-10-18T14:08Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 4126121 alive, waiting
- [watch136] 2026-10-18T15:05Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 4126121 alive, waiting

- 2026-10-18 15:54Z: ofg2h22 A1 C64 done 15:26:51: score 17,605 qps 1.080 ISL 129,540 TTFT 35.22/87.93 TPOT 15.8 gpu 0.879 cpu 0.042 err 0 (ofg3h14 G3 A1 C64 17,669). A2 started 15:26:51.
- [watch136] 2026-10-18T16:02Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 4126121 alive, waiting
- [watch136] 2026-10-18T17:00Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 16:53:15Z -> relaunched

- 2026-10-18 17:05Z: ofg2h22 A2 C64 DONE 16:56:59: score 18,392 qps 1.106 ISL 132,070 TTFT 37.26/81.33 TPOT 12.1 gpu 0.837 cpu 0.073 err 2 (A1 17,605). Stack stopped (stop rc=1 on controller connect error; node then 0 containers, kfd 0). ofg4h4 (G4K1+tok4, C64 C64) launched 17:03:56 on v3-006; env = ofg2h22 with OFF_G 2->4; tokdiff: prefill offload-group-size 2->4 only. 136 still unreachable (~76 h).
- [watch136] 2026-10-18T17:57Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1380588 alive, waiting

- 2026-10-18 18:25Z: /shared_nfs 100% full again (42G avail of 420T). ofg4h4 A1 logs (bench + both server logs) stopped updating at 18:18:48-18:18:58; last bench interim 18:18:17 normal (kv 93%, cpu_kv 93.6%, queue 8r/8w). Node check via spur exec failed (controller connect error). Engine state unknown. Not deleting anything.
- [watch136] 2026-10-18T18:55Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1380588 alive, waiting

- 2026-10-18 19:02Z: ofg4h4: chain_cap2 driver + log writers gone from node (no process found 18:5x; chain log last line 17:26:57); engines healthy (health 200 x3). A1 client kept running, profiling timed out 18:53:24 (sent 3889, completed 3829), container Exited(1) with JSONDecodeError; aiperf artifacts written 18:56; no agentx_conc64.json -> A1 unscored. Disk /shared_nfs 709G->385G free. Continued on live stack: cont_bench.yihou.sh env.ofg4h4 - 64 (flush both legs -> A2 C64) launched 19:00:07, log runs/ofg4h4.cont.log.
  19:00:07 node-side tee to status.tsv -> "Disk quota exceeded" (EDQUOT); 19:03 test write from spur006 to /shared_nfs ok; df 311G free. A2 client started 19:00:09.

- 2026-10-18 19:25Z: ofg4h4 A2 FAILED 19:12:02 (bench rc=1, tee "Disk quota exceeded" on runner.log/stdout; was still in warmup, returned 58/707). /shared_nfs 30G free. Paused: no relaunch until space returns. Stack (engines) still up on v3-006, hold 222453 kept.
- [watch136] 2026-10-18T19:52Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1380588 alive, waiting

- 2026-10-18 19:58Z: /shared_nfs back to 2.7T free. Found ofg4h4 A2 client container still Up (started 19:00:09) - only the host-side tee wrapper failed (rc=1); engines health 200. Letting it run; output dir names unchanged (briefly mv-ed aside and back, same FS).
- [watch136] 2026-10-18T20:49Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 1380588 alive, waiting

- 2026-10-18 21:05Z: ofg4h4 A2 client finished 20:28:58 but exited 1: process_agentic_result JSONDecodeError. Cause found: profile_export.jsonl has exactly one NUL-filled line in each of A1 (line 2473, 398,830 B) and A2 (line 558, 51,158 B) - written during the EDQUOT window. UNOFFICIAL scores: copies with the NUL line dropped (originals untouched) in N/unofficial/ofg4h4-{A1,A2}-nulline-dropped, processed with the same process_agentic_result + runtime.env on the node: A1 16,905 / A2 17,693 tok/s/GPU. Not like-for-like with official scores (records lost in the NUL block; A2 was a cont_bench continuation without the driver).
  (time of the entry above was ~20:55Z, not 21:05Z.)
- 2026-10-18 20:58Z: ofg4h4 stack stopped (stop rc=1; node then 0 containers, kfd 0). /shared_nfs 4.0T free. ofg4h5 (G4K1+tok4, C64 C64; rerun of ofg4h4) launched 20:57:22 on v3-006; tokdiff vs ofg4h4 empty.
- [watch136] 2026-10-18T21:47Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 21:40:11Z -> relaunched
- [watch136] 2026-10-18T22:44Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3055390 alive, waiting

- 2026-10-18 22:55Z: ofg4h5 A1 C64 done 22:48:02: score 18,192 qps 1.112 ISL 129,927 TTFT 34.66/102.97 TPOT 15.7 gpu 0.835 cpu 0.083 err 2. A2 started 22:48:02.
- [watch136] 2026-10-18T23:42Z no change for 136: 135/136/139 ssh timeout, 138 ok; 137 once gave empty output at 20 s limit, recheck ok (~20 s to answer, up 52 d); waiter pid 3055390 alive, waiting

- 2026-10-19 00:25Z: ofg4h5 A2 C64 DONE 00:17:01: score 18,535 qps 1.128 ISL 130,498 TTFT 35.10/80.49 TPOT 13.5 gpu 0.737 cpu 0.165 err 3 (A1 18,192). Queue (G2C56, G3C64, G2C64, G4C64) now complete. Stack left running on v3-006; hold 222453 kept; awaiting user on next work.
- [watch136] 2026-10-19T00:39Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3055390 alive, waiting
- [watch136] 2026-10-19T01:36Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 3055390 alive, waiting

- 2026-10-19 01:55Z: v2 hold 222453 (v3-006) gone from squeue (ran ~19.5 h; all ofg4h5 results already on NFS). Queue complete; told burstkeeper2 to PAUSE submissions and cancel pending amd-spur 222685 so an idle hold does not stop other users containers. Awaiting user.
- [watch136] 2026-10-19T02:34Z 135/136/139 ssh timeout, 137/138 ok; waiter STOP (4 h) at 02:27:02Z -> relaunched
- [burstkeeper2] 2026-10-19T03:26:20Z GAP: keeper not polling 10-18 07:32Z..10-19 03:25Z. 222453 CANCELLED 10-19 01:51:13 (v3-006); 222685 CANCELLED 10-19 02:11:44. resubmitted spur 224284, v2 224285 PENDING

- 2026-10-19 03:35Z: burstkeeper2 reports its polling stopped 10-18 07:32Z-10-19 03:25Z (session not executing). sacct per burstkeeper2: 222453 (v3-006) CANCELLED 10-19 01:51:13 (ran from 10-18 05:46:55); 222685 CANCELLED 02:11:44 (not by burstkeeper2). It resubmitted 224284/224285 having missed the pause; main scancel-ed both (ours, PENDING). Paused, no holds.
- [burstkeeper2] 2026-10-19T03:27:41Z amd-spur hold ended: 224284 CANCELLED Unknown 2026-10-19T03:26:41; replacement 224299 SUBMITTED 
- [burstkeeper2] 2026-10-19T03:28:08Z 224284/224285 CANCELLED 03:26:41-42 ~30s after submit, not by keeper2; spur 224299 PENDING; keeper2 pausing submits pending main confirmation
- [watch136] 2026-10-19T03:32Z 135/136/139 ssh timeout, 137/138 ok; waiter pid 325157 gone, no STOP line (last log 02:36:31Z unreachable) -> relaunched
- [burstkeeper2] 2026-10-19T03:33:19Z 224299 CANCELLED 03:28:14 ~30s after submit (not keeper2). keeper2 halted submissions; observe-only until main confirms

- 2026-10-19 03:5xZ: packup glm52.v0521-offloaderv2-burst-c56c64.packup_20261019 (57M, 854 files, none >4MB, secret scan clean, SHA256SUMS ok); CONCLUSIONS row added. burstkeeper2 paused (no holds). 136 still TCP-unreachable from spur005 (03:3x).

- 2026-10-19 04:3xZ: conc-drop analysis (read-only) -> N/analysis_conc_20261019/rows.jsonl (scripts N/conc_anal.yihou.py, isl_turn.yihou.py, perdp.yihou.py). Window = aiperf profiling start..end. Findings: prefill chunk = 8192 new tok/DP (32768/4); full batches 62% (C48 pf90) -> 67% (G2 C56) -> 77-80% (C64). G2 C56->C64 (same node): qps -9.3%, ISL -6.9% (turn idx 27.4->26.3, conversations 566->596), GPU hit 0.918->0.837, new tok/request +29%, prefill new tok/s +17%, sat. s/1k new tok 0.197->0.236; no-offload tb56n C56->C64 same signature (-16.1%).
- [burstkeeper2] 2026-10-19T04:01:13Z heartbeat: observe-only, no yihou jobs in queue, awaiting main

- 2026-10-19 04:5xZ: prefill KV time series -> N/analysis_conc_20261019/kv_ts.jsonl (script N/kv_ts.yihou.py; 10-min buckets vs aiperf profiling start). GPU KV free ~0.1% throughout profiling for all runs; in-flight (used) share mean over min 10-50: C48 12-14%, C56 21-23% (none 30%), C64 28-32% (none 32%); evictable prefix tokens/DP G2 2.62M->2.30M, G4 2.26M (136)->2.03M. HiCache host (10.0M tok/DP) refills after flush_cache and reaches ~100% at ~+30 min (C64), ~+40-50 (C56), ~+50-60 (C48); tb56n host (3.65M/DP) full before window.

- 2026-10-19 05:2xZ: expert-8 comparison (N/analysis_conc_20261019/rows_x8.jsonl, kv_ts_x8.jsonl, isl_x8.jsonl; conc_anal2 = regex accepts EP tag). No C56 expert-8 run exists: x8p/x8pc/x128p only C48; x8d2 (decode 8 experts) C48 23,591 -> C52 21,713. x8p vs x8pc (same 8 experts EP4, KV 3.67M vs 2.43M/DP): 25,905 vs 22,489, GPU hit .950 vs .847, s/1k new tok .131 vs .152. x128p A1 vs ofg2 A1 (KV 3.32M vs 3.38M, GPU hit .959 vs .960, miss 3.8% vs 3.7%): 24,316 vs 21,157, s/1k new tok .138 vs .191, TTFT 5.6 vs 12.3 s (cross-node 085 vs 288).
- [watch136] 2026-10-19T04:32Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 754320 alive, waiting

- 2026-10-19 05:4xZ: decode-side gauges (N/analysis_conc_20261019/dec_q.jsonl, script N/dec_q.yihou.py; mean/p50/p90/max of 4-DP sums): running flat 15-17 (C48..C64 offload; max_running 96), decode num_queue 0, retracted 0; transfer queue 11-12 (C48) -> 24-25 (C56) -> 38-40 (C64); prealloc queue 0 -> 0.1-0.2 -> 0.8-1.0 (p90 3, max 9-14); decode token_usage 0.42-0.44 -> 0.59-0.60 -> 0.73-0.74 (p90 0.90); gen_throughput 1288-1340 -> 1297-1336 -> 1187-1228.

- 2026-10-19 06:0xZ: decode running vs Littles law (qps x (e2el-ttft)): within 2-7% for all 11 runs (e.g. G2 C56 16.5 vs 16.8 meas; G2 C64 13.6 vs 14.0; none C64 10.5 vs 11.3). Decode KV: running seq_lens/cap 0.22-0.28 at every conc; total used 0.42 -> 0.59 -> 0.73 (difference = prealloc for transfer queue). decode num_queue 0 throughout.

- 2026-10-19 06:3xZ: TTFT stage split by Littles law (n/qps, prefill gauges; script N/stage.yihou.py), sums match TTFT within ~2 s: G2 C56 queue 4.4 / run 4.2 / inflight 14.9 s; G2 C64 11.1 / 4.8 / 23.0; pf90 C48 2.5 / 3.0 / 5.7; x8p C48 1.1 / 1.7 / 2.9. Inflight = prefill done, KV transfer not yet Success (sglang disaggregation/prefill.py process_disagg_prefill_inflight_queue, read in local checkout 0b3bb0cbe3, not the image commit). Prefill kv_used / (n_run+n_inflight) = 120-130k tok/request (~ISL): inflight requests pin prefill KV.

- 2026-10-19 07:0xZ: inflight semantics verified at image commit 11b1fb21de (upstream sgl-project/sglang, fetched to local /tmp/yihou/sgl_11b1): prefill.py L1054-1056 send_kv_chunk(last_chunk) then set_prefill_transfer_queue_entry_time; L1183-1197 poll Success -> release_kv_cache + set_prefill_kv_transfer_finish_time (stage prefill_transfer_kv_cache). Measured sglang histograms (server_metrics_export.csv, script N/hist.yihou.py, out hist.jsonl): prefill_transfer_kv_cache avg pf90 C48 4.9 s, x8p 2.6, G3 C48 5.1, G2 C56 11.6, G3 C56 9.9, G2 C64 18.6, G3 C64 15.1, G4 C64 15.8; kv_transfer_total_mb avg 7.3-8.1 GB/request (p50 ~5 GB) at every conc; decode kv_transfer_alloc_ms avg 0.3 s (C48) -> 1.0-1.4 (C56) -> 2.6-4.9 s (C64), p50 <1 ms; prefill queue_time avg 4.8 -> 8-16 -> 15-32 s.
- [burstkeeper2] 2026-10-19T04:59:28Z heartbeat: observe-only (submits halted since 03:28Z), no yihou-ofs-burst jobs; awaiting main

## 2026-10-19 05:2x UTC - NEW TASK (user): tune G4 offload (base ofg4b A2 23,771) at C56 then C52: (1) diagnose KV-transfer inflight stage, mooncake logging via env, machine suspicion; (2) larger chunked-prefill-size; (3) other checks. OAI allowed again (user 10-19): holds 224693/224694 -A amd-oai -q amd-oai-qos -p amd-spur PENDING QOSGrpNodeLimit; burstkeeper2 resumed (OAI prep + one v2 fallback; never touch 224677 yihou-dsp-batch = other session). Workspace /shared_nfs/yihou_glm52/g4tune_20261019. Image sglang source = upstream 11b1fb21de (md5 of disaggregation/prefill.py, decode.py identical to src.v0521.yihou); Mooncake faae8dd4 + PR4340 HIP port. Research workflow (code + data) running locally.

- 2026-10-19 05:5xZ: research workflow done (5 readers + adversarial verify + plan + critic; W/research_wf.20261019.json). Verified: full context sent each request (decode_prefix_len=0; --disaggregation-decode-enable-radix-cache off, rejected with EAGLE); prefill sender per DP process = SGLANG_DISAGGREGATION_QUEUE_SIZE (4) FIFO queues x 1 thread, queue = sum(decode session ports) % 4, one blocking batch_transfer_sync per chunk; HIP transport: hipMemcpyAsync per block over 64 streams then serial event sync; Success seen once per prefill loop; decode alloc wait lands in bootstrap stage, not transfer stage; MC_DISABLE_HIP_TRANSPORT and MOONCAKE_DISABLE_HIP_DMABUF in our env are read nowhere (no-ops); MC_TE_FILTERS=yihou_no_such_rdma_device -> tcp+hip installed (code read; runtime route unverified); per-DP chunk = 8192, max_prefill_tokens 16384; G4 copies w13+w2 of 19 layers per forward, constant per forward; exposure unmeasured. Mooncake has no per-transfer timing log (MC_TE_METRIC needs WITH_METRICS build) -> plan: instrumented conn.py copy. ls ~/spur-hold
- [watch136] 2026-10-19T05:29Z no change: 135/136/139 ssh timeout, 137/138 ok; waiter pid 754320 alive, waiting
- [burstkeeper2] 2026-10-19T05:57:44Z heartbeat: observe-only, no yihou-ofs-burst jobs; awaiting main

- 2026-10-19 06:2xZ: found prior R032/R036 transfer work (10-10, node 102, C40; ledger v0521dbg/logs/watchdog2.yihou.md): conn.yihou-xferfix.py (YIHOU_XFER_LOG per-shard per-minute stats, YIHOU_XFER_SHARD dest|room, YIHOU_XFER_PAR), prefill.yihou-nonfinal-event.py. Logs: e2e (dest) each prefill DP used ONE shard, worker busy ~75-100% of span, 0.4-2 GiB/s per shard, queue wait mean up to 7-17 s; e2f (room) still one shard per DP (DPi->shard i: room % 4 == DP rank), 2.3-6.9 GiB/s, wait ~1 s. Scores C40 fresh: e2e 19,565 / e2f 19,864 / e2g A1 19,640 (single runs). New (g4tune): conn.yihou-xferfix2.py adds YIHOU_XFER_SHARD=mix (hashed room) + config.yihou.xferfix2.sh (defaults dest/orig/LOG=1 = upstream behaviour + logging) + mounts.yihou.v2-xferfix2.tsv. Dryrun template W/env.g4xdry-v3006: tokdiff vs ofg4h5 = conn/prefill mounts + 3 YIHOU_ env only. Also W/conn.yihou-xfertime.py (per-chunk timing on regbase) kept unused for now. spur006 hung on a recursive grep I started (stopped); using spur007.
- [watch136] 2026-10-19T06:26Z 135/136 ssh timeout; 139 now "Connection refused" (was timeout); 137/138 ok; waiter pid 754320 alive, waiting
- [watch136] 2026-10-19T06:41Z 136 BACK (also 135, 139): up 14 min (boot ~06:26Z). docker.service failed, containerd activating; /mnt/m2m_nobackup not mounted (8x3.5T nvme present, unmounted); / 9.6G free; RAM 2.7T free; VRAM ~284 MiB/GPU, no KFD pids. Waiter saw 136 at 06:27:14Z, started ofg4h3, STOP at up (etcd FAIL: docker daemon not running) 06:28:17Z, DONE 06:28:26Z. Not relaunched. Waiter log also has 2x "sleep: write error: Disk quota exceeded"; /shared_nfs 99% used (5.5T avail).

- 2026-10-19 06:4xZ (watch136): 136/135/139 answer ssh again. 136 rebooted ~06:26Z; docker.service failed (daemon down), containerd activating; /mnt/m2m_nobackup (docker root) NOT mounted, 8 nvme unmounted; / 93% full; VRAM idle. Waiter auto-started ofg4h3 06:27:14Z -> etcd FAIL -> STOP at up 06:28:17Z; nothing relaunched.
- [burstkeeper2] 2026-10-19T06:56:09Z FINAL: keeper2 loop ended (submits halted since 03:28Z after external cancels; no main reply). no yihou-ofs-burst jobs queued
- [watch136] 2026-10-19T07:35Z all 135-139 ssh ok; 136 up 1:09, docker failed / containerd activating, /mnt/m2m_nobackup unmounted, no KFD pids; waiter finished (DONE 06:28Z), not relaunched

- 2026-10-19 07:5xZ: machine-pool skill now in place (live manager on a16-19). Consumers must request via pool. Cancelled our stuck self-submitted v2 hold 224799 (prolog failure x2) and requested pool r-1019154955-1221 (crusoe burst-p2 x1, 24 h, notify infera-glm52-pd-5f). Kept self-submitted OAI holds 224693/224694 (queue position since 05:04Z), pending user decision.

- 2026-10-19 07:53Z: 136 check (read-only): up 1:27; docker failed, containerd activating; /mnt/m2m_nobackup still not mounted (on /dev/vda1); / now 47% used (was 93% at 06:40); docker unreachable. NOT ready; nothing launched.

- 2026-10-19 08:1xZ: pool r-1019154955-1221 fulfilled 07:52Z: job 225074 on crsuse2-m2m-v3-032 (10.245.150.194), 8 GPUs, 0 containers, kfd 0, VRAM 0%, image absent. prep_node 032 seed started 08:07:17Z (fresh cache key). R1 env W/env.g4xb1-v3032.yihou.sh (xferfix2 log-only dest/orig/LOG=1, G4K1, tok4/detok4, C56 C56); dryrun ok; tokdiff vs g4xdry-v3006 template empty (node/IP only).

- 2026-10-19 08:11Z: v3-032 (225074) CANCELLED after exactly 20:00 (burst preemption) during prep (started 08:07:17); R1 not launched. Cache key 032 may be partially seeded - verify .so counts / locks before reuse. Pool resubmitted 225102 (PENDING).

- 2026-10-19 08:20Z: pool replacement job 225102 on crsuse2-m2m-v3-021 (10.245.145.196) since 08:15Z; foreign container ixk3-attempt5 present (kfd 0, left running). prep_node 021 seed 08:19:30 (fresh key). R1 env W/env.g4xb1-v3021 (tokdiff vs v3032 empty). Armed W/launch_after_prep.yihou.sh (waits for prep, checks img 9d755dd6 / .so 16/14 / locks 0 / kfd 0, then chain_cap2 56 56 + pyspy; log W/launch.g4xb1.yihou.log).
- [watch136] 2026-10-19T08:30Z no change: 136 up 2:03, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok

- 2026-10-19 08:5xZ: R1 g4xb1 launched 08:26:33 on v3-021 (job 225102) but job preempted 08:35 (20:00); prefill container got SIGTERM (exit 143 at 08:35:08) while still in startup (waiting HTTP 331 s). Pool re-granted SAME node v3-021 as job 225141 08:36Z. Stopped our leftover g4xb1-etcd. Burst nodes (20 min) cannot cover bring-up (25-35 min) -> R1 not relaunched. Machine probes on v3-021: rocm-smi --showtopo = full XGMI mesh, weight 15 everywhere (W/machine/topo.v3-021.yihou.txt); torch peer copy (image, W/machine/p2p_bw.v3-021.yihou.txt): 59.7 GB/s for one 4 GiB copy GPU0-3 -> GPU4-7, 12.9-13.5 GB/s for the same 4 GiB as 10000 non_blocking block copies from python.
- [watch136] 2026-10-19T09:25Z no change: 136 up 2:58, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok

- 2026-10-19 09:1x-09:3xZ: Mooncake microbench on v3-021 (job 225141, image, production MC_* env; W/machine/mc_bw*.yihou.py + logs). 360 KiB blocks x5825 = 2 GiB per batch_transfer_sync_write: idle GPUs 0.042 s/call (~47-51 GB/s) for 1 pair; 4 pairs concurrent 0.107 s each with 61 registered buffers (no load). 1x2GiB single block ~9 GB/s; 64 KiB blocks ~13.6 GB/s. With a bf16 matmul burn on the SENDER GPU and >=2 pairs: senders on GPU1/2/3 drop to 5.7-6.0 s/call (0.36-0.4 GB/s, ~50x slower), GPU0 stays 0.11-0.13 s (reproduced 4x, also with nbuf=1). Burn on RECEIVER GPUs: no slowdown. MC_HIP_STREAM_PRIORITY=high and HSA_ENABLE_SDMA=1: no change. HSA_ENABLE_SDMA=0 run lost (controller errors / node). Mechanism not established.
  09:4xZ: HSA_ENABLE_SDMA=0 under sender load: senders did not finish within 600 s (queue.Empty), container stopped. Default/SDMA=1 runs above completed.

- 2026-10-19 09:46Z: v3-021 job 225141 alive 1h11m -> launched R1 again as g4xb2 (W/env.g4xb2-v3021, JOB=225141; tokdiff vs g4xb1-v3021 empty; node check img 9d755dd6, .so 16/14, locks 0, kfd 0) chain_cap2 56 56 + pyspy at 09:46:41Z.
- [watch136] 2026-10-19T10:20Z no change: 136 up 3:53, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok

- 2026-10-19 10:5xZ: g4xb2 LOST: job 225141 preempted ~10:34Z (ran 1h58m) during A1 warmup (bench start 10:21:58; logs stop 10:33:46). Prefill/decode/router containers were still Up afterwards (stopped by main). YIHOU-XFER at 10:33 (C56 warmup): every prefill DP used ONE shard; per shard calls~40/min, blocks/call 76-78 (avg block 4.6-5.4 MiB), wall 53-55 s of 60 s busy, 0.25-0.30 GiB/s per shard, queue wait mean 1.2-1.3 s. Pool re-granted v3-021 as job 225205 (~10:35Z). Relaunched R1 as g4xb3 (tokdiff vs g4xb2 empty).
- [watch136] 2026-10-19T11:15Z no change: 136 up 4:48, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok

- 2026-10-19 11:3xZ: g4xb3 LOST: job 225205 ended 11:23Z (~48 min) 7 min into A1 (bench start 11:16:03). Pool re-granted job 225232 on crsuse2-m2m-v3-026 (10.245.152.176) 11:30Z: docker.service failed, containerd activating -> unusable; prep 026 failed (no docker; cp into cache/026 errors). Reported bad node to pool manager (machines-and-clusters-b1).

- 2026-10-19 11:5xZ: pool granted job 225242 on crsuse2-m2m-v3-032 (10.245.150.194) 11:51Z (docker active). prep_node 032 seed 11:52:38 (key 032 was interrupted at 08:11; seed cp -an refills). R2 env W/env.g4xm1-v3032 (YIHOU_XFER_SHARD=mix; tokdiff vs g4xb3 = that one env line). Launcher armed (waits for prep; checks img/.so 16/14/aiter lock_* 0/kfd 0).
- [watch136] 2026-10-19T12:10Z no change: 136 up 5:43, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok

- 2026-10-19 12:1xZ: g4xm1 (R2 mix) launched 11:59:28 on v3-032 but job 225242 cancelled 12:09Z (20 min) during bring-up. Pool node now job 225265 on v3-021 (12:11Z). Stopped leftover g4xb3 containers on v3-021 (all were still Up). R2 relaunched as g4xm2 (tokdiff vs g4xm1 empty).
- [watch136] 2026-10-19T13:04Z no change: 136 up 6:38, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok

- 2026-10-19 13:1xZ: g4xm2 (R2 mix) LOST: job 225265 cancelled 13:11Z; A1 bench started 12:47:16, logs stop 13:09:55. YIHOU-XFER per DP (approx per-minute aggregation over shard lines): warm-up 12:50-12:58 used 1-4 shards/DP, 0.17-0.32 GiB/s per DP, busy 64-98%, wait_mean 0.5-1.9 s = same as R1 baseline warm-up (g4xb2 10:29-10:33, g4xb3 11:18-11:21: 1 shard, 0.22-0.33 GiB/s, busy 66-92%, wait 0.5-1.3 s). From 13:02 (probably profiling phase) 4 shards/DP active, 0.05-11 GiB/s per DP per minute, wait_mean 0.1-0.8 s; NO baseline sample exists for that phase (baseline rounds died in warm-up). Pool resubmitted 225312 (PENDING).

- 2026-10-19 13:2xZ: pool job 225312 on v3-032 since 13:12Z; stopped leftover g4xm1-etcd; R1 baseline relaunched as g4xb4 (SHARD=dest; tokdiff vs g4xm1 = SHARD line only). OAI holds 224693/224694 now PENDING with prolog failure (were QOSGrpNodeLimit).
- [watch136] 2026-10-19T13:59Z no change: 136 up 7:32, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok

- 2026-10-19 14:1xZ: user: OAI available, request via pool and start. Cancelled self-submitted OAI holds 224693/224694 (prolog failure, Restarts=1). Pool request r-1019221412-41b1 (crusoe oai x2, 24h) -> jobs 225333/225334 PENDING QOSGrpNodeLimit; pool standing target t-1019210042-7f5d already holds OAI m2m-250 (job 225298, since 13:42Z) + 225299 pending. Asked pool manager to hand target nodes to g4tune.

- 2026-10-19 14:2xZ: pool lent OAI target node crsuse2-m2m-250 (job 225298, since 13:42Z, not preemptible, MaxWall 24 h -> ~13:42Z 10-20). Stopped our 31-h-old ofg3h13-etcd there. Cache 250 .so 16/14 locks 0, image present. R1 full baseline g4xb5 (env W/env.g4xb5-250, config mc4340n16; tokdiff vs g4xb4 empty) launched chain_cap2 56 56. Pool released r-1019221412-41b1; second OAI node 225299 pending.

- 2026-10-19 14:2xZ: phase-aligned transfer comparison (W/xfer_sum.yihou.py, 5-min windows, minutes since A1 bench start). R1 g4xb4 (dest, v3-032, start 13:48:57): min 16-21 per-DP 0.8-4.1 GiB/s, busy 70-96%, queue wait mean 5.3-11.9 s; min 21-26 wait 4.2-18.6 s. R2 g4xm2 (mix, v3-021, start 12:47:16): min 18-23 4 shards/DP, 1.4-5.6 GiB/s, busy 29-81%, wait 0.25-0.52 s. Early warm-up (min 3-13) both ~0.13-0.30 GiB/s/DP, wait 0.8-1.7 s. Single sample each, different nodes.
- [watch136] 2026-10-19T14:54Z no change: 136 up 8:27, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok

- 2026-10-19 15:1xZ: g4xb4 (R1 burst v3-032) LOST: job 225312 cancelled 14:55Z (~66 min into A1; no JSON). Pool node now job 225364 on v3-017 (10.245.149.32, fresh key 017): prep 15:12:36; R2 g4xm3 (mix) armed (tokdiff vs g4xm2 empty). OAI R1 g4xb5 on m2m-250 continues.
- [watch136] 2026-10-19T15:49Z no change: 136 up 9:22, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok
- [watch136] 2026-10-19T16:44Z no change: 136 up 10:17, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok
- 2026-10-19 16:5xZ: g4xb5 (R1 dest, OAI m2m-250) A1 done 16:28Z: score 20606 qps 1.212 ISL 135094 TTFT 17.95/57.35 TPOT 17.0 gpu 0.903 cpu 0.038 err 2; prefill transfer_kv p50/p90/avg 5.35/18.99/8.04 s, kv_transfer_latency p50 5.37 s, prefill queue avg 9.43 s; decode_transferred p50/p90 12.0/44.6 s. A2 started 16:28:27Z. g4xm3 (R2 mix, v3-017) A1 running since 16:11:57Z. 0 error hits both. 225299 still PD QOSGrpNodeLimit.
- 2026-10-19 17:19Z: no change: g4xb5 A2 running (since 16:28Z), g4xm3 A1 running (since 16:11Z), logs live, 0 error hits; 225299 PD QOSGrpNodeLimit.
- [watch136] 2026-10-19T17:38Z no change: 136 up 11:12, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok
- 2026-10-19 17:49Z: g4xm3 (R2 mix, burst v3-017) A1 done 17:42Z: score 21769 qps 1.314 ISL 131591 TTFT 14.16/52.28 TPOT 17.9 gpu 0.893 cpu 0.044 err 2; prefill transfer_kv p50/p90/avg 1.87/4.03/2.14 s (g4xb5 A1 dest: 5.35/18.99/8.04), kv_transfer_latency p50 1.87 s, prefill queue avg 10.66 s p90 30.6; prefill_forward p50/avg 2.86/4.52 (dest 2.40/3.98); decode_transferred p50/p90 7.26/41.9 s. A2 started 17:42Z. g4xb5 A2 running. 0 error hits.
- 2026-10-19 18:2xZ: g4xb5 (R1 dest, OAI m2m-250) A2 done 17:54Z (SCORED): score 21262 qps 1.223 ISL 138105 TTFT 20.59/61.90 TPOT 13.3 gpu 0.851 cpu 0.073 err 2; prefill transfer_kv p50/p90/avg 7.31/21.76/9.98 s, kv_transfer_latency p50 8.35 s, speed p50 0.86 GB/s, total_mb avg 7730; prefill_forward p50/avg 2.58/4.18; prefill queue avg 9.64 p90 26.4; decode_transferred p50/p90 15.2/51.3; decode alloc p90 574 ms. docker stop of 4 g4xb5 containers on 250. R2-full g4xm4 (SHARD=mix, same node 250, env from g4xb5 with only SHARD+ROUND changed; tokdiff vs g4xb5 = only YIHOU_XFER_SHARD dest->mix, vs g4xm3 empty) launched 18:21:18Z. g4xm3 (v3-017) A2 continues.
- [watch136] 2026-10-19T18:33Z no change: 136 up 12:07, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok
- 2026-10-19 18:5xZ: OAI job 225299 RUNNING on crsuse2-m2m-097 (10.245.144.36; pool target t-1019210042-7f5d 2nd node). Clean (0 containers, VRAM 0.3 GB, docker active, no image). prep_node 097 seed started 18:49:50Z. New harness files: topology.yihou.097.tsv, config.yihou.mc4340n17.sh (n16 + m2m-097). Code read (image): DP attention divides chunked_prefill_size by attn_dp_size (parallel_hook.py:310); PrefillAdder budget rem_input_tokens = max_prefill_tokens=16384 (log). R3 g4xc1 = PREFILL_CHUNK_SIZE 65536 (per-DP 16384 = max_prefill_tokens), else = g4xb5 (dest); tokdiff vs g4xb5 = only prefill 32768->65536. Armed 18:51Z, waits for prep. Caveat: different node from R1 (250).
- 2026-10-19 19:2xZ: g4xm3 (R2 mix, burst v3-017) A2 done 19:08Z (SCORED): score 22495 qps 1.329 ISL 134476 TTFT 15.88/58.43 TPOT 14.4 gpu 0.839 cpu 0.084 err 3; prefill transfer_kv p50/p90/avg 1.96/4.12/2.22 s (g4xb5 A2 dest on 250: 7.31/21.76/9.98), kv_transfer_latency p50 1.96, speed p50 2.94 GB/s; prefill_forward p50/avg 3.20/4.82 (dest 2.58/4.18); prefill queue avg 11.76 p90 32.2 (dest 9.64/26.4); decode_transferred p50/p90 8.31/44.6 (dest 15.2/51.3). Different node from R1; same-node R2 = g4xm4 on 250 (bench start 19:06:35Z). docker stop 4 g4xm3 containers on v3-017; pool release r-1019154955-1221. g4xc1 (chunk 65536, 097) prefill up 19:10Z, decode launching. 0 error hits.
- [watch136] 2026-10-19T19:28Z no change: 136 up 13:02, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok
- 2026-10-19 19:5xZ: /shared_nfs EDQUOT again ~19:35Z (df 417T/420T, 3.7T avail, 100%; writes ok again at 19:50Z). g4xm4 (250): A1 bench FAIL rc=1 19:35:29Z (tee Disk quota exceeded on runner.log/stdout); chain+pipeline gone, server-log copies on NFS stopped 19:35 (docker logs intact); orphan A1 client container still running (ends ~20:16Z, JSON may carry NUL lines like ofg4h4). Armed W/a2_after_orphan.yihou.sh on 250 (waits for A1 client exit, flush both, bench A2 C56; log R/g4xm4.a2.log). g4xc1 (097, chunk 65536): prefill steps of 16384 new tokens/DP took ~3m15s each from 19:26Z (first two 19:23:47/19:23:51 4 s apart; chunk-8192 runs ~1.6 s/step); GPU0-3 100% busy, GPU4-7 0%; py-spy x3 of scheduler_DP0: MainThread in torch synchronize <- process_batch_result_disagg_prefill (prefill.py:962). Warmup returned 9/619 after 1680 s. Cause not measured (kernel-level profile needed). Stack docker-stopped ~19:53Z (prefill needed 2nd try), KFD 0. 097 idle.
- 2026-10-19 20:19Z: g4xm4 orphan A1 client still running (250); a2 waiter armed, nothing yet; 0 error hits (docker logs prefill 30m). 097 idle (awaiting user choice for chunk follow-up). /shared_nfs 3.4T avail.
- [watch136] 2026-10-19T20:23Z no change: 136 up 13:57, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok
- 2026-10-19 20:49Z: g4xm4 orphan A1 client exited 20:32:10Z (aiperf artifacts written, no agentx_conc56.json since the wrapper died); a2 waiter flushed both and started A2 C56 20:32:11Z (expected ~21:58Z). 0 error hits (docker logs 30m). 097 idle.
- [watch136] 2026-10-19T21:18Z no change: 136 up 14:51, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok
- 2026-10-19 21:20Z: no change: g4xm4 A2 running (since 20:32Z), 0 error hits; 097 idle; /shared_nfs 2.9T avail.
- 2026-10-19 21:49Z: no change: g4xm4 A2 running (since 20:32Z), 0 error hits; 097 idle; /shared_nfs 2.8T avail.
- [watch136] 2026-10-19T22:15Z no change: 136 up 15:48, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok
- 2026-10-19 22:2xZ: g4xm4 (R2 mix, SAME node 250 as R1 g4xb5) A2 done 21:57:34Z (SCORED; A2 launched by a2_after_orphan after flush, A1 client ran its full course): score 22854 qps 1.372 ISL 132321 TTFT 14.83/49.37 TPOT 15.0 gpu 0.857 cpu 0.065 err 5. vs g4xb5 A2 dest: 21262 qps 1.223 ISL 138105 TTFT 20.59/61.90 TPOT 13.3. prefill transfer_kv p50/p90/avg 1.89/4.01/2.16 s (dest 7.31/21.76/9.98); kv speed p50 3.09 GB/s (dest 0.86); prefill_forward p50/avg 3.08/4.62 (dest 2.58/4.18); prefill queue avg/p90 10.32/26.4 (dest 9.64/26.4); decode_transferred p50/p90 7.96/37.2 (dest 15.2/51.3); decode alloc p90 1.2 ms (dest 574). Stack left up on 250. 097 idle.
- 2026-10-19 22:49Z: no active round; 250 stack (g4xm4) idle-up, 097 idle; awaiting user decision (mix as baseline? chunk follow-up a/b/c). /shared_nfs 1.6T avail (falling ~0.7T/30min).
- [watch136] 2026-10-19T23:10Z no change: 136 up 16:43, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok
- 2026-10-19 23:19Z: no change; awaiting user decision; /shared_nfs 1.6T avail.
- 2026-10-19T23:49Z: no change; awaiting user decision; /shared_nfs 1.3T avail.
- [watch136] 2026-10-20T00:04Z no change: 136 up 17:38, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok
- 2026-10-20T00:19Z: no change; awaiting user decision; /shared_nfs 3.0T avail.
- 2026-10-20T00:49Z: no change; awaiting user decision; /shared_nfs 2.4T avail.
- [watch136] 2026-10-20T00:59Z no change: 136 up 18:33, docker failed / containerd activating, no KFD pids; all 135-139 ssh ok
- 2026-10-20T01:19Z: no change; awaiting user decision; /shared_nfs 3.2T avail.
