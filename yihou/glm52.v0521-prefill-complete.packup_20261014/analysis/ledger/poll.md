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
