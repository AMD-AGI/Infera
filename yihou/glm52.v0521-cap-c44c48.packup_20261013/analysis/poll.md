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
