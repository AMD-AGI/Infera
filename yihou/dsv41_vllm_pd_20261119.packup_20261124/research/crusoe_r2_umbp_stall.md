# crusoe v3-200: node-wide stall while R2 (UMBP) runs next to other servers (2026-11-21)

Observations only (first-hand, job 302960). Cause not established.

| time (UTC) | event |
|---|---|
| 14:47 | first stall: all servers on the node "permanently unhealthy" / shm_broadcast; R2 e2-r2cru-11211415 had started with other servers; R2 log ends with gloo "Connection closed by peer". |
| 18:13 | R2 e2-r2cru-11211813 started on GPU4,5 (r2-dp2-umbp-c384, /dev/shm 1.4T free); ready 541 s; client 18:22. |
| 18:24 | R1 replicate e3-r1cru3-c256 started on GPU0,1 (graph capture). |
| ~18:30-18:34 | rocm-smi GPU0-5 0% (GPU6,7 = c288 kept 100%). c320 (GPU2,3) and R2 engine cores log "No available shared memory broadcast block found in 60 seconds"; R2 workers: "UMBP standalone transfers are refused while 1 timed-out transfer(s) are still blocked in MORI"; c256 FULL graph capture slowed to 45.7 s/it. free -g: 22 G free, 836 G shared. |
| 18:34 | R2 stopped (stop.sh). |
| 18:35 | GPU2,3 back to 100%; c256 capture back to ~1 it/s. |

Same-time correlation in 2 of 2 R2 starts; GPU6,7 (c288) unaffected in the second. Not shown: which resource is shared (host
memory pressure, MORI/RDMA, shm). Discriminating test (not run): R2 alone on the node vs R2 + one R1, with host memory and
MORI counters sampled. Decision: no R2/UMBP on this shared node for now; c320 (e3-r1cru-c320-11211705) score is contaminated.

## Update 2026-11-22 02:10Z (v3-014, job 303579): same stall WITHOUT R2
4 servers (R1 / e4b / e4c, no UMBP) started ~2.5 min apart from 01:47Z. During graph capture each logged 60 s shm_broadcast
stalls; e4c c224 failed warmup at 02:05:02 (EngineCore_DP1 "RPC call to sample_tokens timed out", all 841 warmup requests
errored); by 02:08Z all 8 GPUs 0% while two clients had 115/215 requests in flight and "returned" frozen. Load average 28.
So R2/UMBP is not necessary for the stall; the 18:30Z R2 correlation remains an observation, not a cause. Common factor in
all three stalls: several vLLM servers starting / capturing concurrently on one crusoe node. Not tested: one server alone;
sequential starts (being tried now).
- 02:12-02:43Z v3-014: sequential start (next server only after previous "ready") -> all 4 ready in 436-466 s each, no stall
  during startup. One run only; stall-free warmup/measurement still to be confirmed.
- 06:30Z v3-014: a SINGLE server (R1 c288, GPU4,5) hit the same signature mid-measurement (shm_broadcast 60 s at 06:29:56 ->
  EngineCore_DP0 sample_tokens RPC timeout 06:30:54, 816 errors) with no server starting nearby (last start 05:45); the other 3
  servers on the node kept 100% and 0 errors. So the stall is not only a concurrent-start effect; per-server, timing unknown.
- 07:30:58-59Z v3-014: TWO servers (R1 c384 GPU0,1; e4b c256 GPU2,3) hit the sample_tokens RPC timeout within 1 s of each other;
  the other two servers kept running. Observation (timing only, cause unknown): failures at 06:30:54 and 07:30:58 on v3-014, and the
  v3-200 stall began ~18:30Z — all near hh:30. Not shown: anything periodic on the node at hh:30. Discriminating check: host
  cron/systemd timers and pool/daemon probes scheduled at :30, and whether a stall recurs ~08:30Z.
  Host timers on v3-014 (read 07:35Z, systemctl list-timers): cve-2026-98432-metrics.timer every ~1 min; sysstat-collect every 10 min
  (:00,:10,...,:30 incl.); sync-groups ~every 30 min (07:14, 07:44); logrotate hourly at :00; cron.hourly at :17. Nothing found that
  runs ONLY at :30. Not checked yet: what cve-2026-98432-metrics.service does; pool-daemon probes from outside the node.
- 08:16-08:18Z v3-014: two servers (e4c c256 GPU0,1; R1 c224 GPU2,3) stalled together again (GPU 0%, shm_broadcast 60 s at 08:17:35 /
  08:17:45), GPU4-7 unaffected. This one is NOT near hh:30 (weakens that hypothesis). The latest server start (R1 c256, GPU4,5) was at
  08:08 with readiness ~08:14, i.e. its graph capture ended just before the stall began. Pattern so far: stalls hit servers on
  GPU0-3 together (07:30, 08:16), GPU4,5 alone (06:30). Not known: cause.
  08:19Z: that stall recovered on its own after ~2-3 min (no RPC timeout this time); stalls can be transient or fatal.
- 08:37Z: no stall around 08:30Z (last shm_broadcast 08:18:35) -> the hh:30 timing pattern did not repeat.
- 09:00:02-03Z v3-014: e4c c256 (GPU0,1) and R1 c224 (GPU2,3) stalled together again and then failed (timeouts, contract_ok=False).
  Tally on v3-014 by GPU pair: GPU0,1 + GPU2,3 together at 07:30 (fail), 08:16 (recovered), 09:00 (fail); GPU4,5 alone at 06:30
  (fail); GPU6,7 none so far. Observation only — a placement effect (GPU0-3 pairs) is suggested but not shown (only 4 events, and the
  failed runs' concurrency/config differ). Discriminating test: swap which config runs on GPU0-3 vs GPU4-7, or leave GPU0-3 idle
  for one cycle (not allowed on crusoe) / run a light load there.
- 09:48-09:50Z v3-014: GPU0,1 (e4b c256) stalled and failed (DP1 sample_tokens timeout 09:50:17); GPU2,3 (R1 c288) logged
  shm_broadcast at 09:49:15 but recovered. Topology (rocm-smi --showhw): GPU0-3 = PCIe domain 0002 (00:01-04), GPU4-7 = domain 0003.
  Tally of v3-014 stall events by domain: 0002 (GPU0-3) at 07:30, 08:16, 09:00, 09:49 (each time both pairs); 0003 (GPU4,5) once at
  06:30. Observation only; a per-domain effect is suggested, not shown.
- 10:04-10:09Z+ v3-014: GPU0-3 (domain 0002) stalled again (both servers, shm_broadcast from 10:05:25); GPU4-7 kept 100%. 6th event;
  5 of 6 on domain 0002. `journalctl -k` since 09:40 returned no amdgpu/xgmi/AER/reset lines (empty output; readability not verified).

## 2026-11-22 17:44Z — job 304125 (v3-014 again), GPU4,5, e4b-cru9-c224-11221729 (no R2/UMBP)
- Client start 17:37:39Z; "No available shared memory broadcast block" at 17:43:06 (DP0) and 17:44:06 (DP0, DP1); at 17:44:32
  EngineCore_DP1 fatal: shm_broadcast dequeue TimeoutError (scheduler stats: running 2, waiting 139). Client got truncated
  responses, aiperf exited 1, CHAIN_DONE rc=1 at 17:44:40. Afterwards GPU4 0% / GPU5 100% until stopped at 17:55.
- Same pair (GPU4,5, PCIe domain 0003) ran e4c c320 = 139,858 (anomaly, half rate) at 14:09-15:45 and R1 c256 = 348,490 (normal)
  at 15:53-17:16. Other pairs at the same time: normal.
- Action: stopped; relaunched as e4b-cru9r-c224-11221755 at 17:55Z on the same pair; chain14 re-armed as chain14b (pid 36226).
- Cause unknown. This is the first stall on 304125 and the first v3-014 stall on GPU4-7 since 06:30Z (GPU4,5).

## 2026-11-22 19:02Z — job 304125 (v3-014), GPU0-3 (PCIe domain 0002), two servers at once (no R2/UMBP)
- e4c-cru9-c256-11221848 (GPU0,1; client start 18:55:45): DP0 "No available shared memory broadcast" 18:59:39, 19:00:39,
  19:01:39, 19:02:39; EngineCore_DP0 fatal at 19:02:39 (shm_broadcast TimeoutError). 0/285 requests ok (still in warm-up). rc=1.
- e4b-cru9-c384-11221721 (GPU2,3; client start 17:29:24, in its measurement window): EngineCore_DP0 fatal 19:02:28,
  "RPC call to sample_tokens timed out", RustFrontend died; client error rate 306/2782 = 11.0% > 10% -> rc=1, run void.
- GPU4-7 (e4b c224 relaunch, R1 c288) unaffected (0 TimeoutError in their logs).
- Watchdog at 19:04 saw GPU1 0% and GPU2 0% (the other GPU of each pair at 100%).
- Action 19:06: chain14b killed; both tags stopped; relaunched sequentially (launch/seqlaunch-1906.log): e4c-cru9r-c256-11221905
  (GPU0,1), then e4b-cru9r-c384 (GPU2,3) after the first is ready; chain14c (pid 43278) keeps GPU4-7 going.
- Tally on v3-014 today: GPU0-3 stalls 07:30, 09:00, 09:49, 10:04, 19:02; GPU4-7 stalls 06:30, 17:44. Cause unknown.

## 2026-11-22 19:52Z — job 304125 (v3-014), GPU4-7 (PCIe domain 0003), two servers at once (no R2/UMBP)
- e4c-cru9-c192-11221938 (GPU4,5; client 19:45:30, warm-up): "No available shared memory broadcast" 19:50:30 (DP0), 19:51:30
  (DP0, DP1); EngineCore DP0+DP1 fatal 19:52:28. 0/192 ok. rc=1.
- e4b-cru9-c192-11221930 (GPU6,7; client 19:38:26, warm-up): broadcast 19:50:28 (DP1), 19:51:28 (DP0, DP1); EngineCore DP0 fatal
  19:52:27 (scheduler: running 2, waiting 9). 0/498 ok. rc=1.
- GPU0-3 (relaunched e4c c256, e4b c384) unaffected and at 100%.
- Action 19:57: chain15 killed; both stopped; relaunched sequentially (launch/seqlaunch-1958.log): e4c-cru9r-c192-11221957 (GPU4,5)
  then e4b-cru9r-c192 (GPU6,7); chain16 (pid 54245) for GPU0,1 -> R1 c192, GPU2,3 -> e4c c384.
- v3-014 today: 17:44 (GPU4,5), 19:02 (GPU0-3, both pairs at once), 19:52 (GPU4-7, both pairs at once). Each episode hits all
  servers in one PCIe domain at the same second while the other domain keeps running. Cause unknown.

## 2026-11-22 20:15Z — job 304125 (v3-014), GPU4,5 again
- e4c-cru9r-c192-11221957 (relaunch after 19:52; client 20:04:54, warm-up): EngineCore_DP0 fatal 20:15:15
  (sample_tokens/shm TimeoutError), RustFrontend errors; 0/502 ok, rc=1.
- GPU4,5 failures today: 17:44 (stall), 19:52 (stall, with GPU6,7), 20:15 (stall); plus the 139,858 half-rate run 14:09-15:45.
  One normal run on the pair in between (R1 c256 348,490, 15:53-17:16).
- Action: stopped; GPU4,5 left idle (no relaunch) pending the user's node decision. GPU6,7 e4b-cru9r-c192 still loading.

## 2026-11-22 23:50Z — job 304125 (v3-014), GPU6,7
- e3-r1cru11-c256-11222320 (R1, no connector; client 23:28:15, warm-up): EngineCore_DP1 fatal 23:50:05; 0/2437 ok, 377 errors,
  rc=1. GPU6 100% / GPU7 0% afterwards. Stopped 23:55. Not relaunched (would end ~01:25, at the job end; pair unreliable).
- Note: R1 (no KV connector) fails the same way, so the connector is not required for these stalls.
- v3-014 episodes this evening: 17:44 (GPU4,5), 19:02 (GPU0-3), 19:52 (GPU4-7), 20:15 (GPU4,5), 23:50 (GPU6,7).

## 2026-11-23 02:22Z — job 305057 (v3-014, new job), GPU4-7 (PCIe domain 0003), both servers at once
- e4c-cru12-c288-11230153 (GPU4,5; client 02:01:53, warm-up 316/3196) and e3-r1cru12-c384-11230201 (GPU6,7, R1 no connector;
  client 02:09:48, warm-up 241/4249): "No available shared memory broadcast" from 02:23:11-12 on both DP ranks of both
  servers; GPU4-7 0% at 02:24; both CHAIN_DONE rc=1 by 02:27. GPU0-3 (R1 c256, e4b c288) unaffected at 100%.
- Same node, new slurm job (304125 was released 01:34Z; 305057 started 01:38Z): the stalls persist across jobs on v3-014.
- Action 02:28: chain19 killed; both stopped; relaunched sequentially (launch/seqlaunch-0228.log): e4c-cru12r-c288 (4,5),
  then e3-r1cru12r-c384 (6,7); chain19b for GPU0,1 -> e4b c224, GPU2,3 -> e4c c256.

## 2026-11-23 04:27Z — job 305057 (v3-014), GPU4-7 (PCIe domain 0003) again, both servers
- e3-r1cru13-c288-11230400 (GPU4,5, R1 no connector; client 04:08:21, warm-up frozen at returned 2,360/3,196 with 257 in flight
  from ~04:27) and e4b-cru13-c384-11230417 (GPU6,7; ready ~04:25, warm-up): GPU4-7 0% from 04:27; both EngineCores fatal
  04:32:10; both CHAIN_DONE rc=1. GPU0-3 unaffected (100%).
- Second failure on this job for both GPU4,5 and GPU6,7 (first 02:22). Per rule: both stopped, GPU4-7 LEFT IDLE. Chain20
  killed; chain21 (pid 29441) keeps GPU0-3 going (R1 c224, e4b c256 next).
- v3-014 episodes since 11-22 17:44: 17:44, 19:02, 19:52, 20:15, 23:50, 02:22, 04:27 — 7 episodes in ~11 h, both PCIe domains,
  R1 and connector configs alike, across two slurm jobs.
