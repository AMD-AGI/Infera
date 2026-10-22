# PLAN v1 (2026-10-12 08:20 UTC) — task v0521cap

Score = AgentX full, warm-stack protocol (A2 = 2nd run after /flush_cache), tok/s/chip. Goal > 23,000 at C44 then C48.
Machines: amd-oai only — 102 (job 207495, to 2026-10-13 05:26) + 2nd node (job 208970, pending QOSGrpNodeLimit).

## Done
- X1 (102, warm e2j stack, C44): 21,773/chip (+0.4 % vs same-stack C40). Decode KV max 0.71-0.87; transfer share max
  0.59-0.71; prealloc queue 0, retract 0 (ledger X1).

## Running
- k1 (102): e2g recipe + prefill HiCache 360 (user), CONC 44, A1 fresh -> A2 warm, /metrics samplers. ETA A2 ~11:30 UTC.
- Research workflow (R1 decode admission code, R2 KV occupancy logs, R3 capacity/scheduling levers, R5 upstream).

## Next (to be fixed after R1-R3 land)
1. Track A on the 2nd oai node as soon as it is allocated: C44/C48 with the larger-capacity candidate from R3
   (e.g. decode mem 0.96-0.97, prefill mem / HiCache, decode cuda-graph bs + max_running sized for per-DP running at C48).
2. Track B: from R1, a config or a YIHOU_* runtime-mount patch capping prealloc/transfer KV to <= 50 % of the decode
   pool (running keeps >= 50 %); verify with /metrics (kv_used − decode_sum_seq_lens, transfer queue, retractions);
   then apply on the best Track A config.
3. C48 rounds once C44 ≥ 22,114.
4. LAST (user): index_share_for_mtp_iteration=true at the best config, warm protocol (watch for decode MAF at C44/C48).

## v2 (08:45 UTC) — after research + adversarial verify (research/*.verify.yihou.md)
Verified corrections that change the plan:
- Per-DP running cap = 16 (max_running 64 / 4 DP; graph ladder captured only to 16) is hit at C44 more than R3 said
  (DP3 repeatedly; transferred reqs queued ~20 s behind it) -> **DECODE_MAX_RUNNING=96 (24/DP) is a primary lever**
  for C44/C48, not insurance.
- Score ≈ λ × mean ISL (~99 % input tokens): normalise C44/C48 comparisons by ISL (X1 +0.4 % = λ +3.1 % × ISL −2.5 %).
- xferfix TTFT evidence is weak (e2f not the best TTFT); e1c (HiCache ~154 GB) had the lowest TTFT 5.14 s -> large host
  pool may cost TTFT; k1 (360) discriminates.
- index_share TRUE (e2d) is the only fresh run without the ~10 % fresh penalty -> keep as the user's last item, but it
  may matter more than expected.
- Track B patch gaps: deferred-release KV (failed transfers, 30 s) not counted; a few ~280k-token prompts dominate
  in-flight -> FIFO cap can head-of-line block; must be A/B'd at C48 (cap 0 vs 0.5), measure TTFT.
Next rounds (2nd oai node when granted, else 102 after k1):
- Q1: e2g + DECODE_MAX_RUNNING=96 + PREFILL_MEM_FRACTION=0.88, CONC 44 (warm protocol).
- Then C48 with Q1/k1 winners (+ decode 0.96), then Track B A/B at C48, then index_share TRUE.

## v3 (11:55 UTC)
- k1 (HiCache 360, C44 warm) 21,342 → keep HiCache 450. C44 is TTFT-bound (qps +0.4…2 % over C40, ISL −3.5 %).
- Running: Q1 on 137 (max_running 96 + prefill 0.88, C44; A2 ≈ 14:00). pc40c on 102 (runshare 0.5, C40; A2 ≈ 15:00).
- Next on whichever node frees first: winners of Q1 + runshare (if no regression) at C44, then C48 (+ decode 0.96).

## v4 (12:58 UTC) — armed queue
- 137: q1 A2 (C44, scored, ≈14:05) → `a3_q1c48.yihou.sh`: C48 as 3rd run on the same warm q1 stack (server args identical;
  CONC is bench-only) → runs/q1/agentx-c48-A3 (≈15:30).
- 102: pc40c A2 (C40 + runshare 0.5, scored vs 22,114, ≈15:00) → smoke2 (230k-token burst, cap must fire) →
  `chain_q3.yihou.sh`: Q1 config + runshare 0.5 at C44 (same node as k1/e2g). If pc40c shows a regression, kill q3
  before its pipeline starts and relaunch without the patch.

## v5 (15:40 UTC)
- Best: 22,306 (137, q1 stack 3rd run, C48, max_running 96 + prefill 0.88). C48 decode KV max 0.99 (DP1) → capacity binds.
- 137 queue: A4 C52 on q1 stack (≈16:50) → q4: q1 config + decode 0.96, C48, fresh stack A1 → A2 (≈20:00).
- 102 queue: pc40d = runshare v2 at C40 (regression check vs 22,114; ≈18:40). If v2 passes → apply v2 on the best config
  (C48/C52) next; if not → v3 or reconsider share.
- Later: mc4340prio (copy-stream priority, 102), index_share TRUE (last, user).
