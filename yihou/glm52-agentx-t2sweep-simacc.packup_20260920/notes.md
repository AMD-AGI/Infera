# Notes — gotchas, failure modes, wrong turns

The full chronological narrative is `spec/sweep-driver-log.md`. This file is the
distilled what/why/how/context for the traps worth knowing before you re-run.

## 1. Forced acceptance ⇒ garbled text is EXPECTED (do not "fix" it, do not gate on it)

- **What:** every T2 point emits degenerate text (`1!…`, tokens repeated). Under
  temperature-0 probing the reply is garbage.
- **Why:** `SGLANG_SIMULATE_ACC_LEN=3.61` does not *report* acceptance, it
  *forces* it. `python/sglang/srt/speculative/spec_utils.py:401-434` sets the
  accept index to the first `simulate_acc_len` positions regardless of what the
  target would accept, and with `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`
  writes the draft's UNVERIFIED tokens straight into `predict`. Degenerate output
  is the designed behaviour.
- **How to think about it:** T2 measures **timing**, not correctness. The emitted
  text — and therefore OSL and any token-count-derived figure — describes the
  DRAFT, not the target. Correctness lives in T1 (real acceptance).
- **Context / wrong turn:** mid-task a probe of the (custom-all-reduce-ON) t2e
  stack showed `1! the, answer, answer…`; this was briefly read as an
  invalidation and a healthy 4.6 h run was torn down. That inference was wrong —
  the answer was one grep of `spec_utils.py` away. **Do not run a coherence gate
  as a pre-condition for a simulated point.** The gate
  (`scripts/coherence_gate.yihou.sh`) is kept because it is exactly right for a
  *real*-acceptance run like T1.

## 2. CONC=128 failed twice — two unrelated, non-GPU causes. NOT an OOM.

**Attempt 1 — shared `/home` NFS hit 0 bytes.** `df` showed `10T 10T 0 100%`.
Our workspace is 16 G of a 10 TB volume shared by hundreds of users; we did not
fill it and cannot control it. Every artifact write failed and the client tree
died. Partial preserved in the workspace as `t2f-agentx-c128-nfsfull/`. Resumed
after the volume recovered to ~700 G free (someone else's cleanup).

**Attempt 2 — Mooncake PD KV-transfer wedge on the decode leg.** Evidence:
`results/t2f-c128-hang/` (`HANG-SUMMARY.txt` + gzipped docker-log tails).
- 00:53:59 last normal decode completion; client froze at returned≈346/1422,
  in_flight=149.
- 01:12+ decode logs `Decode handshake failed` / `Decode transfer failed`
  (Mooncake), including for the `/health` probe — so decode `/health` → 000
  timeout while it still answered `GET /metrics` 200 (an HTTP thread alive over a
  wedged scheduler; the classic "corpse serving /metrics").
- GPU VRAM held ~269 G but read/write activity 0; **0 GPU faults on both legs**
  (live `docker logs`); NCCL watchdog silent; `/home` fine at 665 G.
- Later it partially "recovered" to ~1 request/minute against 148 in flight —
  not useful work.

**Neither is a GPU `Memory access fault`,** so neither is the fault the
`index_share` mitigation targets, and no config variable was changed to chase
them. **Do not write CONC=128 up as "OOMed"** — the mission anticipated a memory
failure; this was not one. Decision (owner's call): stop at 4/5 and pack up — the
curve is defined by 40/56/72/96, CONC=96 already shows capacity collapse, and a
third attempt was not worth ~1 h + recurrence risk.

## 3. CONC 56+ is KV-capacity / queueing limited, not compute limited

- **What:** throughput *collapses* as concurrency rises: 158 k → 117 k → 70 k →
  24 k tok/s (CONC=96 is 15% of CONC=40). TTFT mean explodes 7.4 → 418.6 s.
- **Why it's capacity, not compute:** ITL (per-output-token latency) actually
  *falls* monotonically 13.8 → 13.8 → 12.1 → 11.1 ms across 40→96 — the GPU is
  not the bottleneck. Meanwhile server-measured GPU-cache hit collapses
  93.9% → 65.0% → 27.3% → 11.5% (overall 95.2% → 22.3%) and `kv_gpu_usage_pct`
  pins at 1.0 from CONC=56 up. Requests queue and the KV cache thrashes.
- **How to read the numbers:** **CONC=40 is the only usable operating point.**
  56/72/96 are saturation behaviour, not headroom scaling; do not present them as
  a clean scaling curve.
- **Trap in the tail:** CONC=96 TTFT **p50 = 44.67 s is LOWER than CONC=72's
  p50 = 85.21 s** — an artifact of far fewer/shorter requests completing (808
  profiled vs 1996), not an improvement. The honest number is the **mean,
  418.61 s**. Quote the mean.

## 4. The t2e A/B arm is valid data, not garbage

t2e = identical to t2f but with custom all-reduce **ON** (no
`--disable-custom-all-reduce`). Its CONC=40/56 points
(`results/t2e/agentx_conc{40,56}.json`) are valid timing measurements — the other
half of the single-variable A/B. Deltas t2f vs t2e: **+0.65% total tput at
CONC=40, −4.2% at CONC=56.** So on this simulated-timing workload the
custom-all-reduce flag is close to throughput-neutral. (Its real effect is on
correctness under REAL acceptance — T1 — not visible here.)

## 5. The `index_share=false` GPU-fault mitigation (in force for every point)

`--json-model-override-args '{"index_share_for_mtp_iteration":false}'` is on for
all T2 points. It is a **mitigation, correlational**: 0 GPU memory access faults
in ~17 h of load across three runs and two TP shapes, vs 5 faults in 5 runs with
it ON — but no faulting kernel was ever identified, and upstream (#39517, #37648)
is open with hazard files byte-identical to `origin/main`. Do not treat "0 faults
here" as a proof of fix. Full analysis: `spec/dsa.topk.indexer.bug.analysis.md`.
(A tempting alternative, `SGLANG_DSA_FUSE_TOPK=0`, is **untested** and was briefly
mis-attributed as a garble cause then cleared — see the driver-log ladder history.)

## 6. Monitoring blind spot — scan `docker logs`, not the on-disk follower

- **What:** `server-logs/decode-0.log` / `prefill-0.log` stopped being written at
  23:42 when the log follower died with the process tree in the NFS-full event.
  Their mtime froze. A watcher grepping those files reported `faults=0` from a
  DEAD file for ~90 min.
- **Why it matters:** it *looks* like healthy monitoring; it is monitoring a
  corpse. The true fault count was re-confirmed 0 only by scanning
  `docker logs <container>` directly on both legs.
- **How to avoid:** scan `docker logs <container> 2>&1 | grep -icE 'memory access
  fault|Fatal Python error|Watchdog caught|HSA_STATUS'`, not the follower's
  on-disk file. `scripts/watch.yihou.sh` here still reads the on-disk file — its
  header carries this caveat; prefer a docker-logs scan for any long run.

## 7. Expected-not-a-bug (from the mission landmines)

- **HiCache release lags ~90 s.** After `stop.sh`, prefill GPUs sit at ~266 GB
  before dropping to the ~298 MB baseline. Verify baseline before any relaunch;
  it is not a leak.
- **`Phase warmup progress` parks and stops updating** in the warmup tail; AgentX
  never emits a profiling-phase progress line. Judge liveness by `tot in=/out=`,
  `unique_in_srv`, `tput_in_srv`, not the parked counter.
- **`stop.sh` prints `cleanup completed with errors` on clean teardowns.**
- **Dataset setup takes 4-14 min** before each point's profiling starts.

## 8. `InvalidInferenceResultError` drops — small, pre-existing, not from the workaround

Each point drops a few requests as `records_error_dropped`
(`InvalidInferenceResultError`: "No responses with actual content … only
usage/metadata, null/empty data, or [DONE] markers"). Counts across t2f were
1/2/5/3 (CONC 40/56/72/96) — well under the 10% bar, dropped from profiling not
run failures. This is **pre-existing on this stack**, NOT caused by
`index_share=false`: the 2026-09-18 C40 run with `index_share` ON also dropped 3.
Left open, not concluded.

## 9. Config placement / substitution gotchas

- The `config.yihou.*` files must sit exactly two levels under
  `bench/glm5p2_pd/` — `config.yihou.base.sh` reaches the repo `config.sh` via
  `../..`.
- `config.yihou.full.sh` forces two substitutions because the pinned nightly
  rejects the requested `flydsl`/`aiter` backends: DSA → `tilelang`, top-k →
  `sgl-kernel`. Not a preference.
- `sweep.yihou.dcar.sh` vs `sweep.yihou.sh` differ ONLY by the CONFIG path
  (dcar = custom-all-reduce OFF = t2f; plain = ON = t2e). Do not de-duplicate
  them — the header of the dcar file says why.

## 10. Stale in-config comment about PR #37152 (do not trust it; do not edit it)

`scripts/config.yihou.full.sh:29` carries the comment
*"PREFILL_HICACHE=1, ratio 1.5 — the reason PR #37152 is in the image."* **That
comment is now false for T2.** Line 74 of the same file forces
`IMAGE=infera-sglang:v0519-yihou-0917` (the base, no thin layer), so **PR #37152
is NOT in the image** — HiCache ran on stock kernels (see README "What T2 did and
did NOT measure" and `environment.md`). The comment predates the C40-alignment
image revert and was left in place because the config is a runtime artifact of
exactly what ran and must not be edited after the fact. A cold reader who greps
for `#37152` will hit this line — it is a stale comment, not evidence the patch
was active.
