# GLM-5.2 v0.5.21 PD decode: cap the KV held by in-flight (prealloc → transfer) requests so running requests keep ≥ 50 % of the pool

**Ran:** 2026-10-12 08:40 → 17:15 UTC (research + 2 patch versions + 4 rounds). **Node of record:** `crsuse2-m2m-102`
(spur oai job 207495). **Author:** yihou (with Claude). **Status: INCOMPLETE** — both patch versions work
functionally; v1 regressed C40; **v2's scored no-regression run (warm A2) was never run** (driver host lost 17:18,
node job ended 20:01).

## Goal (user, 2026-10-12, `spec/mission.md`)

Configure the decode leg so running requests always keep at least half of the decode KV pool — the user's rule:
**admit a prealloc request only if `(requested + KV already held) − KV held by running requests < 50 % of the pool`**,
i.e. a slow prefill must not let transfer-queue requests hog decode's pre-allocated pool. Debug until AgentX C40 runs
with **no regression** vs the baseline (e2g A2, C40 warm, 22,114 tok/s/chip), then apply to the best capacity config.

## Result

| criterion | v1 (`decode.yihou-runshare.py`, FIFO break) | v2 (`decode.yihou-runshare2.py`, skip + 5 s anti-starvation) |
|---|---|---|
| cap enforced under a 64 × ~231k-token burst | ✅ decode "pre-allocated usage" max **0.50**, 64/64 ok, 0 faults | ✅ max **0.42**, 64/64 ok, 0 faults |
| running-KV accounting live | ✅ block logs show running > 0 in profiling | ✅ |
| AgentX C40 fresh (A1) | 20,361 (fresh band on 102: 19,335-20,371) | 20,680 |
| AgentX C40 warm (A2, scored) vs 22,114 | ❌ **21,271 (−3.8 %)**, TTFT 6.17 vs 5.30 s | **not run** |

Conclusions so far:
1. No sglang v0.5.21 flag expresses the rule (R1): `num_reserved_decode_tokens` (512/request), `extra_slots`,
   `max_running_requests` are per-request reserves or count caps; host receive is unusable for MLA/DSA;
   `--disaggregation-decode-allocation-policy prefill_complete` shortens in-flight hold time but needs prefill HiCache
   `write_back` (untested). Hence a runtime-mount patch.
2. Under the default `early` policy prefill does not start a request until decode allocated its KV, so a cap delays
   prefill too. v1's FIFO `break` lets one ~0.7 M-token prompt block every request behind it → TTFT +0.9 s → −1.8 % qps
   → regression (inferred from logs: 71-83k blocked checks per DP in the profiling hour, blocks in 4-10 of 60 minutes).
3. At C40 the cap rarely matters for capacity: without it, in-flight KV exceeded 50 % of a DP pool only ~4-5 % of the
   time at C44 (X1) and never caused retraction. It is insurance for C48+ (decode usage reached 0.98-0.99 there,
   still 0 retractions; see the capacity pack-up).

## Folder map

`spec/` mission + PLAN · `analysis/` R1 (admission code model), R2 (occupancy from logs), R5 (upstream) + adversarial
verifications, `working_process.trackB.md` (TB1-TB7, pc40c), full ledger · `patches/` v1/v2 full files + diffs vs
image, configs, mounts · `scripts/harness/` round env/config/chain files + harness core · `scripts/tools/` metrics
sampler, long-prompt burst, residence · `results/` per-run JSON/benchmark logs, round files, smoke logs, /metrics TSVs,
`runs_summary.csv`, reference e2g A2 JSON · `logs/` gzipped server logs (< 4 MB each).

Related: capacity/concurrency pack-up (same day) `../glm52.v0521-cap-c44c48.packup_20261013/`; baseline kit
`/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/`.
