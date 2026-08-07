# GLM-5.2-MXFP4 mix deployment — agentic benchmark at CONCURRENCY 1

**Ran:** 2026-08-06, 09:53:41 → 11:52:03 UTC (three arms, run back-to-back on
one live deployment).
**Node:** `chi2835` — single MI355X (gfx950) node, 8 GPUs, mix (aggregated) worker.
**Status:** **COMPLETE — 3/3 arms, 0 errors, success rate 1.0 on every arm.**

This is **Phase 2 of 3** of the GLM-5.2 mix bench ([`specs/mission.mix.md`](specs/mission.mix.md),
**task 2**). Phase 1 (the fixed-length sweep) is packed separately at
`fixlen.glm52.mix.packup_20260806/`; Phase 3 (agentic under load) is its own packup.

**The deployment is the same live deployment Phase 1 measured** — same container,
created `2026-08-06T07:01:26Z`, never restarted between the phases. Where the
environment is identical this packup restates the essentials and cites
`fixlen.glm52.mix.packup_20260806/environment.md` for the full derivation. This
packup nevertheless stands alone: everything needed to reproduce is inside it.

## Goal

Measure the **single-request latency floor** of GLM-5.2-MXFP4 in a MIX
(aggregated, single-instance) deployment, driven by the *agentic* workload rather
than a synthetic fixed-length one — at **concurrency 1**, at the three Case-A
request shapes (p50 / p90 / p99), with the shared prefix **already warm** so each
request runs at the intended cache-hit rate.

**Spec:** [`specs/mission.mix.md`](specs/mission.mix.md), task 2 — *"测试单conc:
agentic性能 … session = 1, max session = 1, max in flight = 1, turn = 1 … (在刷入
cache 保证 cache hit rate 的情况下, 另外 conc=1 不需要 think time delay)"*, with
**10 repeats** at each of p50 / p90 / p99.

**Success criteria**, as the spec states them — there is no numeric latency bar:

| criterion (from the spec) | actual | verdict |
|---|---|---|
| concurrency held at 1 | `in_flight` ∈ {0,1} on every tick of all three arms; `num_sessions_active` never exceeded 1; the driver never printed its `Hit max_inflight` warning | met |
| turns per session = 1 | `turns_per_session {1,1,1}`, so no think time is reachable | met |
| cache warmed before measuring | ramp is an exclusion window; sustain-phase cache hit 0.8898 / 0.8897 / 0.8900 against an ideal of 0.89 | met |
| ≥ 10 repeats per percentile | **106 / 46 / 28** sustain-phase requests | exceeded |
| runs complete without error | 0 errors, success rate 1.0, on all three arms | met |

## Result — sustain phase only

Produced by [`scripts/analyze_solo.py`](scripts/analyze_solo.py) over the shipped
`results/solo/<arm>/metrics.jsonl`. **Re-derived while assembling this packup and
reproduced to the decimal** — see [`REPRODUCE.md`](REPRODUCE.md) §6.

| arm | n | prompt p50 | gen p50 | cache hit (mean) | TTFT p50 | TTFT p90 | TTFT p99 | E2E p50 | E2E p90 | E2E p99 | TPOT p50 | TPOT p90 | TPOT p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p50 | 106 | 74,013 | 317 | 0.8898 | **1811.7 ms** | 2815.9 | 3413.3 | **5111.1 ms** | 5910.2 | 6798.8 | **10.0 ms** | 10.7 | 11.5 |
| p90 | 46 | 155,013 | 3,297 | 0.8897 | **3674.7 ms** | 4730.9 | 5349.1 | **30781.4 ms** | 33478.7 | 39312.8 | **8.1 ms** | 9.1 | 10.8 |
| p99 | 28 | 235,013 | 16,993 | 0.8900 | **5663.6 ms** | 5854.7 | 7486.4 | **147404.6 ms** | 155157.9 | 166228.8 | **8.3 ms** | 8.8 | 9.4 |

These are `analyze_solo.py`'s percentiles. The driver's own `summary.json` block
reports TTFT up to 2.5 % higher — a **percentile index convention** difference on
the same array (driver uses `sorted[int(q*n)]`, we use `sorted[round(q*(n-1))]`),
established by testing every standard convention rather than assumed. See
[`results/RESULTS.md`](results/RESULTS.md) and [`notes.md`](notes.md) §10. One
source is used for all three metrics on purpose: the driver's block has no E2E
column at all.

Per-run totals, from each arm's own `summary.json`:

| arm | duration | sent | completed | errors | success | qps | cache ideal | cache actual | efficiency | eviction |
|---|---|---|---|---|---|---|---|---|---|---|
| p50 | 785.7 s | 133 | 133 | 0 | 1.0 | 0.1693 | 0.8898 | 0.8831 | 0.9924 | 0.0076 |
| p90 | 1685.6 s | 52 | 52 | 0 | 1.0 | 0.0308 | 0.8899 | 0.8808 | 0.9897 | 0.0103 |
| p99 | 4385.9 s | 30 | **29** | 0 | 0.9667* | 0.0068 | 0.8900 | 0.8795 | 0.9883 | 0.0117 |

\* **The p99 arm's 0.9667 is not a failure.** `errors` is 0 on every tick. The
30th request was sent at elapsed 4302.7 s and was still on the wire when the
4380 s budget expired — the driver computes `success_rate = completed / sent` and
counts a still-running request as not-completed. Traced tick-by-tick from the
shipped `metrics.jsonl`; the reproduction is in [`REPRODUCE.md`](REPRODUCE.md) §7.
It does not touch the reported numbers: that request never entered the
sustain-phase arrays, and the 28 that did all completed.

### The stated SLA line

The workload files carry `sla.e2e_p50_ms: 4500`. **This is documentation only** —
`args.sla_cfg` is assigned at `agent_throughput.py:3351` and never read anywhere
in the driver (verified by grep over the source). Reported here purely as a
measurement against a stated bar:

| arm | E2E p50 measured | stated bar | difference |
|---|---|---|---|
| p50 | 5111.1 ms | 4500 ms | **+611.1 ms over** |

**No explanation is offered for the gap.** Nothing was measured that would
identify one. See [`notes.md`](notes.md) §6 for the two measurements that would
settle it.

### Two observations recorded without explanation

1. **TPOT p50 is *lower* on the longer arms** — 10.0 → 8.1 → 8.3 ms as the prompt
   grows 74K → 155K → 235K and the generation grows 317 → 3,297 → 16,993 tokens.
   No mechanism is claimed. [`notes.md`](notes.md) §5 names what would settle it.
2. **The p50-arm E2E misses the stated 4500 ms bar** (above). Likewise unexplained
   on purpose. [`notes.md`](notes.md) §6.

Both are stated as open questions deliberately. A fluent story here would be
worse than silence: it would discourage running the experiment that settles it.

## Why these three YAMLs look the way they do

Read [`notes.md`](notes.md) §1–§3 before reusing them. In short:

- **The percentile triples are degenerate on purpose** (`{p50: X, p90: X, p99: X}`).
  `PercentileSampler` interpolates ln(value) against normal quantiles, so an equal
  triple has zero slope everywhere and `vmin == vmax == X`. Verified first-hand:
  `PercentileSampler(74000,74000,74000).sample_int()` returned 74000 on every
  draw. That is how "measure the p50 shape" becomes an *exact repeatable request*
  rather than a distribution that merely has that median.
- **`max_sessions: 1` is the real concurrency gate**, not `max_inflight`.
- **`ramp_duration` is a warm-up EXCLUSION window, not a load ramp** — it is what
  makes the shared prefix resident. All reported numbers are sustain-phase only.

## Feature evidence — the deployment under test

Same live deployment as Phase 1, verified still serving one mix worker while this
packup was assembled (`/v1/workers` → 1 worker, `disagg_mode: "mixed"`,
`dp_size: 8`, active). MTP acceptance read off the engine log **scoped to each
arm's own time window** (these logs are appended across the whole session — an
unscoped grep mixes in Phase 1 and Phase 3):

| arm | window (UTC) | n | p10 | median | p90 | at 4.00 |
|---|---|---|---|---|---|---|
| p50 | 09:53:41–10:07:30 | 393 | 2.42 | **2.73** | 3.05 | 0 (0.0 %) |
| p90 | 10:07:44–10:38:40 | 1,266 | 2.62 | **3.55** | 4.00 | 147 (11.6 %) |
| p99 | 10:38:57–11:52:00 | 3,475 | 3.10 | **3.70** | 4.00 | 785 (22.6 %) |

A median **at** 4.00 would be a failure signal (a repetition loop the draft model
predicts perfectly). All three medians sit below it.

## How to reproduce

See [`REPRODUCE.md`](REPRODUCE.md). TL;DR: bring the mix deployment up (390 s cold
start), then `WORKLOAD=specs/mix_solo_p50.yaml TAG=solo_p50 bash run_agentic.sh`
for each arm, then `analyze_solo.py`. Wall clock ~2 h for all three, of which the
p99 arm alone is 73 min. **Offline re-analysis of the shipped results needs no
cluster access at all** — §6.

## Folder map

| path | what |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable reproduction, plus offline re-analysis |
| `environment.md` | exact HW/SW — pinned image digest, driver, resolved server args |
| `notes.md` | why the YAMLs look like that, the SOLO_M1 gap, gotchas, open questions |
| `specs/mission.mix.md` | the originating task spec — **task 2** is this packup |
| `specs/mix_solo_{p50,p90,p99}.yaml` | the three workloads, md5-verified against the cluster copies |
| `patches/` | the SOLO_M1 driver patch + its what/why/how/context note |
| `scripts/` | every script that ran, verbatim — see `scripts/README.md` |
| `results/RESULTS.md` | the full table + how to re-derive every number |
| `results/summary.csv` | machine-readable, one row per arm |
| `results/solo/<arm>/` | `metadata.json`, `metrics.jsonl`, `summary.json` per arm |
| `logs/` | the three driver console logs (gzipped) |
| `env/` | on-node environment snapshot + the engine's own resolved `server_args` |
