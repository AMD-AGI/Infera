# GLM-5.2 MIX single-concurrency agentic latency (Task 2)

**Ran:** 2026-08-06
**Author:** yihou
**Status:** PASS — all three Case-A shapes measured at conc=1 on one frozen MIX server, 100% cache-hit as designed.

## Goal

Task 2 of the GLM-5.2 agentic-bench-on-AMD mission (`mission.mix.md`, repo root):
measure **single-concurrency (conc=1) agentic latency** on the aggregated (MIX)
SGLang deployment, at each Case-A percentile shape, **with the cacheable prefix
pre-warmed so cache-hit is guaranteed** (the mission's "在刷入 cache 保证 cache hit
rate 的情况下"). conc=1 needs no think-time delay (operator decision). Each shape is
measured over 10 sequential repeats.

**Spec:** `mission.mix.md` → Task breakdown item 2 (P50/P90/P99 ISL/OSL, 10 reps each,
session=1 / max-session=1 / max-in-flight=1 / turn=1, cache pre-warmed, no think time).

**Success criteria:** this is a *characterization* task — the bar is completeness
(all three shapes measured, conc=1, cache-hit honored) with the tuned MIX features
verified live, not a threshold number.

## Result

Full Case-A ISL/OSL shapes (P50/P90/P99), 10 sequential reps each, warmed shared
prefix (100% cache-hit):

| shape | ISL | OSL | cache-hit | TTFT p50 (ms) | TTFT p90 (ms) | E2E p50 (ms) | E2E p90 (ms) | E2E mean (ms) | TPOT p50 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| p50 | 74000 | 320 | 100.0% | 424 | 456 | 2480 | 2602 | 2500 | 6.63 |
| p90 | 155000 | 3300 | 100.0% | 601 | 649 | 20053 | 21873 | 20603 | 5.89 |
| p99 | 235000 | 17000 | 100.0% | 866 | 1026 | 103069 | 106468 | 104473 | 6.01 |

Reading: **E2E scales with OSL** (dominated by decode); **TPOT is steady ~6 ms**
across all shapes; **TTFT stays low** (0.4–0.9 s) because the ~89% prefix is
cache-resident, so prefill only processes the small fresh tail each rep.

## How to reproduce

See `REPRODUCE.md`. TL;DR: bring up the MIX server (`scripts/mix_up.sh`), then run
`scripts/lat_conc1.py` **inside the engine container** against the router `:8100`.

## Folder map
- `REPRODUCE.md` — step-by-step reproduction (bring-up + latency driver)
- `environment.md` — exact HW/SW the numbers came from
- `scripts/` — `lat_conc1.py` (driver) + `lat_report.py` (table) + the MIX bring-up (`mix_up.sh`, `mix_worker.sh`, `mix_smoke.sh`)
- `results/` — `lat_summary.md` (the table), `lat_jsonl/` (raw per-request jsonl + `lat_summary.json`)
- `notes.md` — the cache model, the `ignore_eos`/`min_tokens` forcing, gotchas
