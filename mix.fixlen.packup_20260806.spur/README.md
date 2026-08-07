# GLM-5.2 MIX — Task 1 fixlen throughput/latency sweep

**Ran:** 2026-08-06
**Author:** yihou (AMD)
**Status:** PASS — a full 12-round InferenceX-aligned fixlen sweep completed against
one frozen MIX (aggregated) server; all features (DSA, MTP, kvd, kv-aware) proven
live before the sweep.

## Goal

Task 1 of the GLM-5.2 agentic-bench-on-AMD mission: measure the throughput and
latency envelope of a **MIX (aggregated, non-PD) SGLang deployment** of
GLM-5.2-MXFP4 on a single 8×MI355X node, across the Case-A-derived shape/concurrency
grid. This establishes the aggregated-serving baseline the mission compares later
deployments against. It is a **characterization sweep, not a pass/fail gate** — the
"success criterion" is a clean, complete grid on a server whose features are all
verified on.

**Spec:** mission `CLAUDE.md` at the repo root + the operator's bench method (see
`REPRODUCE.md` §3 and `notes.md`). The bench harness is InferenceX-aligned
`sglang.bench_serving`.
**Success criteria:**
1. One frozen MIX server serving GLM-5.2-MXFP4 with DSA + MTP + kvd + kv-aware all
   verified live (smoke passes).
2. All 12 rounds (3 shapes × 4 concurrencies) complete and produce raw jsonl +
   a computed summary table.

## Result

Both criteria met. Smoke proved: 1 worker `disagg_mode=mixed`, coherent answer
(DSA ok), MTP accept-len median 3.12, kvd 8 adapters + entries>0, kv-aware tokenizer
loaded. The 12-round grid ran on one frozen server; headline numbers below (full
table in `results/fixlen_summary.md`).

| shape | ISL | OSL | conc | out tok/s | out/GPU | TTFT p50 (ms) | E2E p50 (ms) |
|---|---|---|---|---|---|---|---|
| p50 | 7400  | 320   | 24 | 970.6  | 121.3 | 427  | 6311   |
| p90 | 15500 | 3300  | 24 | 1855.6 | 232.0 | 354  | 39578  |
| p99 | 23500 | 17000 | 24 | 2133.8 | 266.7 | 1421 | 182708 |

Note: the random dataset has **no shared prefix by construction**, so the cache-hit
column is residue, not a workload property (see `notes.md`).

## How to reproduce

See `REPRODUCE.md`. TL;DR: build `infera/engine-sglang:final-pr` on the node from
this branch, `mix_up.sh` to bring up etcd+kvd+worker+router, `mix_smoke.sh` to prove
features, `fixlen_bench.sh` to run the 12-round sweep.

## Folder map
- `REPRODUCE.md` — ordered, copy-pasteable reproduction from a clean node
- `environment.md` — exact node / image / weights / git SHA the numbers came from
- `scripts/` — the 4 scripts that ran (worker, bring-up, smoke, bench), verbatim
- `results/` — `fixlen_summary.md` (computed table) + `fixlen_jsonl/` (12 raw outputs)
- `logs/` — sweep console + worker + router logs
- `notes.md` — bench method, gotchas, feature-proof evidence, wrong turns
