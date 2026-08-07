# GLM-5.2-MXFP4 mix deployment — fixed-length (fixlen) sweep

**Ran:** 2026-08-06, 07:13:48 → 09:45:14 UTC (2 h 31 min of measurement; the
deployment came up at 07:08).
**Node:** `chi2835` — single MI355X (gfx950) node, 8 GPUs, mix (aggregated) worker.
**Status:** **COMPLETE — 12/12 points, no failures, no aborted requests.**

This is **Phase 1 of 3** of the GLM-5.2 mix bench (`spec/mission.mix.md`, task 1).
Phases 2 (agentic at conc 1) and 3 (agentic under load) are separate packups.

## Goal

Measure the fixed-length serving performance of **GLM-5.2-MXFP4 in a MIX
(aggregated, single-instance) deployment** on one MI355X node with SGLang, at
three request shapes × four concurrencies, aligned with the InferenceX benchmark
convention. This is the throughput/latency baseline the agentic phases are read
against.

**Spec:** [`spec/mission.mix.md`](spec/mission.mix.md), task 1.
**Success criteria:** the spec sets no numeric bar for task 1 — it asks for the
12 measurements themselves, aligned with InferenceX. The bar is therefore
*completeness and trustworthiness*: all 12 points measured, every request
completed, and every feature (DP-attention, MTP, kv-aware, kvd, prefix cache)
proven on with a signal that would go red if it were silently absent.

## Read this first: what ISL means here

**ISL is the fresh remainder, not the full prompt.** This is the single most
misreadable number in the whole packup.

Case A's real inputs are **74K / 155K / 235K** tokens, served at an **89–90 %
prefix-cache hit rate** — so only ~10 % of each prompt is actually computed. The
fixlen arm therefore sends **that fresh remainder**:

| arm | Case-A input | **ISL sent = computed** | OSL |
|---|---|---|---|
| p50 | 74,000 | **7,400** | 320 |
| p90 | 155,000 | **15,500** | 3,300 |
| p99 | 235,000 | **23,500** | 17,000 |

`sglang.bench_serving --dataset-name random` builds every prompt independently,
so there is **no shared prefix by construction** and the sent length *is* the
computed length. The two line up, which is why this substitution is valid.

**This was an explicit user decision, not an inference.** Do not "correct" it
back to 74K/155K/235K — that would double-count the prefix the real workload
never recomputes.

## Result — all 12 points

`--num-prompts = conc × 10` (the InferenceX convention). Every arm completed
its full request count.

| arm | isl | osl | conc | completed | req/s | out tok/s | ttft_p50_ms | ttft_p99_ms | tpot_p50_ms | e2e_p50_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| p50 | 7400 | 320 | 1 | 10 | 0.26 | 82.56 | 1046.69 | 1066.95 | 9.13 | 3967.01 |
| p50 | 7400 | 320 | 8 | 80 | 1.24 | 395.58 | 2012.35 | 4232.94 | 13.55 | 6270.88 |
| p50 | 7400 | 320 | 16 | 160 | 2.12 | 679.82 | 1977.77 | 3333.10 | 16.56 | 7089.55 |
| p50 | 7400 | 320 | 24 | 240 | 2.33 | 746.75 | 2276.03 | 7225.92 | 21.70 | 10312.64 |
| p90 | 15500 | 3300 | 1 | 10 | 0.03 | 112.18 | 1327.57 | 2954.26 | 8.14 | 29152.55 |
| p90 | 15500 | 3300 | 8 | 80 | 0.18 | 606.55 | 4522.62 | 6885.07 | 11.87 | 42588.29 |
| p90 | 15500 | 3300 | 16 | 160 | 0.31 | 1020.36 | 3208.46 | 7870.40 | 14.03 | 50259.41 |
| p90 | 15500 | 3300 | 24 | 240 | 0.40 | 1331.73 | 2690.71 | 7377.46 | 16.52 | 57663.13 |
| p99 | 23500 | 17000 | 1 | 10 | 0.01 | 135.10 | 1440.31 | 1908.67 | 7.13 | 122380.54 |
| p99 | 23500 | 17000 | 8 | 80 | 0.05 | 850.34 | 5368.10 | 8558.46 | 8.86 | 155092.54 |
| p99 | 23500 | 17000 | 16 | 160 | 0.09 | 1480.51 | 4044.73 | 7839.19 | 10.15 | 176184.90 |
| p99 | 23500 | 17000 | 24 | 240 | 0.11 | 1935.51 | 3522.31 | 9206.01 | 11.57 | 200049.47 |

The full column set — p90/p95 percentiles, ITL, input/total throughput, cache-hit
breakdown, peak observed concurrency — is in
[`results/RESULTS.md`](results/RESULTS.md) and
[`results/summary.csv`](results/summary.csv).

### One observation stated without explanation

**TTFT p50 is non-monotonic in concurrency on the p90 and p99 arms.** It *rises*
from conc 1 → 8, then *falls* at conc 16 and again at 24:

| arm | c1 | c8 | c16 | c24 |
|---|---|---|---|---|
| p90 | 1328 | 4523 | **3208** | **2691** |
| p99 | 1440 | 5368 | **4045** | **3522** |

This is recorded as an observation. **No mechanism is offered here** — none was
measured. What would settle it: per-request TTFT paired with that request's
`cached_tokens` (both are in the raw jsonl, `ttfts[]` and `cached_tokens[]`,
index-aligned), to separate "queued longer" from "prefilled less". The
sweep-level `cache_hit_rate_pct` also moves non-monotonically across the same
points (p90: 52.5 / 20.9 / 39.3 / 43.1 %), so the two are worth correlating
per-request before anything is claimed. A second discriminator would be the
scheduler queue depth from the engine log over each run's window.

## Feature evidence gate — measured BEFORE any benchmark ran

A green `/health` proves the process is alive, not that the features are on.
Each signal below goes **red** if its feature is silently absent. All were taken
at 07:11–07:13, before the first measured request.

| feature | signal | reading |
|---|---|---|
| registry | `/v1/workers` | 1 worker, `disagg_mode: "mixed"`, active |
| DSA correctness | chat completion | "The capital of France is Paris." + coherent reasoning — garbage here means the DSA env block did not take |
| DP-attention | resolved `server_args` + live ranks | `dp_size=8`, `enable_dp_attention=True`, 8 scheduler ranks |
| MTP (EAGLE) | accept-len distribution, n=25 | median **2.80**, p10 2.48, p90 3.08, **0 % at 4.00** |
| kv-aware | router policy + tokenizer | `router_policy: "kv-aware"`, tokenizer loaded |
| kvd | adapters + `statctl` | **8 adapters** (one per DP rank) |
| prefix cache | `cached_tokens` on a repeat | 1st `None`, 2nd **1984 / 2018** |

A median accept-len **at** 4.00 would be a failure signal (a repetition loop the
draft model predicts perfectly), not a win. 2.80 is inside the healthy 2–3 band.

**Every row above is re-derivable from the logs shipped in this packup** — you
do not need cluster access to audit the gate. Verified while assembling:
`logs/router.log.gz` yields `router_policy: "kv-aware"` and one
`kv-aware: loaded tokenizer` line; `logs/glm52_mix_base.log.gz` yields 8
`kvd adapter connected` lines and all 8 `DP0..DP7` scheduler ranks; scoping the
accept-len read to the pre-sweep window 07:11–07:13 reproduces the gate figure
exactly (n=25, median 2.80, 0 % at 4.00). Commands in `REPRODUCE.md` §8 and
`notes.md` §4.

## Post-sweep counters — the features stayed on under the full load

Both re-derivable from `logs/glm52_mix_base.log.gz`; see
[`REPRODUCE.md`](REPRODUCE.md) §6.

- **MTP accept-len over the sweep:** n = **37,630**, p10 2.85, median **3.61**,
  p90 3.98, **9.0 % at 4.00**.
- **kvd:** 70,676 entries / 84.8 GB host / 68.7 GB L3 / gets 64,741 /
  sets 266,289 / **hits 62,697 / misses 2,044** / evictions 151,525.

## How to reproduce

See [`REPRODUCE.md`](REPRODUCE.md). TL;DR: claim a MI355X node, `bash
mix_site.sh up` (390 s cold start), `bash mix_site.sh smoke` and read the gate
table, then `bash mix_bench_fixlen.sh`, then `summarize_fixlen.py`.
Wall clock end-to-end: ~2 h 45 min, of which the p99 arm alone is ~1 h 40 min.

## Folder map

| path | what |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable reproduction |
| `environment.md` | exact HW/SW the numbers came from — pinned digests, drivers, fabric |
| `notes.md` | gotchas, the one real defect found, wrong turns, open questions |
| `spec/mission.mix.md` | the originating task spec — **task 1** is this packup |
| `scripts/` | every script that ran, copied verbatim — see `scripts/README.md` for which ones this packup uses |
| `results/RESULTS.md` | full-width results table + how to re-derive it |
| `results/summary.csv` | machine-readable summary, one row per point |
| `results/fixlen/*.jsonl` | per-run metrics, 12 files (see the note in RESULTS.md on what was trimmed) |
| `logs/` | sweep console log, bring-up log, and the three container logs (gzipped) |
| `env/` | on-node environment snapshot + the engine's own resolved `server_args` |

No `patches/` directory: the one defect found was in **our own harness**, fixed
inline in `scripts/mix_common.sh`, and is written up in `notes.md` §2. No
third-party code was patched for this run.

`scripts/run_agentic.sh` and the four `spec/mix_*.yaml` workload files belong to
**Phases 2 and 3**, not to this sweep. They are carried because they share the
same staging directory; `scripts/README.md` marks them clearly. Ignore them when
reproducing.
