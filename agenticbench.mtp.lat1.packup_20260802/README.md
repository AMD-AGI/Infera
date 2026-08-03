# lat1 — the latency floor of the full-feature GLM-5.2 deployment

**Ran 2026-08-02, 05:39:24–06:12:25 UTC** (1,980.9 s) on the **spur**
(`crsuse2-m2m`) cluster, against the *same live deployment* the Case A run of
2026-08-01 measured — kvaware + kvd + MTP + PD + DP-attention, GLM-5.2-MXFP4.

## Goal

Case A measures this deployment under a realistic agentic population. It cannot
say how fast **one** request is, because at 44 in-flight requests a TTFT of
18.9 s is mostly waiting. This run holds Case A's request distribution fixed and
collapses the population to exactly one session issuing one turn at a time, so
TTFT and TPOT become the **service time itself** — the latency limit of the
service, with throughput contention removed.

## Result at a glance

> **Both columns are the SUSTAIN phase**, i.e. the measured window with the
> warm-up excluded (lat1 n=119 of 124; Case A n=2,582 of 2,811). That is the
> like-for-like comparison. `results/summary.json`'s **top-level** `ttft_ms`
> block is the ALL-request figure (p50 2,053 / p90 4,860) and will not match the
> table below by a percent or two — the sustain numbers are in its `phases[]`
> array. TPOT, success rate and cache are whole-run; the driver emits no
> per-phase split for them.

| | lat1 (N=1) | Case A (N≈44) | ratio |
|---|---|---|---|
| **TTFT p50** | **2,027 ms** | 6,733 ms | **3.32×** |
| **TTFT p90** | **4,920 ms** | 18,877 ms | **3.84×** |
| TTFT p99 | 5.8–9.3 s *(range; n too small for a point)* | 33,097 ms | ~4.6× |
| **TPOT p50** | **10.66 ms** | 17.9 ms | **1.68×** |
| TPOT p90 | 12.23 ms | 24.9 ms | 2.04× |
| success rate | **1.0000** (124/124) | 0.9757 | — |
| cache actual / ideal | 0.8897 / 0.8899 | 0.8882 / 0.8899 | — |
| MTP acceptance (engine) | **2.846** (71.2 % @ 4 draft) | 2.736 | — |
| engine faults, both legs | **0** | 0 | — |

**Three findings worth the run:**

1. **70 % of Case A's median TTFT is queueing, not computation** (4,706 of
   6,733 ms). At p90, 74 %. **40 % of its median TPOT is batching contention.**

2. **With queueing removed, TTFT is a clean linear function of prompt length:**
   `TTFT_ms = -319 + 29.33 × input_ktok`, **R² = 0.956** over 124 points. Under
   load the same fit gives **R² = 0.0016** — a request's own length explains
   essentially *none* of its latency; an 8.7 s queue intercept explains it.
   Cost per token is flat to ~160K input tokens and superlinear beyond:
   **the knee is at roughly 160K–200K.**

3. **The 240 s client timeout is confirmed as a client artifact.** Case A lost 39
   requests to it and its output p99 came in 43 % short of spec. lat1 lost
   **zero** — its longest generation (22,219 tok) finished in 236.9 s, 1.3 %
   under the deadline. At Case A's TPOT the same request would have needed 397 s.

Full working, with confidence intervals and every caveat: **`analysis/lat1_latency_floor.md`**.

## Success criteria, against the spec

The originating spec (`spec/agentic.bench.kv.liying.mtp.md`) sets Case A's
targets. lat1 inherits the request distribution but not the load, so the
*throughput* criteria do not apply. What does:

| criterion | source | measured | verdict |
|---|---|---|---|
| input p50/p90/p99 = 74K/155K/235K | Case A spec | 83.0K / 171.3K / 233.5K | **PASS** (p99 within 1 %; p50/p90 run 10–12 % high, sampling noise at n=124) |
| output p50/p90/p99 = 320/3.3K/17K | Case A spec | 390 / 3,104 / 14,946 | **PASS** (p99 within 12 % — *better* than Case A, which was 43 % short) |
| cache hit 88–90 % | Case A spec | **88.9–89.0 %** p1→max | **PASS** |
| acceptance 56 % @ 5 draft | Case A spec | **71.2 % @ 4 draft** | **EXCEEDS** |
| `ttft_p90_ms ≤ 30,000` | YAML `sla:` | 4,920 | **PASS** (6.1× margin) |
| `success_rate ≥ 0.97` | YAML `sla:` | **1.0000** | **PASS** |
| `e2e_p50_ms ≤ 4,500` | YAML `sla:` | 6,570 (composed) | **MISS, 1.46×** — and see below |
| session count ≡ 1 | this run's own requirement | `in_flight` max 1, `sessions_active` max 1 | **PASS** |

**The E2E miss is the informative one.** With queueing entirely removed and the
fastest decode this stack can do, 4.5 s is *still* not reached. That settles what
the Case A analysis had to leave open: the target is not a load problem, it is
incompatible with a 74K-token p50 prompt — which costs ~2.0 s of prefill before
the first token, after which the p50 request still decodes 390 tokens.

## Navigation

| path | what |
|---|---|
| **`REPRODUCE.md`** | the ordered, copy-pasteable command sequence |
| **`analysis/lat1_latency_floor.md`** | the report: ladders, curves, CIs, cross-checks |
| **`notes/notes.lat1.md`** | config rationale + **two defects found and fixed** |
| `environment.md` | hardware, fabric, image digests, git SHA, secrets (names only) |
| `spec/lat1_full.yaml` | the workload (heavily commented; every knob justified) |
| `spec/caseA_full.REFERENCE.yaml` | Case A's workload — diff the two |
| `scripts/` | every script that ran, verbatim |
| `patches/` | the ROCm hicache patch + `GLM52_P1V3`, both load-bearing |
| `results/` | `summary.json`, `metrics.jsonl.gz`, kvd before/after, ladders |
| `logs/` | driver transcript + both legs' tails, gzipped |

## Read this before trusting any percentile

**n = 124.** At concurrency 1 the request rate *is* the service time (16.0 s per
request), so 33 minutes buys 124 samples and n=1,000 would take 4.4 hours. From
order-statistic CIs: **p50 (±11 %) and p90 (±12 %) are solid**; p75/p95 are weak;
**p99 (±25 %) is effectively the third-largest observation and is reported as a
range, never a point.** This is structural, not an oversight — the measurement
requires concurrency 1 and concurrency 1 caps the sample rate. The mitigation is
the TTFT-vs-length curve, which uses all 124 points at once.

## Two defects were found and fixed before the reported run

Both are in `notes/notes.lat1.md` with full mechanism. Briefly:

1. **Cache contamination across runs (severe).** The first attempt returned cache
   hit **~100 %** against a configured 0.89 — only 14–62 uncached tokens per
   request. Cause: the fresh-content seed is the *run-local* `request_id`
   (`agent_throughput.py:2187`), so two runs sharing `random_seed` replay
   byte-identical prompts and hit the previous run's radix tree. lat1 inherited
   Case A's `1337`. **Not kvd** — `gets_total` never moved. Fixed with a distinct
   seed; the probe carries a third one so it cannot warm the full run.
   **Generalisation: any two runs of the same config against a warm engine will
   silently measure cache hits instead of compute.**

2. **numpy seed range.** `20260802999 > 2**32-1` kills the driver at startup.
   Caught by the probe, which is what the probe is for.

The contaminated run's metrics are preserved as
`results/CONTAMINATED_seed1337_metrics.jsonl.gz` — it is the evidence for defect 1.

## What this run does not establish

- **The cached-vs-uncached token cost split.** They are collinear by construction
  at a fixed 0.89 hit rate (ratio 7.98–8.09 across all 124 requests), so the
  regression cannot separate them. The sweep cross-check *bounds* it — cached
  tokens are **not** free — but no coefficient is claimed.
- **A controlled MTP ablation.** No MTP-off arm was run on this image.
- **Multi-turn behaviour.** `turns_per_session: 1` removes it by design.
- **The knee location precisely.** Stated as ~160K–200K; the 200–240K bin has
  n=4 and the 240K+ bin n=1.
