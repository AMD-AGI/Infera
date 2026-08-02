# Solo latency floor — Case A request shape at concurrency exactly 1

**Ran:** 2026-08-01, 16:05–16:42 UTC (30-minute measured window).
**Status:** **PASS.** 145 requests, 0 errors, concurrency verifiably pinned at 1.

Companion to `../caseA.glm52.fullfeature.packup_20260801/`. **Same deployment,
same image, same processes, same hour** — nothing was restarted between the two
runs (proved in `environment.md`), which is what makes the comparison valid.

## Goal

Case A answers *"what does the service do under a realistic agentic load?"* It
cannot answer *"how fast can this service possibly be?"* — every number in it
carries an unknown amount of queueing.

This run removes the queue: **identical request shape, one request in flight,
never two.** Any gap between the two runs is therefore attributable to load
alone.

**Spec:** `spec/solo.yaml` (the workload, with every knob's mechanism documented
inline). Derived from `spec/glm52_crxx_caseA.fix.yaml`; parent mission in
`spec/mission.kv.liying.mtp.bench.md`.

**Success criteria** — this run had no pre-agreed SLA bar (it is an exploratory
latency-floor measurement), so it is judged on whether the *measurement* is
sound:

| criterion | result |
|---|---|
| concurrency genuinely 1, provably | ✅ `in_flight ∈ {0,1}` across all 1,795 rows; `max_inflight` warning never fired |
| request shape identical to Case A | ✅ same triples, same `cache_hit_rate`, same seed, same 400 s ramp |
| per-request E2E and TPOT measured, not derived | ✅ via the `SOLO_M1` driver patch |
| clean run | ✅ 0 driver errors, 0 engine faults on either leg in the window |

## Result at a glance

| | **solo** | Case A (loaded) | ratio |
|---|---|---|---|
| TTFT p50 | **1,563 ms** | 4,531 ms | 2.90× |
| TTFT p90 | **2,899 ms** | 9,170 ms | 3.16× |
| TPOT p50 | **10.68 ms** (floor 7.78) | 14.8 ms | 1.39× |
| E2E p50 | **4,124 ms** | ~11,100 ms | 2.69× |
| MTP acceptance | **2.069** | 2.036 | 1.02× |
| success | **1.000** (0/145 errors) | 0.988 | — |
| emergent qps | 0.066 | 0.75 | — |

### The headline finding: queueing is an additive ~3.9 s, not a multiplier

Bucketing by input length removes the confound that 102 samples cannot reproduce
a 235K-token p99:

| input | solo mean TTFT | Case A mean TTFT | ratio | **difference** |
|---|---|---|---|---|
| 0–50K | 773 ms | 4,641 ms | 6.00× | **+3,868 ms** |
| 50–80K | 1,272 ms | 4,911 ms | 3.86× | **+3,639 ms** |
| 80–120K | 1,985 ms | 5,256 ms | 2.65× | **+3,271 ms** |
| 120–160K | 2,946 ms | 5,671 ms | 1.93× | **+2,725 ms** |
| 160–300K | 4,502 ms | 6,878 ms | 1.53× | **+2,376 ms** |

Reproduce this table with `python3 scripts/compare_vs_caseA.py` (reads this kit's
`results/` and the sibling Case A kit's).

The ratio collapses from 6.0× to 1.5× purely because the denominator grows. The
*penalty* is roughly constant. Operationally: **under load, a small request is
not fast** — its 0.8 s of work sits behind ~3.9 s of other people's prefill. The
service is not latency-fair across request sizes.

### Three secondary findings

1. **`sla.e2e_p50_ms: 4500` is achievable — solo p50 is 4,124 ms.** The Case A
   packup left open whether that target was a deployment shortfall or a bad
   target. Neither: it is a **latency-floor spec**, met at concurrency 1 and
   unmeetable under Case A's load. (This corrects a guess made mid-analysis that
   the target was simply mis-written.)
2. **PD disaggregation is doing its job.** Decode degrades 1.4× under load;
   prefill degrades 2.9×. The decode leg runs a bounded batch while prefill
   absorbs the bursty, size-variable arrival queue.
3. **TPOT gets *faster* with longer generations** (11.99 ms at 0–100 tokens →
   9.31 ms at 2–6K). MTP warm-up and scheduling overhead amortize; growing KV
   does not dominate even at 14K output tokens.

## ⚠️ Two things this kit depends on

**`GLM52_P1V3` (`patches/0004`)** — inherited from Case A and still live on the
decode leg. Stock `infera/engine-sglang:merged-e` **cannot run this request shape
at all**; it dies on a DSA-indexer shape assert under MTP draft-extend. Verified
present in the *loaded* module immediately before this window opened.

**`SOLO_M1` (`patches/0005`)** — new here. The driver records neither
per-request E2E nor per-request TPOT to disk; Case A had to *back-solve* E2E.
Without this patch the two headline ladders of this experiment cannot be
produced. Side benefit: it let us verify the identity
`TTFT + (gen−1)×TPOT = E2E` to **+0.0 ms at every percentile, 102/102** —
retroactively validating the method Case A relied on.

## Navigate

| file | read it for |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | **the ordered, copy-pasteable reproduction** |
| [`environment.md`](environment.md) | hardware, fabric, image, SHAs, and the proof nothing restarted |
| [`analysis/solo_latency.md`](analysis/solo_latency.md) | **every ladder, every comparison, every caveat** |
| [`notes.md`](notes.md) | gotchas, wrong turns, why each knob is what it is |
| [`spec/solo.yaml`](spec/solo.yaml) | the workload, with mechanism comments |
| [`patches/`](patches/) | 0004 = P1V3 (inherited), 0005 = SOLO_M1 (new) |
| [`results/`](results/) | `metrics.jsonl.gz`, `summary.json`, kvd before/after |

## Layout

    README.md              this file
    REPRODUCE.md           exact ordered steps
    environment.md         hw/sw/fabric/paths/secrets + no-restart proof
    notes.md               gotchas and wrong turns
    scripts/               every script that ran, verbatim
    patches/               0004 inherited (load-bearing), 0005 new (measurement)
    results/               metrics.jsonl.gz + summary + kvd snapshots
    analysis/              the full report
    logs/                  driver log, gzipped
    env/                   collect_env output per node
    spec/                  solo.yaml + Case A yaml + the mission

## Honest caveats

1. **Not a capacity measurement.** 0.066 qps, 3,710 uncached TPM/GPU against
   Case A's 53,190. Nothing here says what the service can sustain.
2. **n=102 in the measured window.** p50/p90 solid; **p99 rests on ~1 request**.
   Intrinsic to concurrency 1 — buying samples would have meant shortening
   requests, breaking shape-parity and invalidating every comparison.
3. **Input tail under-sampled** (solo p99 184.6K vs the spec's 235K). The
   bucketed table works around this; the unbucketed ratios do not and are the
   weaker number.
4. **The 3.9 s penalty is one load point.** Whether it is flat, linear or
   knee-shaped in concurrency needs a sweep; this run and Case A are 2 points.
5. **kvd gets = +0.** A single warm stream never reads the spill tier. The
   "Case A does not exercise tiering" open item stays open.
