# Serving SLIs — full percentile ladders

Run `caseA_armB/2026-08-01-15-37-08`, 2026-08-01 15:37:08–16:43:56 UTC.
Deployment: two-node PD over mooncake RDMA (**mlx5_0, GID 3, dma-buf ON**),
DP-attention 8/8 both legs, kv-aware routing ON (Rust router), prefill kvd ON /
decode kvd OFF, **MTP ON** (EAGLE, steps 3, topk 1, draft 4),
`--context-length 262144`.

> **This is the first Case A run on this cluster that completed its full
> window.** Two prior attempts died on the prefill leg with
> `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. The fix was `--mem-fraction-static`
> 0.88 → **0.80** on prefill; see `../notes/notes.config.md`. The decode leg
> additionally carries the **`GLM52_P1V3`** patch (`../patches/apply_p1v3.py`).
> Every number below is post-fix.

TTFT, prompt length, generation length, cache hit and driver-side acceptance are
recomputed here from the **2,811 raw per-request samples** in
`results/metrics.jsonl.gz`, which carries more resolution than the three
percentiles persisted in `summary.json`.

Recompute with:

    zcat results/metrics.jsonl.gz | python3 -c "
    import sys,json,math
    def P(a,p):
        a=sorted(a); k=(len(a)-1)*p/100.0; lo,hi=math.floor(k),math.ceil(k)
        return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(k-lo)
    t=[]
    for line in sys.stdin: t+=json.loads(line).get('new_ttfts') or []
    print([round(P(t,p)*1000) for p in [1,5,10,25,50,75,90,95,99,99.5,99.9]])"

`RAMP` = the first 400 s (`ramp_duration`, a warm-up **exclusion** window, not a
load ramp); `SUSTAIN` = the 3,600 s measured window. Split at
`elapsed_seconds < 400`.

---

## TTFT — Time To First Token (ms)

| Percentile | ALL (n=2811) | SUSTAIN (n=2582) | RAMP (n=229) |
|---|---|---|---|
| min | 830 | 830 | 1,004 |
| p1 | 1,357 | 1,357 | 1,452 |
| p5 | 2,016 | 2,054 | 1,901 |
| p10 | 2,732 | 2,817 | 2,172 |
| p25 | 4,308 | 4,414 | 3,542 |
| **p50** | **6,659** | **6,733** | **5,474** |
| p75 | 11,636 | 11,474 | 14,830 |
| **p90** | **19,238** | **18,877** | **23,532** |
| p95 | 25,734 | 24,833 | 39,768 |
| **p99** | **37,736** | **33,097** | **43,357** |
| p99.5 | 40,987 | 37,689 | 43,992 |
| p99.9 | 42,667 | 41,351 | 44,104 |
| max | 44,124 | 42,195 | 44,124 |
| mean | 9,242 | 9,116 | 10,659 |

**vs SLA `ttft_p90_ms ≤ 30,000` → 18,877 ms sustain. MET, 1.59× margin.**

**Cross-check.** Recomputed ALL p50/p90/p99 = 6,659 / 19,238 / 37,736 ms against
`summary.json`'s 6,658.5 / 19,238.0 / 37,742.0. Agreement to 0.02 %; the residual
is interpolation method (linear here, `np.percentile` default in the driver). Two
independent paths, consistent.

**Shape.** TTFT tracks prompt length superlinearly: p50 73.6K tok → 6,733 ms and
p99 226.4K tok → 33,097 ms — a **4.92×** latency rise against a **3.08×** token
rise, ratio **1.60**. Prefill was in a congestion regime at this load, more so
than vultr's ratio of 1.18 at the same nominal workload. The difference is
population: this run's live sessions climbed 22.6 → 44.1 across the four sustain
quarters (vultr peaked at 30 in-flight; here 44).

**RAMP is worse than SUSTAIN here, which inverts the usual pattern** (vultr's
ramp ran 30–50 % *below* sustain). p90 RAMP 23,532 vs SUSTAIN 18,877 ms. The
cause is visible in the cache ladder below: RAMP p1 cache hit is 44.2 % and its
min is 0 %, i.e. at t=0 the shared prefix is not yet resident and the
synchronized initial cohort of 32 sessions all miss at once. By SUSTAIN the
distribution has collapsed onto 88.9–89.0 %. This is the ramp window doing
exactly its job.

## TPOT — Time Per Output Token (ms, excludes the first token)

| | value |
|---|---|
| samples | 2,811 (filter: `gen_len>1 & gen_time>=50 ms`) |
| mean | 18.9 (52.8 tok/s/req) |
| **p50** | **17.9** |
| p90 | 24.9 |
| p99 | 42.2 |

**Only these four points exist.** Unlike TTFT, the driver persists no
per-request TPOT array in `metrics.jsonl` (verified: the only generation-side
keys are `generation_tps` and `new_generation_lengths`). A finer TPOT ladder is
**not recoverable** from this run's artifacts, and none is fabricated here.

TPOT p50 17.9 ms = **55.9 tok/s per request**, against a decode leg running MTP
at a measured 2.736 accepted tokens per step.

## Prompt length (tokens) — what the driver actually sent

| Percentile | ALL (n=2811) | SUSTAIN (n=2582) | RAMP (n=229) |
|---|---|---|---|
| min | 35,341 | 35,341 | 35,342 |
| p1 | 35,342 | 35,342 | 35,342 |
| p5 | 35,342 | 35,342 | 35,342 |
| p10 | 35,342 | 35,342 | 37,006 |
| p25 | 49,037 | 49,065 | 47,896 |
| **p50** | **73,618** | **73,934** | **71,952** |
| p75 | 110,266 | 110,035 | 111,610 |
| **p90** | **152,264** | **152,716** | **147,331** |
| p95 | 178,452 | 179,886 | 163,531 |
| **p99** | **225,721** | **226,392** | **185,282** |
| p99.5 | 251,155 | 251,353 | 207,185 |
| p99.9 | 260,013 | 260,013 | 248,772 |
| max | 260,013 | 260,013 | 260,013 |
| mean | 85,520 | 85,692 | 83,581 |

**vs the Case A spec (74K / 155K / 235K at p50/p90/p99):** measured
**73.6K / 152.3K / 225.7K**. Within 0.5 % / 1.8 % / 4.0 %. The workload
reproduced its own input distribution.

The p99.9 and max both read **260,013**, which is the `max_input_tokens: 260000`
clamp (plus template). 0.1 % of requests clamped — at `--context-length 262144`,
as intended. At `--context-length 131,072` this would clamp **15.4 %** of the
distribution (measured directly on the reference run's 2,781 raw samples: 428
prompts exceed 131,072). p90 alone is 152,264 tokens, so 131,072 cannot serve
even the p90 request, let alone p99 — which needs ~243K for 225.8K in + 17K out.
That is why 262,144 is the setting for this profile.

## Generation length (tokens)

| Percentile | ALL (n=2811) | SUSTAIN (n=2582) | RAMP (n=229) |
|---|---|---|---|
| min | 30 | 31 | 30 |
| p1 | 31 | 31 | 31 |
| p5 | 31 | 31 | 31 |
| p10 | 31 | 31 | 31 |
| p25 | 87 | 88 | 68 |
| **p50** | **299** | **303** | **233** |
| p75 | 934 | 948 | 788 |
| **p90** | **2,791** | **2,832** | **2,049** |
| p95 | 4,948 | 5,052 | 4,234 |
| **p99** | **9,688** | **10,134** | **7,280** |
| p99.5 | 12,000 | 12,123 | 8,627 |
| p99.9 | 17,818 | 17,850 | 8,980 |
| max | 20,434 | 20,434 | 9,032 |
| mean | 1,019 | 1,039 | 796 |

**vs the Case A spec (320 / 3.3K / 17K at p50/p90/p99):** measured
**299 / 2,791 / 9,688**. p50 within 7 %, p90 within 15 %, **p99 43 % short**.

The p99 gap is the 240 s client timeout truncating the tail, not the model
stopping early — see the error analysis in `README.md`. A 20,434-token
generation (the observed max) needs 366 s of decode at TPOT p50 alone.

## Cache hit rate (%) — per request

| Percentile | ALL (n=2811) | SUSTAIN (n=2582) | RAMP (n=229) |
|---|---|---|---|
| min | 0.00 | 85.30 | 0.00 |
| p1 | 88.51 | 88.84 | 44.17 |
| p5 | 88.88 | 88.89 | 85.15 |
| p10 | 88.91 | 88.91 | 88.87 |
| p25 | 88.92 | 88.92 | 88.91 |
| **p50** | **88.96** | **88.96** | **88.95** |
| p75 | 88.98 | 88.98 | 88.98 |
| **p90** | **88.99** | **88.99** | **88.99** |
| p95 | 88.99 | 88.99 | 88.99 |
| **p99** | **89.00** | **89.00** | **89.00** |
| p99.5 | 89.00 | 89.00 | 89.00 |
| p99.9 | 89.00 | 89.00 | 89.00 |
| max | 89.00 | 89.00 | 89.00 |
| mean | 88.83 | 88.94 | 87.51 |

**vs the Case A target of 88–90 %:** actual **88.82 %** aggregate against an
ideal of **88.99 %** → efficiency **99.81 %**, eviction **0.192 %**.

The sustain distribution is astonishingly tight: p5 through max all read
88.9–89.0 %. Every request nests in the same growing prefix and essentially
nothing is evicted. Note `min` = 0.0 % in the ALL/RAMP column and 85.3 % in
SUSTAIN — the zero is the very first request of the run, which by definition has
no prefix to hit.

## MTP acceptance — two independent sources

**Engine-side** (`accept len` in the decode log, n=16,588 decode batches):

| | value |
|---|---|
| mean | **2.736** |
| min / p25 / p50 / p75 / p90 / max | 1.15 / 2.23 / 2.67 / 3.16 / 3.70 / 4.00 |
| lines reading exactly 4.00 | 494 (**3.0 %**) |

**Driver-side** (`new_acceptance_lengths`, per completed request):

| Percentile | ALL (n=2811) | SUSTAIN (n=2582) | RAMP (n=229) |
|---|---|---|---|
| min | 1.05 | 1.05 | 1.25 |
| p1 | 1.26 | 1.26 | 1.26 |
| p5 | 1.41 | 1.42 | 1.34 |
| p10 | 1.54 | 1.57 | 1.40 |
| p25 | 1.78 | 1.80 | 1.60 |
| **p50** | **1.98** | **2.00** | **1.79** |
| p75 | 2.19 | 2.21 | 1.96 |
| **p90** | **2.50** | **2.50** | **2.24** |
| p95 | 2.65 | 2.65 | 2.52 |
| **p99** | **2.86** | **2.86** | **2.75** |
| p99.5 | 2.91 | 2.91 | 2.83 |
| p99.9 | 2.95 | 2.95 | 2.85 |
| max | 2.97 | 2.97 | 2.85 |
| mean | 2.00 | 2.01 | 1.82 |

**Both land in the healthy band.** Per the project's own criterion, a *sustained*
`accept len: 4.00` is the repetition-loop tell — a draft model perfectly
predicting a degenerate output. Here 4.00 is 3.0 % of batches, transient, against
a 2.736 mean. Vultr's sibling run measured 2.2 % at a 2.80 mean; this is the same
regime.

The driver-side mean (2.0) is lower than engine-side (2.736) because the two
measure different things: the driver averages SSE chunk sizes per request, which
undercounts whenever a chunk boundary does not align with a verify step. The
engine counter is the authoritative one.

**vs the Case A spec's "56 % acceptance @ 5 draft tokens" (= acc_len 1.56):**
measured 2.736 at 4 draft tokens = **68.4 % acceptance**. The shipped assumption
is conservative for this stack.

## Throughput

| | value |
|---|---|
| duration | 4,007.4 s |
| requests | 2,881 sent / **2,811 completed** / 39 errors |
| **success rate** | **97.57 %** (SLA 97 % — **met**) |
| QPS | 0.719 (emergent, closed-loop) |
| input tokens | 240,396,631 → **59,988 tok/s** (7,498/GPU) |
| ├ cached | 213,510,592 (88.8 %) |
| └ **uncached — real prefill work** | 26,886,039 → **6,709 tok/s** |
| generation tokens | 2,864,792 → **714.9 tok/s** |

The distinction between the 59,988 and 6,709 figures matters and is easy to
misreport: the first is what the workload *presented*, the second is what the
prefill leg actually *computed*. At 88.8 % cache hit these differ by ~9×.

## Load-shape sanity: the run measured its configured workload

| | |
|---|---|
| in-flight peak | **44 / 48** — never pinned |
| live sessions peak | 54 / 128 |
| live sessions, sustain quarters | 22.6 → 22.3 → 34.4 → **44.1** |

In-flight never reaching the cap is the load-bearing check: had it pinned,
backpressure rather than the configured birth rate would be setting the load, and
the percentiles above would describe the cap, not the deployment. The previous
spur attempt failed exactly this test.

The session population does climb across quarters (22 → 44). This is the expected
tail-censoring recovery — a workload with a 158.7 s p99 inter-turn delay and up
to 103-turn sessions cannot reach its steady population inside one window. It is
also why TTFT here exceeds vultr's: the last quarter runs at roughly double the
first quarter's concurrency.

## SLA verdict

| target | source | measured | verdict |
|---|---|---|---|
| `ttft_p90_ms ≤ 30,000` | YAML `sla:` | 18,877 (sustain) | **PASS** (1.59×) |
| `success_rate ≥ 0.97` | YAML `sla:` | 0.9757 | **PASS** |
| cache hit 88–90 % | Case A spec | 88.82 % | **PASS** |
| input p50/p90/p99 74K/155K/235K | Case A spec | 73.6K/152.3K/225.7K | **PASS** |
| output p50/p90 320/3.3K | Case A spec | 299/2,791 | **PASS** |
| output p99 17K | Case A spec | 9,688 | **MISS** — client-timeout truncation |
| acceptance 56 % @ 5 draft | Case A spec | 68.4 % @ 4 draft | **EXCEEDS** |
| `e2e_p50_ms ≤ 4,500` | YAML `sla:` | not measured | **see below** |

**`sla.e2e_p50_ms` has no measured counterpart.** `args.sla_cfg` is parsed and
never consumed; the driver emits no end-to-end percentile. It can only be
back-solved, and the routes disagree: TTFT p50 (6,659 ms) + gen p50 (299 tok) ×
TPOT p50 (17.9 ms) = **12.0 s**, while `new_session_times` is a session lifetime,
not a request E2E, and cannot be used. Either way the 4.5 s target is missed by
~2.7×, consistent with vultr's 2.5× miss, and the target looks written against a
shorter generation than this profile produces. Recorded rather than quoted as a
number.
