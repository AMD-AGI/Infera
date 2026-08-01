# Serving SLIs — full percentile ladders

Run `caseA_full/2026-08-01-13-34-46`. Deployment: two-node PD over mooncake RDMA
(ionic RoCE), DP-attention 8/8 on both legs, kv-aware routing ON via the **Rust**
router, prefill kvd ON / decode kvd OFF, **MTP ON**, `--context-length 262144`.

> **Image is `infera/engine-sglang:merged-e` + the `GLM52_P1V3` patch.** Stock
> merged-e **cannot complete this workload** — the decode leg dies on a DSA
> indexer shape assert under MTP draft-extend (twice reproduced, at 125 s and
> 766 s). See `../notes/notes.dsa.mtp.crash.md`. Every number below is
> post-patch.

TTFT is recomputed here from the **2,952 raw per-request samples** in
`results/caseA/metrics.jsonl.gz` (`new_ttfts`, full-precision float seconds),
which carries more resolution than the three percentiles persisted in
`summary.json`.

Recompute with:

    zcat results/caseA/metrics.jsonl.gz | python3 -c "
    import sys,json,math
    def P(a,p):
        a=sorted(a); k=(len(a)-1)*p/100.0; lo,hi=math.floor(k),math.ceil(k)
        return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(k-lo)
    t=[]
    for line in sys.stdin: t+=json.loads(line).get('new_ttfts') or []
    print([round(P(t,p)*1000) for p in [1,5,10,25,50,75,90,95,99,99.5,99.9]])"

---

## TTFT — Time To First Token (ms)

| Percentile | ALL (n=2952) | **SUSTAIN (n=2702)** | RAMP (n=246) |
|---|---|---|---|
| min | 299 | 419 | 299 |
| p1 | 735 | 791 | 362 |
| p5 | 1,348 | 1,543 | 659 |
| p10 | 1,838 | 1,973 | 986 |
| p25 | 2,841 | 2,966 | 1,508 |
| **p50** | 4,377 | **4,531** | 3,060 |
| p75 | 6,436 | 6,660 | 4,435 |
| **p90** | 8,938 | **9,170** | 5,596 |
| p95 | 10,694 | 11,182 | 6,566 |
| **p99** | 16,425 | **16,706** | 8,053 |
| p99.5 | 18,119 | 18,260 | 9,324 |
| p99.9 | 22,405 | 22,461 | 10,051 |
| max | 22,915 | 22,915 | 10,171 |
| mean | 5,015 | 5,186 | 3,140 |

**vs SLA `ttft_p90_ms ≤ 30,000` → 9,170 ms, 3.3× margin.**

**Cross-check.** Recomputed p50/p90/p99 = 4,377 / 8,938 / 16,425 (ALL) against
`summary.json`'s 4,378 / 8,940 / 16,492. Agreement within 0.4 %; the residual is
the interpolation method (linear here, `np.percentile` default in the driver).
Two independent paths, consistent.

**Why the ramp phase is excluded.** RAMP runs 30–50 % below SUSTAIN at every
percentile — p50 3,060 vs 4,531 ms. `ramp_duration=400` is a warm-up exclusion
window sized at roughly one session lifetime, and it is doing real work: at t=0
the 231K-token shared prefix is not yet resident, and the synchronized initial
cohort has not yet dispersed.

**Shape.** TTFT tracks prompt length closely: p50 74.6K tok → 4,531 ms and p99
233.7K tok → 16,706 ms — a 3.69× latency rise against a 3.13× token rise. Mildly
superlinear (ratio 1.18), so prefill was entering, but not deep into, a
congestion regime at this load. Compare the spur run's MTP-OFF ratio of 1.07 at
a lower population.

## TPOT — Time Per Output Token (ms, excludes the first token)

| Percentile | Whole run (n=2952) | ramp | sustain | drain |
|---|---|---|---|---|
| **p50** | **14.9** | 15.2 | **14.8** | 18.6 |
| **p90** | **17.9** | 18.4 | **17.8** | 18.8 |
| **p99** | **20.9** | — | — | — |
| mean | 14.9 (67.2 tok/s/req) | — | — | — |

> ⚠️ **Only these percentiles exist, and the gap cannot be closed from the
> artifacts.** The driver keeps per-request TPOT in memory
> (`metrics.actual_tpots`) but persists only p50/p90/p99; `metrics.jsonl` has no
> `new_tpots` field. A full ladder needs a driver change and a re-run.

All 2,952 samples passed the driver's filter (`gen_len > 1` and
`generation_time ≥ 50 ms`).

**Shape.** The distribution is tight — p50 → p99 rises only 40 %, against TTFT's
3.8×. That matches decode being a steady per-token process whose tail is set by
in-batch contention rather than by prompt length.

**This is the headline MTP result.** 14.8 ms p50 against the spur run's
**31.3 ms** with MTP off — a **2.11× improvement** in per-token latency on the
same model and workload. The two runs differ in cluster (vultr ionic vs spur
mlx5) and in kvd decode wiring, so this is not a single-variable ablation; but
the magnitude matches the measured acceptance length (mean 2.04, below) almost
exactly, which is the mechanism MTP is supposed to deliver.

## MTP acceptance — the mechanism behind the TPOT number

Per-request acceptance length, sustain phase (n=2,702):

| p1 | p5 | p25 | **p50** | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|
| 1.27 | 1.44 | 1.84 | **2.00** | 2.24 | 2.50 | 2.65 | 2.84 | **2.04** |

Server-side, across 19,225 decode-batch log lines: **mean 2.80**, min 1.20, max
4.00.

**Against the workload's declared `acc_len: 1.56`, the deployment delivers
2.04 — 31 % better than the config assumes.** The YAML's "56 % acceptance @ 5
draft tokens" is conservative for this stack.

**On the 4.00 readings.** CLAUDE.md principle 2 flags `accept len: 4.00` as the
repetition-loop tell (a draft model perfectly predicting a degenerate loop).
431 of 19,225 server lines (2.2 %) read exactly 4.00. These are transient
small-batch maxima, not a degenerate state: the per-request p99 is 2.84 and the
distribution is smooth and unimodal, with no mass piling at the ceiling. Had the
model been looping, the *request-level* mean would have climbed toward 4, not
sat at 2.04. Sampling used GLM-5.2's own `generation_config.json` (temp 1.0,
top_p 0.95), not `temperature: 0`.

## Supporting per-request distributions (sustain)

| Percentile | Prompt length (tok) | Generation length (tok) | Inter-arrival (s) |
|---|---|---|---|
| p1 | 35,342 | 31 | 0.52 |
| p5 | 35,342 | 31 | 0.52 |
| p25 | 50,295 | 96 | 1.35 |
| **p50** | **74,267** | **330** | **3.98** |
| p75 | 109,330 | 1,081 | 10.96 |
| **p90** | **153,788** | **2,897** | **28.68** |
| p95 | 179,790 | 4,914 | 53.28 |
| **p99** | **233,655** | **11,097** | **172.12** |
| p99.9 | 260,013 | 20,204 | 864.19 |
| mean | 86,109 | 1,108 | 15.32 |

Inter-arrival is **response + think time**, not think time alone.

The p1/p5 prompt floor at exactly 35,342 tokens is the shared-prefix construction:
every request nests inside the same base, so no prompt can be shorter than it.
p99.9 at 260,013 is the `max_input_tokens: 260000` clamp binding.

## E2E — no SLI exists, but this time the two back-solves agree

`sla.e2e_p50_ms: 4500` has no measured counterpart. `args.sla_cfg` is parsed and
never consumed, and the driver emits no E2E percentile anywhere.

**Route 1 — composition.** `E2E ≈ TTFT + (gen_len − 1) × TPOT`, per-request,
sustain phase:

| | p50 | p90 | p99 | mean |
|---|---|---|---|---|
| derived E2E (s) | **11.1** | **48.7** | **170.5** | **21.6** |

**Route 2 — duty cycle.** `N_inflight / N_session = E2E / (E2E + delay)`:

    mean in-flight 16.0 / mean live sessions 27.0 = 0.592
    mean inter-arrival (response + think) = 15.3 s
    E2E = 0.592 × 15.3 / (1 − 0.592) = 22.2 s

**21.6 s vs 22.2 s — the two independent routes agree within 3 %.** This is the
discrepancy the spur analysis had to leave open (it saw 51 s vs 29.7 s, a 1.7×
gap, attributed to un-instrumented `max_inflight` spin-wait). Here the caps never
bound, so no hidden queueing time exists to reconcile — which is itself the
evidence that the gap in the earlier run was indeed the spin-wait.

vs the YAML's `e2e_p50_ms: 4500`: derived p50 is **11.1 s, 2.5× over target**.
The target is documentation only and is not gated on. Note it was almost certainly
written against a much smaller generation length — at the p50 of 330 output
tokens, 4.5 s would require 13.6 ms of *total* latency per token including
prefill of a 74K prompt, which no configuration of this model achieves.

## Throughput (sustain phase, 3600 s, 2702 completed)

| | value |
|---|---|
| qps | 0.75 |
| input TPM (incl. cache) | 3,877,785 |
| **uncached TPM (real prefill work)** | **425,522** (53,190 /GPU) |
| generation TPM | 49,903 |
| peak prefill | 686,052 tok/s total (85,756 /GPU) |

The guide's Case A reference figure is **~69K uncached TPM/GPU** at N=32. This run
realized 53,190 /GPU at a mean live population of 27 — 77 % of the reference at
84 % of the reference population. Consistent.

## Cache

| | |
|---|---|
| ideal hit rate | 0.8899 |
| **actual hit rate** | **0.8924** |
| efficiency (actual/ideal) | **1.0028** |
| eviction rate | **0.0** |
| total tokens | 254,623,157 (prefix 226,580,963, cached 227,213,056, evicted 0) |
| source | server `usage.prompt_tokens_details.cached_tokens` (`--enable-cache-report`) |

Actual **exceeds** ideal by 0.3 %, because "ideal" assumes only the modelled
prefix is cached while the radix tree also retains some request-unique suffix
across turns of the same session.

## Success rate

2,988 sent, 2,952 completed, **18 errors → 0.988** against an SLA of 0.97. **Met.**

All 18 are the client's hardcoded `aiohttp.ClientTimeout(total=240)`
(`agent_throughput.py:3107`). Verified, not inferred: 18 lines matching
`timed out`, **0** matching `failed: HTTP`, 0 other exceptions. Both engine legs
logged 0 GPU faults, 0 scheduler exceptions, 0 retractions, and 0 DSA asserts
across the full 67-minute window.

The mechanism is the workload's own output tail crossing a fixed client deadline:
p99 generation is 11,097 tokens × 14.9 ms ≈ 165 s, and p99.9 is 20,204 tokens ≈
301 s — past the 240 s cap. 18/2988 = 0.6 %, and the fraction of requests above
~16,000 output tokens is of the same order.

**Compare the spur run: 0.953, 96 timeouts (3.3 %).** The improvement is
mechanical — MTP halved TPOT, so a generation of any given length now finishes in
half the time and far fewer cross the deadline.
