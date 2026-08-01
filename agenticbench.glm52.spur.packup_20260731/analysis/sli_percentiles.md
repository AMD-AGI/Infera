# Serving SLIs — full percentile ladders

Run `caseA_full/2026-07-31-17-10-30`. Deployment: two-node PD over mooncake RDMA,
DP-attention 8/8, kv-aware ON, prefill kvd ON / decode kvd OFF, **MTP OFF**,
`--context-length 262144`.

TTFT is recomputed here from the **2,781 raw per-request samples** in
`results/metrics.jsonl.gz` (`new_ttfts`, full-precision float seconds), which carries
more resolution than the three percentiles persisted in `summary.json`.

Recompute with:

    zcat results/metrics.jsonl.gz | python3 -c "
    import sys,json,math
    def P(a,p):
        a=sorted(a); k=(len(a)-1)*p/100.0; lo,hi=math.floor(k),math.ceil(k)
        return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(k-lo)
    t=[]
    for line in sys.stdin: t+=json.loads(line).get('new_ttfts') or []
    print([round(P(t,p)*1000) for p in [1,5,10,25,50,75,90,95,99,99.5,99.9]])"

---

## TTFT — Time To First Token (ms)

| Percentile | ALL (n=2781) | **SUSTAIN (n=2588)** | RAMP (n=190) |
|---|---|---|---|
| min | 421 | 421 | 505 |
| p1 | 890 | 970 | 596 |
| p5 | 1,499 | 1,640 | 869 |
| p10 | 1,990 | 2,109 | 994 |
| p25 | 2,990 | 3,087 | 1,517 |
| **p50** | 4,444 | **4,543** | 2,920 |
| p75 | 6,448 | 6,558 | 4,742 |
| **p90** | 8,652 | **8,795** | 7,025 |
| p95 | 10,106 | 10,313 | 8,079 |
| **p99** | 12,991 | **13,062** | 9,162 |
| p99.5 | 14,216 | 14,393 | 9,738 |
| p99.9 | 16,288 | 16,325 | 10,524 |
| max | 16,724 | 16,724 | 10,721 |
| mean | 4,983 | 5,092 | 3,487 |

**vs SLA `ttft_p90_ms ≤ 30,000` → 8,795 ms, 3.4× margin.**

**Cross-check.** Recomputed p50/p90/p99 = 4,543 / 8,795 / 13,062 against
`summary.json`'s 4,543 / 8,821 / 13,070. Agreement within 0.3 %; the residual is the
interpolation method (linear here, `np.percentile` default in the driver). Two
independent paths, consistent.

**Why the ramp phase is excluded.** RAMP runs 30–50 % below SUSTAIN at every percentile.
Folding it in would drag p50 down ~2 %. `ramp_duration=400` is a warm-up exclusion
window, and it is doing real work.

**Shape.** TTFT tracks prompt length closely: p50 73.9K tok → 4,543 ms and p99 226.9K tok
→ 13,062 ms, a 3.07× latency rise against a 2.88× token rise. Near-linear, so prefill was
not in a non-linear congestion regime at this load.

## TPOT — Time Per Output Token (ms, excludes the first token)

| Percentile | Whole run (n=2781) | ramp | sustain | drain |
|---|---|---|---|---|
| **p50** | **31.2** | 28.1 | **31.3** | 31.5 |
| **p90** | **32.5** | 30.5 | **32.6** | 31.5 |
| **p99** | **37.9** | 36.2 | **37.9** | 31.5 |
| mean | 30.9 (32.4 tok/s/req) | — | — | — |

> ⚠️ **Only these three percentiles exist, and the gap cannot be closed from the
> artifacts.** The driver keeps per-request TPOT in memory (`metrics.actual_tpots`,
> `agent_throughput.py:327`) but persists only p50/p90/p99 (`:1412-1414`).
> `metrics.jsonl` has no `new_tpots` field. A full ladder needs a driver change and a
> re-run; the spur jobs have since hit walltime TIMEOUT and the nodes were released.

Definition and filter (`agent_throughput.py:354-355`):

```python
if actual_gen_length > 1 and generation_time >= MIN_GENERATION_TIME:   # 50 ms
    self.actual_tpots.append(generation_time / (actual_gen_length - 1))
```

All 2,781 samples passed the filter.

**Shape.** The distribution is tight — p50 → p99 rises only 21 %, against TTFT's 2.9×.
That matches decode being a steady per-token process whose tail is set by in-batch
contention rather than by prompt length.

## Supporting per-request distributions

Same raw samples; these are what explain the two tables above.

| Percentile | Prompt length (tok) | Generation length, sustain (tok) | Inter-arrival (s) |
|---|---|---|---|
| p1 | — | 31 | — |
| p5 | — | 31 | — |
| p25 | — | 88 | — |
| **p50** | **73,862** | **289** | **4.0** |
| p75 | — | 886 | — |
| **p90** | **151,526** | **2,380** | **27.4** |
| p95 | — | 3,608 | — |
| **p99** | **226,854** | **6,326** | **168.6** |
| p99.9 | — | 7,915 | — |
| mean | 85,172 | 802 | 14.8 |

Inter-arrival is **response + think time**, not think time alone.

## E2E — no SLI exists

`sla.e2e_p50_ms: 4500` has no measured counterpart. `args.sla_cfg` is parsed and never
consumed, and the driver emits no E2E percentile anywhere.

Only back-solving is possible. **These are derived, not measured:**

    E2E ≈ TTFT + (gen_len − 1) × TPOT
    p50:  4,543 +   288 × 31.3 ms  ≈  13.6 s
    p90:  8,795 + 2,379 × 32.6 ms  ≈  86.4 s
    p99: 13,062 + 6,325 × 37.9 ms  ≈   253 s     ← exceeds the client's 240 s timeout

The p99 estimate landing above 240 s independently explains the 96 timeouts (3.3 % of
requests sent, same order as the observed 1 − 0.953 = 4.7 %). No separate cause needed.

**An unresolved discrepancy, stated rather than smoothed over.** A second back-solve, from
the duty cycle (`N_inflight / N_session = 28/38 = 0.74`), gives E2E ≈ 51 s, while the
mean-based composition gives `4,983 + 801 × 30.9 ms ≈ 29.7 s`. The two disagree by 1.7×.
The difference should be time spent spinning at the `max_inflight` gate in
`send_realistic_request`, which is not counted in TTFT — but there is no direct evidence
for that decomposition in the artifacts, so it is left open.

## Throughput (sustain phase, 3600 s, 2588 completed)

| | value |
|---|---|
| qps | 0.719 |
| input TPM (incl. cache) | 3,668,448 |
| **uncached TPM (real prefill work)** | **407,291** (50,911 /GPU) |
| generation TPM | 34,997 |
| peak prefill | 522,974 tok/s total (65,372 /GPU) |

## Cache

| | |
|---|---|
| ideal hit rate | 0.8899 |
| **actual hit rate** | **0.8900** |
| efficiency (actual/ideal) | **1.0002** |
| eviction rate | **0.0** |
| source | server `usage.prompt_tokens_details.cached_tokens` (`--enable-cache-report`) |

## Success rate

2919 sent, 2782 completed, **96 errors → 0.953** against an SLA of 0.97.

All 96 are the client's hardcoded `aiohttp.ClientTimeout(total=240)`
(`agent_throughput.py:928`). Verified exactly, not inferred: 96 lines matching
`timed out`, **0** matching `failed: HTTP`, 0 other exceptions. Both engine legs logged
0 GPU faults and 0 scheduler exceptions across the full 67-minute window. Reported as
measured rather than adjusted upward.
