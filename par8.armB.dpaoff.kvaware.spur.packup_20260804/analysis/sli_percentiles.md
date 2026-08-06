# SLI ladder — arm B

All numbers read out of `results/summary.json`. Nothing here is recomputed by hand.

## Whole run

| metric | value |
|---|---|
| duration | 4007.7 s |
| requests sent | 2,907 |
| requests completed | 2,861 |
| errors | 25 |
| success rate | **0.9842** |
| achieved QPS | 0.7253 |

## Latency

| | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| **TTFT (ms)** | 3047 | **2171** | **6404** | 10909 |
| **TPOT (ms)** | 16.6 | **16.0** | 21.1 | 35.0 |

## Request shape as ACTUALLY sampled (vs the YAML's targets)

| | p50 | p90 | p99 | YAML p50/p90/p99 |
|---|---|---|---|---|
| input tokens | 74,853 | 154,176 | 225,679 | 74,000 / 155,000 / 235,000 |
| output tokens | 314 | 2,698 | 10,249 | 320 / 3,300 / 17,000 |

The sampled distribution tracks the spec closely at p50 and p90. The input p99
lands at 225,679 against a 235,000 target and a 260,000 clamp, so the clamp is
**not** truncating the tail — at ctx 262,144 there is headroom above the p99.

## Cache

| | value |
|---|---|
| ideal hit rate (workload construction) | 0.8899 |
| **actual hit rate (server-reported)** | **0.8865** |
| efficiency (actual / ideal) | **0.9962** |
| eviction rate | 0.0038 |
| server reported cached_tokens | True |

`server_reported_cached: true` matters: it means the numbers come from the
engine's `usage.prompt_tokens_details.cached_tokens`, which only populates when
the leg runs with `--enable-cache-report`. Without that flag this column silently
reads 0 and the whole cache row is lost.

## Token totals

| | tokens |
|---|---|
| input_tokens | 246,231,789 |
| cached_tokens | 218,281,088 |
| uncached_tokens | 27,950,701 |
| prefix_tokens | 219,113,726 |
| generation_tokens | 2,925,633 |

## Per phase

| phase | n | qps | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | cache |
|---|---|---|---|---|---|---|---|
| ramp | 142 | 0.355 | 1,425 | 5,851 | 46,303 | 14.0 | 0.8683 |
| sustain | 2,717 | 0.755 | 2,239 | 6,389 | 10,606 | 16.1 | 0.8874 |
| drain | 2 | 0.000 | 10,887 | 10,887 | 10,887 | 20.6 | 0.8899 |

**Read the sustain row, not the whole-run row.** `ramp_duration` is a warmup
EXCLUSION window, not a load ramp — nothing ramps in closed-loop mode. Its
TTFT p99 of 46,303 ms is the synchronized t=0 cohort all arriving at once against
a cold 231K-token shared prefix; that is exactly what the 400 s window exists to
exclude. Sustain's p99 is 10,606 ms, 4.4x lower.

The drain row is 2 requests. It is noise and is shown only for completeness.
