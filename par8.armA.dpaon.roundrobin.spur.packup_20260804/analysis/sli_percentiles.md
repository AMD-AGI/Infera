# SLI ladder — arm A (DPA on + round-robin)

All numbers read out of `results/summary.json`. Nothing is recomputed by hand.

## Whole run

| metric | value |
|---|---|
| duration | 4006.7 s |
| requests sent | 2,323 |
| requests completed | 2,289 |
| errors | 17 |
| success rate | **0.9854** |
| achieved QPS | 0.5798 |

## Latency

| | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| **TTFT (ms)** | 4402 | **3507** | **7225** | 21587 |
| **TPOT (ms)** | 17.5 | **16.5** | 23.2 | 38.3 |

## Request shape as ACTUALLY sampled

| | p50 | p90 | p99 | YAML target |
|---|---|---|---|---|
| input tokens | 75,668 | 155,601 | 237,743 | 74,000 / 155,000 / 235,000 |
| output tokens | 322 | 2,714 | 10,508 | 320 / 3,300 / 17,000 |

## Cache

| | value |
|---|---|
| ideal hit rate (workload construction) | 0.8899 |
| **actual hit rate (server-reported)** | **0.8819** |
| efficiency (actual / ideal) | **0.9911** |
| eviction rate | 0.0089 |

`server_reported_cached: true` means these come from the engine's
`usage.prompt_tokens_details.cached_tokens`, which only populates under
`--enable-cache-report`. Without that flag the column silently reads 0.

## Token totals

| | tokens |
|---|---|
| input_tokens | 199,677,086 |
| cached_tokens | 176,104,704 |
| uncached_tokens | 23,572,382 |
| prefix_tokens | 177,686,526 |
| generation_tokens | 2,358,361 |

## Per phase

| phase | n | qps | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | cache |
|---|---|---|---|---|---|---|---|
| ramp | 115 | 0.287 | 3,451 | 26,166 | 73,927 | 14.9 | 0.8052 |
| sustain | 2,170 | 0.603 | 3,504 | 7,079 | 16,602 | 16.5 | 0.8862 |
| drain | 4 | 1.319 | 14,560 | 19,063 | 19,063 | 19.3 | 0.8084 |

**Read the sustain row.** `ramp_duration` is a warmup EXCLUSION window, not
a load ramp — nothing ramps in closed-loop mode. The ramp row's TTFT tail is
the synchronized t=0 cohort hitting a cold 231K-token shared prefix, which is
exactly what the 400 s window exists to exclude.
