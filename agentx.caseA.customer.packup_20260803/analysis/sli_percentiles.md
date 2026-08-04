# Serving SLIs — full percentile ladders

All ladders **recomputed from aiperf's raw per-request records**
(`../results/c*/profile_export.jsonl.gz`), not copied from any summary line.
Every number is the **profiling phase only** — warmup (8 requests at c8, 12 at
c16) is excluded, exactly as par8 excludes its ramp.

Reproduce any table here:

```bash
zcat ../results/c8/profile_export.jsonl.gz > /tmp/c8.jsonl
python3 ../scripts/analyze.py <dir-containing-profile_export.jsonl>
```

---

## The measured window

| | c8 | c16 |
|---|---|---|
| profiling requests | **231** | **323** |
| warmup requests (excluded) | 8 | 12 |
| cancelled | **0** | **0** |
| context-overflow skips | **0** | **0** |
| window (first start → last end) | 901.2 s | 921.8 s |
| request rate | 0.256 req/s | 0.350 req/s |
| output token rate | 268.1 tok/s | 384.4 tok/s |
| **in-flight max / mean** | **8 / 5.13** | **16 / 11.08** |

**In-flight reached the concurrency ceiling at both points.** This is the
defining property of the run and the reason the latency ladders read the way
they do — see `README.md`.

## TTFT — time to first token

| | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| **c8** | **5,146** | 11,397 | **19,780** | 23,926 | 31,014 | 33,943 | 7,601 |
| **c16** | **14,394** | 21,061 | **30,567** | 40,588 | 74,645 | 83,467 | 16,468 |
| *par8 (ours)* | *1,365* | *2,611* | *4,903* | *6,121* | *9,066* | *13,304* | *2,083* |

*(ms)*

**TTFT is entirely server-side wait.** `http_req_sending` p50 is **0.2 ms** at
both points, and `http_req_waiting` p50 equals TTFT p50 to the millisecond
(5,146 / 14,394). No client-side or network contribution.

### TTFT vs input size — the shape changes between c8 and c16

| input tokens | c8 n | c8 p50 | c8 p90 | c16 n | c16 p50 | c16 p90 |
|---|---|---|---|---|---|---|
| 0 – 50K | 56 | **1,867** | 16,120 | 73 | **11,949** | 31,840 |
| 50 – 100K | 113 | 3,937 | 17,756 | 159 | 13,504 | 28,661 |
| 100 – 160K | 50 | 8,663 | 22,272 | 76 | 15,675 | 32,476 |
| 160 – 220K | 7 | 15,996 | 33,943 | 9 | 19,461 | 83,467 |
| 220 – 300K | 5 | **18,661** | 19,797 | 6 | **27,860** | 34,306 |

**This is the most diagnostic table in the kit.**

- **At c8 the curve is prefill-shaped**: p50 rises **10.0×** across the size
  range, the same monotone super-linear behaviour par8 measured (9.4×). The
  deployment is computing, not queueing.
- **At c16 the curve flattens**: only **2.3×** across the same span, and the
  smallest bucket already costs 11,949 ms. A 40K-token request cannot take 12
  seconds to prefill on a leg that serves 240K in 28 s. It is **waiting**.

The transition between these two shapes is where this deployment's usable
concurrency limit sits. It is between 8 and 16 and this run does not locate it.

### First-turn vs cached-turn — pricing the prefix cache

Split by `turn_index`, at essentially identical input size:

| | n | ISL p50 | **TTFT p50** | TTFT p90 |
|---|---|---|---|---|
| **c8** first turn (cold) | 54 | 69,907 | **8,981** | 24,144 |
| **c8** turn ≥ 1 (cached) | 177 | 70,998 | **3,568** | 17,495 |
| **c16** first turn | 73 | 67,612 | 17,449 | 62,051 |
| **c16** turn ≥ 1 | 250 | 74,300 | 13,561 | 25,565 |

At c8 the cached turns are **2.5× faster on TTFT for a 1.6 % larger prompt**.
That ratio is the prefix cache's value, measured directly.

At c16 the ratio collapses to 1.3× — consistent with queue delay, which is
common to both groups, dominating the compute the cache saves. Same reading as
the flattened size curve above.

## E2E — end-to-end request latency

| | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| **c8** | **12,556** | 23,503 | 44,522 | 68,843 | 117,151 | 173,120 | 20,019 |
| **c16** | **21,620** | 34,418 | 66,089 | 87,818 | 185,801 | 289,856 | 31,626 |
| *par8* | *7,400* | *16,800* | *36,900* | *58,500* | *130,700* | *239,200* | *15,800* |

*(ms)*

E2E is dominated by generation length, as in par8: OSL p50 is 223 tokens but
p99 is 10,527, and at ITL ~14 ms a p99-length request spends ~147 s purely
generating. **No client timeout exists in aiperf's agentic replay**, so the
289.9 s max at c16 completed normally — unlike par8, where max 239.2 s sat just
under a 240 s client timeout.

## ITL — inter-token latency (= our TPOT)

| | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| **c8** | **13.81** | 19.32 | 27.43 | 30.31 | 32.06 | 36.62 | 16.39 |
| **c16** | **14.68** | 19.34 | 29.69 | 32.13 | 34.78 | 35.97 | 17.34 |
| *par8 TPOT* | *14.8* | *—* | *17.7* | *—* | *21.4* | *—* | *—* |

*(ms)*

**p50 agrees with par8 to within 7 %** — 13.8 / 14.7 vs 14.8 ms, measured by a
different driver on a different trace. Independent confirmation that decode is
healthy.

The **upper tail is wider** than par8's (p99 32–35 ms vs 21.4 ms, ratio p99/p50
2.3 vs par8's 1.45). Doubling concurrency moves ITL p50 by only **+6 %** while
TTFT triples, so decode is not the binding resource; **why its tail is fatter
under this driver is not established** by this run. One candidate the data does
not settle: aiperf measures ITL across the whole stream including any gap
introduced by prefill chunking of concurrent requests, which par8's driver
computes differently.

## Workload reproduction — the trace replayed as authored

| | trace (authored) | c8 measured | c16 measured | par8 measured |
|---|---|---|---|---|
| ISL p50 | 74,140 | 70,624 | 73,450 | 75,440 |
| ISL p90 | 146,111 | 141,568 | 142,322 | 157,056 |
| ISL p99 | 245,000 | 241,910 | 241,910 | 230,669 |
| ISL mean | 84,213 | 80,989 | 82,330 | 86,888 |
| OSL p50 | 306 | 223 | 264 | 316 |
| OSL p90 | 2,980 | 2,772 | 2,833 | 2,852 |
| OSL p99 | 16,193 | 10,527 | 10,527 | 10,898 |

**ISL reproduces to within 5 %** — the replay is faithful on the input axis.

**OSL p99 is 35 % below the authored value** (10,527 vs 16,193). The mechanism
is visible in the window: a 16K-output request takes ~230 s to generate at
ITL 14 ms, and the 900 s profiling phase truncates on duration — the longest
outputs are the ones most likely to be in flight when the window closes and
therefore missing from the completed set. **par8 shows the same effect for the
same reason** (its OSL p99 10,898 against a sampler value of 17,098), so this is
a property of finite measurement windows, not of either driver.

## The error class: there isn't one

| | c8 | c16 |
|---|---|---|
| HTTP errors | **0** | **0** |
| cancelled | **0** | **0** |
| context-overflow skips | **0** | **0** |
| success rate | **1.000** | **1.000** |

Contrast par8's 15 errors (0.52 %), all HTTP 502 wrapping an engine 400
`Requested token count exceeds the model's maximum context length of 262144`.

**The customer's trace cannot produce that error class.** Its
`max(in + out) = 258,303` sits below the 262,144 context by construction, and
aiperf additionally logs `--max-context-length=262144: all 200 traces within
limit (largest: 244,937 tokens)` at load time. Our own workload samples input
and output independently with no joint clamp, which is where our 0.52 % comes
from.
