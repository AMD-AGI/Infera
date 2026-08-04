# SLI ladders

All figures from `analyze.py` (copied verbatim from the 20260803 kit, so the
estimands are identical), profiling phase only, cancelled excluded.

## Full ladders — concurrency 8

| | 20260803<br>DPA off, kv-aware pw=20 | round-robin<br>DPA on | kv-aware pw=2<br>DPA on |
|---|---|---|---|
| profiling requests | 231 | 136 | 135 |
| window | 901.2 s | 902.7 s | 914.8 s |
| throughput | 0.256 req/s · 268.1 tok/s | 0.151 req/s · 178.2 tok/s | 0.148 req/s · 181.1 tok/s |
| errors / cancelled / ctx-overflow | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| in-flight max / mean | 8 / 5.13 | 8 / 6.05 | 8 / 6.38 |
| **TTFT** p50 | 5,146 | 22,236 | 25,188 |
| TTFT p90 | 19,780 | 42,693 | 69,938 |
| TTFT p99 | 31,014 | 62,965 | 100,846 |
| TTFT max | — | 74,859 | 118,365 |
| TTFT mean | — | 25,014 | 29,657 |
| **E2E** p50 | 12,556 | 34,966 | 36,135 |
| E2E p90 | 44,522 | 75,353 | 86,409 |
| E2E p99 | — | 202,183 | 209,212 |
| **ITL** p50 | 13.81 | 19.46 | **14.85** |
| ITL p90 | 27.43 | 28.24 | 22.32 |
| ITL p99 | — | 32.53 | 30.03 |
| ISL p50 / mean | — | 65,904 / 79,138 | 65,904 / 78,582 |
| OSL p50 / mean | — | 260 / 1,183 | 240 / 1,227 |

All times in ms. The ISL columns confirm the two legs replayed the **same
frozen trace at the same size distribution** — the demand is identical by
construction, which is the property that makes an open-loop replay comparable
across configurations at all.

## TTFT by input size

| bucket | RR n | RR p50 | RR p90 | pw2 n | pw2 p50 | pw2 p90 |
|---|---|---|---|---|---|---|
| 0–50K | 37 | 16,518 | 30,277 | 36 | 18,205 | 43,641 |
| 50–100K | 64 | 21,830 | 41,487 | 66 | 23,679 | 71,994 |
| 100–160K | 27 | 30,677 | 45,767 | 25 | 30,941 | 79,078 |
| 160–220K | 5 | 38,112 | 49,834 | 5 | 35,222 | 52,159 |
| 220–300K | 3 | 56,037 | 74,859 | 3 | 49,728 | 51,535 |

Spread across the 0–300K span: **3.4×** (RR) and **2.7×** (pw2), against
**10.0×** in the 20260803 c8 run. The 20260803 kit read a wide spread as
prefill-shaped and a narrow one as queue-shaped; both legs here sit at the
narrow end, and at c8 rather than c16.

**This is an observation, not an attribution.** Four things differ from
20260803 simultaneously (DPA, gmu, per-forward chunk, prefill delayer), and the
kv-aware leg additionally ran with an empty cache view. Nothing here isolates
which one narrowed the curve.

## Cache-hit rate — both estimands

The 20260803 kit established that this number has to be stated with its
estimand, because the distribution is multi-modal (first turns at 0 %, prefix-
starved turns spread wide, normal turns near the trace's constructed 88–90 %).

| | token-weighted | per-request p50 | requests reporting the field |
|---|---|---|---|
| 20260803, DPA off, pw=20 | 50.3 % | 88.1 % | 177 / 231 |
| round-robin, DPA on | **12.4 %** | 55.7 % | 38 / 136 |
| kv-aware pw=2, DPA on | **27.4 %** | 69.1 % | 65 / 135 |

`analyze.py`'s single "server cache hit" line is the **token-weighted** figure;
that is the one with a physical meaning (prefill tokens actually skipped).

### An anomaly in the denominator, unresolved

In the 20260803 run, every request that did **not** report
`usage_prompt_cache_read_tokens` was a first turn — 54/54 had
`turn_index == 0`, which is exactly right: a first turn has no prefix to reuse.

Here that no longer holds:

| | records without the field | of those, `turn_index == 0` |
|---|---|---|
| round-robin | 98 | 35 |
| kv-aware pw=2 | 70 | 34 |

So ~60 requests per leg are **turn ≥ 1 yet report no cache field at all**. That
is a different condition from "reported a low hit rate", and it is not
explained. It also means the per-request p50 column above is computed over a
differently-selected subset than 20260803's, so the three per-request numbers
are **not** strictly comparable to each other. The token-weighted column does
not have this problem (a missing field contributes 0 to the numerator and its
full ISL to the denominator, which is the correct treatment either way).

## What is safe to take from this page

1. **Zero errors in every leg.** 271 requests across the two c8 legs, no
   cancellations, no context overflows.
2. **ITL p50 is stable across all three configurations** — 13.8 / 19.5 / 14.9 ms.
   Decode was not the bottleneck in any of them, the same conclusion par8 and
   the 20260803 kit reached.
3. **TTFT and throughput are both markedly worse than the 20260803 posture**,
   by roughly 4–5× and 0.6× respectively. Which of the four simultaneous
   changes is responsible is **not determined** by this data.
