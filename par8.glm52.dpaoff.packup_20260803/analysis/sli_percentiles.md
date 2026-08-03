# Serving SLIs — full percentile ladders

All ladders **recomputed from the raw per-request samples** in
`../results/metrics.jsonl.gz`, not copied from the driver's summary line. Unless
stated otherwise every number is the **sustain phase only** (n = 2,671), which
is the measured window; ramp (n = 174) is warm-up and is excluded.

Reproduce any table here with:

```bash
python3 -c "
import json,gzip,statistics as st
rows=[json.loads(l) for l in gzip.open('../results/metrics.jsonl.gz','rt') if l.strip()]
v=[x for r in rows if r.get('phase')=='sustain' for x in (r.get('new_ttfts') or [])]
s=sorted(v); print([round(s[int(len(s)*f)]*1000) for f in (.5,.75,.9,.95,.99)])
"
```

---

## TTFT — time to first token

| | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| **sustain** | **1,365** | 2,611 | **4,903** | 6,121 | 9,066 | 13,304 | 2,083 |
| ramp | 1,177 | — | 7,006 | — | 36,754 | — | — |
| whole run | 1,353 | — | 4,948 | — | 10,674 | 38,965 | 2,170 |

*(ms)*

The ramp tail (p99 36.8 s, max 39.0 s) is **an artifact of cold cache, not a
result**: the 231K-token shared prefix is not yet resident, so early requests
pay full prefill. It decays within the 400 s exclusion window and is why that
window exists. Quote sustain.

### TTFT scales monotonically with input size

Bucketed over the sustain phase — this is the single most informative table in
the kit, because it shows the prefill leg behaving like a prefill leg:

| input tokens | n | TTFT p50 | TTFT p90 |
|---|---|---|---|
| 0 – 50K | 632 | **623 ms** | 3,333 ms |
| 50 – 100K | 1,219 | 1,036 ms | 4,158 ms |
| 100 – 160K | 580 | 1,815 ms | 4,886 ms |
| 160 – 220K | 205 | 3,411 ms | 6,797 ms |
| 220 – 300K | 35 | **5,863 ms** | 8,605 ms |

p50 rises **9.4×** across a **~4.5×** span of input length — super-linear, as
expected when a growing prompt spills past the per-forward chunk and pays more
chunked-prefill passes. **No knee, no cliff, no stall bucket.** Contrast the
DPA-off *solo* run, where two requests stalled at 11.4 s on a 77K input while a
215K input served in 6.1 s — that signature (large TTFT uncorrelated with size)
is absent here.

## E2E — end-to-end request latency

| | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| sustain | **7.4 s** | 16.8 s | 36.9 s | 58.5 s | 130.7 s | 239.2 s | 15.8 s |

E2E is dominated by **generation length**, not by queueing: gen_len p50 is 316
tokens but p90 is 2,852 and p99 is 10,898. At TPOT p50 14.8 ms, a p99-length
request spends ~161 s purely generating. The E2E ladder is essentially the
gen_len ladder scaled by TPOT plus TTFT.

**max 239.2 s sits just under the driver's 240 s client timeout** — 6 requests
exceeded 200 s, **none timed out**. This is worth watching: a slightly longer
tail, or slightly slower decode, would start converting long-but-valid requests
into client-side timeouts, which would look like server errors.

## TPOT — time per output token

| | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| sustain | **14.8 ms** | 16.2 | **17.7** | 18.8 | 21.4 | 28.9 | 14.8 |

Remarkably tight: p99/p50 = **1.45**. Decode is not contended in this run — a
loaded decode leg would show a fat upper tail. This is the strongest single
piece of evidence that **prefill, not decode, is the binding resource here**.

Equivalent per-request throughput: **67.3 tok/s/request**.

## MTP acceptance

| | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|
| per-request accept len | 2.0 | 2.5 | 2.8 | **3.0** | **2.02** |

Engine-side, from the decode log: mean **2.763** over n = 1,594 batch samples,
min 1.40. Mean acceptance **rate** 0.403.

**Healthy band.** `4.00` would mean the draft model is perfectly predicting a
repetition loop — the degenerate case CLAUDE.md flags as indistinguishable from
KV corruption. Per-request max is 3.0, so no request degenerated.

> **One honest caveat.** The engine-side batch samples contain **61 occurrences
> of exactly `accept len: 4.00` out of 1,594 (3.8 %)**. These are *instantaneous
> per-batch* values, not per-request means, and the per-request ladder above
> caps at 3.0. **Whether those 61 batches are genuine repetition loops is NOT
> established** — the measurement that would settle it is dumping the output
> token IDs of the requests in those batches and checking for repeats. Related
> and unresolved: the driver hardcodes `temperature: 0.0` (see `../notes.md`).

## Cache

| | value |
|---|---|
| actual hit rate | **88.94 %** |
| ideal (no eviction) | 88.98 % |
| **efficiency** | **100.0 %** |
| total tokens | 246,937,309 |
| prefix tokens | 219,741,732 |
| cached | 219,368,000 |
| evicted | 373,732 (**0.2 %**) |

The workload asked for `cache_hit_rate: 0.89`; the engine delivered 88.94 % with
0.2 % eviction. **The prefix cache is doing exactly what the YAML specified.**

This is the number people conflate with kv-aware routing. It is not the same
thing — these hits are sglang's own radix tree *inside* the single prefill
target. See `routing_and_kvaware.md`.

## Throughput

| | p50 | p90 | max |
|---|---|---|---|
| prefill (tok/s, windowed) | 63,901 | 86,316 | **117,870** |
| generation (tok/s, MTP-compensated) | 108.9 | — | 124.4 |

Driver-reported aggregates: peak prefill **646,125 tok/s total / 80,766 tok/s
per GPU**; average context **462,412 tok/min/GPU (7,707 tok/s/GPU)**.

> The two prefill figures use different windows — 117,870 is the 30 s smoothed
> series in `metrics.jsonl`, 646,125 is the driver's instantaneous peak. Both
> are in the kit; do not mix them in one sentence.

## Offered load — what actually happened

| | configured | measured |
|---|---|---|
| initial_sessions | 8 | 8 at t=0 |
| max_sessions | 32 | **reached 32**; steady 21–26 |
| max_inflight | 24 | **max 22**, mean 12.1 — never pinned |
| new_session_rate | 0.10 | 419 rate-based births |
| QPS | *emergent* | **0.72** (sustain 0.74) |

**The cap was not binding.** in-flight peaked at 22 of 24, so the *workload* set
the load, not backpressure — the window is valid. Sessions did hit their cap of
32, which the offline preview predicted (holding `new_session_rate` at Case A's
N=32 value while lowering `initial_sessions` to 8 means births push the
population toward 32 regardless of the starting point).

Session lifetimes: min 8 s, max 2,364 s, mean 508 s. Final: 26 active,
6 retired, **0 abandoned**.

## Errors — all 15, one cause

| | |
|---|---|
| sent | 2,884 |
| completed | 2,850 |
| errors | **15** (0.52 %) |
| success rate | **98.8 %** |

Every error is the same class — **HTTP 502 wrapping an engine 400**:

```
decode 10.2.122.10:30000 error 400: "Requested token count exceeds the model's
maximum context length of 262144 tokens. You requested a total of 265545 tokens"
```

**Not timeouts, not OOM, not crashes.** The request was rejected before
generation began. Zero requests hit the 240 s client timeout (E2E max 239.2 s).

**Root cause is a workload arithmetic gap inherited from Case A**, not an engine
fault: `max_input_tokens: 260000` clamps the *input* only, while `max_tokens` is
sampled independently with an upper bound of ~451K
(`p99 × (p99/p90)² = 17000 × (17000/3300)²`). Nothing checks
`input + output ≤ 262144`. The failure needs the joint tail — input near 260K
**and** an output sample of a few thousand — hence 0.52 %.

Case A full carries the identical gap (18 errors / 2,988 = 0.60 %). **Fixing it
would break comparability**, so it is recorded, not fixed.
