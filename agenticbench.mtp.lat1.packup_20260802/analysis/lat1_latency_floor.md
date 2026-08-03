# lat1 — the latency floor of the full-feature GLM-5.2 deployment

Run `lat1_full/2026-08-02-05-39-24`, 2026-08-02 05:39:24–06:12:25 UTC (1,980.9 s).
Deployment **unchanged from the Case A run**: two-node PD over mooncake RDMA
(mlx5_0, GID 3, dma-buf ON), DP-attention 8/8 both legs, kv-aware routing ON,
prefill kvd ON / decode kvd OFF, **MTP ON** (EAGLE, steps 3, topk 1, draft 4),
`--context-length 262144`, prefill `--mem-fraction-static 0.80`, decode carrying
`GLM52_P1V3`. Not one server flag was touched.

## What this measures, and why Case A could not

Case A answers *what does this deployment do under a realistic agentic
population*. It cannot answer *how fast is one request*, because at 44 in-flight
requests a TTFT of 18.9 s is mostly **waiting**, not **computing**. This workload
holds the request distribution fixed and collapses the population to exactly one
live session issuing exactly one turn at a time, so TTFT and TPOT become the
service time itself.

Everything describing the request is byte-identical to `caseA_full.yaml` — the
input/output percentile triples, `cache_hit_rate: 0.89`, `max_input_tokens`,
`system_prompt_len`, tokenizer, `acc_len`, `mtp_draft_tokens`, `sla`, `gpus`,
`window`. Six concurrency knobs differ, plus `random_seed` (see
`../notes/notes.lat1.md` — it had to change, and the reason is a real defect).

**The concurrency guarantee held**: `in_flight` max **1**, `num_sessions_active`
max **1**, every tick of the run. This is checked, not assumed; the run would be
measuring something else if it had failed.

---

## Headline

| | value | Case A (N≈44) | ratio |
|---|---|---|---|
| **TTFT p50** | **2,027 ms** | 6,733 ms | **3.32×** |
| **TTFT p90** | **4,920 ms** | 18,877 ms | **3.84×** |
| TTFT p99 | 7,134 ms *(±25 %, see below)* | 33,097 ms | 4.64× |
| **TPOT p50** | **10.66 ms** | 17.9 ms | **1.68×** |
| TPOT p90 | 12.23 ms | 24.9 ms | 2.04× |
| TPOT p99 | 13.75 ms | 42.2 ms | 3.07× |
| success rate | **1.0000** (124/124) | 0.9757 | — |
| cache actual / ideal | 0.8897 / 0.8899 | 0.8882 / 0.8899 | — |
| MTP acceptance (engine) | **2.846** | 2.736 | — |

**Read directly: 70 % of Case A's median TTFT is queueing, not computation**
(4,706 of 6,733 ms). At p90 it is 74 %, at p99 78 %. And **40 % of Case A's
median TPOT is batching contention** — a decode step at N=1 costs 10.66 ms
against 17.9 ms in a loaded batch.

---

## The result that actually answers "where is the latency limit"

TTFT is a **clean linear function of prompt length** once queueing is removed:

    TTFT_ms = -319 + 29.33 x (input_ktok)          R² = 0.9563     n = 124

| input bin | n | mean input | mean uncached | mean TTFT | ms/ktok total | ms/ktok uncached |
|---|---:|---:|---:|---:|---:|---:|
| 0–40K | 17 | 36,412 | 4,035 | 960 | 26.4 | 238.0 |
| 40–60K | 19 | 48,973 | 5,422 | 1,184 | 24.2 | 218.4 |
| 60–80K | 21 | 70,191 | 7,755 | 1,716 | 24.5 | 221.3 |
| 80–100K | 23 | 88,942 | 9,819 | 2,176 | 24.5 | 221.7 |
| 100–130K | 19 | 115,138 | 12,694 | 2,973 | 25.8 | 234.2 |
| 130–160K | 5 | 145,448 | 16,027 | 3,678 | 25.3 | 229.5 |
| 160–200K | 15 | 175,256 | 19,309 | 4,708 | 26.9 | 243.8 |
| 200–240K | 4 | 217,078 | 23,910 | 6,303 | 29.0 | 263.6 |
| 240K+ | 1 | 260,013 | 28,653 | 9,284 | 35.7 | 324.0 |

The marginal rate is **34,092 tok/s of presented prompt** — but that number mixes
cached and uncached tokens at a fixed 89:11 ratio, so it is a property of *this
profile*, not of the engine. Against uncached tokens alone the fit gives
**3,749 tok/s**, and the two fits have identical R² (0.9563 vs 0.9565) because
cached and uncached are collinear here by construction (ratio 7.98–8.09, i.e.
essentially constant). **This run therefore cannot separate the cached-token cost
from the uncached-token cost** — that would need a cache-hit sweep, and it is not
claimed here.

**The ms/ktok column is flat from 40K to 160K and then rises**: 24.2 → 24.5 →
24.5 → 25.8 → 25.3, then 26.9 → 29.0 → 35.7. Prefill cost per token is constant
up to ~160K and becomes superlinear beyond it — 48 % more expensive per token at
260K than at 50K. That is the attention term becoming visible. **The knee is at
roughly 160K–200K input tokens**, and it is the single most actionable number
here: below it, latency is predictable from length; above it, it degrades faster
than length.

### Under load, length stops predicting latency at all

Fitting the *same model* to Case A's 2,811 samples:

| run | fit | R² | marginal |
|---|---|---|---|
| **lat1 (N=1)** | `TTFT_ms = -319 + 29.33/ktok` | **0.9563** | 34,092 tok/s |
| **Case A (N≈44)** | `TTFT_ms = 8,681 + 6.56/ktok` | **0.0016** | — |

R² of **0.0016**. Under load, a request's own length explains essentially *none*
of its TTFT; an 8.7 s intercept — the queue — explains it. Bin-matched, the
penalty collapses with size:

| input bin | Case A TTFT p50 | lat1 TTFT p50 | queueing penalty |
|---|---:|---:|---:|
| 0–50K | 5,960 | 960 | **6.21×** |
| 50–70K | 5,907 | 1,416 | 4.17× |
| 70–90K | 6,659 | 1,953 | 3.41× |
| 90–110K | 6,391 | 2,368 | 2.70× |
| 110–140K | 6,939 | 3,045 | 2.28× |
| 140–180K | 7,060 | 4,241 | 1.66× |
| 180–230K | 8,440 | 5,341 | **1.58×** |

Case A's TTFT is nearly **flat** across input length (5,960 → 8,440 ms, 1.4×)
while lat1's rises 5.6×. Under load every request waits behind the same queue, so
small requests are penalised most: a 40K prompt that should take 960 ms takes
5,960 ms. This is the head-of-line cost of mixing 35K and 260K prompts in one
queue, quantified.

### Reproduce both tables above

Run from this kit's root. It reads only this kit and the Case A sibling kit — no
scratch paths. Both fits and the bin table come out of the same 12 lines:

```python
import json, gzip, math

def load(path):
    op = gzip.open if path.endswith('.gz') else open
    t, p = [], []
    for line in op(path, 'rt'):
        r = json.loads(line)
        t += r.get('new_ttfts') or []
        p += r.get('new_prompt_lengths') or []
    n = min(len(t), len(p))
    return t[:n], p[:n]

ca = load('../agenticbench.mtp.caseA.packup_20260801/results/metrics.jsonl.gz')
l1 = load('results/metrics.jsonl.gz')

def fit(xs, ys):                       # least squares + R^2
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sl = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    ic = my - sl*mx
    ss = sum((y-my)**2 for y in ys)
    rs = sum((y-(ic+sl*x))**2 for x, y in zip(xs, ys))
    return sl, ic, 1 - rs/ss

def P(a, q):
    a = sorted(a); k = (len(a)-1)*q/100.0
    lo, hi = math.floor(k), math.ceil(k)
    return a[lo] if lo == hi else a[lo] + (a[hi]-a[lo])*(k-lo)

for nm, (t, p) in (('Case A (N~44)', ca), ('lat1  (N=1)', l1)):
    sl, ic, r2 = fit(p, [x*1000 for x in t])
    print('%-14s TTFT_ms = %8.0f + %6.2f/ktok   R2=%.4f' % (nm, ic, sl*1000, r2))

for lo, hi in [(0,50000),(50000,70000),(70000,90000),(90000,110000),
               (110000,140000),(140000,180000),(180000,230000)]:
    A = [ca[0][i]*1000 for i in range(len(ca[0])) if lo <= ca[1][i] < hi]
    L = [l1[0][i]*1000 for i in range(len(l1[0])) if lo <= l1[1][i] < hi]
    if len(L) < 3:
        continue
    print('%6d-%-6d  A n=%-4d %7.0f | L n=%-3d %7.0f | %.2fx'
          % (lo, hi, len(A), P(A,50), len(L), P(L,50), P(A,50)/P(L,50)))
```

Expected first two lines:

    Case A (N~44)  TTFT_ms =     8681 +   6.56/ktok   R2=0.0016
    lat1  (N=1)    TTFT_ms =     -319 +  29.33/ktok   R2=0.9563

Note this fit uses **all** requests of each run (ramp included), which is why the
lat1 intercept is −319 rather than the sustain-only value; the R² contrast — the
actual finding — is unaffected.

---

## TPOT — and why lat1 lost no requests to the client timeout

| | whole run (n=124) | sustain (n=119) |
|---|---|---|
| samples | 124 (filter `gen_len>1 & gen_time>=50 ms`) | 119 |
| mean | 10.50 ms (95.2 tok/s/req) | — |
| **p50** | **10.66 ms** | **10.66 ms** |
| p90 | 12.18 ms | **12.23 ms** |
| p99 | 13.75 ms | — |

Only these points exist — the driver persists no per-request TPOT array in
`metrics.jsonl` (verified: the only generation-side keys are `generation_tps` and
`new_generation_lengths`). A finer ladder is not recoverable and none is
fabricated. The phase split comes from `summary.json`'s `phases[]` array, which
carries p50/p90 only (no mean, no p99) — hence the dashes.

**The headline tables in this document and in `../README.md` quote the SUSTAIN
column** (12.23 ms), to match the sustain-phase TTFT and the sustain-phase Case A
figures they are compared against. The two differ by 0.4 %, which is the four
ramp requests.

**p50 → p99 spans only 1.29×**, against TTFT's 3.5×. Decode at N=1 is a steady
per-token process with almost no tail; the tail Case A shows (p99 42.2 ms, 3.07×
worse) is entirely in-batch contention.

Case A lost 39 requests to the hardcoded 240 s client timeout
(`agent_throughput.py:2246`) and its output p99 came in **43 % short** of spec.
**lat1 lost zero.** The mechanism is arithmetic:

| | generation | decode time @ TPOT p50 |
|---|---:|---:|
| p50 | 390 tok | 4.2 s |
| p90 | 3,104 tok | 33.1 s |
| p99 | 14,946 tok | 159.4 s |
| **max** | **22,219 tok** | **236.9 s** |

The timeout binds at ≥22,509 tokens. The longest generation in the run needed
236.9 s — **1.3 % under the deadline**. At Case A's 17.9 ms TPOT the same request
would have needed 397 s and been killed. So the output distribution reproduces
here (p99 14,946 vs spec 17,000, within 12 %) where Case A's could not, and the
truncation is confirmed as a client artifact rather than a model behaviour.

---

## Distributions — the workload reproduced its own spec

| | spec p50/p90/p99 | lat1 measured (n=124) | Case A (n=2,811) |
|---|---|---|---|
| input tokens | 74,000 / 155,000 / 235,000 | 83,048 / 171,271 / 233,537 | 73,618 / 152,264 / 225,721 |
| output tokens | 320 / 3,300 / 17,000 | 390 / 3,104 / 14,946 | 299 / 2,791 / 9,688 |
| cache hit | 88–90 % | **88.9–89.0 %** (p1 to max) | 88.8 % |

Input p99 and cache land within 1 %. Input p50/p90 run 12 %/10 % **high** — with
n=124 that is sampling noise on a heavy-tailed distribution, not a
misconfiguration: the same config produced 73,618 at n=2,811. Output p99 is the
one place lat1 is *closer* to spec than Case A, for the timeout reason above.

The cache ladder is the tightest in either run: **p1 through max all read
88.9–89.0 %**. Eviction 0.02 %.

---

## Statistical honesty — what n=124 supports

95 % confidence intervals from order statistics, sustain phase (n=120):

| quantile | value | 95 % CI | verdict |
|---|---:|---|---|
| p10 | 905 ms | 858–1,094 | solid (±13 %) |
| p25 | 1,323 ms | 1,118–1,510 | solid (±15 %) |
| **p50** | **2,029 ms** | **1,805–2,249** | **solid (±11 %)** |
| p75 | 3,041 ms | 2,668–4,027 | weak (±22 %) |
| **p90** | **4,866 ms** | **4,216–5,383** | **solid (±12 %)** |
| p95 | 5,346 ms | 4,939–6,967 | weak (±19 %) |
| p99 | 7,031 ms | 5,796–9,284 | **weak (±25 %)** |

**p50 and p90 are solid. p99 is not a percentile at this sample size** — it is
the 118.8th of 120 order statistics, i.e. essentially the third-largest
observation. It is reported as the range **5.8–9.3 s** and should not be quoted
as a point.

This is a structural limit, not an oversight. At concurrency 1 the run rate is
its own service time: 124 requests in 1,981 s is 16.0 s/request, which is exactly
the mean E2E. Reaching n=1,000 would take **4.4 hours**. The trade was made
deliberately — the measurement needs concurrency 1, and concurrency 1 caps the
sample rate.

---

## E2E and the SLA

`sla.e2e_p50_ms: 4500` still has no measured counterpart (`args.sla_cfg` is
parsed and never read). Composed per request as `TTFT + (gen−1) × TPOT_p50`:

| p1 | p25 | **p50** | p75 | p90 | p99 | mean | max |
|---|---|---|---|---|---|---|---|
| 1.28 s | 3.78 s | **6.57 s** | 17.53 s | 36.21 s | 161.56 s | 16.56 s | 239.76 s |

**p50 6.57 s = 1.46× the 4.5 s target — still missed, but by far less than Case
A's 2.7×.** With queueing entirely removed and the fastest decode this stack can
do, the target is *still* not met at this request profile. That settles what the
Case A analysis had to leave open: the 4.5 s figure is not a load problem, it is
incompatible with a 74K-token p50 prompt. Serving a 74K prompt takes ~2.0 s of
prefill before a single token is emitted, and the p50 request then decodes 390
tokens. No configuration of this model reaches 4.5 s for that shape.

Sanity check on the composition: Σ E2E = 2,054 s against a 1,981 s wall clock,
duty cycle **1.037**. Slightly over 1.0 because TPOT_p50 is applied uniformly to
requests whose own TPOT varied; the agreement to 4 % confirms the closed loop was
genuinely serialised with negligible dead time.

---

## Cross-check against the fixed-length sweep

The `bench_serving` sweep measured this same server at conc=1 with
`--random-range-ratio 1.0`, i.e. **0 % cache hit**:

| point | ISL | sweep TTFT | lat1 nearest bin | lat1 TTFT | ratio |
|---|---:|---:|---|---:|---:|
| p50 | 74,000 (all uncached) | 10,784 ms | 70,191 (7,755 uncached) | 1,716 ms | 6.28× |
| p90 | 155,000 (all uncached) | 24,457 ms | 175,256 (19,309 uncached) | 4,708 ms | 5.19× |

The sweep does **9.5×** and **8.0×** the actual prefill work, and takes 6.3× and
5.2× as long. Implied rates: sweep **6,862 / 6,338 tok/s** on uncached tokens,
lat1 **4,519 / 4,101 tok/s**.

**Cached tokens are therefore not free.** lat1's per-uncached-token rate is
~35 % worse than the sweep's, and the difference is the cost of walking and
attending over ~8× as many cached KV tokens. The exact split cannot be resolved
from this run (see the collinearity note above) — it is stated as a bound, not a
coefficient.

TPOT agrees closely across the two harnesses (sweep 11.53 ms at OSL 320; lat1
10.66 ms at gen p50 390), which is the expected result and a useful check that
the two paths measure the same thing.

---

## Engine health

| check | result |
|---|---|
| requests | 124 sent / **124 completed** / **0 errors** |
| prefill leg faults | **0** (HSA / Fatal Python / Memory access / scheduler exception / Traceback) |
| decode leg faults | **0** |
| retractions, either leg | **0** |
| MTP acceptance (engine, last 4,000 batches) | mean **2.846**, p50 2.75, p90 3.85 |
| `accept len: 4.00` readings | 180 / 4,000 = **4.5 %** |
| kvd `gets` / `hits` / `misses` | 15,210 → 15,210 / 15,210 → 15,210 / 0 → 0 |
| kvd `sets` | 423,410 → 427,202 (**+3,792**) |

Acceptance 2.846 at 4 draft tokens = **71.2 % acceptance**, against the Case A
spec's assumed 56 % @ 5 draft. The 4.5 % of batches reading exactly 4.00 is
higher than Case A's 3.0 % and is expected at N=1: with a single sequence in the
batch there is nothing to average against, so a fully-accepted step reads 4.00
outright. The per-request driver-side p99 is 2.9 with a smooth unimodal
distribution and no mass piling at the ceiling, so this is not the repetition
loop the project's criterion warns about.

**kvd was wired but not exercised** — `gets` did not move, exactly as in Case A.
Prefill kvd only fetches on a radix-tree miss, and at 89 % planned hit with a
single session the in-GPU tree served everything. The +3,792 `sets` show the
write path live. This is the same result as Case A and is a property of the
workload, not a defect.
