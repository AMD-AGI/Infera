# Solo latency floor with prefill DP-attention OFF

**Ran:** 2026-08-02, 05:46–06:23 UTC (400 s ramp + 1800 s measured window).
**Status:** PASS. 145 requests total / **105 in the measured window, 0 errors**,
concurrency verifiably pinned at 1.

**Question:** the DPA-on solo run established a latency floor. Does turning off
DP-attention on the **prefill** leg lower it?

**Answer: yes, and by a lot — prefill TTFT roughly halves.** Paired on identical
request shapes, DPA-off is faster on **49 of 50** and the **median paired
speedup is 2.01×**.

**The framing that makes every number below cohere: DP-attention spends latency
to buy KV capacity, and hence throughput headroom.** Turning it off refunds the
latency and hands the capacity back:

| | DPA-on | DPA-off | |
|---|---|---|---|
| TTFT p50 | 1,563 ms | **778 ms** | latency **refunded** (2.0×) |
| aggregate KV | 22,639,616 tok | **3,263,680 tok** | capacity **surrendered** (−85.6 %) |
| kvd gets | +0 | **+432** | the spill tier now does real work |
| evictions | +1,206 | **+10,249** | 8.5× — the working set no longer fits |

Both halves of that trade are measured here, and they are the same
phenomenon seen from two sides.

---

## 1. What was changed, and what was held fixed

The comparison is against `../../solo.glm52.latencyfloor.packup_20260802/`
(2026-08-01 16:05). Same workload file, same seed 1337, same driver + `SOLO_M1`
patch, same router, same **decode leg process** (PID 2420132, started
2026-08-01 13:29:28 — never restarted, `GLM52_P1V3` still loaded).

Only the prefill leg was relaunched (`TAG=p7`, 05:42:33), with DPA off.

### The single-variable claim, verified rather than assumed

Resolved server args, compared offline before spending a window:

| | dp_size | **ep_size** | chunked_prefill (resolved) | max_prefill | aiter AR fusion |
|---|---|---|---|---|---|
| DPA-ON | 8 | 8 | 8192 | 16384 | False |
| DPA-OFF | **1** | **8** | 8192 | 16384 | False |

- **MoE expert-parallelism is unchanged at 8.** This did not happen by default —
  see §5, trap 1. Without an explicit fix `ep_size` collapses to 1 and the run
  becomes a two-variable change.
- **Per-forward-pass prefill work is identical.** sglang divides
  `chunked_prefill_size` by `dp_size`, so CLI 65536÷8 (DPA-on) and CLI 8192÷1
  (DPA-off) both resolve to **8192 tokens per pass**.
- DSA supports pure-TP attention; `dp_size < tp_size` logs a warning, not an
  error.

### Shape parity (the comparison is void without this)

| | DPA-on | DPA-off | delta |
|---|---|---|---|
| input p50 | 77,107 | 77,117 | +0.0 % |
| input mean | 79,027 | 78,826 | −0.3 % |
| output p50 | 225 | 229 | +1.8 % |
| output mean | 1,583 | 1,528 | −3.5 % |

The shared seed reproduced the request stream almost exactly, so TTFT
differences are **not** explained by one run drawing smaller prompts.

---

## 2. Headline — TTFT

| TTFT | DPA-on | DPA-off | change |
|---|---|---|---|
| min | 689 ms | **327 ms** | −52.6 % |
| **p50** | 1,563 ms | **778 ms** | **−50.2 %** |
| p75 | 1,985 ms | **1,020 ms** | −48.6 % |
| **p90** | 2,899 ms | **1,536 ms** | **−47.0 %** |
| p95 | 3,336 ms | 2,166 ms | −35.1 % |
| **p99** | 4,472 ms | **11,185 ms** | **+150.1 %** ⚠️ |
| mean | 1,668 ms | 1,075 ms | −35.5 % |

Everything up to p95 improves by roughly half. **p99 is the exception and is
analysed in §4 — it is two isolated requests, not a trend.**

Distribution shift, which is the more honest summary at n≈100:

| | DPA-on | DPA-off |
|---|---|---|
| TTFT < 1 s | 26 | **77** |
| TTFT > 2 s | 25 | **7** |
| TTFT > 5 s | 1 | 3 |

### Bucketed by input length

| input | n(on) | n(off) | mean TTFT on | mean TTFT off | speedup |
|---|---|---|---|---|---|
| 0–50K | 25 | 26 | 773 ms | **372 ms** | **2.08×** |
| 50–80K | 34 | 34 | 1,272 ms | **942 ms** | 1.35× |
| 80–120K | 28 | 30 | 1,985 ms | **1,000 ms** | **1.99×** |
| 120–160K | 10 | 10 | 2,946 ms | **2,599 ms** | 1.13× |
| 160–300K | 5 | 5 | 4,502 ms | **3,053 ms** | 1.47× |

Every bucket improves. The two ragged buckets (50–80K at 1.35×, 120–160K at
1.13×) are the ones containing the outliers; the paired analysis below removes
that noise.

### The cleanest evidence: paired identical requests

Because both runs share seed 1337, 50 `(input, output)` shapes appear in both.
Comparing each request against *itself*:

```
DPA-off faster on 49 / 50 paired shapes
median paired speedup           2.01x
speedups cluster                1.7x - 2.1x
the single regression           in=77,117 out=31   1,565 ms -> 11,397 ms
next-worst case                 in=78,914 out=158  1,614 ms ->    930 ms  (1.74x FASTER)
```

A tight cluster at ~2× with one discontinuous outlier is the signature of a
systematic improvement plus one stall — not of a bimodal or unstable service.

---

## 3. Everything else

### TPOT — decode leg untouched, yet ~7 % slower

| TPOT | DPA-on | DPA-off | change |
|---|---|---|---|
| min | 7.78 ms | 7.43 ms | −4.5 % |
| **p50** | 10.68 ms | **11.17 ms** | **+4.5 %** |
| **p90** | 12.75 ms | **14.44 ms** | **+13.2 %** |
| mean | 10.83 ms | 11.64 ms | +7.5 % |

By output-length bucket the regression is present but small and not monotone
(+6.2 % / +14.3 % / +1.1 % / +8.7 %). The decode leg was **not restarted** —
same PID, same weights, same `P1V3` bytecode, engine-side `accept len` mean
2.834 over 1,768 batches (healthy).

**I am not claiming a cause.** The plausible mechanism is that a ~2× faster
prefill hands KV to decode sooner, so decode batches pack more densely and each
token pays slightly more contention. That is consistent with the data but is
**one run against one run** and is not established. Recorded as open.

Note the floor barely moved (7.78 → 7.43 ms), which argues the per-token
compute path itself is unchanged — as it should be, since decode never restarted.

### E2E, MTP, cache

| | DPA-on | DPA-off | change |
|---|---|---|---|
| E2E p50 | 4,124 ms | **3,751 ms** | −9.0 % |
| E2E mean | 16,957 ms | 16,284 ms | −4.0 % |
| MTP acceptance mean | 2.069 | 2.028 | −2.0 % |
| cache hit mean | 88.95 % | 88.83 % | −0.1 pp |

E2E improves far less than TTFT (−9 % vs −50 %) — expected, since E2E at this
shape is dominated by generation time, and generation is a decode concern. **The
`sla.e2e_p50_ms: 4500` target is met more comfortably: 3,751 ms.**

MTP acceptance is unchanged, confirming the decode path was not disturbed.

### kvd finally gets read — because aggregate KV capacity collapsed

| counter | DPA-on run | DPA-off run | ratio |
|---|---|---|---|
| gets | **+0** | **+432** | — |
| hits | +0 | +432 (100 %) | — |
| sets | +1,122 | +11,076 | **9.9×** |
| evictions | +1,206 | +10,249 | **8.5×** |

The DPA-on kit recorded "+0 gets" and left "kvd tiering unmeasured" open. This
run reads the spill tier — and the reason is the central mechanism of this
experiment, not a side effect.

**DP-attention buys aggregate KV capacity.** Under DPA each of the 8 DP ranks
runs its own scheduler owning its own *distinct* KV shard. Under pure TP there
is one scheduler and one KV pool, *replicated* across the 8 TP ranks. Confirmed
structurally in the boot logs, not inferred:

```
DPA-on   [DP0 TP0 EP0] max_total_num_tokens=2829952  max_running_requests=256   <- 2048/8, per-rank scheduler
DPA-off  [TP0 EP0]     max_total_num_tokens=3263680  max_running_requests=2048  <- ONE scheduler
```

| | per rank | ranks with distinct KV | **aggregate addressable KV** |
|---|---|---|---|
| DPA-on | 2,829,952 | 8 | **22,639,616 tokens** |
| DPA-off | 3,263,680 | 1 (replicated) | **3,263,680 tokens** |

**Aggregate capacity falls 85.6 %.** The +15.3 % per-rank growth (attention
weights shard, freeing memory) is a rounding error against that.

The eviction and set counters agree with the capacity number: **8.5×** more
evictions, **9.9×** more sets. Working set no longer fits on the GPU, so blocks
spill to the host tier and get read back — hence the 432 gets.

This is the same trade viewed from the memory side: **DPA spends latency to buy
capacity (and hence throughput headroom).** Turning it off refunds the latency
and hands back the capacity.

---

## 4. The p99 regression, explained

Two requests, and only two, blew past 10 s:

| ttft | input | note |
|---|---|---|
| 12,326 ms | 148,710 | the same shape took **3,339 ms** under DPA-on |
| 11,397 ms | 77,117 | a *mid-sized* prompt — the same run served 215,664 tokens in 6,098 ms |

A 77K prompt taking 11.4 s while a 215K prompt takes 6.1 s in the same run is
not a compute-scaling story. Both legs were checked for faults over the run
window and both are clean (§5, trap 3), so these are not crashes or retries.

**The cause is the capacity collapse from §3, not a mis-tuned cache.** An
earlier draft of this analysis blamed the boot warning

```
HiCache host KV pool (356,160 tokens) is smaller than the device pool
(3,263,680 tokens); L2 cache effectiveness is reduced.
```

on an unre-tuned `--hicache-size`. **That explanation is wrong and is corrected
here.** The host pool is **356,160 tokens in both runs**, and the same warning
was already printed under DPA-on against a 2,829,952-token device pool. Nothing
about the host tier changed.

What changed is the aggregate GPU KV: **22.6 M → 3.3 M tokens (−85.6 %)**. The
working set that previously lived entirely on the GPU no longer fits, so blocks
are evicted (8.5×) and occasionally have to be fetched back through the host
tier. A request that lands on such a fetch pays seconds; a request whose prefix
is still resident does not. That is exactly the observed shape — 77 requests
under 1 s, and two that stall.

So DPA-off makes the common case ~2× faster and exposes a rare slow path that
DPA's larger aggregate cache had been hiding. **This is intrinsic to the trade,
not a configuration bug** — though raising `--hicache-size` (16 GB today) would
widen the tier absorbing the spill and should compress the tail. Untested.

Trimming the two outliers, mean TTFT is 1,601 ms vs **866 ms** = **1.85×**.

---

## 5. Traps hit (all three would have produced wrong or unattributable numbers)

**1. `--ep-size` was scoped to the DPA branch.** In `glm52_leg.sh`,
`--ep-size "$TP"` lived inside `if [ "$DPA" = "1" ]`. Running `DPA=0` therefore
dropped the flag and sglang resolved `ep_size` **8 → 1**, collapsing MoE
expert-parallelism at the same moment as attention DP. Verified offline
(`no --ep-size flag -> ep_size = 1`) *before* launching. Fixed by
`EP_DECOUPLE`: hoist the flag out of the branch. Verified the DPA-on command
line is unchanged by the edit.

**2. `DPA` was hardcoded in `start_leg.sh`.** Line 86 passed a literal `DPA=1`
into the container env, so the outer `DPA=0` was **silently ignored** — the leg
came up with `--enable-dp-attention` anyway. Caught by reading back the live
process command line after launch instead of trusting the launcher's echo.
Fixed by `DPA_PASSTHROUGH` (`DPA="${DPA:-1}"`), default unchanged.

**3. A fault grep matched a boot line.** Scanning today's window on the prefill
leg returned 1 hit, which looked like a real fault. It is the disaggregation
warmup line containing `'num_retractions': 0` matching the pattern `retract`.
Same class as the `abort_on_priority_when_disabled` false positive from the
DPA-on run. **Real fault count: 0 on both legs.**

---

## 6. Verdict and honest caveats

**Turning off prefill DP-attention roughly halves the prefill latency floor**
(median paired speedup 2.01×, 49/50 requests faster) at unchanged MoE
parallelism, unchanged request shape, and an untouched decode leg. E2E p50
improves 9 % and now sits comfortably inside the 4,500 ms SLA.

**It buys that latency by giving up 85.6 % of aggregate KV capacity.** DPA is a
latency-for-capacity trade; this run measures the price of the trade in both
currencies at once. Whether the trade is worth taking is a function of load,
and this run deliberately sits at the load point (concurrency 1) where capacity
is worth least — so it flatters DPA-off by construction.

Caveats, in order of how much they should temper the conclusion:

1. **The tail got worse, and it is the cost side of the same trade.** p99 TTFT
   +150 %, driven by two requests, caused by the working set no longer fitting
   in a 3.3 M-token aggregate pool (evictions 8.5×). Raising `--hicache-size`
   should compress it; untested.
2. **This is concurrency 1 only.** DP-attention exists to serve *concurrent*
   requests across ranks; at one request in flight, 7 of 8 DP ranks are idle by
   construction, which is exactly the regime where DPA should look worst. **This
   result must not be read as "DPA is bad" — it says the DPA-on latency floor
   carries a ~2× per-request overhead that only pays for itself under load.**
   Where the crossover is, is unmeasured.
3. **n = 105, so p99 is ~1 request.** Intrinsic to concurrency 1.
4. **The +7 % TPOT is unexplained.** One run vs one run; cause recorded as open.
5. **kv-aware routing on the prefill leg is now a no-op** (`dp_size=None` →
   1 routing target instead of 8). Inherent to DPA-off, not a separate variable,
   and near-harmless at concurrency 1 — but it means this configuration is not
   the one to use for any kv-aware routing study.

## Reproduce

```bash
# analysis of this run
python3 scripts/solo_analyze.py results/solo_dpaoff/metrics.jsonl.gz --phase sustain
# the comparison tables above
python3 scripts/compare_dpa.py \
    ../solo.glm52.latencyfloor.packup_20260802/results/metrics.jsonl.gz \
    results/solo_dpaoff/metrics.jsonl.gz
```

Deployment: relaunch the prefill leg only, with the two script fixes applied —
`ROLE=prefill MY_IP=10.2.122.10 ETCD_IP=10.2.122.10 MTP=0 TAG=p7 DPA=0 bash scripts/start_leg.sh`.
Do **not** restart decode: it carries the runtime-applied `GLM52_P1V3` patch,
without which it dies within minutes on this request shape.
