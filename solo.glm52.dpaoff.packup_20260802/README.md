# Prefill DP-attention OFF — the latency/capacity trade, measured in both currencies

**Ran:** 2026-08-02, 05:46–06:23 UTC (400 s ramp + 1,800 s measured window).
**Status:** **PASS.** 145 requests, **105 in the measured window, 0 errors**,
concurrency verifiably pinned at 1.

Companion to `../solo.glm52.latencyfloor.packup_20260802/` (the DPA-**on**
baseline, 2026-08-01 16:05). Same workload file, same seed, same driver patch,
same decode-leg **process** — only the prefill leg's attention sharding changed.

## Goal

The baseline kit established a latency floor with DP-attention on. This run asks:
**does turning off prefill DP-attention lower that floor, and what does it cost?**

**Spec:** `spec/solo.yaml` (byte-identical to the baseline's). Parent mission in
`spec/mission.kv.liying.mtp.bench.md`.

**Success criteria** — exploratory, so judged on measurement soundness:

| criterion | result |
|---|---|
| concurrency genuinely 1, provably | ✅ `in_flight ∈ {0,1}` across 1,796 rows; `Hit max_inflight` = 0 |
| request shape identical to the baseline | ✅ input p50 77,117 vs 77,107 (+0.0 %) |
| **exactly one variable changed** | ✅ verified in resolved server args (see below) |
| clean run | ✅ 0 driver errors, 0 engine faults on either leg in-window |

## Result — DP-attention spends latency to buy capacity

Both sides of that trade are measured here, and they are the same phenomenon
seen from two directions:

| | DPA-on | DPA-off | |
|---|---|---|---|
| **TTFT p50** | 1,563 ms | **778 ms** | latency **refunded** (2.0×) |
| **aggregate KV** | 22,639,616 tok | **3,263,680 tok** | capacity **surrendered** (−85.6 %) |
| kvd gets | +0 | **+432** (100 % hits) | the spill tier now does real work |
| evictions | +1,206 | **+10,249** | 8.5× — working set no longer fits |

### The latency side

| TTFT | DPA-on | DPA-off | change |
|---|---|---|---|
| min | 689 ms | **327 ms** | −52.6 % |
| **p50** | 1,563 ms | **778 ms** | **−50.2 %** |
| **p90** | 2,899 ms | **1,536 ms** | **−47.0 %** |
| p95 | 3,336 ms | 2,166 ms | −35.1 % |
| **p99** | 4,472 ms | **11,185 ms** | **+150.1 %** ⚠️ see below |

**The cleanest evidence — paired on identical requests.** Both runs share seed
1337, so 50 `(input, output)` shapes occur in both. Comparing each request
against *itself*:

```
DPA-off faster on   49 / 50 paired shapes
MEDIAN PAIRED SPEEDUP        2.01x
speedups cluster             1.7x - 2.1x
```

Reproduce with `python3 scripts/compare_dpa.py` (no arguments — the baseline is
bundled at `results/baseline_dpaon_metrics.jsonl.gz`).

Distribution shift is the honest summary at n≈100: **TTFT < 1 s on 26 → 77
requests; TTFT > 2 s on 25 → 7.**

### The capacity side, and why p99 got worse

Under DPA each of the 8 DP ranks runs its own scheduler owning a **distinct** KV
shard. Under pure TP there is **one** scheduler and one pool, *replicated*
across TP ranks. Read straight out of the boot logs:

```
DPA-on   [DP0 TP0 EP0] max_total_num_tokens=2829952  max_running_requests=256   <- 2048/8, per-rank
DPA-off  [TP0 EP0]     max_total_num_tokens=3263680  max_running_requests=2048  <- ONE scheduler
```

Per-rank KV *grew* 15.3 % (attention weights shard, freeing memory) — but
aggregate fell **85.6 %**. That is what drives eviction, and it is why the two
TTFT outliers exist: a request whose prefix is still resident is fast, one that
must be re-fetched through the host tier stalls for seconds. Both outliers are
cache stalls, **not** compute regressions and **not** faults (both legs clean).

> An earlier draft blamed an unre-tuned `--hicache-size`. **That was wrong and is
> corrected in `analysis/`**: the host pool is 356,160 tokens in *both* runs and
> the same warning appears in the baseline. Nothing about the host tier changed.

### Everything else

| | DPA-on | DPA-off | |
|---|---|---|---|
| E2E p50 | 4,124 ms | **3,751 ms** | −9.0 %; now well inside `sla.e2e_p50_ms: 4500` |
| TPOT p50 | 10.68 ms | 11.17 ms | +4.5 % — **cause open**, see caveats |
| MTP acceptance | 2.069 | 2.028 | −2.0 % (decode untouched) |
| cache hit | 88.95 % | 88.83 % | flat |

## ⚠️ What this kit depends on

**Four patches, two of them new here.** See `patches/README.md`. The two new
ones share a property worth internalising: **neither failure is visible at
runtime** — both yield a leg that boots, serves, and produces a clean
105-sample run of the *wrong deployment*.

| | without it |
|---|---|
| `0004` `GLM52_P1V3` (inherited, engine) | decode leg **crashes** in minutes |
| `0005` `SOLO_M1` (inherited, driver) | **no E2E / TPOT ladders** |
| `0006` `EP_DECOUPLE` (**new**) | `ep_size` collapses 8→1 — **two-variable run** |
| `0007` `DPA_PASSTHROUGH` (**new**) | `DPA=0` **silently ignored** |

## The single-variable claim, verified before the window was spent

| | dp_size | **ep_size** | chunked_prefill (resolved) | max_prefill | aiter AR |
|---|---|---|---|---|---|
| DPA-ON | 8 | 8 | 8192 | 16384 | False |
| DPA-OFF | **1** | **8** | 8192 | 16384 | False |

MoE expert-parallelism unchanged; per-forward-pass prefill work identical
(sglang divides `chunked_prefill_size` by `dp_size`, so 65536÷8 and 8192÷1 both
give 8192).

## Navigate

| file | read it for |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | **the ordered, copy-pasteable reproduction** |
| [`analysis/solo_latency_dpaoff.md`](analysis/solo_latency_dpaoff.md) | **every ladder, the mechanism, every caveat** |
| [`notes.md`](notes.md) | the three traps, and how each was caught |
| [`environment.md`](environment.md) | hardware, image, SHAs, what was and was not restarted |
| [`patches/README.md`](patches/README.md) | all four patches, why each is load-bearing |
| [`results/`](results/) | this run + the bundled DPA-on baseline |

## Honest caveats

1. **Concurrency 1 flatters DPA-off by construction.** DP-attention exists to
   serve *concurrent* requests across ranks; at one request in flight, 7 of 8
   ranks are idle and the capacity DPA buys is worth least. **This is not
   "DPA is bad"** — it prices the trade at the single load point where the
   capacity side has almost no value. The crossover is unmeasured.
2. **The tail regression is the cost side of the same trade**, not a bug. It may
   compress with a larger `--hicache-size` (16 GB today); untested.
3. **n = 105, so p99 rests on ~1 request.** Intrinsic to concurrency 1.
4. **The +7 % TPOT is unexplained.** Decode was never restarted (same PID, same
   `accept len` health). Plausibly denser decode batching from faster prefill —
   one run vs one run, recorded as open, not asserted.
5. **kv-aware routing on prefill is now a no-op** (`dp_size=None` → 1 routing
   target instead of 8). Inherent to DPA-off; near-harmless at concurrency 1,
   but this config is not the one for a kv-aware routing study.
6. **Not a capacity measurement.** 0.06 qps sustained.
