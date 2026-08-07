# Results — GLM-5.2 mix, agentic Case-A UNDER LOAD

One run: `2026-08-06-11-56-00`, `initial_sessions 8 / max_inflight 16 /
max_sessions 24`. Everything below is re-derivable offline from the files in
this directory — no cluster access. Commands are in
[`../REPRODUCE.md`](../REPRODUCE.md) §6.

Two sources are used and they are **not interchangeable**:

- `load/summary.json` — the driver's own whole-run block. Percentiles use the
  driver's index convention `sorted[int(q*n)]`.
- `analyze_solo.py` over `load/metrics.jsonl.gz` — **sustain phase only**, and
  the **only** source that has an E2E column at all. Convention
  `sorted[round(q*(n-1))]`.

Where both report the same metric they differ by up to ~4 % on the p90 (TTFT p90
7564.6 whole-run vs 7295.8 sustain-only — different populations *and* different
conventions). Do not mix them in one row. This is the same convention gap
established by test in `solo.glm52.mix.packup_20260806/notes.md` §10.

---

## 1. Whole run — `load/summary.json`

| field | value |
|---|---|
| duration | **4006.2 s** |
| requests sent | **1804** |
| requests completed | **1755** |
| errors | **35** |
| success rate | **0.9728** |
| actual average qps | **0.4503** (emergent — closed loop) |

### Latency (whole run, driver convention)

| metric | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| TTFT ms | 4779.9 | **4272.0** | **7564.6** | **16873.5** |
| TPOT ms | 28.80 | **25.31** | **42.66** | **95.85** |

### Request shape actually produced

| metric | p50 | p90 | p99 | mean |
|---|---|---|---|---|
| prompt tokens | **75,345** | **155,462** | **222,770** | 86,435 |
| generation tokens | **317** | **2,443** | **7,478** | 905.8 |

The p50/p90 land on Case A's 74K/155K targets; the p99 comes in at 222,770
against a 235,000 target — this is a *sampled* distribution here (unlike Phase 2,
where the triples were degenerate on purpose), so the realised p99 is an order
statistic of 1,755 draws, not a pinned value.

### Cache

| field | value |
|---|---|
| ideal hit rate | **0.8899** |
| actual hit rate | **0.8806** |
| efficiency (actual/ideal) | **0.9896** |
| eviction rate | **0.0104** |
| total input tokens | 151,694,265 |
| cached | 133,581,952 |
| uncached (real prefill work) | 18,112,313 |
| evicted | 1,405,986 |

Actual tracks ideal to within 1 %. The workload nests every request inside one
shared prefix, so this measures cache *accounting* and one hot radix path — it
does not exercise tiering under eviction pressure (the YAML says so itself).

---

## 2. Phase table — the driver's own output

Copied from the driver's console block (`logs/agentic_load.log.gz`, tail).
`drain` is a single request and reports `n/a` for rates.

| phase | dur(s) | reqs | qps | input TPM | cached TPM | uncached TPM | visible TPM | reason TPM | cache% | TTFT p50 | TTFT p90 | TPOT p50 | TPOT p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ramp | 400 | 117 | 0.29 | 1,538,151 | 1,220,006 | 318,145 | 6,851 | 4,703 | 79.3% | 4329.2 ms | 18799.1 ms | 26.2 ms | 70.5 ms |
| sustain | 3600 | 1637 | 0.45 | 2,356,388 | 2,089,970 | 266,418 | 18,111 | 7,087 | 88.7% | 4266.8 ms | 7349.1 ms | 25.3 ms | 41.7 ms |

Per-GPU: divide TPM by 8.

**The ramp is a warm-up exclusion window, not a load ramp** (`ramp_duration: 400`
in the YAML, whose own comment says so). Its 79.3 % cache hit against sustain's
88.7 %, and its TTFT p90 of 18.8 s against sustain's 7.3 s, is that window doing
its job: the 231K-token shared prefix becomes resident during it. Nothing from
ramp enters the reported sustain numbers.

### Other driver-reported aggregates

| item | value |
|---|---|
| peak prefill throughput | **561,399 tok/s** total = **70,175 tok/s/GPU** |
| average context throughput | **284,007 TPM/GPU** (4,733 tok/s/GPU) |
| generation throughput | **68.2 tok/s** (MTP compensated) |
| inter-arrival (response + think) | mean 12.8 s, p50 4.1 s, p90 27.4 s, p99 152.9 s |
| sessions | total **24** (initial 8, +228 rate-based); retired 6; **abandoned 0** |
| session lifetimes | min 17 s, max 3397 s, mean 686 s |
| final prefix size | 5,603 tokens (min = max = mean) |

---

## 3. Sustain-phase per-request — `analyze_solo.py`, n = 1637

**This is the only table with an E2E column.** Reproduced while assembling this
packup, from the gzipped `metrics.jsonl` shipped here:

```
===== load — sustain phase, n=1637 requests =====
  prompt tokens : p50 75,450   (n=1637)
  gen tokens    : p50 320   mean 923.6
  cache hit     : mean 0.8883
  TTFT  ms      : p50    4266.8  p90    7295.8  p99   12335.5  mean    4604.5  n=1637
  E2E   ms      : p50   13931.9  p90   70313.9  p99  188236.2  mean   28353.5  n=1637
  TPOT  ms      : p50      25.3  p90      41.6  p99      77.3  mean      27.6  n=1637
```

| metric | p50 | p90 | p99 | mean |
|---|---|---|---|---|
| TTFT ms | 4266.8 | 7295.8 | 12335.5 | 4604.5 |
| **E2E ms** | **13931.9** | **70313.9** | **188236.2** | **28353.5** |
| TPOT ms | 25.3 | 41.6 | 77.3 | 27.6 |

The E2E spread is wide — p90 is 5.0× the p50 and p99 is 13.5× it. **No
explanation is offered**; see [`../notes.md`](../notes.md) §4 for what would
settle it and for the one first-hand fact that bounds it (the p99 sits under the
driver's own 240 s client budget, and the maximum completed request is 239.0 s).

`analyze_solo.py` drops `new_tpots` entries equal to 0.0 — that value is the
SOLO_M1 marker for "the driver filtered this sample" (gen_len ≤ 1 or
gen_time < 50 ms), not a zero-latency token.

---

## 4. Saturation — both caps were hit

Measured over the **3588 sustain ticks** in `metrics.jsonl` (the run has 3992
ticks total: 399 ramp / 3588 sustain / 5 drain).

| counter | cap | max | mean | **ticks at the cap** |
|---|---|---|---|---|
| `in_flight` | 16 | 16 | 15.30 | **2588 / 3588 = 72.1 %** |
| `num_sessions_active` | 24 | 24 | 22.42 | **1685 / 3588 = 47.0 %** |

Full distributions:

```
in_flight           {6:2, 7:2, 8:5, 9:17, 10:32, 11:92, 12:94, 13:139, 14:239, 15:378, 16:2588}
num_sessions_active {14:3, 15:26, 16:68, 17:91, 18:112, 19:147, 20:187, 21:207, 22:369, 23:693, 24:1685}
```

`in_flight` never fell below 6 and `num_sessions_active` never below 14 — the
system was continuously loaded, not bursty.

The driver also printed its own warning, **once**, at elapsed 406.6 s (the
first tick of sustain):

```
WARNING: Hit max_inflight (16) - sessions are being throttled; offered load is capped
```

**`max_inflight` was the binding constraint** — it is at its cap on 72.1 % of
ticks against `max_sessions`'s 47.0 %.

The workload file states the interpretation itself, in its own comment on
`max_sessions`:

> *"Hitting it means sessions are dying slower than they are born, i.e. the
> server is saturated — treat it as a failure signal."*

Both caps were hit. **This run is at saturation and the throughput numbers are
capacity-limited by the offered-load caps, not a free-running measurement of what
the deployment can do.** No claim is made here about *which* phase (prefill
compute, decode compute, or cache) is the limiter — that was not measured. See
[`../notes.md`](../notes.md) §3.

---

## 5. The 35 errors — all one class, with per-error log lines

**Corrected while assembling this packup.** It was previously believed the driver
recorded no per-error detail. It does. See [`../notes.md`](../notes.md) §5.

| observation | evidence |
|---|---|
| all 35 errors are **client-side request timeouts** | 35 `Request N timed out` lines in `logs/agentic_load.log.gz`, 35 unique request ids |
| **no other error class occurred** | 0 × `error:`, 0 × `failed: HTTP`, 0 × `Traceback` in the same log |
| the **engine rejected nothing** | time-scoped scan of the engine log over 11:56–13:05: **0** error/abort/reject/invalid/Traceback lines, **1802 / 1802 HTTP 200** |
| errors accumulated **gradually**, not in a burst | first at elapsed 658.8 s, then 921.5 / 1026.8 / 1149.2 / 1162.2 / 1205.4 …; the running rate sits at 0.4 % early and 1.4–1.97 % for the rest of the run |
| **no session was abandoned** | `num_sessions_abandoned` = 0 on the final tick |
| the two records agree 1:1 | each driver-log timeout pairs with exactly one `metrics.jsonl` error increment; max time offset 1.9 s, mean 1.0 s (tick period ~1 s) |

The client budget is `aiohttp.ClientTimeout(total=240)`
(`agent_throughput.py:2253`, the realistic-mode send path). The **completed**
requests bear directly on this:

| completed sustain E2E | value |
|---|---|
| p99 | 188.2 s |
| p99.9 | 228.6 s |
| **max** | **239.0 s** |
| completed with E2E > 240 s | **0** |

**What is NOT claimed.** That the timeouts were caused by the server being slow,
by the queue, by saturation, or by anything else. A timeout is the *client giving
up*; nothing measured here shows what the server was doing for those 35 requests,
and the driver discards the request on timeout without recording a server-side
correlator. See [`../notes.md`](../notes.md) §5 for the measurement that would
settle it.

---

## 6. Against the workload's stated SLA block

`sla:` in the YAML is **documentation only** — `args.sla_cfg` is parsed at
`agent_throughput.py:3351` and never consumed; nothing in the run gates on it.
The workload's own comment says so. Reported purely as a comparison against a
stated target:

| stated target | value | measured | verdict |
|---|---|---|---|
| `success_rate` ≥ 0.97 | 0.97 | **0.9728** | met, by 0.0028 |
| `ttft_p90_ms` < 30000 | 30,000 ms | **7,564.6 ms** (whole run) | met, 4.0× inside |
| `e2e_p50_ms` < 4500 | 4,500 ms | **13,931.9 ms** (sustain) | **not met — 3.1× over** |

The E2E bar is a stated target being compared against a **loaded** measurement,
and the run is at saturation (§4). Phase 2 measured the same bar unloaded at
concurrency 1 and got 5,111.1 ms — also over, by 611 ms
(`solo.glm52.mix.packup_20260806/notes.md` §6, where its provenance is recorded
as unestablished). No further interpretation is offered.

---

## 7. Feature counters after the run — the features stayed on

Read live from the still-running deployment. **Both are cumulative across all
three phases** (one unrestarted container since 07:01:26Z) and cannot be
attributed to this run alone.

**MTP acceptance**, whole engine log (07:02:13 → 13:02:48, all three phases):

```
n=51,685  p10 2.73  MEDIAN 3.55  p90 3.98  at 4.00: 4,716 (9.1 %)
```

Scoped to **this run's window only** (11:56:00 → 13:05:00) it reads:

```
n=8,918   p10 2.48  MEDIAN 3.14  p90 3.82  at 4.00: 400 (4.5 %)
```

Both are below 4.00. A median **at** 4.00 would be a failure signal (a repetition
loop the draft model predicts perfectly), not a win. The two readings differ
materially, which is exactly why every engine-log grep must be time-scoped —
`../notes.md` §7.

**kvd**, cumulative:

| counter | value | Phase-1 end reading | **delta ≈ this run + Phase 2** |
|---|---|---|---|
| entries | 71,879 | 70,676 | +1,203 |
| host bytes | 85.4 GB | 84.8 GB | — |
| L3 (long) bytes | 68.7 GB | 68.7 GB | 0 |
| gets | 69,126 | 64,741 | **+4,385** |
| sets | 486,924 | 266,289 | **+220,635** |
| hits | 67,060 | 62,697 | **+4,363** |
| misses | 2,066 | 2,044 | +22 |
| evictions | 319,447 | 151,525 | **+167,922** |

Phase-1 end readings are from `fixlen.glm52.mix.packup_20260806/notes.md` §7. The
delta spans the Phase-2 → Phase-3 boundary loosely (Phase 2's own mid-point read
was gets 69,019 / hits 66,975), so treat the subtraction as an order of magnitude,
not a measurement of this run.

Registry, checked after the run: **1 worker**, `disagg_mode: "mixed"`,
`dp_size: 8`, `kv_block_size: 64`, `status: active`.
