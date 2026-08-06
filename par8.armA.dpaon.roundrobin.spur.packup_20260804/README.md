# par8 arm A — DP-attention ON + **round-robin** routing, on **spur**

**Ran 2026-08-04 03:16:37 – 04:23:24 UTC** (4,006.7 s = 66.8 min, one full cycle:
ramp 400 + sustain 3,600 + drain). Two-node PD on the **crsuse2-m2m** (spur)
cluster, against the **final-PR branch** image.

## What this run is

One arm of a two-arm pair. Both arms run the **par8 workload byte-identical**
(only the tokenizer path is retargeted to this cluster's filesystem); they differ
in the deployment:

| | **arm A (this kit)** | arm B (`../par8.armB.dpaoff.kvaware.spur.packup_20260804/`) |
|---|---|---|
| prefill DP-attention | **on (dp8)** | off (pure TP8) |
| router policy | **round-robin** | kv-aware (pw 20.0 / dw 2.0) |
| `--mem-fraction-static` prefill | **0.70** — forced, see below | 0.70 |

**This arm took two attempts.** The first (`GMU=0.80`) aborted 60 s into its
measurement window with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. That was not a
setback to be papered over — it is a real result about this deployment. See
[§ The 0.80 crash](#the-080-crash-round-robin-costs-activation-memory).

> **Caveat on that crash: its log did not survive.** The GMU-0.70 restart reused
> the same log tag and truncated it in place. The crash is reported from
> first-hand reading at the time and is **not independently checkable from this
> kit**; `logs/README.md` states exactly what was lost and what still corroborates
> the conclusion.

## Result at a glance

| | measured |
|---|---|
| duration | **4,006.7 s** |
| requests | 2,323 sent / **2,289 completed** |
| **success rate** | **0.9854** |
| errors | **17** — all client-side 240 s timeouts; **0** HTTP failures |
| **TTFT p50 / p90** (sustain) | **3,504 / 7,079 ms** |
| TTFT p99 (sustain) | 16,602 ms |
| TPOT p50 / p90 | 16.5 / 21.5 ms |
| **cache hit** | **88.19 %** actual vs 88.99 % ideal — efficiency **99.11 %** |
| **MTP acceptance** | **2.720** engine-measured (n = 14,399) |
| **kvd** | `entries 0 → 47,975`, `host_bytes 0 → 84.6 GB`, **14,864 gets, 14,864 hits** |
| engine faults | **0** on both legs across the full window |

### Against the workload's own SLA block

| bar | source | measured | verdict |
|---|---|---|---|
| success rate ≥ 0.97 | `par8.yaml` `sla.success_rate` | **0.9854** | **PASS** |
| TTFT p90 < 30,000 ms | `par8.yaml` `sla.ttft_p90_ms` | **7,079 ms** | **PASS** (4.2× margin) |
| e2e p50 < 4,500 ms | `par8.yaml` `sla.e2e_p50_ms` | **not recorded** | this driver's `summary.json` emits no e2e percentile object |
| full cycle completes | the instruction | 4,006.7 s | **PASS** |
| in-flight not pinned at cap | method | touched 24 on **31 of 3,820** ticks (0.8 %) | **PASS** — brushed, not pinned |

## The headline: round-robin actually spreads, and it is measurable two ways

This is what the arm was for. Prefill batches served, **per DP rank**:

| routing | DP0 | DP1 | DP2 | DP3 | DP4 | DP5 | DP6 | DP7 |
|---|---|---|---|---|---|---|---|---|
| kv-aware (earlier acceptance run, same branch/image) | 1,405 | 1,497 | **1** | **1** | **1** | **1** | **1** | **1** |
| **round-robin (this run)** | **515** | **495** | **508** | **493** | **508** | **502** | **475** | **499** |

±4 % around the mean across all 8 ranks, against a distribution where six of eight
ranks previously did **one batch each for an entire run**.

The router's own pick log agrees, and shows the two pools rotating
**independently** — 290 or 291 picks to every one of the 16 targets:

```
291 picked=10.245.156.167:30000#dp0      291 picked=10.245.152.164:30001#dp0
291 picked=10.245.156.167:30000#dp1      291 picked=10.245.152.164:30001#dp1
...                                      ...
290 picked=10.245.156.167:30000#dp7      290 picked=10.245.152.164:30001#dp7
```

That per-pool independence is deliberate: a single shared counter would advance
by two per PD request and pin every prefill pick to the same target
(`infera/router/policy/round_robin.py:26-32`).

**This settles the open question** from the earlier acceptance run, where prefill
`cache_view_size` showed dp0/dp1 populated and dp2..dp7 at zero and two
hypotheses were live: (A) the ranks are not computing at all, or (B) they compute
but their KV events never reach the router. **Neither.** The ranks are alive and
take work the moment the router sends it; kv-aware simply concentrates on ranks
that already hold the prefix, and the concentration is self-reinforcing — no
traffic → empty cache view → never the cheapest candidate → no traffic.

## The 0.80 crash: round-robin costs activation memory

The first attempt at this arm ran the same `--mem-fraction-static 0.80` every
DPA-on leg on this stack had used. It aborted after 60 s:

```
:0:rocdevice.cpp:3582 … HSA_STATUS_ERROR_OUT_OF_RESOURCES … Available Free mem : 52 MB
Fatal Python error: Aborted
```

with `token usage: 0.05`, `#running-req: 0` — **the KV pool is empty**, so this is
activation memory, not KV exhaustion.

The mechanism is the spreading above. In the seconds before the abort, **4–5 DP
ranks were prefilling concurrently**; under kv-aware only 1–2 ever are. Each holds
its own chunk's activations, and 56 GB outside the static reservation does not fit
the peak.

| | attempt 1 (`GMU 0.80`) | **attempt 2 (`GMU 0.70`)** |
|---|---|---|
| avail mem after pool | 56.4 GB | **85.2 GB** (+51 %) |
| `max_total_num_tokens` | 2,939,264 | 2,387,200 (−19 % KV pool) |
| outcome | abort at t+60 s | **4,006.7 s, 0 faults** |

**So the routing policy and an engine memory knob are coupled.** Round-robin on a
DPA prefill leg needs a lower `mem-fraction-static` than kv-aware does — a
relationship not documented by either component. Cost: 19 % of the KV pool, which
at a peak `token usage` of ~0.05 was never the binding resource.

> Direction matters and is counter-intuitive: **prefill** activation OOM is fixed
> by **LOWERING** `mem-fraction-static`, the opposite of the decode-side retract
> fix. Diagnose by phase — decode retract → raise; prefill `HSA_STATUS_ERROR` at
> low token usage → lower. This is the fourth independent confirmation on this
> stack (Case A 0.88→0.80; nodpa 0.80→0.70; arm B 0.80→0.70; this arm 0.80→0.70).

## The five features, each with a positive signal

| feature | signal | result |
|---|---|---|
| **PD** | router `/health` + registered workers | `{"status":"ok","active_workers":2}`, both legs registered with `kv_events` endpoints |
| **DPA** | resolved `server_args` + per-rank batches | **both** legs `dp_size=8`; prefill 475–515 batches/rank, decode all 8 active |
| **round-robin** | router pick log | 16 targets, **290–291 picks each**, prefill and decode pools rotating independently |
| **MTP** | `accept len` in the decode log | **2.720** over 14,399 samples — healthy band, not the degenerate 4.00 |
| **kvd** | `statctl` delta, before → after | `entries 0 → 47,975`, `host_bytes 0 → 84.6 GB`, `long_bytes 0 → 297 GB`, **gets 14,864 / hits 14,864** |

**The kvd row is stronger here than anywhere else in this series.** Earlier runs
proved write-back only (`gets_total = 0` — the GPU radix served everything). This
run reads **14,864 gets with 14,864 hits**: L3 host offload was actually read back,
at a 100 % hit rate, 121,835 evictions in. That is the tier doing its job under
pressure, not merely being connected.

**RDMA is real, not silently degraded**: `MC_FORCE_TCP` and `GID is NULL` both
count **0** in both leg logs. TCP fallback works and looks fine; it is merely slow.

## Comparing the two arms — read the caveat first

| | **arm A** (DPA on + round-robin) | **arm B** (DPA off + kv-aware) |
|---|---|---|
| duration | 4,006.7 s | 4,007.7 s |
| sent / completed | 2,323 / 2,289 | 2,907 / 2,861 |
| success rate | **0.9854** | 0.9842 |
| **QPS** | 0.580 | **0.725** |
| **TTFT p50** (sustain) | 3,504 ms | **2,239 ms** |
| **TTFT p90** (sustain) | 7,079 ms | **6,389 ms** |
| TTFT p99 (sustain) | 16,602 ms | **10,606 ms** |
| TPOT p50 | 16.5 ms | 16.0 ms |
| cache actual / efficiency | 0.8819 / 99.11 % | **0.8865 / 99.62 %** |
| MTP accept len | 2.720 | 2.789 |

> **The two arms differ in TWO variables** — prefill DP-attention *and* routing
> policy — because that is what the operator specified. **No single-variable
> attribution is available from this pair.** Arm B is faster on every latency
> percentile and carries 25 % more throughput, but that gap is the *combined*
> effect of (DPA off) and (kv-aware), and this data cannot split it.
>
> One prior result bears on the DPA half in isolation: the spur nodpa kit
> (`../agenticbench.mtp.nodpa.packup_20260802/`) measured DP-attention costing
> **1.65–1.93× TTFT at concurrency 1** in a controlled single-variable comparison.
> That direction is consistent with the gap here, but the load is 24× lower there
> and nothing licenses transferring the magnitude.

What *would* separate them: par8 re-run with DPA on + kv-aware, and DPA off +
round-robin — the two missing cells of the 2×2. Neither exists.

## Navigate

| file | what |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | ordered, copy-pasteable reproduction — **including the GMU trap** |
| [`environment.md`](environment.md) | nodes, image digests, git SHA, resolved server args, full env snapshots |
| [`notes.md`](notes.md) | the crash, the two leg-script fixes, gotchas, wrong turns |
| [`analysis/`](analysis/) | routing distribution, per-phase ladder, feature evidence, **arm A vs arm B** |
| [`spec/`](spec/) | `par8.yaml` as run + the Case A parent it derives from |
| [`scripts/`](scripts/) | every script that ran, verbatim |
| [`patches/`](patches/) | the leg-script edits, as diffs with what/why/how/context |
| [`results/`](results/) | `summary.json`, metrics, kvd before/after, router log + metrics |
| [`logs/`](logs/) | both engine logs, driver, build, **and attempt 1's crash log** |
| [`env/`](env/) | `collect_env.sh` snapshots from both live nodes |
