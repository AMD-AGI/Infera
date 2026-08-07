# GLM-5.2-MXFP4 mix deployment — agentic Case A UNDER LOAD

**Ran:** 2026-08-06, 11:56:00 → 13:02:46 UTC (one run, 4006.2 s).
**Node:** `chi2835` — single MI355X (gfx950) node, 8 GPUs, mix (aggregated) worker.
**Status:** **COMPLETE — 1804 sent / 1755 completed, 35 errors, success rate 0.9728.**
**Offered load:** `initial_sessions 8` / `max_inflight 16` / `max_sessions 24`.

This is **Phase 3 of 3** of the GLM-5.2 mix bench
([`specs/mission.mix.md`](specs/mission.mix.md), **task 3**). Phase 1 (fixed-length
sweep) is at `fixlen.glm52.mix.packup_20260806/`; Phase 2 (agentic at concurrency 1)
is at `solo.glm52.mix.packup_20260806/`.

**All three phases ran on the SAME unrestarted container** — created
`2026-08-06T07:01:26Z`, never restarted or reconfigured. This packup restates the
environment essentials so it stands alone, and cites the other two for the fuller
derivations. Everything needed to reproduce is inside this folder.

---

## Read this first: the run is at SATURATION

**Both offered-load caps were hit, and `max_inflight` was the binding one.**
Measured over the 3588 sustain ticks in the shipped `metrics.jsonl`:

| counter | cap | max | mean | **ticks at the cap** |
|---|---|---|---|---|
| `in_flight` | 16 | 16 | 15.30 | **2588 / 3588 = 72.1 %** |
| `num_sessions_active` | 24 | 24 | 22.42 | **1685 / 3588 = 47.0 %** |

The driver printed its own warning once, at the first sustain tick:

```
WARNING: Hit max_inflight (16) - sessions are being throttled; offered load is capped
```

The workload file states the interpretation in its own words, in the comment on
`max_sessions`:

> *"Hitting it means sessions are dying slower than they are born, i.e. the
> server is saturated — treat it as a failure signal."*

**So: the throughput figures below are capacity-limited by the caps, not a
free-running measurement of what this deployment can do.** Read them as "what the
deployment delivered at this offered load", never as "the deployment's ceiling".

**What is NOT claimed: which phase is the bottleneck.** Prefill compute, decode
compute and cache were not separated — nothing here measures that. See
[`notes.md`](notes.md) §3 for the two measurements that would settle it.

---

## Goal

Measure GLM-5.2-MXFP4 in a MIX (aggregated, single-instance) deployment under the
**Case-A agentic workload at load**, at the offered load the mission specifies.

**Spec:** [`specs/mission.mix.md`](specs/mission.mix.md), task 3 — *"压测 agentic
场景, init session = 8, max in flight = 16, max session = 24."*

**Success criteria.** The spec sets no numeric bar for task 3; it asks for the
measurement at that offered load. The bar is therefore completeness and
trustworthiness: the run completes, the request shape is provably Case A's, and
the features are provably still on. All three hold — see below and §7 of
[`results/RESULTS.md`](results/RESULTS.md).

The workload file separately carries an `sla:` block. It is **documentation only**
(`args.sla_cfg` is parsed at `agent_throughput.py:3351` and never consumed), so
nothing gated on it, but it is a stated target and is reported against:

| stated target | measured | verdict |
|---|---|---|
| `success_rate` ≥ 0.97 | **0.9728** | met, by 0.0028 |
| `ttft_p90_ms` < 30000 | **7,564.6 ms** (whole run) | met, 4.0× inside |
| `e2e_p50_ms` < 4500 | **13,931.9 ms** (sustain) | **not met — 3.1× over** |

The E2E line compares a stated target against a **loaded** measurement, on a run
that is at saturation. Phase 2 measured the same bar unloaded at concurrency 1 and
was also over it (5,111.1 ms); that packup records the bar's provenance as
unestablished. No further interpretation is offered here.

---

## This IS Case A's request shape — verified by machine diff

The only thing that changed relative to Case A is the *offered load* (and a path).
`specs/mix_load.yaml` is
`Optimus-AgenticBench/agent/workloads/glm52_crxx_caseA.fix.yaml` with **exactly
four** changed lines. Comments stripped, both files 27 significant lines:

```diff
-  initial_sessions:      32
+  initial_sessions:      8
   new_session_rate:      0.10
-  max_sessions:          128
-  max_inflight:          48
+  max_sessions:          24
+  max_inflight:          16
   ramp_duration:         400
   sustain_duration:      3600
   system_prompt_len:     2000
   max_prompt_tokens:     260000
-  tokenizer:             /path/to/GLM-5.2-MXFP4
+  tokenizer:             /mnt/vast/xiaobo/models/GLM-5.2-MXFP4
```

Three are the mission's load knobs; the fourth is a site path. **Everything else
is Case A verbatim** — the percentile triples (input 74K/155K/235K, output
320/3300/17000, turns 3/20/103, inter-turn delay 4/31/240 s), `cache_hit_rate 0.89`,
`new_session_rate 0.10`, `ramp 400 / sustain 3600`, `random_seed 1337`,
`max_input_tokens 260000`, and the `sla:` block.

Re-run the diff yourself: [`REPRODUCE.md`](REPRODUCE.md) §2. The shipped file is
md5 `092c7fc2a6f7ab77601d8ab63a38b618`, byte-identical to the cluster copy that ran.

---

## Result

### Whole run — `results/load/summary.json`

| field | value |
|---|---|
| duration | 4006.2 s |
| sent / completed | **1804 / 1755** |
| errors | **35** |
| success rate | **0.9728** |
| qps | 0.4503 (emergent — closed loop) |

| metric | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| TTFT ms | 4779.9 | 4272.0 | 7564.6 | 16873.5 |
| TPOT ms | 28.80 | 25.31 | 42.66 | 95.85 |
| prompt tokens | 86,435 | 75,345 | 155,462 | 222,770 |
| generation tokens | 905.8 | 317 | 2,443 | 7,478 |

Cache: ideal **0.8899**, actual **0.8806**, efficiency **0.9896**, eviction **0.0104**.

### Sustain phase, per request — `analyze_solo.py`, n = 1637

**The only source with an E2E column.** Re-derived while assembling this packup
from the shipped `metrics.jsonl.gz`, offline:

| metric | p50 | p90 | p99 | mean |
|---|---|---|---|---|
| TTFT ms | **4266.8** | 7295.8 | 12335.5 | 4604.5 |
| **E2E ms** | **13931.9** | **70313.9** | **188236.2** | 28353.5 |
| TPOT ms | **25.3** | 41.6 | 77.3 | 27.6 |

prompt p50 75,450 · gen p50 320 (mean 923.6) · cache hit mean 0.8883.

The E2E spread is very wide — p90 is **5.0×** the p50, p99 is **13.5×** it.
**No explanation is offered**; nothing measured here identifies one.
[`notes.md`](notes.md) §4 names what would settle it and records the one
first-hand bound: the maximum *completed* request is 239.0 s, and the driver's
client budget is 240 s.

### Phase table — the driver's own output

| phase | dur(s) | reqs | qps | input TPM | cached TPM | uncached TPM | visible TPM | reason TPM | cache% | TTFT p50 | TTFT p90 | TPOT p50 | TPOT p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ramp | 400 | 117 | 0.29 | 1,538,151 | 1,220,006 | 318,145 | 6,851 | 4,703 | 79.3% | 4329.2 ms | 18799.1 ms | 26.2 ms | 70.5 ms |
| sustain | 3600 | 1637 | 0.45 | 2,356,388 | 2,089,970 | 266,418 | 18,111 | 7,087 | 88.7% | 4266.8 ms | 7349.1 ms | 25.3 ms | 41.7 ms |

Per-GPU: divide TPM by 8. `ramp` is a **warm-up exclusion window, not a load
ramp** — its job is to make the 231K-token shared prefix resident, and its 79.3 %
cache hit vs sustain's 88.7 % is that job being done. Nothing from ramp enters the
sustain numbers.

Also from the driver: peak prefill **561,399 tok/s** total (**70,175 tok/s/GPU**);
average context **284,007 TPM/GPU**; generation **68.2 tok/s** (MTP compensated);
inter-arrival mean 12.8 s / p50 4.1 s / p90 27.4 s / p99 152.9 s; sessions total
**24** (initial 8, +228 rate-based), retired 6, **abandoned 0**; session lifetimes
min 17 s / max 3397 s / mean 686 s.

### The 35 errors — all client-side timeouts; the engine rejected nothing

| observation | evidence |
|---|---|
| all 35 are **client request timeouts** | 35 `Request N timed out` lines, 35 unique ids, in `logs/agentic_load.log.gz` |
| **no other error class** | 0 × `error:`, 0 × `failed: HTTP`, 0 × `Traceback` |
| **engine rejected nothing** | engine log scoped to 11:56–13:05: **0** error/abort/reject lines, **1802/1802 HTTP 200** |
| **gradual, not a burst** | first at elapsed 658.8 s; running rate 0.4 % early, 1.4–1.97 % thereafter |
| **no session abandoned** | `num_sessions_abandoned` = 0 |
| the two records agree 1:1 | each log timeout pairs with one `metrics.jsonl` increment; max offset 1.9 s |

The client budget is `aiohttp.ClientTimeout(total=240)`. Of the **completed**
requests, the maximum E2E is **239.0 s** and **none** exceeds 240 s.

**What is NOT claimed: why the 35 timed out.** A timeout is the client giving up;
nothing measured shows what the server was doing for those requests, and the
driver discards them without a server-side correlator. [`notes.md`](notes.md) §5
names the measurement that would settle it.

> **Correction carried into this packup.** It was previously believed the driver
> recorded no per-error detail and that the count in `metrics.jsonl` was the only
> record. Reading the log showed otherwise: the driver prints one line per error
> and every one of the 35 is the same class. `notes.md` §5.

---

## Feature evidence — the deployment stayed as configured

Read live from the still-running deployment after the run. Registry: **1 worker**,
`disagg_mode: "mixed"`, `dp_size: 8`, `kv_block_size: 64`, active.

**MTP acceptance.** Whole engine log, all three phases (07:02:13 → 13:02:48):
n = **51,685**, p10 2.73, **median 3.55**, p90 3.98, **9.1 %** at 4.00.
Scoped to this run's window only (11:56–13:05): n = **8,918**, p10 2.48,
**median 3.14**, p90 3.82, **4.5 %** at 4.00. Both below 4.00 — a median *at* 4.00
would be a failure signal, not a win. That the two readings differ is precisely why
every engine-log grep must be time-scoped ([`notes.md`](notes.md) §7).

**kvd**, cumulative across all three phases (one unrestarted container): 71,879
entries / 85.4 GB host / 68.7 GB L3 / gets 69,126 / sets 486,924 / **hits 67,060 /
misses 2,066** / evictions 319,447. Against Phase 1's end reading the delta is
gets +4,385, sets +220,635, hits +4,363, evictions +167,922 — spanning the
Phase-2/3 boundary loosely, so an order of magnitude, not a measurement of this
run. Full table in [`results/RESULTS.md`](results/RESULTS.md) §7.

---

## Three things deliberately left unexplained

Each is a measurement with no mechanism attached, and each names what would settle
it. A fluent story in any of these places would be worse than silence — it would
discourage running the experiment that finds the truth.

| # | observation | where |
|---|---|---|
| 1 | **which phase is the bottleneck** at saturation — prefill compute, decode compute, or cache | [`notes.md`](notes.md) §3 |
| 2 | **why 35 requests hit the client's 240 s budget** | [`notes.md`](notes.md) §5 |
| 3 | **why E2E p90/p99 (70 s / 188 s) sit so far above the 13.9 s p50** | [`notes.md`](notes.md) §4 |

---

## How to reproduce

See [`REPRODUCE.md`](REPRODUCE.md). TL;DR: bring the mix deployment up (390 s cold
start), then
`WORKLOAD=specs/mix_load.yaml TAG=load bash run_agentic.sh`, wait ~67 min, then
`analyze_solo.py`. **Offline re-analysis of the shipped results needs no cluster
access at all** — §6, and it reproduces every number in this README to the decimal.

## Folder map

| path | what |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable reproduction + the offline audit path |
| `environment.md` | exact HW/SW — pinned image digest, driver, resolved server args |
| `notes.md` | the saturation reading, the three open questions, gotchas |
| `specs/mission.mix.md` | the originating task spec — **task 3** is this packup |
| `specs/mix_load.yaml` | the workload, md5-verified against the cluster copy |
| `patches/` | the SOLO_M1 driver patch (carried from Phase 2) + how to apply it |
| `scripts/` | every script that ran, verbatim — see `scripts/README.md` |
| `results/RESULTS.md` | the full tables + how to re-derive every number |
| `results/summary.csv` | machine-readable, one row per metric |
| `results/load/` | `metadata.json`, `summary.json`, `metrics.jsonl.gz` (per-tick arrays) |
| `logs/` | driver console log, engine log **sliced to this run's window**, router, kvd (all gzipped) |
| `env/` | on-node environment snapshot + the engine's own resolved `server_args` |
