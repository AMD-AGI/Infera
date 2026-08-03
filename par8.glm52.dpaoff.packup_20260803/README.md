# PAR8 — Case A request shape at reduced load, prefill DP-attention OFF

**Ran 2026-08-03 09:28:21 – 10:35:06 UTC** (4,005 s = 66.8 min, one full cycle:
ramp 400 + sustain 3600 + drain 5). Two-node PD on **chi2835 (prefill) +
chi2879 (decode)**.

## What this run is

Case A's **request profile held byte-identical**, with exactly **three** offered-load
knobs changed, plus the prefill leg running **pure TP8 (DP-attention OFF)**:

| knob | Case A | par8 |
|---|---|---|
| `initial_sessions` | 32 | **8** |
| `max_sessions` | 128 | **32** |
| `max_inflight` | 48 | **24** |
| prefill DP-attention | ON (dp8) | **OFF (TP8)** |
| prefill `--chunked-prefill-size` | 65536 | **16384** |

A machine diff of the two YAMLs (`spec/`) confirms **only** those three fields
differ, plus the tokenizer path. Everything else — the percentile triples, the
0.89 cache construction, `new_session_rate 0.10`, seed 1337, the 400/3600 window
— is Case A verbatim.

> The chunk-size change is **not** a free parameter. sglang divides
> `--chunked-prefill-size` by `dp_size` *only* when DP-attention is on
> (`server_args.py:4902`). Reusing Case A's 65536 under DPA-off would have made
> per-forward prefill work **8× larger**, not equal. 16384 was chosen to hold
> per-forward work near the measured sweet spot. See `notes.md`.

## Headline result

| | par8 (this run) | Case A full (DPA on, 32/48) |
|---|---|---|
| requests | 2,884 sent / **2,850 completed** | 2,988 / 2,952 |
| success | **98.8 %** | 98.8 % |
| duration | 4,005 s | 4,006 s |
| **TTFT p50** | **1,353 ms** | 4,378 ms |
| **TTFT p90** | **4,948 ms** | 8,940 ms |
| TTFT p99 | 10,674 ms | 16,492 ms |
| TPOT p50 | **14.8 ms** | *not recorded* (see below) |
| E2E p50 / p90 | 7.4 s / 36.8 s | *not recorded* |
| cache hit | **88.8 %** (ideal 89.0, eff. 99.8 %) | 89.2 % |
| accept len (per-request) | 2.02 | 2.02 |
| in-flight max / mean | **22 / 12.1** | 30 / 15.4 |
| sessions active max | 32 (capped) | 44 |

### ⚠ The TTFT gap is NOT attributable to DP-attention

**This run changed two things at once** — offered load *and* prefill DPA. The
−69 % TTFT p50 is the combined effect and **cannot be split between them from
this data**. Anyone quoting "DPA-off is 3× faster" from this kit is misreading it.

What *would* separate them: par8's exact YAML re-run with prefill DPA **on**
(and chunk back to 65536). That run does not exist yet.

The one place the variable *is* isolated is the solo pair
(`../solo.glm52.latencyfloor.packup_20260802` vs
`../solo.glm52.dpaoff.packup_20260802`), at concurrency 1: TTFT p50
1,165 → 805 ms. That is a controlled −31 %, at a load 20× lower than this run.

## The five features, each with a positive signal

| feature | signal | result |
|---|---|---|
| **PD** | `/v1/workers` both disagg modes | prefill + decode, both `active` |
| **DPA** | live `scheduler_DP*` count | prefill **0** (TP8, by design) / decode **8/8** |
| **kv-aware** | per-rank pick distribution, Rust router policy log | decode **all 8 ranks**, 10.6–14.3 % (2,884 picks). Prefill **1 target** — see below |
| **MTP** | `accept len` in decode log | engine mean **2.76** (n=1,594) / per-request **2.02**; healthy band, not degenerate 4.00 |
| **kvd** | `statctl` deltas | prefill **+57,870 sets / +10,193 evictions / 0 gets**; decode all-zero **by design** |

### Two structural findings this run pinned down

**1. kv-aware routing did no cache steering in this run — on either leg.**
Not a misconfiguration; a consequence of the deployment:

- **Prefill**: DPA-off ⇒ `dp_size=1` ⇒ `expand_targets()` yields **one** target.
  Nothing to choose between. (2,884/2,884 picks to the same target.)
- **Decode**: `cache_hits` **0 on all 2,884 picks**. The decode leg runs
  `ChunkCache`, not a radix tree, so its router-side KV view is permanently
  empty and the cost function's overlap term cancels — routing degenerates to
  pure least-loaded. Verified directly: subscribing to the decode engine's
  kv-event socket yields **0 messages in 15 s** while prefill's yields 30.

**2. MTP and decode-side radix cache are mutually exclusive upstream — so
every turn re-transfers the full prompt KV.** sglang hard-raises on
`--disaggregation-decode-enable-radix-cache` + `--speculative-algorithm`. With
decode radix off, `decode_prefix_len` is always 0, and the prefill→decode
transfer window is the **entire** prompt. A prefill-side cache hit saves
*compute*, not *bytes*. Full code trace and upstream research in `notes.md`.

## Navigate

| file | what |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | ordered, copy-pasteable reproduction |
| [`environment.md`](environment.md) | nodes, image digests, the exact deployment delta |
| [`analysis/`](analysis/) | every number: SLI ladders, YAML-vs-measured, routing |
| [`notes.md`](notes.md) | traps, the node eviction, the two structural findings |
| [`spec/`](spec/) | `par8.yaml` + the Case A parent it derives from |
| [`scripts/`](scripts/) | every script that ran, verbatim |
| [`patches/`](patches/) | CHUNK_PASSTHROUGH + the inherited three |
| [`results/`](results/) | raw metrics, kvd counters, router log, env snapshots |
| [`logs/`](logs/) | driver + both engine logs (gzipped) |
