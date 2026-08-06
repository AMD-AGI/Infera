# Customer AgentX Case-A bench on infera GLM-5.2 PD — CruSoe/spur

**Ran 2026-08-04 04:00–04:53 UTC** on the CruSoe (spur / crsuse2-m2m) cluster.
Three concurrency points, 900 s each, against a two-node infera PD deployment
with **prefill DP-attention OFF**.

The benchmark is the customer's, taken verbatim from
[ROCm/MAD PR #173](https://github.com/ROCm/MAD/pull/173) `scripts/AgentX_CaseA/`.
`replay_caseA.sh` was **not modified** — md5 `7cde1afc627c7e4868eac0fd13741baa`
matches the PR blob. Everything site-specific was passed as environment.

## Result

| | C=2 | C=8 | C=16 |
|---|---:|---:|---:|
| requests (profiling) | 72 | 229 | 303 |
| **TTFT p50 / p90** | 2,503 / 14,458 ms | 6,698 / 23,775 ms | 19,496 / 42,484 ms |
| TTFT p99 | 24,085 ms | 33,681 ms | 83,575 ms |
| **ITL p50 / p90** | 11.8 / 15.6 ms | 13.3 / 18.5 ms | 14.7 / 23.3 ms |
| **E2E p50 / p90** | 8.6 / 28.6 s | 13.9 / 38.8 s | 27.6 / 72.3 s |
| output throughput | 98 tok/s | 258 tok/s | **402 tok/s** |
| input throughput | 6,772 tok/s | 19,899 tok/s | **26,447 tok/s** |
| ISL mean / p99 | 85,314 / 244,873 | 80,811 / 240,623 | 81,173 / 237,218 |
| server cache read / prompt | 58.6 % | 66.1 % | 66.6 % |
| **`submission_valid`** | **true** | **true** | **true** |

Zero engine faults on either leg across all three runs.

## All five features verified live, not assumed

| feature | evidence | result |
|---|---|---|
| **PD** | `/v1/workers` | prefill `dp_size=null` + decode `dp_size=8`, both `active` |
| **DPA** | live command lines | prefill **no** `--enable-dp-attention` (by design) / decode **8** `scheduler_DP*` |
| **MTP** | decode log, n=5,564 batches | mean accept len **3.06**, p50 2.98 — healthy band |
| **kvd** | `statctl` deltas | prefill **+168,866 sets / +121,621 evictions / 0 gets**; decode all-zero by design |
| **RDMA** | `MC_FORCE_TCP`/`GID is NULL` count | **0** on both legs — real mlx5 RDMA, no TCP fallback |

## The headline finding: 88 % constructed reuse measures as 66 %

The corpus is built for ~88 % prefix reuse and `verify_caseA.py` passes **13/13**
axes on it. The server's own `usage.prompt_cache_read_tokens` reports **50.8 %**
(C=8) overall. That gap is **not** a cache defect. Decomposed by turn:

| segment | C=8 | C=16 |
|---|---:|---:|
| turn 0 (session's first turn) | n=53, **0.0 %** | n=73, **0.0 %** |
| turns 1–2 | n=71, **70.1 %** | n=99, **67.1 %** |
| turns 3+ | n=105, **66.4 %** | n=131, **65.5 %** |
| overall | 50.8 % | 49.7 % |

Two independent effects, both structural:

1. **Turn 0 is cold by construction, and the scenario forces it to stay cold.**
   `inferencex-agentx-mvp` locks `--cache-bust first_turn_prefix`
   (`inferencex_agentx_mvp.py`), injecting a unique marker at the head of every
   first user turn so a recycled trace cannot inherit the previous play's
   prefix. 23 % of C=8's profiling requests are turn 0, contributing exactly
   **zero** cache read. That alone drags an ~66 % per-turn rate down to ~51 %.
2. **The remaining ~66 % vs the corpus's ~88 % is a block-granularity effect,
   not a miss.** The trace counts reuse in its own 64-token `hash_ids` blocks;
   the server counts whole KV pages it could actually skip. Partial trailing
   blocks and page alignment cost the difference. **This second part is
   measured, not explained** — separating the two would need per-request
   `cached_tokens` correlated against the trace's own `hash_ids` overlap, which
   this run did not capture.

The rate being **flat across concurrency** (66.1 % → 66.6 %) says the cache is
not being evicted under load at these levels — consistent with kvd's
121,621 evictions being spillover-tier, not hot-tier, churn.

## What scales and what doesn't

Doubling C=8 → C=16 buys **+55 % output throughput** (258 → 402 tok/s) and costs
**+191 % TTFT p50** (6.7 → 19.5 s). ITL barely moves (13.3 → 14.7 ms p50), so
the decode loop is not the bottleneck — the queue in front of prefill is. With
ISL p50 ≈ 72 K and DPA off, a single prefill scheduler serves the whole node,
and 16 in-flight long prompts serialize against it.

`Theoretical Prefix Cache Hit` (aiperf's own column, ~50.8 %) and the
server-measured 66.1 % are **different quantities** and should not be compared:
the former is computed from the trace, the latter reported by the engine.

## Deviations from the reference kits, each deliberate

| | value | why |
|---|---|---|
| prefill `--chunked-prefill-size` | **65536** | operator decision. sglang divides this by `dp_size` **only** under DPA (`server_args.py:4902`), so with DPA off the engine resolves it to 65536, not 8192 — 8× the per-forward batch. Paired with GMU 0.70 to absorb the activation peak. |
| prefill `--mem-fraction-static` | **0.70** | the pairing above; the vultr par8 kit takes the other route (chunk 16384 + GMU 0.80) |
| decode `--mem-fraction-static` | 0.85 | unchanged from the spur Case-A kit |
| fabric | **mlx5_0**, GID 3, dma-buf ON | spur is configured oppositely to vultr (ionic/GID 1/dma-buf off there) |
| image | rebuilt `final-pr` | see `notes/notes.md` — every node holding a prebuilt image had its GPUs occupied by other tenants |

## Navigate

| path | what |
|---|---|
| `results/summary.csv` | the customer script's own output, both points |
| `results/c2|c8|c16/` | full aiperf CSV + JSON per point |
| `results/kvd_*.json` | kvd counters before/after |
| `results/env_*.txt` | node, kernel, driver, image digest, live command line |
| `scripts/` | every script that ran, plus the aiperf Dockerfile |
| `spec/` | the customer's bench, verbatim |
| `logs/` | both engine logs (gzipped) |
| `notes/notes.md` | the traps, and what this run could not answer |
