# par8 arm B — prefill DP-attention OFF + kv-aware routing, on **spur**

**Ran 2026-08-03 12:44:04 – 13:51:12 UTC** (4,007.7 s = 66.8 min, one full cycle:
ramp 400 + sustain 3,600 + drain). Two-node PD on the **crsuse2-m2m** (spur)
cluster, against the **final-PR branch** image.

## What this run is

One arm of a two-arm pair requested by the operator. Both arms run the **par8
workload byte-identical** (only the tokenizer path is retargeted to this
cluster's filesystem); they differ in the deployment:

| | arm A | **arm B (this kit)** |
|---|---|---|
| prefill DP-attention | on (dp8) | **off (pure TP8)** |
| router policy | round-robin | **kv-aware** (pw 20.0 / dw 2.0) |

**Arm A now has its own kit**:
[`../par8.armA.dpaon.roundrobin.spur.packup_20260804/`](../par8.armA.dpaon.roundrobin.spur.packup_20260804/).
It took two attempts — the first OOMed 60 s in and its allocation expired before a
rerun could finish; the rerun completed cleanly. `notes.md` § "What arm A's crash
established" here records the finding from that crash, which was written before
the rerun existed; the arm A kit is now the fuller account, including a
side-by-side comparison of the two arms.

This is the **spur** sibling of `../par8.glm52.dpaoff.packup_20260803/`, which
ran the same workload and the same DPA-off prefill on **vultr**. The two clusters
have different fabrics and the numbers are **not** directly comparable — see
[§ Cross-cluster](#cross-cluster-comparison-read-the-caveat-first).

## Result at a glance

| | measured |
|---|---|
| duration | **4,007.7 s** (ramp 400 + sustain 3,600 + drain) |
| requests | 2,907 sent / **2,861 completed** |
| **success rate** | **0.9842** |
| errors | **25** — all client-side 240 s timeouts; **0** HTTP failures |
| **TTFT p50 / p90** (sustain) | **2,239 / 6,389 ms** |
| TTFT p99 (sustain) | 10,606 ms |
| TPOT p50 / p90 | 16.1 / 21.1 ms |
| **cache hit** | **88.65 %** actual vs 88.99 % ideal — efficiency **99.62 %**, eviction 0.38 % |
| **MTP acceptance** | **2.789** engine-measured (n = 17,674) |
| in-flight | max 24 cap; **never pinned** during sustain |
| engine faults | **0** on both legs across the full window |

### Against the workload's own SLA block

| bar | source | measured | verdict |
|---|---|---|---|
| success rate ≥ 0.97 | `par8.yaml` `sla.success_rate` | **0.9842** | **PASS** |
| TTFT p90 < 30,000 ms | `par8.yaml` `sla.ttft_p90_ms` | **6,389 ms** | **PASS** (4.7× margin) |
| e2e p50 < 4,500 ms | `par8.yaml` `sla.e2e_p50_ms` | **not recorded** | see below |
| full cycle completes | the instruction | 4,007.7 s | **PASS** |
| in-flight not pinned at cap | method | not pinned | **PASS** |

> `e2e_p50_ms` has no counterpart in this driver's `summary.json` (it emits
> `ttft_ms` / `tpot_ms` / per-phase blocks, no end-to-end percentile object). The
> vultr par8 kit reported 7.4 s against this 4.5 s bar and called it a **latency-floor
> spec, not a capacity spec** — met only at concurrency 1. Nothing here changes
> that reading, but this run **did not measure it**, so it is marked not-recorded
> rather than inherited.

## The five features, each with a positive signal

| feature | signal | result |
|---|---|---|
| **PD** | router `/health` + registered workers | `{"status":"ok","active_workers":2}`, prefill + decode both registered with `kv_events` endpoints |
| **DPA** | resolved `server_args` per leg | prefill `enable_dp_attention=False`, `dp_size=1` (**by design**); decode `dp_size=8`, all 8 ranks decoding 2,030–2,338 batches each |
| **kv-aware** | router log line at startup | `router-policy=kv-aware overlap_weight=1 prefill=20.0 decode=2.0` |
| **MTP** | `accept len` in the decode log | **2.789** mean over 17,674 samples — healthy band, not the degenerate 4.00 |
| **kvd** | `infera-kvd adapter connected` per DP rank | prefill **8** (one per rank); decode **0** *by design* — `_skip_kvd_on_decode_leg` |

**RDMA is real, not silently degraded**: `MC_FORCE_TCP` and `GID is NULL` both
count **0** in both leg logs. On this fabric that check is load-bearing — TCP
fallback works and looks fine, it is merely slow.

## Cross-cluster comparison — read the caveat first

| | **arm B (spur, this kit)** | par8 (vultr, `../par8.glm52.dpaoff.packup_20260803/`) |
|---|---|---|
| requests | 2,907 / 2,861 | 2,884 / 2,850 |
| success | 0.9842 | 0.988 |
| **TTFT p50** | **2,239 ms** | **1,353 ms** |
| **TTFT p90** | **6,389 ms** | **4,948 ms** |
| TTFT p99 | 10,606 ms | 10,674 ms |
| TPOT p50 | 16.1 ms | 14.8 ms |
| cache hit | 88.65 % | 88.8 % |
| accept len | 2.789 (engine) | 2.02 (per-request) |

> **The ~1.65× TTFT gap is NOT attributable to anything this run configured.**
> The fabrics differ: spur is **one mlx5 rail with dma-buf**; vultr is **8 ionic
> rails with peermem**. Chunk also differs (65,536 global here vs 16,384 there —
> see `notes.md` §2, the two reference kits disagree and the disagreement is
> recorded rather than resolved). Two uncontrolled variables, one of them the
> transport itself. Quote the *within-cluster* numbers; the cross-cluster row is
> context, not a measurement.

## Known gaps in this kit

Stated up front rather than discovered on a cold read:

| gap | why | impact |
|---|---|---|
| **no `collect_env.sh` snapshot** | both allocations were reclaimed at the 24 h wall clock (SIGTERM, `ExitCode=143`) ~11 h after the run; the script was never executed while the nodes were live | hardware facts in `environment.md` are reconstructed **from the run's own logs** and are marked as such |
| **no kvd after-state** | same cause — `statctl` could not be reached post-run | the kvd *before* baseline (all-zero) is kept; the delta is unmeasured. The `adapter connected: 8` signal still proves the wiring |
| **no router pick distribution** | `/tmp/router.log` lives inside the container and died with it | kv-aware is proven *configured* (startup line) but its per-rank steering is **not** measured on this arm |

None of these three is recoverable without re-running. They are gaps in
*evidence*, not signs the run misbehaved.

## Navigate

| file | what |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | ordered, copy-pasteable reproduction |
| [`environment.md`](environment.md) | nodes, image digests, git SHA, resolved server args |
| [`notes.md`](notes.md) | the two leg-script fixes, arm A's crash, gotchas, wrong turns |
| [`analysis/`](analysis/) | per-phase ladder, feature evidence, the SLA table worked through |
| [`spec/`](spec/) | `par8.yaml` as run + the Case A parent it derives from |
| [`scripts/`](scripts/) | every script that ran, verbatim |
| [`patches/`](patches/) | the two leg-script edits, as diffs with what/why/how/context |
| [`results/`](results/) | `summary.json`, `metrics.jsonl.gz`, kvd baselines |
| [`logs/`](logs/) | both engine logs + driver + build log (gzipped) |
| [`env/`](env/) | the exact env files the legs were launched with |
