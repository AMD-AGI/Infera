# Phase 0 — bring-up and feature proof

**Ran:** 2026-08-01 09:10–09:45 UTC · **Nodes:** chi2879 (prefill) + chi2867 (decode)
**Image:** `infera/engine-sglang:merged-e` — chi2879 `bfcb6462fa30`, chi2867 `27667ee43291`
**Status:** **PASS** — all six rows green, after one real bug found and fixed.

This is the first time `ctx=262144` **and** MTP have run together on this stack, and
the first time the **Rust** router has carried live traffic.

## The bug: `MC_GID_INDEX` was hardcoded, and is node-dependent

The decode leg died on the first boot — all 8 DP ranks, during init:

    rdma_context.cpp:1132  GID is NULL, please check your GID index by specifying MC_GID_INDEX
    rdma_context.cpp:200   Failed to open device ionic_4 on port with GID 1
    rdma_transport.cpp:932 Disable device ionic_4
    RuntimeError: Mooncake Transfer Engine initialization failed.
    RuntimeError: Rank 0 scheduler died during initialization (exit code: -3)

**The prefill leg on chi2879 had zero such errors.** That asymmetry is the whole
diagnosis: the kit's `glm52_leg.sh` hardcodes `MC_GID_INDEX=1`, which is correct on
chi2879 and wrong on chi2867. Both nodes expose two RoCE v2 GIDs per ionic port — a
link-local `fe80::` and a routable `fd93::` — but at **different indices**:

| node | idx0 | idx1 | idx2 |
|---|---|---|---|
| chi2879 | `fe80::` link-local | **`fd93::` routable** | empty |
| chi2867 | `fe80::` link-local | **empty** | **`fd93::` routable** |

Verified identical across all 8 NICs on each node. `show_gids` confirms it
independently (`n_gids_found=2`, indices 0 and 2 on chi2867).

The link-local idx0 is **not** a fallback: it is not routable across the fabric and
crashes MoRI at `ionic.cpp:414`.

**Fix** (`scripts/glm52_leg.sh`, "BENCH DELTA"): discover the index instead of
hardcoding it — first GID that is neither empty nor `fe80::`. Falls back to the
hardcoded 1 if discovery finds nothing, so a node where discovery fails behaves as
before rather than silently differently.

Resolves to **1 on chi2879** (unchanged) and **2 on chi2867** (the fix). After it:
`Using user-specified GID index: 2` ×64, mooncake init failures **0**.

> Why the earlier kits never hit this: every prior cross-node run used chi2879 as
> prefill and either never got decode past init on chi2867, or ran both roles on
> hosts where idx1 happened to be routable. The hardcode was a latent node-dependency.

### A second, smaller trap on the way

The fix appeared not to work: after re-staging to `/mnt/vast` the log still read
`GID index: 1`. Cause — `reset_node.sh` `docker cp`s the leg script into the container
**once at container creation**, so editing the shared-fs copy between rounds silently
runs the *old* script. Now `start_leg.sh` re-copies on every launch. This is the same
class as the stale-`__pycache__` trap: the artifact you edited is not the artifact that
ran.

## Feature-proof matrix — all six green

| # | feature | signal | result |
|---|---|---|---|
| 1 | **PD** | `/v1/workers` both modes active | `10.2.122.10:30000 prefill active dp_size=8`<br>`10.2.122.44:30000 decode active dp_size=8` |
| 2 | **DPA** | 8 `sglang::scheduler_DP*` per node | **8 / 8** on both |
| 3 | **RDMA** | mooncake init failures, `MC_FORCE_TCP` | **0 / 0**; 194 ionic mentions; GID idx discovered |
| 4 | **MTP** | decode `accept len` | **1.52 – 2.67** (11 samples). *Not* 4.00 |
| 5 | **kvd** | restart-replay: gets/hits climb, sets flat | **gets 102, hits 102, sets 102 unchanged, misses 0** |
| 6 | **kv-aware** | per-DP-rank picks + `cache_hits` | **prefill `cache_hits` max 51, nonzero 31/36** |

### Row 5 — the attribution test, not a latency test

A speed-up proves nothing: sglang's in-GPU radix cache serves a repeated prefix without
touching L3. Restarting the prefill engine (190 s) empties that cache while the kvd
daemon and its L3 keep running, so any reuse afterwards can only be L3.

| | gets | hits | sets | misses |
|---|---:|---:|---:|---:|
| after first reuse run | 0 | 0 | 102 | 0 |
| **after restart + replay** | **102** | **102** | **102 (unchanged)** | 0 |

`sets` staying put is the load-bearing part: reads, not re-writes. Identical to the
historic baseline.

### Row 6 — kv-aware under the Rust router

The Rust router routes only `/health`, `/v1/workers`, `/v1/models`, `/metrics` and the
completion endpoints (`rust/router/src/handlers.rs:33-38`). There is **no**
`/v1/admin/cache-view`, and `total_blocks()` is unrouted — so the per-rank signal comes
from the policy log line (`rust/router/src/policy.rs:314`), parsed by
`scripts/cache_view.py`.

    72 pick decisions
    === Prefill  (w_overlap=20.0) ===
      10.2.122.10:30000#dp0    36 picks
      cache_hits: max=51 mean=43.9 nonzero=31/36
      request_blocks: max=51 mean=45.3
    === Decode   (w_overlap=2.0) ===
      10.2.122.44:30000#dp0    36 picks
      cache_hits: max=0  mean=0.0  nonzero=0/36

Three things this establishes:

1. **Routing is per-DP-rank.** The pick key is `worker#dp0`, not `worker` — the router
   subscribed `ranks=8` per worker and scores each rank separately.
2. **`cache_hits` max = 51** — the same 51 the python router produced in G0/G1. Under
   MTP the prefill leg's kv-events carry **bigram** pairs; unfixed, the Rust
   `as_u32_vec` dropped them all and the view read **0**. This is group E's bigram fix
   observed on a live path for the first time.
3. **decode `cache_hits` = 0 is expected** — the decode radix cache is deliberately off
   under MTP (branch patch: `--no-enable-kv-events` semantics on the decode leg).

⚠️ **All 36 picks landed on `#dp0`.** Not yet evidence of a bug: `prefix_reuse.py` sends
one stable shared prefix, so every request's best-overlap rank is legitimately the same
one, and cost ties break to the first candidate. Whether the scorer *spreads* across
ranks is a real question for the fixlen sweep, where concurrency ≥32 makes
`active_blocks` diverge. Recorded as open, not concluded.

## Correctness

`probe.py` **4/4** through the router. Sampling left at the probe's default; the
multi-chunk needle test belongs with the sweep config and is deferred to Phase 1's
first long-ISL round, where it is a by-product rather than a separate cold start
(mission Rule 4: do not over-spend on needle).

`prefix_reuse.py` **16/16 + 16/16**, twice (before and after the restart).

## Deployment under test — frozen for all subsequent phases

    two-node PD over mooncake RDMA (ionic RoCE, GID idx discovered per node)
    DP-attention 8/8 both legs      --dp-size 8 --enable-dp-attention --ep-size 8
    kv-aware routing                ON, Rust backend, w_prefill 20.0 / w_decode 2.0
    kvd (infera HiCacheStorage)     prefill ON (--hicache-size 16), decode skipped by design
    MTP                             decode leg, EAGLE steps=3 topk=1 draft=4
    --context-length                262144
    --chunked-prefill-size          65536   (= 8192/rank at dp8)
    --cuda-graph-max-bs             128
    --max-running-requests          2048    (engine reports 256 effective)
    --enable-cache-report           ON      (bench delta; else cache-hit reads 0)
    mem-fraction-static             0.88 prefill / 0.85 decode
    max_total_num_tokens            3,260,672 per rank (prefill, at ctx 262144)

`max_total_num_tokens` is unchanged from the ctx=32768 runs — the pool is sized by
memory, not by context length, so raising ctx cost no KV capacity.

## Open

- **`hicache-size 16` GB gives a 356,160-token host pool against a 3,260,672-token
  device pool.** The engine warns L2 effectiveness is reduced. Left at 16 GB
  deliberately: it is the value every prior validated run used, and changing it here
  would make this deployment non-comparable to them. Revisit only if kvd hit-rate is
  the bottleneck in Case A.
- Whether kv-aware spreads picks across DP ranks under real concurrency (see above).
