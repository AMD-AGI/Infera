# kvaware / kvd on the sglang engine — what infera actually implements

Source-verified 2026-07-30 against branch `yihou.dev.glm5.2.mxfp4.experiment`
@ `362192e7`, and against sglang 0.5.15.post1 inside
`infera/engine-sglang:pd-unified`. Every claim carries a file:line.

## Engine support at a glance

| | sglang | vLLM | ATOM |
|---|---|---|---|
| kvaware (KV events) | yes, **native** — no site hook | yes | yes, but via `.pth` site hooks, OFF by default |
| kvd | yes — `InferaKvdBackend(HiCacheStorage)` | yes — `InferaKvdConnector(KVConnectorBase_V1)` | **no** |

`manual/features/kv_cache_offload.md:14-18` still says KV-Cache Offload is
"**vLLM only (for now)**". That is **stale** — sglang has a working kvd path via
HiCache. Reported, not fixed (out of scope for this experiment).

---

## kvaware — the three layers

### Layer 1: what sglang already provides
- `--kv-events-config` and its ZMQ publisher —
  `sglang/srt/managers/scheduler_components/kv_events_publisher.py`. Binds one
  PUB socket per DP rank at `base + attn_dp_rank`.
- `RadixCache` and its `insert` / `evict` / `reset` semantics.
- `--disaggregation-decode-enable-radix-cache` — sglang's own switch.

### Layer 2: what infera builds itself (engine-agnostic)
The whole routing + index plane; sglang has no equivalent.

| Component | File |
|---|---|
| `KvEventProbe` — coalesces engine-block events into index-block events (xxhash, releases the GIL) | `infera/kv/probe.py` |
| ZMQ publisher / snapshot producer / NATS relay / index | `infera/kv/{publisher,snapshot,nats_relay,index}.py` |
| Chained XXH3-64 hashing (`parent_hash ‖ token_ids`) | `infera/router/kv_event/hasher.py` |
| kv-aware scoring `cost = w*(request_blocks − hits) + active_blocks` | `infera/router/policy/kv_event_aware.py` |
| Tokenizer digest + canary (router/worker must agree byte-for-byte) | `infera/kv/compat_key.py` |

Router-side knobs: `--router-tokenizer-path` (required), `--kv-overlap-weight`,
`--kv-prefill-overlap-weight` (PD, typical 20.0), `--kv-decode-overlap-weight`
(PD, typical 2.0) — `infera/server/args.py:139-175`. `--index-block-size`
(default 64) is the **router's coalescing unit, NOT the engine's KV page size**
(`infera/engine/sglang/args.py:206-212`).

### Layer 3: what infera wrote specifically to adapt sglang
1. `attach_to_radix_cache()` — monkey-patches a live `RadixCache.insert/evict/reset`
   (`infera/engine/sglang/kv_probe.py:54`). Hooks at **node level**, not per-page,
   precisely so ROCm+AITER `page-size 1` stays cheap: one Python dispatch covers
   a whole node.
2. `_find_radix_cache()` — sglang versions disagree on where the cache hangs;
   tries `tree_cache` / `radix_cache` / `_radix_cache`
   (`infera/engine/sglang/kv_wiring.py:196`).
3. Port allocation + `--kv-events-config` JSON synthesis
   (`infera/engine/sglang/worker.py:75-85`). **This is where the bug in
   `patches/` lived.**
4. Auto-append `--disaggregation-decode-enable-radix-cache` on the decode leg —
   only when kv-events are on **and** backend is mooncake, because mori/nixl
   reject the flag (`infera/engine/sglang/args.py:251-263`).

> Two event paths coexist: sglang's native publisher (via `--kv-events-config`)
> **and** infera's own monkey-patched probe plane. See notes.md — whether the
> probe plane actually attaches across the subprocess boundary is **unverified**.

---

## kvd — the three layers

### Layer 1: what sglang already provides
- The whole HiCache machinery: `--enable-hierarchical-cache`, `--hicache-ratio` /
  `--hicache-size` / `--hicache-io-backend` / `--hicache-mem-layout`,
  `HostKVCache`, `HiRadixCache`, prefetch policy.
- The `HiCacheStorage` ABC — `sglang/srt/mem_cache/hicache_storage.py`
  (`batch_get_v2` / `batch_set_v2` / `batch_exists_v2` / `get` / `set` /
  `clear` / `get_stats` …).
- The `dynamic` backend loader (imports by `module_path` + `class_name`).
- The constraint that `enable-hierarchical-cache` and `disable-radix-cache` are
  mutually exclusive (`server_args.py:_handle_cache_compatibility`).

### Layer 2: what infera builds itself
The entire `infera/kvd/` daemon — `server.py`, `store.py`, `shared_arena.py`
(memfd zero-copy), `tablespace.py`, `striped_long_region.py`,
`mooncake_long_region.py`, `lmcache_long_region.py`. Tiering L1 HBM → L2 host RAM
→ L3 NVMe/NFS → L4 distributed, the UDS protocol, retention semantics. Engine-
independent: vLLM reuses the same daemon through a different connector.

### Layer 3: what infera wrote specifically to adapt sglang
1. `InferaKvdBackend(HiCacheStorage)` — `infera/engine/sglang/kvd_adapter.py:165`.
   Contains a **sync↔async bridge**: sglang's `CacheController` is synchronous,
   `KvdClient` is async, so the adapter spins a background event loop and
   dispatches via `run_coroutine_threadsafe`.
2. `kvd_wiring.py` — expands one `--infera-kvd-socket` flag into: probe the
   daemon (5 s, **refuse to start** if dead) → register the backend → append
   `--enable-hierarchical-cache --hicache-storage-backend dynamic
   --hicache-storage-backend-extra-config {...}` to the **child** argv. It must
   patch argv, not `ServerArgs`, because sglang runs in a subprocess that
   re-parses argv from scratch (`kvd_wiring.py:_finish_wiring` docstring).
3. `hicache_validate.py` — warns at CRITICAL when `--hicache-ratio < 1.5`, where
   sglang's `prefetch_capacity_limit` computes to ~0 and L3 is written but never
   read.
4. `hipfile_shim.py` — GPU-direct read path.
5. TP/PP rank folded into the kvd namespace (`compat_key`) so multi-rank workers
   don't collide.

---

## The coupling that is easy to miss

A decode leg sets `disable_radix_cache=True` on its own ("KV cache is forced as
chunk cache for decode server"), and sglang forbids hicache alongside it — so
**kvd is illegal on a decode leg by default**. What makes it legal is the
auto-appended `--disaggregation-decode-enable-radix-cache`, and that append is
gated on kv-events being enabled.

**Therefore: turning kvaware off also disables kvd on the decode leg.**

Verified by driving sglang's own `ServerArgs.from_cli_args` over the matrix
(`--model-path` GLM-5.2, tp8, in-container):

| # | combo | verdict |
|---|-------|---------|
| 1 | mix baseline | OK |
| 2 | mix + hicache | OK |
| 3 | mix + DPA + hicache | OK |
| 4 | PD-prefill + DPA + hicache | OK |
| 5 | PD-decode + DPA + hicache | **FAIL** — hicache vs disable-radix-cache |
| 6 | #5 + `--disaggregation-decode-enable-radix-cache` | OK |
| 7 | #4 + `--disable-radix-cache` | FAIL (same conflict) |
| 8 | #5 + `--disaggregation-decode-enable-offload-kvcache` | FAIL (same conflict) |

Direct probe of the mechanism:

```
[decode leg, NO hicache]              disable_radix_cache=True
[decode leg + decode-radix flag]      disable_radix_cache=False
[prefill leg, NO hicache]             disable_radix_cache=False
```
