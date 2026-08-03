# kv-aware routing — what it did, and why it could not do more

This run was expected to show per-DP-rank cache steering. **It shows none, on
either leg.** That is a property of the deployment, not a fault, and pinning
down *why* is the most transferable result in this kit.

All pick data below comes from the Rust router's policy log
(`rust/router/src/policy.rs:314`), captured as `../results/router.log.gz` and
parsed with `../scripts/cv_scoped.py`.

> **Scoping is mandatory.** `/tmp/router.log` lives inside the container and is
> appended all day across leg restarts. An unscoped read of this exact file
> mixes in picks addressed to `10.2.122.3` — the **previous** decode leg on
> chi2878, a different deployment. Always pass the run's start timestamp:
> `cv_scoped.py /tmp/router.log 2026-08-03T09:28`.

---

## Measured distribution — full window (09:28:21 – 10:35:05, 5,768 picks)

### Decode — all 8 ranks, near-uniform

| rank | picks | share |
|---|---|---|
| dp0 | 373 | 12.9 % |
| dp1 | 387 | 13.4 % |
| dp2 | 346 | 12.0 % |
| dp3 | **411** | **14.3 %** |
| dp4 | 306 | 10.6 % |
| dp5 | 363 | 12.6 % |
| dp6 | 391 | 13.6 % |
| dp7 | 307 | 10.6 % |
| **total** | **2,884** | |

`cache_hits`: **max 0, mean 0.0, nonzero 0 / 2,884.**

Spread 10.6 – 14.3 %, ratio max/min = **1.34**. This is load balancing.

### Prefill — one target

| target | picks | share |
|---|---|---|
| `10.2.122.78:30000` | 2,884 | **100 %** |

`cache_hits`: **max 3,615, mean 1,201.2, nonzero 2,876 / 2,884 (99.7 %).**

---

## Why decode cannot steer by cache

The cost function is identical in both router backends
(`rust/router/src/policy.rs:277-289`, `infera/router/policy/kv_event_aware.py:169-174`):

```
cost(target) = w_overlap × (total_blocks − prefix_hits) + active_blocks(target)
pick = argmin cost,  tie-break least-loaded
```

With `prefix_hits ≡ 0` on every decode rank, the first term is
`w_overlap × total_blocks` — **identical across all 8 ranks** — so it cancels
out of the argmin entirely. What remains is `active_blocks`: pure least-loaded.
The measured 1.34 spread is exactly that.

**`prefix_hits` is 0 because the decode leg has no prefix tree.** Its resolved
args are `disable_radix_cache=True` and
`disaggregation_decode_enable_radix_cache=False`, so
`mem_cache/registry.py:78-93` builds a **`ChunkCache`**, not a `RadixCache`:

| ChunkCache behaviour | code |
|---|---|
| `match_prefix()` always returns **empty** | `chunk_cache.py:64-71` |
| `insert()` is a **no-op** | `chunk_cache.py:73-75` |
| `disable` property hardcoded **True** | `chunk_cache.py:58-60` |
| never emits `BlockStored` — inherits `BasePrefixCache`, not `KVCacheEventMixin`, so `take_events()` returns `[]` | `chunk_cache.py:35`; `base_prefix_cache.py:325-326` |

So the router's KV view for that worker is permanently empty. **Verified
directly on the wire**, not inferred — subscribing to each engine's kv-event
socket during the run:

| engine | endpoint | messages in 15 s |
|---|---|---|
| prefill | `10.2.122.78:63985` | **30** (`BlockStored` streaming) |
| decode | `10.2.122.10:29361` | **0** |

ChunkCache still allocates and holds each request's KV in the token pool
(`chunk_cache.py:43-52, 82-84`) and frees it at request finish
(`chunk_cache.py:79-85`). It is a *policy* object, not a storage layer — the KV
exists, it is simply never shared across requests.

## Why the decode leg has no radix cache: MTP forbids it

Not a configuration choice. `arg_groups/pd_disaggregation_hook.py:29-56` raises:

```python
if server_args.disaggregation_decode_enable_radix_cache:
    ...
    if server_args.speculative_algorithm is not None:
        raise ValueError(
            "--disaggregation-decode-enable-radix-cache is incompatible "
            "with speculative decoding "
            f"(--speculative-algorithm {server_args.speculative_algorithm})")
else:
    server_args.disable_radix_cache = True
    logger.warning("KV cache is forced as chunk cache for decode server")
```

The mission requires MTP. **MTP and decode-side radix are mutually exclusive in
sglang v0.5.15.post1 — you get one.** Our launcher never touches any radix flag;
this is upstream's default path.

**Upstream states no reason for the exclusion.** It was added silently in
[PR #19746](https://github.com/sgl-project/sglang/pull/19746) (merged
2026-05-01), whose body never mentions speculative decoding; none of its 30
review comments question it; the docs assert it without rationale; and
[PR #28238](https://github.com/sgl-project/sglang/pull/28238) later preserved it
without explaining it. The only reasoning anywhere is an outside contributor's
in [PR #32170](https://github.com/sgl-project/sglang/pull/32170) — **unmerged,
CI red** — which argues the opposite: *"the radix tree itself is already
EAGLE-aware … the mutual exclusion is a conservative gate rather than a
structural limitation."*

Reading the code, three real conflicts exist, any one sufficient to justify a
guard (details in `../notes.md`). **Which one motivated it is not determined.**

## Why prefill cannot steer either

DPA off ⇒ `dp_size = 1` ⇒ `expand_targets()` (`infera/router/policy/target.py:45`)
yields **one** routing target instead of eight. `/v1/workers` reports
`"dp_size": null` for the prefill worker. There is nothing to choose between.

**This is inherent to DPA-off, not an independent variable.** It also means:
**this configuration must not be used for a kv-aware routing study.**

## The distinction that matters: cache ≠ kv-aware

Prefill `cache_hits` averaged **1,201 blocks** (× 64 tokens/block ≈ 77K tokens)
with 99.7 % of picks non-zero, and the run achieved **88.9 % prefix hit rate**.
Those hits are **real**. They are also **entirely sglang's own radix tree inside
the single prefill engine** — the router did not route them there, because there
was nowhere else to route them.

> **The prefix cache worked. kv-aware routing did not run.** One `cache_hits`
> number covers both, which is how they get conflated.

## The one run where kv-aware genuinely worked

`../caseA.glm52.fullfeature.packup_20260801` — prefill DPA **on**, 8 targets:

| | prefill (w=20.0) | decode (w=2.0) |
|---|---|---|
| distribution | dp0 **175** … dp3 **35** | dp0 141, others 58–62 |
| skew (max/min) | **5.0×** | 2.4× |
| `cache_hits` | max **2,422**, mean 493.5 | **0** |

Prefill deliberately skewed toward cache locality; decode near-uniform. **The
decode leg's zero `cache_hits` is the same finding as this run's** — it is
structural and reproduces across both.

## Consequence: every turn re-transfers the full prompt KV

Decode radix off also disables the PD transfer optimisation. The prefill→decode
send window is `[decode_prefix_len, len(origin_input_ids))`
(`disaggregation/prefill.py:273-281`), and `decode_prefix_len` defaults to **0**
(`disaggregation/base/conn.py:134-135`), set non-zero *only* by a decode-side
radix match (`disaggregation/decode.py:878-889`, else-branch at `:904-908`).

```
MTP on → decode ChunkCache → decode_prefix_len = 0 → start_send_idx = 0
       → the ENTIRE prompt KV crosses the wire, every turn
```

**A prefill-side cache hit does not reduce this.** `maybe_send_cached_prefix_chunk`
(`prefill.py:969-983`, [PR #29316](https://github.com/sgl-project/sglang/pull/29316))
only sends the cached prefix *earlier*, overlapping transfer with the suffix
forward. It advances `start_send_idx`, so the total is unchanged — **a reorder,
not a reduction.**

At this workload's mean prompt of 86,888 tokens, every one of 2,850 requests
shipped its full prompt KV over RoCE. **Whether that is a measurable bottleneck
here is unmeasured.** The discriminating measurement: per-request transfer bytes
and time from the prefill leg (`get_transfer_metric` / `transfer_total_bytes` in
`disaggregation/common/conn.py`) correlated against TTFT.

## What to run next to actually study kv-aware

| goal | configuration | note |
|---|---|---|
| prefill per-rank cache steering | par8.yaml, prefill **DPA on**, CHUNK back to 65536 | also the missing control for this run's TTFT delta |
| decode per-rank cache steering | **not reachable** with MTP on | would require dropping MTP, violating the mission |
| does full-prompt re-transfer cost us? | instrument `transfer_total_bytes` vs TTFT | no config change needed |
