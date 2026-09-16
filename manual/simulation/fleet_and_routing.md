# Fleet and routing

```{admonition} One-pager
:class: tip
**What:** several engine replicas behind a router, each with a real
content-addressed KV block cache. **Why:** cache hit rate is a property of
routing, not a constant — and the `kv` policy scores the same cost function the
[deployed router](../features/kv_aware_routing.md) does. **Cost:** you have to
supply content, either a prefix pool or a trace.
```

## Simulate a fleet

```bash
inferasim inference ... \
  --request-rate 6 --arrival-model poisson --des-num-requests 200 \
  --des-instances 4 --des-routing kv \
  --des-num-prefixes 8 --des-prefix-len 2048 --des-block-size 512
```

```text
  Fleet: 4 instance(s), routing=kv | prefix pool: 8 prefixes
  KV block cache: block_size=512 tok, capacity/instance=unbounded, evictions=0, block-reuse=44.8%
  Prefix-cache hit rate: 89.5% of requests (avg 1833 cached tok/req; per-instance 88–90%)
```

## Hit rate is emergent, not configured

A prompt is an ordered sequence of block-hash ids, and a hit is the longest
*contiguous leading run* of blocks already resident on the replica the request
was routed to. The cache is finite and LRU-evicts under pressure, so **hit rate
is an emergent property** of content, capacity and routing rather than a number
you supply.

This is the structural difference from
[`--prefix-cache-hit-rate`](projection_runs.md#declare-prefix-reuse) on the
analytical path, which asserts a rate. Here, changing the routing policy changes
the hit rate, because that is what routing policies do.

## Routing policies

| `--des-routing` | Behaviour |
|---|---|
| `kv` | The serving router's own policy: minimise `--des-overlap-weight` × blocks missed + replica load. The weight dials reuse against balance (`0` = pure load balance). |
| `prefix_aware` | Consistently hash the leading block to a home replica, so same-prefix requests co-locate. Misses ≈ number of prefixes, independent of fleet size. |
| `round_robin` / `random` | Ignore locality, so every replica re-warms every prefix. Misses ≈ prefixes × replicas. |

The trade-off is real in both directions. Locality-seeking policies maximise
reuse but can overconcentrate a hot prefix onto one replica, raising its decode
pressure and lengthening the makespan. The report shows pooled latencies
alongside the per-replica hit-rate spread so you can see both halves — a
suspiciously good hit rate next to a wide per-instance spread is a hot-spotting
warning, not a win.

Because `kv` scores the same cost function the deployed router does, sweeping
`--des-overlap-weight` projects what retuning that weight in production would
cost **before** you change it there.

## Cache capacity and eviction

| Flag | Meaning |
|---|---|
| `--des-block-size` | tokens per cache block (the unit of matching and eviction) |
| `--des-kv-blocks` | cap per-replica capacity in blocks; unset means unbounded |
| `--des-instances` | number of engine replicas behind the router |
| `--des-overlap-weight` | `kv` policy: cache locality versus load balance |

Leave `--des-kv-blocks` unset to find the hit rate an infinite cache would give
you — the ceiling. Set it to study eviction pressure, which is where the
`evictions=` and `block-reuse=` fields in the report start moving and where a
policy that looked best on an unbounded cache can lose.

## A single engine

Set `--des-instances 1` to study one engine's automatic prefix caching as
temporal reuse across a stream, with no routing in the picture. This isolates
"does my traffic reuse anything at all, given this cache size?" from "is my
router sending reuse to the right place?", and it is worth answering in that
order.

## Next steps

- Feed it real content: [Workloads and traces](workloads.md).
- Give prefill and decode separate pools: [Disaggregated serving](disaggregation.md).
- Compare with the deployed policy: [KV-aware routing](../features/kv_aware_routing.md).
