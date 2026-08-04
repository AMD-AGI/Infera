# round-robin vs kv-aware — why this A/B is inconclusive

## What the two policies are

They are **mutually exclusive** — `infera/router/policy/factory.py:56-59`
registers exactly two builders, and `infera/server/launch_rust.py:30` lists the
same two as the Rust backend's supported set. There is no "kv-aware with a
round-robin tie-break"; selecting round-robin turns cache-locality scoring off
entirely.

Both fan a rank-multiplexed worker (`dp_size>1`, `dp_rank is None`) into one
target per DP rank via `expand_targets`
(`infera/router/policy/target.py`), then differ only in how they pick:

- `RoundRobinPolicy` — one counter per candidate-set, `targets[idx % len]`.
- `KvEventAwarePolicy` — `min` over
  `w × (blocks − hits) + active_blocks`.

## Why this A/B is only possible with prefill DPA on

With the 20260803 posture the prefill leg had `dp_size = null`, so
`expand_targets` returned **one** target for the prefill pool. Both policies
then pick the same thing, every time. **The router policy was a no-op on the
prefill side of that entire run** — only the 8 decode ranks were being rotated
or scored.

Turning prefill DPA on is what creates 8 prefill targets and makes the policy a
real variable. That is why the deployment change and the policy change were
made together, and it is also why they cannot be separated here.

## The dp_rank does reach the engine — verified by code read

Not router-side bookkeeping. Three mechanisms, all present in both the Python
and Rust routers:

| mechanism | Python | Rust |
|---|---|---|
| `X-Data-Parallel-Rank` header | `infera/router/dp_routing.py:19-25`, applied `disagg.py:56-62` | `rust/router/src/dp.rs:10`, applied `proxy.rs:146-148` |
| `disagg_prefill_dp_rank` in the decode body | `dp_routing.py:28-35`, applied `disagg.py:261,604` | `disagg.rs:98-100` |
| `bootstrap_room` rewritten so `room % dp_size == dp_rank` | `dp_routing.py:39-46` | `disagg.rs:77` |

The third is load-bearing: SGLang's default `follow_bootstrap_room` balancer
derives the prefill rank from `bootstrap_room % dp_size` and rejects the leg
with `KVTransferError` if it landed elsewhere.

## Round-robin: worked exactly as specified

From `env/router.roundrobin.log.gz`, 239 prefill picks over the run:

```
Prefill dp0 30   dp1 30   dp2 30   dp3 30   dp4 30   dp5 30   dp6 30   dp7 29
Decode  dp0 30   dp1 30   dp2 30   dp3 30   dp4 30   dp5 30   dp6 30   dp7 29
```

Perfectly uniform, and the prefill and decode pools rotate **independently** —
the per-candidate-set counter doing what its docstring says (a single shared
counter would advance twice per PD request and pin the parity).

Its cache consequence is the expected one: **12.4 %** token-weighted, the
lowest of the three legs. Spraying a conversation's turns across 8 ranks
destroys the prefix locality this workload is built on.

## kv-aware pw=2: the policy never ran

From `env/router.kvaware_pw2.log.gz`, 153 prefill picks, of which 37 carried a
non-empty prompt:

```
mean request_blocks : 1436     <- correct; the trace's own mean is 1,315
mean cache_hits     :    0.0   <- on every one of the 37
max  active_blocks  :    0     <- on every one of the 37
```

Both terms of the cost function were identically zero, so `min()` degenerated
to "first target in the list". The rank histogram

```
dp0 65   dp1 45   dp2 26   dp3 10   dp4 4   dp5 3   dp6 0   dp7 0
```

is that degenerate ordering, **not** the locality/load trade-off `pw=2.0` was
computed to produce. I initially read this skew as the weight working; that
reading was wrong and is retracted.

The remaining 116 picks show `request_blocks=0` because they are short probe /
warm-up requests below one 64-token block.

### What was eliminated

| hypothesis | probe | outcome |
|---|---|---|
| router subscribed to a stale port | `/v1/workers` `kv_events_endpoint` vs the leg's own `tcp://*:N` | both **26803** — match |
| per-rank publishers never bound | `ss -ltn` in the container | **26803–26810**, all 8 listening |
| ZMQ link down | `ss -tn \| grep 26803` | **ESTAB**, both directions |
| engine has no cache to publish | `#cached-token > 0` in prefill batch lines | **60 / 1129** batches |
| kvd (L3) dead | `statctl` before → after | gets **576 → 1,864**, hits == gets, misses 0 |

The engine caches, the socket is up, the ports are right — and the router's
view is still empty. **Cause not determined.**

### The next two probes, in order

1. **Does the prefill leg emit `BlockStored` at all?** Its log contains
   `kv_events` twice, both at startup (`KvEventPublisher started:
   endpoint=tcp://0.0.0.0:5557` — note this is infera's *own* publisher on 5557,
   a different socket from the sglang-side block at 26803+rank). Confirm which
   socket is supposed to carry `BlockStored` for a **prefill-role** leg.
2. **If events do flow, compare hashes.** The router computes block hashes with
   its own `BlockHasher` + tokenizer (`--kv-tokenizer-path`); the engine
   publishes ids from its own tokenization. A mismatch yields exactly this
   signature: correct `request_blocks`, zero `hits`. The 20260803 kit's
   `feature_proof.sh` and par8's `cache_view.py` are the existing tools for
   this.

Until one of those resolves, **no statement about kv-aware's effect on this
workload is supported by this run**, and `pw=2.0` remains a derivation
(`overlap_weight_derivation.md`) with no measurement behind it.

## What the A/B *did* establish

1. **Round-robin is measurably wrong for this workload** — 12.4 % vs 27.4 %
   token-weighted cache hit, at equal offered demand. That comparison is valid
   because the degenerate kv-aware leg still happened to concentrate load
   (dp0-heavy), which preserves more prefix reuse than uniform spraying does.
   The mechanism is affinity, not the policy's scoring.
2. **Both policies sustain the load with zero errors** — 271 requests, 0
   failures, 0 cancellations, 0 context overflows.
3. **Neither reaches the 20260803 posture's 50.3 %**, which ran with prefill
   DPA *off* and therefore a single prefill target holding one shared radix
   tree. Fanning prefill into 8 ranks partitions that tree 8 ways; whether a
   correctly-functioning kv-aware policy recovers the loss is exactly the
   question this run failed to answer.
