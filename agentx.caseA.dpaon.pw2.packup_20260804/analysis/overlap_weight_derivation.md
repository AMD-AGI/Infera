# How `pw=2.0` was derived

The operator's instruction was to pick "a suitable value" for the kv-aware
weight from the AgentX workload's own distribution rather than reuse par8's
`pw=20`. This file is the derivation. **It is a derivation, not a measurement —
the run that was supposed to test it produced an empty cache view (see
`../README.md`), so nothing here has been confirmed against a result.**

## The cost function

`infera/router/policy/kv_event_aware.py:174-179` (Rust twin:
`rust/router/src/policy.rs`):

```python
def cost(t: RouteTarget) -> float:
    total = len(hashes_for[...])          # blocks in THIS request
    hits  = self._cache_hits(t, ...)      # blocks target t already holds
    return w_overlap * (total - hits) + w_mm * mm_miss + active(t)
```

with `active(t) = len(self._active_block_refs[t.route_key])` — a **refcounted
set of distinct in-flight block hashes**, deduped across requests that share a
prefix. No multimodal content in this trace, so `w_mm * mm_miss` is 0.

**Both surviving terms are in units of KV blocks.** That is what makes `w`
interpretable: it is the exchange rate between "prefill work I can skip" and
"queue I am willing to join".

Choosing between a warm target and a cold one, the cost difference is

```
Δcost = w × hits(warm) + [active(warm) − active(cold)]
```

so the policy switches away from the warm rank exactly when

```
active(warm) − active(cold)  >  w × hits
```

## The workload's numbers

Block size is **64** (`kv_block_size` from `/v1/workers`).

Measured over all **1,778** requests of the frozen corpus by walking each
session's `hash_ids` and accumulating the prefix (`spec/` holds the corpus
generator; the walk is 12 lines and reproduced in `../REPRODUCE.md` step 9):

| | p10 | p50 | p90 | mean |
|---|---|---|---|---|
| blocks per request | 581 | **1,158** | 2,282 | 1,315 |
| blocks still missing on a **warm** rank | 67 | **192** | 1,436 | 529 |
| ⇒ `hits` = the difference | 514 | **966** | 846 | — |

Cross-checked against what actually ran: the measured c8 run reports
mean `request_blocks = 1,436` in the router log and ISL p50 65,904 tok
(= 1,030 blocks) in `profile_export.jsonl`. Same order, so the corpus walk is a
fair model of the live request stream.

## Sizing the load term

At concurrency 8 with the prefill leg fanned out to 8 DP ranks
(`expand_targets`, `infera/router/policy/target.py`), the steady state is
**~1 in-flight request per rank** — confirmed by the run: in-flight max 8,
mean 6.38.

So `active(t)` for a rank holding one typical request is ~1,158 blocks; holding
two is ~2,316. That is the whole dynamic range the load term has at this
concurrency.

## Putting the two together

| `w` | cache gain `w × 966` | rank must hold this many blocks before the policy leaves it | ≈ in-flight requests |
|---|---|---|---|
| 0.5 | 483 | 483 | 0.4 |
| 1.0 | 966 | 966 | 0.8 |
| **2.0** | **1,932** | **1,932** | **1.7** |
| 3.0 | 2,898 | 2,898 | 2.5 |
| 5.0 | 4,830 | 4,830 | 4.2 |
| **20.0** (par8) | **19,320** | **19,320** | **16.7** |

**`pw=20` cannot lose.** It requires a rank to be sitting on ~17 concurrent
requests before cache locality yields — unreachable when the whole benchmark
runs 8 in flight. The load term is decorative; every request in a conversation
pins to one rank and the other seven idle.

That was harmless in the 20260803 run **because prefill had exactly one
target** (DPA off ⇒ `expand_targets` returns 1 target ⇒ every policy picks the
same thing). `pw=20` had therefore never been exercised on a fanned-out prefill
pool. Turning DPA on is what makes the weight matter.

**`pw=2.0` sits at ~1.7 in-flight requests**, which is the point where the
target load (1/rank) is respected but a *single* extra queued request is not
enough to abandon a 966-block prefix. That is the intended behaviour at c8×8
ranks: keep the conversation's affinity, spill only when a rank is genuinely
doubled up.

`dw` stays at **2.0**, unchanged from par8. Decode is memory-bound on KV and a
prefill-time hit does not help its inner loop — the class docstring
(`kv_event_aware.py:60-66`) says exactly this, and nothing in this workload
argues against it.

## What would confirm or refute this

None of the below was run.

1. **Fix the empty cache view first.** With `cache_hits ≡ 0` the weight is
   multiplied by a constant and any sweep measures nothing.
2. Then sweep `pw ∈ {1, 2, 5, 20}` at c8, reading three things per leg: TTFT
   p50, the token-weighted cache-hit rate, and the rank histogram from
   `router.log`. The prediction this derivation makes is a **non-monotone**
   TTFT curve — low `pw` sheds locality, high `pw` sheds parallelism.
3. The concurrency dependence is explicit in the table: the balancing `w`
   scales with in-flight-per-rank. A value tuned at c8 is **not** expected to
   hold at c16 or c32, and this file should not be cited as if it were
   concurrency-independent.
