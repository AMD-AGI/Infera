# Routing distribution — the point of this arm

Two independent measurements of the same thing: does round-robin actually spread
work across DP ranks, and does kv-aware actually not?

## Measurement 1 — the router's own pick log

From `results/armA2_router.log.gz`, every `pick policy=round-robin` line, counted
by target:

| target | picks | | target | picks |
|---|---:|---|---|---:|
| `10.245.156.167:30000#dp0` (prefill) | 291 | | `10.245.152.164:30001#dp0` (decode) | 291 |
| `…30000#dp1` | 291 | | `…30001#dp1` | 291 |
| `…30000#dp2` | 291 | | `…30001#dp2` | 291 |
| `…30000#dp3` | 290 | | `…30001#dp3` | 290 |
| `…30000#dp4` | 290 | | `…30001#dp4` | 290 |
| `…30000#dp5` | 290 | | `…30001#dp5` | 290 |
| `…30000#dp6` | 290 | | `…30001#dp6` | 290 |
| `…30000#dp7` | 290 | | `…30001#dp7` | 290 |

**16 targets, 290–291 picks each.** A spread of exactly 1, which is the remainder
of 2,323 requests over 8 ranks — i.e. the rotation is *perfect*, not merely even.

Reproduce from this kit:

```bash
zcat results/armA2_router.log.gz | grep -oE 'picked=[^ ]+' | sort | uniq -c
```

### The two pools rotate independently

Note that prefill `#dp0` and decode `#dp0` both got 291 — the counters did not
interleave. That is deliberate: a PD request calls `pick()` **twice**, once per
pool. A single shared counter would advance by two per request, and with exactly
two pools the parity would pin every prefill pick to the same target. `RoundRobin`
keys its counter on the *candidate set*
(`infera/router/policy/round_robin.py:26-32`), so each pool keeps its own
rotation. This run is that design working.

## Measurement 2 — what the engine actually computed

The pick log says what the router *decided*. The engine log says what each rank
*did*. From `logs/armA2_prefill.log.gz`, prefill batches by DP rank:

| DP0 | DP1 | DP2 | DP3 | DP4 | DP5 | DP6 | DP7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 515 | 495 | 508 | 493 | 508 | 502 | 475 | 499 |

Mean 499, range 475–515, **±4 %**. (Batch count exceeds pick count because a
155K-token prompt is chunked into many forward batches.)

```bash
zcat logs/armA2_prefill.log.gz | strings \
  | grep -oE 'DP[0-7] TP[0-7] EP[0-7]\] Prefill batch' | grep -oE 'DP[0-7]' \
  | sort | uniq -c
```

## The contrast that makes this worth measuring

Same branch, same image, same workload family, same 8-rank DPA prefill leg — only
the routing policy differs:

| routing | DP0 | DP1 | DP2 | DP3 | DP4 | DP5 | DP6 | DP7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **kv-aware** (earlier acceptance run) | 1,405 | 1,497 | **1** | **1** | **1** | **1** | **1** | **1** |
| **round-robin** (this run) | 515 | 495 | 508 | 493 | 508 | 502 | 475 | 499 |

Under kv-aware, **six of eight prefill ranks did one batch each across a full
run**. Corroborated at the time by the router's own gauges:
`infera_policy_cache_view_size` read dp0=45,813 / dp1=45,833 / dp2..dp7 = **0**,
and `infera_policy_active_blocks` likewise only dp0/dp1.

### What this settles

Two hypotheses were live after that run:

| | claim | verdict |
|---|---|---|
| **A** | dp2..dp7 are not computing at all — DP-attention is not truly 8-way | **refuted** — the same ranks take ~500 batches each the moment the router sends work |
| **B** | all 8 compute, but only 2 ranks' KV events reach the router | **refuted as stated** — the ranks were idle, not silently working |

The actual mechanism is a **self-reinforcing affinity loop**: kv-aware's cost
function is `w·(request_blocks − hits) + active_blocks`. A rank with an empty
cache view has zero hits, so its cost is maximal, so it is never the cheapest
candidate, so it receives no traffic, so it stores no blocks, so its view stays
empty. Ranks dp0/dp1 won the first few picks and the loop closed behind them.

This is not a bug — it is the policy doing exactly what it is designed to do (89 %
of every request is a shared prefix, and concentrating that prefix on few ranks
maximises hit rate). It is a **property worth knowing**, because it means the
effective prefill parallelism under kv-aware on a nested-prefix workload is
whatever the first few picks establish, not `dp_size`.

## The cost of spreading

Splitting work across 8 ranks means 8 ranks hold activations concurrently. In the
seconds before attempt 1's abort, **4–5 DP ranks were prefilling at once**; under
kv-aware only 1–2 ever are. That is why this arm cannot run at
`--mem-fraction-static 0.80` and arm B / the acceptance run can. Full mechanism in
`../notes.md` §1 and `../README.md`.

| | kv-aware | round-robin |
|---|---|---|
| concurrent prefilling ranks | 1–2 | **4–5** |
| workable `mem-fraction-static` | 0.80 | **0.70** |
| KV pool (`max_total_num_tokens`) | 2,939,264 | 2,387,200 (−19 %) |

## What is NOT shown here

`infera_router_picks_total` and `infera_router_pick_cache_hits` are **empty** in
`results/armA2_router_metrics.txt` — HELP/TYPE headers with no samples. Those
counters are instrumented in the kv-aware policy path only; round-robin does not
emit them. The pick log above is the substitute, and it is strictly more detailed
(per-target counts rather than a histogram).

`infera_request_duration_seconds` *is* populated and shows 2,323 `outcome="ok"`
observations, 2,322 of them under 5 ms — that is the router's own decision +
dispatch latency, not the end-to-end request, so it measures routing overhead
rather than service time.
