# Arm 2 — prefill DP-attention ON (dp8), kv-aware weights re-derived

**Ran 2026-08-04 05:32:39–05:49:12 UTC**, one point (C=8, 900 s), same decode
leg, same corpus, same customer script.

## What changed from arm 1

| | arm 1 (DPA off) | arm 2 (this) |
|---|---|---|
| prefill DP-attention | **OFF** (pure TP8) | **ON, dp8** |
| prefill `--mem-fraction-static` | 0.70 | **0.70** (see below — 0.80 crashed) |
| `--chunked-prefill-size` passed | 65536 | 65536 |
| ...**resolved by the engine** | **65536** (not divided) | **8192** (÷ dp_size) |
| routable prefill targets | **1** | **8** |
| `--kv-prefill-overlap-weight` | 20.0 | **5.0** |
| `--kv-decode-overlap-weight` | 2.0 | **1.0** |

## Why 5.0 / 1.0 and not the 20.0 / 2.0 default

The cost function (`infera/router/policy/kv_event_aware.py:50`) is:

```
cost(w) = overlap_weight * (request_blocks - hits(w)) + active_blocks(w)
```

Both terms are in **KV blocks** (page_size 64). Measured from arm 1's 229
profiling requests:

| quantity | p50 | p90 | mean |
|---|---:|---:|---:|
| `request_blocks` | 1,092 | 2,283 | 1,263 |
| **missing blocks** (`request_blocks - hits`) | **269** | **1,805** | 621 |

`active_blocks` is the refcounted set of distinct in-flight blocks on that
worker — so one in-flight request of average size contributes ~1,263.

That sets the balance point:

| W | overlap term @p50 | load term wins once the worker holds… |
|---:|---:|---|
| 1 | 269 | 0.2 requests |
| 2 | 538 | 0.4 requests |
| **5** | **1,345** | **1.1 requests** |
| 20 | 5,380 | 4.3 requests |

**W=20 means locality outweighs load until a rank is already ~4 requests deep.**
At C=8 across 8 ranks that is the whole budget — the router would keep stacking
onto a warm rank while others idle. W=5 puts the crossover at ~1 request, so
locality decides between comparably-loaded ranks and load decides once one
starts to pile up. That is the intended regime here.

> The 20.0 default is documented as tuned for a ~0.89 hit rate. This workload's
> **realized** hit rate is 66 % (and 0 % on turn 0, which the scenario's
> `cache-bust` forces) — so the miss term is ~3× larger than the default assumes,
> and re-using 20.0 would over-weight locality by that same factor.

Decode 2.0 → 1.0 for the same reason and one more: the decode leg runs
`ChunkCache`, so its router-side KV view is empty and the overlap term is
identically zero regardless of weight. 1.0 is the honest "route by load" value.

## Result — C=8, both arms

| | arm 1 (DPA off) | **arm 2 (DPA on)** | delta |
|---|---:|---:|---|
| requests | 229 | **181** | −21 % |
| **TTFT p50** | 6,698 ms | **13,578 ms** | **+103 %** |
| **TTFT p90** | 23,775 ms | **32,506 ms** | +37 % |
| TTFT p99 | 33,681 ms | 44,036 ms | +31 % |
| ITL p50 | 13.26 ms | 14.07 ms | +6 % |
| E2E p50 | 13,874 ms | 22,754 ms | +64 % |
| **output throughput** | **258.5 tok/s** | **220.2 tok/s** | **−15 %** |
| input throughput | 19,899 tok/s | 15,664 tok/s | −21 % |
| server cache read/prompt | 66.1 % | **66.2 %** | flat |
| `submission_valid` | true | **true** | — |
| engine faults | 0 | **0** | — |

**DPA-on is worse on every latency and throughput axis at this concurrency.**
The cache hit rate is unchanged (66.1 → 66.2 %), which says the difference is
not a caching effect.

The mechanism is visible in the resolved chunk size. sglang divides
`chunked_prefill_size` by `dp_size` **only** under DPA (`server_args.py:4902`),
so the same CLI 65536 becomes **8192 per forward** here versus **65536** in arm
1 — an 8× smaller per-forward batch. For a workload whose ISL p50 is ~68 K
tokens, that is 8 chunks where arm 1 needed 1, and the per-chunk overhead is
paid 8×.

> **This is a two-variable comparison, not a clean DPA ablation.** DPA and the
> resolved chunk size moved together, because they are coupled by that division.
> Holding per-forward work equal would need `CHUNK=524288` under DPA — untested,
> and its activation peak would almost certainly not fit.

## Per-rank load balance — the reason to run this arm at all

Sampled every 15 s for the whole run (`results/dpa8_c8/rank_samples.jsonl`,
87 samples; analyser `scripts/rank_balance.py`).

```
DP0   236   21.1 %  #########################
DP1   211   18.8 %  ######################
DP2   211   18.8 %  ######################
DP3   152   13.6 %  ################
DP4   122   10.9 %  #############
DP5    72    6.4 %  #######
DP6    73    6.5 %  #######
DP7    43    3.8 %  ####

total=1,120  mean=140  min=43  max=236  max/min=5.49x  CV=0.494
per-tick delta CV: mean 1.454, median 1.258 over 44 active ticks
```

**The load is not balanced, and it skews monotonically toward the low ranks.**
Three snapshots show it worsening as the run proceeds:

| elapsed | max/min | CV |
|---|---:|---:|
| ~5 min | 2.91× | 0.342 |
| ~14 min | 4.77× | 0.493 |
| ~17 min (final) | **5.49×** | **0.494** |

A monotone DP0 > DP1 > … > DP7 gradient is **not** what a cost function
minimising `W*(miss) + active` produces from a symmetric workload — that would
scatter. The shape points at the dispatch order rather than the policy: ties
broken toward the lowest-indexed rank, plus `min()` being stable over a target
list that is presumably in rank order.

**This is a finding, not a conclusion.** What would settle it, and was not run:

1. Re-run with `--router-policy round-robin`. If the gradient persists, it is
   dispatch/tie-break, not the kv-aware cost function.
2. Parse the router's own `picked= cache_hits= active_blocks=` log lines for
   the per-decision inputs. **Not available for this run** — `/tmp/router.log`
   lives inside the container and was overwritten when the router was restarted
   to change the policy. Capture it *before* any restart next time.
3. Sweep W ∈ {1, 5, 20} at fixed concurrency and watch CV. If CV is flat in W,
   the weight is not the lever.

Note the sampler measures **prefill batch counts per rank**, not tokens. Ranks
handling longer prompts do more work per batch, so the token-level skew may
differ from 5.49×. Token-level accounting was not captured.

## The crash that preceded this run — GMU 0.80 under DPA

The first attempt at this arm used `GMU=0.80` (the value the spur Case-A kit
records as good) and **died 3 minutes in**:

```
:0:rocdevice.cpp:3582 ... HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 362 MB
Fatal Python error: Aborted
[2026-08-04 05:17:53 DP2 TP2 EP2] Prefill batch ... token usage: 0.05
```

`token usage: 0.05` — the KV pool was 5 % full. **Not KV exhaustion; DP-attention
activation memory.** Under dp8 each rank holds its own chunk activations, and
this corpus's ISL p99 of 245 K is well above what the Case-A kit exercised.

Fixed by **lowering** `mem-fraction-static` to 0.70 — the same
counter-intuitive direction the Case-A kit documents (prefill activation OOM ⇒
*lower* GMU; decode retract ⇒ *raise* it).

Artifacts kept: `logs/prefill_dpa8.gmu080_crashed.log.gz`,
`results/dpa8_c8/rank_samples.attempt1_crashed.jsonl`.

## Files

| path | what |
|---|---|
| `results/dpa8_c8/summary.csv` | the customer script's own row |
| `results/dpa8_c8/profile_export_aiperf.{csv,json}` | full aiperf metrics |
| `results/dpa8_c8/profile_export.jsonl.gz` | per-request records |
| `results/dpa8_c8/rank_samples.jsonl` | **87 × 15 s per-rank load samples** |
| `results/dpa8_c8/rank_samples.attempt1_crashed.jsonl` | 35 samples from the GMU 0.80 attempt |
| `results/dpa8_c8/dpa8_kvd_{before,after}_*.json` | kvd counters |
| `analysis/rank_balance_dpa8.txt` | the analyser's full output |
| `scripts/sample_ranks.sh` | the sampler |
| `scripts/rank_balance.py` | the analyser |
| `scripts/router_tuned.sh` | kv-aware @ 5.0 / 1.0 |
| `scripts/env_prefill_dpa8.sh` | DPA=1 CHUNK=65536 GMU=0.70 |
