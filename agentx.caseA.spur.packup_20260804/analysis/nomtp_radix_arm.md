# Arm 3 — decode MTP OFF, decode radix cache ON

**Ran 2026-08-04 06:56:45–07:11:45 UTC**, one point (C=8, 900 s), same nodes,
same corpus, same customer script, **prefill restored to arm 1** (DPA off) and
**router restored to arm 1 weights** (20.0 / 2.0).

## MTP and the decode radix cache are ONE switch, not two

This is the finding that shapes the whole arm. They cannot be varied
independently — `infera/engine/sglang/args.py:261-278`:

```python
if (known.enable_kv_events and mode == "decode"
        and transfer_backend == "mooncake"
        and "--disaggregation-decode-enable-radix-cache" not in remaining):
    if sglang_parsed.speculative_algorithm is not None:
        logger.info("...incompatible with --speculative-algorithm %s; "
                    "not appending it...")          # <-- arm 1 lands here
    else:
        remaining.append("--disaggregation-decode-enable-radix-cache")
```

SGLang rejects the decode radix cache outright under speculative decoding, so
infera declines to append the flag whenever EAGLE is on. **Turning MTP off is
what legalises the decode radix cache.** Asking for "MTP off" and asking for
"decode radix cache on" are the same request against this stack.

Verified in the resolved `server_args` of the spawned subprocess, not inferred:

| field | arm 1 (MTP on) | arm 3 (MTP off) |
|---|---|---|
| `speculative_algorithm` | `'EAGLE'` | *absent* |
| `disable_radix_cache` | `True` | **`False`** |
| `disaggregation_decode_enable_radix_cache` | `False` | **`True`** |
| `Tree cache initialized: impl=` | `ChunkCache` ×8 | **`RadixCache` ×8** |
| `accept len` occurrences in log | 5,564 batches | **0** |

## The third variable, and how it was pinned

`MTP=0` moves **four** things in the leg script, not one:

| # | what | arm 1 | arm 3 | handling |
|---|---|---|---|---|
| 1 | EAGLE spec-dec | on | off | **under test** |
| 2 | decode tree cache | ChunkCache | RadixCache | **under test** (coupled to #1, see above) |
| 3 | `--num-reserved-decode-tokens` | 256 | *would be 512* | **pinned to 256** |
| 4 | `--disable-custom-all-reduce` | on | on | already decoupled |

**#3 would have ridden along silently.** `RESERVED_TOK` lives *inside* the
`MTP_ARGS` block (`glm52_leg_spur_mtp.sh:198`), so dropping MTP drops the flag
and SGLang's own default (**512**, verified from `ServerArgs`) takes over — a
2× change in reserved decode tokens attributed to nothing. Pinned via
`EXTRA_ARGS="--num-reserved-decode-tokens 256"` and confirmed present in the
spawned argv.

**#4 was a trap that had already been fixed.** The leg script's comment records
that `CUSTOM_AR` used to follow `MTP`, which meant `MTP=0` silently re-enabled a
known-broken aiter all-reduce kernel on gfx950. It is now independent; both arms
pass `--disable-custom-all-reduce`.

## Result — C=8, arm 1 vs arm 3

| metric | arm 1 (MTP + ChunkCache) | arm 3 (no MTP + RadixCache) | delta |
|---|---:|---:|---:|
| requests (profiling) | 229 | 175 | −23.6 % |
| ISL mean | 80,811 | 77,956 | −3.5 % |
| **TTFT p50** | 6,698 ms | **4,224 ms** | **−36.9 %** |
| **TTFT p90** | 23,775 ms | **16,999 ms** | **−28.5 %** |
| TTFT p99 | 33,681 ms | **26,061 ms** | −22.6 % |
| **ITL p50** | **13.26 ms** | 22.21 ms | **+67.5 %** |
| ITL p90 | **18.49 ms** | 24.86 ms | +34.5 % |
| E2E p50 | **13,874 ms** | 14,375 ms | +3.6 % |
| E2E p90 | **38,774 ms** | 69,078 ms | +78.2 % |
| **output throughput** | **258.5 tok/s** | 211.0 tok/s | **−18.4 %** |
| server cache read/prompt | 50.8 % | 49.5 % | flat |
| `submission_valid` | true | **true** | — |
| engine faults | 0 | **0** | — |
| client errors | 0 | 1 (`ClientOSError` 104, 0.57 %) | — |

**The two axes move in opposite directions — and it is ONE cause, not a
trade-off.** The full chain is below; every link is measured.

### Why TTFT improved: the bench is CLOSED-LOOP, so arrival rate fell

**The TTFT gain is not the server getting faster. It is the load getting
lighter.**

`aiperf/timing/strategies/agentic_replay.py` issues turn N+1 **from turn N's
return callback** (`handle_credit_return`, :1214 — *"if not the final turn,
dispatch the next turn honoring trace `delay_ms`"*), and a lane's slot is
*"held until its whole TREE drains"* (:138). `--concurrency 8` is therefore
**8 serial trajectory chains in parallel**, and trace `delay_ms` is think-time
layered *on top of* the response — not an absolute timeline. **Arrival rate is
an output of server speed, not an input.**

Measured from `request_start_ns` (`scripts/pd_bottleneck.py --jsonl`):

| | requests | span | **arrival rate** | inter-arrival p50 |
|---|---:|---:|---:|---:|
| arm 1 | 231 | 897.3 s | **0.257 req/s** | 2.48 s |
| arm 3 | 176 | 897.2 s | **0.196 req/s** | 3.35 s |

**−24 % arrival rate at identical prefill configuration.** The chain closes:

```
MTP off  →  every token a full forward       →  ITL +68 %  (accept len was 3.06)
         →  each turn returns later          →  lane turnover slows
         →  arrival rate 0.257 → 0.196 req/s (-24 %)
         →  prefill #queue-req 2.00 → 0.50
         →  TTFT p50 -37 %
```

**Consequence: TTFT cannot be read from this bench on its own.** Making decode
slower "improves" it. Always report arrival rate beside it.

### The bottleneck did not move: both arms are prefill-bound

Five-state queue-depth analysis (`analysis/pd_bottleneck_arm1_vs_arm3.txt`).
A queue grows immediately upstream of the slowest stage and stays near-empty
downstream of it, so the verdict is read from *which* queue is backed up.

| state | counter | side | arm 1 | arm 3 |
|---|---|---|---|---|
| 1 prefill input | `#queue-req` | prefill | **mean 2.00, p95 8, max 12** | mean 0.50, p95 2, max 7 |
| 2 prefill outbound | `#inflight-req` | prefill | mean 1.69, p95 4, max 8 | mean 1.21, p95 3, max 5 |
| 3 decode admission | `#prealloc-req` | decode | **0 on all 8 ranks** | **0 on all 8 ranks** |
| 4 decode transfer-in | `#transfer-req` | decode | max 1–3 per rank | max 0–1 per rank |
| 5 decode running | `#running-req` | decode | mean ≈1.0/rank | mean ≈1.0/rank |
| — | `#retracted-req` | decode | **0** | **0** |

- **prefill-bound, both arms.** State 1 is the only deep queue; state 2 is
  shallow; decode never queued at its admission gate.
- **Transfer exonerated.** The decisive test is the *sender's* outbound queue —
  a stuck transfer path backs up state 2. It is shallow in both arms
  (p95 ≤ 4), and `MC_FORCE_TCP` / `GID is NULL` count 0 on both legs.
- **Decode nowhere near saturation.** Per-rank cap is 256 (the DP-adjusted
  value); occupancy is **0.13 %** (arm 1) and **0.27 %** (arm 3), with KV pool
  `token usage` p95 of 3–10 %.

> **Two counting traps, both hit while producing this table.**
>
> **Decode counters are PER RANK.** Decode runs dp8, so each of the 8 schedulers
> prints its own `#running-req`. Averaging across ranks yields a number that is
> not any queue's depth. Per-rank running ≈1.0 reads like an idle engine; the
> cross-rank per-second sum is **p50 6, max 10** for arm 3 — i.e. the C=8 load
> is present and simply spread over 8 ranks. Prefill has DPA **off**, so its
> lines carry no `DPn` tag and its two counters are a single global queue.
>
> **States 2 and 4 have different start points, same end point.** Prefill
> `#inflight-req` starts when *prefill finishes computing*; decode
> `#transfer-req` starts when *decode sends the KV landing address*
> (`decode.py:1090-1111`, `send_metadata`), and both end at `KVPoll.Success`
> (`prefill.py:757` / `decode.py:1728`). Prefill's window therefore *contains*
> decode's, plus the wait for the address to arrive. State 2 > state 4 is the
> normal ordering, **not** evidence that decode is failing to keep up.

### The ITL cost is the direct price of removing EAGLE

Arm 1's measured mean accept length was **3.06** tokens/step; without it every
token costs a full forward. 13.26 → 22.21 ms is **1.68×**, close to what a
3.06-accept draft predicts once verify overhead is netted out.

**Throughput follows ITL**: −18.4 %. For this workload (OSL mean 1,121) the
decode loop dominates total time. E2E p50 is essentially flat (+3.6 %) while E2E
p90 blows out +78 %, so the long-output tail is where losing MTP hurts most.

## Per-turn cache decomposition — both arms

| segment | arm 1 | arm 3 |
|---|---:|---:|
| turn 0 | n=53, **0.0 %** | n=45, **0.0 %** |
| turns 1–2 | n=72, 70.1 % | n=60, 73.0 % |
| turns 3+ | n=106, 66.4 % | n=71, 65.6 % |
| overall | 50.8 % | 49.5 % |

Turn 0 is 0.0 % in both — the scenario's `--cache-bust first_turn_prefix` lock,
unchanged by anything on the serving side.

**The decode radix cache did not raise the measured hit rate.** That is
consistent with what the PD architecture implies (prefix reuse is decided on the
prefill leg, and `usage.prompt_cache_read_tokens` reports the prefill-side
match) but this arm does **not** isolate what the decode radix cache
contributes, because it could not be turned on without turning MTP off.

## ⚠ What this arm does NOT establish

**It is not an MTP ablation, and it is not a radix-cache ablation.** Two things
move together and the coupling is enforced upstream in SGLang, not by choice
here. Any attribution of the TTFT gain to the radix cache, or of the ITL loss to
MTP alone, is unsupported by this data — the ITL half is well-explained by
EAGLE's absence, the TTFT half is not yet explained by anything measured.

To separate them would need SGLang to accept
`--disaggregation-decode-enable-radix-cache` alongside `--speculative-algorithm`
(it does not), or a third configuration with MTP off and the radix cache
*explicitly suppressed* — which infera has no flag for and would need a code
change.

**Nothing here measures what the decode radix cache contributes.** The
server-measured cache rate is flat, but that metric reports the *prefill*-side
prefix match, so it would be flat either way. The decode radix cache's effect is
simply unobserved.

**The arms are not iso-load.** Because the driver is closed-loop, arm 3 offered
24 % fewer requests in the same window. Every queue-depth number therefore
compares two different load levels; the *bottleneck verdict* (prefill-bound,
transfer exonerated) holds in both independently, but the queue *magnitudes* are
not directly comparable. An iso-load comparison would need an open-loop driver
(`--request-rate` instead of the agentic-replay strategy), which would no longer
be the customer's benchmark.

**Not run, and would sharpen this:** arm 3 at `--num-reserved-decode-tokens 512`
(the unpinned default) to confirm the pin was not itself doing work.

## The 1 client error

`ClientOSError(104, 'Connection reset by peer')`, 1 of 176 (0.57 %). **Both
engine legs report zero faults** (`HSA_STATUS_ERROR` / `Fatal Python error` /
`Traceback` / `Memory access fault` all count 0), so this is a client-side
socket reset, not an engine failure. aiperf's error-adjusted metrics are
reported alongside the raw ones in the CSV; the difference is within noise
(TTFT p50 4,249 vs 4,224 ms).

## Files

| path | what |
|---|---|
| `scripts/pd_bottleneck.py` | five-state queue analyser + arrival-rate probe |
| `analysis/pd_bottleneck_arm1_vs_arm3.txt` | its output for both arms |
| `results/nomtp_c8/summary.csv` | the customer script's own row |
| `results/nomtp_c8/profile_export_aiperf.{csv,json}` | full aiperf metrics |
| `results/nomtp_c8/profile_export.jsonl.gz` | per-request records |
| `results/nomtp_c8/nomtp_kvd_{before,after}_*.json` | kvd counters |
| `scripts/env_decode_nomtp.sh` | MTP=0 + the `RESERVED_TOK` pin |
| `scripts/run_nomtp.sh` | C=8 wrapper |
| `logs/decode_nomtp.log.gz` | decode leg (RadixCache ×8, zero `accept len`) |
