# Every par8.yaml knob vs. what was measured

The point of this file: a workload YAML states *intent*. Whether the engine and
driver reproduced that intent is a separate question, answered only by the raw
samples. Each row below gives the knob, the conversion rule, and the measurement.

Sampler reference values were regenerated offline with seed 1337 over 50,000
draws using the driver's own `PercentileSampler`, so "expected" is the
distribution the config actually specifies — not the three percentiles as read.

---

## Request profile — all UNCHANGED from Case A

### `input_tokens: {p50: 74000, p90: 155000, p99: 235000}`

| | p50 | p90 | p99 | mean | max |
|---|---|---|---|---|---|
| sampler (offline, n=50k) | 74,021 | 153,284 | 230,557 | 86,035 | 477,356 |
| **measured (sustain)** | **75,440** | 157,056 | 230,669 | **86,888** | 260,013 |

Agreement to **~1 %** on p50/p90/p99 and mean. The workload shape reproduced.

`max` differs by design: `max_input_tokens: 260000` clamps every draw
(`sampling.py:177`, `min(self.input.sample_int(), self.max_input_tokens)`), and
the measured max of 260,013 is that clamp plus the chat-template overhead the
tokenizer adds around the content.

> **The sampler's unclamped vmax is 540,182** — more than 2× the engine's
> context. The clamp is doing real work on every tail draw, not decorating.

### `output_tokens: {p50: 320, p90: 3300, p99: 17000}`

| | p50 | p90 | p99 | mean | max |
|---|---|---|---|---|---|
| sampler (offline) | 321 | 3,327 | 17,098 | 1,425 | 305,464 |
| **measured (sustain)** | **316** | 2,852 | 10,898 | 1,087 | 22,371 |

p50 agrees to 1.6 %. **p90 is 14 % low and p99 is 36 % low** — and this is
expected, not a defect. Two mechanisms, both mechanical:

1. **`max_prompt_tokens`/context truncation of the joint tail.** A draw needing
   input ≈ 260K *and* output ≈ 17K exceeds the 262,144 context and is rejected
   before generation (that is the entire error class — see
   `sli_percentiles.md`). Those samples are removed from the *completed* set,
   and they are exactly the largest-output ones.
2. **The measured value is a client-side re-tokenisation, not the engine's
   count.** The driver accumulates streamed text and re-encodes it
   (`agent_throughput.py:2305`), so token-boundary differences shift it. It
   never reads `usage.completion_tokens`.

**Output length control itself is exact.** `max_tokens = generation_length` with
`ignore_eos: true` (`agent_throughput.py:2229-2236`) — no `stop` field, so
`max_tokens` is the sole terminator. Neither early stop nor overrun is possible.

> **The 31-token spike is the sampler's floor, not an engine early-stop.**
> 303 of 2,768 completions (11 %) are exactly 31 tokens. `PercentileSampler`
> clamps at `vmin = p50²/p90 = 320²/3300 = 31.03 → 31` (`sampling.py:106`), and
> offline the same clamp catches **9.9 %** of draws. Measured 11 % ≈ expected
> 9.9 %. *(Measured min is 30, one below the floor — attributable to the
> re-tokenisation above; not separately verified.)*

### `turns_per_session: {p50: 3, p90: 20, p99: 103}`

Sampler mean **9.6** turns (not 3 — the p99 tail dominates, which is why the
window is sized in hours). Measured: 2,884 requests over 427 sessions ≈ **6.8
requests/session**, and 26 of 32 sessions were **still alive** at cutoff, so the
realised mean is a right-censored underestimate. Consistent, not comparable.

### `inter_turn_delay_s: {p50: 4, p90: 31, p99: 240}`

| | p50 | p90 | p99 | mean |
|---|---|---|---|---|
| sampler (offline) | 4.0 | 31.5 | 241.4 | 18.3 |
| **measured inter-arrival** | **4.1** | 29.5 | 202.5 | 16.8 |

Near-exact at p50. Measured inter-arrival is *response time + think time*
observed per session, so the tail compresses slightly against pure think time.
`MAX_THINK_TIME` (900 s) shows up as the measured max of exactly 900.0 s.

### `cache_hit_rate: 0.89`

| | value |
|---|---|
| ideal (construction) | 88.98 % |
| **measured actual** | **88.94 %** |
| efficiency | **100.0 %** |
| evicted | 373,732 tok (0.2 %) |

Exact. Requires `--enable-cache-report` on the leg or this reads 0.

## Offered load — the three CHANGED knobs

| knob | Case A | par8 | measured |
|---|---|---|---|
| `initial_sessions` | 32 | **8** | 8 at t=0 |
| `max_sessions` | 128 | **32** | **reached 32**, steady 21–26 |
| `max_inflight` | 48 | **24** | **max 22**, mean 12.1 — never pinned |
| `new_session_rate` | 0.10 | 0.10 *(held)* | 419 rate-based births |

**The interaction is the story.** `new_session_rate: 0.10` was solved for N=32
via Little's law, `N = rate × E[turns] × (E2E + E[delay])`. Holding it while
lowering `initial_sessions` to 8 means births push the population toward 32
regardless of the start. The offline preview said so; the run confirmed it.

So **`initial_sessions: 8` set only the starting point, not the operating
point.** The run's actual concurrency is a *measurement*: 21–26 live sessions,
12.1 mean in-flight. Anyone reading "8 sessions" off the filename is wrong.

Solving with the measured values: `0.10 × 9.6 × (15.8 + 16.8) = 31.3` ≈ the
observed cap of 32. Little's law closes.

## Measurement window

| knob | value | measured |
|---|---|---|
| `ramp_duration` | 400 | 400.0 s, 174 reqs, excluded |
| `sustain_duration` | 3600 | 3,600.0 s, **2,671 reqs**, 0.74 QPS |
| *(drain)* | — | 4.7 s, 5 reqs |
| **total** | 4,000 | **4,005.1 s** |

`ramp_duration` is a **warm-up exclusion window, not a load ramp** — nothing
ramps in closed-loop mode. Its job is to let the shared prefix become resident;
the ramp-phase TTFT p99 of 36.8 s vs sustain's 9.1 s shows it earning its keep.

## Speculative-decoding accounting knobs

| knob | value | note |
|---|---|---|
| `acc_len` | 1.56 | **Only used for the driver's MTP-compensated TPS arithmetic.** Does not configure the engine. |
| `mtp_draft_tokens` | 5 | ditto — the engine ran `--speculative-num-draft-tokens 4` |
| `mtp_overhead_factor` | 1.0 | ditto |

**These three are bookkeeping, not configuration**, and two of them disagree
with the deployment. The engine's measured acceptance was **2.02** per request
(vs. the YAML's 1.56) with **4** draft tokens (vs. 5). So the driver's
"Generation: 109.0 tok/s (MTP compensated)" is computed with the wrong
multiplier and should not be quoted as a throughput result. The honest number is
the uncompensated per-request rate: **67.3 tok/s/request** (= 1/TPOT).

## SLA block — documentation only

`args.sla_cfg` is parsed and never consumed; `runner.py`'s gate reads its own
CLI flags, which were not passed. Evaluated manually:

| target | value | measured | verdict |
|---|---|---|---|
| `e2e_p50_ms` | 4,500 | **7,400** | **FAIL 1.64×** |
| `ttft_p90_ms` | 30,000 | **4,903** | **PASS 6.1×** |
| `success_rate` | 0.97 | **0.988** | **PASS** |

The E2E miss is a **latency-floor spec measured under load**. The solo kits
established that 4,500 ms is met only at concurrency 1; at 12 mean in-flight
this target is not the right bar. See `README.md` in this directory.

## Misc

| knob | value | note |
|---|---|---|
| `gpus: 8` | per-GPU reporting divisor only | |
| `window: 30` | metrics smoothing | the `prefill_tps` series is 30 s-smoothed |
| `random_seed: 1337` | same as Case A and both solo runs | |
| `system_prompt_len: 2000` | **not sent** in profile mode | prompts are cut from the shared base |
| `tokenizer` | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | the one field changed besides the three knobs |
