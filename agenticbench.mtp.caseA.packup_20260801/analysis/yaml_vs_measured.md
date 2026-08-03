# Every workload knob against its measured value

Source YAML: `spec/caseA_full.yaml` (derived from the bench repo's
`agent/workloads/glm52_crxx_caseA.fix.yaml` @ `1cf01cb`).
Run: `caseA_armB/2026-08-01-15-37-08`.

A knob with no measured counterpart is not padded with a guess — it carries its
**blast radius** instead: what silently changes if it is wrong.

---

## `profile:` — the percentile-calibrated request shape

These are the knobs the driver *realizes* by sampling, so each has a direct
measurement.

| knob | nominal | measured | Δ | conversion mechanism |
|---|---|---|---|---|
| `input_tokens.p50` | 74,000 | **73,618** | −0.5 % | driver samples a 3-point log-interpolated distribution, then pads/cuts the prompt to the sampled length |
| `input_tokens.p90` | 155,000 | **152,264** | −1.8 % | same |
| `input_tokens.p99` | 235,000 | **225,721** | −4.0 % | same; the negative bias at the tail is the `max_input_tokens` clamp pulling the top 0.1 % down |
| `output_tokens.p50` | 320 | **299** | −6.6 % | sampled length becomes `max_tokens` on the request; the model may stop earlier |
| `output_tokens.p90` | 3,300 | **2,791** | −15.4 % | same |
| `output_tokens.p99` | 17,000 | **9,688** | **−43.0 %** | **truncated by the 240 s client timeout, not by the model** — see blast radius below |
| `turns_per_session.p50/p90/p99` | 3 / 20 / 103 | **not directly emitted** | — | driver samples a turn budget per session and retires it when exhausted; only `lifetimes` (min 16 s, max 3,575 s, mean 709 s) and the 79 retirements are persisted. p99=103 turns is **window-censored** — at a 14.1 s mean inter-arrival, 103 turns needs ≈24 min of pure think time and cannot complete inside 3,600 s |
| `inter_turn_delay_s.p50` | 4 | **4.2** | +5 % | `new_inter_arrival_times` = response + think time |
| `inter_turn_delay_s.p90` | 31 | **27.3** | −12 % | same |
| `inter_turn_delay_s.p99` | 240 | **158.7** | −34 % | same; censored — a 240 s delay late in the window never completes |
| `cache_hit_rate` | 0.89 | **0.8882 actual / 0.8899 ideal** | −0.2 % | driver builds each turn to nest in the session's prefix; the server reports `cached_tokens` (needs `--enable-cache-report`) |
| `max_input_tokens` | 260,000 | **260,013 = observed max** | clamp active | hard cut before send; 0.1 % of requests hit it |

## MTP knobs — reporting-only, and one of them is a trap

| knob | value | what it does | measured counterpart |
|---|---|---|---|
| `acc_len` | **1.0** | **Reporting only.** Verified first-hand at `agent_throughput.py:1002`: `generation_tps_mtp = completion_tokens * acc_len / (generation_time * mtp_overhead_factor)`. It scales the printed "(MTP compensated)" throughput and **nothing else** — it does not reach the server, the load, or any latency. | engine-measured acceptance **2.736** |
| `mtp_draft_tokens` | **4** | Divisor for the acceptance *rate* at `:1571` (`acc_len / mtp_draft_tokens`). Set to the server's actual `--speculative-num-draft-tokens 4`. | server config, matched |
| `mtp_overhead_factor` | 1.0 | Denominator in the same formula. | — |

**Why `acc_len: 1.0` and not the shipped 1.56.** This server already speculates,
so its measured `completion_tokens / generation_time` *already contains* the
speedup. Multiplying by 1.56 would report ~1.56× the tokens/s actually delivered
— a real measurement multiplied by a model of itself. 1.0 is the identity and
keeps `generation_tps` honest. The vultr sibling run shipped 1.56; its
`generation_tps` rescales by ÷1.56 for comparison. Acceptance is reported here
from the engine's own counters, never inferred from this knob.

## Load-shape knobs

| knob | value | measured | reading |
|---|---|---|---|
| `initial_sessions` | 32 | 32 started by t=33 s | as configured |
| `new_session_rate` | 0.10 /s | 422 rate-based births / 4,007 s = **0.105 /s** | as configured |
| `max_sessions` | 128 | peak live **54** | 2.4× headroom, never bound |
| `max_inflight` | 48 | peak **44** | **never pinned** — the load-bearing check |
| `ramp_duration` | 400 s | 229 requests excluded | warm-up exclusion, not a load ramp |
| `sustain_duration` | 3,600 s | 2,582 requests, 4,007 s total incl. drain | as configured |
| `gpus` | 8 | per-GPU reporting only | 7,498 input tok/s/GPU |
| `window` | 30 s | smoothing for the live display | no effect on persisted samples |
| `random_seed` | 1337 | — | fixes the sampled distribution; two runs at the same seed draw the same prompts |

**`new_session_rate` was left at the shipped 0.10.** The previous spur attempt
raised it to 0.115 on a linear-scaling assumption, predicted 26 in-flight,
measured 44–48, pinned the cap and had to be aborted. Vultr's Step-3 re-solve
landed 0.0941 — a 6 % move, inside noise. The correct action for both clusters
was to leave it alone.

## Knobs with no measured counterpart

| knob | value | blast radius if wrong |
|---|---|---|
| `sla.e2e_p50_ms` | 4,500 | **None — it is never consumed.** `args.sla_cfg` is parsed and dropped; the driver emits no E2E percentile. Back-solving gives ≈12.0 s (TTFT p50 + gen p50 × TPOT p50), a 2.7× miss, but no code path gates on it. |
| `sla.ttft_p90_ms` | 30,000 | **This one IS checked** — `runner.py:slo_satisfied()` compares `sustain_ttft_p90_ms`. Measured 18,877 → pass. Lowering it below ~19,000 would flip this run to a fail. |
| `system_prompt_len` | 2,000 | **Not sent in profile mode** — prompts are cut to the sampled `input_tokens` instead. Changing it does nothing here. |
| `max_prompt_tokens` | 260,000 | Unused for retirement in profile mode; declared to match `max_input_tokens`. A mismatch would let a session grow past the clamp and silently truncate mid-conversation. |
| `tokenizer` | `/shared_nfs/.../GLM-5.2-MXFP4` | **A placeholder path aborts the run at startup.** The shipped YAML carries `/path/to/GLM-5.2-MXFP4` and must be edited. Wrong-but-valid tokenizer → every length in this document is wrong by its vocab ratio, silently. |

## The one number outside the driver's control

**The 240 s client timeout is hardcoded** at `agent_throughput.py:929` as
`aiohttp.ClientTimeout(total=240)` and is not exposed as a knob. It is the direct
cause of both remaining imperfections in this run:

- the 39 errors (1.4 %), and
- the `output_tokens.p99` shortfall (9,688 vs 17,000).

Arithmetic: at TPOT p50 17.9 ms, a 17,000-token generation needs **304 s of
decode alone**, before TTFT. The observed max completed generation was 20,434
tokens = 366 s — which only completed because its TPOT ran below p50. A profile
that asks for a 17K-token p99 and a client that hangs up at 240 s are mutually
inconsistent; the workload cannot realize its own tail on any server fast enough
to matter. Raising the constant (not the server) is the fix.
