# Every YAML knob vs. what was measured

Derived after the run, from `results/summary.json`, `results/metrics.jsonl.gz`,
`results/caseA_full_report.txt`, and the driver source
(`Optimus-AgenticBench @ 1cf01cbf`, `agent/agent_throughput.py`).

Config of record: `workloads/caseA_full.yaml` (derived from
`spec/glm52_crxx_caseA.fix.yaml`). Run: `caseA_full/2026-07-31-17-10-30`,
ramp 400 s + sustain 3600 s, 2919 requests sent.

Every knob in the YAML gets a row. Where a knob has no measured counterpart, the
row states its blast radius and the mechanism instead of leaving a blank.

---

## A. `profile` — request shape

| YAML knob | Nominal | Measured | Conversion rule / mechanism |
|---|---|---|---|
| `input_tokens` | p50 74,000<br>p90 155,000<br>p99 235,000 | **p50 73,862**<br>**p90 151,526**<br>**p99 226,854**<br>mean 85,172 | Off by 0.2 % / 2.2 % / 3.5 %. `PercentileSampler` interpolates the triple, then `max_input_tokens` clamps. → sets prefill work and chunk count: `chunks = ceil(input / 8192)`, so ~9 chunks at p50 and ~28 at p99. |
| `output_tokens` | p50 320<br>p90 3,300<br>p99 17,000 | **p50 286**<br>**p90 2,283**<br>**p99 6,301**<br>mean 802 | p50 close; **p90/p99 well below nominal**. The 96 client-side 240 s timeouts preferentially kill long generations, so the right tail is censored. Sent as `max_tokens` + `ignore_eos`. → drives E2E: `E2E ≈ TTFT + (gen_len − 1) × TPOT`. |
| `turns_per_session` | p50 3 / p90 20 / p99 103<br>(mean 9.5) | **p50 3 / p90 18 / p99 46**<br>mean 7.20 / max 69 | p50 exact, p90 close, **p99 truncated** by the 3600 s window (a p99 session's modelled lifetime is ~3557 s). Sampled once at birth into `session_max_turns`; independent of server speed. → the multiplier in the demand rate `λ = rate × E[turns]`. |
| `inter_turn_delay_s` | p50 4 / p90 31 / p99 240<br>(mean 18.4) | inter-arrival<br>**p50 4.0 / p90 27.4 / p99 168.6**<br>mean 14.8 | Careful: the reported inter-arrival is **response + think**, not think alone. Capped by `MAX_THINK_TIME`. → the duty cycle `E2E / (E2E + delay)` is what converts live-session count into in-flight count. |
| `cache_hit_rate` | 0.89 | **actual 0.8900**<br>ideal 0.8899<br>efficiency 1.0002<br>eviction 0.0 | Four-decimal match, zero eviction. Prompt = shared prefix of `round(0.89 × input)` tokens + request-unique fresh tokens. Read from the server's `usage.prompt_tokens_details.cached_tokens`, not modelled. → **every request nests in the SAME prefix, so the radix tree holds one hot path and there is no L3 pressure** — this is why kvd saw +131,162 sets but only +18 gets. |
| `max_input_tokens` | 260,000 | in effect, clamps ~1.4 % | Fitting a lognormal to the spec triple puts 16.1 % above a 131,072 clamp vs 1.4 % above 262144. At ctx=131072 the probe showed p90 **and** p99 both pinned at exactly 131,072. → forced `--context-length 262144`. |
| `system_prompt_len` | 2,000 | **not in effect** | Not sent in profile mode — prompts are cut from the shared base. Only sizes each session's (unused) token slice. Blast radius: **none**, beyond being echoed into `metadata.json`. |

## B. Offered load — the load-bearing group

| YAML knob | Nominal | Measured | Conversion rule / mechanism |
|---|---|---|---|
| `initial_sessions` | 32 | 32 live at t=0 | Warm-up transient only. `ramp_duration=400 s` ≈ one session lifetime, sized so the synchronized t=0 cohort dies off and births take over. **No effect on steady state.** |
| `new_session_rate` | 0.110<br>(spec original 0.10) | **458 births observed**<br>unthrottled expectation 435<br>z = 1.17 | One Bernoulli draw per ~1 s tick, **constant, no feedback** (`agent_throughput.py:2510`). → **demand rate `λ = rate × E[turns] = 0.110 × 9.5 = 1.045 req/s`, independent of server speed** — the driver prints exactly this at `:3935` ("offered rate is E2E-independent"). Measured qps 0.719 < 1.045; the gap is the 96 timeouts plus turns still unfinished at window close. |
| `max_sessions` | 128 | **never throttled** | Gate is `if new_session_rate > 0 and active_sessions < max_sessions`, where `active_sessions = count_active_sessions()` (`:2417`) counts not-retired-and-not-abandoned, and retired sessions free their slot. The `num_sessions_total` in `metrics.jsonl` is the **cumulative list length** — it reached 128 at t=968 s, which is the list filling up, not the gate binding. Observed births are statistically indistinguishable from unthrottled (z = 1.17). → **⚠️ this refutes the limitation recorded in `RESULTS.md` / `notes.md` ("max_sessions=128 was reached at t≈1089 s, so new-session creation stopped"). That claim read the list length as the live count and is wrong. Not yet corrected in those files.** |
| `max_inflight` | 48 | **peak 46, never pinned**<br>p50 28 / p90 36 / p99 42 | Spin-wait at the entry of `send_realistic_request` (`:2164-2170`). → **it is simultaneously the only guard against session-layer open-loop divergence and a contaminant that invalidates the measurement the moment it binds.** Contrast the first attempt at rate 0.145: pinned at 48 (25 of the last 120 ticks at the cap, mean 44.2) with sessions climbing 40 → 57; aborted. |

### The conversion chain for this group, back-solved from measurement

    λ         = rate × E[turns] = 1.045 req/s        (constant, open-loop)
    N_session = measured p50 38
    N_inflight / N_session = E2E / (E2E + delay)
              → 28 / 38 = 0.74  →  E2E ≈ 51 s

E2E ≈ 51 s is far above the spec's assumed 15 s. That is why the "48 vs target 32"
headroom got eaten: at the spec's 15 s the duty cycle is only 15/33.4 ≈ 45 % (32
sessions → ~14 in-flight, 3.4× headroom), but the measured duty cycle is 74 %.

**Two layers, two behaviours.** Within a session the loop is closed — it awaits each
response before the think time. Across sessions it is open: births are a fixed Bernoulli
draw reading no server state, and each session's total request count is sampled at birth.
So a slower server changes the standing population, not the demand. With both caps
removed and `λ > μ`, N diverges monotonically.

## C. Speculative decoding

| YAML knob | Spec original | This run | Mechanism |
|---|---|---|---|
| `acc_len` | 1.56 | **1.0** | MTP is off. Left at 1.56 the driver reports a fabricated MTP-adjusted TPS. **No measured counterpart** — `acceptance_length` degenerates to ~1 without speculative decoding and is deliberately not quoted. |
| `mtp_draft_tokens` | 5 | **1** | As above. |
| `mtp_overhead_factor` | 1.0 | 1.0 | Unchanged. A multiplier on the TPS accounting; 1.0 is the identity. Blast radius: **none**. |

## D. Measurement window

| YAML knob | Nominal | Measured | Mechanism |
|---|---|---|---|
| `ramp_duration` | 400 | ramp phase 400 s / 190 req / qps 0.475 | **Not a load ramp** — nothing ramps in closed-loop mode. It is a warm-up exclusion window. Ramp-phase TTFT p50 is 2,920 ms against sustain's 4,543 ms, so excluding it is load-bearing, not cosmetic. |
| `sustain_duration` | 3600 | 3600 s / 2588 req / qps 0.719 | Total 4007 s including a 2.2 s drain. → **directly causes the turns p99 truncation**: a p99 session's modelled lifetime (~3557 s) is the window itself. Longest observed session: 3219 s. |

## E. Request framing / metadata

| YAML knob | Nominal | Measured | Mechanism |
|---|---|---|---|
| `max_prompt_tokens` | 260,000 | echoed in metadata | **Not used for retirement** in profile mode. Declared only to match `max_input_tokens` — the original inherited a 200000 default that silently contradicted the 260000 clamp. Blast radius: **none**. |
| `tokenizer` | real model path | in effect | A placeholder path aborts the run. Used for exact token counts. |
| `random_seed` | 1337 | in effect | Fixes the profile sampling sequence. ⚠️ Once `max_inflight` binds the driver warns `traffic timing may diverge from seed (non-deterministic)`. It did not bind here, so this run is reproducible. |

## F. `sla` block — parsed, never consumed

`args.sla_cfg` is read and never used; `runner.py::slo_satisfied()` checks its own CLI
flags (`--slo-ttft-p90-ms` / `--slo-success-rate`). These rows document intent, not gating.

| YAML knob | Target | Measured | Mechanism |
|---|---|---|---|
| `sla.e2e_p50_ms` | 4,500 | **driver emits no E2E** | No SLI exists. Back-solvable only (see `sli_percentiles.md`), and the two back-solve routes disagree by 1.7× — the gap is `max_inflight` spin-wait, which is not counted in TTFT. Left open. |
| `sla.ttft_p90_ms` | 30,000 | **8,795 ms** | 3.4× margin. The only metric `runner.py` actually compares — and it reads it from CLI, not from here. |
| `sla.success_rate` | 0.97 | **0.953** | Missed. All 96 errors are the client's hardcoded `aiohttp.ClientTimeout(total=240)` (`agent_throughput.py:928`): 96 lines `timed out`, **0** `failed: HTTP`, 0 other exceptions. Both engine legs logged 0 GPU faults and 0 scheduler exceptions for the entire window. |

## G. Misc

| YAML knob | Nominal | Measured | Mechanism |
|---|---|---|---|
| `gpus` | 8 | 8 | Per-GPU reporting only: uncached TPM 407,291 → **50,911 /GPU**. |
| `window` | 30 | 30 | Metrics smoothing window (s). Affects only the `*_window` display fields, no summary statistic. |

## H. Knobs the `.fix.yaml` deleted from the original spec

| YAML knob | Original value | This run | Why removed |
|---|---|---|---|
| `initial_qps` / `max_qps` | 0.05 → 0.40 | **deleted** | The open-loop QPS ramp exists only on the open-loop path (`agent_throughput.py:1085`, `:3505`); the realistic closed-loop path never reads it. The driver prints `initial_qps/max_qps have no effect here` at startup. |
| `session_lifetime_mean/median` | 600 / 400 | deleted | `turns_per_session` owns session length here. |
| `think_time_mean/shape` | 10.0 / 1.0 | deleted | `inter_turn_delay_s` replaces the gamma think time. |

> ⚠️ **`metadata.json` still echoes `think_time_mean: 10.0`, `session_lifetime_mean: 600.0`,
> `generation_length_mean: 1`, `new_tokens_mean: 8000`.** These are unused defaults, not
> the effective configuration for this run. They are easy to misread as settings.

---

## One correction this analysis forces

`RESULTS.md` and `notes.md` both state that `max_sessions=128` was reached at t≈1089 s
and that new-session creation stopped for the remainder. **That is wrong.** It read
`num_sessions_total` (cumulative list length, which hit 128 at t=968 s) as the live-session
count, but the birth gate tests `count_active_sessions()`, and retired sessions free their
slot. 458 births against an unthrottled expectation of 435 (binomial sd 19.7, z = 1.17)
shows the gate never bound. Neither file has been corrected yet.
