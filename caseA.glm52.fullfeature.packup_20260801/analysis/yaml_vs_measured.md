# Every YAML knob vs. what was measured

Derived after the run, from `results/caseA/summary.json`,
`results/caseA/metrics.jsonl.gz`, `logs/caseA_full.log`, and the driver source
(`Optimus-AgenticBench @ 1cf01cb`, branch `fix/realistic-profile-session-driver`,
`agent/agent_throughput.py`).

Config of record: `caseA.yaml`, a verbatim copy of the repo's
`agent/workloads/glm52_crxx_caseA.fix.yaml` with only the tokenizer path
substituted. Run: `caseA_full/2026-08-01-13-34-46`, ramp 400 s + sustain 3600 s,
2,988 requests sent.

> Image: `infera/engine-sglang:merged-e` **+ `GLM52_P1V3`**. Stock merged-e cannot
> complete this workload; see `../notes/notes.dsa.mtp.crash.md`.

Every knob in the YAML gets a row. Where a knob has no measured counterpart, the
row states its blast radius and the mechanism instead of leaving a blank.

---

## A. `profile` — request shape

| YAML knob | Nominal | Measured | Conversion rule / mechanism |
|---|---|---|---|
| `input_tokens` | p50 74,000<br>p90 155,000<br>p99 235,000 | **p50 74,559**<br>**p90 153,312**<br>**p99 233,656**<br>mean 86,254 | Off by +0.8 % / −1.1 % / −0.6 % — within 1.1 % at every percentile. `PercentileSampler` interpolates the triple, then `max_input_tokens` clamps. → sets prefill work and chunk count: `chunks = ceil(input / 8192)`, so ~9 chunks at p50 and ~29 at p99. |
| `output_tokens` | p50 320<br>p90 3,300<br>p99 17,000 | **p50 328**<br>**p90 2,810**<br>**p99 11,098**<br>mean 1,085 | p50 exact (+2.5 %); p90 −15 %; **p99 −35 %**. The 18 client-side 240 s timeouts preferentially kill long generations, so the right tail is censored — but far less than the spur run (p99 6,301, −63 %), because MTP halved TPOT so fewer long generations hit the deadline. Sent as `max_tokens` + `ignore_eos`. |
| `turns_per_session` | p50 3 / p90 20 / p99 103<br>(mean 9.50) | 422 births, 92 retirements,<br>mean session **732 s**,<br>max **3,978 s** | Sampled once at birth into `session_max_turns`; independent of server speed. **The p99 tail is still truncated** — a p99 session needs ~103 × 37 s ≈ 3,800 s, essentially the whole 4,006 s window, so only the very longest observed session (3,978 s) approaches it. → the multiplier in the demand rate `λ = rate × E[turns]`. |
| `inter_turn_delay_s` | p50 4 / p90 31 / p99 240<br>(mean 18.0) | inter-arrival<br>**p50 3.98 / p90 28.68 / p99 172.12**<br>mean 15.32 | Careful: the reported inter-arrival is **response + think**, not think alone. p50 and p90 land within 1 % and 7 % of nominal; p99 is 28 % low, the same window-censoring as `turns_per_session`. → the duty cycle `E2E / (E2E + delay)` converts live-session count into in-flight count: measured 0.592. |
| `cache_hit_rate` | 0.89 | **actual 0.8924**<br>ideal 0.8899<br>efficiency **1.0028**<br>eviction **0.0** | Actual slightly **exceeds** ideal: "ideal" counts only the modelled prefix, while the radix tree also retains request-unique suffix across turns of one session. Read from the server's `usage.prompt_tokens_details.cached_tokens`, not modelled. → **every request nests in the SAME prefix, so the radix tree holds one hot path** — which is why kvd saw +72,844 sets against only +452 gets. |
| `max_input_tokens` | 260,000 | **in effect** — p99.9 pinned at 260,013 | Clamps the sampled lognormal tail. Requires `--context-length 262144` on both legs; at ctx=131072 the p90 *and* p99 would both pin at 131,072 and the input distribution would not be the specified one. |
| `system_prompt_len` | 2,000 | **not in effect** | Not sent in profile mode — prompts are cut from the shared base. Only sizes each session's (unused) token slice. Blast radius: **none**, beyond being echoed into `metadata.json`. |

## B. Offered load — the load-bearing group

| YAML knob | Nominal | Measured | Conversion rule / mechanism |
|---|---|---|---|
| `initial_sessions` | 32 | 32 live at t=0 | Warm-up transient only. `ramp_duration=400 s` ≈ one session lifetime, sized so the synchronized t=0 cohort dies off and births take over. **No effect on steady state.** |
| `new_session_rate` | 0.10 | **422 births observed**<br>unthrottled expectation 401<br>(sd 19.0, **z = +1.13**) | One Bernoulli draw per ~1 s tick, **constant, no feedback**. → **demand rate `λ = rate × E[turns] = 0.10 × 9.50 = 0.95 req/s`, independent of server speed** — the driver prints exactly this ("offered rate is E2E-independent"). Measured qps 0.75 < 0.95; the gap is the 18 timeouts plus turns still unfinished at window close. **Left at the shipped value — see the re-solve below.** |
| `max_sessions` | 128 | **never throttled** — 0 ticks with live ≥ 128; max live 44 | `num_sessions_total` reaching 128 is the **cumulative list length**, not the live count; the birth gate tests `count_active_sessions()` and retired sessions free their slot. Births indistinguishable from unthrottled (z = +1.13). |
| `max_inflight` | 48 | **peak 30, never pinned**<br>p50 17 / p90 22 / p99 27<br>0 ticks at the cap | Spin-wait at the entry of `send_realistic_request`. → **it is simultaneously the only guard against session-layer divergence and a contaminant that invalidates the measurement the moment it binds.** It did not bind, so no hidden queueing time exists — which is why the two E2E back-solves agree here (21.6 s vs 22.2 s) and disagreed by 1.7× in the spur run. |

### Step 3 re-solve — and why the rate was deliberately NOT changed

The guide's procedure is to measure E2E, then re-solve
`rate = N_target / (E[turns] × (E2E + E[delay]))`.

    probe measured E2E = 17.8 s
    rate = 32 / (9.50 × (17.8 + 18.0)) = 0.0941 /s

Shipped value is 0.10, so the re-solve moves it by 6 % — inside noise. **The
shipped rate is confirmed by the guide's own formula.**

The probe nonetheless showed a steady population of only ~17 against a predicted
34. That gap was diagnosed as **tail censoring, not a rate error**: a 1,000 s
probe cannot realize an inter-turn delay whose p99 is 240 s nor a turn budget
whose p99 is 103 turns, so the realized per-turn cycle (13.7 s) came out 2.6×
short of the predicted 35.8 s, and session lifetime with it.

The prediction was that a 4,000 s window would let both realize closer to spec
and N would rise on its own at the unchanged rate. **It did:**

| | probe (1,000 s) | full (4,006 s) |
|---|---|---|
| live sessions p50 | 17 | **27** |
| live sessions by quarter | 19.5, 16.2, 14.4, 19.0 | **22.1, 22.3, 27.6, 36.0** |
| mean session lifetime | 318 s (est.) | **732 s** |
| max session lifetime | 1,005 s (window-bound) | **3,978 s** |
| mean inter-arrival | 13.7 s | **15.3 s** |

Final live population 36–44, straddling the nominal 32.

**Had the probe's N=17 been "corrected" by roughly doubling the rate**, the full
run would have landed near N≈70, in-flight would have pinned at 48, and
backpressure rather than the config would have set the load — the exact failure
the spur run hit and the guide warns about ("response is superlinear").

### The conversion chain, back-solved from measurement

    λ         = rate × E[turns] = 0.95 req/s        (constant, open-loop)
    N_session = measured mean 27.0
    N_inflight / N_session = E2E / (E2E + delay)
              → 16.0 / 27.0 = 0.592  →  E2E ≈ 22.2 s

Cross-checks against the composition route (`TTFT + (gen−1)×TPOT`, mean 21.6 s)
within 3 %.

**Two layers, two behaviours.** Within a session the loop is closed — it awaits
each response before the think time. Across sessions it is open: births are a
fixed Bernoulli draw reading no server state. So a slower server changes the
standing population, not the demand.

## C. Speculative decoding — MTP is ON here

| YAML knob | Shipped | This run | Mechanism |
|---|---|---|---|
| `acc_len` | 1.56 | **1.56 as shipped**; measured **2.04** | MTP is on, so the shipped value is left alone and becomes a directly checkable claim. Per-request acceptance p50 2.00 / mean **2.04** (n=2,702); server-side decode-batch mean **2.80** over 19,225 lines. **The deployment beats the config's assumption by 31 %.** Note the driver uses `acc_len` only for its MTP-compensated TPS accounting, so that figure is now conservative. |
| `mtp_draft_tokens` | 5 | **5 as shipped** | Draft tokens per step. With acceptance 2.04 of 5 drafted, the realized acceptance *rate* is ~41 %, against the config's implied 56 % (1.56/5 ≈ 31 %… the YAML's own arithmetic is inconsistent with its comment; the measurement stands regardless). |
| `mtp_overhead_factor` | 1.0 | 1.0 | A multiplier on the TPS accounting; 1.0 is the identity. Blast radius: **none**. |

> Contrast the spur packup, which overrode these to 1.0 / 1 **because MTP was off
> there**. Leaving them at the shipped values is only correct when MTP is on.

**The measured payoff:** TPOT p50 **14.8 ms** vs the spur run's **31.3 ms**, a
**2.11×** improvement — closely tracking the 2.04 mean acceptance, which is the
mechanism. (Not a single-variable ablation: different cluster fabric and kvd
decode wiring. But the magnitude matches the mechanism.)

## D. Measurement window

| YAML knob | Nominal | Measured | Mechanism |
|---|---|---|---|
| `ramp_duration` | 400 | ramp phase 400 s / 246 req / qps 0.61 | **Not a load ramp** — nothing ramps in closed-loop mode. It is a warm-up exclusion window. Ramp TTFT p50 is 3,060 ms against sustain's 4,531 ms, so excluding it is load-bearing, not cosmetic. Ramp cache% 91.5 vs sustain 89.0 — the initial cohort shares one prefix more tightly than the steady-state mix. |
| `sustain_duration` | 3600 | 3600 s / 2,702 req / qps 0.75 | Total 4,006 s including a 4.5 s drain. → **still the binding constraint on the turn tail**: a p99 session needs ~3,800 s, and the longest observed was 3,978 s. The guide calls 3,600 s "the honest number"; it is honest but not generous. |

## E. Request framing / metadata

| YAML knob | Nominal | Measured | Mechanism |
|---|---|---|---|
| `max_prompt_tokens` | 260,000 | echoed in metadata | **Not used for retirement** in profile mode. Declared only to match `max_input_tokens`. Blast radius: **none**. |
| `tokenizer` | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | in effect | The shipped placeholder `/path/to/GLM-5.2-MXFP4` aborts the run; substituting the real path is the one edit made to the `.fix.yaml`. Used for exact token counts. |
| `random_seed` | 1337 | in effect | Fixes the profile sampling sequence. ⚠️ Once `max_inflight` binds the driver warns `traffic timing may diverge from seed`. It did not bind here, so this run is reproducible. |

## F. `sla` block — parsed, never consumed

`args.sla_cfg` is read and never used; `runner.py::slo_satisfied()` checks its own
CLI flags (`--slo-ttft-p90-ms` / `--slo-success-rate`). These rows document
intent, not gating.

| YAML knob | Target | Measured | Verdict |
|---|---|---|---|
| `sla.e2e_p50_ms` | 4,500 | **driver emits no E2E**; derived p50 **11,100 ms** | **Missed, 2.5×** — but derived, and the target is not gated. Two independent back-solve routes agree within 3 % (21.6 s vs 22.2 s mean), unlike the spur run's 1.7× gap. The target appears written against a much shorter generation than the p50 of 330 tokens this profile actually produces. |
| `sla.ttft_p90_ms` | 30,000 | **9,170 ms** | **Met, 3.3× margin.** |
| `sla.success_rate` | 0.97 | **0.988** | **Met.** All 18 errors are the client's hardcoded `aiohttp.ClientTimeout(total=240)`: 18 lines `timed out`, **0** `failed: HTTP`, 0 other exceptions. Both engine legs logged 0 GPU faults, 0 scheduler exceptions, 0 retractions for the entire window. (Spur: 0.953, missed.) |

## G. Misc

| YAML knob | Nominal | Measured | Mechanism |
|---|---|---|---|
| `gpus` | 8 | 8 | Per-GPU reporting only: uncached TPM 425,522 → **53,190 /GPU**, against the guide's ~69K reference at N=32 (this run averaged N=27). |
| `window` | 30 | 30 | Metrics smoothing window (s). Affects only the `*_window` display fields, no summary statistic. |

## H. Knobs the `.fix.yaml` deleted from the original spec

| YAML knob | Original value | This run | Why removed |
|---|---|---|---|
| `initial_qps` / `max_qps` | 0.05 → 0.40 | **deleted** | The open-loop QPS ramp exists only on the open-loop path; the realistic closed-loop path never reads it. The driver prints `initial_qps/max_qps have no effect here` at startup. **This also means `runner.py --auto-search` and its QPS sweep are inert on this workload** — they bisect on a variable with no effect. |
| `session_lifetime_mean/median` | 600 / 400 | deleted | `turns_per_session` owns session length here. |
| `think_time_mean/shape` | 10.0 / 1.0 | deleted | `inter_turn_delay_s` replaces the gamma think time. |

> ⚠️ **`metadata.json` still echoes unused defaults** (`think_time_mean`,
> `session_lifetime_mean`, `generation_length_mean: 1`, `new_tokens_mean`). These
> are argparse defaults, not the effective configuration for this run, and are
> easy to misread as settings. Note `generation_length.mean` in the run summary
> prints "(target: 1)" for the same reason — the *target* is a stale default; the
> measured 1,085 is real.

---

## What this run establishes that the prior kits could not

| claim | evidence |
|---|---|
| the merged stack runs Case A **with all five features on at once** | 4,006 s, 2,952 completed, 0.988 success, kvaware+kvd+MTP+PD+DPA all verified live |
| MTP is worth ~2× on TPOT at this workload | 14.8 ms vs spur's MTP-OFF 31.3 ms, mechanism confirmed by acceptance 2.04 |
| the shipped `acc_len: 1.56` is conservative | measured 2.04 per-request / 2.80 server-side |
| kv-aware + kvd serve correctly under a shared-prefix agentic load | cache efficiency 100.3 %, kvd 452 gets / 452 hits / **0 misses** |
| **stock merged-e has a decode-leg crash under MTP + DPA** | reproduced twice; root-caused to the `GLM52_P1V2` trim's one-sided guard; fixed by `GLM52_P1V3` |

The last row is the most consequential finding of the whole task: the "latest full
feature support" image cannot run this workload unpatched.
