# Notes — why the design is what it is, gotchas, and open questions

Written as **what / why / how / context**. The point is that a reader learns why
a step matters, not just that it exists.

---

## 1. The percentile triples are degenerate ON PURPOSE

**What.** Each workload YAML pins its shape with an *equal* triple:

```yaml
input_tokens:  { p50: 74000, p90: 74000, p99: 74000 }
output_tokens: { p50: 320,   p90: 320,   p99: 320   }
```

This looks like a mistake — three percentiles all set to the same number — and it
is not.

**Why it works.** `PercentileSampler` (`agent/sampling.py:80`) interpolates
**ln(value)** piecewise-linearly against the standard-normal quantiles
z = (0, 1.2816, 2.3263). With all three control points equal, every segment has
**zero slope**, so `_interp_lnv(z)` returns `ln(X)` for every z. The clamps agree:

```python
vmin = p50 * p50 / p90   # = X when all equal
vmax = p99 * (p99 / p90) ** 2   # = X when all equal
```

so `vmin == vmax == X` and `sample()` cannot return anything else.

**How it was confirmed — first-hand, not by reading the arithmetic.**
`PercentileSampler(74000, 74000, 74000).sample_int()` returns `74000` on every
draw. The measured data agrees: `prompt_length` in every `summary.json` has
`mean == p50 == p90 == p99` (74012.99 / 155012.98 / 235013.00 — the ~13-token
offset is the chat template, not sampler spread).

**Context.** This is what turns "measure the p50 request shape" into **an exact,
repeatable request** rather than a distribution that merely has that median. At
n = 28–106 repeats, a real distribution would have swamped the latency signal
with request-size variance. Note the *output* side still varies slightly
(gen p50 317 vs the requested 320) — the model stops when it stops; the sampler
only fixes the `max_tokens` request.

---

## 2. `max_sessions: 1` is the real concurrency gate — `max_inflight: 1` is not

**What.** Both are set to 1. Only one of them is doing the work.

**Why.** The control loop (`agent_throughput.py:2509`) spawns a new session only
while

```python
if new_session_rate > 0 and active_sessions < max_sessions:
```

and `count_active_sessions()` (`:2417`) is

```python
return sum(1 for s in sessions if s.is_available() or s.in_flight)
```

— an **in-flight session counts as active**. So while a request is on the wire,
`active_sessions == 1 == max_sessions` and no second session can be born. With
`turns_per_session = 1` the session then retires and the next tick spawns a
fresh one. Exactly one request is ever outstanding.

`max_inflight` is a *backpressure* valve on a different code path
(`:1098–1103`): it busy-waits when in-flight reaches the cap, and prints
`WARNING: Hit max_inflight (…) - traffic timing may diverge from seed` the first
time it fires.

**How we know it never fired.** `grep -c "Hit max_inflight"` over all three
driver logs returns **0, 0, 0**. And the per-tick record confirms the property
directly rather than by absence:

| arm | `in_flight` over all ticks | `num_sessions_active` |
|---|---|---|
| p50 | `{0: 61, 1: 722}` | `{0: 60, 1: 723}` |
| p90 | `{0: 43, 1: 1638}` | `{0: 42, 1: 1639}` |
| p99 | `{0: 29, 1: 4346}` | `{0: 28, 1: 4347}` |

Never 2. The zeros are the gaps between a session retiring and the next spawning.

**Context / trap.** If you reuse these YAMLs at higher concurrency, **raise
`max_sessions`** — raising only `max_inflight` changes nothing, because the
session gate binds first. That is the opposite of the intuition the two names
suggest.

---

## 3. `ramp_duration` is a warm-up EXCLUSION window, not a load ramp

**What.** `ramp_duration: 180` on all three arms. It does **not** ramp the
offered load — at `max_sessions: 1` there is nothing to ramp. It is a period
whose requests are *tagged and excluded* from the reported statistics.

**Why it is there.** The mission requires the measurement be taken *在刷入 cache
保证 cache hit rate 的情况下* — with the shared prefix already resident. Each
request nests inside the same ~0.89 prefix; the first few requests build that
prefix in the radix tree and pay for it. Reporting them would measure cache
construction, not steady-state serving.

**How the evidence shows it worked.** The ramp/sustain contrast in each
`summary.json` is unambiguous:

| arm | ramp completed | ramp cache hit | ramp TTFT p99 | sustain cache hit | sustain TTFT p99 |
|---|---|---|---|---|---|
| p50 | 26 | 0.8556 | **20875.8 ms** | 0.8898 | 3413.3 ms |
| p90 | 5 | 0.7968 | **16015.1 ms** | 0.8897 | 5349.1 ms |
| p99 | 1 | 0.5869 | **18462.2 ms** | 0.8900 | 7486.4 ms |

The first request of each arm pays a 16–21 s cold TTFT. None of that reaches the
reported table. The p99 arm is the clearest case: its single ramp request ran at
a 0.587 hit rate, and sustain then held 0.890.

**Context.** `analyze_solo.py` enforces this — it skips every row where
`phase != "sustain"`. If you re-analyse with your own script and forget that
filter, your TTFT p99 will be several-fold too high and you will be measuring the
wrong thing.

---

## 4. There is no think time, and `inter_turn_delay_s` is dead weight kept for provenance

**What.** The YAMLs still carry `inter_turn_delay_s: { p50: 4, p90: 31, p99: 240 }`,
inherited verbatim from `glm52_crxx_caseA.fix.yaml`. It has **no effect**.

**Why it cannot fire.** In `session_loop` the turn-budget check comes **before**
the sleep (`agent_throughput.py:2386–2401`):

```python
if session_max_turns is not None and session.request_count >= session_max_turns:
    session.retired = True
    break                        # <-- exits here at turns=1
...
think_time = min(MAX_THINK_TIME, profile.inter_turn_delay.sample())
await asyncio.sleep(think_time)  # <-- never reached
```

With `turns_per_session = 1`, `request_count` hits the budget on the first
request and the loop breaks before the sleep. This satisfies the mission's
"conc=1 不需要 think time delay" structurally rather than by setting a value to 0.

**Why it was kept anyway.** Deleting it would make the file look like a
*different* workload from the Case-A original it was derived from. The three
solo YAMLs differ from each other in **only** the `input_tokens` /
`output_tokens` triples and `sustain_duration` (plus comments) — verified by
`diff`. Keeping unreachable fields verbatim is what makes that diff meaningful.

**Context / trap.** `retired = True` matters as much as `break`. The comment in
the source spells it out: a session that only breaks still reads as live and
holds a slot against `max_sessions` forever, starving new spawns. At
`max_sessions: 1` that would deadlock the whole run after one request.

---

## 5. OPEN QUESTION: TPOT p50 is *lower* on the longer arms

**What was measured.** Sustain-phase TPOT p50 across the three arms:

| arm | prompt | gen tokens | **TPOT p50** | TPOT p90 | TPOT p99 |
|---|---|---|---|---|---|
| p50 | 74,013 | 317 | **10.0 ms** | 10.7 | 11.5 |
| p90 | 155,013 | 3,297 | **8.1 ms** | 9.1 | 10.8 |
| p99 | 235,013 | 16,993 | **8.3 ms** | 8.8 | 9.4 |

Decode gets *faster per token* as the prompt more than triples and the generation
grows 50×. The whole distribution shifts, not just the median — TPOT p99 falls
monotonically 11.5 → 10.8 → 9.4 ms.

**What is NOT claimed.** No mechanism. Nothing was measured that would identify
one. A fluent-sounding story here would be worse than silence: it would
discourage running the experiment that actually settles it.

**What would settle it.** Three measurements, in increasing cost:

1. **Per-request TPOT against that request's generation length**, within a single
   arm. Both arrays are already in the shipped `metrics.jsonl`
   (`new_tpots` and `new_generation_lengths`, index-aligned by the SOLO_M1
   patch). If TPOT falls with generation length *inside one arm* — where prompt
   length is pinned — that separates a per-request fixed-cost amortisation effect
   from anything to do with prompt length. **This needs no cluster access.**
2. **MTP acceptance per arm, correlated with TPOT.** The engine log already shows
   acceptance rising across the arms (median 2.73 / 3.55 / 3.70 — §"Feature
   evidence" in `README.md`), and TPOT is per *accepted* output token. Whether
   that accounts for the shift is arithmetic that has **not been done here**; it
   requires pairing per-request acceptance with per-request TPOT, and the
   engine-log acceptance figures are per-batch, not per-request. The driver does
   record `new_acceptance_lengths` per request — start there.
3. **An A/B with MTP disabled**, same three shapes. Expensive (three more runs)
   but it is the only one that isolates speculation from everything else.

Measurement 1 is cheap and offline; do it first.

---

## 6. OPEN QUESTION: the p50 arm misses the stated 4500 ms E2E bar

**What was measured.** The p50 workload carries `sla.e2e_p50_ms: 4500`. The
measured sustain-phase E2E p50 is **5111.1 ms** — over by **611.1 ms**.

**First, what the bar is and is not.** It is **documentation only**. The driver
assigns `args.sla_cfg = workload.get("sla")` at `agent_throughput.py:3351` and
**never reads it again** (verified by grep over the whole source). Nothing in the
run enforces, checks, or reports against it. So this is a comparison *we* are
making against a number *we* wrote into the file, not a driver verdict.

**What is NOT claimed.** No explanation for the 611 ms. It is not attributed to
prefill, to decode, to the router, to cache misses, or to anything else — none of
those was measured against this bar.

**What would settle it.** The E2E decomposes as `TTFT + (gen_len - 1) × TPOT`.
For this arm: 1811.7 + 316 × 10.0 ≈ 4972 ms, close to the measured 5111 ms p50
(the residual is because a p50 of a sum is not the sum of p50s). So the question
reduces to which of the two terms is above expectation:

1. **Compare TTFT p50 against the fixlen p50 arm at conc 1.** Phase 1 measured
   TTFT p50 **1046.7 ms** for ISL 7,400 / OSL 320 at concurrency 1
   (`fixlen.glm52.mix.packup_20260806/README.md`) — the fresh-remainder
   equivalent of this arm's 74K-at-89 %-hit prompt. Here it is 1811.7 ms. The two
   are not the same request (one has a 66K-token cached prefix to look up, the
   other has no prefix at all), and **whether that difference is the whole story
   has not been measured.** Pairing per-request TTFT with `new_cache_hit_rates`
   from the shipped `metrics.jsonl` is the cheap, offline first step.
2. **Whether 4500 ms was ever the right bar for this shape.** The value was
   carried in from the Case-A profile. What it was derived from, and on what
   hardware, is not recorded in anything in this packup. Establishing its
   provenance may dissolve the question entirely.

Both are open. Neither is answered here.

---

## 7. The p99 arm's `success_rate: 0.9667` is clock truncation, not a failure

**What.** `summary.json` for the p99 arm reads `requests_sent: 30`,
`requests_completed: 29`, `success_rate: 0.9667` — while `errors: 0`.

**Why.** Tick-by-tick from `metrics.jsonl`: request 30 was sent at elapsed
**4302.7 s**, and the run's budget (180 ramp + 4200 sustain) expires at 4380 s.
A request of this shape takes ~147 s end-to-end. It was still in flight
(`in_flight: 1`) on the final tick. `success_rate` is `completed / sent`, so an
outstanding request reads as a shortfall.

**How to confirm.** `REPRODUCE.md` §7 has the exact snippet; it prints

```
sent ->30 at elapsed=  4302.7 (sustain)
final: sent=30 done=29 errors=0 inflight=1
```

**Context — why it does not touch the results.** The truncated request never
produced a TTFT/E2E/TPOT sample, so it never entered the sustain arrays. The
reported n = 28 are all completed requests. The `errors` counter, which is what
would move on a real failure, is 0 on every tick of every arm.

**Trap for the next run.** If you shorten `sustain_duration` on a long-generation
arm, this gets worse — the tail request is always at risk. Size `sustain_duration`
as a whole multiple of the expected E2E, or just accept the reported success rate
will be `(n-1)/n` and check `errors` instead.

---

## 8. Nested-ssh quoting silently produced a division-by-zero

**What.** Reading the MTP acceptance distribution per arm needs an `awk` script
run inside the container, two ssh hops away. Written as a one-liner:

```bash
ssh jump "ssh node 'docker exec ctr bash -c \"strings …\" | awk \"…\"'"
```

it returned, three times:

```
awk: cmd. line:1: fatal: division by zero attempted
```

**Why.** The quoting is mangled across the three shell levels, so the awk pattern
never matched anything, `NR` stayed 0, and the `END` block divided by it. The
failure mode is deceptive: awk *ran*, so it looks like a data problem (an empty
time window) rather than a quoting problem.

**How it was fixed.** Stage the script as a **file** and run the file:

```bash
cat > /tmp/acc_win.sh <<'EOF'   # quoted heredoc — no interpolation anywhere
...
EOF
scp /tmp/acc_win.sh jump:/tmp/ && ssh jump "scp /tmp/acc_win.sh node:/tmp/; ssh node 'bash /tmp/acc_win.sh 09:53:41 10:07:30 p50'"
```

That produced the readings in `README.md` immediately.

**Context.** This is a known recurring trap, not a one-off. Any time a command
crosses more than one shell boundary and contains quotes, stage a script file.
The same class of failure silently no-ops rather than erroring in most other
shapes — here it happened to hit a divide-by-zero and became visible, which was
lucky.

---

## 9. Scope EVERY engine-log grep by time window

**What.** The engine, router and kvd logs are appended across the **entire
session**. At the time this packup was assembled that file contained Phase 1's
fixlen sweep (07:13–09:45), these three solo arms (09:53–11:52), **and** Phase 3's
loaded run, which started afterwards and was still running.

**Why it matters.** An unscoped `grep` mixes all three phases into one
distribution. The whole-solo-window acceptance read is
`n=5134 median=3.60 at-4.00=18.2 %`; the per-arm reads are
`2.73 / 3.55 / 3.70` with at-4.00 of `0.0 / 11.6 / 22.6 %`. Those are materially
different conclusions about the same deployment.

**How.** The lines carry timestamps; filter on them. `REPRODUCE.md` §8 has the
script. The windows for this phase:

| arm | window (UTC) |
|---|---|
| p50 | 09:53:41 → ~10:07 |
| p90 | 10:07:44 → ~10:38 |
| p99 | 10:38:57 → ~11:52 |

**Also.** Use `strings` on these logs — they carry binary bytes and a bare `grep`
then reports only "binary file matches". And never poll for readiness by grepping
a log for a ready line: it matches the *previous* run's line and returns early.

---

## 10. The driver's own percentiles differ from ours — index convention, established by test

**What.** Each `summary.json` carries a sustain-phase TTFT/TPOT percentile block,
and it disagrees with `analyze_solo.py` by up to 2.5 % on the same array:

| arm | metric | driver | `analyze_solo.py` |
|---|---|---|---|
| p50 | TTFT p90 | 2857.6 | 2815.9 |
| p90 | TTFT p90 | 4850.0 | 4730.9 |
| p99 | TTFT p90 | 6039.4 | 5854.7 |

**Why.** Two different percentile index rules over the same data:

- driver: `sorted[int(q * n)]` — the ceiling / "higher" convention
- `analyze_solo.py`: `sorted[round(q * (n - 1))]` — nearest-rank on `n-1`

**How it was established — by test, not by reading one implementation.** Every
standard convention was evaluated against the driver's reported values on all
three arms: `floor q*n`, `ceil q*n - 1`, and numpy's `linear`, `higher`,
`lower`, `nearest`, `midpoint`. The driver matched `numpy method='higher'`
exactly on every point tested and matched `linear` on none. At n = 28–106 the
two conventions can land on adjacent samples, and one sample is the whole gap.
Where the indices coincide (all the p99 columns, TTFT p99 everywhere) the two
agree to the decimal — which is itself the confirmation that the underlying data
is identical.

**Context.** Neither convention is more correct. The reported table uses
`analyze_solo.py` for all of TTFT / E2E / TPOT because the driver's block has **no
E2E column at all** — SOLO_M1 adds the raw array, not a percentile summary. Mixing
the two sources would silently compare percentiles taken by different rules. If
you quote `summary.json` numbers, quote them for all three metrics, or state the
convention.

---

## 11. Smaller things worth knowing

- **The kvd counters are cumulative across all phases** and cannot be attributed
  to this one. Read live while assembling: 71,283 entries / 84.9 GB host /
  68.7 GB L3 / gets 69,019 / sets 391,322 / hits 66,975 / misses 2,044 /
  evictions 247,577. Phase 1 recorded gets 64,741 / hits 62,697 at its end, so
  the solo arms added roughly 4,278 gets and 4,278 hits — but that subtraction
  spans the phase boundary loosely and is offered as an order of magnitude, not
  a measurement.
- **Never greedy-decode this model.** `--temperature 1.0 --top-p 0.95` is the
  checkpoint's own `generation_config`. At temperature 0 it repeats, MTP pins
  acceptance at 4.00, and the run reads like KV corruption. Results are therefore
  **not bit-reproducible** — compare distributions.
- **`--tokenizer` on the CLI overrides the YAML's `tokenizer:` field.** The
  driver announces this (`Skipped (CLI override): 1 parameters`). Harmless here
  (both point at the same path) but a silent divergence waiting to happen.
- **`--dashboard-mode` is mandatory** in `run_agentic.sh`. Without it nothing
  structured is persisted and the run is unrecoverable once the terminal scrolls.
- **No load knobs on the CLI, by design.** `run_agentic.sh` passes only the YAML.
  Passing `--initial-sessions` would silently shadow the file and make the run
  unreproducible from it.
- **The driver prints `PyTorch was not found` at startup.** Expected — it only
  uses `tokenizers`. Not a problem.
- **`system_prompt_len: 2000` produces a 5,603-token system prompt.** The driver
  reports this on startup. The requested value is a target for a synthetic
  generator, not an exact length.
- **The 235K p99 prompt sits inside `context_len=262144` with ~27K to spare.** A
  larger Case-A shape would need `--context-length` raised.
- **Reasoning tokens are billed against the same budget as content**
  (`--reasoning-parser glm45`). The p99 arm generated 38,609 reasoning tokens out
  of 492,453 total.
