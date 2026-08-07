# Notes — the saturation reading, the open questions, and the gotchas

Written as **what / why / how / context**. The point is that a reader learns why
a step matters, not just that it exists.

Three sections (§3, §4, §5) are **deliberately unresolved**. Each states a
measurement and stops there. That is the intended deliverable, not a gap: a
fluent-sounding mechanism in any of those places would discourage running the
experiment that actually settles it.

---

## 1. This is Case A's request shape at a different offered load — proven by machine diff

**What.** `specs/mix_load.yaml` differs from
`Optimus-AgenticBench/agent/workloads/glm52_crxx_caseA.fix.yaml` by **exactly four
lines**, verified by diffing both files with comments and blank lines stripped
(both reduce to 27 significant lines):

| line | Case A | this run | why |
|---|---|---|---|
| `initial_sessions` | 32 | **8** | mission task 3 |
| `max_sessions` | 128 | **24** | mission task 3 |
| `max_inflight` | 48 | **16** | mission task 3 |
| `tokenizer` | `/path/to/GLM-5.2-MXFP4` | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | site path |

**Why it matters.** Three of the four are the mission's load knobs and the fourth
is a path. Everything that defines *what a request looks like* is untouched: the
percentile triples (input 74K/155K/235K, output 320/3300/17000, turns 3/20/103,
inter-turn delay 4/31/240 s), `cache_hit_rate 0.89`, `new_session_rate 0.10`,
`ramp 400 / sustain 3600`, `random_seed 1337`, `max_input_tokens 260000`, and the
`sla:` block. So this measures **Case A at a different offered load**, not a
different workload — and a comparison against any other Case-A result is
legitimate on the request-shape axis.

**How to re-verify.** `REPRODUCE.md` §2 gives the exact command. Do the diff with
comments stripped: the two files differ in comment text as well (this one carries
`# mission task 3` markers), and an unstripped diff buries the four real changes.

**Context / trap.** The mission's own wording asks for these three knobs and
nothing else. If you re-derive this file by hand instead of copying it, diff it
mechanically before running — a stray edit to a percentile triple would silently
change the workload while keeping the name.

---

## 2. The ramp is a warm-up EXCLUSION window, not a load ramp

**What.** `ramp_duration: 400`. Nothing ramps. The driver runs closed-loop; the
offered load is set by the live-session count from tick zero.

**Why it is there.** Every request nests inside the same shared prefix (the
workload models a ~231K-token shared prefix at `cache_hit_rate 0.89`). The first
requests build that prefix in the radix tree and pay for it. Reporting them would
measure cache construction, not steady-state serving. The YAML's own comment sizes
the window at "roughly one session lifetime (~320 s) … so the 231K-token shared
prefix is resident before measurement starts".

**How the evidence shows it worked.**

| | ramp | sustain |
|---|---|---|
| cache hit | **79.3 %** | **88.7 %** |
| TTFT p90 | **18,799.1 ms** | **7,349.1 ms** |
| TPOT p90 | 70.5 ms | 41.7 ms |

The ramp's TTFT p90 is 2.6× sustain's, at a 9.4-point lower cache hit rate. None
of it reaches the reported sustain numbers.

**Context.** `analyze_solo.py` enforces this — it skips every row whose
`phase != "sustain"`. If you re-analyse with your own script and forget that
filter, the ramp's 117 requests will drag the tail and you will be measuring the
wrong thing. Note this differs from the driver's own `summary.json` top-level
block, which is **whole-run** and therefore *does* include ramp; that is the main
reason the two sources disagree (see §6).

---

## 3. OPEN QUESTION: the run is at saturation — but WHICH phase is the limiter was not measured

**What was measured.** Over the 3588 sustain ticks:

| counter | cap | max | mean | ticks at the cap |
|---|---|---|---|---|
| `in_flight` | 16 | 16 | 15.30 | **2588 / 3588 = 72.1 %** |
| `num_sessions_active` | 24 | 24 | 22.42 | **1685 / 3588 = 47.0 %** |

Neither counter ever dropped far: `in_flight` bottomed at 6,
`num_sessions_active` at 14. The driver printed
`WARNING: Hit max_inflight (16) - sessions are being throttled; offered load is
capped` once, at elapsed 406.6 s — the first tick of sustain — and the condition
then persisted for most of the run.

**What that means, stated plainly.** Case A's own YAML defines the reading, on the
`max_sessions` line:

> *"Hitting it means sessions are dying slower than they are born, i.e. the
> server is saturated — treat it as a failure signal."*

Both caps were hit. **`max_inflight` is the binding one** (72.1 % vs 47.0 %). The
throughput numbers in this packup are therefore **capacity-limited by the offered-
load caps**. They describe what the deployment delivered at this offered load. They
are **not** a measurement of the deployment's ceiling, and must not be quoted as one.

**What is NOT claimed.** Which phase is the limiter — **prefill compute, decode
compute, or cache**. Nothing here separates them. The temptation is strong (the
uncached TPM, the TTFT, and the TPOT are all sitting right there) but none of those
three numbers distinguishes a queue that is long because prefill is slow from one
that is long because decode is slow, and no queue-depth data was captured at all.

**What would settle it.**

1. **The engine's own scheduler queue-depth counters**, scoped to this run's
   window. SGLang's `Prefill batch` / `Decode batch` log lines carry
   `#running-req`, `#queue-req` and token usage per iteration. The shipped
   `logs/engine_loadwindow.log.gz` is already sliced to 11:55–13:05, so this needs
   **no cluster access** — parse those lines and look at where requests are
   waiting. This is the cheap first step and it is the discriminating one.
2. **Re-run with the caps raised** (e.g. `max_inflight` 32, `max_sessions` 48) and
   see whether completed-request throughput moves. If it does not, the caps were
   not the limiter and something in the engine is; if it does, this run was
   client-capped and the ceiling is higher. Costs one hour of node time.
3. The `llm-pd-bottleneck-finder` skill applies queueing theory to exactly the
   counters in (1) — but note it is written for disaggregated P≠D deployments, and
   this is a **mix** worker where prefill and decode share the same 8 GPUs. Its
   framing needs adapting before its conclusions transfer.

Do (1) first. It is offline and the data is already in this packup.

---

## 4. OPEN QUESTION: E2E p90 is 5× the p50, and p99 is 13.5×

**What was measured.** Sustain-phase per-request end-to-end latency, n = 1637:

| p50 | p90 | p99 | mean |
|---|---|---|---|
| **13,931.9 ms** | **70,313.9 ms** | **188,236.2 ms** | 28,353.5 ms |

The mean sits at 2.0× the median, so the distribution is heavily right-skewed.
Meanwhile TTFT is comparatively tight (p50 4266.8 → p99 12335.5, only 2.9×) and so
is TPOT (25.3 → 77.3, 3.1×).

**One first-hand bound, and it is worth having.** The completed requests stop just
short of the driver's own client budget:

| completed sustain E2E | value |
|---|---|
| p99 | 188.2 s |
| p99.9 | 228.6 s |
| **max** | **239.0 s** |
| count above 240 s | **0** |

The client budget is `aiohttp.ClientTimeout(total=240)`
(`agent_throughput.py:2253`). So the *observed* E2E distribution is **censored at
240 s by construction** — anything slower could not appear as a completed request;
it would have become one of the 35 timeouts (§5). Any percentile in the far tail
must be read as a percentile of the surviving population.

**What is NOT claimed.** That the spread is caused by the request-size mix, by
queueing at saturation, by the tail of the generation-length distribution, or by
anything else. Note the workload *does* sample a wide shape by design (generation
p50 317 → p99 7,478 tokens, a 24× range), so a wide E2E spread is not by itself
surprising — but **whether the size mix accounts for this spread has not been
computed**, and until it is, nothing about the server follows from it.

**What would settle it.** In increasing cost:

1. **Regress per-request E2E on that request's own generation length**, within this
   run. Both arrays are index-aligned in the shipped `metrics.jsonl`
   (`new_e2es`, `new_generation_lengths`, plus `new_ttfts` and `new_tpots`). If
   E2E ≈ TTFT + (gen_len − 1) × TPOT holds per request with a tight residual, the
   spread is the size mix and there is nothing further to explain. If the residual
   is large, the extra time is being spent somewhere the three components do not
   account for. **Needs no cluster access** — do this first.
2. **The same decomposition split by whether the request was queued**, using the
   engine-side scheduler counters from §3(1). This separates "big request" from
   "waited behind other requests".
3. **Re-run at a lower offered load** (e.g. `max_inflight 4`) with the same request
   shape. Phase 2 is effectively the concurrency-1 end of that sweep already; a
   middle point would show whether the spread is load-induced.

---

## 5. OPEN QUESTION: 35 requests hit the client's 240 s budget — cause unmeasured

> **This section corrects an earlier belief.** It was thought the driver recorded
> no per-error detail and that the `errors` counter in `metrics.jsonl` was the only
> record. **That is wrong** — the driver prints one line per error, the console log
> was captured, and all 35 lines are the same class. The stronger statement below
> is first-hand from that log.

**What was measured.** 35 errors out of 1804 sent (1.94 % of sent; success rate
0.9728). Everything known about them:

| observation | how it was established |
|---|---|
| **all 35 are client-side request timeouts** | 35 × `Request N timed out` in `logs/agentic_load.log.gz`, 35 **unique** request ids |
| **no other error class occurred at all** | 0 × `Request N error: <exc>`, 0 × `failed: HTTP`, 0 × `Traceback` in the same log |
| **the engine rejected nothing** | engine log scoped to 11:56–13:05: **0** lines matching error/abort/reject/invalid/Traceback, and **1802 / 1802 HTTP 200** |
| **they accumulated gradually, at a stable rate** | first at elapsed 658.8 s, then 921.5 / 1026.8 / 1149.2 / 1162.2 / 1205.4 …; running rate 0.41 % at the first, 1.4–1.97 % for the rest of the run. Not a burst, not a cliff. |
| **no session was abandoned** | `num_sessions_abandoned` = 0 on the final tick; `session_abandon_rate` is 0.0 in the config |
| **the two independent records agree 1:1** | each driver-log timeout pairs with exactly one `metrics.jsonl` error increment; max time offset 1.9 s, mean 1.0 s, against a ~1 s tick period |

The budget is `aiohttp.ClientTimeout(total=240)` at `agent_throughput.py:2253`
(the realistic-mode send path — this run is realistic mode, forced by the
`profile:` block). The timeout covers the **whole** request including streaming,
not just connection setup.

**What is NOT claimed — and this is the important part.** *Why* those 35 requests
took longer than 240 s. Specifically, none of the following is established:

- that the server was slow for those requests (a timeout is the **client giving
  up**; the server may have still been streaming perfectly well)
- that saturation (§3) caused them
- that they were the largest requests
- that anything was wrong at all — a 240 s budget against a workload whose p99
  generation is 7,478 tokens is a design choice, and §4 shows completed requests
  reaching 239.0 s, i.e. the population is pressed right up against the budget

**What would settle it.** In increasing cost:

1. **Log the request's planned shape alongside the timeout.** The driver knows
   `plan.request_id`, the sampled prompt length and the sampled `max_tokens` at
   send time, and currently prints only the id. One line of instrumentation at
   `agent_throughput.py:2353` would say whether the 35 were the biggest requests.
   If they were, the answer is "the budget is too small for the p99 shape" and the
   question closes.
2. **Correlate the 35 request ids with the engine's own request log** over the same
   window, to see whether the server finished them after the client hung up. The
   engine reported 1802 HTTP 200 against 1804 sent — a gap of 2, not 35, which
   already hints the server completed most of them, **but the accounting boundary
   between "sent" and "logged in this window" was not established** and that gap
   should not be over-read.
3. **Raise the client budget** (e.g. to 600 s) and re-run. If the errors vanish and
   the E2E tail simply extends past 240 s, they were censoring, not failures.

Do (1) — it is a one-line change and it is decisive.

---

## 6. The two result sources are not interchangeable — different population AND different convention

**What.** This packup reports from two places, and they disagree:

| metric | `summary.json` (whole run) | `analyze_solo.py` (sustain only) |
|---|---|---|
| TTFT p50 | 4272.0 | 4266.8 |
| TTFT p90 | **7564.6** | **7295.8** |
| TTFT p99 | **16873.5** | **12335.5** |
| TPOT p50 | 25.31 | 25.3 |

**Why, two reasons compounding.**

1. **Different population.** `summary.json`'s top-level block is the **whole run**
   — it includes the 117 ramp requests, whose TTFT p90 is 18.8 s. `analyze_solo.py`
   is **sustain only** (n = 1637). That is most of the p99 gap.
2. **Different percentile convention.** The driver uses `sorted[int(q*n)]`;
   `analyze_solo.py` uses `sorted[round(q*(n-1))]`. This was established by test,
   not by reading one implementation — see
   `solo.glm52.mix.packup_20260806/notes.md` §10, where every standard convention
   was evaluated against the driver's own output and it matched numpy's `higher`
   exactly.

Note the driver's own **phase table** (reproduced in `results/RESULTS.md` §2) *is*
per-phase, and its sustain TTFT p90 of 7349.1 ms is close to `analyze_solo.py`'s
7295.8 ms — the residual there is convention alone.

**Context / rule.** Quote one source for a whole row. The reported E2E table uses
`analyze_solo.py` for all of TTFT/E2E/TPOT because **the driver's block has no E2E
column at all** — SOLO_M1 adds the raw per-request array, not a percentile summary.
Mixing the two would silently compare percentiles taken by different rules over
different populations.

---

## 7. Scope EVERY engine-log grep by time window — this run shares its log with two others

**What.** The engine, router and kvd logs are appended across the **entire
session**. The engine log spans **07:02:13 → 13:02:48** and contains all three
phases: Phase 1's fixlen sweep (07:13–09:45), Phase 2's three solo arms
(09:53–11:52), and this run (11:56–13:03).

**Why it matters — with this run's own numbers as the demonstration.** MTP
acceptance, read two ways over the same file:

| scope | n | p10 | median | p90 | at 4.00 |
|---|---|---|---|---|---|
| whole log (all 3 phases) | 51,685 | 2.73 | **3.55** | 3.98 | **9.1 %** |
| **this run's window only** (11:56–13:05) | 8,918 | 2.48 | **3.14** | 3.82 | **4.5 %** |

A median of 3.55 vs 3.14 and an at-4.00 rate of 9.1 % vs 4.5 % are materially
different characterisations of the same deployment. Both are reported in this
packup, each labelled with its scope. **Neither is wrong; quoting one as the other
would be.**

**How.** The lines carry timestamps; filter on them. `REPRODUCE.md` §7 has the
script, and `scripts/scan_err.sh` exists precisely for this — it takes a window
prefix as its second argument. `scripts/accept_len.sh` does the **unscoped**
version; use it only on a log you know belongs to one run.

**Also.** Use `strings` on these logs — they carry binary bytes and a bare `grep`
then reports only "binary file matches". And never poll for readiness by grepping
a log for a ready line: it matches the *previous* run's line and returns early.

**Context — what this packup ships.** `logs/engine_loadwindow.log.gz` is the engine
log **already sliced** to 11:55–13:05 (227 KB gz, vs 1.0 MB gz for the full file).
Use it and the scoping problem cannot bite you. The full log is not shipped here —
it is in `fixlen.glm52.mix.packup_20260806/logs/` for the earlier window, and lives
on the node at `/tmp/glm52_mix_base.log` inside the container.

---

## 8. The SOLO_M1 driver patch is load-bearing here too

**What.** The staged driver carries a local patch, **SOLO_M1**, never committed
upstream. It appends `new_e2es` and an index-aligned `new_tpots` to each metrics
tick. `patches/solo_m1_per_request_e2e_tpot.patch` in this packup is the same file
shipped in `solo.glm52.mix.packup_20260806/patches/`, carried so this packup stands
alone; that packup's `patches/README.md` holds the full rationale.

**Why it matters to *this* phase.** Every E2E number in this packup — the p50
13,931.9 ms, the p90 70,313.9 ms, the p99 188,236.2 ms, and the 239.0 s maximum
that bounds §4 and §5 — exists **only** because of it. Upstream records neither
per-request end-to-end latency nor an index-aligned TPOT array, so without the
patch the E2E column would have to be back-solved from TTFT and TPOT, and the
censoring observation in §4 could not be made at all.

**How to apply it cold** (the staged driver is on a shared mount, not in this repo):

```bash
git clone <Optimus-AgenticBench> agbench && cd agbench
git checkout 1cf01cbf169d9370a0bc8fe574055c5e975d1be9
patch -p1 < <this packup>/patches/solo_m1_per_request_e2e_tpot.patch
```

**How to verify it took**, after a run:

```bash
python3 -c "
import json
r=json.loads(open('<results>/metrics.jsonl').readline())
assert 'new_e2es' in r and 'new_tpots' in r, 'SOLO_M1 patch is NOT applied'
print('SOLO_M1 present')"
```

**Context.** The baseline is provable, not assumed: the staged tree keeps
`agent_throughput.py.orig` alongside the patched file, and that `.orig` md5s
identical to the pristine checkout at `1cf01cb`, so the diff between them is
exactly the patch. Details in the Phase-2 packup's `environment.md`.

---

## 9. Smaller things worth knowing

- **`analyze_solo.py` works unchanged on this run** despite its name and docstring
  saying "concurrency-1". Nothing in it assumes concurrency 1 — it concatenates the
  per-tick arrays row-wise and filters to sustain. Verified by running it: it
  reproduces the shipped table to the decimal.
- **`new_e2es` is in SECONDS**, not milliseconds; `analyze_solo.py` multiplies by
  1000. Reading the raw array as ms understates E2E by 1000× and will make you
  think the tail is fine. (Caught exactly this way while assembling this packup.)
- **The `drain` phase is one request** with `duration_s: 0.0`, and the driver
  prints `n/a` for its rates. It is excluded from every reported number here.
- **`num_sessions_total` is 24, not 236.** The driver reports "Total sessions: 24
  (initial: 8, +228 rate-based)" — 24 is the *live population cap*, and 228 is the
  cumulative count of sessions born over the run. Do not read 24 as "only 24
  sessions ran".
- **`max_inflight` binds before `max_sessions` here** — the opposite of Phase 2,
  where `max_sessions: 1` was the real gate and `max_inflight` never fired. Which
  one binds depends on the ratio; do not carry the Phase-2 intuition over.
- **Never greedy-decode this model.** `--temperature 1.0 --top-p 0.95` is the
  checkpoint's own `generation_config`. At temperature 0 it repeats, MTP pins
  acceptance at 4.00, and the run reads like KV corruption. Results are therefore
  **not bit-reproducible** — compare distributions.
- **`--dashboard-mode` is mandatory** in `run_agentic.sh`. Without it nothing
  structured is persisted and the run is unrecoverable once the terminal scrolls.
- **No load knobs on the CLI, by design.** `run_agentic.sh` passes only the YAML;
  passing `--initial-sessions` would silently shadow the file and make the run
  unreproducible from it. This matters more here than anywhere else in the bench —
  the load knobs *are* the experiment.
- **`--tokenizer` on the CLI overrides the YAML's `tokenizer:` field.** Both point
  at the same path here, so it is harmless, but the driver announces the override
  and a silent divergence is waiting to happen.
- **The driver prints `PyTorch was not found` at startup.** Expected — it only uses
  `tokenizers`.
- **`system_prompt_len: 2000` produced a 5,603-token prefix** (the driver reports
  `Final prefix sizes: min=5,603, max=5,603, mean=5,603`). The requested value is a
  target for a synthetic generator, not an exact length.
- **The p99 prompt came in at 222,770 tokens against a 235,000 target.** Unlike
  Phase 2, this workload samples a real distribution, so the realised p99 is an
  order statistic of 1,755 draws. It sits inside `context_len=262144`; a larger
  Case-A shape would need `--context-length` raised.
- **`chunked_prefill_size` and `max_running_requests` are GLOBAL budgets that
  DP-attention divides by `dp_size`** — 65536 → 8192 and 256 → 32 per rank. Verified
  in this image's own source (`srt/server_args.py:4902`,
  `srt/model_executor/pool_configurator.py:541-543`); full analysis in
  `fixlen.glm52.mix.packup_20260806/notes.md` §3. These are per-rank values, **not
  flags that failed to take**. At this run's load they matter: the per-rank
  `max_running_requests=32` is the engine-side admission limit sitting behind the
  client's `max_inflight=16`.
- **Reasoning tokens are billed against the same budget as content**
  (`--reasoning-parser glm45`). This run generated 456,672 reasoning tokens out of
  1,589,656 total (28.7 %).
