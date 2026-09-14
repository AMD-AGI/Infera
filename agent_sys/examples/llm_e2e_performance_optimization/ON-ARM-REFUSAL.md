# Why `run_profiling_mode_on` refused — and why `stack_window_s=0` is a silencer, not a fix

**Written 2026-09-06, m35.** Forensic read of run `20260906T113811-fdb0bd`, the
best board of the day. **No GPU touched; chain 6 was live throughout.**

This refusal is upstream of everything: it stopped `merge_profiling_evidence`,
which has never run, which is why `profiling_evidence` has never existed, which
is why m3/m4/m5 have never started. **Nobody had written down why it refused.**

---

## 1. Materials check FIRST — the refusals are attributable

```
$ bash assets/lib/refusal_saw_something.sh <run>
ok  validation.60b6091e….output_v  files: 30  versions: v1
ok  validation.27c7b833….output_v  files: 17  versions: v0
ok  validation.64619ce7….output_v  files: 39  versions: v0     <- the _on zone
3 zone(s), none empty
```

**The `_on` zone was handed 39 files.** Not a zero-file zone, so everything below
is about the artefact rather than about staging. **Reported before the diagnosis,
because after it would be worthless.**

Verdicts in that zone: **2 false, 3 true.**

---

## 2. The two refusals, verbatim — and both read the file

### `check_kernel_table` on `kernel_table`

```
note:    (note) 130 kernels, top 25 cover 82.5%, shares sum to 100.01
note:    layout: structured_text (items/text.json) — the shape this validator grades
PROBLEM: no launcher frames were resolved (the torch_trace handoff carries no stack
         window) and this round wanted at least 10 in the head. Set --var
         stack_window_s=0 and min_launchers_in_top_n to 0 to say that is intended
```

### `check_trace_coverage` on `profiling_mode_on.profile_result`

```
note:    (note) re-parsed 1788696891.2799642-TP-2.trace.json.gz: 1395036 events,
         99892 GPU kernels, 11.02s — manifest agrees
note:    (note) 4 rank(s), 399528 GPU kernel events
PROBLEM: items/result/stacks_manifest.json is missing — the round was asked for a
         stack window and this handoff carries none, so no launcher frame can be
         resolved from it. Set --var stack_window_s=0 to say that is intended
```

### Applying the test in `CLAUDE.md`: does the refusal quote a number from inside a file?

**Yes, both — decisively, and in the strongest available form.**

`check_trace_coverage` **re-parsed the trace from scratch** and reports 1395036
events / 99892 GPU kernels / 11.02 s, then says *"manifest agrees"*. That is two
independent routes to one number: the manifest's claim and the validator's own
recount. **A refusal that recounts cannot be looking at an empty directory.**

`check_kernel_table` computes `130 kernels, top 25 cover 82.5%, shares sum to
100.01` from the table itself.

**Worth being precise about the mixed shape:** the *notes* prove the artefact was
read; the *PROBLEM* lines are absence claims (`stacks_manifest.json is missing`).
By the criterion alone an absence claim proves nothing — **here it is redeemed by
the 39-file count and by the recount in the same report.**

**A by-product worth keeping:** `check_trace_coverage` counted **4 ranks and the
manifest agreed**, so **`--var expect_ranks=4` was correct for this run.** One
launch variable positively confirmed rather than assumed.

---

## 3. Producer defect or launch-variable mismatch? **Producer, and the mechanism is timing.**

The refusals say *"the round was asked for a stack window and this handoff
carries none."* The round did ask: `shared.yaml:173`,
`E2E_STACK_WINDOW_S: '${stack_window_s:-3}'`, and this run did not override it.

**So the producer tried.** `assets/load/replay.sh:123-136` takes the `>0` branch
and runs a second capture — and the attempt left its own log:

```
$ cat .../load.profiling_mode_on/capture_stacks.log
===== 1/6 preflight =====
  traces are governed by mount /data/yihou/e2e_flow/pmon (rw) in yihou_e2e_chain_…_pmon
  control plane ON (probe -> 400 invalid role, as expected)
  mixed workers registered: 1
  ABORT: no aiperf load in flight. Start 06_aiperf_replay.sh first.
```

`capture.sh:130` requires a live load: `load_running(){ pgrep -f 'aiperf profile'; }`.

### The load was genuinely over — this is timing, not a broken detector

I considered both, because `pgrep -f <literal>` is exactly the fragile-detector
shape this project keeps paying for. **The timestamps settle it:**

```
aiperf.log          last written 12:14:58     <- the load ENDED here
capture.log         12:15:29, window recorded inside it: 12:14:51 .. 12:15:19
capture_stacks.log  12:15:29  (the abort)
```

- the **measurement** capture started **12:14:51**, while the load was alive → preflight passed;
- the **stack** capture started after that window closed at **12:15:19**, roughly
  **21 seconds after the load had finished** → preflight correctly refused.

> **`pgrep` was right. There really was no load in flight.** The defect is that
> **the measurement window outlives the AIPerf replay**, so the second capture —
> which by construction needs load — is scheduled into a period when there is
> none. `capture.sh` checks at preflight, i.e. at start, which is why the first
> capture survived the load ending mid-window and the second did not.

### The producer treats this as non-fatal on purpose

`replay.sh:114-121`, verbatim:

> *"**Not fatal when it fails.** The ranking is complete without it and the
> launcher block is an enrichment, so losing the stack window must not cost the
> round its aiperf report and its traces. Whether a round missing it is
> acceptable is `check_trace_coverage`'s call."*

**So the design is: producer tries, fails softly, validator decides.** The
validators decided *not acceptable* — **which is the package working exactly as
written.** The refusal is correct.

**This is not a mocked-vs-real variable mismatch.** I checked the class that cost
the other cluster three launches: `expect_ranks` was **confirmed correct** (§2);
`adhoc_cases` and `bench_rounds` are m5 knobs and do not reach the `_on` arm.
**No variable in that class is implicated.**

**Caveat, stated rather than glossed:** this run does not record its own launch
line, and its orchestrator is long dead, so I cannot read back with certainty
that `stack_window_s` was unset. **What settles it:** the refusal text itself —
a validator only says *"the round was asked for a stack window"* when
`E2E_STACK_WINDOW_S > 0`, and `replay.sh` only writes `capture_stacks.log` on the
same condition. **Two artefacts agree that it was >0**, which is stronger than a
launch record would have been.

---

## 4. Does `stack_window_s=0` interact with the `_on` arm's checks? **Yes — it is named by both refusals as the way to silence them.**

Not a cleared suspect; the opposite. Both refusals name it explicitly, and
`m2_profiling.yaml:131` documents it: *"Set to 0 when `--var stack_window_s=0`
says the round is deliberately taken without it."*

**Setting it to 0 does make both refusals go away, and it fixes nothing.**

| | with `stack_window_s=3` (default) | with `stack_window_s=0` |
|---|---|---|
| stack capture attempted | yes — and aborts, load already gone | **no** |
| `stacks_manifest.json` | absent | absent |
| `check_trace_coverage` | **refuses** | passes |
| `check_kernel_table` | **refuses** (also needs `kernel_table_min_launchers=0`) | passes |
| launcher frames for m3 | none | none |

> **The artefact is identical in both columns. Only the question changes.** `=0`
> converts *"you asked for this and did not get it"* into *"you did not ask"* —
> a correct thing to say, and it declares a capability abandoned rather than
> repaired.

**And it has a downstream price, which is why it belongs in this document:**
`identify.py:9` resolution **level 1** (`trace_python_stack`) reads exactly the
launcher block that a stack window produces. On the first cluster level 1
resolved all five operators at `resolve_ratio: 1.0` and the fallback was never
exercised. **With `=0`, m3 falls to level 2 — `kernel_finder`, fed by
`magpie_root` — which has never completed a real scan on this cluster.** And **[CORRECTED 2026-09-06T14:13:13Z: Magpie HAS now completed a real scan on this cluster. On run `20260906T130845-298750` `check_kernel_table` PASSED, which it cannot do without a kernel table, so Magpie produced one. Observed by m2 from the verdict side; I had listed this as an unknown and leaned on it. It weakens the m3-degradation reasoning: a thin worklist now needs a cause other than Magpie never having run.]** 
`min_resolve_ratio` defaults to `0.0`, so nothing refuses if it resolves nothing.

**Note the trap in the refusal's own advice:** it says *"set …
`min_launchers_in_top_n` to 0"*. **That is the validator's args field. The `--var`
is `kernel_table_min_launchers`** (`steps/common.yaml:162`), and an unrecognised
`--var` produces byte-identical `show` output — no error, no warning. Recorded as
bug-record §7c; run `041f89` passed the correct name.

---

## 5. What would settle what remains open

| open question | the one thing that settles it |
|---|---|
| Does the stack capture succeed if the load outlives the measurement window? | a run with the AIPerf replay longer than `warmup + E2E_WINDOW_S`, then `grep CAPTURE_OK .../capture_stacks.log`. **`--var trace_end_ms` is the load's length knob.** |
| Is `merge_profiling_evidence` cheap, or does it have its own failure? | a run that gets past `_on` validation. **It has never executed**; no code read substitutes. |
| Does m3 resolve anything at level 2 on this cluster? | the first real `operator_workset` — then `python3 m3_extract.py <workset>` and read `entry_function`. |

## 6. The recommendation I would make, and its cost

**`stack_window_s=0` is the right call for reaching `packup` and the wrong call
for a chain that wants launcher attribution.** It buys progress by giving up the
resolution route that worked on the other cluster.

**The cheaper repair, if a later round wants it:** lengthen the load rather than
shorten the ask — make the AIPerf replay outlast the measurement window so the
stack capture's preflight finds load in flight. **That is a `--var trace_end_ms`
change, not a code change**, and it is untested. I have not run it and am not
claiming it works; I am naming it because it addresses the mechanism instead of
the symptom.

---

*Stopped here at the hold boundary. Everything above is read from `fdb0bd`'s own
artefacts and the package source; nothing is inferred from a run I did not open.*

---

## 7. Amendment 2026-09-06T13:28:36Z — "refused; the run stopped here" was imprecise, and the precise version is better

The leader challenged step 1: **`output_validating` is a STATE, not a verdict**, and
two worlds fit it — a real refusal, or validation still running when the run was
killed. Neither of us had established which. **The challenge was right as method;
the evidence already existed and settles it.**

### A refusal definitely happened — two artefacts, not an inference

```
verdict.json  validation-u2o0w8ix   {"a56d7e08…": false}   + validator_report.txt "REFUSED"
verdict.json  validation-js25li68   {"88274848…": false}   + validator_report.txt "REFUSED"
```

**World 1. Not "the task sat in a state" — two verdicts on disk say `false`.**

### But what happened AFTER the refusal I had NOT established, and it is the documented death

From `store/event`:

```
validation_failed  task=64619ce7 (run_profiling_mode_on)  "output_validation did not pass"
validation_failed  task=b80e029c (m2_profiling)           from_task=64619ce7
validation_failed  task=b8d93b5d (main)                   from_task=b80e029c

escalated  64619ce7   "validation_failed: the task is terminal and there is nothing to push"
escalated  b80e029c   same
escalated  b8d93b5d   same
escalated  b8d93b5d   same, target: user
```

**Three `validation_failed` are ONE refusal propagating up the task tree** —
leaf → `m2_profiling` → `main`, each naming its `from_task`. **A ledger counting
these as three failures would report three where there is one**, which is exactly
the trap `mission.verify.e2e.md` defect 3 names, visible here in the event stream.

**Four `escalated`, all with the same `why`: *"the task is terminal and there is
nothing to push"*, the last addressed to `target: user`.** That is
`mission.verify.e2e.md` **defect 2 in its exact documented form** — a program body
exits with a failure, there is no agent to instruct, the escalation climbs to the
root and is addressed to a user who is not there.

**And there is no terminal event.** Event kinds present: `phase_done`,
`subgraph_done`, `validation_failed`, `escalated`. **No `done`, no run-complete.**
The run never concluded on its own; it sat with an unanswered escalation at the
root until it was superseded.

### So the corrected sentence

**Not** *"refused; the run stopped here"* — that reads as if validation halted the
run. **The accurate version:**

> **Two validators refused. The refusal propagated to the root as a single
> failure. The escalation had no receiver, so nothing retried and nothing
> concluded, and the run was still sitting there when it was superseded.**

**Nothing in §1–§6 changes** — the materials check, the recounted numbers, the
timing diagnosis and the `stack_window_s=0` analysis all stand. **What changes is
that the run's end is now attributed to the no-receiver escalation rather than
left implied.** The blocker upstream of m3/m4/m5 is real, and the run's death is
a second, separate, already-documented defect that happened to it.

---

## 8. The repair, made actionable — and it reattributes the cause (2026-09-06T13:31:01Z)

**UNTESTED. Not run. Stated as arithmetic and a hypothesis.**

### 8.1 The knob exists, and it is `trace_end_ms`

| where | what |
|---|---|
| `shared.yaml:149` | `E2E_TRACE_END_MS: '${trace_end_ms:-180000}'` — **this is the one m2's load path reads** |
| `steps/m5_integration.yaml:126` | `E2E_TRACE_END_MS: '${trace_end_ms:-60000}'` — same `--var`, **different default** (bug record §3) |
| `assets/load/replay.sh:86` | passes it to the replay as `TRACE_END_MS` |
| `assets/load/aiperf_replay.sh:40` | `TRACE_END_MS="${TRACE_END_MS:-120000}"` — the script's own fallback |
| `assets/load/aiperf_replay.sh:100` | `--fixed-schedule-end-offset $TRACE_END_MS` — **the actual load length** |

**`m2_profiling.yaml` does not declare it**, so m2's arm takes the one in
`shared.yaml`.

*(Three words were lost from this line in the first commit: an unescaped
backtick pair in an unquoted heredoc ran `shared.yaml` as a command. The shell
said `command not found` and I checked the written file rather than the exit
status, which is the only reason it did not ship. Same family as `2>/dev/null`
on a command whose failure matters — the diagnostic was there and the artefact
is what needed reading.)*

### 8.2 The arithmetic — and it reattributes the cause

The two captures run **in sequence inside one load**:

```
measurement capture   WARMUP_S + WINDOW_S       = 60 + 10 = 70 s   (m2_profiling.yaml:370-371)
stack capture         WARMUP_S=0 + STACK_WINDOW = 0  + 3  =  3 s   (m2_profiling.yaml:367)
                                                        ------
                    the load must outlast at least          73 s   + both captures' setup
```

**Every launch today passed `--var trace_end_ms=60000`** — checked, all six
launch records. **A 60-second load cannot cover a 73-second sequence.**

> **So this is NOT a producer code defect, and §3's heading is narrowed by this
> section.** The ordering works at the package's own defaults —
> `aiperf_replay.sh`'s fallback is **120000** and `shared.yaml`'s is **180000**,
> both comfortably over 73 s. **The failure was introduced by a deliberate
> time-saving override**, and nothing warns that `trace_end_ms` has a floor set
> by `warmup_s + window_s + stack_window_s`.

**The relationship between those four variables is unstated anywhere.** That is
the defect: not the ordering, but an **unenforced dependency between launch
variables** — the same family as the mocked/real class, arriving as a
*constraint* rather than a mismatch.

### 8.3 The value I would pass

```
--var trace_end_ms=120000        # aiperf_replay.sh's own fallback; 47 s of margin over the 73 s floor
```

The cheaper alternative, if wall-clock matters more than fidelity:
`--var warmup_s=20` drops the floor to 33 s and keeps a 60 s load — **but warmup
exists to let the engine reach steady state, so shortening it trades a known
quantity for an unknown one.** I would move the load, not the warmup.

### 8.4 What else moves when it moves — three cleared, one real

| threshold | function of | moves? |
|---|---|---|
| `check_trace_coverage.max_span_ratio` (4.0) | `max_span = window_s × ratio` (`check.py:96-97`) — **`window_s`, not the load** | **No.** Cleared. Observed span 11.02 s against `window_s=10` → ratio 1.1 |
| `check_trace_coverage.min_gpu_kernels_per_rank` (1000) | trace content | **No** — a longer load gives *more* kernels |
| `check_bench_result.min_requests` (50) | request count | **No** — 437 recorded at 60 s; more at 120 s |
| **`check_no_regression.stock_vs_m2`** | m5's stock arm vs **m2's bench** | **YES, and this is the one to watch** |

**Why the last one moves:** the replay window selects *which slice* of the
Mooncake trace runs, the trace carries `hash_ids`, and prefix hit rate decides
how much prefill there is. **A different window is a different workload.** So
**m2 and m5 must use the same `trace_end_ms`** — which they do today, because
one `--var` feeds both declaration sites. **If anyone raises it for m2 alone,
`stock_vs_m2` is comparing different workloads and will refuse for a reason that
looks like a regression.**

### 8.5 Status

**Untested.** I have not run it and the hold ended before I could. What settles
it: a run with `--var trace_end_ms=120000`, then
`grep CAPTURE_OK <run>/…/load.profiling_mode_on/capture_stacks.log`. **A
`CAPTURE_OK` there is the whole test.**
