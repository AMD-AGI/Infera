# Notes — gotchas, wrong turns, and why each knob is what it is

Written as what / why / how / context. The traps here are mostly *measurement*
traps: this experiment's failure mode is not a crash, it is a run that completes
green while measuring something other than what you think.

---

## 1. `max_sessions`, not `max_inflight`, is what pins concurrency

**What.** Concurrency 1 is enforced by `max_sessions: 1`. `max_inflight: 1` is a
belt-and-braces valve that never fired.

**Why it matters.** The obvious knob is the wrong one. `max_inflight` throttles
by *blocking inside* `send_realistic_request` — it spins on
`await asyncio.sleep(0.01)` until the window has room. That produces concurrency
1, but it also means the *driver* is the bottleneck, and the driver prints
`Hit max_inflight` and warns that "offered load is capped". Numbers taken under
an active backpressure gate are not a clean latency measurement: request
dispatch is no longer governed by the session model.

**How the right one works** (`agent_throughput.py`, `run_realistic` control
loop): sessions are spawned only while `active_sessions < max_sessions`, and
`count_active_sessions()` counts a session with `in_flight == True`. So while a
request is on the wire, no second session can be born. Nothing ever queues, so
the valve is never touched.

**Context / how to verify.** Two independent checks, both must pass:

```bash
grep -c 'Hit max_inflight' logs/solo_full.log        # want 0
# and, from the raw metrics:
zcat results/metrics.jsonl.gz | python3 -c "
import sys,json; print(sorted({json.loads(l)['in_flight'] for l in sys.stdin}))"
# want [0, 1]
```

Ours: 0 warnings, `{0, 1}`. If you ever see `[0,1,2]`, the workload file that
loaded was not the one you think.

---

## 2. `turns_per_session: {1,1,1}` also removes the think time — on purpose

**What.** Setting all three percentiles to 1 makes each session issue exactly
one request and retire.

**Why it works exactly.** `PercentileSampler(1,1,1)` degenerates:
`vmin = p50²/p90 = 1` and `vmax = p99·(p99/p90)² = 1`, so `sample_int()` returns
1 with probability 1. Verified offline before the run rather than assumed:

```python
from agent.sampling import PercentileSampler
s = PercentileSampler(1,1,1)
assert {s.sample_int() for _ in range(2000)} == {1}
```

**The non-obvious consequence.** In `session_loop`, the turn-budget check
`break`s **before** `await asyncio.sleep(think_time)`. So `inter_turn_delay_s`
never executes. This is what makes the duty cycle 96.4 % instead of the ~44 %
you would get if the 18.4 s mean think time still applied — and it is why the
30-minute window yields 102 samples rather than ~50.

**Context.** `inter_turn_delay_s` is deliberately left in `spec/solo.yaml` even
though it is unreachable. Deleting it would make the file read like a different
workload; it is not. `new_inter_arrival_times` is empty in the output for the
same structural reason (the driver only appends it from a session's second turn
onward) — **not a bug**, and the analyzer prints `NO SAMPLES` rather than
silently reporting nothing.

---

## 3. The driver silently discards the two things this experiment measures

**What.** Per-request E2E is never persisted; per-request TPOT is persisted only
as three summary percentiles.

**Why.** `total_time` is computed at `agent_throughput.py:2310`, handed to a rate
tracker, and dropped. `metrics.actual_tpots` lives in memory and dies with the
process. `save_metrics_loop` writes `new_ttfts` but has no counterpart for
either.

**Consequence if unnoticed.** You get a green 37-minute run whose output cannot
answer the question. Case A hit the softer version of this and had to
*back-solve* E2E from `TTFT + (gen−1)×TPOT`.

**How fixed.** `patches/0005` (`SOLO_M1`) adds `new_e2es` and `new_tpots` to the
1 Hz JSONL. Additive only — no existing value changes.

**The one subtlety worth knowing.** `actual_tpots` is *filtered*
(`gen_len > 1 and gen_time >= 50 ms`), so it is **not** index-aligned with
`new_ttfts` and cannot be sliced by the same cursor. Slicing it with
`last_distributions_index` would silently misalign TPOT against TTFT — every row
shifted by however many requests had been filtered out, which is exactly the
kind of bug that produces plausible, wrong numbers. So the patch adds a
*separate* `actual_tpots_aligned` that appends on **every** request, writing
`0.0` where the sample was filtered. The analyzer drops zeros; it does not treat
them as fast tokens.

**Payoff.** With a measured E2E in hand, the composition identity becomes
checkable: predicted `TTFT + (gen−1)×TPOT` minus measured E2E is **+0.0 ms at
every percentile, 102/102**. That retroactively validates Case A's back-solve.

---

## 4. Preview's warnings are wrong here — expected, and must be ignored

**What.** `--mode preview` emits three warnings for this config:

```
- max_inflight=1 throttles even at zero latency; ... the session population runs away
- max_sessions=1 is reached even at zero latency
- initial_sessions=1 is far from the steady-state range 23-58
```

**Why they are wrong.** Preview models offered load with the open-loop Little's-
law formula `N = rate × turns × (E2E + delay)`. With `new_session_rate: 1.0` that
predicts a runaway population. It does not know that the real control loop gates
spawning on `active_sessions < max_sessions`.

**How to respond.** Do not "fix" them. Read the one line that *is* load-bearing:

    turns_per_session   1.0   1.0   1.0   1.0    1/1/1

and confirm the input/output triples match Case A's. Then verify the population
empirically at runtime (`In-flight` must never read 2).

**Context.** This is the mirror image of the Case A trap, where the probe showed
`N≈17` against a nominal 32 and the tempting "fix" — doubling the birth rate —
would have pinned `max_inflight` and let backpressure set the load. In both
cases the model disagrees with the loop, and the loop is what runs.

---

## 5. Grepping the engine logs for faults gives a false positive

**What.** The Case A kit's fault scan, run verbatim, returns `1` on the prefill
leg and `3` on the decode leg — on a run that was actually clean.

**Why.** Two independent causes:
- The `server_args=ServerArgs(...)` dump is one enormous line containing the
  substring **`abort_on_priority_when_disabled=False`**, which matches `abort`.
- These logs are **appended all day**. The decode leg's other two hits are
  `Aborted by AbortReq` at 13:51 and 14:22 — inside the *Case A* window
  (13:34–14:41), hours before this run.

**How to scan correctly.** Scope by timestamp, and keep using `strings` (server
logs contain binary bytes, so plain `grep` misbehaves):

```bash
strings $W/logs/p6_decode.log \
  | grep -E "^\[2026-08-01 16:[0-4][0-9]" \
  | grep -icE "Expected lengths.size|Aborted|OUT_OF_RESOURCES|Traceback|exception"
```

Ours: **0 on both legs** within 16:00–16:49.

**Context.** This generalizes: any long-lived appended log needs the window
scoped before you can claim a run was clean. Reporting "3 aborts" here would
have been a false alarm that discredited a good run.

---

## 6. An env snapshot older than the run is only OK if you can prove no restart

**What.** `env/*.txt` is stamped 11:21 UTC; the run is 16:05 UTC.

**Why it's acceptable here.** Direct evidence, not assertion:
`docker inspect` shows both `bench_run` containers started 09:18, and
`ps -eo lstart` shows the `sglang.launch_server` processes started 11:47
(prefill) and 13:29 (decode) — both still the same PIDs at 16:05. Same
processes, same weights, same patched bytecode as Case A.

**How to check.** See the command block in `environment.md`.

**Context.** Had either process restarted, the P1V3 patch would have been lost
(it is applied at runtime inside the container, not baked into the image) and
the run would have crashed within minutes — so the clean run is itself
corroborating evidence.

---

## 7. Verify the *loaded* module, never the file on disk

**What.** Both patches (`SOLO_M1` on the driver, `GLM52_P1V3` on the engine) are
checked by importing the module and inspecting its source, after deleting
`__pycache__`.

**Why.** Stale bytecode has invalidated a full experiment in this tree before. A
patched `.py` with a cached `.pyc` gives you a green run of unpatched code.

**How.**

```bash
# driver
$W/venv/bin/python -c "import agent.agent_throughput as m, inspect; \
  print(inspect.getsource(m).count('SOLO_M1'))"        # 8

# engine, inside the decode container
docker exec bench_run python3 -c "import sglang.srt.layers.attention.dsa.dsa_indexer as m, inspect; \
  print(inspect.getsource(m).count('GLM52_P1V3'))"     # 3
```

`solo_run.sh` additionally hard-gates on the driver marker and refuses to launch
without it — a guard rail so a forgotten patch costs seconds, not 37 minutes.

---

## 8. `pkill -f infera.kvd` kills the engine too

**What.** Never run it bare. Use `scripts/restart_kvd.sh`.

**Why.** `-f` matches against the full command line as a **regex**, and `.` is a
wildcard. `infera.kvd` therefore matches the engine's own
`--infera-kvd-socket ...` argument. The kvd daemon and the engine both die.

**Context.** Inherited hazard, not hit during this run, but it is one keystroke
away any time you touch kvd on these nodes.

---

## 9. What the numbers cannot support

Stated here as well as in the analysis, because caveats are the first thing
dropped when a table gets copied into a slide.

- **n = 102 in the measured window.** p50/p90 solid; **p99 rests on ~1
  request**. Intrinsic to concurrency 1. Buying samples would mean shortening
  requests, which breaks shape-parity with Case A and invalidates every
  comparison in the report.
- **The input tail is under-sampled** (solo p99 184.6K vs the spec's 235K). The
  bucketed TTFT table exists precisely to work around this; the unbucketed
  ratios do not, and are the weaker number.
- **The ~3.9 s queueing penalty is measured at one load point** — Case A's ~27
  live sessions. Whether it is flat, linear, or knee-shaped in concurrency needs
  a sweep. This run and Case A are two points.
- **Not a capacity measurement.** 0.066 qps, 3,710 uncached TPM/GPU against Case
  A's 53,190.
- **kvd tiering is still unexercised** (+0 gets). A single warm stream nesting in
  one prefix is the *least* demanding case for the spill tier, not the most.
