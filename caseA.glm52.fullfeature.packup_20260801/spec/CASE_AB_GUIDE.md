# Running the GLM-5.2 `crxx` Case A / Case B workloads

Operational guide for `agent/workloads/glm52_crxx_caseA*.yaml` and
`glm52_crxx_caseB*.yaml`. These two configs behave differently from every other
workload in this repo, and most of the ways they go wrong are silent — the run
completes, prints a summary, and the numbers are simply not the workload you
asked for. Read the [pitfalls](#pitfalls) section before your first real run.

## Use the `.fix.yaml` files

| file | use it? |
|---|---|
| `glm52_crxx_caseA.fix.yaml` | yes |
| `glm52_crxx_caseB.fix.yaml` | yes |
| `glm52_crxx_caseA.yaml`, `glm52_crxx_caseB.yaml` | original spec, kept for provenance — do not run |

The originals describe the *request shape* correctly (that part is unchanged in
the corrected files) but describe *load* in open-loop vocabulary — QPS, ramp
rate, in-flight cap — for a driver that runs these workloads closed-loop. Every
one of those knobs is inert here, and the measurement window they specify is
roughly 6x too short to reproduce the turn distribution they calibrate.

These files also assume the corrected driver. Against an older
`agent_throughput.py` they silently under-load: the turn budget gets halved, the
lognormal session lifetime censors the tail, and `max_inflight` is ignored.

## The one thing to understand: these run closed-loop

A `profile:` block selects the realistic session/turn runner. In that mode each
session issues **one request at a time** and waits for the full response before
starting its inter-turn delay. Nothing anywhere asks for a request rate.

So offered load is set by the number of live sessions, and that population is a
birth–death equilibrium, not a setting. By Little's law:

```
N_live  =  new_session_rate x E[turns] x (E2E + E[inter_turn_delay])
```

Two consequences worth internalizing before you tune anything:

**E2E is on the right-hand side.** The population you get depends on how fast
the server answers. A slower server means longer-lived sessions means a *larger*
standing population. You cannot set `N_live` directly; you solve for
`new_session_rate` given an E2E you have actually measured.

**The offered request rate does not depend on E2E.** Substituting, the rate
reduces to `new_session_rate x E[turns]`. A session issues its whole turn budget
however long each turn takes, so a slower server changes the population but not
the demand. Demand here is birth-limited. If you want more load, raise
`new_session_rate` (and `initial_sessions` with it) — nothing else moves it.

## Step 1 — dry run (no GPU, no server)

```bash
python3 -m agent.agent_throughput \
  --workload-config agent/workloads/glm52_crxx_caseA.fix.yaml \
  --mode preview \
  --model glm5.2-mxfp4 \
  --tokenizer /path/to/GLM-5.2-MXFP4
```

`--model` is required by the parser even though preview never contacts a server;
any string works. The tokenizer path must be real — preview measures actual
prompt-build cost with it.

This prints the realized distributions against the spec triples, solves Little's
law across a range of E2E, reports where `max_inflight` / `max_sessions` start to
bind, and tells you whether `sustain_duration` is long enough for the p99
session. It ends with either `Plan looks self-consistent.` or a numbered list of
problems. Fix them before spending GPU hours.

To check only that the sampler reproduces the percentiles, the older offline
validator still works:

```bash
cd agent/workloads && python3 validate_crxx_profile.py glm52_crxx_caseA.fix.yaml
```

## Step 2 — the real run

```bash
python3 -m agent.agent_throughput \
  --workload-config agent/workloads/glm52_crxx_caseA.fix.yaml \
  --server http://127.0.0.1:30000 --model glm5.2-mxfp4 \
  --tokenizer /path/to/GLM-5.2-MXFP4 \
  --name caseA --data-dir ./benchmarks --dashboard-mode
```

No `--mode` flag: the `profile:` block selects the runner. Swap in
`glm52_crxx_caseB.fix.yaml` for Case B. From an installed package the entry
point is `agent-bench agent` with the same flags.

## Step 3 — re-solve `new_session_rate`

The shipped rates assume E2E ≈ 15 s. That is a placeholder. After the first run,
read the achieved mean E2E off the summary and re-solve:

```
new_session_rate  =  N_target / (E[turns] x (measured_E2E + E[delay]))
```

using `E[turns]` and `E[delay]` from the table below. Then raise
`initial_sessions` to the same `N_target` so the run starts at equilibrium
instead of drifting toward it through the whole warmup.

If you skip this step the population silently lands somewhere other than your
target, and every throughput number is for a load level you did not choose.

## Case A vs Case B

Sampled from the shipped triples (400K draws, seed 1337). These are the numbers
to plug into Little's law — note how far the means sit from the p50s, which is
the whole reason these configs need percentile calibration.

| | Case A | Case B |
|---|---|---|
| `E[input]` (after clamp) | 86,023 tok | 95,138 tok |
| `E[output]` | 1,433 tok | 622 tok |
| `E[turns]` | 9.50 (p50 is 3) | 20.98 (p50 is 5) |
| `E[inter_turn_delay]` | 18.0 s | 17.5 s |
| turns p99 | 103 | 144 |
| input clamp | 260K | **520K** |
| cache-hit target | 0.89 | 0.88 |
| mean session @ E2E=15 s | 313 s | 683 s |
| p99 session @ E2E=15 s | 3,397 s | 4,686 s |
| `new_session_rate` for N=32 @ E2E=15 s | 0.102 /s | 0.047 /s |
| resulting offered rate | 0.97 req/s | 0.98 req/s |
| resulting uncached TPM / GPU (8 GPUs) | ~69K | ~84K |

Case B is not "Case A with different numbers." Its sessions live 2.2x longer, so
the same population needs less than half the birth rate — copying Case A's
`0.10` into Case B targets N=68, not 32. Its output is a third as long, making it
far more prefill-dominated. And its 520K input clamp is a hard requirement on the
engine that Case A never exercises.

## Pitfalls

**QPS knobs do nothing.** `initial_qps`, `max_qps`, and any QPS-shaped CLI flag
are read and ignored in this mode. The driver now prints a note saying so at
startup. This extends to tooling: `runner.py`'s sweep and `--auto-search` work by
rewriting `max_qps` / `initial_qps` in a temp config, so **a QPS sweep over these
workloads runs the identical load at every point** and `--auto-search` bisects on
a variable with no effect. To sweep capacity, sweep the
`initial_sessions` + `new_session_rate` pair instead.

**`ramp_duration` is a warmup exclusion window, not a ramp.** Nothing ramps
closed-loop. It exists so the synchronized t=0 session cohort dies off and the
large shared prefix becomes resident before measurement starts. Setting it to 0
puts cold-cache requests into your sustain statistics.

**The original `sustain_duration: 600` truncates the tail you calibrated.** A p99
session is ~3,400 s in Case A and ~4,700 s in Case B. Measuring for 600 s cuts
off the long sessions that `turns_per_session` exists to reproduce, so the run
under-represents exactly the behavior the spec cares about. The corrected files
use 3,600 s (A) and 4,800 s (B); total run time including warmup is ~67 min and
~92 min. There is no shorter honest window.

**Hitting `max_sessions` is a failure signal, not a cap to tune.** It means
sessions are dying slower than they are born — the server is saturated and the
population is running away. Raising the cap does not fix it; it just lets the
runaway continue. Same for `max_inflight`: it is now genuinely enforced, so the
value of `32` in the original files will bind and throttle the workload, at which
point backpressure rather than your config decides the load. Set it to the
server's `--max-running-requests` if you want to model a real admission limit,
otherwise leave the headroom the corrected files use.

**Case B needs a 520K context engine.** `max_input_tokens: 520000` is a
requirement, not a hint. If the server's `--max-model-len` is below it, the p99
tail errors out and the input distribution you measure is not the specified one.
About 0.9% of Case B requests land on that clamp.

**The cache-hit model does not test cache tiering.** Every request nests inside
the *same* shared prefix, so the radix tree holds one hot path and there is no
eviction pressure. `cache_hit_rate` here verifies cache accounting and gives
realistic prefill savings; it does not exercise HiCache / NVMe offload. For that
you want a growing-prefix workload such as `code_agent_200k.yaml`.

**A placeholder tokenizer path aborts the run.** Both files ship with
`/path/to/GLM-5.2-MXFP4`. Replace it in the YAML or override with `--tokenizer`.

**The `sla:` block is documentation only.** It is parsed and never consumed.
`runner.py` gates on `sustain_ttft_p90_ms` and success rate taken from its own
`--slo-ttft-p90-ms` / `--slo-success-rate` flags. If you want a real gate, pass
those flags; the `e2e_p50_ms` and `tps_min` targets in the YAML are not checked
by anything.

**Short runs show noisy percentiles.** At ~1 req/s, a few minutes is a few
hundred requests, far too few to resolve a p99. Judge percentiles only over a
full `sustain_duration`.

## Reading the output

Artifacts land in `benchmarks/<name>/<timestamp>/`. `summary.json` carries the
phase breakdown; the **sustain** row is your result — ramp is warmup and drain is
the post-window tail. Sanity-check a run before trusting it:

- live session count should sit near your `N_target` and be flat, not climbing
- the observed cache-hit rate should be close to `cache_hit_rate`
- `max_inflight` should not have triggered its throttling warning
- mean E2E should be near whatever you assumed when solving for the rate; if it
  is not, re-solve and rerun

Note that reported QPS in this mode is an emergent measurement, not a target, and
the driver labels it that way.
