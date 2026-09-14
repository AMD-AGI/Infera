# serve_profiled

Restart GLM-5.3-Flash with decode CUDA graphs **off** and the torch-profiler
control plane **on**.

Same script as `serve_baseline` with two flags flipped. Keeping the two rounds on
one implementation is what stops them drifting into two deployments that differ
in more than the axis under test.

## Why the restart is unavoidable

CUDA graphs are a start-up flag; there is no way to toggle them on a live engine.
And the two rounds want opposite settings:

- Graphs **on** is what makes throughput worth quoting — measured 7× at
  concurrency 1 on this model.
- Graphs **off** is what makes a trace attributable. With graphs on the profiler
  records one launch per decode step instead of the kernels inside it, so the
  ranking downstream would be a single opaque row.

One deployment cannot be both, so there are two.

## Why it depends on run_baseline

Its first act is `mix_up.sh`'s idempotent teardown, which destroys the baseline
deployment. The edge therefore has to mean "the baseline has been measured", not
"the baseline exists" — wiring it to `serve_baseline` would let agent_sys
schedule it alongside `run_baseline` and tear the deployment down mid-measurement.

## It does not honour PD_REUSE_DEPLOYMENT

That variable is a development aid for iterating on handoff shape without paying
a cold start each time. Here it would skip the bring-up that is the entire
content of this task and hand the trace round a graphs-on engine. The entry
script sets it to 0 unconditionally.

The safety net if it were ever bypassed: `round.sh` records the engine's observed
argv and the router's actual invocation, and `check_service_live` decides the
round from those rather than from anything this task declares about itself.
