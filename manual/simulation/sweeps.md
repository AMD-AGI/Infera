# Sweeps and tuning

```{admonition} One-pager
:class: tip
**What:** two ways to search the configuration space instead of projecting one
point at a time — a scripted grid, or an LLM-driven search. **Why:** the whole
reason to make a projection cheap is to run a lot of them. **Cost:** zero GPUs
on both paths.
```

## Sweep the configuration space

From Python, for scripted searches:

```python
from infera.projection.core.projection.inference_projection.sweep import sweep

res = sweep(
    "gpt_oss_120B",
    tp=[1, 2, 4, 8], ep=[1, 2, 4, 8], pp=[1],
    concurrency=[1, 8, 32, 128],
    isl=1024, osl=1024,
    gpu_arch="mi355x", hbm_gb=288.0,
    valid=lambda tp, ep, pp: ep <= tp,      # your own legality rules
)

for p in res.points:
    if p.feasible:
        print(p.tp, p.ep, p.concurrency, round(p.ttft_ms, 1),
              round(p.tpot_ms, 2), round(p.decode_tps_per_gpu, 1))
```

Sweeps force `--profiling-mode simulate`, so they need **zero GPUs** — a search
that measured every point would not be a search. This is the one place the
uncalibrated path is the default rather than the opt-in, and it is why a sweep
produces a *ranking to confirm* rather than numbers to quote. Rank in the sweep,
then re-run the top few against an [anchor](anchors.md).

**Infeasible points are kept**, annotated with `p.reason`, rather than dropped.
This is deliberate: a missing point is ambiguous between "did not fit" and "was
never tried", and those call for different next actions. Pass `workload=` to
project against your own experiment config instead of the packaged default.

Rank on `decode_tps_per_gpu` rather than aggregate throughput unless every point
in the sweep has the same GPU count — otherwise the sweep just rediscovers that
more GPUs are faster.

## Search with the tuning agent

When you would rather describe the space than enumerate it:

```bash
inferasim-tune --inference --workload <workload.yaml> --target-cluster <cluster.yaml>
```

Two stages: a deterministic seed sweep for a warm start, then an LLM-driven
search that continues from the incumbent, proposing recipes and scoring them
through the projector. It searches 36 serving levers against any of 24 projected
metrics, and — the part that makes it worth using over a grid — it enforces
latency SLOs as hard constraints, so `max_throughput` returns the fastest config
that still keeps the promise instead of the largest batch that fits.

Note the `--inference` flag: the agent tunes *training* configurations by
default.

Full documentation, including the objective list and the SLO block, is on
[Tuning agent](tuning_agent.md).

## Then spend the GPU hours

A sweep's output is a shortlist, not a decision. The intended loop is to search
in simulation, confirm the top few candidates on hardware with the
[benchmarking path](../reference/benchmarking.md), and — if the confirmation
disagrees — [harvest an anchor](anchors.md) in that regime so the next search
starts from a calibrated kernel.

## Next steps

- Ground the scorer in hardware: [Anchors and calibration](anchors.md).
- Add tails to the shortlist: [Simulation runs](simulation_runs.md).
