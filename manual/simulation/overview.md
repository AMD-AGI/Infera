# Simulation overview

```{admonition} One-pager
:class: tip
**What:** InferaSim answers "how would this serving configuration behave?"
without standing up the configuration. **Why:** so you can screen thousands of
candidate deployments and spend GPU hours only on the shortlist.
**Cost:** one measured anchor per execution regime. There is also a path that
needs no GPU, no engine and no cluster at all, but it is not the default — see
[Fidelity sources](#fidelity-sources).
```

**InferaSim** is Infera's serving simulator. It projects time-to-first-token,
inter-token latency, throughput and KV-cache footprint for a serving recipe, and
it simulates a fleet of engines under arrival-driven load. The governing idea is
**measure sparsely, transport analytically**: benchmark one cheap sub-scale
anchor on a real GPU, then project every other recipe — parallel shape, batch,
concurrency, context length — from that one measurement instead of re-measuring
each one.

## Use InferaSim when you want to answer questions such as

- **What parallel shape should I serve this model with?** Is TP=8 worth double
  the GPUs over TP=4 for this input/output mix, on a throughput-per-GPU basis?
- **How many replicas do I need for a given arrival rate and latency SLO?** And
  at what offered rate does the p99 fall off a cliff?
- **Will this fit?** What is the maximum concurrency an HBM budget supports once
  weights, KV cache and the activation working set are accounted for?
- **What will my real trace do to cache hit rate?** Given a recorded workload,
  how much prefix reuse does each routing policy actually realise, and what does
  overconcentration cost in tail latency?
- **Is disaggregation worth it here?** How large is the continuous-batching
  interference tax I would be buying my way out of?
- **What does this cost?** The same ranking in dollars per million tokens rather
  than tokens per second.

## Components

| Component | Entry point | Role |
|---|---|---|
| **Analytical projector** | `inferasim inference` | Closed-form steady-state **means**: TTFT, ITL/TPOT, throughput, memory, feasibility. Runs on a calibrated cost kernel by default; `--profiling-mode simulate` runs it uncalibrated. |
| **Discrete-event simulator (DES)** | `inferasim inference --arrival-model poisson` (or `--des-mooncake-trace`) | Event-driven scheduler, queue and fleet model producing **distributions** — p50/p90/p99, queue wait, saturation, per-replica cache behaviour. |
| **Anchor harvester** | `inferasim anchor` | Runs the real serving engine once on real hardware and saves a calibration artifact the two engines above both honour. |
| **Benchmark mode** | `inferasim inference --profiling-mode benchmark` | The same harvest, driven from the projection config, so measurement and projection happen in one step. |
| **Sweep API** | `from infera.projection.core.projection.inference_projection.sweep import sweep` | Scripted grid search over recipes, with your own legality predicate. Zero GPUs. |
| **Tuning agent** | `inferasim-tune --inference` | A deterministic seed sweep followed by an LLM-driven search over 36 serving levers, optimising any of 24 projected metrics under hard latency SLOs. |

Two console scripts are installed by `pip install ".[projection]"`: `inferasim`
(projection and simulation) and `inferasim-tune` (recipe search).
`infera-projection` and `infera-tuning` remain as aliases, and everything is
also reachable as `python -m infera.projection.cli`.

## Workflow

```{graphviz}
digraph inferasim_flow {
  rankdir=LR; bgcolor="transparent"; compound=true;
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc" fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.22,0.13"];
  edge [fontname="Helvetica,Arial,sans-serif" fontsize=10 color="#5577cc"];

  SPEC [label="Serving spec\nmodel · hardware\nrecipe · workload" fillcolor="#f4f4f4" color="#999999"];
  ANCHOR [label="Anchor (optional)\none measured run\non real hardware" fillcolor="#fff3cd" color="#caa300"];
  KERNEL [label="Cost kernel\nshared by both engines"];

  subgraph cluster_engines {
    label="Two engines, one cost kernel"; labelloc=b; fontsize=10; fontname="Helvetica,Arial,sans-serif";
    color="#cccccc"; style=dashed;
    PROJ [label="Analytical projector\nsteady state"];
    DES  [label="Discrete-event simulator\narrivals · queues · fleet"];
  }

  MEANS [label="Means\nTTFT · TPOT · throughput\nmemory · feasibility" fillcolor="#e8f0fb" color="#5577cc"];
  DIST  [label="Distributions\np50 / p90 / p99 · queue wait\nsaturation · cache hit rate" fillcolor="#e6f5ec" color="#2fa36b"];
  DECIDE [label="Shortlist" fillcolor="#f4f4f4" color="#999999"];
  HW [label="Confirm on hardware" fillcolor="#f4f4f4" color="#999999"];

  SPEC -> KERNEL;
  ANCHOR -> KERNEL [label="calibrates" style=dashed];
  KERNEL -> PROJ;
  KERNEL -> DES;
  PROJ -> MEANS;
  DES -> DIST;
  MEANS -> DECIDE;
  DIST -> DECIDE;
  DECIDE -> HW [label="spend GPU hours here" penwidth=1.9];
}
```

Four things describe a run: the **model** (an architecture preset), the
**hardware** (GPU architecture and HBM budget), the **recipe** (parallel shape,
dtypes, concurrency — the thing you are searching over) and the **workload**
(input/output lengths, arrival pattern, prefix reuse).

The two engines answer different questions but share one cost kernel, so an
anchor loaded for one is honoured by the other. When you ask for a simulation,
the DES report is printed *in addition to* the analytical one rather than
replacing it.

## How the tools differ

| Tool | Function | What it does not do |
|---|---|---|
| **Analytical projector** | Closed-form means at a fixed concurrency, plus the memory and feasibility verdict. Milliseconds per point. | No queueing, no arrival process, no percentiles. It cannot tell you a p99, and its TTFT is prefill *service* time rather than the response time a client sees. |
| **Discrete-event simulator** | Runs a scheduler step loop over an arrival stream or a trace: admission, batch packing, a content-addressed KV block cache per replica, and routing across replicas. | Models one scheduler policy (unified batch), one device-resident cache tier, and no autoscaler or worker-startup delay. Disaggregation is not event-driven here. |
| **Anchor harvest / benchmark mode** | Measures the real engine, as deployed, across a batch ladder — so the projector's latency is grounded in a kernel and graph regime that actually ran. | Calibrates **latency only**. Every memory number stays analytical. An anchor certifies its own regime: change dtype, kernel backend or graph mode and it no longer transfers. |
| **Sweep API** | Exhaustive scripted grids over recipes, keeping infeasible points annotated with a reason rather than dropping them. | Forces `--profiling-mode simulate`, so it never measures. It ranks recipes, not kernel or scheduler flags. |
| **Tuning agent** | Navigates the space instead of gridding it, optimising a chosen metric *subject to* latency SLOs it treats as hard constraints. | It optimises against the projector, so its answer inherits the projector's boundaries. Its `memory-real` mode ranks on a coarse heuristic, not a projection. |
| **Real serving benchmark** ([Benchmarking](../reference/benchmarking.md)) | Ground truth on real hardware, including everything the simulator declines to model. | Costs GPU hours per point, which is the reason InferaSim exists. |

The last row is not a competitor. The intended loop is: search in simulation,
then confirm the shortlist on hardware. See
[Boundaries and verification](boundaries.md).

## Fidelity sources

One flag selects where latency comes from:

| `--profiling-mode` | Needs a GPU | Meaning |
|---|---|---|
| `benchmark` | yes, unless an anchor already matches | **The default.** Measure on real hardware, then project from that measurement. |
| `simulate` | no | Analytical kernel models, uncalibrated. Opt-in, and forced for sweeps. |
| `both` | yes | Run each and report them side by side. |

**Measuring is the default on purpose.** Correlation against real serving has
been established for the calibrated path; the purely analytical kernel models
have not earned the same claim. A default is a claim, so the mode you get for
saying nothing is the one whose numbers are defensible, and the uncalibrated
projection is something you ask for.

`benchmark` measures by serving the model for real, so it also needs a serving
engine and `--bench-model` — a structural config names an architecture, not a
checkpoint. It never silently degrades to `simulate`: if it cannot measure, it
says so and stops, because a number quietly downgraded to analytical would be
read as a measured one.

```{admonition} The default does not mean every run needs a GPU
:class: note
A matching anchor in `--anchor-store` (or `INFERASIM_ANCHOR_STORE`) satisfies
the default without touching hardware — the run loads it and projects. So the
usual working pattern is *one* measured harvest per regime, followed by any
number of GPU-free projections. On a host with neither an anchor nor an
accelerator the run stops and tells you the three ways forward:
`--load-benchmark`, `--anchor-store`, or `--profiling-mode simulate`.
```

The no-GPU path is fully supported and is what sweeps run on — it is simply not
what you get by default.

## Where anchors fit

The **anchor** is what makes a measured default affordable: a saved artifact
from one cheap measured run that calibrates the analytical path afterwards, with
no GPU in play.

```bash
# once, on any ROCm host with a serving engine
inferasim anchor --model openai/gpt-oss-120b --benchmark-gpus 1 --save anchor.json

# then, anywhere
inferasim inference ... --load-benchmark anchor.json
```

This is the whole economic argument for the tool. Measuring is by far the
expensive part, and one anchor is reused across every projection that shares its
execution regime. See [Anchors and calibration](anchors.md) for what defines a
regime and how many GPUs a harvest should use.

## Choosing an entry point

| Goal | Start here |
|---|---|
| Project one recipe and read the report | [Projection runs](projection_runs.md) |
| Get p90/p99 latency under an offered load | [Simulation runs](simulation_runs.md) |
| Reproduce a fixed-concurrency benchmark harness | [Simulation runs](simulation_runs.md#reproduce-a-fixed-concurrency-benchmark) |
| Ground the numbers in real hardware | [Anchors and calibration](anchors.md) |
| Run with no GPU, engine or cluster at all | [Fidelity sources](#fidelity-sources) |
| Drive the simulation from a recorded trace | [Workloads and traces](workloads.md) |
| Study cache reuse and routing across replicas | [Fleet and routing](fleet_and_routing.md) |
| Size prefill and decode pools separately | [Disaggregated serving](disaggregation.md) |
| Grid a space you can enumerate | [Sweeps and tuning](sweeps.md) |
| Find the best recipe under a latency SLO | [Tuning agent](tuning_agent.md) |
| Check whether a feature is modeled at all | [Modeled features](modeled_features.md) |
| Size HBM and KV rather than time anything | [The memory projection](projection_runs.md#the-memory-projection) |
| Find out what the tool will not tell you | [Boundaries and verification](boundaries.md) |
