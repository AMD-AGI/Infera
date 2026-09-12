# Anchors and calibration

```{admonition} One-pager
:class: tip
**What:** an anchor is a saved artifact from one measured run that grounds every
later projection in real hardware. **Why:** measuring is the expensive part, so
you measure once per *execution regime* and transport analytically from there.
**Cost:** one short serving run on one ROCm host, reused indefinitely.
```

Calibration is what makes a projection trustworthy, which is why
[measuring is the default](overview.md#fidelity-sources). The mechanism that
keeps that affordable is the **anchor**: a measured benchmark artifact indexed
by a regime signature, which the analytical projector and the DES both honour.

Harvest one per regime and every later projection in that regime is GPU-free
while still being calibrated. This is the intended steady state — not the
uncalibrated `simulate` path, and not a GPU per question.

## Harvest an anchor

Harvest once on any ROCm host with a serving engine. `--model` is the checkpoint
to serve, so it is a Hugging Face id or a local path rather than a preset name:

```bash
inferasim anchor --model openai/gpt-oss-120b --benchmark-gpus 1 --save anchor.json
```

`--serving-backend {vllm,sglang,atom}` picks the engine, which is launched
through the same adapters the platform serves with — so the anchor describes the
engine as deployed, rather than as a benchmark harness happened to start it. The
anchor covers one engine: routing across replicas is simulated, not measured, so
measuring through the router would count that layer twice.

Then project any recipe from it, with no GPU:

```bash
inferasim inference ... --load-benchmark anchor.json
```

### Measuring and projecting in one step

To skip saving an anchor first, ask the projection itself to measure. This is the
same harness, driven from the config:

```bash
inferasim inference ... --profiling-mode benchmark \
  --bench-model openai/gpt-oss-120b --save-benchmark anchor.json
```

Prefer the two-step form when more than one recipe is in play: measuring is by
far the expensive part, and `--load-benchmark` reuses one measurement across
every projection that shares its regime.

## What one run actually measures

One run sweeps more than the point it was asked for. It covers the CUDA-graph
capture ladder up to `--max-concurrency`, because batch is an axis projections
transport along and a lone measured point would be held flat across it. Decode
is then looked up by padding a batch **up** to the nearest measured size, the
way the engine pads it up to the nearest captured one. Decode latency is a
staircase, not a curve, and this is why it is looked up rather than interpolated.

The served anchor takes that ladder from vLLM's default shape, since the engine
runs in another process and its real capture list cannot be read from here — a
default launch captures the default ladder. The offline anchor (`--offline`)
reads the list off the built engine instead, and also measures decode against
context so the projector can fit the attention KV term rather than assume decode
is flat in context. Naming batches explicitly with `--batches` overrides the
ladder on either path.

```{admonition} Measurement calibrates latency only
:class: important
The memory projection is analytical throughout — weights, KV cache and the
activation working set are computed from the model shape and the parallel layout.
`--profiling-mode benchmark` does not change a single memory number.
```

The anchor JSON is engine-neutral: a `"backend"` field plus per-batch
decode/prefill measurements. A different harvester can be added without touching
the projector.

## Regimes: when an anchor stops transferring

Recipe parameters split into two kinds.

| Kind | Parameters | Behaviour |
|---|---|---|
| **Regime-defining** | dtype, kernel/attention backend, graph mode, speculative decoding, the model itself | Swap the kernel or execution path. Two recipes differing on any of them are **not** transportable from one another — each regime needs its own anchor. |
| **Transportable** | parallel shape (TP/EP/PP), batch, concurrency, sequence and context length, layer count | Move analytically from an existing measurement. |

Reconstruction picks the nearest in-regime anchor and transports it to the target
recipe by driving the same projector, so the physics lives in one place.

### The anchor store

Anchors live in a directory indexed by regime signature, named by
`--anchor-store` or `INFERASIM_ANCHOR_STORE`. Point a run at one and it looks
for a matching anchor itself, which is what lets the measured default be
satisfied without a GPU. The lookup is conservative and says what it did: an
anchor at a different dtype, attention backend, cudagraph mode or speculation
setting describes different kernels, so it is **reported and refused** rather
than quietly applied.

Setting this once per host is the difference between "measuring is the default"
meaning *one* harvest and it meaning one harvest per question.

If anchor-calibrated results look wrong after you changed dtype or backend, this
is why: an anchor certifies its own execution regime. Harvest another.

## How many GPUs a harvest should use

A 1-GPU anchor cannot observe cross-GPU communication — TP all-reduce, EP
all-to-all — which is exactly the cost that matters at scale. So a warmup wants
more than one GPU, but only up to a point. The rule is a single line:
`min(tp, 4)`, stepped down to a degree that divides `tp`. There is no sweep and
no rung to climb; one anchor is measured and the projector restores every target
from it.

Four is the **operational default, not a universal accuracy guarantee**. It is
large enough to observe multi-rank collectives, remains only one doubling from an
eight-GPU target, and avoids making every warmup wait on a full node. Targets
more than one doubling from their anchor are marked **extrapolated**. Use
`--load-benchmark-scaling` or a target-width confirmation when the parallelism
hop, the topology, or the risk attached to the decision warrants another
measurement.

## Next steps

- Project from the anchor: [Projection runs](projection_runs.md).
- Simulate load with the same calibration: [Simulation runs](simulation_runs.md).
- Understand what calibration cannot fix:
  [Boundaries and verification](boundaries.md).
