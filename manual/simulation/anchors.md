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

## Harvest flag reference

`inferasim anchor` forwards its flags straight to the harness, so
`inferasim anchor --help` is authoritative. The groups below are what the flags
are *for*; most harvests only touch the first two.

### What to measure

| Flag | Default | Meaning |
|---|---|---|
| `--model` | *(required)* | HF id or local path of the checkpoint to serve. |
| `--serving-backend {vllm,sglang,atom}` | `vllm` | Which engine's kernels the anchor describes. Two engines serving one config are two measurements and never share a cache entry. Ignored under `--offline`, which is vLLM-only. |
| `--tp` / `--pp` | `1` / `1` | **Target** parallel shape to project to — not necessarily the shape that runs. |
| `--benchmark-gpus` | *(all of `tp*pp`)* | GPUs the run may actually use. Below `tp*pp`, parallelism is reduced in `pp → ep → tp` order to fit and the projector restores the target. `--tp 8 --benchmark-gpus 1` measures TP=1 and projects TP=8. |
| `--save` | *(required)* | Where to write the anchor JSON. |

### Shape of the measurement

`--concurrency` is the usual way to set the batch axis: it derives the sweep
from the engine's own CUDA-graph capture sizes, which is the ladder decode is
later looked up against. Name batches explicitly only when you want to override
that.

| Flag | Default | Meaning |
|---|---|---|
| `--concurrency` | *(unset)* | Sweep the capture ladder up to this concurrency. Overrides `--batch`/`--batches`. |
| `--batch` | `16` | Single reference batch when no sweep is requested. |
| `--batches` | *(unset)* | Explicit comma list, e.g. `4,8,16,32,64`. |
| `--input-len` | `1024` | Prompt length the step is measured at. |
| `--output-len` | `1024` | Recorded in `meta` only; decode length comes from `--decode-steps`. |
| `--decode-steps` | `32` | K in the K-token minus 1-token difference that isolates the steady-state decode step. |
| `--decode-context-grid` | `input_len × {1,2,4}` | Context lengths to time decode at, so the projector fits the attention KV term instead of assuming decode is flat in context. Offline capture mode. |
| `--max-model-len` | *(from config)* | Engine context limit. |
| `--gpu-mem-util` | `0.9` | vLLM `gpu_memory_utilization`. |

### Reduce, measure, restore

Depth reduction is the counterpart to parallelism reduction: build a shallow
model, fit step latency against layer count, and evaluate the fit at the real
depth.

| Flag | Default | Meaning |
|---|---|---|
| `--bench-layers` | *(unset)* | Comma list of **reduced** layer counts to measure and restore from, e.g. `4,8`. |
| `--full-layers` | *(HF config)* | Depth to restore to when using `--bench-layers`. |
| `--num-hidden-layers` | *(unset)* | Legacy single sub-scale run with **no** restore. Prefer `--bench-layers`. |

### Anchoring prefill

Decode escapes the analytical roofline by being measured. Prefill only escapes
it if you measure prefill too, which is why the probe is on by default on the
served path.

| Flag | Default | Meaning |
|---|---|---|
| `--prefill-anchor` / `--no-prefill-anchor` | on (served) | Difference mean TTFT across two prompt lengths at concurrency 1 to price prefill from measurement instead of the roofline. Turning it off saves the probe runs and pays the roofline bias. |
| `--prefill-anchor-short` | `input_len / 2` | Short probe length. The long probe is always `--input-len`, so the fitted rate covers the lengths the anchor is used at. |
| `--prefill-anchor-points` | `0` (two points) | Probe this many lengths and fit `fixed + per-token + per-token²`. Two points can only draw a chord, so curvature in prompt length lands inside the intercept — which is then carried across TP as if it were fixed cost. |
| `--prefill-packed-points` | `4` | Also probe this many simultaneous sequences at fixed length, so a step that packs many sequences is measured rather than inferred. The length probe cannot supply this: at concurrency 1, token count and attention context are the same number. |
| `--prefill-anchor-validate` | off | Probe a third, interior length so pairwise slopes can be compared — a linearity check, at the cost of one more client run. |

### MoE routing and expert placement

Expert-load imbalance changes which rank is busiest, so a dummy-weight run needs
a routing distribution imposed on it. With real weights the trained router
supplies one and `--routing-dist none` is the constant-free choice.

| Flag | Default | Meaning |
|---|---|---|
| `--enable-expert-parallel` | off | Shard experts across ranks (EP=TP) instead of tensor-slicing each expert. Exposes busiest-rank and all-to-all effects. |
| `--routing-dist {zipf,uniform,normal,none}` | `zipf` | Token→expert distribution for the benchmark; `none` uses the model's own router. |
| `--zipf-s` | `1.0` | Zipf skew exponent — `0` is uniform, larger is more skewed. |
| `--moe-imbalance` | *(unset)* | Target imbalance `I = max/mean` tokens per expert; solves for the Zipf exponent at the model's expert count and overrides `--zipf-s`. Random data sits at low `I`, domain-clustered traffic higher. |
| `--load-format` | `dummy` | `dummy` for random weights (needs an imposed distribution), `auto`/`safetensors` for real ones. |

### Regime-defining engine settings

Everything here changes which kernels run, so it changes the regime the anchor
certifies. An anchor measured without speculation cannot be transported to a
target that uses it.

| Flag | Default | Meaning |
|---|---|---|
| `--quantization` | *(from config)* | e.g. `fp8`, `mxfp4`. |
| `--kv-cache-dtype` | auto | e.g. `fp8`. |
| `--speculative-method` | *(unset)* | e.g. `deepseek_mtp` (NextN head) or `ngram`. Changes how many tokens a step emits. |
| `--speculative-num-tokens` | *(unset)* | Draft tokens per step, bounded by the checkpoint's `num_nextn_predict_layers` for `deepseek_mtp`. |
| `--speculative-draft-model` | *(unset)* | Draft checkpoint for methods that need a separate one. |
| `--enforce-eager` | off | Disable graph capture — which removes the pad-up staircase decode is normally looked up against. |
| `--no-aiter` | off | Disable AITER kernels (enabled by default on ROCm). |
| `--prefix-caching` | off | Enable vLLM prefix caching. In the offline repeated-prompt sweep this measures near-100%-hit lookup latency, not cold prefill. |
| `--server-args` | `""` | Engine flags as one string, e.g. `'--max-num-seqs 512 --enable-chunked-prefill'`. Parsed by vLLM's own parser, so a flag means what it means to a real server — this is what lets a reduced-scale run screen a serving variant. |
| `--env KEY=VAL` | *(none)* | Repeatable environment override applied before vLLM is imported, for levers the engine reads from the environment rather than a flag. |
| `--trust-remote-code` | off | Required by remote-code architectures. |
| `--skip-tokenizer-init` | auto with `dummy` | Benchmark drives token ids directly. |

### Sampling, determinism, and caching

| Flag | Default | Meaning |
|---|---|---|
| `--seed` | `0` | Single RNG seed for random token content. |
| `--seeds` | `0,1,2` | Comma list swept inside **one** engine build. Each seed re-rolls token content and adds an independent sample, so the artifact carries a per-batch mean and standard deviation instead of a point with no error bar. |
| `--random-tokens` | auto with real weights | Independent random token ids per sequence. |
| `--vocab` | `30000` | Upper bound for random token ids. |
| `--cache-dir` | `$INFERASIM_BENCH_CACHE` | Cache keyed by run config; a hit never builds the engine, which is nearly all the wall time. Caching is off when neither is set. |
| `--no-cache` / `--force` | off | Ignore the cache / re-run and overwrite the entry. |
| `--offline` | off | Measure through the offline `LLM()` entrypoint instead of a real server. Off by default because the two do not resolve the same kernels, so an offline anchor can mispredict a served target badly. |

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
