# Tuning agent — `inferasim-tune`

```{admonition} One-pager
:class: tip
**What:** an automated search that proposes serving recipes, scores them through
the projector, and returns the best one *subject to your latency budget*.
**Why:** the serving space is too large and too awkward to grid, and the
interesting answer is almost always constrained rather than maximal.
**Cost:** no GPU by default. It optimises against the projector, so it inherits
the projector's [boundaries](boundaries.md).
```

The [sweep API](sweeps.md) grids a space you enumerate. The tuning agent
navigates a space you describe, over **36 serving levers**, against **any of 24
projected metrics**, under latency constraints it will not violate.

## Run it

```bash
inferasim-tune --inference \
  --workload <workload.yaml> \
  --target-cluster infera/projection/examples/tuning/target_cluster_mi355x_inference.yaml \
  --out-dir runs/tune-1
```

```{admonition} --inference is not optional
:class: warning
The agent tunes **training** configurations by default. Serving is a different
search surface, selected either by the `--inference` flag or by
`optimization.mode: inference` in the target-cluster YAML. The packaged
`target_cluster_mi355x_inference.yaml` sets that key, so `--inference` is
redundant with it — but pass a training-mode cluster file without the flag and
you will get a training search that never mentions TTFT.
```

## Two stages

**1. Deterministic seed sweep.** A systematic warm start over the serving space,
ordered by expected impact: TP (latency) → batching and concurrency
(throughput) → KV quantization (capacity) → weight quantization → combined →
chunked prefill → speculative decoding → EP (for MoE). This stage is
reproducible and needs no LLM. `--seed-budget` caps it (default `12`).

**2. LLM-driven search.** Continues from the warm-started incumbent, proposing
recipes, reading back the result of each, and keeping notes across rounds. It
sees the legal axes, the architecture, the cluster, the objective and its
direction, the active latency budgets, and the trial history.

`--seed-only` (alias `--no-agent`) runs stage 1 alone, which is the right way to
get a reproducible baseline. `--agent-only` skips the seeds and runs the LLM
against an existing history.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--workload` | *required* | the workload YAML |
| `--target-cluster` | *required* | the cluster and optimization YAML |
| `--inference` | off | tune serving rather than training; overrides `optimization.mode` |
| `--out-dir` | `./tuning_runs/<target_cluster.name>` | where trials, summary and scratchpad are written |
| `--mode {dry,memory-real,full}` | `full` | evaluator depth — see [Evaluation modes](#evaluation-modes) |
| `--profiling-mode {simulate,benchmark}` | `simulate` | how `full` mode scores a candidate |
| `--dry-run` | off | shorthand for `--mode dry` |
| `--seed-only` / `--no-agent` | off | deterministic stage only, no LLM |
| `--agent-only` | off | skip seeds, search from existing history |
| `--resume` | off | reuse an existing trials file in `--out-dir`. On the serving path this is a no-op: an existing `inference_trials.jsonl` is always loaded and deduplicated, so a re-run continues rather than repeating. |
| `--seed-budget` | `12` | seed candidates before the LLM takes over |

## Objectives

Set `optimization.objective` in the target-cluster YAML. The default is
`decode_throughput_tps_per_gpu`. Friendly aliases are accepted —
`max_throughput`, `min_ttft`, `min_latency`, `min_itl`, `tpot`,
`max_concurrency`, `tput_per_gpu` and others resolve to the canonical names
below.

This list is the direct answer to "what can it predict?" — anything here can be
the thing being optimised, and everything here is reported for every trial.

**Throughput**

| Objective | Meaning |
|---|---|
| `total_throughput_tps_per_gpu` | total tokens/s/GPU, prompt plus generation — the headline ranking |
| `total_throughput_tps` | the same, fleet total |
| `decode_throughput_tps_per_gpu` | generation tokens/s/GPU (**the default**) |
| `decode_throughput_tps` | generation tokens/s, fleet |
| `prefill_throughput_tps_per_gpu` | prompt tokens/s/GPU |
| `prefill_throughput_tps` | prompt tokens/s, fleet |

Total and decode-only throughput are **different orderings**. At a 144:1
prompt-to-generation ratio the prompt decides the winner, so a prefill-heavy
agentic fleet should not be ranked on decode throughput.

**Latency and interactivity**

| Objective | Meaning |
|---|---|
| `ttft_ms` | mean time to first token *(minimize)* |
| `itl_ms` | mean inter-token latency / TPOT *(minimize)* |
| `request_latency_ms` | mean end-to-end request latency *(minimize)* |
| `interactivity_tok_s_per_user` | per-user generation rate |
| `per_request_decode_tps` | per-request generation throughput |
| `decode_step_ms_pure` | uncontended decode step time *(minimize)* |

**Capacity and memory**

| Objective | Meaning |
|---|---|
| `max_concurrent_sequences` | sequences the KV pool holds |
| `max_sustainable_concurrency` | concurrency the pool sustains |
| `memory_per_gpu_gb` | HBM footprint *(minimize)* |
| `kv_cache_gb` | KV footprint *(minimize)* |
| `weights_gb` | resident weight footprint *(minimize)* |
| `activation_gb` | activation working set *(minimize)* |

**Interference and communication**

| Objective | Meaning |
|---|---|
| `mixed_step_fraction_pct` | prefill interference in decode *(minimize)* |
| `tpot_pollution_pct` | TPOT inflation from that interference *(minimize)* |
| `decode_step_ms_mixed` | step time when a prefill chunk lands *(minimize)* |
| `prefill_comm_ms` / `decode_comm_ms` | collective time per phase *(minimize)* |
| `replica_gpus` | GPUs a replica costs *(minimize)* |

Metrics marked *(minimize)* are negated internally, so scoring is always
"higher is better" regardless of direction.

```{admonition} Deliberately unsupported
:class: note
`avg_power_w`, `joules_per_output_token`, `joules_per_total_token`, `mfu`,
`tflops_per_s_per_gpu` and `iteration_ms` are rejected rather than accepted and
scored as absent. The serving projection reports none of them, and an objective
that silently scores `None` reads as a failed search rather than as a missing
metric.
```

## Latency SLOs — the constraint that makes it useful

Set budgets in milliseconds under `optimization.slo`. A trial that misses any of
them is rejected exactly like an over-memory one.

```yaml
optimization:
  objective: max_throughput
  slo:
    ttft_ms: 500
    tpot_ms: 25
    request_latency_ms: null   # null leaves it unconstrained
```

| SLO key | Constrains |
|---|---|
| `ttft_ms` | time to first token |
| `tpot_ms` / `itl_ms` | per-token latency |
| `request_latency_ms` | end-to-end request latency |

This is the difference between a search that is useful and one that is not.
**Without an SLO, `max_throughput` always walks to the largest batch that fits**,
and the interactive configuration never appears in the results at all. With one,
the search becomes the constrained problem serving actually poses: maximise
throughput *subject to* a latency promise.

The rejection reason names the budget and the overshoot rather than just failing
the trial — `TTFT 812.4 ms > 500.0 ms` — because that string is what the LLM
planner reads back. It is the signal telling it to trade concurrency away rather
than to try another dtype.

## What it is allowed to vary

A proposal may set any subset of these; unspecified fields inherit the
profile-anchored baseline.

| Group | Levers |
|---|---|
| Parallel shape | `tp`, `pp`, `ep`, `attention_dp` (`cp` exists but its legal set is fixed at `[1]` — context parallelism is not searched for serving) |
| Load | `batch_size`, `max_concurrency`, `input_len`, `output_len`, `request_rate`, `arrival_model` |
| Precision | `weight_dtype`, `kv_cache_dtype`, `moe_expert_dtype` |
| Scheduler and graphs | `chunked_prefill_size`, `max_num_batched_tokens`, `cudagraph_mode`, `kv_block_size`, `kv_cache_memory_fraction` |
| Speculation | `speculative_num_tokens`, `speculative_acceptance_rate`, `speculative_draft_cost_factor` |
| Collectives | `tp_allreduce_algo`, `ep_a2a_algo`, `use_turbo_deepep`, `quick_reduce`, `fuse_rmsnorm_allreduce`, `fused_kernels` |
| MoE | `ep_load_balance`, `redundant_experts` |
| Attention | `attention_backend`, `sparse_attention_topk` |
| Disaggregation | `disaggregate`, `prefill_tp`, `decode_tp`, `decode_replicas`, `transfer_backend` |

Legality is derived from the architecture and the cluster before any trial runs,
so the search is told which TP/PP/EP/batch/dtype values are admissible instead
of discovering it by failing.

`prefix_cache_hit_rate` is deliberately **not** a lever. It is a property of the
traffic, not a knob to tune — an agentic trace resends a long transcript and
reprefills almost none of it, so leaving it at `0` tunes for a workload nobody
runs. Set it under `optimization.inference` instead.

## Configuration

```yaml
target_cluster:
  name: "mi355x-inference"
  num_nodes: 1
  gpus_per_node: 8
  gpu_arch: mi355x
  hardware_config: examples/hardware_configs/mi355x.yaml

available_for_benchmark:
  has_gpu: false
  benchmark_gpus: 0

optimization:
  mode: inference
  objective: max_throughput
  memory_safety_margin: 0.10
  hbm_capacity_gb: 288.0
  inference:
    input_len: 4096
    output_len: 256
    max_concurrency: null       # defaults to the trial batch size
    prefix_cache_hit_rate: 0.0
  slo:
    ttft_ms: 500
    tpot_ms: 25
  budget:
    max_proposals: 16
    max_perf_calls: 16
    max_benchmark_calls: 0
    max_rounds: 1
    max_rlm_iterations: 8

agent:
  llm:
    provider: litellm
    timeout: 300
    max_tokens: 8000
```

The `budget` block is the cost control: `max_perf_calls` caps projector
evaluations and `max_benchmark_calls` caps GPU measurements (default `0`, i.e.
none). `max_rounds` and `max_rlm_iterations` bound the LLM loop.

Two notes on that file. `agent.llm.provider` appears in the packaged examples
but is not read — the provider comes from the `model` string. And
`optimization.hbm_capacity_gb` is where the memory ceiling comes from, combined
with `memory_safety_margin`: a trial projecting above
`hbm_capacity_gb × (1 - margin)` is rejected the same way an SLO miss is.

Packaged examples live in `infera/projection/examples/tuning/` — note that only
inference target-cluster examples ship, so a training run needs its own.

## Evaluation modes

On the serving path only `--mode dry` is special-cased: it synthesises metrics
so the loop can be exercised without the projector at all. `memory-real` and
`full` both run a real `inferasim inference` projection.

```{admonition} --profiling-mode does not reach the serving path
:class: warning
`--profiling-mode` is read only by the training branch. On the inference branch
the switch is **`available_for_benchmark.has_gpu`** in the target-cluster YAML:
set it to `true` and the agent measures, leave it `false` and every trial is
scored on the cost model. This is why the packaged inference example ships
`has_gpu: false` — passing `--profiling-mode benchmark` alongside it changes
nothing.
```

When measurement is enabled, the agent prefers a cached anchor over spawning a
run, controlled by environment rather than flags:

| Variable | Effect |
|---|---|
| `INFERASIM_INFER_BENCH_CACHE` | directory of cached anchors, looked up as `<model>_tp{tp}_pp{pp}_ep{ep}.json` |
| `INFERASIM_INFER_BENCH_ARTIFACT` | force one artifact for every trial |
| `INFERASIM_INFER_BENCH_CACHE_ONLY` | reject a trial rather than measuring when the cache misses |

`INFERASIM_INFER_BENCH_CACHE_ONLY` is the one to set if you want measured
fidelity without ever letting a search start a GPU run. When no cached artifact
matches, the agent falls back to the nearest anchor below the target
parallelism.

On the training path, `--mode memory-real` runs a real memory projection but
ranks on a **deliberately coarse throughput heuristic** rather than a
projection. The LLM is told as much; do not read it as a performance number.

## What it writes

In `--out-dir` (default `./tuning_runs/<target_cluster.name>`):

| File | Contents |
|---|---|
| `inference_trials.jsonl` | one record per trial: index, timestamp, config, result, source, legality and the reason it was rejected |
| `inference_summary.json` | the mode, the objective, and the best config with its result |
| `inference_scratchpad.txt` | the agent's durable notes — plans, hypotheses and observations that survive across rounds |
| `trials/inf_trial_<tag>.yaml` | the workload overlay each trial was actually projected from |

`inference_trials.jsonl` is the artifact worth keeping. Every trial is in it,
including the rejected ones with the budget they missed, which makes the search
auditable rather than a single recommended number. Trials are deduplicated by
config signature, so re-running does not re-score work you already have.

The serving path writes no plot; `trials.png` is produced by the training path
only.

## Requirements and limits

- The LLM stage needs `dspy` and `python-dotenv` — install with
  `pip install ".[projection-tuning]"`. Without them the agent **skips stage 2**
  with a warning and you get the deterministic seed sweep, which is exactly what
  `--seed-only` does. Stage 1 results are still written and still usable.
- The model is any LiteLLM provider string under `agent.llm.model`, defaulting
  to `openai/gpt-4o`. The key is taken from `agent.llm.api_key` or, failing
  that, `OPENAI_API_KEY` → `ANTHROPIC_API_KEY` → `LLM_API_KEY`. `LLM_MODEL` and
  `OPENAI_API_BASE` also work. With no key you get stage 1 only.
- The LLM search stage drives DSPy's interpreter, which wants `deno` on `PATH`
  (`~/.deno/bin` is also checked). This does not affect `--seed-only`.
- `max_perf_calls` and `max_benchmark_calls` are enforced. `max_proposals` is
  passed to the planner as guidance but is **not** a hard gate, so treat
  `max_perf_calls` as the real cost ceiling.
- The agent scores through the projector, so it can only distinguish what the
  projector distinguishes. Flags that do not enter the serving spec project to
  the same number, and a search over them would produce confident noise. See
  [Boundaries and verification](boundaries.md).
- `optimization.axes` is parsed but only governs the training search; it does
  not restrict the serving levers.

## Next steps

- Enumerate a space yourself instead: [Sweeps and tuning](sweeps.md).
- Ground the scorer in hardware first: [Anchors and calibration](anchors.md).
- Every lever the projector understands: [Modeled features](modeled_features.md).
