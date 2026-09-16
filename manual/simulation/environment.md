# Environment variables

Every `INFERASIM_*` variable the simulator reads, grouped by what it affects.
Nearly all of them have a CLI flag that wins over the env var; the env form
exists so a host can be configured once instead of every command being
decorated. The serving stack's own variables are a separate list — see
[Environment variables](../reference/environment.md) in the reference section.

```{tip}
Only one of these belongs in a normal setup: `INFERASIM_ANCHOR_STORE`. Point it
at a directory of harvested anchors and the measured default is satisfied
without a GPU. Everything else on this page is tuning or diagnosis.
```

## Anchors and model resolution

| Env | Default | What it does |
|---|---|---|
| `INFERASIM_ANCHOR_STORE` | *(unset)* | Directory of anchors indexed by regime signature. A run looks here for a match before deciding it needs to measure. Flag: `--anchor-store`. |
| `INFERASIM_ALLOW_FOREIGN_ANCHOR` | *(unset)* | Skip the check that the anchor's model matches the target's. Off by default because an anchor describes the checkpoint it measured; set it only when you know two ids name the same weights. |
| `INFERASIM_BENCH_MODEL` | *(unset)* | Checkpoint to serve while measuring. Flag: `--bench-model`. |
| `INFERASIM_MODEL` | *(unset)* | Preset spelling (`gpt_oss_120B`) used for anchor matching. It is treated as a checkpoint id only when it looks like one — i.e. contains `/` or exists on disk. |
| `INFERASIM_ROOT` | *(walk up to `configs/models`)* | Project root the tuning agent resolves configs against. Useful when a workload YAML lives outside the tree. |

## Harvesting an anchor

| Env | Default | What it does |
|---|---|---|
| `INFERASIM_BENCH_CACHE` | *(unset)* | Cache directory for `inferasim anchor`, keyed by run config. A hit skips building the engine, which is nearly all the wall time. Flag: `--cache-dir`. |
| `INFERASIM_BENCH_SERVING_BACKEND` | `vllm` | Engine the harness launches. Flag: `--serving-backend`. |
| `INFERASIM_BENCH_SERVER_ARGS` | *(unset)* | Engine flags a checkpoint needs before it will load at all — a remote-code architecture, a non-default attention backend. Flag: `--bench-server-args`. |
| `INFERASIM_BENCH_DISABLE_CAR` | *(unset)* | Fall back to RCCL all-reduce. Custom all-reduce allocates an IPC shared buffer that can fail on some ROCm GPU-subset topologies — a 4-GPU subset while another tenant holds a GPU, for instance. |
| `INFERASIM_BENCH_WARMUP_ITERS` | `20` | Warmup iterations before timing. |
| `INFERASIM_BENCH_NUM_ITERATIONS` | `64` | Timed iterations per measurement. |
| `INFERASIM_BENCH_AGGREGATE` | `median` | How repeated timings collapse to one number. Median by default so one slow first iteration — a JIT compile warmup missed — does not poison the result; `mean` recovers the historical behaviour. |
| `INFERASIM_BENCH_MOE_KERNEL_PAD` | `1` | MoE kernel padding patch during benches. `0` disables it. |
| `INFERASIM_BENCH_MOE_LAYER_WHOLE` | `0` | Time the MoE layer as a whole instead of per-module. |

The tuning agent has its own artifact-reuse variables, because it scores
thousands of candidates and must not measure per candidate:
`INFERASIM_INFER_BENCH_CACHE` (directory of `<model>_tp{tp}_pp{pp}_ep{ep}.json`
artifacts), `INFERASIM_INFER_BENCH_ARTIFACT` (one forced artifact), and
`INFERASIM_INFER_BENCH_CACHE_ONLY` (mark a trial illegal rather than fall back
to measuring when no artifact matches). See [Tuning agent](tuning_agent.md).

## Restoring a measurement to another shape

| Env | Default | What it does |
|---|---|---|
| `INFERASIM_RESTORE_SCALING` | `origami` | How a measured anchor is extrapolated to another TP. `origami` scales the measured step by the simulator's TP-scaling ratio — validated to beat a two-point measured fit at high TP. `fit` forces the measured shardable/invariant fit; `blind` is the naive TP⁻¹. Present for A/B testing the law, not for routine use. |

## Analytical backend

These matter on the `simulate` path and inside restoration, where the simulator
supplies the ratio between two shapes.

| Env | Default | What it does |
|---|---|---|
| `INFERASIM_GEMM_BACKEND` | *(auto)* | GEMM simulation backend to construct. |
| `INFERASIM_GPU_ARCH` | *(probe the local device)* | Hardware profile to model, e.g. `gfx950`. Required when projecting for hardware the host does not have. |
| `INFERASIM_GPU_CLOCK_MHZ` | *(profile default)* | Clock override. Resolution is CLI, then env, then the profile. |
| `INFERASIM_GPU_DEVICE` | `0` | Which local device to read a hardware profile from when no arch is named. |

## Collective cost model

The defaults here are measured constants, not preferences. vLLM dispatches small
decode messages to its own one-shot/two-shot all-reduce rather than RCCL, so
charging RCCL's latency put a flat, batch-independent and TP-independent
overhead into every decode step. The two ceilings exist for the opposite
failure: a prefill message is `tokens × hidden`, hundreds of times larger than
anything the small-message fit was measured over, and applying that fit there
charged more communication than the whole step cost.

| Env | Default | What it does |
|---|---|---|
| `INFERASIM_INFER_AR_FLOOR_US` | `10.0` | Fixed per-all-reduce overhead. |
| `INFERASIM_INFER_A2A_FLOOR_US` | `30.0` | Fixed per-all-to-all overhead. |
| `INFERASIM_CUSTOM_AR_MAX_MB` | `8` | Above this message size the measured small-message all-reduce fit is abandoned for the analytical bandwidth model — matching where vLLM itself stops using its custom kernel. |
| `INFERASIM_A2A_MEASURED_MAX_MB` | `32` | The same ceiling for expert-parallel all-to-all. |

## MoE modelling

| Env | Default | What it does |
|---|---|---|
| `INFERASIM_MOE_SIM_KERNEL` | `turbo`/`legacy` by flag | Which expert-GEMM kernel model to price with. Serving projections should use `vllm_fused`; the Megatron training path keeps `turbo`/`legacy`. |
| `INFERASIM_MOE_IMB_ROOFLINE` | `1` | Apply expert-load imbalance inside the roofline, per parallel view. `0` falls back to an outer multiplier. |
| `INFERASIM_ORIGAMI_IMB_PERVIEW` | `1` | Use a per-view imbalance for the benchmark and target views. `0` restores the older single-imbalance behaviour, in which imbalance cancels in the ratio — kept for before/after validation. |
| `INFERASIM_ZIPF_S` | `1.0` | Zipf skew for the injected benchmark routing strategy. Flag: `--zipf-s`. |
| `INFERASIM_ROUTING_POOL` | `65536` | Rows of pre-drawn routing assignments, so a timed decode step pays a gather rather than sampling. Floored at `8192`. |

## Diagnostics

Each of these prints and changes nothing else.

| Env | What it prints |
|---|---|
| `INFERASIM_DEBUG_BREAKDOWN` | Per-component split of a projected step — attention, MLP, quantisation. |
| `INFERASIM_DEBUG_DES_STEPS` | Every mixed step the DES prices, with its decode count, prefill tokens, and context buckets. |
| `INFERASIM_DEBUG_RESTORE` | The floor-preserving TP restoration applied to each measured decode point. |

## Related

- [Anchors and calibration](anchors.md) — what the harvest flags do.
- [Boundaries and verification](boundaries.md) — what none of these can fix.
