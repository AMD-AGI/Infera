# Non-autoregressive models (dLLM-Sim)

```{admonition} One-pager
:class: tip
**What:** latency, throughput and memory of masked-diffusion language models
(LLaDA, Dream, block diffusion) on AMD GPUs, single node to multi-node.
**Why:** a refinement step is not a decode step — it recomputes the whole
canvas, bidirectionally, and sends every position's hidden state — so the
autoregressive projection prices the wrong shape.
**Cost:** one YAML file, no GPU: `inferasim dllm --config my_run.yaml`; or
a few measured steps, transported: `--anchor-store DIR`.
```

## What a refinement step costs

A masked-diffusion model starts from `gen_len` mask tokens after the prompt
and unmasks them over `steps` forwards, `block_len` positions at a time left to
right. Each forward is a prefill-shaped pass with three differences from an
autoregressive prefill:

* **Bidirectional attention.** Every query attends to every key: the full
  `q × kv` product, twice the causal triangle.
* **The head runs over every predicted position.** At a 126k vocabulary the
  LM head is one of the largest GEMMs in the step, and its logits — upcast to
  FP32 by the samplers — are the largest activation.
* **It repeats.** A request is `steps` of these, not one prefill and many
  cheap decodes, so the activation payload `D = B × S × H × Pb` crosses the
  collectives on every step.

What each step recomputes depends on the cache mode:

| `cache` | Per step recomputes | Attends to |
|---|---|---|
| `none` | prompt + whole canvas (LLaDA as released) | everything |
| `prefix` | current and later blocks; full refresh per block (Fast-dLLM) | everything |
| `dual` | current block; full refresh per block (Fast-dLLM) | everything |
| `block` | current block; prompt prefilled once (BD3-LM, SDAR) | prompt + finished blocks + block |

`prefix` and `dual` are approximations the decoder accepts for speed; their
quality against `none` is the decoder's to validate, not the simulator's.

## How it is priced

```{graphviz}
digraph dllm {
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc" fontname="Helvetica,Arial,sans-serif" fontsize=11];
  edge [color="#5577cc"];
  CFG [label="Config frontend\nmodel dims"]; TR [label="Trace frontend\ntorch.compile on meta"];
  OPS [label="Op list\nGEMM / FAv3 problems"]; ST [label="Step model\nOrigami compute\n+ projector comm + PP"];
  ANC [label="Anchor\nmeasured steps"];
  WL [label="Step schedule\nper cache mode"]; SIM [label="Static batch\nand Poisson DES"];
  CFG -> OPS; TR -> OPS; OPS -> ST; WL -> SIM; ST -> SIM; ANC -> SIM [label="x sim ratio"];
}
```

* **Kernels are priced as the autoregressive projector prices them.** Both
  frontends reduce a model to the same records, and every GEMM and attention
  record carries the problem it runs as. GEMMs go to Origami's tile model
  (capped at the roofline, as the projector caps it); attention goes to the
  FAv3 tile model built on Origami — `SDPASimulator`, the projector's attention
  model — at the step's real shape: non-causal, one problem per distinct
  `(q, kv)` in a ragged batch, GQA heads and KV-cache bytes included.
  Element-wise kernels take the larger of their compute and memory bounds.
  Low-precision weights add the activation cast they need.
* **Origami is required.** As in the projector, a host without Origami is an
  error, not a quiet change of model. `hardware.backend: roofline` (or
  `--backend roofline`) is the explicit opt-out: every kernel at the larger of
  its compute and memory bounds on the achievable peaks Origami's roofline
  uses (MI355X: 2.5 PF dense BF16, doubled for FP8 and quadrupled for MXFP4;
  8 TB/s HBM).
* **Convolutions** are lowered to their implicit GEMM — `M = N × prod(out
  spatial)`, `N = C_out / groups`, `K = C_in / groups × prod(kernel)`, the
  groups as the batch; transposed convolutions in their scatter form — so
  Origami prices them like a linear layer, with their real tensors as their
  bytes. Depthwise convolutions (fewer than 16 filters or taps per group) run
  as direct kernels and stay on the roofline.
* **Trace frontend.** `dllm.trace.trace_model(model, *inputs)` compiles any
  PyTorch module with a custom `torch.compile` backend on meta tensors and
  counts each FX node with PyTorch's own FLOP formulas. Scaled-dot-product
  attention is one FAv3 problem, causal when `is_causal`.
* **Communication is the projector's, unchanged.** The collective model is
  `InferenceCollectiveModel` built as the projector builds it (its link
  defaults, the architecture's datasheet interconnect, a calibrated
  `INFERASIM_COLL_HW` profile), and a forward's collectives are charged by the
  projector's rules: a dense layer pays `tp_allreduce_ms`; a MoE layer pays it
  scaled by `_moe_tp_allreduce_count / _dense_tp_allreduce_count` — one
  reduction once `EP == TP`, since the all-to-all combines the experts — plus
  `ep_a2a_ms`; a pipelined forward pays `pp_p2p_ms`, every stage boundary and
  the sampled tokens' return. Each is exposed by `overlap_keep` against the
  same layer type's compute, up to `comm.overlap` — the `refine` phase's
  ceiling, the projector's `prefill_overlap`, 0 by default.
* **Quantized collectives (opt-in).** With `comm.quant_scope` set to
  `inter_node` or `all`, a collective is priced at `comm.payload_dtype`'s width
  and pays a memory-bound pack and unpack per round — one kernel each way
  with `comm.pack: fused`, the framework ops' measured 7.5× traffic and ten
  kernels with `eager`. The report prints the break-even: a fused pack wins
  on the NIC and, on an xGMI mesh, for full-canvas payloads at any TP; an
  eager one rarely wins anywhere.
* **Contention (opt-in).** `contention.enabled` charges the part of a
  collective that overlap hid: RCCL takes CUs from compute-bound kernels and
  HBM bandwidth from memory-bound ones, so over that window compute runs at
  `r = 1 − [f_mem × hbm_share + (1 − f_mem) × cu_share]` and the hidden part
  costs `(1 − r)` of itself. Off by default, so a default run is the
  projector's communication exactly.
* **Pipeline parallelism** cuts a step into micro-batches: the first crosses
  every stage, each later one adds the slowest stage, and the forward's
  `pp_p2p_ms` is charged once at the pipeline's exposure.
* **Memory.** Weights per rank, the KV cache a cached mode keeps for the
  whole canvas, and the step's activation peak (logits, or the widest MLP
  intermediate). `max_batch` is the largest batch that fits.
* **Serving.** `simulate_serving` admits Poisson arrivals at step boundaries
  while the replica has room; a batch mixes requests at different blocks and
  cache phases and each step is priced at that ragged composition. The first
  finished block is the time to first token for a server that streams blocks.

* **Runtime.** The op graph is the arithmetic; `runtime` is what runs it.
  `engine` (the default) fuses q/k/v and gate/up, runs each norm and RoPE as
  one kernel, samples with a fused kernel, attends with a split-KV decode
  kernel and writes the cache in place. `hf_eager` is the Hugging Face
  reference implementation every dLLM ships as: five separate projection
  GEMMs, 34 element-wise kernels a layer at a 4.4 µs floor and 4.85× the
  fused traffic, an FP32 sampler (18 bytes a logit), SDPA's forward kernel —
  one workgroup per (sequence, head, 128-row query tile) walking the whole KV
  at half of one CU's rate, so a small-`q` step does not spread over the
  device (`ops.fmha_fwd_ms`) — and a cache extended by concatenation, which
  rereads and rewrites all of it every cached step. Its MoE layers loop over
  experts in Python (`moe_dispatch`): `loop_all`, the Mixtral, OLMoE and
  Qwen3-MoE reference, visits every expert behind a host sync, about 110 µs
  for one tokens reach and 55 µs for one they do not; `loop_touched`, the
  DeepSeek-style loop LLaDA2 ships, visits only the touched ones at 47.5 µs;
  `grouped` is one grouped GEMM. Any field can be set in a dict over a preset.
* **Sustained clock.** `hardware.gpu_clock_mhz` prices GEMMs, attention and
  the roofline at the clock the part holds under load. MI325X at its 1000 W
  cap holds about 1510 MHz, not the 2100 MHz Origami's profile assumes; at
  the profile clock every compute-bound kernel is 1.4× optimistic. MI325X
  defaults to 1510; `INFERASIM_GPU_CLOCK_MHZ` or an explicit
  `gpu_clock_mhz` overrides it, and other parts keep their profile's clock
  until one is measured.

`InferaSimBackend` takes compute from the projector itself instead, run at
the `refine` phase — bidirectional, with the head over every position — less
the projector's own collectives, which this package charges with the same
model.

## Anchored mode

As with autoregressive runs ([anchors](anchors.md)), a measurement can be
carried to a configuration nobody measured. A dLLM anchor is a few measured
refinement steps — a full forward over `q` positions and the FP32 confidence
sampler — of one model, precision and layout:

```bash
inferasim dllm anchor --hf GSAI-ML/LLaDA-8B-Instruct --model llada_8B \
    --batch 1 --batch 8 --seq 512 --seq 1024 --out anchors/llada_8B_mi355x.json
```

A projected step is the nearest measured point, scaled by what the simulator
says changes between the two:

`step(target) = measured(point) × sim(target) / sim(point on the anchor's layout)`

Both simulations price the same way, so the measurement supplies what the
simulator does not model — framework overhead, the sampler's real cost, the
kernels the implementation actually picks — and the ratio supplies how the
step scales with batch, canvas, cache mode, TP/EP/PP and GPU. At a measured
point on the measured layout the projection is the measurement.

An anchor transports only within its regime: model, weight, activation and
KV-cache precision, attention backend, graph mode and runtime (a harvested
anchor is `hf_eager`; both sides of the ratio are priced with the anchor's
runtime). A target that differs
on any key both state is refused with what differed. `--anchor FILE` uses one
anchor; `--anchor-store DIR` (or `INFERASIM_ANCHOR_STORE`, or `anchor.store`)
picks the store's anchor for the regime, preferring the target GPU.

## Configuration

```yaml
model: llada_8B              # dLLM preset (llada_8B, dream_7B, sdar_*, llada_moe_7B_A1B, llada2_mini), any Megatron preset, or a dict
dtypes: {weight: bf16, act: bf16, kv_cache: bf16}
hardware: {arch: mi355x, backend: origami}   # origami | roofline, plus any DLLMHardware field (gpu_clock_mhz, ...)
runtime: engine              # engine | hf_eager, or {preset: hf_eager, sampler: fused, ...}
parallel: {tp: 8, pp: 1, ep: 1, replicas: 4, pp_microbatches: 0}
workload: {prompt_len: 512, gen_len: 256, block_len: 32, steps: 64,
           cache: dual, logits: all, cfg: false}
comm: {payload_dtype: fp8, quant_scope: inter_node, overlap: 0.0, pack: fused}   # pack: fused | eager
contention: {enabled: false, comm_cu_fraction: 0.0625, hbm_traffic_factor: 3.0}
anchor: {store: null, file: null, attention_backend: null, graph_mode: null}
run:
  static_batches: [1, 8, 32]
  serving: {rates: [4, 16, 64], num_requests: 200, seed: 0}
```

`--model`, `--tp`, `--pp`, `--ep`, `--cache`, `--steps`, `--batch`, `--rate`,
`--backend`, `--anchor` and `--anchor-store` override the file; `--json` writes every number the table prints. An example
lives in `dllm/examples/llada_8B_mi355x_4node.yaml`.

## Validation on MI325X

One 8x MI325X node (gfx942, ROCm 7, PyTorch 2.14, transformers 4.46), LLaDA-8B
and Dream-7B as released on Hugging Face, Qwen2.5-7B as the autoregressive
baseline: Dream keeps Qwen2.5-7B's dimensions, so the pair differs only in the
decoding schedule. The scripts are in `bench/dllm/` and every number below is
in `bench/dllm/results_mi325x/`. `hf_eager` was calibrated on LLaDA's kernel
census alone; Dream and Qwen are held out.

**Refinement steps without a cache** (`steps.py`, `sim_vs_meas.py`; 20 shapes
each, batch 1-16 by canvas 256-2048, `runtime: hf_eager`, 1510 MHz):

| | simulated / measured | anchored, 1 point | leave-one-out |
|---|---|---|---|
| LLaDA-8B (calibration) | 1.02 geomean, 0.89-1.10 | 3.9-4.1% MAPE | 2.6% |
| Dream-7B (held out) | 1.00 geomean, 0.96-1.04 | 2.1% MAPE, max 6% | 1.9% |

At the profile's 2100 MHz with the `engine` runtime the same steps simulate at
0.57-0.59x: the clock is 1.4x of it, the unfused element-wise chain (62 ms of
a 248 ms LLaDA step at batch 8, canvas 1024) most of the rest. Dream's GEMMs
simulate 11% fast and LLaDA's element-wise multiplier overstates Dream's by
about as much, so part of Dream's agreement is cancellation. Peak activation
memory is within 1% of `max_memory_allocated`.

**Cached steps** (`replay.py`). The simulator's own step schedule for every
cache mode, replayed on the model: each `(q, kv)` step a forward of `q`
positions against `kv - q` cached ones, then the eager sampler. Dream, 256
prompt and 256 generated tokens, blocks of 32, 1-4 tokens per step:

| simulated / measured | none | prefix | dual | block |
|---|---|---|---|---|
| batch 8 | 0.98 | 0.99 | 0.92 | 0.90 |
| batch 1, wall | 1.04 | 0.74 | 0.64 | 0.56 |
| batch 1, GPU busy | | 0.91 | 0.95 | 0.93 |

At batch 1 a cached step's 9-10 ms of GPU work is shorter than the Python that
issues its ~1200 kernels, so the wall time is host time the simulator does not
model; an anchor taken at batch 1 carries it. LLaDA's released code does not
support a cache (its own forward asserts against one); forcing it falls back
to materialized attention with a float bias, which the simulator does not
model, and its cached rows simulate at 0.42-0.76x.

**Autoregressive baseline** (`ar.py`): Qwen2.5-7B `generate`, greedy, 256 new
tokens, is priced as block diffusion with one-position blocks over an exact
cache. It simulates at 0.77-0.82x of measurement at batch 1-32; GEMMs at
`M <= 32` are where Origami is fast (about 20% at `M = 1`).

**When the dLLM wins** (`summarize.py`). Latency for the same 256 tokens,
Qwen AR over Dream, measured and simulated:

| tokens per step | batch | none | prefix | dual | block |
|---|---|---|---|---|---|
| 1 | 1 | 0.52 / 0.41 | 0.69 / 0.78 | 0.70 / 0.93 | 0.68 / 0.97 |
| 2 | 1 | 1.06 / 0.82 | 1.38 / 1.52 | 1.39 / 1.79 | 1.39 / 1.93 |
| 4 | 1 | 2.11 / 1.65 | 2.73 / 2.88 | 2.78 / 3.32 | 2.51 / 3.82 |
| 1 | 8 | 0.09 / 0.08 | 0.26 / 0.22 | 0.59 / 0.53 | 0.70 / 0.65 |
| 2 | 8 | 0.19 / 0.16 | 0.49 / 0.41 | 1.01 / 0.90 | 1.40 / 1.27 |
| 4 | 8 | 0.38 / 0.32 | 0.89 / 0.74 | 1.57 / 1.37 | 2.72 / 2.47 |

The simulator picks the winner in 26 of the 28 configurations; both misses
are within 10% of break-even. A dLLM beats AR when it commits more tokens per
step than its step costs AR steps. At batch 1 on HF every step is a ~10 ms
floor of small kernels whatever its `q`, so two tokens a step is enough. At
batch 8 a full-canvas step is 8 x 512 rows against AR's 8, and only `dual`
and `block`, which recompute one block, keep the ratio low. Simulated further
(`crossover.py`, prompt 512): with the `engine` runtime the break-even at
batch 32 is 8 tokens a step for `block` and out of reach for the others,
because an engine removes AR's kernel floors too and leaves it weight-read
bound; MI355X moves each threshold down about one step.

**Kernels** (`kernels.py`). HBM streams at 4.18 TB/s; an 8192-cube BF16 GEMM
at 754-789 TF/s. SDPA's forward kernel costs about the same causal as
bidirectional, and at `q = 32` against a long cache it is 10-15x the FAv3
model's split-KV time (34-175 us at KV 544-4128), which is what `hf_eager`'s
`attention: fmha_fwd` prices. The eager FP32 sampler moves 18 bytes a logit
against the fused kernel's 4.

**Collectives** (`comm_bench.py`, `comm_cmp.py`; RCCL over 8 ranks). MI300X
and MI325X are an xGMI mesh: seven 64 GB/s links a direction, one to each
peer, so a TP2 pair drives one of them. With that profile in
`INTERCONNECT_PROFILES`, all-reduce at TP 2/4/8, the EP all-to-all and a
pipeline send land within 6% of RCCL from 32 MB to 512 MB; the switched
default was 2-17x fast. Below 8 MB, RCCL's 25-50 us per call is above the
model, which prices a graph-captured engine's kernels. An FP8 two-shot
all-reduce puts 1.63 ms on the wire for 512 MB against 3.0 ms in BF16, and
its eager pack, reduce and unpack cost 5.8 ms more (`comm.pack: eager`). With
a fused pack, FP8 saves 43%, 37% and 25% of a full-canvas step's all-reduce at
TP 2, 4 and 8, and loses below about 512 rows, so a `dual` or `block` step
should not quantize (`quant_tradeoff.txt`).

**Tensor-parallel steps** (`tp_step.py`, `tp_compare.py`). Dream sharded by
hand the way Megatron shards it -- q/k/v, gate and up by rows, o and down by
columns with an all-reduce after each, the LM head by vocabulary, KV heads
replicated past four ranks -- at TP 1/2/4/8, batch 1 and 8, canvas 512-2048:

| | TP 1 | TP 2 | TP 4 | TP 8 |
|---|---|---|---|---|
| simulated / measured | 1.01-1.05 | 0.94-1.04 | 0.83-1.06 | 0.66-1.08 |

The two outliers are batch 1 on a 512 canvas at TP 4 and 8, about 9 ms of GPU
work under the eager host floor. At TP 2 and batch 8 the all-reduce is a third
of the step; priced with the switched-node default instead of the mesh, the
same steps simulate at 0.62-0.69x.

**Seven models, simulate and anchored** (`wsteps.py`, `wreplay.py`,
`anchor_eval.py`, `e2e_eval.py`; `results_mi325x/workloads/`). Dense Dream-7B,
LLaDA-8B, SDAR-1.7B and SDAR-8B, and MoE LLaDA-MoE-7B-A1B, LLaDA2.0-mini and
SDAR-30B-A3B, at 29-53 step shapes each: full steps over canvases of 256-4096
at batch 1-32, and cached steps of 32 or 128 positions against 512-4096
(LLaDA, LLaDA-MoE and LLaDA2 have no cache in their released code). Every
model has the same four anchors — batch 8 and 1, a full step and a cached one —
and is scored on the shapes it does not contain. MAPE:

| | simulate, full | simulate, cached | anchored, 4 points | leave-one-out |
|---|---|---|---|---|
| LLaDA-8B | 4.6% | | 5.2% | 2.1% |
| Dream-7B | 1.7% | 17.4% | 6.5% | 4.1% |
| SDAR-8B | 4.8% | 19.6% | 9.8% | 4.2% |
| SDAR-1.7B | 11.6% | 27.8% | 11.4% | 7.4% |
| LLaDA-MoE-7B-A1B | 11.9% | | 5.4% | 2.9% |
| LLaDA2.0-mini | 16.9% | | 12.8% | 6.0% |
| SDAR-30B-A3B | 9.2% | 7.9% | 5.0% | 3.0% |

The expert loop's per-visit costs were fitted on a separate microbenchmark
(`moe_loop.py`, `moe_fit.py`), not on these models; priced as one grouped GEMM
the three MoE models simulate at 70-87% error, a step 4-11x faster than
measured. Whole generations replayed from the simulator's schedules (Dream,
four cache modes; SDAR, block 4 and 32; 64-256 steps; batch 1-32):

| | simulate | anchored |
|---|---|---|
| Dream-7B (33 runs) | 6.1%, worst 12% | 6.4%, worst 23% |
| SDAR-8B (18) | 28.8%, worst 40% | 4.8%, worst 18% |
| SDAR-1.7B (18) | 41.2%, worst 54% | 11.1%, worst 23% |

A cached SDAR step costs 14-20 ms at any size up to batch 8: host time from
its Python-level attention path, which the simulator does not model and the
cached anchor carries. SDAR-4B is not in the set: its remote code needs
transformers 4.56, whose flex-attention mask it then rejects.

Still open:

1. **A serving engine.** Every measurement here is the HF reference
   implementation; the `engine` runtime is the simulator's, not measured.
2. **Quality.** Tokens per step is the schedule's input; what each cache mode
   and step count costs in accuracy is the decoder's to measure.
3. **Contention**, **the serving DES**, **pipeline micro-batching**,
   **convolutions**, inter-node collectives and **MI355X**.

## Boundaries

* Anchors are harvested from Hugging Face models without a KV cache
  (`q = kv`); cached shapes are reached by the simulator's ratio.
* Host time is not modelled: a step is its GPU time. An eager step at batch 1
  with a cache is host-bound (see above) and needs an anchor at that shape.
* The pipeline's sends inherit the collective model's placement of an
  intra-node hop: with `EP > 1` it is priced at pod rather than node
  bandwidth, as it is for autoregressive runs.
* Flow-matching heads (`vocab_head: false`) are priced as a projection to a
  latent; the ODE solver's own cost per step is not modelled.
* The step schedule is fixed per request; confidence-adaptive samplers that
  stop early are modelled only through `steps`.
* Expert parallelism is confined to the tensor-parallel group (no attention
  data parallelism).
* Convolutions are priced as the implicit GEMM MIOpen's implicit-GEMM solvers
  run. Where MIOpen picks Winograd, FFT or a direct solver instead, the time
  is that GEMM's, not the solver's.
