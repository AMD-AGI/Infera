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
  and pays a memory-bound pack and unpack per round. The report prints the
  break-even: on the node fabric the pack usually costs what it saves; on the
  NIC it wins.
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
KV-cache precision, attention backend and graph mode. A target that differs
on any key both state is refused with what differed. `--anchor FILE` uses one
anchor; `--anchor-store DIR` (or `INFERASIM_ANCHOR_STORE`, or `anchor.store`)
picks the store's anchor for the regime, preferring the target GPU.

## Configuration

```yaml
model: llada_8B              # dLLM preset (llada_8B, dream_7B), any Megatron preset, or a dict
dtypes: {weight: bf16, act: bf16, kv_cache: bf16}
hardware: {arch: mi355x, backend: origami}   # origami | roofline, plus any DLLMHardware field
parallel: {tp: 8, pp: 1, ep: 1, replicas: 4, pp_microbatches: 0}
workload: {prompt_len: 512, gen_len: 256, block_len: 32, steps: 64,
           cache: dual, logits: all, cfg: false}
comm: {payload_dtype: fp8, quant_scope: inter_node, overlap: 0.0}
contention: {enabled: false, comm_cu_fraction: 0.0625, hbm_traffic_factor: 3.0}
anchor: {store: null, file: null, attention_backend: null, graph_mode: null}
run:
  static_batches: [1, 8, 32]
  serving: {rates: [4, 16, 64], num_requests: 200, seed: 0}
```

`--model`, `--tp`, `--pp`, `--ep`, `--cache`, `--steps`, `--batch`, `--rate`,
`--backend`, `--anchor` and `--anchor-store` override the file; `--json` writes every number the table prints. An example
lives in `dllm/examples/llada_8B_mi355x_4node.yaml`.

## What must be validated before the numbers are quoted

The simulated path is not fitted to a diffusion-model measurement yet; an
anchor fixes its absolute level, not how it scales. In order of how much of
the step each item decides:

1. **Refinement-step time on MI355X** for LLaDA-8B and Dream-7B at several
   canvas lengths and batches, in the HF reference samplers and in any
   serving engine that runs them — harvested as anchors, and the anchor
   ratio checked by transporting one point to the others.
2. **Bidirectional attention efficiency.** The FAv3 model against
   Flash/AITER kernels at non-causal `q = kv` shapes, and at the `dual` and
   `block` shapes (`q ≪ kv`).
3. **LM head and sampler.** The head over every position and the FP32 softmax
   and ranking — time and peak memory.
4. **Cache-mode step counts and quality.** Steps per block and accuracy of
   `prefix`/`dual` against `none`, per Fast-dLLM.
5. **Collectives at dLLM payloads.** All-reduce and PP hops at
   tens-to-hundreds of MB per layer, intra-node and over RoCE.
6. **Quantized collectives.** Pack/unpack kernel time and the accuracy cost of
   FP8 and MXFP4 activations on the wire.
7. **Contention.** Overlap microbenchmarks to fit `comm_cu_fraction` and
   `hbm_traffic_factor`.
8. **Activation memory peaks** against `torch.cuda.max_memory_allocated`.
9. **Preset dimensions** against the checkpoints' `config.json`.
10. **The serving DES** against a real dLLM server under Poisson load.
11. **Pipeline micro-batching** of a refinement step, if an engine supports it.
12. **Convolutions.** Per layer shape, MIOpen's chosen solver and time on
    MI355X against Origami on the lowered implicit GEMM.

## Boundaries

* Anchors are harvested from Hugging Face models without a KV cache
  (`q = kv`); cached shapes are reached by the simulator's ratio.
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
  is that GEMM's, not the solver's (validation item 12).
