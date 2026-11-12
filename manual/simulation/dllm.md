# Non-autoregressive models (dLLM-Sim)

```{admonition} One-pager
:class: tip
**What:** latency, throughput and memory of masked-diffusion language models
(LLaDA, Dream, block diffusion) on AMD GPUs, single node to multi-node.
**Why:** a refinement step is not a decode step — it recomputes the whole
canvas, bidirectionally, and sends every position's hidden state — so the
autoregressive projection prices the wrong shape.
**Cost:** one YAML file, no GPU: `inferasim dllm --config my_run.yaml`.
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
  OPS [label="Op list\nFLOPs, bytes, dtype"]; ST [label="Step model\ncompute + comm\n+ contention + PP"];
  WL [label="Step schedule\nper cache mode"]; SIM [label="Static batch\nand Poisson DES"];
  CFG -> OPS; TR -> OPS; OPS -> ST; WL -> SIM; ST -> SIM;
}
```

* **Ops.** Both frontends reduce a model to the same records — FLOPs, bytes
  read and written, operand precision. An op costs the larger of its compute
  and memory bounds plus a launch, on the achievable peaks InferaSim's roofline
  uses (MI355X: 2.5 PF dense BF16, doubled for FP8 and quadrupled for MXFP4;
  8 TB/s HBM). Low-precision weights add the activation cast they need.
* **GEMMs and convolutions.** Every GEMM op carries the `(M, N, K, batch)`
  problem it runs as. With `hardware.gemm_backend: origami` (or `auto` on a
  host that has Origami) that problem is priced by Origami's tile model,
  capped at the same roofline; otherwise by the roofline. A convolution is
  lowered to its implicit GEMM — `M = N × prod(out spatial)`,
  `N = C_out / groups`, `K = C_in / groups × prod(kernel)`, the groups as the
  batch; transposed convolutions in their scatter form — so it gets the same
  kernel model as a linear layer, with its real tensors as its bytes.
  Depthwise convolutions (fewer than 16 filters or taps per group) run as
  direct kernels and stay on the roofline.
* **Trace frontend.** `dllm.trace.trace_model(model, *inputs)` compiles any
  PyTorch module with a custom `torch.compile` backend on meta tensors and
  counts each FX node with PyTorch's own FLOP formulas. Scaled-dot-product
  attention is priced as one fused kernel and halved when `is_causal`.
* **Collectives** are InferaSim's own calibrated models (the vLLM custom
  all-reduce fit, the all-to-all fits, the NIC model and the per-hop engine
  overhead), priced at the precision they cross the wire at. With
  `comm.quant_scope` set to `inter_node` or `all`, a collective is sent at
  `comm.payload_dtype` and pays a memory-bound pack and unpack per round. The
  report prints the break-even: on the node fabric the pack usually costs what
  it saves; on the NIC it wins.
* **Overlap and contention.** `comm.overlap` is the fraction of a layer's
  collectives issued concurrently with compute. Overlap is not free: RCCL takes
  CUs from compute-bound kernels and HBM bandwidth from memory-bound ones, so
  over the overlap window compute runs at
  `r = 1 − [f_mem × hbm_share + (1 − f_mem) × cu_share]`.
* **Pipeline parallelism** cuts a step into micro-batches: the first crosses
  every stage, each later one adds the slowest stage, and the unmasked tokens
  return to the first stage before the next step starts.
* **Memory.** Weights per rank, the KV cache a cached mode keeps for the
  whole canvas, and the step's activation peak (logits, or the widest MLP
  intermediate). `max_batch` is the largest batch that fits.
* **Serving.** `simulate_serving` admits Poisson arrivals at step boundaries
  while the replica has room; a batch mixes requests at different blocks and
  cache phases and each step is priced at that ragged composition. The first
  finished block is the time to first token for a server that streams blocks.

On a ROCm host with Origami, `InferaSimBackend` replaces the roofline compute
with InferaSim's projector run at the `refine` phase — the measured GEMM and
attention models, bidirectional, with the head over every position.

## Configuration

```yaml
model: llada_8B              # dLLM preset (llada_8B, dream_7B), any Megatron preset, or a dict
dtypes: {weight: bf16, act: bf16, kv_cache: bf16}
hardware: {arch: mi355x, gemm_backend: auto}   # auto | origami | roofline, plus any DLLMHardware field
parallel: {tp: 8, pp: 1, ep: 1, replicas: 4, pp_microbatches: 0}
workload: {prompt_len: 512, gen_len: 256, block_len: 32, steps: 64,
           cache: dual, logits: all, cfg: false}
comm: {payload_dtype: fp8, quant_scope: inter_node, overlap: 0.0}
contention: {enabled: true, comm_cu_fraction: 0.0625, hbm_traffic_factor: 3.0}
run:
  static_batches: [1, 8, 32]
  serving: {rates: [4, 16, 64], num_requests: 200, seed: 0}
```

`--model`, `--tp`, `--pp`, `--ep`, `--cache`, `--steps`, `--batch` and `--rate`
override the file; `--json` writes every number the table prints. An example
lives in `dllm/examples/llada_8B_mi355x_4node.yaml`.

## What must be validated before the numbers are quoted

Nothing in dLLM-Sim is fitted to a diffusion-model measurement yet. In order of
how much of the step each item decides:

1. **Refinement-step time on MI355X** for LLaDA-8B and Dream-7B at several
   canvas lengths and batches, in the HF reference samplers and in any
   serving engine that runs them, against both backends.
2. **Bidirectional attention efficiency.** Flash/AITER kernels at non-causal
   `q = kv` shapes, and at the `dual` and `block` shapes (`q ≪ kv`).
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

* No anchors yet: dLLM runs cannot be harvested or calibrated the way
  autoregressive ones are.
* Flow-matching heads (`vocab_head: false`) are priced as a projection to a
  latent; the ODE solver's own cost per step is not modelled.
* The step schedule is fixed per request; confidence-adaptive samplers that
  stop early are modelled only through `steps`.
* Expert parallelism is confined to the tensor-parallel group (no attention
  data parallelism).
* Convolutions are priced as the implicit GEMM MIOpen's implicit-GEMM solvers
  run. Where MIOpen picks Winograd, FFT or a direct solver instead, the time
  is that GEMM's, not the solver's (validation item 12).
