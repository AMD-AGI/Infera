# Tool 2 — per-operator decode/prefill micro-bench, DeepSeek-V4.1-Flash on vLLM (MI355X)

Port of the AKLR per-operator micro-bench (`aklr/benchmarks/models/{glm52,deepseekv4}` @ 33689332, read-only
copies in `../../research/aklr_src/`) to vLLM 0.34.0rc1.dev897+g9c1d45c21 (image
`infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2`, the llying decode-engine lineage). Feature set = the GLM-5.2
port's plus the deepseekv4 additions, the same bar as the SGLang port (`../../sglang_inproc/tool2_perop/`).
Parity table: `PARITY_tool2_vllm.md`. License: Apache-2.0, AMD header on every adapted file, provenance in each.

How it works: the engine is built in-process (`LLM.from_engine_args`, `external_launcher`, one process per GPU
under torchrun). Each case is an exact batch built by vLLM's own `vllm bench forward` builder (a
`SchedulerOutput` without the scheduler) and run through the real `GPUModelRunner.execute_model`. Tool 2 takes
over the eager model call: under the ForwardContext the runner built it records every operator's real inputs
in one forward (module pre-hooks + call-site wrappers where vLLM's ROCm code bypasses a module), replays each
operator alone in its own CUDA graph (cold weights rotated through a 512 MB ring), multiplies one
representative per layer kind by the kind's layer count, and checks the sum against the full forward:
`ratio = roll-up / (target forward + logits [+ drafter])`. Inputs are synthetic (random tokens, random KV
bytes in the case's blocks, seeded balanced routing): performance measurements, never correctness evidence.

## Layout

| file | holds |
|---|---|
| `dsv41/profile_decode.py` | decode / DSpark-verify entry: per-op graph timing, roll-up, report, operator shapes; `--CONC` sweep, `--full-step N`, `--trace`, `--export-only`, `--no-kernels` |
| `dsv41/profile_prefill.py` | prefill entry: every chunk of one request's TTFT; `--prefix-cache-hit` |
| `dsv41/model_bench.py` | engine args, batches + forward takeover, layer kinds, op tree, recording/replay, rotation, roll-up, memory |
| `dsv41/spec_dspark.py` | DSpark drafter row (vLLM bench forward's DraftBench), draft ops, `--full-step` stepper + phases |
| `vllm_bench.py` | shared: world/lockstep/agreed, graph/eager timing, ForwardContext snapshot/replay, AKLR balanced routing, versions, tables |
| `kernel_breakdown.py` | AKLR Kineto breakdown calibrated to graph time, `Ranking.targets()` top-5 |
| `shape_export.py` | `AKLR_EXPORT=1` call records (aiter APIs / dispatcher ops / modules, each with its tool-2 row) + `operator_shapes.json` |
| `main.py` | e2e stage on real text (real engine, chunked prefill, 256 decode steps) |
| `presets/dsv41_dp2.args` | llying's decode leg minus PD/serving knobs (DP2 TP1, aiter MoE, mxfp4 indexer + sparse logits, fp8 KV, block 128, Engram cpu_offload, FULL_DECODE_ONLY up to 1728) |
| `presets/dsv41_dspark.args` | DSpark 5, synthetic acceptance 3.51 (as the server) |
| `scripts/run_t2v.sh` | one point in the kit container under torchrun, with provenance |
| `tests/test_tool2v_cpu.py` | CPU-only tests (19) |

## Usage

Container: `../scripts/create_container.sh` with `CONTAINER=yihou-inproc-v2 GPUS=a,b KIT_DIR=<kit copy> MODEL_DIR WORK_DIR CACHE_DIR`.
From `scripts/` on the node (NPROC = DP x TP):

```bash
P=../presets/dsv41_dp2.args; D=../presets/dsv41_dspark.args
# shape A (per DP rank 223 requests x 6 query tokens at 173k context) = CONC 446 over DP2, with DSpark
ARGS_FILES="$P $D" ./run_t2v.sh NAME decode --ISL 173000 --CONC 446 --reps 10   # per-op report, then (second torchrun job) --full-step 3
ARGS_FILES="$P $D" ./run_t2v.sh NAME decode --isl-file ../presets/isl_shapeA_per_dp_rank.txt --CONC 448 --reps 10
ARGS_FILES="$P $D" ./run_t2v.sh NAME decode --ISL 173000 --CONC 446 --full-step 8
ARGS_FILES="$P $D" ./run_t2v.sh NAME decode --ISL 173000 --CONC 446 --trace
ARGS_FILES="$P $D" ./run_t2v.sh NAME decode --ISL 173000 --CONC 446 --export-only
ARGS_FILES="$P"    ./run_t2v.sh NAME prefill --ISL 65536 --reps 3 --prefix-cache-hit 0.5 --gpu-memory-utilization 0.85
ARGS_FILES="$P"    ./run_t2v.sh NAME e2e decode --ISL 8192 --CONC 64 --text-file /out/text.txt
```

Bench-own flags: `--ISL` (context per request before the step), `--reps`, `--repeat`, `--out`, `--result-dir`,
`--CONC` (global, split over DP; several values per load), `--full-step N`, `--trace`, `--export-only`,
`--no-kernels`, `--prefix-cache-hit`, `--kv-fill random|zeros`, `--routing balanced|native`,
`--isl-file F` (per-request contexts per DP rank, e.g. `presets/isl_shapeA_per_dp_rank.txt`), `--kv-len-jitter F` (vLLM bench forward's per-request context jitter; 0 = AKLR's fixed ISL),
`--no-implied-full-step`. Every vLLM `EngineArgs` flag passes through; forced: compilation/cudagraph NONE (not
under `--full-step`), `external_launcher`, prefix caching off, `auto` KV -> fp8_e4m3.

## Reading a report

Rows are `kind|row` (x layer count) or global (x calls per step).

| row | contents (vLLM call path) |
|---|---|
| `mhc_attn` / `mhc_ffn` | mHC seams (`mhc_pre_delayed` / `mhc_pre_delayed_mxfp8`: post fold + pre + Sinkhorn + the norm, Mega-MHC) |
| `mhc_post` | separate post before the Engram (Engram layers) |
| `engram` | Engram gate (`layer.engram`: wkv + gated update) — inside the layer in vLLM |
| `engram_lookup` | Engram host-table lookup (`engram.prepare_embeddings`, cpu_offload), issued from the model loop |
| `attn.wqkv_a` / `attn.wq_b` / `attn.wo_a` / `attn.wo_b` | q/kv down-projection, q up-projection, MXFP8 wo_a BMM, output projection |
| `attn.qkv_norm` / `attn.kv_insert` | fused q/kv RMSNorm + group quant; q norm + RoPE + KV insert |
| `attn.compressor_kv` / `attn.compressor` / `attn.compressor_insert` | kv-source compressor score GEMM, compression, compressed-cache insert |
| `attn.indexer_w` / `attn.indexer` / `attn.indexer_topk` | indexer weights GEMM, index K/Q (wk, wq_b, RoPE, MXFP4 quant), sparse-logits top-k |
| `attn.mla` | sparse MLA attention (`forward_mqa`) |
| `mlp.experts` | vLLM's MoE runner: gate, routing, DP dispatch all-gather, routed MXFP4 experts, shared expert, combine |
| `mlp.experts.gate` / `mlp.experts.shared_experts` | router GEMM / shared expert (both called inside the runner) |
| `mlp.experts.<rest>` | top-k routing, DP dispatch/combine collectives, routed a8w4 experts |
| `embed_tokens`, `engram_hasher`, `engram_gather`, `hc_post`, `hc_head`, `norm`, `logits` | globals |
| `draft model` | DSpark propose (all draft forwards of a step), one row; first draft layer's ops in their own table |

**Coverage** lists hooks / call sites never called (on gfx950 the Mega-MHC seam applies attn_norm / ffn_norm
itself; the preshuffled projections are timed through their call sites). **Not replayable** lists rows that
ran but failed alone. **Top kernels** = Kineto groups calibrated to graph time; `target` = top 5 non-collective.
**Operator shapes**: `<result-dir>/aklr_export/rank<r>/operator_shapes.conc<C>.json` (schema
`tool2-operator-shapes/1`) and, with `AKLR_EXPORT=1`, `calls.pid*.jsonl` (every aiter API / dispatcher op call with
argument shapes and its tool-2 row).

## Validation

n03-33, DP2 TP1 + DSpark 5, shape A (per DP rank 223 requests x 6 query tokens at 173k context):

| point | result |
|---|---|
| per-op, uniform 173k, CONC 446 | forwards 96.564 ms, roll-up 90.328 ms, ratio 0.935 |
| per-op, per-request list, CONC 448 | forwards 97.363 ms, roll-up 91.675 ms, ratio 0.942 |
| `--full-step 5` | 103.38 ms (uniform) / 104.70 ms (list); vllm-tool3 104.3 ms |
| prefill 64k, 50% hit | TTFT 752.3 ms, ratio 1.008 |

Hot ops at shape A (Top kernels): aiter fused_moe a8w4 30.9%, NCCL MoE combine/dispatch 25.7% (collective), mxfp8 block GEMMs
6.5%, sparse MLA 4.7%, mega_mhc 4.3%, indexer mxfp4 logits 4.0%. Shapes per rank: fused_moe x [2676, 5120] bf16 (1338 x DP2 after
the naive dispatch), w13 [384, 2304, 2560] / w2 [384, 5120, 576] fp4x2, top-6; GEMMs at M=1338: wq_b N32768 K1280, wo_b N5120 K8192,
wqkv_a N1792 K5120; wo_a BMM [1338, 32768] x [8192, 4096] fp8; sparse MLA q [1338, 64, 512] fp8 on a [626271, 32, 512] fp8 cache.
Full table, cross-checks and known limitations: `PARITY_tool2_vllm.md` "GPU evidence".

## Open

- `engram_lookup` isolated 0.233 ms/call vs 4.5-6.2 ms in place (e2e trace, tool3, `--trace`): not reproduced standalone.
- Prefill chunks at prefix 0 (16k tokens) exhaust vLLM's skinny-GEMM capture workspace (8 streams) when rows are captured one by
  one; prefix-hit and later chunks pass.
- `--full-step` runs the compiled engine: balanced routing does not reach the gates there (native routing, as the server).

- Balanced routing touches every expert at shape-A token counts; the server routes by content. Compare with
  `--routing native` before reading `mlp.experts` as the serving cost.
- `<rest>` rows are differences; the attention parent replayed in a graph takes vLLM's multi-stream overlap path,
  its children are timed alone, so `attn.<rest>` can read negative.
- No per-row correctness check of attribution exists (upstream TODO).
