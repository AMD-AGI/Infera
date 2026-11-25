# PARITY — tool 2 (per-op micro-bench) for DeepSeek-V4.1-Flash on vLLM vs the GLM-5.2 port

Reference: AKLR @ 33689332 `aklr/benchmarks/models/glm52/{model_bench,profile_decode,profile_prefill}.py`,
shared `sglang_bench.py`, `kernel_breakdown.py`, `main.py`, and the deepseekv4 port (copies in
`../../research/aklr_src/`); row structure mirrors `../../sglang_inproc/tool2_perop/PARITY_tool2.md`.
Target: vLLM 0.34.0rc1.dev897+g9c1d45c21, image `infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2`
(the llying decode engine lineage, infera 240c72a4f / nightly 9c1d45c2), DeepSeek-V4.1-Flash, MI355X (gfx950).
Model code read: `vllm/models/deepseek_v41/amd/{model,rocm,mhc}.py`, `deepseek_v41/{attention,common/engram}.py`,
`deepseek_v4/amd/model.py` (MoE), `model_executor/layers/fused_moe/runner/moe_runner.py`, `v1/worker/gpu/model_runner.py`,
`vllm/benchmarks/forward/*` (extracted from the image to `/data/yihou/inproc_bench_20261125/src_v2/` on n03-33).

Status: **impl** = as in glm52; **adapt** = same function, different mechanism (why, first-hand from the code read);
**n/a** = not applicable (why). Evidence: GPU point names under `/data/yihou/inproc_bench_20261125/work_v2/iterations/`
(filled in "GPU evidence" below); "CPU" = `tests/test_tool2v_cpu.py` (19 tests).

## The one structural adaptation

AKLR loads through SGLang's `ModelRunner` and builds a `ScheduleBatch`/`ForwardBatch` itself. vLLM has no
public "forward a hand-built batch" entry, but its own `vllm bench forward` (shipped in the image) builds a
`SchedulerOutput` for an exact batch and runs it through the real `GPUModelRunner.execute_model` (input prep,
attention metadata, DP sync). Tool 2 uses that builder and **takes over the eager model call**
(`runner.model.forward`, as `vllm/benchmarks/forward/capture.py` does): inside it, under the ForwardContext the
runner built, it records, times the reference and every op, and profiles — then returns a real output.
Everything below follows from this.

## Feature rows

| # | glm52 feature | status | how / why | evidence |
|---|---|---|---|---|
| 1 | `profile_decode.py` entry | impl | `dsv41/profile_decode.py` (torchrun, one process per GPU) | GPU |
| 2 | `profile_prefill.py` entry | impl | `dsv41/profile_prefill.py` | GPU |
| 3 | per-model `model_bench.py` + shared bench module | impl | `dsv41/model_bench.py`, `vllm_bench.py` | all |
| 4 | load: one process per rank, dead rank -> non-zero exit | adapt | torchrun + `distributed_executor_backend=external_launcher` (vLLM's in-process SPMD mode; DP rank derived from RANK); torchrun fails the job on a dead rank | GPU |
| 5 | `publish_once` / initializers | n/a | vLLM's `LLM.from_engine_args` builds configs, workers, KV cache | — |
| 6 | `build_server_args`: force eager, `auto` KV -> fp8, CONC % dp | adapt | `build_engine_args`: compilation mode NONE + cudagraph NONE (else the runner replays its graph and `model(**inputs)` is never called) unless `--full-step`; `auto` -> `fp8_e4m3`; prefix caching off; `max_num_seqs/batched_tokens` raised to the case | CPU (dry-run), GPU |
| 7 | `--ISL --reps --repeat --out` | impl | + `--result-dir`; `--ISL` = context per request before the step | GPU |
| 8 | `--CONC` total, split over DP, several values per load | impl | `[UNSUPPORTED]` when not divisible, `[BLOCKED_CAPACITY]` when blocks / max-model-len do not fit (from vLLM's builder), agreed across ranks, never reduced | GPU |
| 9 | `--full-step N` | adapt | separate engine with the preset's graphs (FULL_DECODE_ONLY); step 0 adds the requests (untimed), then N+2 cached-request steps at the same context, each `execute_model` + `sample_tokens` synced, fastest wins; counts `run_fullgraph` replays; phases by CUDA events (graph forward / sample / propose). vLLM's runner holds request state, so a fresh batch per sample (glm52) would put request-add work inside the clock | GPU |
| 10 | `--trace` | impl | one eager step (+ drafter) under torch.profiler, top 35, chrome trace on rank 0 | GPU |
| 11 | `--prefix-cache-hit` | adapt | first chunk's prefix built as a prefix-cache hit by vLLM's builder (DSv4.1 recomputes the last SWA window, as vLLM's scheduler does) | GPU |
| 12 | record inputs: pre-hook, first call wins | impl + adapt | pre-hooks on every module row; **adapt**: vLLM's ROCm attention reads preshuffled `fused_wqa_wkv`/`wo_b` weights directly and the MXFP8 wo_a BMM is a function, so those rows wrap their call sites (`_fused_wqa_wkv_gemm`, `_wo_b_after_wo_a`, `rocm_mxfp8_wo_a_bmm`, ...); the two mHC seams share one module/function and are told apart by the mix weights passed in | CPU, GPU |
| 13 | ambient state re-scoped on replay | adapt | vLLM's state is the ForwardContext (per-layer attention metadata, slot mappings, DP metadata, batch descriptor): every field snapshotted at the call and restored + re-entered (`override_forward_context`) on replay | GPU |
| 14 | inputs recorded before in-place mutation (deepseekv4) | impl | tensor snapshot per (id, data_ptr, version): unchanged tensors cloned once per recording, in-place writes get a new clone; descends into dataclasses (QuantizedActivation) | CPU |
| 15 | `bench()`: 20 calls/graph, 3 warm, reps, min over repeat, pool freed | impl | captured inside `vllm.distributed.parallel_state.graph_capture(device)` (vLLM's capture context) | GPU |
| 16 | prefill inner = `max(1, min(20, 16384 // extend))` | impl | unchanged | CPU |
| 17 | cold-weight rotation ring | impl + adapt | unchanged plan (CPU test); call-site rows rotate the module they read (`Op.wmod`) | CPU, GPU |
| 18 | parent rows cycled over real layers (deepseekv4) | impl | `layer`/`attn`/`mlp` replayed over every layer of the kind | GPU |
| 19 | seeded balanced routing (seed 1234), draft too | adapt | the AKLR draw (uniform distinct experts, seed 1234) injected as router logits at the `GateLinear` output with `e_score_correction_bias` zeroed (vLLM's routing-fixture mechanism): vLLM's MoE runner routes inside monolithic aiter kernels and has no `topk_output` argument; unreached gates are reported; `--routing native` diagnostic | CPU, GPU |
| 20 | fake KV: random bytes < 0x7C, no prefill forward | adapt | only the case's blocks (vLLM's builder allocates disjoint blocks), every KV tensor incl. the MXFP4 indexer cache; `--kv-fill zeros` = vLLM bench forward's init | GPU |
| 21 | MTP KV: draft pool randomized | impl | draft KV groups are in the same block pool -> filled too | GPU |
| 22 | layer kinds from the built model + roll-up | adapt | same kind names as the SGLang port (`moe+c{r}+kvsrc+indexer+engram+late`); tree depth 3 under `mlp.experts` (gate and shared expert run inside vLLM's MoE runner); Engram is inside the layer in vLLM | CPU, GPU |
| 23 | `assert_lockstep` + `agreed()` | impl | over vLLM's world CPU group | GPU |
| 24 | decode reference: whole model in one graph | adapt | target forward + logits (`compute_logits` on the logits rows) in one graph: vLLM computes logits in `sample_tokens`, SGLang inside the forward | GPU |
| 25 | prefill reference: eager forward, TTFT = sum | impl | eager forward + last-token logits | GPU |
| 26 | rank-max under EP / min otherwise | adapt | max whenever the MoE spans ranks (vLLM without EP TP-shards experts over DP x TP with naive dispatch/combine collectives) | GPU |
| 27 | Summary | impl | | GPU |
| 28 | Config | impl | + KV setup, routing, env | GPU |
| 29 | Versions | impl | bench (git or sha256 of files), vllm, aiter, torch, rocm, triton, flydsl, tilelang, aklr | GPU |
| 30 | Memory, per rank | adapt | KV tensor storage per cache group, block pool and the case's share, device weights, Engram host table | GPU |
| 31 | Hotspots (+ Kineto kernels under each row) | impl | | GPU |
| 32 | Per layer kind | impl | | GPU |
| 33 | Roll-up | impl | globals x calls per step (the model `mhc_post` also runs for the DSpark aux hidden states) | CPU |
| 34 | Coverage | impl | module hooks and call sites never called | CPU, GPU |
| 35 | Not replayable | impl | | CPU |
| 36 | dummy banner + filename tag | impl | | — |
| 37 | auto report name | impl | `<tag>_<decode|mtp>_isl_conc_tp_dp_kvdtype_date.md` + `.json` summary | GPU |
| 38 | MTP: spec worker built in the scheduler's order | adapt | the runner's own speculator (DSpark) from the engine | GPU |
| 39 | MTP: per-forward graph timing summed as the reference | adapt | target verify forward + logits graph + drafter propose graph (vLLM bench forward's DraftBench, all drafts accepted) | GPU |
| 40 | MTP: draft model one row; draft per-op rows | impl | `draft model` = propose; first draft layer's ops `draft|<row>` in their own table | GPU |
| 41 | MTP: throughput with simulated acceptance, ITL | adapt | acceptance from the speculative config (`rejection_sample_method=synthetic`, `synthetic_acceptance_length`), the same knob the server runs | GPU |
| 42 | MTP: implied second run with `--full-step 3` | adapt | `scripts/run_t2v.sh` launches it as a second torchrun job (a fresh engine with graphs); an in-process execv under the same torchrun agent store hung in distributed init on GPU | shapeA_*_fs5 |
| 43 | MTP: DSA seed carried between steps | n/a | GLM-specific; the runner carries DSpark state itself across `--full-step` steps | code read |
| 44 | prefill refuses speculation | impl | refuses `--speculative-config` | code path |
| 45 | prefill chunk plan | adapt | chunk = `--max-num-batched-tokens` per DP rank (each rank schedules its own); each chunk is a chunked request on its prefix | CPU, GPU |
| 46 | Chunk timeline | adapt | columns prefix / new / padded tokens (no MHA/MLA switch in DSv4.1) | GPU |
| 47 | growth table | impl | | GPU |
| 48 | `main.py` e2e | adapt | real engine, real text, real chunked prefill, 256 decode steps; vLLM has no teacher forcing through its engine API -> greedy tokens, ignore_eos | GPU |
| 49 | Kineto breakdown + Top kernels with top-5 targets | impl | `kernel_breakdown.py` (+ `vllm` package, `/dist-packages/`, NCCL marks); call-site children get Kineto child ranges | CPU, GPU |
| 50 | shape export `AKLR_EXPORT` / `--export-only` | adapt | no `aklr` in the image: `shape_export.py` records aiter APIs (vLLM module bindings + `rocm_aiter_ops`), dispatcher ops and module rows, each with the active tool-2 row; plus `operator_shapes.json` per CONC (args, weights, derived M/N/K, experts/top-k, heads, calls/step, ms/step, top kernels) | CPU, GPU |
| 51 | capacity refusal | impl | | GPU |
| 52 | `ck_moe_stage1` M<16 hole | n/a (check) | vLLM routes MoE through aiter `fused_moe` a8w4 (FlyDSL); see Top kernels | GPU |
| 53 | `--no-kernels` | impl | | — |
| 54 | `validate_model` | impl | DeepseekV41ForCausalLM / DeepseekV41LLMForCausalLM | GPU |

## GPU evidence

Node smci355-ccs-aus-n03-33, GPUs 4-7, 2026-11-25; container image `infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2`
(sha256:51ceb17daa7d...). Model DeepSeek-V4.1-Flash, real weights. Preset `presets/dsv41_dp2.args` (DP2 TP1), + `presets/dsv41_dspark.args`
where marked. Points under `/data/yihou/inproc_bench_20261125/work_v2/iterations/`, each with command/env/versions/kit hashes/
container inspect/runtime log/launch status.

| point | what | result | rows exercised |
|---|---|---|---|
| smoke_d3 | decode ISL 8192, CONC 8, DSpark | forwards 16.643 ms, roll-up 16.630, ratio 0.999 | 1,3,4,6-10,12-34,38-40,49 |
| smoke_p3 | prefill ISL 20000 (gpu-mem 0.85) | TTFT 477.6 ms over 2 chunks, ratio 1.006 | 2,25,44-47 |
| shapeA_u173k_c446_r2 | decode shape A, uniform 173k, CONC 446 (223 x 6 tok/rank), DSpark, AKLR_EXPORT=1 | forwards 96.564 ms (target+logits 89.8, draft 6.76), roll-up 90.328, ratio 0.935, 16212 tok/s at accept 3.51; 0 rows skipped; 40/40 target + 3/3 draft gates balanced | 8,17-24,26-35,40,41,49,50 |
| shapeA_list_c448_r2 | same, per-request contexts `presets/isl_shapeA_per_dp_rank.txt` (224/rank, mean 174k, max 830k) | forwards 97.363 ms, roll-up 91.675, ratio 0.942, 16151 tok/s | same |
| shapeA_u173k_c446_fs5 | `--full-step 5`, uniform | best synced step 103.38 ms (samples 103.38-103.80): graph forward 93.22, sample 1.65, propose 6.74; 8/8 full-graph replays; 15142 tok/s | 9,41,42 |
| shapeA_list_c448_fs5 | `--full-step 5`, list | 104.70 ms (104.70-105.05): 94.41 / 1.72 / 6.82; 8/8 replays; 15020 tok/s | 9 |
| shapeA_trace | `--trace` shape A | eager step 201.4 ms, 133 distinct kernels, 5470 launches, chrome trace on rank 0 | 10 |
| prefill_isl64k_hit50 | prefill ISL 65536, 50% prefix hit | TTFT 752.3 ms over 2 chunks, ratio 1.008 (1.010, 1.007) | 2,11,25,45-47 |
| prefill_isl64k | prefill ISL 65536, no hit | **failed**: chunk 1 (prefix 0) rows hit vLLM's `wvSplitKrc: no free capture workspace slot; more than 8 distinct streams have captured a graph containing wvSplitKrc`, then a capture with unjoined work killed the run | known limitation |

Cross-checks:
- Step time vs vllm-tool3 at shape A: tool3 104.3 ms (in-process) / 104.27 ms (vllm serve); tool 2 `--full-step` 103.38 (uniform) /
  104.70 (list) ms, within 1%. The per-op reference (96.6 ms) excludes sampling and the runner's CPU work, as in glm52.
- The e2e decode trace of the same engine (llying run) ranks MoE stage1 a8w4, NCCL, MoE gemm2, engram lookup, sparse MLA, mega_mhc on top;
  tool 2's Top kernels at shape A: fused_moe 30.9%, NCCL reduce-scatter 13.1% + group-end 12.6%, mxfp8 GEMMs 6.5%, sparse MLA 4.7%,
  mega_mhc 4.3%, indexer logits 4.0%.

Known limitations / open (observed, not investigated):
- `engram_lookup` isolated: 0.233 ms/call (0.466 ms/step); in place (e2e trace, tool3, this tool's `--trace`: `_engram_lookup_kernel`
  2 x 5.79 ms) 4.5-6.2 ms. The standalone replay of the host-table lookup does not reproduce the in-place cost; the gap lands in the
  ratio shortfall (0.935).
- Under `--full-step` (compiled engine, graphs) the balanced-routing gate hook never fires (40/40 gates reported unreached): full-step
  routing is the model's own, as in the server.
- Prefill chunks starting at prefix 0 at 16k tokens exceed vLLM's skinny-GEMM capture workspace (8 streams) when every row is captured
  separately; prefix-hit / later chunks pass.
- `hc_head` is not a row under `AKLR_EXPORT=1` (its torch.ops wrapper conflicts with the export's dispatch mode); `engram_gather` has no
  GPU kernel at decode (no Kineto sample).
- AKLR `--export-only` path (row 50) exercised through AKLR_EXPORT=1 on the shape-A points rather than a separate `--export-only` run.
