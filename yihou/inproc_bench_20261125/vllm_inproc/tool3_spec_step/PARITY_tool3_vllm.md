# PARITY_tool3_vllm.md — GLM-5.2 tool 3 -> vLLM DeepSeek-V4.1 port, feature by feature

Reference: `/home/yihou/dev/git.16-19/infera.dev.yihou.sglang.bench.fast.script/sglang_decode_internal_bench_and_profiling/`
(GLM-5.2-MXFP4, sglang 402df1e1, EAGLE 5/6/1) and KB handoff `kbk-tool-inprocess-step-bench`. Structure follows the
SGLang DSv4.1 port's `PARITY_tool3.md`. Port: this directory (DeepSeek-V4.1-Flash, vLLM 0.34.0rc1.dev897+g9c1d45c21 in
`infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2`, DSpark K=5, synthetic acceptance 3.51, llying decode leg DP2 TP1).
Status: **implemented** (same behaviour), **adapted** (same function, changed where vLLM/the model forces it; how +
why, first-hand), **N.A.** (why), **added**. GPU evidence: iteration names under
`smci355-ccs-aus-n03-33:/data/yihou/inproc_bench_20261125/work_v3/iterations/` (section "GPU validation" below).

Engine facts the adaptations rest on (code read of the pinned image source, copied out to
`n03-33:/data/yihou/inproc_bench_20261125/src_v3/`):
- V1 `vllm/v1/worker/gpu_worker.py`: the only GPU runner is `vllm.v1.worker.gpu.model_runner.GPUModelRunner` (V2).
  `Worker.execute_model(SchedulerOutput)` then `Worker.sample_tokens(grammar)` is one engine step; the DSpark drafter
  runs inside `sample_tokens` (`speculator.propose`). Request state (context length, last sampled token, drafts) lives
  on device (`req_states`); a SchedulerOutput carries counts and placeholder drafts only
  (`spec_decode/utils.py::DraftTokensHandler`, async-scheduler `[-1]*K`).
- V2 `spec_decode/rejection_sampler.py:172-180` + `rejection_sampler_utils.py` (SYNTHETIC_MODE): synthetic acceptance
  is per request and per draft position, `u = tl_rand32(seed_req, pos) < conditional_rate_i`; accepted tokens are the
  real draft tokens. `config/speculative.py::_acceptance_length_to_rates`: 3.51, K=5 -> [1, 1, 0.51, 0, 0].
- V3 `config/parallel.py:1188-1240`: `external_launcher` makes world = PP*TP*DP and derives the DP rank from `RANK`;
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` (EngineCore in process). `distributed/parallel_state.py:1912` skips the DP port
  logic for it.
- V4 `v1/worker/gpu/states.py:76,118`: a new request's draft slots are zeroed; `core/sched/scheduler.py:1354-1376`
  (`pad_spec_decode`) pads an admitted request whose single remaining token is to be decoded to 1 + K tokens.
- V5 `v1/worker/gpu/model_runner.py:1488-1690`: `gather_batch_req_state` runs "one new prompt token over existing
  context" as a decode row; `dispatch_cg_and_sync_dp` picks FULL / PIECEWISE / NONE and syncs DP;
  `cudagraph_manager.run_fullgraph` replays the target graph; DSpark replays its own graph through
  `DFlashSpeculator.query_cudagraph_manager.run_fullgraph` (`spec_decode/dflash/speculator.py:631-690`).
- V6 `distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`: vLLM's own fake-KV decode benchmark connector
  (fills each new request's first ISL-1 tokens of KV; packed uint8 layouts and circular buffers are zero-filled).
- V7 decode leg command `dsv41_vllm_pd_20261119/results/llying_repro_steplog_20261124/e2e_trace/decode.cmd`: DP2 TP1,
  no `--enable-expert-parallel` (MoE experts sharded across the DP x TP group), FULL_DECODE_ONLY graphs.

## bench/profile_decode.py — CLI flags

| GLM-5.2 flag | status | port | evidence |
|---|---|---|---|
| `--model-path` (required) | implemented | passed as engine `--model` | all |
| `--batch-size` (global C, default 16) | implemented | global over DP ranks | all |
| `--input-len` (default 70000) | adapted | default 173000 (shape A mean context); exclusive with spec | GPU table |
| `--input-len-spec` uniform/bimodal/normal/list/@file | implemented | `bench/isl_spec.py` verbatim; per DP rank | shape A point (list:@presets/isl_shapeA_per_dp_rank.txt) |
| `--output-len` (default 10000) | adapted | default 1024 (enough steps for steady state at 3.51: ~300 steps) | GPU table |
| `--accept-length` (3.61, bound [1, 6]) | adapted | default 3.51 -> `synthetic_acceptance_length`; bound [1, K+1] | all |
| `--accept-method` match-expected / multinomial | adapted | `length` (vLLM's minimum-variance schedule from the length, V2) / `rates` (`--accept-rates`, vLLM `synthetic_acceptance_rates`). SGLang's methods do not exist in vLLM | CPU tests; GPU: length |
| `--accept-token-mode` real-draft-token / fixed | adapted | only `real-draft-token`: the vLLM kernel always stores the draft token on accept (V2); `fixed` has no vLLM counterpart | code read |
| `--tp-size` (8) / `--ep-size` (1) / `--enable-dp-attention` | adapted | `--tp-size` (1), `--dp-size` (2), `--ep-size` 1 or DP*TP (`--enable-expert-parallel`); SGLang DP attention == vLLM DP for a MoE model (attention per DP rank, MoE across DP x TP, V7) | all |
| `--result-dir` (required) | implemented | | all |
| `--disable-cuda-graph` | implemented | -> `--enforce-eager` | eager smoke |
| `--max-steps` (0 = full OSL) | implemented | | smokes |
| `--warmup-steps` (3) | implemented | then finish + re-admit (exact ISL) | all |
| `--seed` | implemented | prompts, ISL generator, KV fill, engine `--seed`, sampler seed stream (np.random reseeded before each admission) | all |
| `--log-interval` | implemented | | all |
| profiling flags | implemented | `profiling_yihou.py`; ranks are DP x TP world ranks | profile point |
| `--enable-profile-cuda-graph` (ServerArgs) + `--profile-graph-capture-trace` | adapted | vLLM has no such flag; `--profile-graph-capture-trace` wraps `GPUModelRunner.capture_model` in torch.profiler (inventory) | profile point (CAPTURE_PROFILE=1) |
| pass-through ServerArgs | implemented | `parse_known_args` -> EngineArgs CLI; `presets/dsv41_llying_decode.args` | all |
| (new) `--num-speculative-tokens`, `--spec-method`, `--draft-sample-method` | added | build the owned `--speculative-config` (DSpark 5, probabilistic, synthetic, adaptive verification off as in V7) | config_yihou.json |
| (new) `--kv-init random/decode-bench/zeros` | added | see initialize_kv | GPU table |
| (new) `--graph-sizes engine/local` | added | engine (default): the served capture list; local: exactly local x (1+K) (GLM's "one graph = local batch") | |
| (new) `--master-port` | added | torch.distributed rendezvous of the spawned ranks | |

## bench/profile_decode.py — behaviour

| GLM-5.2 feature | status | port (how / why) | evidence |
|---|---|---|---|
| `--help` stdlib-only; plain python, rejects torchrun | implemented | `tests/test_batch_state.py::test_help_requires_no_torch_or_vllm` | CPU |
| `validate_args` | implemented / adapted | accept bound K+1; rates validation | CPU |
| `resolve_input_lens` once in parent (Plan 1) | implemented | | CPU + hetero point |
| `server_cli`: bench owns topology + graph flags | adapted | `engine_cli` owns TP/DP/EP/PP/DCP/PCP sizes, launcher, speculative config, seed, model; graph sizes are NOT owned by default (served list, see `--graph-sizes`) | `tests/test_topology.py::CliTests` |
| injected defaults (EAGLE 5/6/1, fp8 KV, DSA backends, graph bs, radix off, ...) | adapted | speculative config built by the bench; `--no-enable-prefix-caching`, `--trust-remote-code` injected; KV dtype / attention / MoE backends come from the llying preset, not hard-coded | config_yihou.json, engine_resolved_yihou.json |
| `configure_acceptance` (env before import) | adapted | vLLM's synthetic mode is an engine config field (`rejection_sample_method=synthetic`), not an env; the bench asserts the RESOLVED config (synthetic, rates, adaptive verification off) before running | engine_resolved_yihou.json |
| `create_workers` in scheduler order | adapted | `LLMEngine.from_engine_args(usage_context=OPENAI_API_SERVER)`: the engine's own construction (load, KV sizing, capture), identical to `vllm serve`; the worker is `engine.model_executor.driver_worker.worker` | all GPU points |
| ParallelState from the topology | adapted | engine derives it (V3); `compare_resolved_parallel` raises on any mismatch | all |
| `initialize_kv` + `validate_kv_layout` | adapted | fills every KV cache group's reserved blocks through `runner.kv_caches_by_layer` (the per-layer, block-indexed tensors the KV connectors see): `random` = GLM rule generalized (1-byte storage: bytes in [0, 0x7C), finite under every fp8/e8m0/fp4/bf16-high-byte reading; wider floats uniform[-1, 1]); circular buffers (compressor-state rings) zeroed as vLLM's DecodeBenchConnector does (V6); unknown dtypes raise; a tensor whose first dim is not num_blocks raises. Inventory in `memory.kv_init_inventory`, groups in `engine_resolved_yihou.json` | all |
| `allocate_batch` (reserve once; capacity, row width, context checks) | adapted | `BatchPlan`: per request and per KV group block lists sized for ISL + max_accept x ceil(OSL/min_accept) + 2(1+K) (requests progress independently, V2); SWA groups get null blocks before the window, rings 1 block; checks num_blocks, `max_num_seqs`, `max_num_batched_tokens`, `max_model_len` before allocating | memory.* in every result |
| string request ids offset by DP rank | implemented | `bench-<epoch>-<offset+i>` | all |
| `bootstrap` (one-token untimed target+draft forward) | adapted | one-token admission SchedulerOutput (V5: runs as a decode row); asserts 1 token emitted and device context == ISL. The serving padded 1+K admission is not used (V4: zeroed draft slots would be accepted synthetically and the loop would not start at ISL) | all |
| warmup then reset to exact ISL | adapted | finish the warmed requests and re-admit fresh ones on the same blocks | all |
| reseed acceptance coins before the measured loop | adapted | per-request sampler seeds are `np.random` draws at admission (V2 path for unseeded requests, as served); `np.random.seed(seed)` before each admission makes the measured stream independent of warmup | realized accept identical across warmup counts (GPU) |
| `prepare_dp_metadata` | adapted | not needed: the runner's `dispatch_cg_and_sync_dp` all-reduces DP token counts every step, exactly as in serving; every DP rank runs the same loop | all |
| `commit_result` | adapted | next SchedulerOutput carries the committed context per request; the device context (`req_states.num_computed_tokens`) is read back and checked | all |
| cross-rank progress check every iteration | adapted | 7-int signature over the gloo world group; accept is a digested VECTOR (per-request acceptance, V2) | all |
| `validate_worker_progress` | implemented | against the device context | all |
| `count_graph_executions` (target / draft / draft_extend) | adapted | wraps `run_fullgraph` of the target `CudaGraphManager` and the DSpark `query_cudagraph_manager`; `draft_extend` null (DSpark has none, V5) | GPU table |
| `target_graph` per step, `target_graph_iterations` | implemented | + `draft_graph` / `draft_graph_iterations` | all |
| terminal policy | adapted | per request (V2): a request that reached OSL keeps decoding in the fixed batch, tokens paid not counted; run ends when all are done; `request_completion_iteration_min/max` | all |
| per-rank JSON, steps JSONL, all_gather aggregate, rank-0 result | implemented | + `engine_resolved_yihou.json` | all |
| result fields | implemented / adapted | all GLM fields; `pinned_vllm`, `pinned_image`; added step percentiles, host execute/sample p50, acceptance rates/bounds, kv_init, cudagraph_mode | result_yihou.json |
| `module_source_paths` | adapted | model_runner, rejection_sampler, speculator module | all |
| `main()` asserts (topology, PP1/CP1/DCP1/MoEDP1, a2a none, spec, no dummy) | adapted | PP/DCP/PCP owned (=1), dummy load refused, synthetic + no adaptive verification asserted on the resolved config; a2a backend: N.A. (vLLM all2all is chosen by the engine from the topology; not overridden) | all |
| config_yihou.json byte-compatible when features off | implemented | same omission rule; engine CLI redacted | equivalence smoke |
| `--profile` env in parent before spawn | implemented | | profile point |
| rank spawning `mp.spawn` | implemented | `torch.multiprocessing.spawn`, rank env (RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*) set per child for the external launcher; ranks `os._exit(0)` after results (engine teardown skipped) | all |

## bench/batch_state.py, topology.py, isl_spec.py

| GLM-5.2 | status | note |
|---|---|---|
| `DecodeAccounting` | adapted | per-request raw vs useful tokens; committed context = ISL + raw; completion iteration per request |
| `validate_worker_progress`, `count_graph_executions`, `module_source_paths` | implemented / adapted | counter wraps a named method (`run_fullgraph`) |
| `required_token_capacity` | adapted | `required_tokens_per_request` + `blocks_for_request` (per KV group kind) + `accept_bounds` |
| `validate_kv_layout` | adapted | `kv_fill_rule` (refuses unknown storage) |
| `DecodeTopology` | adapted | (tp, dp, ep) vLLM vocabulary; representative ranks = TP rank 0 of each DP replica |
| `lens_digest`, `rank_progress_signature`, `validate_rank_progress`, `aggregate_rank_summaries` | implemented / adapted | accept vector digested |
| `isl_spec.py` | implemented | verbatim copy |

## bench/profiling_yihou.py

| GLM-5.2 feature | status | note |
|---|---|---|
| torch.profiler window, start/stop latch, close at loop end | implemented | |
| no `_ProfilerTorch` (rank-subset barrier deadlock); trace filename convention | adapted | the engine's worker profiler (`--profiler-config`, EngineCore RPC on every rank) is not used for the same reason; name `decode-<id>-DP-<d>-TP-<t>[-EP-<r>].trace.json.gz` |
| DeviceTimer per runner after warmup | adapted | vLLM has no DeviceTimer: CUDA-event stage timer around target forward (graph replay or eager forward), `GPUModelRunner.sample`, `speculator.propose`; resolved lazily, outside capture |
| top-N tables, annotation exclusion, text tables | implemented | |
| MEM / CUDA_PROFILER activities | implemented | MEM max entries from `BENCH_MEM_PROFILE_MAX_ENTRIES` (was an SGLang env) |
| `profile_meta` | implemented | env snapshot of VLLM_/HIP_/AITER_/TORCH_ |
| capture inventory | adapted | `install_capture_trace` around `GPUModelRunner.capture_model` |

## bench/compare_server.py, bench/check_compare_paths.py

| GLM-5.2 | status | note |
|---|---|---|
| compare_server: fixed waves, per-wave + aggregate TPOT | implemented | `wave_sizes`, `decode_budget`, `summarize_waves` (CPU tests) |
| initial_state `sikl` | adapted | `synthetic` (bench fill, OSL decode tokens) |
| initial_state `fake-server` (zero KV, one-hot draft state, OSL-1) | adapted | `server-like`: KV filled as the vLLM DecodeBenchConnector server does (`--kv-init decode-bench`), admission emits token 1, OSL-1 decode tokens; drafter state comes from the real admission step (DSpark has no hidden/topk state to fake) |
| DP1 only | adapted | DP supported |
| rejects `--input-len-spec` | implemented | |
| check_compare_paths graph / graph_repeat / eager / eager_repeat | adapted | eager forced through the engine's dispatcher (`need_eager=True`) for target and drafter; compares accept, context, sampled ids, drafts (exact) |

## scripts/

| GLM-5.2 script | status | port |
|---|---|---|
| `create_container.sh` | adapted | `common/lib.sh kit_create_container` (yihou-inproc-v3*, owner=yihou, ro model/kit, /cache, --pid host, ROCR_VISIBLE_DEVICES) + digest assert; env `presets/dsv41_decode.env` |
| `run_decode.sh` | implemented | + launch_started.json, provenance (redacted), ARG CHECK, ARGS_FILES |
| `run_profile.sh` (all knobs, MAX_STEPS=0 guard, markers) | implemented | defaults re-pinned to shape A (DP2 TP1, C448, ISL 173000, OSL 1024, 3.51); MEMUTIL replaces MEMFRAC; DP_SIZE added |
| `sweeps/run_tp8_ep8_dpa.sh`, `run_tp8_ep1_dpa.sh` | adapted | `sweeps/run_dp2_tp1.sh` / `run_dp2_tp1_ep.sh` (single-variable EP pair) |
| `verify_point.py` | adapted | vLLM topology gate; + graph replay gate, accept-within-tolerance gate, profiled-run rejection |
| `collect_sweep.py` | implemented | trap #17 import fixed |
| `graph_visibility_probe.py` | implemented | verbatim |
| `equivalence_smoke.sh` | implemented | |
| (new) `sync_kit.sh`, `arg_check.py`, `lib_v3.sh`, `server_reference.sh`, `server_tpot.py`, `make_isl_shape.py` | added | kit copy + provenance; ARG CHECK; real `vllm serve` + DecodeBenchConnector + `vllm bench serve`; TPOT comparison; shape-A ISL list |

## tests/

| GLM-5.2 test file | status | port |
|---|---|---|
| `test_isl_spec_yihou.py` | implemented | `tests/test_isl_spec.py` verbatim |
| `test_batch_state_yihou.py` | adapted | per-request terminal policy, acceptance rates/bounds, block sizing, fill rules, help import test |
| `test_topology_yihou.py` | adapted | vLLM topology, accept-vector signature, aggregate, CLI ownership |
| `test_comparison_yihou.py` | adapted | in `tests/test_scripts.py` (waves, budget) |
| (new) `tests/test_scripts.py` | added | gates, collect import, ARG CHECK, run_profile guard, stage timer report, secret scan |

## GPU validation (smci355-ccs-aus-n03-33, 2026-11-25; bench GPUs 0,1 in `yihou-inproc-v3`, server GPUs 2,3)

All points under `/data/yihou/inproc_bench_20261125/work_v3/iterations/`. Engine: DP2 TP1, preset
`dsv41_llying_decode.args` (aiter MoE, fp8_e4m3 KV, mxfp4 indexer, block 128, FULL_DECODE_ONLY, max capture 1728,
max-model-len 1048576, gpu-memory-utilization 0.95, engram CPU offload), DSpark K=5 synthetic 3.51 (rates
[1, 1, 0.51, 0, 0]). Resolved KV: 450,560 blocks per rank with graphs (486,276 eager), 6 groups (4 SWA, window 128,
32-token blocks; 1 compressed MLA + indexer, 128-token blocks; 1 compressor-state ring) — `engine_resolved_yihou.json`.

| point | what | result | gate |
|---|---|---|---|
| v3_e1_eager | eager smoke C8, ISL 2048, OSL 64, 6 steps | runs end to end; step 90.5 ms; graph replays 0/0 (eager); `verify_point --allow-eager` fails it as intended (incomplete smoke) | smoke |
| v3_g1_graph_c448_8k | graph, C448 (224/rank), ISL 8192, OSL 256, complete | step p50 100.53 ms, TPOT 30.27 ms, 7401 tok/s/GPU, accept 3.5074, target/draft replays 77/77 both ranks, host execute 1.7 ms + sample 1.3 ms per step | PASS |
| **v3_A_u173k_c448** | **shape A uniform**: C448, ISL 173000, OSL 1024, warmup 10, random KV | **step p50 104.30 ms (p10 104.0 / max 105.0), TPOT 30.46 ms, 14,708 tok/s = 7354 tok/s/GPU, realized accept 3.514781414237936, 299 steps, replays 299/299 both ranks**; KV block utilization 77.4 % | PASS |
| v3_A_list_c448 | shape A ISL list (presets/isl_shapeA_per_dp_rank.txt: mean 174,338, 106,822..829,824) | step p50 104.80 ms; one 20,837 ms step at iteration 128 on both ranks (inside the output sync), so TPOT 50.96 ms; accept 3.508011664261664 | PASS |
| v3_A_list_c448_r2 | repeat of the list point | step p50 104.51 ms; slow steps 191 ms + 1049 ms at iterations 74/75 (not 128); TPOT 31.33 ms, 7151 tok/s/GPU; accept 3.508011664261664 (bit-identical to run 1) | PASS |
| v3_A_u173k_dbfill | shape A uniform with `--kv-init decode-bench` (vLLM DecodeBenchConnector contents) | step p50 108.14 ms (+3.7 % vs random fill), TPOT 31.60 ms, accept 3.514781414237936 (identical to random fill) | PASS |
| p_v3_A_u173k_graphon | profile, graph on, 40 steps, window 20-25, all ranks, capture trace | stage timer per step: target_forward 94.9 ms, target_sample 1.75 ms, draft_propose 6.9 ms; 127 distinct kernels; top: nccl 25.9 ms/step, mfma_moe1 afp8_wfp4 20.6, _engram_lookup 12.3, gemm2_a4w4 11.9, _sparse_mla_fp8 4.5 (same kernel families as llying's trace-d); capture inventory written | profiled (gate rejects it, as intended) |
| v3_paths_c64_8k | check_compare_paths C64 ISL 8192, 2 steps | accept + committed context identical in graph / graph_repeat / eager / eager_repeat; sampled ids and drafts differ already between graph and graph_repeat; eager modes 0 graph replays | info |
| v3_cmp_A_u173k_serverlike | compare_server, 448 requests in one wave, server-like (decode-bench fill, token 1 from admission) | decode TPOT 31.51 ms, accept 3.5148, 298 steps | info |
| eqsmoke_noprofile_{patched,pristine}_yihou | equivalence_smoke vs bench_snapshot of v3_A_u173k_c448 | EQUIVALENCE ok (no differences in iterations, accept, histogram, tokens, graph counts); TPOT 7.909 vs 7.946 ms. The snapshot is the same code, so this shows repeatability, not a feature-off comparison | ok |
| collect_sweep | prefix v3_A_u173k | 2 points pass, exit 0 | ok |

### Equivalence against the real engine (v3_srv3_A_u173k_o16k)

`vllm serve` with the same preset, DP2 TP1, the same speculative config, `CanonicalDecodeBenchConnector`
(`connector/canonical_decode_bench.py`: the stock DecodeBenchConnector makes this image refuse fp8_e4m3 KV — run
v3_srv_shapeA_u173k failed with `ValueError: --kv-cache-dtype fp8_e4m3 needs gfx950 and the aiter sparse MLA kernel
..., which is off with a KV connector`; llying's Mooncake passes the same check through `canonical_kv`),
`--api-server-count 1 --enable-logging-iteration-details`; client `vllm bench serve` random ISL 173000, OSL 16384,
448 requests at max concurrency 448, ignore_eos. (OSL 1024 in v3_srv2 never had 224 requests running at once:
admission took ~110 s.)

| measure | server | bench (v3_A_u173k_c448) | delta |
|---|---|---|---|
| step time, decode-only iterations with 224 running per engine (server: engine "iteration elapsed time" = CUDA events from `execute_model` start to `sample_tokens` end; bench: host wall per step incl. sync) | median 104.27 ms (n=2473; engine 0 104.29, engine 1 104.26; p10/p90 103.82/104.86) | p50 104.30 ms | -0.03 % |
| steady TPOT from the 10 s stat logger (Running == 224, Waiting == 0): 224 / gen throughput | 29.70 ms (23 intervals) | 30.46 ms (fixed-batch TPOT incl. the tail) | -2.5 % |
| client TPOT (latency - TTFT)/(OSL - 1) | mean 24.81 / median 25.33 ms | 30.46 ms | -18.6 % / -16.9 % |

Confounds (not corrected): the client TPOT spans each request's whole life, most of it with fewer than 224 running
(admission ramp, mean TTFT 90.6 s, and the drain); server contexts during the steady window are 173k + tokens already
generated; server KV prefix is the connector fill (constant 0.015 / zeros), whose bench twin (dbfill) is 3.7 % slower
than the random fill the server number matches; the server runs async scheduling; different per-request seeds.

## Status counts

Rows above (flags, behaviour, modules, scripts, tests): **33 implemented, 41 adapted, 4 implemented/adapted, 6 added,
0 N.A.** (the SGLang-only `--accept-token-mode fixed` choice, `moe_a2a_backend` and `--enable-profile-cuda-graph` are
counted as adapted: the function is kept or replaced, the value has no vLLM counterpart). Every adapted row names its
engine fact (V1-V7) or GPU run.

Not exercised on GPU: `--ep-size 2` (sweeps/run_dp2_tp1_ep.sh), `--graph-sizes local`, `--accept-method rates`,
profiling activities MEM / CUDA_PROFILER, `--disable-cuda-graph` with `--profile`, compare_server `synthetic`
initial state, TP > 1 inside a DP rank.

## Open

- The list point's single-step stalls (20.8 s, 1.0 s) are not reproducible by iteration and overlapped other
  containers' weight loads on the host; cause not established. Repeat the point on a quiet host before quoting its TPOT.
- KV contents change step time: decode-bench fill 108.14 ms vs random 104.30 ms at the same shape; which fill matches
  a real prefill is not measured.
- Graph/eager and repeated-graph runs from identical state give identical acceptance and context but different sampled
  ids and drafts (also graph vs graph_repeat); whether that is sampler RNG state or kernel nondeterminism is not established.
- The per-request ISL distribution of shape A is modelled (presets/isl_shapeA_per_dp_rank.txt from the step-log mean
  and max), not measured.
