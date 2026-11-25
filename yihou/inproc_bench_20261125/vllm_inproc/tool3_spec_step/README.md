# Tool 3 (vLLM, DeepSeek-V4.1): internal DSpark whole-step decode bench & profiling

vLLM port of the GLM-5.2 "tool 3" (`sglang_decode_internal_bench_and_profiling`, KB handoff
`kbk-tool-inprocess-step-bench`), mirroring the SGLang DSv4.1 port in `../../sglang_inproc/tool3_spec_step/`.
It benchmarks and profiles **one full speculative decode step** (target verify of 1 + K tokens -> synthetic
rejection sampling -> DSpark draft of K tokens) at a fixed batch and context by calling vLLM's internals
directly: **no HTTP server, no Scheduler calls, no PD connector**. Pinned image
`infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2` (sha256:51ceb17d..., vllm `0.34.0rc1.dev897+g9c1d45c21`,
infera `0007cd808`; the llying decode leg ran infera `240c72a4f`, whose 14 extra commits touch only the router
and the example scripts).

Feature-by-feature mapping to the GLM-5.2 tool: `PARITY_tool3_vllm.md`.

```
bench/      profile_decode.py (driver), batch_state.py, topology.py, isl_spec.py (verbatim), profiling_yihou.py,
            compare_server.py (fixed waves), check_compare_paths.py (graph vs eager)
scripts/    create_container, sync_kit, run_decode, run_profile, sweeps/, verify_point, collect_sweep, arg_check,
            graph_visibility_probe (verbatim), equivalence_smoke, server_reference (vllm serve + DecodeBenchConnector),
            server_tpot, lib_v3, make_isl_shape
connector/  canonical_decode_bench.py: DecodeBenchConnector declaring canonical KV (server reference only)
presets/    dsv41_llying_decode.args / dsv41_decode.env: llying's decode-leg engine flags and env the bench does not own
tests/      CPU unit tests (python3 -m pytest tests -q)
```

## How it works

Each spawned rank (one per DP x TP rank; vLLM's `external_launcher`, the launcher `vllm bench forward` uses)
builds `LLMEngine.from_engine_args(..., usage_context=OPENAI_API_SERVER)`, i.e. the same EngineCore -> Executor ->
Worker -> `GPUModelRunner` (V2) construction, KV-cache sizing and CUDA-graph capture as `vllm serve`. The Scheduler
is constructed but never called. The bench then:

1. reserves, once, every KV block each request will touch (all KV cache groups: compressed MLA, SWA, indexer,
   compressor-state rings), checks capacity / `max_num_seqs` / `max_num_batched_tokens` / `max_model_len` up front;
2. fills those blocks with finite synthetic values (`--kv-init random|decode-bench|zeros`);
3. admits the batch with a one-token, untimed step (last prompt token over ISL-1 tokens of synthetic KV): target
   sampling + DSpark proposal give a valid draft state, context is then exactly ISL;
4. warms up N steps, finishes the requests and re-admits fresh ones (exact ISL again);
5. runs the measured loop: per step a `SchedulerOutput` (1 + K tokens per request, K drafts, as the scheduler sends
   it), `Worker.execute_model` -> `Worker.sample_tokens` -> output copy -> `take_draft_token_ids` -> synchronize ->
   cross-rank progress check, until every request emitted OSL useful tokens.

| real | synthetic |
|---|---|
| model weights (dummy load format refused), MoE routing, attention / MoE / GEMM / comm kernels and backend selection | prefix KV contents (`--kv-init`), prompt token ids (uniform random) |
| DSpark drafter, synthetic rejection sampler, sampler, FULL CUDA-graph capture and replay (target + drafter) | acceptance: vLLM's `rejection_sample_method=synthetic` (per request, per position) |
| DP sync (`dispatch_cg_and_sync_dp`), engram CPU offload, KV layout and block tables | no Scheduler, no async scheduling overlap, no admission/churn, no KV transfer |

It measures **steady-state decode cost at a fixed batch and context**, not end-to-end serving.

## Quick start (on the GPU node; containers `yihou-inproc-v3*`, label owner=yihou)

```bash
KIT=/data/yihou/inproc_bench_20261125/kit_v3                    # local copy (the docker daemon cannot read NFS homes)
bash <repo>/vllm_inproc/tool3_spec_step/scripts/sync_kit.sh <node>:$KIT     # from the workstation
export KIT_DIR=$KIT MODEL_DIR=/data/yihou/dsv41_vllm_pd_20261119/models/DeepSeek-V4.1-Flash \
       WORK_DIR=/data/yihou/inproc_bench_20261125/work_v3 CACHE_DIR=/data/yihou/inproc_bench_20261125/cache_v3
GPUS=2,3 bash $KIT/vllm_inproc/tool3_spec_step/scripts/create_container.sh      # once
S=$KIT/vllm_inproc/tool3_spec_step/scripts
# one throughput point at the llying decode shape (dry-run first: it prints the exact command)
ARGS_FILES=presets/dsv41_llying_decode.args bash $S/run_decode.sh --dry-run v3_shapeA \
    --tp-size 1 --dp-size 2 --batch-size 448 --input-len-spec list:@/kit/.../isl_shapeA.txt \
    --output-len 1024 --accept-length 3.51 --warmup-steps 10
# drop --dry-run and run detached:  setsid nohup bash ... > logs/v3_shapeA.log 2>&1 &
python3 $S/verify_point.py $WORK_DIR/iterations/v3_shapeA --expect-heterogeneous   # gate; exit 0 = pass
MAX_STEPS=40 PROFILE_START=20 PROFILE_NUM=5 bash $S/run_profile.sh p_shapeA        # not a performance number
bash $S/sweeps/run_dp2_tp1.sh 256 448 ; python3 $S/collect_sweep.py $WORK_DIR/iterations --output summary.csv; echo exit=$?
```

`runtime.log` can carry long single-line progress output: never pipe it through `tail`; use `grep -a`.
Engine construction (weight load + JIT + capture of every decode graph size) prints little for many minutes;
a silent run is not a hung run.

## Outputs (`$WORK_DIR/iterations/<name>/`)

```
result_yihou.json          aggregated result (rank 0)          rank_{0..N}_yihou.json   per-rank summary
steps_yihou.jsonl          per-step trace (rank 0)             steps_rank_N_yihou.jsonl per-rank traces
config_yihou.json          bench args + engine CLI (redacted) + speculative config
engine_resolved_yihou.json the engine's resolved config: parallel, cache (num_blocks), scheduler, compilation,
                           speculative (synthetic rates), runner/speculator classes, KV cache group inventory
command.txt git_head.txt code.diff code_hashes.sha256 bench_snapshot/ container-inspect.json (redacted)
command.json env.json versions.json kit_hashes.sha256  (common/provenance.py, redacted)
launch_started.json (before launch)  launch_status.json (exit code + wall s)  runtime.log console.log
profile/                   only with --profile: *.trace.json.gz, top_ops_rank_N_yihou.{json,txt},
                           device_timer_rank_N_yihou.json, profile_meta_yihou.json, capture_rank_N_yihou.* (capture trace)
```

Key fields: `effective_token_latency_ms_per_user` (TPOT), `step_seconds_p50/p10/p90`, `output_tokens_per_second(_per_gpu)`,
`realized_accept_length`, `verify_iterations`, `useful_output_tokens`, `terminal_extra_tokens`,
`request_completion_iteration_min/max`, `graph_execution_counts(_by_rank)` (target / draft; draft_extend null),
`target_graph_iterations`, `draft_graph_iterations`, `phase_seconds`, `memory` (blocks reserved, utilization,
fill inventory), `host_execute_seconds_p50`, `host_sample_seconds_p50`.

Metric identities: `TPOT_ms = elapsed * 1000 * C / useful`, `tok/s = useful / elapsed`, `tok/s/GPU = tok/s / (DP*TP)`;
elapsed = max over ranks; DP replicas are summed. `step_seconds_p50` is the per-step wall of rank 0 (comparable to
the serving step log's `dt_ms`). Internal TPOT is not the client TPOT `(latency - TTFT) / (OSL - 1)` (see
`scripts/server_tpot.py`).

## Heterogeneous input lengths

As in the GLM-5.2 tool: `--input-len-spec` (`uniform:N`, `bimodal:A,B,ratio`, `normal:mean,std[,lo,hi]`,
`list:a,b,...`, `list:@file`), mutually exclusive with `--input-len`, describing ONE DP rank's local batch
(C/dp values), resolved once in the parent and shipped to every rank. `input_len` is `null` for a mixed run;
`verify_point.py --expect-heterogeneous` gates it. Difference from GLM-5.2: blocks are reserved per request
(vLLM block tables are per request), so a short request does not pay for the longest one.

## Acceptance (differs from SGLang)

vLLM's synthetic sampler accepts draft i of request r iff `rand(seed_r, position) < conditional_rate_i`; 3.51 with
K=5 resolves to unconditional rates [1, 1, 0.51, 0, 0], so each request accepts 3 or 4 tokens per step,
independently. Consequences: requests finish at different steps (the batch stays fixed; extra tokens are
`terminal_extra_tokens`), the cross-rank check digests the per-request accept vector, and
`realized_accept_length` is a deterministic sample of 3.51 (same seed / ISL vector / OSL / batch -> bit-identical),
not 3.51 itself. Per-request seeds are the sampler's own unseeded draws (`np.random`), reseeded by the bench before
each admission so every DP rank and every epoch gets the same stream.

## Profiling (opt-in, off by default)

Same flags and outputs as GLM-5.2 (`--profile --profile-start-step --profile-num-steps --profile-ranks
--profile-activities --profile-top-n --profile-with-stack --[no-]profile-record-shapes --[no-]device-timer
--profile-graph-capture-trace`). Ranks are DP x TP world ranks. vLLM has no DeviceTimer: the bench brackets the
target forward (graph replay or eager `model.forward`), `GPUModelRunner.sample` and `speculator.propose` with CUDA
events (`target_forward` / `target_sample` / `draft_propose`). `--profile-graph-capture-trace` traces
`GPUModelRunner.capture_model` (inventory, not timing). Rules carried over: rank kernels with
`top_by_self_device_time_device_kernels_only`; a profiled run is not a performance measurement; attribute in the
graph mode you care about; never profile while anything else uses the GPUs.

## Equivalence against the real engine

`scripts/server_reference.sh` starts `vllm serve` with the same preset flags, the same synthetic speculative config
and vLLM's `DecodeBenchConnector` (new requests get ISL-1 tokens of dummy KV and are decoded, the PD decode path
without a prefill node), then `vllm bench serve` with fixed ISL/OSL and ignore_eos at one concurrency.
`scripts/server_tpot.py SERVER --internal POINT` compares client TPOT and the server's steady 10 s stat intervals
against the bench, listing the confounds. `bench/compare_server.py` runs the bench in fixed waves of the same
request count (`--initial-state server-like --kv-init decode-bench`) for the like-for-like internal side.

## Traps

- The bench owns `--tensor/--data-parallel-size`, `--enable-expert-parallel`, `--distributed-executor-backend`,
  `--speculative-config`, `--seed`, PP/DCP/PCP sizes, `--model`; passing them raises.
- `max_num_seqs` must cover the local batch and `max-num-batched-tokens` one verify step (local x (1 + K)); checked
  before anything is allocated (the vLLM counterpart of GLM trap #1).
- The 475 GB checkpoint does not fit one 288 GB GPU (kit VALIDATION.md): DP2 TP1 fits because vLLM shards the MoE
  experts over the DP x TP group (MoE TP = 2) without `--enable-expert-parallel`.
- A new request's draft slots are zero-initialized in the V2 runner, so the serving scheduler's padded 1 + K
  admission step would accept synthetic drafts of token 0; the bench admits with one token instead (untimed).
- Ranks exit with `os._exit(0)` after writing results (engine/NCCL teardown is skipped).

## Results (n03-33, 2026-11-25; details and gates in PARITY_tool3_vllm.md "GPU validation")

| point | step p50 | TPOT | tok/s/GPU | realized accept |
|---|---|---|---|---|
| shape A uniform: DP2 TP1, C448 (224/rank), ISL 173000, OSL 1024 | 104.30 ms | 30.46 ms | 7354 | 3.5148 |
| shape A ISL list (mean 174k, one request at 830k), run 2 | 104.51 ms | 31.33 ms (run 1: 50.96 ms, one 20.8 s stall) | 7151 | 3.5080 |
| real `vllm serve` + DecodeBench connector, same flags, 224 running per engine | 104.27 ms (engine iteration time) | 29.70 ms (stat logger) | | 3.50 (client) |
| llying decode leg step log, steady window (reference shape A) | 101.5 ms | | | 3.51 configured |

## Verifying a result before you believe it

```bash
python3 scripts/verify_point.py <point> [--expect-accept <constant>] [--expect-heterogeneous] ; echo exit=$?
python3 scripts/collect_sweep.py <iterations> --output summary.csv --prefix dp2_tp1_noep_c ; echo exit=$?
python3 -m pytest tests -q                   # CPU only
python3 bench/profile_decode.py --help       # stdlib only
```
