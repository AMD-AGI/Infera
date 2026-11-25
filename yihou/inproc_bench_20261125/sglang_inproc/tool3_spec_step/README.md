# Tool 3 (DeepSeek-V4.1): SGLang internal DSPARK speculative-decode bench & profiling

Port of the GLM-5.2 "tool 3" (`sglang_decode_internal_bench_and_profiling`, KB handoff
`kbk-tool-inprocess-step-bench`) to **DeepSeek-V4.1-Flash on MI355X**, image
`lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111` (sha256:6a0b7952..., sglang `6be0d75ae`).
It benchmarks and profiles **one full speculative decode iteration** (DSPARK draft -> target verify ->
accept -> draft-KV commit) at a fixed batch and context by calling SGLang's internals directly: **no HTTP
server, no Scheduler, no PD path**. The driver builds the target `TpModelWorker` and the `DSparkWorkerV2`
on every TP rank in the Scheduler's construction order and spawns its own ranks.

Feature-by-feature mapping to the GLM-5.2 tool: `PARITY_tool3.md`. GPU evidence: `VALIDATION_tool3.md`.

```
bench/      the engine (profile_decode.py, batch_state.py, topology.py, isl_spec.py, profiling_yihou.py,
            compare_server.py, check_compare_paths.py)
scripts/    runners and gates (create_container, sync_kit, run_decode, run_profile, sweeps/, verify_point,
            collect_sweep, arg_check, graph_visibility_probe, equivalence_smoke, server_reference, server_tpot)
presets/    dsv41_ix5034.args: the #5034 engine flags the bench does not own
tests/      CPU unit tests (python3 -m pytest tests -q)
```

## What it does, and what it does not

| | |
|---|---|
| real model weights, real MoE routing (EP4 / DPA4), real attention / MoE / GEMM / comm kernels | |
| real DSPARK speculative decode (block 5: gamma 5, verify window 6), draft KV injected from target hidden states as in serving | |
| CUDA-graph capture and replay (target verify graph + DSPARK draft graph), exactly as the server does | |
| prefix KV: **real prefill of synthetic random token ids** (GLM-5.2 wrote random bytes; see PARITY "initialize_kv") | |
| simulated acceptance (`SGLANG_SIMULATE_ACC_LEN`, one scalar per iteration, engine's own hook) | |
| no scheduler, no batching policy, no request arrival, no overlap schedule, no radix cache | |

It measures **steady-state decode cost at a fixed batch and context**, not end-to-end serving.

## Quick start (on the GPU node; containers are named `yihou-inproc-*`, label owner=yihou)

```bash
KIT=/data/yihou/inproc_bench_20261125/kit_t3                    # local copy (the docker daemon cannot read NFS homes)
bash <repo>/sglang_inproc/tool3_spec_step/scripts/sync_kit.sh <node>:$KIT     # from the workstation
export KIT_DIR=$KIT MODEL_DIR=/data/yihou/.../DeepSeek-V4.1-Flash WORK_DIR=/data/yihou/inproc_bench_20261125/work_t3 \
       CACHE_DIR=/data/yihou/inproc_bench_20261125/cache_t3
GPUS=0,1,2,3 bash $KIT/sglang_inproc/tool3_spec_step/scripts/create_container.sh   # once
S=$KIT/sglang_inproc/tool3_spec_step/scripts
# one throughput point at the #5034 shape (dry-run first: it prints the exact command)
ARGS_FILES=presets/dsv41_ix5034.args bash $S/run_decode.sh --dry-run t3_c64 \
    --tp-size 4 --ep-size 4 --enable-dp-attention --batch-size 64 --max-running-requests 64 \
    --input-len 8192 --output-len 1024 --accept-length 3.51 --warmup-steps 10
# drop --dry-run and run detached:  setsid nohup bash ... > logs/t3_c64.log 2>&1 &
python3 $S/verify_point.py $WORK_DIR/iterations/t3_c64          # gate; exit 0 = pass
# one profiled point (top-N kernels + DeviceTimer per runner); not a performance measurement
MAX_STEPS=40 PROFILE_START=20 PROFILE_NUM=5 bash $S/run_profile.sh p_c64
# sweeps (EP4 and EP1, single-variable pair) and the sweep gate
bash $S/sweeps/run_tp4_ep4_dpa.sh 16 32 64 ; python3 $S/collect_sweep.py $WORK_DIR/iterations --output summary.csv; echo exit=$?
```

`runtime.log` carries long single-line progress bars: never pipe it through `tail`; use `grep -a`.
CUDA-graph capture and JIT print little for minutes; a silent run is not a hung run (cold start measured
~4 min here with a warm JIT cache).

## Outputs (`$WORK_DIR/iterations/<name>/`)

```
result_yihou.json          aggregated result (rank 0)          rank_{0..3}_yihou.json   per-rank summary
steps_yihou.jsonl          per-iteration trace (rank 0)        steps_rank_N_yihou.jsonl per-rank traces
config_yihou.json          resolved bench args + server CLI (redacted) + simulated-acceptance env
command.txt git_head.txt code.diff code_hashes.sha256 bench_snapshot/ container-inspect.json (redacted)
command.json env.json versions.json kit_hashes.sha256  (common/provenance.py, redacted)
launch_started.json (before launch)  launch_status.json (exit code + wall s)  runtime.log console.log
profile/                   only with --profile: *.trace.json.gz, top_ops_rank_N_yihou.{json,txt},
                           device_timer_rank_N_yihou.json, profile_meta_yihou.json
```

Key fields: `effective_token_latency_ms_per_user` (TPOT), `output_tokens_per_second(_per_gpu)`,
`realized_accept_length`, `verify_iterations`, `useful_output_tokens`, `terminal_extra_tokens`,
`graph_execution_counts` / `graph_execution_counts_by_rank` (target / draft; draft_extend is null: DSPARK
has none), `target_graph_iterations`, `captured_decode_bs`, `phase_seconds`, `memory` (incl. KV pool
inventory and tree cache kind), `target_backend` / `draft_backend`, `dspark_verify_mode`.

Metric identities: `TPOT_ms = elapsed * 1000 * C / useful`, `tok/s = useful / elapsed`; elapsed = max over
ranks; under DPA one representative per DP replica is summed. Internal TPOT is not the client TPOT
`(latency - TTFT) / (OSL - 1)` of a serving benchmark (see `scripts/server_tpot.py`).

## Heterogeneous input lengths

As in the GLM-5.2 tool: `--input-len-spec` (`uniform:N`, `bimodal:A,B,ratio`, `normal:mean,std[,lo,hi]`,
`list:a,b,...`, `list:@file`), mutually exclusive with `--input-len`, describing ONE attention-DP rank's local
batch (C/dp values), resolved once in the parent and shipped to every rank. `input_len` is `null` for a mixed
run; `verify_point.py --expect-heterogeneous` gates it. Difference from GLM-5.2: the prefix is a real prefill
of each request's own length, so a short request is not sized for the longest one in the pool rows it
prefills (decode headroom is allocated by the engine's own `prepare_for_decode`).

## Profiling (opt-in, off by default)

Same three mechanisms and outputs as GLM-5.2 (`--profile --profile-start-step --profile-num-steps
--profile-ranks --profile-activities --profile-top-n --profile-with-stack --[no-]profile-record-shapes
--[no-]device-timer --profile-graph-capture-trace`, plus `--enable-profile-cuda-graph` passed through).
Rules carried over: rank kernels with `top_by_self_device_time_device_kernels_only`; a profiled run is not a
performance measurement; attribute in the graph mode you care about; never profile while anything else uses
the GPUs. DeviceTimer categories here are the decode-graph runner's `forward_mode.name.lower()` per runner
(`by_runner.target` / `by_runner.draft`), not EAGLE's `eagle_draft` / `eagle_draft_extend`.

## Traps (carried over, and new ones found in this port)

- `--max-running-requests <C>` is mandatory (GLM-5.2 trap #1; DSPARK also resolves a default when unset).
- The bench owns `--tp/--ep/--dp-size` (and the `--tensor/--expert/--data-parallel-size` aliases),
  `--enable-dp-attention`, `--moe-a2a-backend`, `--cuda-graph-bs-decode`, `--cuda-graph-max-bs-decode`.
- sglang 6be0d75ae resolves `--enable-dp-attention --dp-size 4` into `attn_dp_size=4, dp_size=1`; the
  resolved view must be checked on `attn_dp_size` (an earlier check on `dp_size` rejected a correct config).
- DSPARK + DP attention requires `--enable-dp-lm-head` (engine check); the preset carries it.
- `resolve_forward_inputs(batch, None)` is only valid for prefill staging; on decode it dereferences the
  FutureMap. The non-overlap spec-v2 path installs `next_draft_input` directly.
- Simulated acceptance is drawn from a private generator seeded from `--random-seed`
  (`spec_utils.simulated_acc_len_generator`), not the process RNG: the bench re-creates it before the
  measured loop so `realized_accept_length` does not depend on the number of warmup steps.

## Verifying a result before you believe it

```bash
python3 scripts/verify_point.py <point> [--expect-accept <constant>] ; echo exit=$?
python3 scripts/collect_sweep.py <iterations> --output summary.csv --prefix tp4_ep4_dpa_on_c ; echo exit=$?
python3 -m pytest tests -q                   # CPU only
python3 bench/profile_decode.py --help       # stdlib only
```
`realized_accept_length` is deterministic given (accept, OSL, seed): one scalar is drawn per iteration from the engine's
simulated-acceptance generator, independent of C and ISL. Every point at the same OSL must reproduce it bit for bit:
3.51 / OSL 1024 / seed 1234 -> `3.5625` (288 iterations; the verify_point default), OSL 4096 -> `3.520618556701031`
(1164 iterations). A short run gives something else entirely.
