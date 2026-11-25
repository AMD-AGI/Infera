# In-process inference + bench kits for SGLang and vLLM

Two kits that call the engines' internal code directly. No HTTP server, no router and no client are
involved. Use them for quick inference, step-level benchmarks and kernel profiles. They follow the
knowledge-base tools `kbk-decode-bench-tools-overview` and `kbk-tool-inprocess-step-bench` (GLM-5.2 /
SGLang): drive the engine's own objects at a pinned version, state what is real and what is synthetic,
gate every point, and record provenance. The presets target DeepSeek-V4.1-Flash on MI355X; the code is
model-agnostic.

## Which tool answers which question

| depth | tool | engine code it drives | question it answers |
|---|---|---|---|
| whole engine, offline | `sglang_inproc/bench/infer_engine.py` | `sglang.Engine` (tokenizer manager + scheduler + detokenizer, in-process) | "does it generate sensible text?" (`generate`); "offline tok/s of the full engine for N x ISL/OSL?" (`throughput`) |
| whole engine, offline | `vllm_inproc/bench/infer_llm.py` | `vllm.LLM` | same two questions for vLLM |
| engine step | `vllm_inproc/bench/engine_step_bench.py` | V1 `EngineCore` in-process (real scheduler, executor, sampler, spec decode), stepped by hand | "what does each engine step cost (CPU included) at batch C, ISL, OSL; what is the decode steady-state tok/s?" |
| model runner | `sglang_inproc/bench/step_bench.py` | `ModelRunner` via `sglang.benchmark.one_batch.load_model` (+ `ScheduleBatch`, `ForwardBatch`) | "what do real prefill and one plain decode step cost at batch C / context L on this TP/DP/EP topology?" |
| model forward | `vllm_inproc/bench/forward_bench.py` | vLLM's `vllm bench forward` (`GPUModelRunner.execute_model`, no scheduler, one-shot CUDA graph, drafter included) | "how long does the forward of this exact batch shape take (verify / decode / prefill / chunk mixes)?" |

Rule of thumb, as in the KB overview: use the shallowest tool that answers the question. Use the
serving bench (AgentX kit) for anything about scheduling, admission, KV transfer or PD.

## Layout

```
common/       redact.py (secret redaction) provenance.py stats.py verify_point.py collect.py lib.sh (container + run helpers)
sglang_inproc/ bench/{step_bench,infer_engine}.py  scripts/{create_container,run_point}.sh  presets/dsv41*
vllm_inproc/   bench/{engine_step_bench,forward_bench,infer_llm}.py  scripts/...  presets/dsv41* (+ forward cases)
tests/        CPU-only tests (python3 -m pytest tests -q)
research/     read-only copies of the upstream sources the kits wrap (sglang one_batch, vllm benchmarks/forward)
```

## Quick start (one MI355X node, docker)

```bash
KIT=/data/yihou/inproc_bench/kit            # copy of this dir the docker daemon can read (rsync; NFS homes are root-squashed)
export KIT_DIR=$KIT MODEL_DIR=/data/yihou/models/DeepSeek-V4.1-Flash WORK_DIR=/data/yihou/inproc_bench/work
CACHE_DIR=/data/yihou/inproc_bench/cache_sgl  GPUS=0,1,2,3 bash $KIT/sglang_inproc/scripts/create_container.sh
CACHE_DIR=/data/yihou/inproc_bench/cache_vllm GPUS=4,5,6,7 bash $KIT/vllm_inproc/scripts/create_container.sh
cd $KIT
# SGLang ModelRunner step bench, #5034 topology (TP4 DPA4 EP4), global C=64:
ARGS_FILES=sglang_inproc/presets/dsv41_tp4_dpa4_ep4.args \
  bash sglang_inproc/scripts/run_point.sh sgl_c64_8k step_bench --batch-size 64 --input-len 8192 --output-len 128
# vLLM engine step bench, two GPUs (TP2; the 475 GB checkpoint does not fit one 288 GB GPU), DSpark with synthetic acceptance 3.51 (llying's decode leg):
ARGS_FILES="vllm_inproc/presets/dsv41_tp2.args vllm_inproc/presets/dsv41_dspark_synth351.args" \
  bash vllm_inproc/scripts/run_point.sh v_c64_8k engine_step_bench --batch-size 64 --input-len 8192 --output-len 128
# vLLM forward bench over the AgentX-like case list (real weights):
ARGS_FILES="vllm_inproc/presets/dsv41_tp2.args vllm_inproc/presets/dsv41_dspark_synth351.args" \
  bash vllm_inproc/scripts/run_point.sh v_fwd forward_bench --load-format auto --case-file /kit/vllm_inproc/presets/dsv41_agentx_cases.txt
# quick inference
ARGS_FILES=sglang_inproc/presets/dsv41_tp4_dpa4_ep4.args bash sglang_inproc/scripts/run_point.sh sgl_hello infer_engine generate --chat --prompt "What is 17*23?"
# gates + table
python3 common/verify_point.py $WORK_DIR/iterations/sgl_c64_8k
python3 common/collect.py $WORK_DIR/iterations --output $WORK_DIR/summary.csv
```

`run_point.sh NAME TOOL [args]` refuses an existing `NAME`. It writes `/out/iterations/NAME/` with
`command.json`, `env.json`, `versions.json`, `kit_hashes.sha256`, `container-inspect.json`, `runtime.log`,
`launch_status.json`, plus the tool's `result.json`, `config.json` and `steps.jsonl`. Knobs:
`ARGS_FILES`, `MODEL`, `TORCHRUN_NPROC` (forward bench on >1 GPU), `DRY_RUN=1`, `CONTAINER`.

## Metric identities (state them with every number)

- SGLang `step_bench`: `decode_tok_s = C / median_step_s`, `tpot_ms = median_step_s * 1000`, and
  `tok_s_per_gpu = decode_tok_s / tp_size`. A step ends when the slowest rank ends (per-step max over ranks).
- vLLM `engine_step_bench`: `steady_tok_s = steady output tokens / steady seconds` over decode steps
  while all C requests run. `tpot_ms = steady_s * 1000 * C / steady_tokens`. With spec decode,
  `tokens_per_req_step` is the realized accepted+bonus tokens per step.
- vLLM `forward_bench`: `median_us` per forward replay (max over ranks). `decode_out_tok_s_per_replica_all_accepted` (and `_per_gpu`, divided by TP×PP)
  assumes every draft is accepted (upstream semantics); divide by the real acceptance before comparing to serving.
- None of these is the client TPOT `(latency - TTFT) / (OSL - 1)` of a serving benchmark.

## Boundary

| covered | not covered |
|---|---|
| real weights, real MoE routing, real attention/MoE/GEMM/comm kernels, CUDA graphs, TP/DP-attention/EP inside one node (SGLang); TP inside one node (vLLM engine step); TP/DP/EP via torchrun (vLLM forward) | HTTP, tokenizer/detokenizer cost (except `infer_*`), request arrival, admission, KV transfer, PD, HiCache/offload tiers, multi-node |
| real prefill (SGLang step bench, vLLM engine step) then decode on that KV | prefix-cache hits (disabled), heterogeneous ISL (uniform only) |
| vLLM: DSpark/MTP spec decode with synthetic acceptance (engine step) or all-accepted drafts (forward) | SGLang step bench: speculative decoding (rejected; plain decode only). Use `infer_engine throughput` for the full spec path |
| random token ids in [0, 10000) | realistic text statistics (indexer top-k on random tokens may differ from real prompts) |

## Desensitization

- `common/redact.py` redacts env names that look like secrets (TOKEN, SECRET, PASSWORD, API-KEY, ...),
  token-shaped values (`hf_`, `ghp_`, `github_pat_`, `sk-`, `xox?-`, `AKIA`) and `user:pw@` in URLs. It runs
  before `command.json` and `env.json` are written.
- `verify_point.py` fails a point if a token-shaped string shows up in its provenance files.
  `tests/test_kits.py::test_no_secrets_in_repo_files` scans the kit itself.
- No credential is needed. Models load from a local path, and Docker Hub pulls are anonymous.

## Validation

See `VALIDATION.md` (GPU runs on smci355-ccs-aus-n03-33, 2026-11-25).
