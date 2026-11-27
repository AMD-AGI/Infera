# DeepSeek-V4.1-Flash vLLM decode benchmark

This suite starts one vLLM server container on ROCm MI355X and drives its
`/v1/completions` endpoint with exact-length token-ID prompts. Each request's
first `ISL - 1` prompt tokens get synthetic KV blocks. The engine runs the last
prompt token and then generates `OSL - 1` further tokens. GPU inference uses
the model weights and the decode kernels. The sample text has no accuracy value.

## Run

Set `MODEL_DIR` to a local DeepSeek-V4.1-Flash model snapshot and choose a
dedicated `RUN_ROOT` on a data disk or NFS mount with sufficient free space.
Set `IMAGE` to a locally available vLLM image with ROCm support for
DeepSeek-V4.1, DSpark and grouped `DecodeBenchConnector` KV caches. Replace
the example paths and image name, then run from the suite directory.

```bash
export MODEL_DIR=/path/to/local/DeepSeek-V4.1-Flash
export RUN_ROOT=/mnt/data/dsv41-fast-bench
export IMAGE='<compatible-local-vllm-image>'
export GPUS=0,1

ISL=8192 OSL=128 NUM_PROMPTS=8 WARMUP=2 CONCURRENCIES=4 ./vllm/run.sh plan smoke-vllm
ISL=8192 OSL=128 NUM_PROMPTS=8 WARMUP=2 CONCURRENCIES=4 ./vllm/run.sh run smoke-vllm
ISL=32768 OSL=4096 NUM_PROMPTS=128 WARMUP=4 CONCURRENCIES=16,32,64 ./vllm/run.sh run sweep-vllm
```

The preset uses DP2 × TP1, block size 128, the MXFP4 indexer, FP8 E4M3 KV, AITER
MoE and sparse MLA, CPU-offloaded Engram, and full decode-only CUDA graphs. Set `DP_SIZE` and
`TP_SIZE` together when changing the GPU topology. Their product must equal the
length of `GPUS`. DSpark uses five draft tokens and synthetic rejection sampling
with target acceptance length 3.51. Override `VLLM_DSPARK_ACCEPT_LENGTH` within
`[1, 6]` to change that workload. Additional vLLM options follow `--`.

The native `DecodeBenchConnector` fills the actual hybrid KV cache groups. The
local subclass specifies that the known finite synthetic fill can use the same
AITER sparse MLA path as the PD decode worker. The stock connector disables
this path in affected ROCm builds, which then reject `fp8_e4m3`. vLLM fills newly
admitted requests' KV on the GPU during the measured run. Prefix caching is disabled.
The response's `usage.prompt_tokens_details.cached_tokens` reports the prefix
prepared by the connector and should equal `ISL - 1` for each request.

Report ISL, OSL, concurrency, topology and the acceptance setting alongside
performance. A point with KV preemptions can include prompt recompute after
resume. Check `vllm:num_preemptions_total` or the server log before interpreting
the point as decode-only. Verify the active GPU batch in the server log for
steady-state comparisons. HTTP request concurrency includes admission and drain.

The DSpark preset completed the default sweep on two MI355X GPUs using
DP2 / TP1 and met the exact prompt, output and cached-token count checks.
