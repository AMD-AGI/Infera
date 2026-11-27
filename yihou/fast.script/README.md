# DeepSeek-V4.1-Flash decode-only PD benchmarks

Each suite starts one engine container and sends fixed-length synthetic token
IDs directly to the decoder. The default sweep uses ISL 10,000, OSL 500, 16
warmups and 128 measured requests at each of concurrency 16 and 32. Both
engines receive the same measured prompts at a given seed.

| Suite | MI355X default | Synthetic handoff |
| --- | --- | --- |
| [SGLang](sglang/README.md) | TP4 / attention DP4 / EP4 | Native fake transfer, with the bootstrap rooms distributed across DP ranks |
| [vLLM](vllm/README.md) | DP2 / TP1 | Native DecodeBenchConnector and the ROCm FP8 KV adapter |

## Run

Use allocated MI355X GPU IDs, Docker, and a locally available engine image.
Set `MODEL_DIR` to the DeepSeek-V4.1-Flash checkpoint. Use a dedicated
`RUN_ROOT` on a data disk or NFS mount with sufficient free space. Docker
mounts the model read-only and writes cache and temporary files under the
individual run directory. Place the suite where the Docker daemon can read
it. Run these commands from the suite directory and replace the example paths
and image name.

```bash
export MODEL_DIR=/path/to/local/DeepSeek-V4.1-Flash
export RUN_ROOT=/mnt/data/dsv41-fast-bench

GPUS=0,1,2,3 ./sglang/run.sh plan baseline-sglang
GPUS=0,1,2,3 ./sglang/run.sh run baseline-sglang

export IMAGE='<compatible-local-vllm-image>'
GPUS=0,1 ./vllm/run.sh plan baseline-vllm
GPUS=0,1 ./vllm/run.sh run baseline-vllm
```

For a quick first run, add
`ISL=1024 OSL=32 NUM_PROMPTS=8 WARMUP=4 CONCURRENCIES=4` to the command.
`plan` renders the launch and client arguments without touching Docker. Each
`run` uses a fresh tag and stops its own container on exit. Use
`./sglang/run.sh --help` or `./vllm/run.sh --help` for the shared knobs. Each
engine accepts additional options after `--`. After readiness, the runner first
checks a short request per selected GPU and records the result in `smoke.jsonl`.
The same engine stays up across the concurrency sweep, so its topology,
graph options, and context limit apply to every point.

## Results

Each `RUN_ROOT/<engine>/<tag>` contains the server log, effective commands,
image digest, allocated GPU IDs, and client `benchmark.jsonl`. The runner
prints this directory as soon as the container starts. The `summary` events
report output tok/s, output tok/s per selected decode GPU, and p50/p95 TTFT,
TPOT and latency in milliseconds. TTFT starts at the HTTP send and ends at the
first token counter increment. TPOT averages the interval between the first
and last received token over the remaining output tokens. Aggregate throughput
uses the measured phase's wall time, including request preparation, admission,
and drain. Inspect the server log for steady-state decode batch sizes.

The client fails a point if any request has a wrong length, an incomplete
stream, or a terminal error. vLLM must also report exactly `ISL - 1` cached
tokens per request. SGLang must report no retractions when it supplies the
per-request count. If either engine exports a cumulative retraction or
preemption counter, the measured phase requires zero increase.

Synthetic KV exercises decode with the requested context shape. SGLang
defaults to the native fake transfer with speculation off. vLLM defaults to
DSpark with simulated acceptance. The SGLang handoff supplies a synthetic
first token. vLLM fills KV at request admission and executes the final prompt
token to produce its first output. Output counts and client timing include
these handoffs. Treat synthetic input lengths as context lengths and use the
metrics for decode serving performance.

## MI355X validation

The default SGLang preset completed the sweep on four GPUs with speculation
disabled. The default vLLM preset completed the sweep with DSpark on two GPUs
using DP2 / TP1. Both runs met the exact request-length checks. The optional
SGLang DSpark mode passed startup validation. Request-level GPU validation for
that mode is pending.
