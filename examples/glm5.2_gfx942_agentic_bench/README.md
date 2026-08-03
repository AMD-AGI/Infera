# GLM-5.2-FP8 SGLang PD on gfx942 — agentic bench

Runnable package for GLM-5.2-FP8 on two gfx942 nodes: SGLang prefill/decode
disaggregation over Mooncake RDMA, MTP speculative decoding, and Infera kv-aware
routing — then an agentic multi-turn benchmark driven entirely by tooling that
ships inside the image.

The engine image is built directly from `deploy/docker/Dockerfile.sglang.gfx942`.
No runtime patching of SGLang, Mooncake, or Infera is applied on top of it.

The benchmark uses SGLang's built-in `sglang.benchmark.serving --dataset-name
agentic-trace` against a public Apache-2.0 corpus, so reproducing it needs no
private tooling.

## Topology

Two nodes, 8 GPUs each. The node names and IPs below are placeholders —
substitute your own.

| Node | Role |
|---|---|
| `node-0` | etcd + router + prefill + bench driver |
| `node-1` | decode |

Every script is driven by environment variables; you should not need to edit any
of them. `env.sh` holds the defaults. Export at least these on **both** nodes:

```bash
export PREFILL_IP=10.0.0.1                    # node-0, on the data network
export DECODE_IP=10.0.0.2                     # node-1, on the data network
export MODEL=/your/path/GLM-5.2-FP8           # local weights dir
export DATA_DIR=/your/path/agentic-data       # where the trace dataset lives
export IMAGE=infera:sglang-gfx942-glm52
```

If your nodes resolve by name, `PREFILL_NODE` / `DECODE_NODE` derive the IPs
instead. `IB_DEVICE` (default `mlx5_0`) and `MC_GID_INDEX` (default `3`) must
match the RDMA rail the two nodes share.

## Scripts

| Script | Where | Purpose |
|---|---|---|
| `build_image.sh` | host | Build `IMAGE` from `Dockerfile.sglang.gfx942`. |
| `preflight_rdma.sh` | host, both nodes | RDMA device visibility + optional cross-node fabric check. |
| `host_container.sh` | host, both nodes | Start/remove the long-lived engine container. |
| `launch/launch_etcd.sh` | host, prefill node | etcd, from the official etcd image. |
| `launch/launch_router.sh` | container, prefill node | Infera kv-aware router. |
| `launch/launch_prefill.sh` | container, prefill node | SGLang prefill leg. |
| `launch/launch_decode.sh` | container, decode node | SGLang decode leg. |
| `smoke.sh` | container, prefill node | Worker list + one chat request + RDMA hand-off. |
| `weka_to_agentic_trace.py` | container, prefill node | Public traces → SGLang `agentic-trace` dataset. |
| `run_agentic_trace.sh` | container, prefill node | Run the benchmark and rescore it. |
| `score_agentic_trace.py` | container, prefill node | Recompute cache metrics (see §4.3). |
| `stop.sh` | container, both nodes | Stop router/engine processes. |

## 1. Build the image

Run on both nodes, or build once and push to a registry:

```bash
cd examples/glm5.2_gfx942_agentic_bench
bash build_image.sh
```

The build context is the repository root and the Dockerfile is
`deploy/docker/Dockerfile.sglang.gfx942` — that single file is the whole image
definition.

## 2. Bring up

Cross-node PD moves KV over the fabric on every request, so check RDMA first. On
both hosts:

```bash
bash preflight_rdma.sh     # active RDMA port count must be non-zero
bash host_container.sh
```

On the prefill host — etcd runs on the host, the rest inside the container:

```bash
bash launch/launch_etcd.sh
docker exec -it infera-glm52-gfx942 bash
  bash launch/launch_router.sh
  bash launch/launch_prefill.sh
```

On the decode host:

```bash
docker exec -it infera-glm52-gfx942 bash
  bash launch/launch_decode.sh
```

Cold start takes 15–25 min — both legs load GLM-5.2 and the MTP nextn layer, and
the log goes quiet while they do. Follow `logs/prefill.log` on the prefill node
and `logs/decode.log` on the decode node; don't kill a slow load.

## 3. Smoke test

Once both workers are registered, on the prefill node inside the container:

```bash
bash smoke.sh
```

One prefill plus one decode worker, a coherent answer, and `installTransport,
type=rdma` in the decode log means the router paired the legs and KV moves over
RDMA rather than falling back to TCP.

## 4. Agentic benchmark

### 4.1 Build the dataset

The corpus is [`semianalysisai/cc-traces-weka-062126-256k`][corpus] (Apache-2.0),
Claude Code agent traffic. It carries only per-turn token counts and KV block ids
— no text — so `weka_to_agentic_trace.py` synthesizes filler text while preserving
the real structure: exact per-turn lengths and the block-level prefix reuse.

[corpus]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k

```bash
source env.sh                       # for TRACE, MODEL and OUTPUT_LEN
export HF_HOME=/your/path/hf_cache
huggingface-cli download semianalysisai/cc-traces-weka-062126-256k --repo-type dataset
SRC=$HF_HOME/hub/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots/*/traces.jsonl

python3 weka_to_agentic_trace.py "$SRC" -o "$TRACE" \
  --output-len "$OUTPUT_LEN" --min-turns 4 --max-context 100000 \
  --verify 20 --tokenizer "$MODEL"
```

`--max-context` fits the corpus to what the deployment can prefill; `--dry-run`
reports the resulting distribution without writing, which is the cheap way to pick
it. At 100000 this yields 295 conversations, p50 peak context 78,848 tokens, and
`--verify` reproduces every checked turn's length exactly.

### 4.2 Run

```bash
NUM_PROMPTS=60 CONC=4 bash run_agentic_trace.sh
```

`NUM_PROMPTS` counts **conversations**, not requests — 60 conversations averaging
8 turns is ~448 requests. The script flushes both legs' caches, runs
`sglang.benchmark.serving` against the router, and then rescores the result.
Sweep capacity by varying `CONC`.

### 4.3 Read the result

`sglang.benchmark.serving` mis-reports the input side in multi-turn mode: it keeps
the conversation-level `prompt_len` for every turn, so its summary can print
`Total input tokens: 0` next to a cache hit rate above 100%. Its per-request
`cached_tokens` come from the server and are correct.

`score_agentic_trace.py` therefore recomputes against the dataset's verified
per-turn lengths, and reports the number worth comparing across tools —
**efficiency**, actual cache hits over what a growing-prefix session could reuse
at best. It prints, for example:

```
  actual hit rate               84.01 %
  ideal  hit rate               84.09 %
  efficiency (a/i)              99.90 %
  tokens lost to evict           1,600 (0.10% of ideal)
```

Efficiency near 100% with almost no eviction means the run is below the pressure
point and kv-aware routing has nothing to distinguish itself on. Raise `CONC`
until eviction appears; that is where the routing policy starts to matter.

## 5. Stop

Inside each engine container:

```bash
bash stop.sh
```

Then on the hosts:

```bash
docker rm -f infera-glm52-gfx942
docker rm -f infera-glm52-etcd    # prefill host only
```

## Notes & gotchas

1. **`--output-len` must equal `--sharegpt-output-len`.** The converter solves each
   turn's filler size with the reply length baked in (`ignore_eos` is on by
   default, so replies are exactly that long). `OUTPUT_LEN` in `env.sh` feeds the
   run side; pass the same value when building the dataset or lengths drift turn
   over turn.
2. **`--warmup-requests` replays a whole conversation** in multi-turn mode, which
   pre-warms the cache the run is measuring. `run_agentic_trace.sh` defaults it
   to 0.
3. **`flush_cache` is a no-op while requests are in flight** and still returns
   success. Make sure nothing is running before a measured run.
4. **`--enable-cache-report` is required on the engines** for `--cache-report` to
   see `cached_tokens`; both launch scripts already pass it.
5. **kv events are on for prefill only.** Prefill-side prefix locality is the win;
   enabling them on the decode leg can make SGLang reject the speculative disagg
   flags. Set `KV_EVENTS=1` on the decode launch to override.
6. **Advertise the data-network IP.** `--advertise-host` / `SGLANG_HOST_IP` must be
   the address the peer node can reach, and `MC_GID_INDEX=3` with
   `--disaggregation-ib-device mlx5_0` keeps Mooncake on the RDMA rail.
