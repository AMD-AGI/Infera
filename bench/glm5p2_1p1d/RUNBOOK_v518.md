# GLM-5.2 v0.5.18 1P1D reproduction guide

This guide covers only the v0.5.18 GLM-5.2-MXFP4 deployment on two MI355X
nodes: one prefill node, one decode node, Mooncake KV transfer, and the Infera
router. The scripts use environment overrides from `config.sh`; no script
editing is required for a normal run.

## 1. Prerequisites and node selection

- The control host can SSH to both GPU nodes without an interactive password.
- Docker, ROCm devices and `rocm-smi` are available on both nodes.
- The model path is readable at the same absolute path on both nodes.
- This workspace is available at the same absolute path on both nodes, because
  fabric, evaluation, and AgentX containers bind-mount it.
- The prefill and decode nodes must remain fixed for a complete comparison
  curve.

Run all commands from:

```bash
cd /home/liyingli/bench_agentx/Infera/bench/glm5p2_1p1d
```

Export one run identity before inventory or deployment. Without this, each
new shell may generate a different timestamp:

```bash
export RUN_ID=v518_ionic8_c8_$(date -u +%Y%m%d_%H%MZ)
export RESULTS_DIR=/home/liyingli/bench_agentx/Infera/bench/glm5p2_1p1d/results
export RUN_ROOT="$RESULTS_DIR/$RUN_ID"
```

Inspect the default candidate pool. This discovery step is intentionally
independent of `config.sh`, because the deployment pair has not been selected
yet. It writes both the evidence table and an automatically selected pair:

```bash
bash inventory_nodes.sh
```

Specific candidates can be supplied as positional arguments. Every artifact
for this experiment is kept below `results/$RUN_ID/`; inventory writes to
`results/$RUN_ID/inventory/`. For an individual stage, load the selected pair
and site overrides before `config.sh`:

```bash
# Override these before sourcing config when the site differs.
export MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
export DATA_NET=10.245.
export RDMA_DEVICE=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export MC_TE_FILTERS="$RDMA_DEVICE"
export MC_GID_INDEX=1
export MOONCAKE_DISABLE_HIP_DMABUF=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export HOST_RDMA_LIB=/lib/x86_64-linux-gnu/libionic.so
export HOST_RDMA_MOUNT=/host-libionic/libionic.so

source "$RUN_ROOT/inventory/selected.env"
source ./config.sh
```

`config.sh` requires two distinct, explicitly selected nodes and resolves
their data-plane IPs when sourced.
The end-to-end workflow runs inventory before loading `config.sh`; explicit
`PREFILL_NODE`/`DECODE_NODE` values are preferences, not hard requirements. If
either is busy, unhealthy, short on disk, or missing an ionic rail, the
workflow selects the next two suitable nodes. Nodes with explicit IOMMU
passthrough/disable options (`iommu=pt`, `iommu.passthrough=1`, `iommu=off`,
or `amd_iommu=off/pt`) are excluded: dma-buf registration can return success
while peer DMA silently drops writes. A resumed runtime stage may start late
only when its pair, image, source, and effective-config gate files prove every
production prerequisite on the exact current pair; otherwise it restarts at
build. Fabric preflight is diagnostic and is not one of those resume gates.

For the canonical end-to-end run, execute:

```bash
bash run_v518_workflow.sh
```

It performs cleanup, selected-node inventory, image build/distribution,
diagnostic preflight, correctness, C8 AgentX for 3600 seconds, and analysis.
A preflight failure is recorded with its raw exit code and the workflow
continues; correctness and AgentX determine their own outcomes. If a stage is
fixed after a failure, resume from it without repeating earlier stages, for
example:

```bash
WORKFLOW_START_AT=preflight bash run_v518_workflow.sh
```

Changing READ/WRITE or GPU-count diagnostic settings does not invalidate the
correctness/AgentX config gate. Run focused diagnostics directly with
`preflight.sh`; inventory and smoke retain the cluster's strict 8-rail/8-GPU
requirements.

The following sections describe the same stages individually.

## 2. Build the image

Build the pinned SGLang/AITER image on the prefill node, build the Infera
layer, copy the exact image to the decode node, and verify matching image IDs:

```bash
bash build_v518_image.sh all
```

The build can also be resumed by stage:

```bash
bash build_v518_image.sh build
bash build_v518_image.sh distribute
bash build_v518_image.sh verify
```

Build artifacts are written to `results/$RUN_ID/build/`.

`rocm-llm-bench/Dockerfile` pins the xiaobochen SGLang/AITER forks at
`402df1e` / `2c71811`. Current Infera `main` supplies the official v0.5.18
upgrade and shared patches; this benchmark build opts in only to the
fork-specific rejection-sampling guard.

## 3. Validate the ionic fabric

Run the node-local registration checks and the two-node fabric checks:

```bash
bash preflight.sh all
```

The netperf stage keeps the public probe's full selected-device matrix. On this
rail-isolated fabric, expected cross-rail failures remain diagnostic evidence
and do not justify narrowing the shared probe. Mooncake registers and transfers
a real GPU buffer on every GPU. This kit explicitly defaults to producer-push
RDMA WRITE, matching SGLang prefill `send_kvcache`; set
`INFERA_PREFLIGHT_MOONCAKE_OPCODE=read` to compare receiver-pull behavior. The
public preflight default remains READ. Artifacts and the raw exit status are
written to `results/$RUN_ID/preflight/`.

## 4. Start and stop the 1P1D service

The default `SIMULATE_ACC_LEN=3.61` is for deterministic performance runs.
Correctness checks must start the service with an explicitly empty value so
that SGLang uses real draft acceptance:

```bash
SIMULATE_ACC_LEN= bash launch.sh
```

The router endpoint is:

```text
http://<PREFILL_IP>:8000
```

Inspect the deployment:

```bash
curl -fsS "http://$PREFILL_IP:$ROUTER_PORT/health"
curl -fsS "http://$PREFILL_IP:$ROUTER_PORT/v1/workers"
ssh "$PREFILL_NODE" docker logs -f glm52-pd-prefill
ssh "$DECODE_NODE" docker logs -f glm52-pd-decode
ssh "$PREFILL_NODE" docker logs -f glm52-pd-router
```

Stop only this experiment's containers:

```bash
bash stop.sh
```

Complete correctness and AgentX runs hold a node-pair lock. To intentionally
abort an active run from another shell, use `FORCE_STOP=1 bash stop.sh`.
`stop.sh` reports residual HBM; the next `launch.sh` enforces the idle limit.

## 5. Correctness

### 5.1 Fixed complete sequence

The top-level script always performs the same sequence:

1. node-local RDMA mode preflight;
2. launch with real MTP acceptance;
3. smoke;
4. long-context;
5. GSM8K;
6. stop.

Run it with no arguments:

```bash
bash run_correctness.sh
```

Each module owns its own result and exit status. There is no shared PASS file
and no correctness gate in the AgentX sweep. If a module fails, the sequence
exits immediately and leaves the deployment running for inspection. Stop it
manually after debugging.

### 5.2 Run modules independently

Start one real-acceptance deployment, run only the checks needed, then stop it:

```bash
SIMULATE_ACC_LEN= bash launch.sh

bash eval/smoke.sh
bash eval/long_context.sh
bash eval/gsm8k.sh

bash stop.sh
```

The modules do not read each other's artifacts:

- `eval/smoke.sh` checks router/worker pairing, one deterministic Jupiter response,
  one required tool call, MTP acceptance evidence, and fatal PD/RDMA/GPU log
  patterns.
- `eval/long_context.sh` flushes both leg caches, sends a 250K-token prompt through
  the router, and verifies the server-reported prompt token count.
- `eval/gsm8k.sh` independently flushes both leg caches, runs InferenceX
  `run_lm_eval`, copies the remote artifacts back, and validates GSM8K
  `exact_match` against the configured threshold.

For a quick GSM8K diagnostic subset:

```bash
CORRECTNESS_EVAL_LIMIT=20 bash eval/gsm8k.sh
```

The default threshold for `glm5.2` is the global GSM8K minimum of `0.90`
because `thresholds.yaml` has no model-specific `glm5.2` override.

### 5.3 Correctness artifacts

```text
results/$RUN_ID/correctness/
├── preflight.log
├── launch.log
├── smoke.log
├── smoke/
│   ├── run.log
│   ├── workers.json
│   ├── jupiter.json
│   ├── tool_call.json
│   ├── prefill.log
│   ├── decode.log
│   ├── gpu_prefill.log
│   ├── gpu_decode.log
│   ├── rdma_prefill.log
│   ├── rdma_decode.log
│   ├── prefill_container.json
│   └── decode_container.json
├── long-context.log
├── long-context/
│   ├── run.log
│   ├── response.json
│   └── summary.json
├── gsm8k.log
├── gsm8k/
│   ├── run.log
│   ├── runner.log
│   ├── client.env
│   └── results*.json
└── stop.log
```

When a module is invoked directly, it uses the same module directory under
`results/$RUN_ID/correctness/`.

## 6. AgentX performance

Correctness and performance are intentionally independent. Running the
correctness modules first is recommended, but `run_agentx_sweep.sh` does not
read their output or enforce a gate.

Run the single required concurrency point. The default is C8 for 3600 seconds:

```bash
bash run_agentx_sweep.sh
```

With no positional concurrencies, the script uses
`AGENTX_CONCURRENCIES` from `config.sh`.

To run one point manually:

```bash
export CONC=8

SIMULATE_ACC_LEN=3.61 \
MAX_RUNNING_REQUESTS="$CONC" \
CUDA_GRAPH_MAX_BS="$CONC" \
    bash launch.sh

SIMULATE_ACC_LEN=3.61 \
MAX_RUNNING_REQUESTS="$CONC" \
CUDA_GRAPH_MAX_BS="$CONC" \
    bash agentx_point.sh "$CONC"

bash stop.sh
```

Each successful point writes its own `agentx_conc<N>.json` and point-local
`PASS` marker under `results/$RUN_ID/agentx/c<N>/`. A matching validation
signature records the pair, image digest, source/config identities, topology,
routing mode, and duration. A stale or missing signature archives and reruns
the point instead of reusing its PASS. The marker belongs only to the
performance sweep's retry/plot workflow; it is not a correctness gate.

HF trace data and the AIPerf runtime cache live in the shared workspace at
`$WORKSPACE_ROOT/.cache/agentx`. They are reused across retries, RUN_IDs, and
dynamically selected nodes; result files remain RUN_ID-local.

## 7. Analyze the result

After one or more AgentX points complete:

```bash
bash analyze_agentx.sh "$RUN_ROOT"
```

Outputs:

```text
results/$RUN_ID/analysis/curve/results.csv
results/$RUN_ID/analysis/curve/pareto.png
```

## 8. Unified artifact layout

```text
results/$RUN_ID/
├── inventory/
├── build/
├── preflight/
├── correctness/
├── agentx/
│   └── c8/
├── analysis/
└── validation-report.zh-CN.md
```

`RESULTS_DIR` selects the parent directory and `RUN_ROOT` defaults to
`$RESULTS_DIR/$RUN_ID`. Override either before sourcing `config.sh`.

## 9. Important defaults

- Image: `infera/engine-sglang:glm52-v518-402df1e-2c71811`
- Model: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`
- Topology: prefill TP8/EP1/DP1 and decode TP8/EP1/DP1
- DP attention: disabled on both legs
- Context length: model-native 1,048,576; no explicit server cap
- Chunked prefill: 32,768
- Static memory fraction: 0.85 on both legs
- MTP: 5 steps, 6 draft tokens
- Simulated acceptance: 3.61 by default; empty for correctness
- Maximum running requests / CUDA graph batch: 32 by default
- HiCache: enabled on prefill, disabled on decode
- RDMA devices: `ionic_0..7`, GID index 1
- Mooncake dma-buf: enabled; same-name destination-HCA affinity enabled
- Host ionic provider: mounted at `/host-libionic/libionic.so` so its ABI
  matches the host kernel driver
- Container `nofile` limit: `65536:65536`
- AgentX concurrency: 8
- AgentX duration: 3600 seconds

See `config.sh` for the complete override list.

## 10. Troubleshooting

If a correctness module fails, inspect its `run.log` first, then the leg logs
captured by `eval/smoke.sh`. The fixed sequence deliberately leaves the
deployment running on failure:

```bash
ssh "$PREFILL_NODE" docker logs glm52-pd-prefill
ssh "$DECODE_NODE" docker logs glm52-pd-decode
ssh "$PREFILL_NODE" docker logs glm52-pd-router
```

If generated text is wrong, confirm the server was launched with:

```bash
SIMULATE_ACC_LEN= bash launch.sh
```

If `ibv_devinfo` reports that the ionic driver does not support the kernel ABI,
verify both `HOST_RDMA_LIB` and `HOST_RDMA_MOUNT`. Omitting this mount makes the
container use its bundled, ABI-incompatible provider even though the host RDMA
devices themselves are healthy.

If launch waits for GPU memory, inspect active processes and clean up only the
containers belonging to this experiment:

```bash
ssh "$PREFILL_NODE" rocm-smi --showpids
ssh "$DECODE_NODE" rocm-smi --showpids
bash stop.sh
```

If another workflow still owns the node-pair lock, either let it finish or
explicitly abort it with `FORCE_STOP=1 bash stop.sh`.

If GSM8K fails after producing results, inspect:

```text
results/$RUN_ID/correctness/gsm8k/runner.log
results/$RUN_ID/correctness/gsm8k/results*.json
```
