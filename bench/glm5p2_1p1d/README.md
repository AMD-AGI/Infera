# GLM-5.2 AgentX 1P1D on two MI355X nodes

This directory contains the reproducible v0.5.18 TP8/EP1 experiment and the
earlier v0.5.17 investigation. All scripts accept environment overrides from
`config.sh`; always keep one performance curve on one fixed node pair.

For a v0.5.18-only step-by-step operating guide, see
[`RUNBOOK_v518.md`](RUNBOOK_v518.md).
The completed C1/C2 live validation is recorded in
[`VALIDATION_v518_20260905_0053.zh-CN.md`](VALIDATION_v518_20260905_0053.zh-CN.md).

## Current v0.5.18 experiment

- Prefill: one MI355X node, TP8/EP1/DP1.
- Decode: one MI355X node, TP8/EP1/DP1 plus EAGLE MTP.
- Image base: `lmsysorg/sglang:v0.5.18-rocm720-mi35x`.
- SGLang fork: `xiaobochen-amd/sglang@402df1e1e453e1e85ec0f5ac4052d36598cc691a`.
- Aiter fork: `xiaobochen-amd/aiter@2c71811b32c8ce2e1266aedaec199df7d90f597d`.
- Transfer: Mooncake dma-buf over `ionic_0..7`, GID 1, with destination affinity.
- Ionic userspace provider: injected from the host to match the kernel ABI.
- AgentX: C8 for 3600 seconds.

The image build first reproduces `rocm-llm-bench/Dockerfile`, then uses that
image as `SGLANG_BASE_IMAGE` for Infera's v0.5.18 Dockerfile. The official
v0.5.18 upgrade and shared DSA/PD/Responses/ROCm patches come from current
Infera `main`; this kit adds only the pinned fork optimization layer,
fork-specific opt-in patch, and benchmark orchestration.

### Reproduce one run

```bash
cd /home/liyingli/bench_agentx/Infera/bench/glm5p2_1p1d

export RUN_ID=v518_ionic8_c8_$(date -u +%Y%m%d_%H%MZ)
export RUN_ROOT="$PWD/results/$RUN_ID"

# Inventory/select an idle pair, then build, preflight, correctness,
# C8/3600s AgentX, and analysis.
bash run_v518_workflow.sh
```

The workflow prefers exported `PREFILL_NODE`/`DECODE_NODE` values when they
pass inventory, but automatically selects replacement nodes from
`crsuse2-m2m-{135..142}` when either is unavailable. Override the candidate
pool with the space-separated `INVENTORY_NODES` environment variable. For the
dma-buf path, inventory excludes explicit IOMMU passthrough/disable modes
(`iommu=pt`, `iommu.passthrough=1`, `iommu=off`, and `amd_iommu=off/pt`)
because registrations can succeed while peer writes are silently dropped.

Fabric preflight is diagnostic evidence and never gates correctness or AgentX.
This kit defaults its Mooncake probe to producer-push RDMA WRITE, matching
SGLang `send_kvcache`, and verifies every byte on all eight destination GPUs in
both node directions. The public preflight default remains receiver-pull READ;
set the benchmark opcode explicitly when comparing the two directions.

Running correctness before performance is recommended, but the two workflows
are independent. `run_correctness.sh` covers RDMA/Mooncake preflight, Jupiter,
tool calls, an MTP burst, a 250K-token request and full GSM8K validation. It
explicitly disables `SGLANG_SIMULATE_ACC_LEN`.

The AgentX run performs a clean server start and sets
`--max-running-requests` plus `--cuda-graph-max-bs` to 8. Only the decode leg
receives `SGLANG_SIMULATE_ACC_LEN=3.61`. A point is valid only when InferenceX writes
the aggregate JSON and the failed-request ratio is at most 10%.
Hugging Face trace data and the AIPerf runtime cache are shared under
`$WORKSPACE_ROOT/.cache/agentx`, so retries, new RUN_IDs, and dynamically
selected nodes reuse the first successful download. Point outputs remain
isolated under their RUN_ID.

Artifacts live under `results/<RUN_ID>/`. Each point records the exact nodes,
image IDs, `/server_info`, workers, AIPerf command and records, three metrics
endpoints, GPU samples and all service logs. `analyze_agentx.sh` copies only
points with a `PASS` marker into the plotted curve.

## Legacy v0.5.17 experiment

- Prefill: `crsuse2-m2m-137`, TP8 + EP8, DP-attention off.
- Decode: `crsuse2-m2m-138`, TP8 + EP8 + DP-attention DP8.
- Model: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.
- Router and etcd run on the prefill node.
- Safe transport baseline: Mooncake dma-buf over `mlx5_0`, GID 3.

All values can be overridden through environment variables in `config.sh`.

### Legacy run

```bash
# 1. Build deploy/docker/Dockerfile.sglang as BASELINE_IMAGE on both nodes.

# 2. Build cumulative A/B layers on each node from this directory.
docker build -f Dockerfile.correctness \
  -t infera/engine-sglang:glm52-1p1d-8f04a68-corr-1 .
docker build -f Dockerfile.runtime \
  -t infera/engine-sglang:glm52-1p1d-8f04a68-runtime-1 .
docker build -f Dockerfile.optimized \
  -t infera/engine-sglang:glm52-1p1d-8f04a68-opt-1 .

# 3. Check each node and measure the cross-node fabric.
bash preflight.sh all

# 4. Launch the accepted runtime-1 default, then verify it.
bash launch.sh
bash eval/smoke.sh

# 5. Full synthetic concurrency sweep.
bash bench.sh synthetic 8 16 32 64 128

# 6. Validate the large-buffer fallback near the context limit while idle.
bash eval/long_context.sh

# 7. One-hour AgentX-MVP run at concurrency 14.
bash bench.sh agentic

# 8. Stop only this experiment's containers.
bash stop.sh
```

Artifacts are written under `results/`. The interpreted records are
`REPORT.md` and `REPORT.zh-CN.md`.

### Legacy version-specific AgentX choices

The pinned SGLang v0.5.17 ROCm EAGLE path verifies non-greedy requests with
argmax unless the still-unmerged
[SGLang #37134](https://github.com/sgl-project/sglang/pull/37134)
rejection-sampling stack is backported. Both benchmark modes therefore request
`temperature=0`, matching the API contract to the path that actually runs.

AIPerf 0.12 synthesizes one AgentX warmup request per concurrency lane. The old
`--warmup-requests-per-lane` flag no longer exists, and the
`inferencex-agentx-mvp` scenario now rejects `--trace-idle-gap-cap-seconds`.
`bench.sh` intentionally omits both options, gives requests a 10-minute drain
window after the one-hour measurement, and uses a 30-minute per-request
timeout. The finite drain is important in AIPerf 0.12: a future DAG join can
otherwise survive the sending cutoff with no HTTP request on the wire, making
an infinite grace period wait forever even after every response was exported.

### Why the transport baseline uses mlx5_0

Both selected nodes have eight 400 Gb/s ionic rails and one 200 Gb/s mlx5
interface. Peer-memory is not loaded. The ionic provider exposes dma-buf but not
ODP, so uncapped registration can pin and duplicate the GPU KV pool. `mlx5_0`
does expose ODP and is therefore the safe no-pin baseline.

The faster ionic mode remains an optimization experiment, not a default. It
must set an explicit model-specific `--max-total-tokens` cap and leave enough
VRAM for weights plus twice the registered KV pool.

### Legacy image policy

The baseline image is built from the checked-out
`deploy/docker/Dockerfile.sglang`, including its pinned SGLang, Mooncake,
GLM-5.2 DSA/PD patches and ROCm HiCache fixes. Images that happened to exist on
the nodes were used only to probe hardware capabilities.

Upstream changes are cumulative so each A/B adds one attributable group:

- `Dockerfile.correctness`: SGLang #37133 only (fp32 GLM correction bias).
- `Dockerfile.runtime`: #37133 plus #37118 (HIP graph helper registration).
- `Dockerfile.optimized`: the runtime layer plus the dependency-free Aiter
  #5121 offset gate; retained as an experimental arm, not the default.

All three are thin layers over the locally built baseline and use
`bench/glm5p2_1p1d/` as their Docker build context. The deployed decode leg
captures EAGLE target-verify and draft graphs even though prefill graphs are
disabled, which is why the #37118 runtime layer remains in the accepted image.
The pinned Aiter already handles large buffers through its global-load/store
fallback; a 250,016-token request passed without #5121, so that patch is not
required for this stack.

## Multi-P/D migration

Keep this accepted 1P1D tree and its historical results unchanged. New
profile-driven 1P1D parity and multi-P/D studies use
[`../glm5p2_pd/README.md`](../glm5p2_pd/README.md); the no-copy parity process
is documented in
[`../glm5p2_pd/MIGRATION.md`](../glm5p2_pd/MIGRATION.md).
