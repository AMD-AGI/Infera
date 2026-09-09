# GLM-5.2 multi-P/D

Hand-operated deployment and AgentX scripts for one or more Prefill and Decode
workers. There is no study runner, resume state, service contract, node lock, or
statistics layer. Run each step separately and fix a failed step before moving
on.

The accepted `../glm5p2_1p1d` kit and all of its results remain unchanged.

## 1. Configure

```bash
cp config.sh.example config.sh
cp topology.tsv.example topology.tsv
```

`config.sh` has three groups of values:

- shared paths, ports, image, RDMA, and router settings;
- `PREFILL_*` worker settings;
- `DECODE_*` worker settings.

`topology.tsv` contains only the role, SSH node, and data-plane IP:

```text
role	node	data_ip
prefill	worker-01	10.245.0.11
prefill	worker-02	10.245.0.12
decode	worker-03	10.245.0.13
```

Each node may appear once. Worker ports and names are derived from row order and
printed before launch. `CONTROL_NODE` must be one of the listed nodes.

Every command accepts temporary `VAR=value` overrides:

```bash
./launch.sh DECODE_MAX_RUNNING=64 DECODE_SIMULATE_ACC_LEN=
./agentx_bench.sh DECODE_MAX_RUNNING=64 CONC=8 DURATION=3600
```

When AgentX checks a service started with temporary deployment overrides, pass
the same overrides again so the expected and live settings can be compared.
For settings used repeatedly, edit `config.sh` instead.

Use `CONFIG=/path/config.sh` and `TOPOLOGY=/path/topology.tsv` to select other
files. The repository, config, topology, model, and output paths must be visible
at the same absolute paths on all selected nodes.

## 2. Check nodes

```bash
./check_nodes.sh
```

This checks SSH, GPU count, selected-GPU VRAM, and existing containers owned by
`CONTAINER_PREFIX`. `launch.sh` repeats the same check immediately before use.

There is deliberately no node lock. Two controllers can both pass the idle
check at the same time, so do not launch overlapping experiments manually.

## 3. Build and distribute the image

The default `all` mode reproduces the accepted two-stage v0.5.18 build:

```bash
./build_image.sh
```

It first builds `ROCM_LLM_BENCH_DIR/Dockerfile`, which pins the optimized
SGLang and AITER commits, as `ROCM_OPT_BASE_IMAGE`. It then builds Infera with
that exact image passed as `SGLANG_BASE_IMAGE`, distributes the final `IMAGE`,
and verifies its imports, SGLang/AITER commits, router binary, and Image ID on
every topology node.

Each operation can also be run separately:

```bash
./build_image.sh build
./build_image.sh distribute
./build_image.sh verify
```

Build and launch logs print the image reference and each node's actual Image ID
for later comparison. Launch does not compare against a historical record and
does not block a changed Image ID.

If the configured final image already exists on every node, skip building and
run only `./build_image.sh verify`.

## 4. Check the P→D fabric

For a new cluster, image, or RDMA configuration:

```bash
./preflight.sh
```

This runs a byte-verified Mooncake WRITE for every Prefill→Decode pair. It does
not run the older all-node READ/netperf diagnostic matrix. Preflight is an
independent command; no later script reads its result.

## 5. Launch

```bash
./launch.sh
```

Launch performs only:

```text
node idle check
  → etcd
  → all Prefill and Decode workers
  → worker health
  → router
  → router health and worker discovery
```

It leaves the service running. It does not start correctness or AgentX.
Failures leave containers and logs available for debugging. Inspect them with:

```bash
ssh NODE docker logs CONTAINER
ssh NODE docker inspect CONTAINER
```

The command prints the exact node, role, port, container, and log locations.

## 6. Correctness

Simulated MTP acceptance can produce incorrect text. Relaunch without it before
correctness:

```bash
./stop.sh
./launch.sh DECODE_SIMULATE_ACC_LEN=

./eval/smoke.sh OUT_DIR=results/smoke
./eval/gsm8k.sh OUT_DIR=results/gsm8k
./eval/long_context.sh OUT_DIR=results/long-context TOKENS=250000
```

- `smoke.sh` checks health, P/D discovery, factual chat, tool calls, and a small
  concurrent burst.
- `gsm8k.sh` runs the official InferenceX task in a host-network container on
  the control node. It refuses a live Decode worker with simulated acceptance.
- `long_context.sh` sends one direct long-context OpenAI request.

Each output directory must be empty so reruns cannot mix artifacts.

## 7. AgentX

For a performance point, launch the desired service configuration and run:

```bash
./agentx_bench.sh \
    CONC=8 \
    DURATION=3600 \
    OUT_DIR=results/agentx-c8
```

These are the only normal AgentX inputs. Before replay, the script obtains the
following from the live router, workers, containers, and hosts:

- served model and P/D worker URLs/counts;
- TP, EP, DP, DPA, max-running, HiCache, MTP, and simulated acceptance;
- metrics URLs, GPU model, CPU DRAM, and actual container Image IDs.

It compares live worker settings with `config.sh` and stops on a mismatch. The
complete InferenceX environment is saved as `OUT_DIR/runtime.env`; raw
`/v1/workers`, server-info, container-inspect, and hardware snapshots are under
`OUT_DIR/service/`. The environment also records the InferenceX commit and
whether its tree is dirty. These are readable run records, not input
configuration.

The GLM-5.2-specific metadata is intentionally fixed in the adapter:

```text
MODEL_PREFIX=glm5.2
FRAMEWORK=sglang
PRECISION=fp4
IS_MULTINODE=true
DISAGG=true
```

## 8. Analyze and plot

Put manually-run concurrency points under one directory, then analyze that
directory:

```bash
./agentx_bench.sh CONC=1 OUT_DIR=results/sweep/c1
./agentx_bench.sh CONC=2 OUT_DIR=results/sweep/c2
./agentx_bench.sh CONC=4 OUT_DIR=results/sweep/c4

./analyze_agentx.sh RESULT_DIR=results/sweep
```

This writes `results/sweep/results.csv` and `results/sweep/pareto.png`.
Throughput per chip is divided by
`num_prefill_gpu + num_decode_gpu`, so arbitrary 1P1D, 2P2D, and asymmetric
P/D topologies are handled correctly. Different P/D worker, TP, EP, and DPA
shapes are drawn as separate curves.

The plot includes the official InferenceX GLM-5.2 reference curves bundled at
`tools/ref/InferenceX_GLM-5.2_interactivity.csv`. The current snapshot was
retrieved from the public InferenceX API on 2026-09-09; it contains 58 points
through benchmark date 2026-09-03. Refresh it explicitly when needed:

```bash
python3 tools/update_inferencex_ref.py
# Or refresh immediately before plotting:
./analyze_agentx.sh RESULT_DIR=results/sweep UPDATE_REFERENCE=1
```

Normal plotting is offline and never changes the reference snapshot.

## 9. Stop

```bash
./stop.sh
```

Stop derives exact container names from config and topology, gracefully stops
router/workers/etcd, and then prints residual GPU state. The default timeout is
300 seconds because interrupting HiCache teardown can leave host memory pinned.
If graceful stop fails, the script leaves the container for manual inspection
instead of immediately forcing removal.

## Files

The normal entry points are:

- `check_nodes.sh`
- `build_image.sh`
- `preflight.sh`
- `launch.sh` and its remote worker helper `engine.sh`
- `eval/smoke.sh`, `eval/gsm8k.sh`, `eval/long_context.sh`
- `agentx_bench.sh`
- `analyze_agentx.sh`
- `stop.sh`

Implementation helpers are limited to `lib/common.sh`,
`tools/wait_healthy.py`, `tools/validate_preflight.py`, and
`tools/agentx_env.py` (used only by AgentX). Analysis uses
`tools/collect_agentx.py`, `tools/plot_agentx.py`, and the optional explicit
reference updater.

## Local checks

```bash
python3 -m unittest discover -s tests -v
bash -n ./*.sh ./eval/*.sh ./lib/*.sh
```

No old `results` or `report` directory is copied into this kit. New commands
write only to the output or log directory shown on screen.
