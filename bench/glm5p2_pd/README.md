# GLM-5.2 multi-P/D

Hand-operated deployment and AgentX scripts for one or more Prefill and Decode
workers. Deployment and hardware measurement have no study runner, service
contract, or node lock: run each step separately and fix a failed step before
moving on. The offline projection sweep below is the only resumable study
driver in this directory.

The accepted `../glm5p2_1p1d` kit and all of its results remain unchanged.

## Offline 1P1D projection sweep

Projection is deliberately split into two independent stages:

- `projection_sweep.py` creates the schedule, runs InferaSim, and writes only
  append-only JSONL records plus one human-readable CLI report per attempt;
- `analyze_projection_sweep.py` reads those records and writes derived CSV,
  Pareto, shortlist, and summary artifacts to a separate directory.

Run the scanner in the same Python environment as a working `inferasim`
installation with the Origami backend:

```bash
RUN_DIR=results/projection-1p1d

./projection_sweep.py --dry-run
./projection_sweep.py --output-dir "$RUN_DIR"
```

The raw run directory is the stable experiment record:

```text
RUN_DIR/
  run_config.json
  raw/
    schedule.jsonl
    projections.jsonl
    reports/                  # one human-readable .txt report per attempt
```

`schedule.jsonl` contains every intended call before execution starts.
`projections.jsonl` is append-only and stores the full native performance,
per-pool memory, resolved configuration, CLI arguments, status, and timing for
every attempt. `reports/` retains the corresponding original InferaSim output,
reproduction command, and Prefill/Decode memory limits. Thus both machines and
people can inspect the raw scan without rerunning InferaSim.

Resume or retry failures without rewriting completed raw records:

```bash
./projection_sweep.py --output-dir "$RUN_DIR" --resume

# Add this flag to the same command to append a new attempt for failed points:
#   --retry-errors
```

Analyze the same raw records repeatedly without changing them:

```bash
./analyze_projection_sweep.py "$RUN_DIR" \
  --output-dir "$RUN_DIR/analysis/baseline"

./analyze_projection_sweep.py "$RUN_DIR" \
  --interactivity-floor 144 \
  --output-dir "$RUN_DIR/analysis/interactivity-144"
```

Analysis may run while a scan is incomplete. Scheduled but unfinished
points are marked `pending`; completed points remain available for intermediate
inspection. Each analysis directory contains `results.csv`, `feasible.csv`,
global and per-budget Pareto CSVs, `shortlist.csv`, `summary.txt`, and an
`analysis_config.json` recording the exact raw-data hashes. It also writes
`memory_limits_by_combination.csv` with each P/D combination's effective
memory max concurrency and `memory_point_actions.csv` with explicit `CUT`,
and `SUPPLEMENT` points. For a memory boundary above C64, supplemental points
are selected from C96/C128/C192/C256/C512 below that boundary, followed by the
exact max-concurrency boundary.

Apply a reviewed memory action table with `apply_memory_followup.py`. It stages
and runs all supplemental projections before deleting any Cut point, then
atomically rewrites the JSONL files, removes Cut reports, archives the applied
plan under `RUN_DIR/memory-plan-applied`, and discards stale derived analysis.

The default matrix has 448 strategy/concurrency configurations:

- exact fleet budgets `8,12,16` GPUs;
- pool widths `4,8`, producing `P4+D4`, `P4+D8`, `P8+D4`, and `P8+D8`;
- independent Prefill and Decode modes `TP`, `TP+EP`, `TP+DPA`, and
  `TP+EP+DPA`;
- `EP=TP` and `attention-DP=TP` whenever the corresponding mode is enabled;
- concurrency `1,2,4,8,16,32,64`;
- fixed workload `ISL=111787`, `OSL=911`, prefix hit `0.97369`;
- Mooncake KV-transfer bandwidth `37.85 GB/s` per TP rank, measured from the
  TP4/EP4 C8 run;
- speculative draft cost factor `0.05`;
- speculative `k=5` drafted tokens at acceptance `0.79067`, so a verify step
  spans `k+1=6` query positions and emits `sum(a^0..a^5)=3.610` tokens --
  matching the engine, which is launched with `--speculative-num-steps 5
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6` and pinned to
  `SGLANG_SIMULATE_ACC_LEN=3.61`. SGLang's `num-draft-tokens` counts the whole
  verify tree (root + drafted), which is why `k` is 5 and not 6.

  The `results/projection-*` directories predate that correction: they were
  produced at `k=6` / acceptance `0.765763`, which is the pair that yields the
  same 3.610 tokens per step but over a 7-position verify step the engine never
  runs. Each directory's `run_config.json` records the pair it used. The error
  is confined to decode step cost (roughly the 7/6 ratio, plus a slightly
  different draft term); it does not touch the memory model, so the
  feasibility and `max_conc` tables built from those runs still hold.

This first pass intentionally uses only that single representative workload;
there is no profile bucketing or cross-workload aggregation. The matrix is
therefore exactly 448 raw projections. TP1 is HBM-infeasible. TP2 reaches only
11 projected resident sequences for this workload, so the production scan
accepts only TP4/TP8. Attention-DP subdivides TP and does not add GPUs.

The Pareto x-axis is InferaSim's modeled mean interactivity, not P90. This
script deliberately does not apply the TP4/EP4 AgentX correction or a shared
GPU Anchor to other TP/EP regimes. Calibrate each shortlisted regime
independently, then obtain P90 from its real AgentX run.

### TP8+DPA replica scan above 16 GPUs

Keep each worker at TP8/EP1/attention-DP8 and scale with P/D replicas rather
than creating a cross-node TP16 worker. Inspect the proposed 26-point 24/32-GPU
schedule with:

```bash
./projection_sweep.py --dry-run \
  --budgets 24,32 --pool-widths 8 --modes tp_dpa \
  --replica-plan \
    '1x2:32,64,96,128,168;2x1:32,64,96,128,168;1x3:32,64,96,128,168;2x2:64,96,128,192,256,336;3x1:32,64,96,128,168'
```

For global concurrency `C`, memory is projected at
`ceil(C / prefill_replicas)` and `ceil(C / decode_replicas)` on each worker.
The analyzer reports both per-replica and fleet max concurrency. Remove
`--dry-run` and add a new `--output-dir` only when the schedule is approved.

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
- `projection_sweep.py`
- `analyze_projection_sweep.py`
- `stop.sh`

Implementation helpers are limited to `lib/common.sh`,
`tools/wait_healthy.py`, `tools/validate_preflight.py`, and
`tools/agentx_env.py` (used only by AgentX). AgentX analysis uses
`tools/collect_agentx.py`, `tools/plot_agentx.py`, and the optional explicit
reference updater. Projection analysis is isolated in
`analyze_projection_sweep.py` and reads only the raw run contract.

## Local checks

```bash
python3 -m unittest discover -s tests -v
bash -n ./*.sh ./eval/*.sh ./lib/*.sh
```

No old `results` or `report` directory is copied into this kit. New commands
write only to the output or log directory shown on screen.
