# REPRODUCE

Order matters. Every step below has a verification, and each verification exists
because skipping it has silently produced a wrong result on this stack before.

## 0. Offline — validate the plan before holding a node

```bash
python3 -m agent.agent_throughput --mode preview \
  --workload-config workloads/caseA_probe.yaml \
  --model glm5.2-mxfp4 \
  --tokenizer /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
```

Confirms the tokenizer loads and the realized distributions match the spec triple.
Costs no GPU time; catches a bad tokenizer path before you are holding two nodes.

## 1. Hold two nodes, build the image on each

```bash
sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00   # x2
```

Health-gate each: `torch.cuda.is_available()` must be `True 8`. Spur has nodes that
enumerate 8 GPUs and report `False`.

Build on **both** nodes rather than moving the image. `docker save` of a 79 GB image
backgrounded inside `spur exec` dies at namespace teardown. `export
DOCKER_CONFIG=/tmp/dockercfg` before every docker call — docker 29's buildx plugin
discovery fails on the default path.

Stage 2 must print `kvaware+kvd self-check OK`.

## 2. Containers, kvd daemon, etcd

```bash
scripts/start_ctr.sh          # container + /shared_nfs mount + GPU gate
scripts/start_services.sh     # kvd daemon on /tmp/kvd/kvd.sock; etcd on prefill node
```

## 3. Apply both patches, in both containers, and prove they landed

```bash
python patches/patch_mooncake_early_send_wait_event.py
python patches/patch_hicache_rocm_host_alloc.py
find /sgl-workspace/sglang -name '*.pyc' -path '*disaggregation*' -delete
```

Then verify **bytecode**, not source:

```bash
strings .../pool_host/__pycache__/common.cpython-310.pyc | grep -c GLM52_ROCM_HOST_ALLOC   # want 1
```

This is not ceremony. A stale `__pycache__` entry silently running unpatched bytecode
has invalidated a full experiment in this repo. The ROCm patch deliberately contains a
real module-level string literal so the marker survives compilation — a comment would
not.

## 4. Boot both legs

```bash
scripts/boot.sh prefill 262144 1     # kvd ON
scripts/boot.sh decode  262144 0     # kvd OFF (operator instruction)
scripts/router.sh
```

Cold start is ~5-8 min (weights, then tilelang JIT + DP cudagraph capture). Eight live
`sglang::scheduler_DP*` processes means it is working, not hung.

## 5. Gate — all rows, before spending a measured window

| row | prefill | decode |
|---|---|---|
| `ready to roll` | 1 | 1 |
| `Memory access fault` | **0** | **0** |
| `Scheduler hit an exception` | 0 | 0 |
| `infera-kvd adapter connected` | 8 | 0 (by design) |
| `Attached hybrid DSA pool stack` | 8 | — |
| `Errno 98` **after** the ready line | 0 | 0 |
| `disaggregation_decode_enable_radix_cache=True` | — | 1 |
| `context_length=262144` | 1 | 1 |

The `Errno 98` row must be checked *after* the ready line: a port collision on
`--kv-snapshot-port` lets the leg log `ready to roll` and *then* die during etcd
registration, so it looks healthy and simply never appears in `/v1/workers`.

Server logs contain binary bytes — use `strings <log> | grep`, not plain `grep`.

Then confirm both workers are actually registered:

```bash
curl -s http://<prefill-ip>:8190/v1/workers
```

Never probe a leg's own port directly; it hangs. Always go through the router.

## 6. Correctness

```bash
docker exec agbench python3 scripts/correctness.py http://127.0.0.1:8190
```

Short factual 4/4 **and** needle 5/5 at ~120K tokens. The short prompts are one
prefill chunk and say nothing about the multi-chunk path the mooncake patch fixes;
the needle test is the one that covers the deployment being benchmarked.

## 7. Probe run — calibration only, not deliverable

```bash
bash scripts/run_bench.sh probe caseA_probe      # ramp 400 + sustain 600
```

Read off the steady-state live-session band and the in-flight range. Percentiles from
a 600 s window at ~0.6 qps are noise; do not quote them.

## 8. Re-solve the offered load

The closed form `rate = N / (E[turns] x (E2E + E[delay]))` needs one scalar E2E, and
the generation distribution is too skewed to supply one honestly (p50 212 tok vs p99
6,372 tok gives estimates 3.5x apart). Scale empirically from the probe's observed
population instead.

**Do not assume the response is linear.** It is not. Scaling 0.10 -> 0.145 on a
linear assumption predicted 26 in-flight and produced 44-48, pinning `max_inflight`
and forcing an abort. Step up conservatively from a known-stable point.

## 9. Case A full run

```bash
bash scripts/run_bench.sh full caseA_full        # ramp 400 + sustain 3600, ~67 min
```

`--dashboard-mode` is **mandatory** and is set inside `run_bench.sh`. All of
`summary.json`, `metrics.jsonl` and `metadata.json` are written inside
`if dashboard_mode and benchmark_name and data_dir:` (`agent_throughput.py:1674`).
Without the flag the run prints its report to stdout and persists nothing structured,
losing the percentiles and the session time series entirely.

Abort criteria, checked live:

* `In-flight` pinned at `max_inflight` -> backpressure is setting the load; the run is
  not the configured workload. Abort, lower the rate, re-run.
* live sessions climbing monotonically -> server saturated. Same response.

Snapshot kvd counters before and after:

```bash
docker exec agbench python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock
```

## 10. kvd read-back proof (separate experiment)

A latency win proves nothing — sglang's in-GPU radix cache serves a repeated prefix
without touching L3. The only clean attribution is restart-and-replay:

```bash
bash scripts/kill_engine.sh                      # engine only; kvd daemon SURVIVES
# poll rocm-smi until all 8 GPUs read VRAM 0%
scripts/boot.sh prefill 262144 1
docker exec agbench python3 scripts/replay_probe.py http://127.0.0.1:8190
```

Want `gets_total` and `hits_total` to climb while `sets_total` stays **flat**.

Do not restart the container to do this: it would repopulate nothing, and it would
silently drop both runtime patches from step 3.
