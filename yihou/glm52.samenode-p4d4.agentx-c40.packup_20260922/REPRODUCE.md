# REPRODUCE — same-node 1P1D P4D4, AgentX CONC=40

Cold start. Everything referenced is either inside this kit or an absolute path
called out explicitly.

## 0. What you need before you start

| requirement | value / where it comes from |
|---|---|
| a node | **one** machine with 8× MI355X (gfx950). This ran on `crsuse2-m2m-137` (`10.245.153.247`). **Do not use `crsuse2-m2m-276`** — see `environment.md`. |
| GPUs | all 8 free, at the ~284 MiB idle baseline, no KFD processes |
| image | `infera-sglang:v0519-yihou-0917-nextnfix-hicache`, present on the node |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (408 GB, shared NFS, read-only) |
| host RDMA lib | `/lib/x86_64-linux-gnu/libionic.so` on the host — `engine.sh` bind-mounts it; without it the container reports `Found 0 HCAs` |
| repo | this repo at the branch/SHA in `environment.md` |
| InferenceX | cloned automatically by `agentx_bench.sh` at the pinned ref; or point `INFERENCEX_DIR` at an existing clone |
| secrets | **none required.** No API keys, no registry login. Cluster access is plain ssh (or `spur exec` on spur nodes). HF is not contacted — the model is local. |

There is **no build step**. The image is used as-is.

## 1. Stage the harness

Copy `scripts/bench-harness/` (the patched harness) anywhere visible to the node
over NFS — `$HOME` works, and that is what this run used. Let `$H` be that path
and `$WS` its parent (the harness's `$DIR/../..`, which gets bind-mounted into
the AgentX client container).

The harness differs from the tracked repo in exactly the ways in `patches/`.
Do **not** substitute the repo's own `bench/glm5p2_pd/` — it cannot express this
topology.

## 2. Write the topology — two rows, one node

`scripts/topology.yihou.137.tsv`, tab-separated:

```
role	node	data_ip
prefill	crsuse2-m2m-137	10.245.153.247
decode	crsuse2-m2m-137	10.245.153.247
```

Prefill **must** be row 0 — every port is derived from the row index. Swap the
node/IP for your machine. The stock `tools/topology.py` rejects this file; the
vendored one accepts it.

Verify:

```bash
python3 $H/tools/topology.py rows scripts/topology.yihou.137.tsv
# 0  prefill-0  prefill  crsuse2-m2m-137  10.245.153.247
# 1  decode-0   decode   crsuse2-m2m-137  10.245.153.247
```

## 3. Confirm the node is clean

```bash
ssh crsuse2-m2m-137 'rocm-smi --showmeminfo vram | grep "Used Memory" \
  | awk "{s+=\$NF} END{printf \"total_MB=%.0f\n\", s/1048576}"; \
  rocm-smi --showpids | grep -c "^[0-9]"'
# expect total_MB≈2272 (8 × ~284 MiB) and 0 KFD processes
```

If a previous run just died, **wait**. A crashed HIP process can strand VRAM for
~20 s after the container is removed.

## 4. Bring the deployment up

```bash
cd $H
PATH="$WS/bin:$PATH" ./launch.sh \
  CONFIG=$WS/scripts/config.yihou.sn.p4d4.sh \
  TOPOLOGY=$WS/scripts/topology.yihou.137.tsv \
  REMOTE_BENCH_DIR=$H \
  CONTROL_NODE=crsuse2-m2m-137 BUILDER_NODE=crsuse2-m2m-137 \
  "PREFILL_EXTRA_ENV=MC_LOG_LEVEL=INFO" "DECODE_EXTRA_ENV=MC_LOG_LEVEL=INFO" \
  OUT_DIR=$WS/launch
```

`$WS/bin` on `PATH` is only needed for a **spur** node, where `ssh` is refused and
the shim re-dispatches through `spur exec`. On an ssh-reachable node it passes
straight through and is harmless.

Three things this depends on, all of them non-obvious:

- **`ENGINE_PORT_STRIDE=256`** (set in the config). Without it the two legs get
  engine ports 29001/29002, SGLang's derived internal blocks overlap, and prefill
  dies at `zmq.error.ZMQError: Address already in use`.
- **the same-node start gate** (in the patched `launch.sh`). It waits for the
  already-started leg on a node to answer `/health` before starting the next one.
  Without it the two legs' RCCL inits race and one dies with
  `NCCL error: unhandled cuda error`. Override with `SAME_NODE_START_GATE=off`
  only to reproduce the failure.
- **per-role `AITER_JIT_CACHE_ROOT`** (in the config). One host, two legs, one
  `/tmp` — they would otherwise share a JIT cache and can race on first compile.

Expect **~25-35 min**: ~4 min weight load per leg, then CUDA-graph capture
(decode only; prefill's graph is disabled by the resolved config), then warmup.

### Verify before spending 20 minutes on a benchmark

```bash
ssh crsuse2-m2m-137 'for p in 29001 29257; do printf "%s=" $p; \
  curl -fsS -o /dev/null -w "%{http_code}\n" http://10.245.153.247:$p/health; done; \
  curl -fsS -o /dev/null -w "router=%{http_code}\n" http://10.245.153.247:28000/health'
# all three must be 200
ssh crsuse2-m2m-137 'curl -fsS http://10.245.153.247:28000/v1/workers'
# must list TWO workers, both on the SAME ip, one prefill one decode
```

In each engine log, check `Found N HCAs` is **N≥1** per rank (it is one per rank
— that is correct, not a defect) and that `HIP transport installed for intra-node
GPU P2P` appears. Then a one-request smoke test:

```bash
ssh crsuse2-m2m-137 'curl -sS http://10.245.153.247:28000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"glm5.2-mxfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}"'
```

It must return a completion. **The text will be garbled** — that is simulated
acceptance working as configured, not a failure.

## 5. Run the benchmark

```bash
cd $H
PATH="$WS/bin:$PATH" ./agentx_bench.sh CONC=40 \
  CONFIG=$WS/scripts/config.yihou.sn.p4d4.sh \
  TOPOLOGY=$WS/scripts/topology.yihou.137.tsv \
  CONTROL_NODE=crsuse2-m2m-137 \
  INFERENCEX_DIR=<an existing InferenceX clone, or omit to let it clone> \
  OUT_DIR=$WS/agentx-c40
```

`config.sh` already sets `AGENTX_DURATION=1200` and
`AGENTX_WARMUP_REQUESTS_PER_LANE=1`, i.e. **running `config.sh` IS fast mode** —
`AIPERF_EXPERIMENTAL_FAST=1` is not plumbed through and is not needed.

Takes ~40 min wall (warmup + 1,200 s profiling + reporting). Success is the line
`AgentX passed: <OUT_DIR>/agentx_conc40.json`.

## 6. Collect the result

```bash
python3 -c 'import json;d=json.load(open("'"$WS"'/agentx-c40/agentx_conc40.json"));
m=d["request_metrics"];print("tput", m["throughput"]["total"]["tokens_per_second"]);
print("per_gpu", m["throughput"]["per_gpu"]["total_tput_tps"]);
print("ttft_p50", m["latency"]["ttft"]["p50"]);
print("profiled", d["request_accounting"]["records_profiled"])'

# spec_accept_length — take it while the deployment is still up
ssh crsuse2-m2m-137 'curl -fsS http://10.245.153.247:29257/metrics \
  | grep "^sglang:spec_accept"'
```

**Trust only ranks whose counters moved.** A rank that served no decode tokens
reports `spec_accept_length 0.0`. All four ranks moved in this run.

## 7. Tear down

```bash
cd $H
PATH="$WS/bin:$PATH" ./stop.sh \
  CONFIG=$WS/scripts/config.yihou.sn.p4d4.sh \
  TOPOLOGY=$WS/scripts/topology.yihou.137.tsv \
  CONTROL_NODE=crsuse2-m2m-137
```

Then **verify** the GPUs return to the idle baseline with no KFD processes
before launching anything else, and wait longer still.

## Success criteria, and what this run actually got

| criterion | result |
|---|---|
| `agentx_conc40.json` produced against a live router | **yes** |
| `docker ps` shows both legs on one node | **yes** — `glm52-pd-yihou-sn-p4d4-{prefill-0,decode-0}` on `crsuse2-m2m-137` |
| `spec_accept_length` reported | **yes** — 3.725 / 3.575 / 3.450 / 3.625, mean 3.59 |
| HiCache off (Phase A) | **yes** — both legs |
| error rate | **0 / 843 = 0.000 %**; 2 `InvalidInferenceResultError` dropped |

**Phase B (prefill HiCache on) was not reached** and is not part of this kit.
