# REPRODUCE — same-node P4D4, triton DSA + index_share ON, CONC=40 full

Cold start. Everything referenced is inside this kit or an absolute path called
out explicitly.

## 0. Prerequisites

| requirement | value |
|---|---|
| a node | **one** machine, 8× MI355X (gfx950). This ran on `crsuse2-m2m-137` (`10.245.153.247`), plain `ssh`. **Do not use `crsuse2-m2m-276`** — it segfaults this workload; see the 20260922 packup. |
| GPUs | all 8 free at the ~284 MiB idle baseline, no KFD processes |
| host RAM | ≥ 1 TB free. Prefill allocates **two** host regions per rank: 153.81 GB HiCache pool + 35.25 GB DSA indexer ⇒ ~756 GB total across 4 ranks |
| image | `infera-sglang:v0519-yihou-0917-nextnfix-hicache`, present on the node |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (408 GB, shared NFS, read-only) |
| host RDMA lib | `/lib/x86_64-linux-gnu/libionic.so` — `engine.sh` bind-mounts it; without it the container reports `Found 0 HCAs` |
| InferenceX | cloned by `agentx_bench.sh` at the pinned ref, or point `INFERENCEX_DIR` at an existing clone |
| secrets | **none.** No API keys, no registry login. HF is not contacted — the model is local. |

No build step; the image is used as-is.

## 1. Stage the harness

Copy `scripts/bench-harness/` (the **patched** harness) somewhere visible to the
node over NFS — `$HOME` works. Let `$H` be that path and `$WS` its parent (which
`agentx_bench.sh` bind-mounts into the client container).

It differs from the tracked repo in exactly three files, all in `patches/`. Do
**not** substitute the repo's own `bench/glm5p2_pd/` — it cannot express this
topology.

## 2. Topology — two rows, one node

`scripts/topology.yihou.137.tsv`, tab-separated, **prefill first** (every port
derives from the row index):

```
role	node	data_ip
prefill	crsuse2-m2m-137	10.245.153.247
decode	crsuse2-m2m-137	10.245.153.247
```

Verify: `python3 $H/tools/topology.py rows scripts/topology.yihou.137.tsv`
→ row 0 prefill, row 1 decode. The **stock** `tools/topology.py` rejects this
file; the patched one accepts it.

## 3. Confirm the node is clean

```bash
ssh crsuse2-m2m-137 'rocm-smi --showmeminfo vram | grep "Used Memory" \
  | awk "{s+=\$NF} END{printf \"total_MB=%.0f\n\", s/1048576}"; \
  rocm-smi --showpids | grep -c "^[0-9]"'
# expect total_MB≈2272 and 0 KFD processes
```

## 4. Bring up

```bash
cd $H
PATH="$WS/bin:$PATH" ./launch.sh \
  CONFIG=$WS/scripts/config.yihou.triton.sh \
  TOPOLOGY=$WS/scripts/topology.yihou.137.tsv \
  REMOTE_BENCH_DIR=$H \
  CONTROL_NODE=crsuse2-m2m-137 BUILDER_NODE=crsuse2-m2m-137 \
  "PREFILL_EXTRA_ENV=MC_LOG_LEVEL=INFO" "DECODE_EXTRA_ENV=MC_LOG_LEVEL=INFO" \
  OUT_DIR=$WS/launch
```

The config carries all four deltas (`triton`/`triton`, `index_share=true`,
`PREFILL_HICACHE=1`, 3600 s / warmup 10) **and** `ENGINE_PORT_STRIDE=256`, so
`agentx_bench.sh` will see the same values when it sources the same file. That
matters: passing them only on `launch.sh`'s command line causes
`agentx_bench: ERROR: prefill-0: live hicache=True, config expects False`.

### EXPECT DECODE TO FAIL THE FIRST TIME

On this shape the two legs' RCCL communicators race, and the built-in same-node
start gate **does not reliably prevent it** (observed twice). Decode typically
dies with:

```
RuntimeError: NCCL error: unhandled cuda error
Rank 0 scheduler died during initialization (exit code: -3)
```

**Recovery, which worked both times:** leave prefill running, wait until it is
`/health` 200 and settled, then start decode by hand:

```bash
ssh crsuse2-m2m-137 docker rm -f glm52-pd-yihou-sn-p4d4-decode-0
ssh crsuse2-m2m-137 bash $H/engine.sh decode decode-0 10.245.153.247 4,5,6,7 \
  29257 28999 25558 28802 glm52-pd-yihou-sn-p4d4-decode-0 10.245.153.247:22379 \
  CONFIG=$WS/scripts/config.yihou.triton.sh \
  TOPOLOGY=$WS/scripts/topology.yihou.137.tsv \
  CONTROL_NODE=crsuse2-m2m-137 "DECODE_EXTRA_ENV=MC_LOG_LEVEL=INFO" \
  SERVER_LOG=$WS/decode-retry.log </dev/null
```

Settle times that worked: ~9 min (this run) and ~17 min (the previous one).
Whether the wait is causal is **unproven** — see `notes.md`.

Then start the router (note the quoting — zsh mangles `-v $M:$M:ro`):

```bash
M=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
ssh crsuse2-m2m-137 docker run -d --init --name glm52-pd-yihou-sn-p4d4-router \
  --network host -e INFERA_PD_DP_RANK_AFFINITY=true -v "${M}:${M}:ro" \
  infera-sglang:v0519-yihou-0917-nextnfix-hicache \
  python3 -m infera.server --host 0.0.0.0 --port 28000 --router-backend rust \
  --discovery-backend etcd --etcd-endpoint 10.245.153.247:22379 \
  --request-transport http --kv-event-transport zmq \
  --router-tokenizer-path "${M}" --router-policy kv-aware \
  --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0
```

### Verify before spending 2.5 hours

```bash
ssh crsuse2-m2m-137 'for p in 29001 29257 28000; do printf "%s=" $p; \
  curl -fsS -o /dev/null -w "%{http_code}\n" http://10.245.153.247:$p/health; done'
ssh crsuse2-m2m-137 'curl -fsS http://10.245.153.247:28000/v1/workers'
```

All three 200; **two** workers on the **same** IP, one prefill one decode. Then
check `/get_server_info` on both legs for `dsa_prefill_backend=triton`,
`dsa_decode_backend=triton`, `json_model_override_args` containing
`"index_share_for_mtp_iteration":true`, prefill `enable_hierarchical_cache=True`
and decode `False`, decode `disable_custom_all_reduce=True`. A one-request smoke
test must return a completion — **the text will be garbled**, which is simulated
acceptance working as configured.

## 5. Benchmark

```bash
cd $H
PATH="$WS/bin:$PATH" ./agentx_bench.sh CONC=40 \
  CONFIG=$WS/scripts/config.yihou.triton.sh \
  TOPOLOGY=$WS/scripts/topology.yihou.137.tsv \
  CONTROL_NODE=crsuse2-m2m-137 \
  "SSH_OPTS=-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=1000" \
  INFERENCEX_DIR=<an existing InferenceX clone, or omit to clone> \
  OUT_DIR=$WS/agentx-c40
```

The `ServerAliveInterval` is deliberate: the previous attempt's client died at
89 % of the window with the driver hung on what is suspected to be a dropped
long-lived ssh channel.

Wall time ≈ 20 min warmup (444 requests) + 3600 s profiling + reporting ≈ 1 h 25 m.
Success is `AgentX passed: <OUT_DIR>/agentx_conc40.json`.

**Watch the log by mtime, not content** (`stat -c %Y`). The previous failure mode
— client container gone, log silent, driver hung — is invisible to content
checks, and the highest-risk moment is near the *end* of the window.

## 6. Collect

```bash
# while the deployment is STILL UP
ssh crsuse2-m2m-137 'curl -fsS http://10.245.153.247:29257/metrics \
  | grep "^sglang:spec_accept"'
```

Trust only ranks whose counters moved; a rank that served no decode tokens
reports `0.0`. All four moved in this run.

## 7. Tear down

```bash
cd $H
PATH="$WS/bin:$PATH" ./stop.sh CONFIG=$WS/scripts/config.yihou.triton.sh \
  TOPOLOGY=$WS/scripts/topology.yihou.137.tsv CONTROL_NODE=crsuse2-m2m-137
```

Then verify the GPUs return to the ~284 MiB idle baseline with no KFD processes.
HiCache host-pool teardown lags; the ~756 GB of host RAM comes back a little
after the containers go.

## Success criteria vs what this run got

| criterion | result |
|---|---|
| `agentx_conc40.json` from a **full** 3600 s window | **yes** — 3,613.6 s |
| both legs on one node | **yes** — both workers on `10.245.153.247` |
| triton DSA on both legs | **yes** — confirmed in live `/get_server_info` |
| `index_share_for_mtp_iteration = true` | **yes** — confirmed live |
| `spec_accept_length` from ranks that moved | **yes** — all four, mean 3.5625 |
| error rate | **0 / 4157** |
