# RUNBOOK — GLM-5.2-MXFP4 same-node 1P1D P4D4 on crsuse2-m2m-102, image v0521, AgentX full C44

Everything is driven from the login host. Node access goes only through the spur ssh shim.
Nothing here deletes a file. Containers are stopped, never `rm`-ed, so their logs survive.

```bash
export H=/shared_nfs/yihou_glm52/agentx44_20261009/harness
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH
export SPUR_SSH_JOBS=crsuse2-m2m-102:200276
export CONFIG=$H/config.yihou.102-v0521.sh
export YIHOU_ROUND=c44a        # new value for every fresh bring-up (scopes names + $WS/runs/<round>)
N=crsuse2-m2m-102; IP=10.245.147.130
```

Ports (from the config): etcd 24501, router 38100, prefill 39101, decode 39357,
bootstrap 38198. Containers: `glm52-v0521-yihou-102-$YIHOU_ROUND-{etcd,prefill-0,decode-0,router}`.
Run dir: `/shared_nfs/yihou_glm52/agentx44_20261009/runs/$YIHOU_ROUND/` (`status.tsv`, `server-logs/`).

## 0. Before the first launch (once per image)

```bash
# a) pin the image ID: fill EXPECTED_IMAGE_ID in $CONFIG with this value
ssh $N "docker image inspect --format '{{.Id}}' infera-sglang:v0521-yihou-1009"

# b) the new image must still carry the transport behaviour the recipe depends on.
#    The v0.5.21 BASE mooncake differs from v0519's: it has no MOONCAKE_DISABLE_HIP_DMABUF
#    and no MC_DISABLE_HIP, and has a new MC_USE_HIP_IPC. The Dockerfile rebuilds mooncake
#    at faae8dd4; confirm that happened (want: HIP transport string present, MC_TE_FILTERS present).
ssh $N "docker run --rm --network none --entrypoint bash infera-sglang:v0521-yihou-1009 -c \
  'd=\$(python3 -c \"import mooncake,os;print(os.path.dirname(mooncake.__file__))\"); \
   for s in MC_TE_FILTERS \"HIP transport installed\" MC_DISABLE_HIP MOONCAKE_DISABLE_HIP_DMABUF MC_USE_HIP_IPC \" HCAs\"; do \
     printf \"%-30s %s\n\" \"\$s\" \$(strings \$d/engine*.so | grep -c \"\$s\"); done; \
   python3 -c \"import sglang;print(sglang.__version__)\"'"

# c) re-render and review (starts nothing). Image ID mismatch only warns in DRY_RUN.
bash $H/run.c44.yihou.sh dryrun > $H/dryrun.yihou.txt 2>&1
```

## a. Bring-up (with the same-node RCCL race handling)

```bash
cd $H && nohup bash run.c44.yihou.sh up > ../runs/$YIHOU_ROUND.up.log 2>&1 &
watch -n 30 cat ../runs/$YIHOU_ROUND/status.tsv
```

`up` does, in order: refuse if any `$CONTAINER_PREFIX-*` container exists → wait for all 8
GPUs < 400 MiB (HiCache host-pool / VRAM release is slow by design; it is not a hang) →
etcd → prefill (GPUs 0-3) and wait for `/health` 200 (cold ≈ 36 min: 450 GB host pool +
4 × ~103 GB indexer pools + cold AITER JIT) → settle `DECODE_SETTLE_S` (default 120 s) →
decode (GPUs 4-7), **up to 3 attempts** a1/a2/a3 → router from the **new** image with
`INFERA_PD_DP_RANK_AFFINITY=true` → check `/v1/workers` is exactly {39101, 39357}.

RCCL race: decode may die in `ncclCommInitRank` (`hipIpcGetMemHandle failed : invalid
argument` / `NCCL error: unhandled cuda error`). It is a race, there is no fix, only retry.
**Retry decode only. Never restart prefill.** Each failed decode container is renamed
`...-decode-dead-aN` (kept for its log). If all three fail:

```bash
bash $H/run.c44.yihou.sh retry-decode a4      # then: bash $H/run.c44.yihou.sh router
```

Decode at `DECODE_MEM_FRACTION=0.95` was capacity-only in Stage 2 (v0519: 2,773,376 tokens,
11.25 GB left); 0.92 is the highest value validated under load. If decode OOMs at
startup or under load, that is the likely cause: `bash run.c44.yihou.sh retry-decode a5`
with `DECODE_MEM_FRACTION=0.92` exported **and** the same value passed to agentx_bench.

## b. Census (transport acceptance; reads the server logs with `strings`)

```bash
bash $H/run.c44.yihou.sh census
```

Per leg, want: `4 x Found 0 HCAs`, `HIP installed` (`HIP transport installed`) = 4,
`HIP inst FAIL` (`Failed to install HIP transport`) = 0, `rdma installs` = 0,
`Invalid argument` = 0. `isHipReachable` is printed as information only: proven r013
logs contain 0 on both legs (it is a function name in the .so, not a log line), so a
non-zero count is new behaviour worth reading, not a failure by itself. Also `hipIpc failed` = 0, OOM/Traceback = 0/0. Also prints
`max_total_num_tokens` / `available_gpu_mem` and the host-pool allocation lines
(prefill: ~450.00 GB hierarchical pool). `Found 1 HCAs` or more means rdma got installed:
stop, the KV transfer will fail.

## c. Live config audit (`/get_server_info` + `docker inspect` on both legs)

```bash
bash $H/run.c44.yihou.sh audit      # writes runs/$YIHOU_ROUND/audit/*.json, exits 1 on any FAIL
```

Checks tp/dp/ep=4/4/1, DPA, mem_fraction 0.85/0.95, DSA triton/triton, fp8_e4m3,
`{"index_share_for_mtp_iteration":true}`, prefill HiCache on with size 450 /
write_through / kernel / page_first, decode HiCache off, shared-experts fusion ON
(`disable_shared_experts_fusion=False`), `max_total_tokens=None`, aiter allreduce
fusion on, decode `disable_custom_all_reduce=True` + EAGLE 5/1/6; container env
`MC_TE_FILTERS=yihou_no_such_rdma_device`, `MOONCAKE_DISABLE_HIP_DMABUF=0`,
`SGLANG_OPT_USE_TOPK_V2=false`, `HSA_NO_SCRATCH_RECLAIM` 0/1, `MC_DISABLE_HIP` absent,
decode `SGLANG_SIMULATE_ACC_LEN=3.61`. An `ABSENT` line means v0.5.21's server_info
does not expose that key; read it rather than ignore it.

## d. Smoke request

```bash
bash $H/run.c44.yihou.sh smoke
```

Must return a `"text"` field. With simulated acceptance ON the text is expected to be
garbled; this only proves the KV moved P→D through the router.

## e. AgentX full, CONC=44 (≈ 20-25 min warmup + 3600 s + reporting)

```bash
cd $H/bench-harness
nohup ./agentx_bench.sh CONC=44 CONFIG=$CONFIG YIHOU_ROUND=$YIHOU_ROUND \
  OUT_DIR=/shared_nfs/yihou_glm52/agentx44_20261009/runs/$YIHOU_ROUND/agentx-c44 \
  > /shared_nfs/yihou_glm52/agentx44_20261009/runs/$YIHOU_ROUND/agentx-c44.log 2>&1 &
```

The config supplies DURATION 3600, warmup 10/lane, failed-request threshold 0.10,
InferenceX `$WS/cache/InferenceX` (already cloned at 918524ff), SSH_OPTS with
ServerAliveInterval=30. agentx_bench re-sources `$CONFIG` and validates the live
deployment against it (any knob overridden at bring-up must be passed here too, e.g.
`DECODE_MEM_FRACTION`). Success: `AgentX passed: .../agentx_conc44.json`.
Watch the log by **mtime** (`stat -c %Y`), not content.
While the stack is still up, also capture
`ssh $N "curl -s http://$IP:39357/metrics | grep '^sglang:spec_accept'"`.

## f. Teardown

```bash
cd $H/bench-harness
./stop.sh CONFIG=$CONFIG YIHOU_ROUND=$YIHOU_ROUND CONTAINER_STOP_TIMEOUT=300
ssh $N 'docker ps -a --format "{{.Names}}\t{{.Status}}" | grep glm52-v0521-yihou-102'   # all Exited
ssh $N 'rocm-smi --showmeminfo vram | grep -i "used"'   # wait for ~0.3 GiB on all 8 before the next launch
```

`stop.sh` only touches names that contain `yihou` and start with this round's prefix,
stops (SIGTERM, then `docker kill` on timeout) and does not remove. It does not stop the
renamed `...-decode-dead-aN` containers (they have already exited). HiCache host-pool
release after stop is slow by design.
