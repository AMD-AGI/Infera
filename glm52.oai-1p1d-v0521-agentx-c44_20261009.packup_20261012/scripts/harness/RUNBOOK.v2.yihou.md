# RUNBOOK v2 — GLM-5.2-MXFP4 same-node 1P1D P4D4, image v0521, E1 / E2 on any amd-oai node

Everything is driven from the login host. Node access only through the spur ssh shim
(the config puts `/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin` first on PATH and
sets `SPUR_SSH_JOBS=$YIHOU_NODE:$YIHOU_JOB`). Nothing here deletes a file; containers are
stopped, never `rm`-ed.

| exp | config | deltas (all else = phase-1 c44e recipe) |
|---|---|---|
| E1 relaxed | `config.yihou.E1-relaxed.sh` | `--hicache-ratio 1.5`, no `--hicache-size` (H1); index_share false (H2); `--no-enable-dsa-fused-indexer` + `SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER=1` both legs (H3); decode mem 0.85 (H8); CONC 40 |
| E2 target | `config.yihou.E2-target.sh` | `--hicache-size 450` + prefill `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512` (H1); index_share false (H2); fused indexer v0521 default ON (H3); decode mem 0.95 (H8); CONC 44 |

Both: triton/triton DSA, empty DSA_TOPK_BACKEND, `SGLANG_OPT_USE_TOPK_V2=false`, EAGLE 5/1/6 +
`SGLANG_SIMULATE_ACC_LEN=3.61`, decode `--disable-custom-all-reduce`, EP1, shared-experts fusion
ON, no `--max-total-tokens`, TP4/DP4/DPA both legs, kv fp8_e4m3, `MC_TE_FILTERS=yihou_no_such_rdma_device`
(never `MC_DISABLE_HIP`), the two fix mounts (`mounts.yihou.v2.tsv`), `ENGINE_PORT_STRIDE=256`,
per-role caches under `$WS/cache/<nodekey>/`, full mode 3600 s / warmup 10 per lane.

## Burst nodes 227 / 067 (2026-10-10 ~05:20 UTC): E1 on 227, E2 on 067. NFS is rw on both, so NODE_RW_ROOT stays empty.

Images built with `stage_image.yihou.sh` (logs `$WS/logs/stage.<node>.yihou.log`): clone fb8ae9c9
clean, BUILD_EXIT=0, engine.so f22b1364..., verify_v0521.sh fail=0. Pre-checks were read-only: our
ports and derived blocks are free, libionic.so and /dev/infiniband are present, the login host
reaches both IPs.

| node | job | ens3 IP | EXPECTED_IMAGE_ID | dryrun |
|---|---|---|---|---|
| crsuse2-m2m-227 | 205515 | 10.245.151.183 | sha256:256a5daa18f56dbb15562dd5bf11299c93f368685ba3b6a8f09c61229d0c173f | dryrun.e1a-227.yihou.txt (E1) |
| crsuse2-m2m-067 | 205548 | 10.245.157.102 | sha256:6a73feca0eb4b34b007a2597ee90baf77c684a7b70c5f64574ff35785a669fc7 | dryrun.e2a-067.yihou.txt (E2) |

```bash
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-227 YIHOU_JOB=205515 YIHOU_IP=10.245.151.183 \
  EXPECTED_IMAGE_ID=sha256:256a5daa18f56dbb15562dd5bf11299c93f368685ba3b6a8f09c61229d0c173f \
  CONFIG=$H/config.yihou.E1-relaxed.sh YIHOU_ROUND=e1a
# E2: YIHOU_NODE=crsuse2-m2m-067 YIHOU_JOB=205548 YIHOU_IP=10.245.157.102 \
#     EXPECTED_IMAGE_ID=sha256:6a73feca0eb4b34b007a2597ee90baf77c684a7b70c5f64574ff35785a669fc7 \
#     CONFIG=$H/config.yihou.E2-target.sh YIHOU_ROUND=e2a
cd $H && nohup bash run.v2.yihou.sh up > ../runs/$YIHOU_ROUND.up.log 2>&1 &
```

## ROCm debug agent (lean, both legs, on by default)

`DEBUG_AGENT=1` (knob; `YIHOU_OVERRIDES="DEBUG_AGENT=0"` to turn it off). engine.sh (WS copy) adds
`HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 HSA_ENABLE_DEBUG=1
ROCM_DEBUG_AGENT_OPTIONS=--disable-linux-signals` using dedicated `-e` flags. There is no
`--output`: the report goes to stderr, into the server log next to `Memory access fault`, with
docker timestamps. There is no `--precise-memory` and no `--all`. census counts
`kernel_code_entry` / `stopped, reason` / `Memory access fault` per server log (want 0) and prints
the first hits. audit checks the three env values.

## Same-node IPC: IPC_MODE_LEGACY (default 1) / ENGINE_PID_HOST (default 0)

Default = the end-to-end-validated fix: `HSA_ENABLE_IPC_MODE_LEGACY=1` on BOTH legs (sanity round
sanity-v0519-legacy on 136: SANITY PASS, decode first try, KV transfer 0 fails). `--pid host` is
an untested alternative arm (`YIHOU_OVERRIDES="ENGINE_PID_HOST=1 IPC_MODE_LEGACY=0"`).

Root cause of the same-node "RCCL race", found first-hand by repro: ROCr default IPC binds the
abstract socket `\0hsa<ns-pid>` in the shared host netns. Two private PID namespaces can give a
prefill rank and a decode rank the same pid, and then `hipIpcGetMemHandle` fails with invalid argument.

- `ENGINE_PID_HOST=1` adds `--pid host` to BOTH engine containers, so pids are unique.
- `IPC_MODE_LEGACY=1` is the alternative: `HSA_ENABLE_IPC_MODE_LEGACY=1` on BOTH legs. Never
  use it on one leg only: on 136 prefill could not open decode's legacy handles
  (hipIpcOpenMemHandle 201/17).
- Both knobs live in the shared config, so the legs cannot diverge. `retry-decode` (ipc_guard)
  refuses if the live prefill differs from the config.
- Override example: `YIHOU_OVERRIDES="ENGINE_PID_HOST=0 IPC_MODE_LEGACY=1"`.

census runs `tools/pidcensus.yihou.sh` on the node (read-only). For each engine process it shows
the host pid, the namespace pid and whether `@hsa<pid>` is present, then the @hsa socket count and
any namespace-pid collisions between the legs (want none). audit and agentx_env check
HostConfig.PidMode and the legacy env.

## NODE_RW_ROOT (nodes that mount /shared_nfs read-only: 136, 138)

NFS is the read side only: model, harness, mounts, configs, InferenceX. On those nodes `$WS` is
bind-mounted `:ro`. Everything the node writes goes under `NODE_RW_ROOT`:
- `SERVER_LOG` → `$NODE_RW_ROOT/runs/<round>/server-logs/`
- AITER/triton/hf/xdg caches → `$NODE_RW_ROOT/cache/<nodekey>/…`
- tmpfs mount points, the router's caches
- the AgentX client's output, aiperf runtime, HF and HOME (agentx_bench writes `runtime.node.env`
  pointing there)

The default is `/mnt/m2m_nobackup/yihou_v0521dbg` when `YIHOU_JOB` is empty (138: 7.3 TB free
nvme, world-writable sticky dir). It is empty (old behaviour, all on NFS) when `YIHOU_JOB` is set.
An explicit `NODE_RW_ROOT=` forces NFS.

Copy-back to `$WS/runs/<round>/` on the login host (base64 tar over ssh):
- `census` pulls `server-logs/` first.
- `bench` (agentx_bench) pulls the client output on success and on failure.
- `stop` pulls the whole node run dir.

Caches stay on the node, which keeps it warm for the next round.

## Burst nodes 238 / 213: CANCELLED (jobs held 0 GPUs). Kept for the record only.

Images staged by `stage_image.yihou.sh` (logs `$WS/logs/stage.<node>.yihou.log`): clone fb8ae9c9
clean, BUILD_EXIT=0, engine.so sha256 f22b1364..., verify_v0521.sh fail=0, sglang 0.5.21.dev20261008+g11b1fb21de.

| node | job | ens3 IP | EXPECTED_IMAGE_ID | dryrun |
|---|---|---|---|---|
| crsuse2-m2m-238 | 205500 | 10.245.147.235 | sha256:32a71e3bd30d4a2467a544ca56e6029e610d4b8f6a05cee8a0b699b077179282 | dryrun.e1a-238.yihou.txt |
| crsuse2-m2m-213 | 205501 | 10.245.145.240 | sha256:a8c5538ea1cbc126cdcb006b78fd3f6d9ca54efe33567667235c59ed3aeddd21 | dryrun.e2a-213.yihou.txt |

Check at 03:45 UTC: 238 had a FOREIGN listener on 10.245.147.235:39101 (= our prefill port), with
a foreign container `chaox_geak` and ~270 GB VRAM used on all 8 GPUs (alloc-watch 03:41).
Re-check before `up`; if 39101 is still taken, move the engine ports (the derived blocks move with them):
`export YIHOU_OVERRIDES="ENGINE_PORT_BASE=39701"` (prefill 39701, decode 39957), then re-run dryrun.

```bash
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-238 YIHOU_JOB=205500 YIHOU_IP=10.245.147.235 \
  EXPECTED_IMAGE_ID=sha256:32a71e3bd30d4a2467a544ca56e6029e610d4b8f6a05cee8a0b699b077179282 \
  CONFIG=$H/config.yihou.E1-relaxed.sh YIHOU_ROUND=e1a
# E2: YIHOU_NODE=crsuse2-m2m-213 YIHOU_JOB=205501 YIHOU_IP=10.245.145.240
#     EXPECTED_IMAGE_ID=sha256:a8c5538ea1cbc126cdcb006b78fd3f6d9ca54efe33567667235c59ed3aeddd21
#     CONFIG=$H/config.yihou.E2-target.sh YIHOU_ROUND=e2a
cd $H && nohup bash run.v2.yihou.sh up > ../runs/$YIHOU_ROUND.up.log 2>&1 &
```

## Quick start: E1 on crsuse2-m2m-138 (plain ssh, image already built)

138 mounts `/shared_nfs` READ-ONLY (findmnt `ro`, 03:50 UTC), and so does 136. Handled by
NODE_RW_ROOT (above), which defaults on for plain-ssh nodes. Dryruns: `dryrun.e1a-138.yihou.txt`,
`dryrun.e2a-136.yihou.txt`.
Earlier read-only checks (03:4x UTC): ens3 = 10.245.157.237/20, libionic.so and /dev/infiniband
present, image ID sha256:546c1105..., 8 x MI355X at 284 MiB, ports and derived blocks free, login
host reaches 10.245.157.237. Foreign containers `dev_xb` and `yihou-expert8-tools-138` hold
/dev/kfd. NOT checked: engine.so sha f22b1364 (needs a container run).

```bash
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-138 YIHOU_IP=10.245.157.237 YIHOU_JOB=   # empty JOB = plain ssh
export EXPECTED_IMAGE_ID=sha256:546c1105f0e8c676e2d5d67fb293ab0177dd03a7813b9de5ef825827fea8e110
export CONFIG=$H/config.yihou.E1-relaxed.sh YIHOU_ROUND=e1a
# optional, ~1 min: ssh $YIHOU_NODE "docker run --rm --network none --entrypoint bash infera-sglang:v0521-yihou-1009 \
#   -c 'sha256sum \$(python3 -c \"import mooncake.engine as e; print(e.__file__)\")'"   # want f22b1364...
bash $H/run.v2.yihou.sh dryrun > $H/dryrun.$YIHOU_ROUND-138.yihou.txt 2>&1    # done: dryrun.e1a-138.yihou.txt
cd $H && nohup bash run.v2.yihou.sh up > ../runs/$YIHOU_ROUND.up.log 2>&1 &     # then steps 5-9 below
```

Second node (crsuse2-m2m-136, IP 10.245.154.168, image sha256:a3a450255b5fe8b70dee5ed0515f6d1bb8b3b525760fa299f7dd91e02e55c31d):
same, with those values and e.g. `CONFIG=$H/config.yihou.E2-target.sh YIHOU_ROUND=e2a`. Two
nodes use different `$WS/cache/<nodekey>` and topology files, so they do not collide; ports
repeat but are per host.

## 0. Variables (fill per node)

`YIHOU_JOB` must be set: a spur job id routes ssh through the spur exec shim, and an empty value
means plain ssh (the config then removes the shim from PATH).

```bash
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-NNN        # the allocated node
export YIHOU_JOB=NNNNNN                  # spur job id holding it
export YIHOU_IP=a.b.c.d                  # its data IP (step 1b)
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH
export SPUR_SSH_JOBS=$YIHOU_NODE:$YIHOU_JOB
N=$YIHOU_NODE
```

## 1. Allocation / node checks

```bash
spur queue | grep "$YIHOU_JOB"                       # job RUNNING on $N, time left
ssh $N hostname                                      # shim reaches the node
# a) data IP + NIC. The config assumes NIC ens3 (YIHOU_NIC=... to override; YIHOU_NIC= (empty)
#    lets engine.sh resolve it from YIHOU_IP on the node).
ssh $N "ip -o -4 addr show | awk '{print \$2, \$4}'"
# b) GPUs visible to the job and idle (8 x MI355X, ~0.3 GiB used each)
ssh $N 'rocm-smi --showproductname | grep -m1 MI3; rocm-smi --showmeminfo vram | grep -i used'
#    If rocm-smi says "No AMD GPUs specified" (job holds no GPU cgroup), export
#    SPUR_SSH_ROCM_BYPASS_NODES=$N SPUR_SSH_ROCM_IMAGE=infera-sglang:v0521-yihou-1009 (after step 2);
#    the gate and agentx_env.py both call rocm-smi.
# c) libionic (engine.sh mounts it), host RAM (450 GB x 4 ranks + 4 x ~103 GB indexer pools for E2)
ssh $N 'ls -l /lib/x86_64-linux-gnu/libionic.so; free -g | head -2'
# d) no foreign containers on the GPUs / ports 24501 38100 39101 39357 38198 35657 38901
ssh $N 'docker ps --format "{{.Names}}\t{{.Status}}"; ss -ltn | grep -E ":(24501|38100|39101|39357|38198|35657|38901) "'
```

## 2. Base pull + image build on the node (cold: base pull + Mooncake/libionic fetch need network)

```bash
ssh $N 'docker pull lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008'
# Build context = the private clone (HEAD fb8ae9c9 carries the v0521 Dockerfile + patches).
C=/shared_nfs/yihou_glm52/agentx44_20261009/infera-src.yihou
git -C $C log --oneline -1; git -C $C status --short     # expect fb8ae9c9 and a clean tree
ssh $N "cd $C && docker build --network host \
  --build-arg SGLANG_BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008 \
  --build-arg BUILD_MOONCAKE=1 --build-arg REQUIRE_MOONCAKE_CROSS_HOST_ROUTING=1 \
  -f deploy/docker/Dockerfile.sglang.v0521.yihou -t infera-sglang:v0521-yihou-1009 . \
  > /shared_nfs/yihou_glm52/v0521dbg_20261010/logs/build.$N.yihou.log 2>&1; echo BUILD_EXIT=\$?"
# Verify: loaded mooncake engine.so must be the proven one (sha256 f22b1364...).
ssh $N "docker run --rm --network none --entrypoint bash infera-sglang:v0521-yihou-1009 -c \
  'sha256sum \$(python3 -c \"import mooncake.engine as e; print(e.__file__)\"); \
   python3 -c \"import sglang;print(sglang.__version__)\"'"
#   want: f22b1364501c748d8a6946f0bf3731057d585229fc20efdcad453205747cd56d ; 0.5.21.dev20261008+g11b1fb21de
# Full marker check (read-only, CPU container), same script as phase 1:
ssh $N "docker run --rm --network none -v /shared_nfs:/shared_nfs --entrypoint bash infera-sglang:v0521-yihou-1009 \
  /shared_nfs/yihou_glm52/agentx44_20261009/build/tools.yihou/verify_v0521.sh" \
  > /shared_nfs/yihou_glm52/v0521dbg_20261010/logs/verify.$N.yihou.log 2>&1; tail -3 /shared_nfs/yihou_glm52/v0521dbg_20261010/logs/verify.$N.yihou.log
# Pin the image ID for THIS node (engine.sh refuses a mismatch; run.v2 `up` refuses an empty one):
export EXPECTED_IMAGE_ID=$(ssh $N "docker image inspect --format '{{.Id}}' infera-sglang:v0521-yihou-1009")
echo $EXPECTED_IMAGE_ID
```

`EXPECTED_IMAGE_ID` and `IMAGE` are forwarded by run.v2 to the node-side engine.sh. For
agentx_bench/stop run through run.v2 they are forwarded too; keep the export in the shell.

## 3. Pick the experiment and round

```bash
export CONFIG=$H/config.yihou.E1-relaxed.sh   YIHOU_ROUND=e1a     # or:
# export CONFIG=$H/config.yihou.E2-target.sh  YIHOU_ROUND=e2a
# New YIHOU_ROUND for every fresh bring-up (scopes container names + $WS/runs/<round>).
# Knob overrides that must reach the node too: YIHOU_OVERRIDES="DECODE_MEM_FRACTION=0.92 ..."
bash $H/run.v2.yihou.sh dryrun > $H/dryrun.$YIHOU_ROUND.yihou.txt 2>&1   # starts nothing, no ssh
$H/tokdiff.yihou.sh $H/dryrun.E1.yihou.txt $H/dryrun.$YIHOU_ROUND.yihou.txt   # only the intended deltas
```

Containers: `glm52-v0521-yihou-<nodekey>-$YIHOU_ROUND-{etcd,prefill-0,decode-0,router}`.
Ports: etcd 24501, router 38100, prefill 39101, decode 39357, bootstrap 38198.

## 4. Bring-up (GPU-idle gate, prefill, decode with RCCL-race retry, router)

```bash
cd $H && nohup bash run.v2.yihou.sh up > ../runs/$YIHOU_ROUND.up.log 2>&1 &
watch -n 30 cat ../runs/$YIHOU_ROUND/status.tsv
```

Refuses if any `$CONTAINER_PREFIX-*` exists → waits for all 8 GPUs < 400 MiB → etcd → prefill
(GPUs 0-3) `/health` 200 (cold up to ~45 min) → settle 120 s → decode (GPUs 4-7), up to 3
attempts a1/a2/a3 → router with `INFERA_PD_DP_RANK_AFFINITY=true` → `/v1/workers` == {39101, 39357}.
RCCL race (`hipIpcGetMemHandle failed : invalid argument` / `unhandled cuda error` in decode):
**retry decode only, never restart prefill**. Dead decodes are renamed `...-decode-dead-aN`.
If all three fail: `bash $H/run.v2.yihou.sh retry-decode a4` then `bash $H/run.v2.yihou.sh router`.

## 5. Census (reads server logs with `strings`)

```bash
bash $H/run.v2.yihou.sh census | tee ../runs/$YIHOU_ROUND/census.txt
```

Per leg: `4 x Found 0 HCAs`, `HIP transport installed` = 4, rdma installs 0, Invalid argument 0,
hipIpc failed 0, OOM/Traceback 0/0.
New in v2:
- `fused idx ON` = count of `gfx950 fused DSA indexer enabled`: **E1 0, E2 4** per leg.
  `fused idx decl` (declined this call / not used) is informational (c44e decode: 4).
- host pools (prefill): `Allocating kv hierarchical KV host pool: N tokens, X GB` ×4 —
  **E2 450.00 GB**; **E1 ≈ 1.5 × device pool ≈ 153-154 GB** (c44e prefill: 2,270,656 device tokens,
  450 GB = 10,016,064 tokens → 1.5× ≈ 3.41 M tokens ≈ 153 GB). Plus the DSA indexer pool line(s).
- prefill container env: `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512` **present in E2, absent in E1**;
  `SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER=1` present in E1 only.
- decode eager-fallback counter: `NCCL INFO` lines after the last `Capture ... CUDA graph end` and
  after `The server is fired up`, per rank. **Want 0 after fired-up** (c44d/c44e, index_share
  false: 0; c44c, index_share true: ~17.8k per rank). Re-run census after the bench too.

## 6. Live config audit

```bash
bash $H/run.v2.yihou.sh audit       # -> runs/<round>/audit/*.json, exit 1 on any FAIL
```

v2 additionally checks `hicache_ratio` (prefill 1.5, decode None), `hicache_size` (E1 0 / E2 450),
`enable_dsa_fused_indexer` (E1 False / E2 None), mem_fraction (decode E1 0.85 / E2 0.95), env
`SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER` and `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB`.

## 7. Smoke

```bash
bash $H/run.v2.yihou.sh smoke       # must return "text" (garbled is expected: simulated acceptance)
```

## 8. AgentX full mode (CONC from the config: E1 40, E2 44)

```bash
cd $H && nohup bash run.v2.yihou.sh bench > ../runs/$YIHOU_ROUND.bench.log 2>&1 &
# result: runs/$YIHOU_ROUND/agentx-c$CONC/agentx_conc$CONC.json ; log runs/$YIHOU_ROUND/agentx-c$CONC.log
```

agentx_bench.sh re-sources `$CONFIG` and agentx_env.py refuses on any live/config mismatch
(v2 also checks hicache ratio/size, the fused-indexer flag, the writer env and the chunk env).
InferenceX is at `$WS/cache/InferenceX` (918524ff, already present). Watch the log by mtime.
`bench` also saves `sglang:spec_accept*` from decode `/metrics` after the run. ≈ 20-25 min warmup
+ 3600 s + reporting.

## 9. Teardown

```bash
bash $H/run.v2.yihou.sh stop        # stop.sh: SIGTERM (300 s) then kill; never rm
ssh $N 'docker ps -a --format "{{.Names}}\t{{.Status}}" | grep glm52-v0521-yihou'
ssh $N 'rocm-smi --showmeminfo vram | grep -i used'   # wait for ~0.3 GiB on all 8 before the next up
```
