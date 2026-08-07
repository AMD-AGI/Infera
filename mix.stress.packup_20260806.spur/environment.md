# Environment — GLM-5.2 MIX Task 3 agentic stress

Same MIX deployment as the Task 1 fixlen sweep (`mix.fixlen.packup_20260806/`)
and the Task 2 latency run (`mix.latency.packup_20260806/`) — same node, same
image, same frozen server. The facts are copied here so this packup is
self-contained.

## Hardware

| item | value |
|---|---|
| Cluster | AMD crsuse spur (`crsuse2-m2m`, `amd-spur` partition) |
| Node | `crsuse2-m2m-036` (single node; slurm job **44901**) |
| Data-plane IP | `10.245.148.191` |
| NIC | `ens3` |
| GPU | 8 × AMD Instinct **MI355X** (gfx950) |
| Deployment | single-node aggregated (MIX) — **no PD, no mooncake, no RDMA** |

MIX is single-node: there is no KV transfer / RDMA fabric to capture. Only the
data-plane NIC (`ens3`) matters, because the worker binds `SGLANG_LOCAL_IP_NIC` /
`GLOO_SOCKET_IFNAME` to it.

## Software

| item | value |
|---|---|
| Repo branch | `dev.yihou.glm52.mix.experiment` |
| Repo commit | `d1a97b2b17732664580c6cbf7c5aa1ff4bef51bd` |
| Engine image | `infera/engine-sglang:final-pr` — **built on the node** from branch source |
| Dockerfile | `deploy/docker/Dockerfile.sglang` (in-tree at the above SHA) |
| Base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` (ROCm 7.2.0, MI35x) |
| etcd | `quay.io/coreos/etcd:v3.5.14` (needs `--entrypoint /usr/local/bin/etcd`) |

> **Pin the base digest.** Recorded here by floating tag only; the `sha256` was
> not captured at build. Resolve it on the node with:
> `docker inspect --format '{{index .RepoDigests 0}}' lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`

### Build-time patches baked into the image (in-tree, applied by the Dockerfile)

The image bakes the branch's DSA + ROCm-hicache patches; they live in the repo at
the pinned SHA (not duplicated into this packup):

- `deploy/docker/patches/sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py`
  — DSA indexer DP-padded-rows patch with the **P1V3 `_p1v2_rows` reconciliation**
  (`min(real, padded)`) for the reversed IDLE-rank case under MTP draft-extend.
  The `_p1v2_rows` **bytecode marker was verified at build** (identifier, not comment).
- `deploy/docker/patches/sglang_rocm/patch_hicache_rocm_host_alloc.py`
  — routes hicache `ALLOC_MEMORY_FUNCS` to `pin_memory` (`hipHostMalloc`); without
  it gfx950 (`xnack-`) hard-aborts with a GPU memory-access fault on the host VA.
- `deploy/docker/scripts/apply_sglang_dsa_patches.sh` — applies the DSA arm at build.

### Key engine config the numbers are sensitive to

Mix worker (`scripts/mix_worker.sh`): TP8 · DPA **off** · EP8 · MTP EAGLE
(steps 3 / topk 1 / draft 4) + `--disable-custom-all-reduce` · kvd on
(hicache-size 16; kvd daemon L2 32 GiB RAM + L3 file 64 GiB) · KV dtype
`fp8_e4m3` · context-length 262144 · chunked-prefill 8192 ·
`--mem-fraction-static 0.85` · `--max-running-requests 256` · `--reasoning-parser glm45`.

DSA-ROCm env (mandatory on gfx950 — without it the model serves garbage):
`SGLANG_OPT_USE_TILELANG_INDEXER=1`, `SGLANG_OPT_USE_TOPK_V2=0`,
`SGLANG_OPT_USE_JIT_NORM=0`, `SGLANG_USE_AITER=1`,
`SGLANG_ROCM_FUSED_DECODE_MLA=0`. (Full env in `scripts/mix_worker.sh`.)

Router (`infera.server`, in `scripts/mix_up.sh`): `--router-policy kv-aware`,
tokenizer = the weights dir, on `:8100`.

## Agentic driver runtime (Task 3 specific)

Unlike Task 2's latency driver (which ran *inside* the engine container), the
Task 3 agentic driver runs **on the NODE HOST** — the engine container has no
`agent` module. It is launched from a **staged venv** and talks to the router on
`http://10.245.148.191:8100` (the node's data-plane IP).

| item | value |
|---|---|
| Driver | `python -m agent.agent_throughput` (INFERA driver; **NOT** AgentX/customer bench) |
| Python | staged venv `/shared_nfs/yihou_agentbench/venv/bin/python` |
| Router endpoint | `http://10.245.148.191:8100` (host → router) |
| Served name | `glm5.2-mxfp4` |
| Tokenizer | `/shared_nfs/GLM-5.2-MXFP4` (the weights dir) |
| Workload | `scripts/stress_caseA.yaml` (`--mode realistic`, closed-loop) |
| Data dir | `/shared_nfs/yihou_final_pr/mix/results/stress` |

## Dependencies (absolute paths, not in repo)

| dep | path |
|---|---|
| Weights | `/shared_nfs/GLM-5.2-MXFP4` (HF cache `/shared_nfs/models--amd--GLM-5.2-MXFP4`) |
| Bind-mount | `/shared_nfs` mounted into the container at the same path |
| Tokenizer | the weights dir (`/shared_nfs/GLM-5.2-MXFP4`) |
| Agent venv | `/shared_nfs/yihou_agentbench/venv` (has the `agent` module + deps) |
| Served name | `glm5.2-mxfp4` |

## Secrets (names + source only — never values)

- **Docker registry login** for pulling the base / pushing the built image — from
  the operator's registry creds, written to `DOCKER_CONFIG=/var/tmp/dockercfg_yihou`
  (crsuse `/tmp` is root-owned; `/var/tmp` is writable by `yihou`).
- **Cluster access** — spur allocation on `crsuse2-m2m` (job 44901); commands run
  via `spur exec`. The agentic driver runs on the node host, not through
  `docker exec`.
- No API keys / etcd auth / S3 creds needed — etcd runs unauthenticated on the
  node IP.
