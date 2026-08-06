# Environment

Captured by `scripts/collect_env.sh` on **both live nodes immediately after the
run**, before the allocations were released. Full raw output in `env/`.

## Digest

| | |
|---|---|
| cluster | **spur** (`crsuse2-m2m`), Slurm-compatible `spur` scheduler |
| access | login node → `spur exec <job> …`. **ssh to compute nodes is blocked** |
| **prefill node** | **crsuse2-m2m-010**, `ens3` = `10.245.156.167`, spur job **35682** |
| **decode node** | **crsuse2-m2m-081**, `ens3` = `10.245.152.164`, spur job **35683** |
| GPUs | 8 × **AMD Instinct MI355X** `gfx950` (`0x75a3`) per node |
| CPU | AMD EPYC 9575F 64-Core (256 threads) |
| RAM | 2.7 TiB per node |
| kernel | `6.8.0-107-generic` |
| **GPU driver** | **6.14.14** (both nodes) |
| ROCm | **7.2.0** (from the base image) |
| **image** | `infera/engine-sglang:final-pr`, built independently on each node |
| image id, prefill 010 | `sha256:5f6b7b5e1cb5b1696300d374b911f9a83b371ac521aed40c0b78d7fc0a640128` |
| image id, decode 081 | `sha256:b13bcd7c82f2e01ac7f8e4e78c0816754dd33609c3431de4b0e285c5f420bc27` |
| **base image** | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| **base digest** | `sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d` |
| sglang | 0.5.15.post1 |
| repo | `AMD-AGI/Infera`, branch **`yihou.dev.glm52.agentic.final.pr`** |
| **commit** | **`97c2ff5c0a594d60c52ec7add2e5c84f652bf734`** |
| model weights | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (shared NFS, 408 GB) |
| workspace | `/shared_nfs/yihou_final_pr/` |

> The two image ids differ **by design** — each node built independently from the
> same source tarball. Equivalence is established by the **bytecode gate**, never
> by digest: both nodes printed `BYTECODE_GATE OK` (9/9) at container start.

## RDMA fabric — nine rails present, exactly one used

These nodes carry **both** fabrics, which the earlier arm-B nodes' capture did not
make visible:

```
link ionic_0/1 … ionic_7/1   state ACTIVE   netdev enP2p0s9 … (8 rails)
link mlx5_0/1                state ACTIVE   netdev ens3
```

| | |
|---|---|
| KV NIC **actually used** | **mlx5_0** only |
| how it is pinned | `MC_MS_AUTO_DISC=0` + `MC_MS_FILTERS=mlx5_0`, and `--disaggregation-ib-device mlx5_0` |
| `PORT_ACTIVE` seen inside the container | **1** — the container sees only mlx5 |
| GID index | **3** (mlx5 RoCEv2 routable, pinned via `GID=3`) |
| dma-buf | **ON** (`MOONCAKE_DISABLE_HIP_DMABUF=0`) — spur has no `peermem`, so dma-buf via mlx5 ODP is the *only* GPUDirect path |
| transport health | `MC_FORCE_TCP` = **0**, `GID is NULL` = **0**, both legs |

**The 8 ionic rails are deliberately unused.** Auto-discovery is off precisely so
mooncake cannot wander onto them: the vultr sibling kits' leg script *scans* for
ionic rails and this cluster's ionic fabric is not the KV path here. Anyone
reading `env/*.txt` and seeing 8 ionic links should not conclude the run used
them — `--disaggregation-ib-device mlx5_0` in both leg command lines is
authoritative.

## Deployment — the exact resolved args

Read out of each leg's boot log (`server_args=ServerArgs(...)`), not assumed.

### prefill (crsuse2-m2m-010) — DP-attention **ON**

```
--model-path /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4  --tp-size 8
--dp-size 8  --enable-dp-attention  --ep-size 8
--enable-prefill-delayer  --prefill-delayer-max-delay-ms 5000
--mem-fraction-static 0.70    <- FORCED. 0.80 aborts under round-robin; see README
--context-length 262144  --kv-cache-dtype fp8_e4m3  --page-size 64
--chunked-prefill-size 65536  <- GLOBAL; resolves to 8192/rank under dp8
--max-running-requests 2048  --cuda-graph-max-bs 128  --watchdog-timeout 3600
--nsa-prefill-backend tilelang  --nsa-decode-backend tilelang
--enable-hierarchical-cache  --hicache-size 32  --hicache-storage-backend infera-kvd
--disaggregation-mode prefill  --disaggregation-transfer-backend mooncake
--disaggregation-ib-device mlx5_0  --disaggregation-bootstrap-port 8998
--disable-custom-all-reduce  --enable-cache-report
```

Resolved: `dp_size=8`, `mem_fraction_static=0.7`, `chunked_prefill_size=8192`
(per rank — 65,536 ÷ 8), `disable_radix_cache=False`.

Memory pool: **avail mem 85.17 GB** after the pool, `max_total_num_tokens=2387200`.

### decode (crsuse2-m2m-081) — DP-attention ON, MTP ON

```
--tp-size 8  --dp-size 8  --enable-dp-attention  --ep-size 8
--mem-fraction-static 0.85          <- untouched; the decode leg never OOMed
--speculative-algorithm EAGLE  --speculative-num-steps 3
--speculative-eagle-topk 1  --speculative-num-draft-tokens 4
--num-reserved-decode-tokens 256
--disaggregation-mode decode  --disaggregation-ib-device mlx5_0
--disable-custom-all-reduce  --enable-cache-report
   (no --enable-hierarchical-cache: KVD=0 on this leg, by design)
```

Resolved: `dp_size=8`, `disable_radix_cache=True` (a PD decode leg sets this
itself), `max_total_num_tokens=3085504`, avail mem **42.12 GB** after the pool.

> **`--disable-custom-all-reduce` is on BOTH legs and is not optional.** The aiter
> custom all-reduce kernel deadlocks on gfx942/gfx950 during EAGLE verify at high
> concurrency (sglang #28815 / #31071 / PR #31478). It is a defect in that kernel
> on this arch, unrelated to the DSA patches this branch carries. It defaults OFF
> on both arms so that "MTP on vs off" never becomes a two-variable comparison.

### router (on the prefill node)

```
python3 -m infera.server --host 0.0.0.0 --port 8190
  --discovery-backend etcd --etcd-endpoint 10.245.156.167:2379
  --request-transport http --kv-event-transport zmq
  --router-policy round-robin
  --router-tokenizer-path /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
  --kvd-socket-path /tmp/kvd/kvd.sock
```

Startup line read back from the router log: `router-policy=round-robin`
(no `overlap_weight=` line — that is kv-aware's, and its absence is the
discriminator between the two arms' routers).

`--router-tokenizer-path` is present but **inert** here: the argument is
`required=True` at the parser level regardless of policy, and
`_build_round_robin(**_)` discards every kwarg. The **overlap weights** are
deliberately *not* passed, since nothing would consume them.

Backend is the **python** router (`--router-backend` defaults to `python`;
`infera/server/args.py:68`).

## Supporting services

| service | where | config |
|---|---|---|
| etcd | prefill node, host network | `quay.io/coreos/etcd:v3.5.14`, client `:2379` |
| infera-kvd | **both** nodes, one per node | `--max-bytes 64G --long-path /tmp/kvd-long --long-bytes 512G` |

Each node's engine talks to its **own local** kvd socket — two separate stores,
not two views of one. `--max-bytes` is absolute deliberately: sglang's
`--hicache-ratio` default once sized the host pool at 355 GB *per DP rank*, and a
TB-scale pinned host allocation can wedge a spur node at kernel level.

This arm exercised both tiers: `host_bytes` reached **84.6 GB** against the 64 GB
`--max-bytes` cap with `long_bytes` at **297 GB** — i.e. spillover into the long
tier was live, with 121,835 evictions.

## Environment variables that affect the result

```
# GLM-5.2 DSA-ROCm recipe (mandatory on gfx950)
SGLANG_USE_AITER=1            SGLANG_ROCM_FUSED_DECODE_MLA=0
SGLANG_OPT_USE_TILELANG_INDEXER=1
SGLANG_OPT_USE_TOPK_V2=0      SGLANG_OPT_USE_JIT_NORM=0
SAFETENSORS_FAST_GPU=1        HIP_FORCE_DEV_KERNARG=1
# transport
MC_GID_INDEX=3  MC_DISABLE_HIP_TRANSPORT=1  MC_MS_AUTO_DISC=0  MC_MS_FILTERS=mlx5_0
MOONCAKE_DISABLE_HIP_DMABUF=0  RDMAV_FORK_SAFE=1
NCCL_IB_DISABLE=1  NCCL_IGNORE_CPU_AFFINITY=1  HSA_NO_SCRATCH_RECLAIM=1
# determinism / readiness
PYTHONHASHSEED=0              <- stable block hashes -> stable kvd keys across restarts
INFERA_ENGINE_READY_TIMEOUT=3600
SGLANG_DP_USE_GATHERV=1       (set on both legs here; DPA is on for both)
# boot-deadlock avoidance
TORCHINDUCTOR_COMPILE_THREADS=1        <- =4 was NOT sufficient on a cold cache
TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_final_pr/inductor_cache
TRITON_CACHE_DIR=/shared_nfs/yihou_final_pr/triton_cache
```

`INFERA_ENGINE_READY_TIMEOUT` is the **current** name. The predecessor kits
exported `INFERA_SGLANG_READY_TIMEOUT`, which this branch no longer reads —
main's `e190d65` generalised the knob to cover the vLLM worker too. Copying the
old name across leaves the 1,800 s default in force and turns a slow cold start
into a spurious "engine never became ready".

## Bench driver

| | |
|---|---|
| repo | `/home/yihou/dev/git/Optimus-AgenticBench` |
| branch / commit | **not captured** — the vultr sibling used `fix/realistic-profile-session-driver` @ `1cf01cb` |
| venv | `/shared_nfs/yihou_agentbench/venv/bin/python3` |
| ran from | the **login node**, not inside a container |
| workload | `spec/par8.yaml`, md5 **`968b1543155839135dc9eaf6dd142626`** |

## Required secrets

Named, never valued:

| secret | source |
|---|---|
| docker registry auth | `DOCKER_CONFIG=/tmp/dockercfg` per node; the build pulls only the public `lmsysorg/sglang` base, so an empty config suffices |
| cluster access | spur account `yihou`, Slurm account `amd-primus`, QOS `amd-burst-qos`. No key material in this kit |
| etcd | no auth configured (cluster-internal, host network) |

No API keys, tokens, or S3 credentials are involved. No secret value appears in
any script, log, or env capture in this kit.
