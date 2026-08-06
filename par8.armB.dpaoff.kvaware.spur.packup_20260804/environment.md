# Environment

> **Capture caveat, stated first.** `scripts/collect_env.sh` was **never run
> against these nodes**. Both spur allocations were reclaimed by the scheduler at
> the 24 h wall clock (SIGTERM, `JobState=FAILED Reason=NonZeroExitCode
> ExitCode=143:0`) roughly 11 hours after the measurement finished, before the
> snapshot was taken. Everything below is therefore reconstructed **from the
> run's own logs and the build logs**, which are first-hand but narrower than the
> script's output. Fields the logs do not carry are marked **not captured**.
> The script is shipped in `scripts/` so a rerun can close the gap.

## Digest

| | |
|---|---|
| cluster | **spur** (`crsuse2-m2m`), Slurm-compatible `spur` scheduler |
| access | login node → `spur exec <job> …`. **ssh to compute nodes is blocked** |
| **prefill node** | **crsuse2-m2m-250**, `ens3` = `10.245.158.155`, spur job **33490** |
| **decode node** | **crsuse2-m2m-251**, `ens3` = `10.245.151.18`, spur job **33491** |
| GPUs | 8 × AMD Instinct MI355X `gfx950` per node (`torch.cuda.device_count()` = 8, gated at container start) |
| CPU / RAM | **not captured** |
| GPU driver / ROCm | ROCm **7.2.0** (from the base image tag); host driver **not captured** |
| **image** | `infera/engine-sglang:final-pr`, built independently on each node |
| image id, prefill 250 | `sha256:f9b311995179ac33217adee42a80b20331f59487416c0a3d4b28bde460092122` |
| image id, decode 251 | `sha256:e676db2147bd0f3e6f35b0c038e7ce6af3c330c01b645404240052bc2c635043` |
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

## RDMA fabric

| | |
|---|---|
| KV NIC | **mlx5_0** — a single rail, unlike vultr's 8 ionic rails |
| `PORT_ACTIVE` in container | **1** on both nodes (this is the expected count on spur, not a degraded one) |
| GID index | **3** (mlx5 RoCEv2 routable, pinned — `GID=3` in the leg script) |
| dma-buf | **ON** (`MOONCAKE_DISABLE_HIP_DMABUF=0`) — spur has no `peermem`, so dma-buf via mlx5 ODP is the *only* GPUDirect path |
| transport health | `MC_FORCE_TCP` = **0**, `GID is NULL` = **0**, both legs |

`MC_MS_AUTO_DISC=0` with `MC_MS_FILTERS=mlx5_0` forces mooncake onto that one
NIC. Auto-discovery is off deliberately: the vultr sibling kit's script scans for
ionic rails and `exit 1`s when it finds none.

## Deployment — the exact resolved args

Read out of each leg's boot log (`server_args=ServerArgs(...)`), not assumed.

### prefill (crsuse2-m2m-250) — **DP-attention OFF**

```
--model-path /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4  --tp-size 8
--mem-fraction-static 0.70          <- 0.80 does NOT boot on this arm; see notes.md §3
--context-length 262144  --kv-cache-dtype fp8_e4m3  --page-size 64
--chunked-prefill-size 65536        <- GLOBAL budget, NOT divided (no DPA); notes.md §2
--ep-size 8                         <- HELD FIXED across arms (EP_DECOUPLE); notes.md §1
   (no --dp-size, no --enable-dp-attention, no --enable-prefill-delayer)
--max-running-requests 2048  --cuda-graph-max-bs 128  --watchdog-timeout 3600
--nsa-prefill-backend tilelang  --nsa-decode-backend tilelang
--enable-hierarchical-cache  --hicache-size 32  --hicache-storage-backend infera-kvd
--disaggregation-mode prefill  --disaggregation-transfer-backend mooncake
--disaggregation-ib-device mlx5_0  --disaggregation-bootstrap-port 8998
--disable-custom-all-reduce  --enable-cache-report
```

Resolved: `dp_size=1`, `enable_dp_attention=False`, `mem_fraction_static=0.7`,
`chunked_prefill_size=65536`, `disable_radix_cache=False`.

Memory pool: **avail mem 83.08 GB** after the pool, `max_total_num_tokens=2821248`.

### decode (crsuse2-m2m-251) — DP-attention ON, MTP ON

```
--tp-size 8  --dp-size 8  --enable-dp-attention  --ep-size 8
--mem-fraction-static 0.85          <- untouched; the decode leg never OOMed
--chunked-prefill-size 65536        (inert here — no prefill work on this leg)
--speculative-algorithm EAGLE  --speculative-num-steps 3
--speculative-eagle-topk 1  --speculative-num-draft-tokens 4
--num-reserved-decode-tokens 256
--disaggregation-mode decode  --disaggregation-ib-device mlx5_0
--disable-custom-all-reduce  --enable-cache-report
   (no --enable-hierarchical-cache: KVD=0 on this leg, by design)
```

Resolved: `dp_size=8`, `disable_radix_cache=True` (a PD decode leg sets this
itself), `max_total_num_tokens=3085504`, avail mem **42.24 GB** after the pool.

> **`--disable-custom-all-reduce` is on BOTH legs and is not optional.** The aiter
> custom all-reduce kernel deadlocks on gfx942/gfx950 during EAGLE verify at high
> concurrency (sglang #28815 / #31071 / PR #31478). It is a defect in that kernel
> on this arch, unrelated to the DSA patches this branch carries.

### router (on the prefill node)

```
python3 -m infera.server --host 0.0.0.0 --port 8190
  --discovery-backend etcd --etcd-endpoint 10.245.158.155:2379
  --request-transport http --kv-event-transport zmq
  --router-policy kv-aware
  --router-tokenizer-path /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
  --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0
  --kvd-socket-path /tmp/kvd/kvd.sock
```

Startup line read back from the router log:
`router-policy=kv-aware overlap_weight=1 prefill=20.0 decode=2.0`

Backend is the **python** router (`--router-backend` defaults to `python`;
`infera/server/args.py:68`). The Rust backend was not used on this arm.

## Supporting services

| service | where | config |
|---|---|---|
| etcd | prefill node, host network | `quay.io/coreos/etcd:v3.5.14`, client `:2379` |
| infera-kvd | **both** nodes, one per node | `--max-bytes 64G --long-path /tmp/kvd-long --long-bytes 512G` |

Each node's engine talks to its **own local** kvd socket — these are two separate
stores, not two views of one. `--max-bytes` is an absolute cap deliberately:
sglang's `--hicache-ratio` default sized the host pool at 355 GB per DP rank once,
and a TB-scale pinned host allocation can wedge a spur node at kernel level.

## Environment variables that affect the result

Set by `scripts/glm52_leg_spur_mtp.sh` on both legs:

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
SGLANG_DP_USE_GATHERV=1       (decode only; set when DPA=1)
# boot-deadlock avoidance
TORCHINDUCTOR_COMPILE_THREADS=1        <- =4 was NOT sufficient on a cold cache
TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_final_pr/inductor_cache
TRITON_CACHE_DIR=/shared_nfs/yihou_final_pr/triton_cache
```

`INFERA_ENGINE_READY_TIMEOUT` is the **current** name. The predecessor kits
exported `INFERA_SGLANG_READY_TIMEOUT`, which this branch no longer reads —
main's `e190d65` generalised the knob so it covers the vLLM worker too. Copying
the old name across leaves the 1800 s default in force and turns a slow cold
start into a spurious "engine never became ready".

## Bench driver

| | |
|---|---|
| repo | `/home/yihou/dev/git/Optimus-AgenticBench` |
| branch / commit | **not captured** — the vultr sibling used `fix/realistic-profile-session-driver` @ `1cf01cb` |
| venv | `/shared_nfs/yihou_agentbench/venv/bin/python3` |
| ran from | the **login node**, not inside a container (the driver is pure HTTP + a tokenizer; running it on the host it measures would perturb the result) |
| workload | `spec/par8.yaml`, md5 **`968b1543155839135dc9eaf6dd142626`** |

## Required secrets

Named, never valued:

| secret | source |
|---|---|
| docker registry auth | `DOCKER_CONFIG=/tmp/dockercfg` per node; the build pulls only the public `lmsysorg/sglang` base, so an empty config suffices |
| cluster access | spur account `yihou`, Slurm account `amd-primus`, QOS `amd-burst-qos`. No key material in this kit |
| etcd | no auth configured (cluster-internal, host network) |

No API keys, tokens, or S3 credentials are involved. No secret value appears in
any script or log in this kit.
