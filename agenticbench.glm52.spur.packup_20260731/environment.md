# Environment

## Cluster

| | |
|---|---|
| cluster | crsuse **spur** (`spur exec <job> <cmd>`; ssh to compute nodes is banned) |
| partition / qos | `amd-spur` / `amd-burst-qos` |
| prefill node | job **19254**, `crsuse2-m2m-034`, `ens3` = **10.245.153.38** |
| decode node | job **19255**, `crsuse2-m2m-227`, `ens3` = **10.245.151.183** |
| GPUs | 8 x **AMD Instinct MI355X** (gfx950, `sramecc+`, **`xnack-`**) per node |

`xnack-` matters: there is no page-migration fallback, so a GPU dereference of an
unmapped host address aborts the process rather than faulting in a page. That is the
whole mechanism behind the kvd bug fixed here.

## Software

| | |
|---|---|
| sglang | **0.5.15.post1** |
| torch | **2.9.1+rocm7.2.0.git7e1940d4** |
| HIP | **7.2.26015-fc0010cf6a** (ROCm 7.2.0) |
| image | `infera/engine-sglang:kvaware-kvd`, built per node from the `infera.kv.fix` worktree |
| engine entry | `python3 -m infera.engine.sglang` — the **infera wrapper**, not `sglang.launch_server` |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (408 GB, `GlmMoeDsaForCausalLM`, 78 layers, 256 experts, ships `chat_template.jinja`) |
| bench | Optimus-AgenticBench, `agent.agent_throughput`, `--mode realistic` |

### Pinned image digests (read out of the build log, not the floating tag)

    infera/engine-sglang:kvaware-kvd
      FROM docker.io/infera/engine-sglang:kvaware-kvd-base
           @sha256:7112f4f511a26dbfa7d6673fdded703579d15f2e66c7e32ae84472ed3f26c3ac
      which is FROM docker.io/lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x
           @sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d

Dockerfile: `deploy/docker/Dockerfile.sglang.kvaware-kvd` in the `infera.kv.fix`
worktree. Its stage-2 self-check must print `kvaware+kvd self-check OK`.

**Gap:** the final built image's own `sha256` Id was not captured before the jobs
reached walltime TIMEOUT and the nodes were released. The two base digests are pinned
and exact, and the build is deterministic from them plus the git SHAs below.

### Repo state — all three trees pinned

| repo / worktree | branch | commit | clean? |
|---|---|---|---|
| `infera.kv.fix` (image source) | `yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr` | **`52d71195498f9caaf8b84bcca3276a366b1e8010`** | **clean** |
| `Optimus-AgenticBench` (the bench) | `fix/realistic-profile-session-driver` | **`1cf01cbf169d9370a0bc8fe574055c5e975d1be9`** | **clean** |
| `infera.yihou.glm5.2.mxfp4` (this kit) | `yihou.dev.glm5.2.mxfp4.experiment` | `1c6d318da227b545ea83b7289bcb54fb54f94e37` | 4 untracked |

Both the image source and the bench were at clean trees, so those two SHAs fully
determine the code that produced these numbers.

### Bench driver invocation

    python3 -m agent.agent_throughput \
      --mode realistic \
      --workload-config workloads/caseA_full.yaml \
      --server http://10.245.153.38:8190 \
      --model glm5.2-mxfp4 \
      --tokenizer /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 \
      --dashboard-mode \
      --name caseA_full --data-dir <out>

`--dashboard-mode` is **not** cosmetic — without it no structured artifact is written
at all. See `notes.md`.

Driven from the **login node**, deliberately: the bench is pure HTTP plus a tokenizer,
and running the load generator inside the prefill container would place it on the host
it is measuring. Python venv at `/shared_nfs/yihou_agentbench/venv`.

## External dependencies (absolute paths, not in the repo)

| what | where |
|---|---|
| model weights | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` |
| bench source | `/home/yihou/dev/git/Optimus-AgenticBench` (`pip install -e .` into the venv) |
| bench venv | `/shared_nfs/yihou_agentbench/venv` |
| scratch, logs, bench artifacts | `/shared_nfs/yihou_agentbench/` |
| kvd L3 long tier | `/tmp/kvd-long` in-container (`--long-bytes 512G`) |

`/home` must not hold large artifacts — it has filled up and destroyed a 28 GB image
tar on this cluster before.

## Secrets required (names and sources only — no values)

| secret | source |
|---|---|
| docker registry login | team registry account; `export DOCKER_CONFIG=/tmp/dockercfg` before **every** docker call (docker 29 buildx plugin discovery fails on the default path) |
| cluster access | Spur job allocation via `sbatch`; `spur exec` only — no SSH keys, ssh to compute nodes is banned |

No API keys, tokens, S3 or etcd credentials are involved. etcd runs unauthenticated on
the prefill node's private data-plane IP. No secret value appears anywhere in this kit.

**Gap:** CPU model and RAM per node were not captured before the nodes were released.
Neither is load-bearing for these numbers (the workload is GPU- and fabric-bound), but
they are absent rather than assumed.

The wrapper entry is not a detail: `sglang.launch_server` bypasses the kvaware/kvd
wiring entirely, so a run launched that way measures neither feature.

## Transport

Spur, not vultr. The two clusters are configured oppositely and the wrong block
silently drops to TCP:

    IBDEV=mlx5_0        MC_GID_INDEX=3        MOONCAKE_DISABLE_HIP_DMABUF=0
    MC_MS_AUTO_DISC=0   MC_MS_FILTERS=mlx5_0  NIC=ens3

Verified in-run: `MC_FORCE_TCP` hits **0**, `mlx5_0` present 26x in the prefill log.
The 8 ionic NICs on this node lack ODP and are not used for KV.

## Deployment under test

    two-node PD over mooncake RDMA
    DP-attention 8/8 both legs   (--dp-size 8 --enable-dp-attention --ep-size 8)
    kv-aware routing             ON
    kvd (infera HiCacheStorage)  PREFILL ON / DECODE OFF   (operator instruction)
    MTP                          OFF                       (operator instruction)
    --context-length             262144
    --chunked-prefill-size       65536  (= 8192 per rank at dp 8)
    --hicache-size               32     (absolute GB; never --hicache-ratio)
    --enable-cache-report        on     (else the bench's cache_hit% reads 0)
    --disable-custom-all-reduce  on     (aiter kernel deadlocks on gfx950)
    mem-fraction-static          0.88 prefill / 0.85 decode
    router                       port 8190, kv-aware policy

`--hicache-size` is absolute by deliberate choice: the default `--hicache-ratio 2.0`
sizes off `max_total_num_tokens` and has computed to 355 GB per DP rank on this stack.

## Patches applied at runtime, inside the container

| patch | why |
|---|---|
| `patch_hicache_rocm_host_alloc.py` | **found and fixed in this work.** ROCm `hipHostRegister` maps host pages at a device VA != the host VA, but sglang's hicache stores raw host `data_ptr()`s in device-side pointer tables that a GPU kernel dereferences. Routes `ALLOC_MEMORY_FUNCS` to `pin_memory` on HIP. Without it, prefill kvd GPU-faults. |
| `patch_mooncake_early_send_wait_event.py` | AMD-AGI/Infera PR #56. Every Case A request is multi-chunk prefill; without this, mooncake PD can ship half-written KV. It does not crash — long prompts just come back partially wrong. |

Both are self-locating and idempotent, and both delete stale `.pyc`. Bytecode was
verified after applying (`strings <pyc> | grep -c <MARKER>` = 1 on each node) because
a stale `__pycache__` entry silently reverting a patch has invalidated a full
experiment on this stack before.

## Not enabled, deliberately

* **MTP / speculative decoding** — operator instruction; the DPA+PD+MTP fix is not
  merged with kv-aware yet. No `--speculative-*` flag appears anywhere.
* **kvd on the decode leg** — operator instruction.
* PR #56's `kv_event` bigram fix (bigram radix keys only exist under EAGLE/MTP) and
  its `dsv4_gfx942` detection fix (gated on `is_gfx942()`; this is gfx950).
