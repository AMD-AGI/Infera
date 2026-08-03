# Environment

## Cluster

| | |
|---|---|
| cluster | crsuse **spur** (`spur exec <job> <cmd>`; ssh to compute nodes is banned) |
| partition / qos | `amd-spur` / `amd-burst-qos` |
| prefill node | job **24300**, `crsuse2-m2m-253`, `ens3` = **10.245.157.89** |
| decode node | job **24301**, `crsuse2-m2m-236`, `ens3` = **10.245.146.87** |
| walltime | 24 h holds, submitted 2026-08-01 08:34 UTC |

Both nodes identical:

| | |
|---|---|
| CPU | AMD EPYC 9575F 64-Core, **236** logical cores |
| RAM | **2,751 GB** |
| GPUs | 8 × **AMD Instinct MI355X** (`0x75a3`), **`gfx950`**, 118 CU |
| ROCm (host) | **7.0.1** |
| ROCm (in image) | **7.2.0** |
| docker | **29.6.1** |
| local docker root | `/mnt/m2m_nobackup/docker` (28 TB, ~20 % used) |

`gfx950` is **`xnack-`**: there is no page-migration fallback, so a GPU
dereference of an unmapped host address aborts the process rather than faulting
a page in. That is the whole mechanism behind the ROCm hicache patch in
`patches/`.

**Node selection gotcha.** The first hold (`24299`) landed on `crsuse2-m2m-232`
and immediately went `PENDING Reason=JobHoldMaxRequeue` — a node that accepts
placement and then fails the job at launch. Resubmitting with
`--exclude=crsuse2-m2m-232` landed a good node on the first try.

## Fabric

Each node exposes **8 × ionic + 1 × mlx5_0**; only `ens3` (the mlx5 netdev)
carries an IP.

| | |
|---|---|
| KV transport NIC | **`mlx5_0`**, state `4: ACTIVE`, fw **28.43.3608** |
| GID index | **3** (RoCEv2 routable) |
| dma-buf | **ON** (`MOONCAKE_DISABLE_HIP_DMABUF=0`) |
| mooncake discovery | `MC_MS_AUTO_DISC=0`, `MC_MS_FILTERS=mlx5_0` |

Spur has **no peermem module**, so a bare `ibv_reg_mr` on a GPU pointer EFAULTs
and **dma-buf via mlx5 is the only GPUDirect path**. The 8 ionic NICs lack ODP
and are not used for KV. This is the exact opposite of the vultr cluster
(8 × ionic + peermem, dma-buf OFF, GID 1) where the branch was originally
validated — getting it wrong drops silently to TCP, which is *correct but slow*
and looks entirely healthy.

Verified in-run: `MC_FORCE_TCP` **0** on both legs, `mlx5_0` present **26×** per
leg log.

## Software

| | |
|---|---|
| sglang | **0.5.15.post1** |
| torch | **2.9.1+rocm7.2.0.git7e1940d4** |
| HIP | **7.2.26015-fc0010cf6a** |
| engine entry | `python3 -m infera.engine.sglang` — the **infera wrapper**, not `sglang.launch_server` |
| router entry | `python3 -m infera.server` — **not** `infera.router` (a package with no `__main__`) |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` |
| bench | sglang's own `python/sglang/bench_serving.py`, `--dataset-name random` |

The wrapper entry is not a detail: `sglang.launch_server` bypasses the
kvaware/kvd wiring entirely, so a run launched that way measures neither feature.

### Images — built per node, and why the ids differ

    infera/engine-sglang:merged-mtp
      crsuse2-m2m-253 (prefill)  sha256:42a303e5820cce9fa58ee10968dac8e4b87cb9e5eddc00f47ffa6bd524b7ec91
      crsuse2-m2m-236 (decode)   sha256:ff7b02eb6f1c6c184c2fb9a46dd18a48a2c18f50bfc35e0e0f306d34baf899f9

    FROM lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x
      @sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d

**The two ids differ and that is expected** — each node built independently from
the same source, so Rust router objects and layer timestamps differ. **Do not
check for equal digests.** Equivalence is established by the 8-assertion
bytecode gate in `scripts/start_ctr.sh`, which passed `BYTECODE_GATE OK` on both.

Built in two stages (`scripts/build_image.sh`):

1. `deploy/docker/Dockerfile.sglang` → `infera/engine-sglang:merged-mtp-base`
2. `deploy/docker/Dockerfile.sglang.kvaware-kvd` → `:merged-mtp`
   (must print `kvaware+kvd self-check OK`)

The base tag is **pinned and must stay pinned**: the GLM-5.2 DSA patches are
context diffs applied at `--fuzz=0` against sglang
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`, so a base bump **fails the build**
rather than mis-applying silently.

### Repo state

| repo / worktree | branch | commit | clean? |
|---|---|---|---|
| `infera.merge.liying.kv.mtp` (image source, this kit) | `yihou.dev.glm52.merged.experiment` | **`b92a1e81380f7583ef030b2f4a56426149f9a412`** | **NO — see below** |
| `Optimus-AgenticBench` | `fix/realistic-profile-session-driver` | `1cf01cbf169d9370a0bc8fe574055c5e975d1be9` | clean |

**The image source tree was deliberately dirty**, by operator decision. Two
uncommitted changes, both reproduced in `patches/`:

    M  deploy/docker/Dockerfile.sglang            (adds the sglang_rocm patch layer)
    ?? deploy/docker/patches/sglang_rocm/         (the ROCm hicache host-alloc fix)

Without them the prefill leg GPU-faults the moment kvd writes back at
long-context scale on gfx950. They are **not** on the branch; whether they should
be is a decision deferred to after the experiment. See `patches/README.md`.

Everything else in the image comes from the branch at `b92a1e8`, so that SHA
plus the two patches fully determine the engine.

## External dependencies (absolute paths, not in any repo)

| what | where |
|---|---|
| model weights | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (symlink → `models--amd--GLM-5.2-MXFP4/snapshots/386bd0e4…`) |
| bench source | in-image: `/sgl-workspace/sglang/python/sglang/bench_serving.py` |
| scratch, logs, build ctx, results | `/shared_nfs/yihou_agbench_mtp/` |
| kvd L3 long tier | `/tmp/kvd-long` in-container (`--long-bytes 512G`) |

`/home` must not hold large artifacts — it has filled up and destroyed a 28 GB
image tar on this cluster before. Anything > ~500 MB goes to `/shared_nfs`.

## Secrets required (names and sources only — no values)

| secret | source |
|---|---|
| docker registry login | team registry account; `export DOCKER_CONFIG=/tmp/dockercfg` before **every** docker call (docker 29's buildx plugin discovery fails on the node's root-owned default path) |
| cluster access | Spur job allocation via `sbatch`; `spur exec` only — **no SSH**, ssh to compute nodes is blocked by an `AllowUsers` whitelist |

No API keys, tokens, S3 or etcd credentials are involved. etcd runs
unauthenticated on the prefill node's private data-plane IP. **No secret value
appears anywhere in this kit** (checked across scripts, logs and results).

## Deployment under test

    two-node PD over mooncake RDMA (mlx5_0, GID 3, dma-buf ON)
    DP-attention 8/8 both legs    (--dp-size 8 --enable-dp-attention --ep-size 8)
    kv-aware routing              ON   (prefill weight 20.0 / decode 2.0)
    kvd (infera HiCacheStorage)   PREFILL ON / DECODE OFF   (operator instruction)
    MTP (EAGLE)                   DECODE ON  (steps 3, topk 1, draft 4) / PREFILL OFF
    --context-length              262144
    --chunked-prefill-size        65536   (= 8192 per rank at dp 8)
    --hicache-size                32      (absolute GB; NEVER --hicache-ratio)
    --enable-cache-report         on
    --disable-custom-all-reduce   on the DECODE leg only (follows MTP)
    mem-fraction-static           0.88 prefill / 0.85 decode
                                  ^ NOTE (added after the fact): 0.88 is SAFE for
                                  this fixed-length sweep and was NOT changed here,
                                  but it aborts the prefill leg under Case A's
                                  ragged agentic workload with
                                  HSA_STATUS_ERROR_OUT_OF_RESOURCES. Case A needs
                                  0.80. See
                                  ../agenticbench.mtp.caseA.packup_20260801/notes/notes.config.md
    router                        port 8190, kv-aware policy
    KV pool                       3,260,992 tokens/rank (167.72 GB, fp8_e4m3)

`--hicache-size` is absolute by deliberate choice: the default
`--hicache-ratio 2.0` sizes off `max_total_num_tokens` and has computed to
355 GB *per DP rank* on this stack, which can wedge a spur node at kernel level.

`--disable-custom-all-reduce` follows MTP rather than being set globally, so that
"MTP on vs off" stays a one-variable comparison. The aiter custom all-reduce
kernel deadlocks on gfx950 during EAGLE verify at high concurrency.

## Gaps, stated rather than guessed

- **No kvd-off A/B.** kvd was ON (prefill) for the whole sweep, so no performance
  claim is made for it. Its read path is proven separately and structurally
  (`notes/kvd_serving_proof.md`), not by a latency comparison.
- **No repeat runs.** One measurement per sweep point; no confidence intervals.
- **`rocm-smi --showproductname` did not emit a parseable gfx line on the host**;
  the arch was read from `rocminfo` inside the container (`gfx950`,
  `Marketing Name: AMD Instinct MI355X`). The `sramecc+/xnack-` suffix pair was
  not captured on this run and is carried over from the prior spur kits — flagged
  as second-hand rather than measured here.
