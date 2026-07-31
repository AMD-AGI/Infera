# Environment

## When

2026-07-31, 10:29–10:53 UTC (build context created through the last stress run).

| step | time (UTC) |
|---|---|
| build context tarball | 10:31 |
| image built, both nodes | 10:31–10:33 |
| containers started | 10:34 |
| prefill leg ready | 10:39:50 |
| decode leg ready | 10:44:03 |
| router up (port 8180) | 10:46 |
| probe 4/4 | 10:47 |
| stress c32 / c128 / c128_r2 | 10:47–10:48 |

## Nodes

Held with `sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00`. Both passed the GPU
health gate (`GPUGATE True 8`) — spur has nodes that enumerate 8 GPUs and report `False`.

| leg | spur job | node | ens3 IP | server port |
|---|---|---|---|---|
| prefill | 17443 | `crsuse2-m2m-099` | 10.245.152.84 | 30000 |
| decode | 17444 | `crsuse2-m2m-227` | 10.245.151.183 | 30001 |

Router runs on the **prefill** node, port **8180**, prometheus **29180**. Ports 8160/8170
were used by earlier runs the same day; a fresh port per restart is deliberate — see
`notes.md`.

**Substitute your own job ids and IPs.** These jobs will not exist for you.

## Hardware

| item | value |
|---|---|
| GPU | AMD Instinct **MI355X** × 8 (**gfx950**) |
| CPU | AMD EPYC 9575F 64-Core × 2 sockets (236 logical CPUs) |
| RAM | 2751 GiB |
| KV NIC | **mlx5_0**, RoCEv2, **GID index 3**, netdev `ens3` |
| KV transport | mooncake RDMA + **dma-buf** (`MOONCAKE_DISABLE_HIP_DMABUF=0`) |

**Provenance:** the CPU/RAM figures are inherited from an earlier node of the same SKU in
the same partition, not re-measured on these two. Run `scripts/collect_env.sh` on each
node if an exact per-node record matters.

> **Why dma-buf and not peermem.** Spur has **no peer-memory kernel module**, so a bare
> `ibv_reg_mr` on a device pointer fails with EFAULT and dma-buf is the only GPUDirect
> path. mlx5 supports ODP, so the dma-buf MR is a dynamic attach — no pin, no VRAM
> doubling. This is the *opposite* of the vultr cluster (ionic + peermem, no ODP), where
> the same setting duplicates the KV pool in VRAM and crashes at KV setup.
> **Never enable HIP transport for cross-node PD.**

## Software

| item | value |
|---|---|
| image | `infera.yihou.sglang.converged:1.0` |
| image id | `sha256:b10b3c4bc318a96900e160543c4da2a59f4c2385d87615c03d6e6b1b2d44809e` |
| image size | 108 GB (uncompressed, as reported by `docker images`) |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| base digest | `sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d` |
| sglang commit | `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` (release/v0.5.15) |
| ROCm | 7.2.0 |
| Dockerfile | `deploy/docker/Dockerfile.sglang` |
| Infera branch | `worktree-dsa-hip-dp-rows-fix.rebase` |
| Infera commit | `0d3e34a` |
| rebased onto | `origin/main` at `8692fb4` |

This image differs from the previous kit's in three ways beyond the patch reshuffle: it is
built from `Dockerfile.sglang` rather than the now-deleted `Dockerfile.sglang.dmabuf`, and
so it **also** contains the infera python package and the Rust router, which the minimal
dma-buf image did not. That makes it closer to a real deployment, not further from one.

The image was built **independently on each node** from the same build context
(`/shared_nfs/yihou.temp/converged_build/buildctx.tar.gz`, 646 KB: `deploy/docker`,
`pyproject.toml`, `README.md`, `infera`, `rust`). Both builds ran the bytecode verification
gate and passed — see `logs/image_build.log`. Building twice is faster than moving the
image; a backgrounded `docker save` inside `spur exec` is killed at namespace teardown even
under `nohup`/`setsid`.

Verified in the image before use:

```
$ docker run --rm --entrypoint bash infera.yihou.sglang.converged:1.0 -c \
    'cd /sgl-workspace/sglang && git status --short --untracked-files=no python/sglang/srt'
 M python/sglang/srt/disaggregation/decode.py
 M python/sglang/srt/layers/attention/dsa/dsa_indexer.py
 M python/sglang/srt/layers/attention/dsa_backend.py
 M python/sglang/srt/managers/schedule_batch.py
 M python/sglang/srt/managers/scheduler_components/dp_attn.py
 M python/sglang/srt/model_executor/forward_batch_info.py
 M python/sglang/srt/models/deepseek_nextn.py
 M python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py
 M python/sglang/srt/speculative/eagle_worker_v2.py
```

Nine files — the same nine as the previous kit. `deepseek_nextn.py` is now modified by
main's patch script rather than by our diff; the resulting line is identical apart from the
trailing comment our diff added, which is how the provenance was confirmed.

Mooncake was rebuilt with dma-buf compiled in (`DMABUF_SYM=3`, `HIP_GATE=1` from `strings`
on the built `engine.cpython-310-*.so`), and the infera package imports.

## Model

`/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` — an absolute path on shared storage,
not in the image. Quark MXFP4 quantized, bf16 nextn layer.

## Server configuration

Both legs, from `scripts/pd_leg_exp.sh`:

```
--tp-size 8 --dp-size 8 --enable-dp-attention --ep-size 8
--nsa-prefill-backend tilelang --nsa-decode-backend tilelang
--kv-cache-dtype fp8_e4m3 --context-length 32768
--cuda-graph-max-bs 128 --max-running-requests 2048
--disaggregation-mode {prefill|decode} --disaggregation-transfer-backend mooncake
--disaggregation-ib-device mlx5_0
--disable-custom-all-reduce
```

Decode leg additionally:

```
--speculative-algorithm EAGLE --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--num-reserved-decode-tokens 256
```

`mem-fraction-static` is 0.88 on prefill, 0.85 on decode.

**MTP is enabled on the decode leg only** (`PREFILL_MTP=0`), matching every prior
measurement in this series except the IndexShare arm. It is a known limitation, not a
recommendation: with MTP off on prefill the draft KV pool is never registered for RDMA, so
the two legs register a different buffer count, and the guard's fourth term is permanently
true rather than genuinely rank-divergent — so the rank-split case patch 4 exists for has
never been exercised here.

`--disable-custom-all-reduce` is required: the aiter custom all-reduce kernel deadlocks on
gfx942/gfx950 during EAGLE verify at high concurrency.

## Things deliberately NOT enabled

- `--enable-hierarchical-cache` / `--infera-kvd-socket` — hicache L3 write-back GPU-faults
  on gfx950, and a large host allocation can wedge a spur node at kernel level (D-state,
  unkillable).
- HIP transport (`MC_DISABLE_HIP_TRANSPORT=1`) — a peer cannot open a local HIP IPC
  segment, so it breaks cross-node PD.
- `index_share_for_mtp_iteration=false` — this run keeps IndexShare **on**, because the
  patches are what is under test. See `notes.md` §2.

## Required secrets

None. The model is on shared storage, the base image is public on Docker Hub, and no
registry push is performed. `DOCKER_CONFIG=/tmp/dockercfg` must be exported before every
docker call on spur (docker 29 buildx plugin discovery fails on the default path) but holds
no credentials for this run.
