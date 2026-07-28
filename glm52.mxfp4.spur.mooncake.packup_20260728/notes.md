# Notes / gotchas — GLM-5.2 mooncake PD on spur (what / why / how / context)

The most valuable output. These are the traps that cost time; each is what/why/how/context.

## ★1 — The transport MUST switch clusters: vultr ionic+peermem → spur mlx5+dmabuf

**What:** the source recipe (glm5.2.mxfp4.packup_20260727, folders 06/07) ran on *vultr* with all-8
ionic NICs, peermem, dmabuf OFF, GID1. On *spur* that config crashes at KV registration.

**Why:** spur nodes have **no peermem kernel module** → a bare `ibv_reg_mr` on a GPU pointer EFAULTs,
so mooncake can't register the KV pool for RDMA at all. The only GPUDirect path is **dma-buf**
(`ibv_reg_dmabuf_mr`). And dma-buf only works without pinning on a NIC with **ODP** — spur's 8 ionic
NICs have NO ODP (dma-buf there pins+doubles the whole KV pool → HIP-209 crash), but the 1 mlx5_0
HAS ODP (dynamic attach → no pin). So: force mooncake onto **mlx5 only**, dmabuf ON.

**How (pd_leg_spur.sh):** `DMABUF=1` (→ `MOONCAKE_DISABLE_HIP_DMABUF=0`), `--disaggregation-ib-device
mlx5_0`, `MC_GID_INDEX=3` (mlx5 RoCEv2 routable, verified via show_gids: idx3 = v2 IPv4-mapped),
`MC_MS_AUTO_DISC=0 MC_MS_FILTERS=mlx5_0` (stop mooncake grabbing the 8 ionic rails), NIC=ens3,
`MC_DISABLE_HIP_TRANSPORT=1` + unset `MC_ENABLE_HIP_TRANSPORT` (cross-node must stay RDMA, not HIP-IPC).

**Context:** verified real RDMA in the prefill log: `installTransport, type=rdma` on `mlx5_0` GID
idx 3, ×8 TP ranks, 0 `MC_FORCE_TCP`, 0 `KVTransferError`. The image's mooncake was rebuilt with
`USE_HIP_DMABUF=ON` (Dockerfile.sglang.dmabuf) so the dmabuf branch is compiled in
(`DMABUF_COMPILED_IN=yes`, `LINKS_HSA=yes`); the base image silently drops it.

## ★2 — docker build on spur: buildx config + exec-namespace teardown

**What:** `docker build --build-arg ...` fails `unknown flag: --build-arg`; and a backgrounded build
inside `spur exec` dies mid-extract with no image.

**Why:** (a) docker 29 has no classic builder — `build` IS buildx, whose plugin discovery needs a
writable `DOCKER_CONFIG` (the default `/opt/spur/.docker/config.json` is permission-denied → plugins
don't load). (b) each `spur exec` is a fresh PID+mount namespace; a `setsid`/`nohup &` build *client*
gets killed when the exec returns, even though dockerd is a host daemon.

**How:** (a) `export DOCKER_CONFIG=/tmp/dockercfg; mkdir -p $DOCKER_CONFIG` before any docker call;
buildx needs `--load`. (b) Don't background a build client — run the mooncake rebuild INSIDE a
host-owned detached container (`docker run -d` + `docker cp` context in + `docker exec -d bash -c
'build > log'`), which dockerd keeps alive, then `docker commit`. (This is how the image was built;
`Dockerfile.sglang` proper can't build on-node anyway — its Rust router needs crates.io, blocked=403.)

## ★3 — spur has bad GPU nodes: gate on torch.cuda.is_available()

**What:** node crsuse2-m2m-243 held fine, `torch.cuda.device_count()`=8, but
`torch.cuda.is_available()`=False and `rocminfo`=0 gfx950 → sglang dies "No accelerator available".

**Why:** a bad/half-initialized GPU node — a known spur flaky-node pattern. A fresh container on the
same node reproduced it, so it's the node, not the container.

**How:** always health-gate a freshly-held node before use:
`docker run --rm ... python3 -c 'import torch;print(torch.cuda.is_available())'` must print True.
If False, `scancel` and resubmit with `--exclude=<bad node>`.

## ★4 — DP-attention + MTP together crash (run one or the other)

**What:** `DPA=1 MTP=1` on the decode leg crashes during EAGLE draft-extend:
`RuntimeError: Expected lengths.size(0) == B` in `dsa_topk_backend.py fast_topk_v2`.

**Why:** the DSA sparse-attention indexer's topk kernel doesn't handle the DP-attention batch layout
in the draft-extend forward path. The two features are individually fine.

**How:** run **Config A (DPA, no MTP)** for high-conc throughput, or **Config B (MTP, no DPA)** for
faster decode — never fused. Matches the reference kits, which kept 06(MTP) and 07(DPA) separate.

**Context:** each passes conc=128 512/512 on spur. A=7218 tok/s / TPOT 31.3ms; B=8990 tok/s /
TPOT 19.2ms (1.6× faster decode via spec-dec, accept_len ~2.9).

## ★5 — MTP needs the pd-unified nextn 1-line patch (image-specific)

**What:** without the patch, the decode MTP draft head crashes `size of tensor a (3072) must match
b (6144)` in deepseek_nextn.py load_weights (eh_proj).

**Why:** GLM-5.2 quark exclude list has submodule `model.layers.78.eh_proj` but stock nextn checks
the bare `model.layers.78` → eh_proj built MXFP4-packed (3072) vs bf16 ckpt (6144).

**How:** bind-mount `patches/deepseek_nextn.unified_patch.py` (line 363 appends `.eh_proj` so the
submodule quark-exclude matches → eh_proj bf16) over the container's stock file on the DECODE leg.
Our image's stock nextn is 423 lines with the bug at line 363 — identical to the pd-unified base the
patch was made for, so it's a drop-in. (The rc6 patch is a DIFFERENT file; don't cross-use.)
No `SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA` env needed on this image.

## ★6 — kvd L3 write-back GPU-faults on gfx950; kv-aware routing is fine

**What:** enabling infera kvd (`--infera-kvd-socket`, which auto-adds `--enable-hierarchical-cache`)
brings the daemon + backend up cleanly, but the first real prefix-cache *write* → `Memory access
fault by GPU node-N` → all TP ranks die. Happens with BOTH `--hicache-io-backend kernel` (ROCm
falls back page_first→layer_first CUDA JIT) and `direct` (page_first_direct). `write_through_selective`
delays but doesn't avoid it.

**Why:** sglang 0.5.15.post1's hicache→storage GPU write-back path isn't compatible with GLM-5.2's
DSA/MLA KV layout on gfx950 (the write-back kernel touches GPU memory in a CUDA-shaped way).

**How:** kv-aware ROUTING needs only kv-events (ZMQ), NOT hicache — so it works perfectly:
router `pick policy=kv-aware ... cache_hits=31 request_blocks=31`, prefix-reuse warm req
cached_tokens=1984/2029, TTFT 1.87s→0.31s. For a working prefix-cache demo use the engine's NATIVE
radix cache (`infera_worker_nokvd.sh`, `--enable-cache-report`), no hicache/kvd. The kvd daemon +
8× `Creating dynamic storage backend infera-kvd` prove the wiring is correct; only the GPU write-back
kernel is the blocker.

**Context:** the kvd daemon L3 self-check is healthy (WRITE 20.7 / READ 43.7 GB/s). This is a real
sglang+ROCm bug to file upstream, not a config mistake.

## ★7 — big hicache allocation wedges a spur node at kernel level

**What:** `--hicache-ratio 2.0` (default) on the full model tries 8×319GB host KV + 8×73GB indexer
≈ 3.1TB pinned host alloc; with `direct` io the page-locking blocks in `__flush_workqueue` (D-state,
uninterruptible). `kill -9` can't reap it, `docker rm -f` fails ("did not receive an exit event"),
`docker exec` setns then fails — node crsuse2-m2m-321 was lost until reboot.

**Why:** TB-scale pinned-memory registration + a stuck kernel workqueue.

**How:** always cap with a small fixed `--hicache-size` (20-40 GB), never a ratio, when experimenting.
If a node wedges, abandon it (scancel) — don't fight it.

## Other useful facts
- Cold start ~7-8 min (weights ~2min + DP cudagraph). 069=prefill, 321=decode (321 later wedged; the
  kv-aware test moved to 069 single-node).
- Router: use a fresh `--prometheus-port` on every restart (`Address already in use` otherwise);
  `pkill -9 -f sglang_router` and kill leftover `sglang::router` by PID before relaunch.
- Node→node image move: `docker save`→NFS tar→`docker load` (spur bans ssh between compute nodes).
- etcd: infera talks to it via the **v3 HTTP/JSON gateway** (httpx) — no etcd3 python lib needed.
- infera pkg install on-node: `SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 pip install --no-build-isolation .`
  (deps pure-python from pypi; pypi=200, dockerhub=401-but-pulls-work, crates.io=403-blocked).
