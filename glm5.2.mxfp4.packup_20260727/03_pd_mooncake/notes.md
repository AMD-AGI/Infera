# Notes — 03 PD mooncake RDMA

## The story: TCP failure (rc6) → RDMA success (pd-unified)

**First attempt (rc6 image) — FAILED.** mooncake on the base rc6 image could only produce correct
output over **TCP** (`MC_FORCE_TCP=1`), and TCP collapsed at conc=64 (50/256 successful, median TTFT
51 s, prefill log flooded with `KVTransferError: remote mooncake session ... is not alive`). The
intended RDMA path failed because #2682's bundled mooncake installs a HIP transport unconditionally
and hardcodes its priority above RDMA, so a cross-node send picks intra-node **HIP-IPC**, which
cannot open a peer's handle cross-node (`hipIpcOpenMemHandle Error 17/201`). The full failed run is
preserved under `tcp_fail_appendix/`.

**Fix — the `pd-unified` image (Infera PR #19).** It rebuilds the base image's bundled mooncake
in place so PD KV-transport is decided at **runtime**:
- `MC_DISABLE_HIP_TRANSPORT=1` is the default (infera `rocm_rdma_env.py`) → cross-node PD stays on
  **RDMA**, not HIP-IPC. This is the core fix for the wall above.
- dma-buf (`ibv_reg_dmabuf_mr`) is now compiled in (PR #19 fixed a CMake propagation bug that
  silently dropped it), but **OFF by default** on the AMD ionic fleet (no ODP → dma-buf pins +
  duplicates the KV pool in VRAM). Default = bare `ibv_reg_mr` + amdgpu **peermem**.

With this image, GLM-5.2 mooncake PD works over real RDMA: conc=64 = 256/256, 0 transfer errors,
5147 tok/s (≈ the mori RDMA result). **No `MC_FORCE_TCP`.**

## How we confirmed it's really RDMA (not a silent TCP fallback)

The prefill leg log prints, once per NIC:

    rdma_context.cpp:75] HIP dmabuf disabled via MOONCAKE_DISABLE_HIP_DMABUF

— i.e. mooncake is in the RDMA `rdma_context` path (registering GPU memory via bare `ibv_reg_mr` +
peermem, dma-buf branch disabled), on all 8 ionic NICs. And there were **zero** `TcpTransport:
listen`, `hipIpcOpenMemHandle`, `not alive`, or `KVTransferError` lines in either leg. Contrast the
TCP appendix, whose decode log is full of `TcpTransport: listen on port ...`.

## Why this works on chi2878/chi2879 (kernel/driver facts)

The bare `ibv_reg_mr` + peermem GPU-direct path needs the amdgpu `ib_peer_mem` driver loaded — it
is, on both nodes. (The prior "peermem hard-faults" concern from earlier notes was on a *different*
node/kernel; here chi2878/chi2879 also report `CONFIG_PCI_P2PDMA=y`, and the peermem path ran clean
through conc=64 with zero faults.) The dma-buf path is available (`DMABUF=1` in `pd_leg.sh`) but not
needed for the pass and carries the ionic 2× pin cost — leave it OFF unless you cap the KV pool.

## Method source

The exact working method for the pd-unified image came from the PR #19 test packup
`sglang_unified_pd_test.packup_20260727` (which validated this image on DeepSeek-V4 at conc=128 with
zero transfer failures). We swapped its DSv4 recipe (`--attention-backend dsv4`, page-256) for the
GLM-5.2 DSA recipe (tilelang indexer, fp8_e4m3, no dsv4 backend) and kept its mooncake env
(dmabuf-off, HIP-off, all-8-ionic, `RDMAV_FORK_SAFE=1`) + `sglang_router` mini-LB.

## Gotchas

- **Image distribution:** pd-unified is a local build, not on a public registry. `docker save` over
  NFS is very slow (78 GB); stream `docker save | ssh <dst> docker load` over the data-plane instead.
- **Router JSON quoting:** drive correctness with `probe.py` (urllib), not inline `curl -d '{...}'`
  through nested ssh+docker-exec (quotes get mangled → HTTP 400).
- **Router false circuit-break:** after any failed round, `pkill -9 -f sglang_router` before relaunch
  (a stale router remembers dead workers).
- **decode.log caveat:** see `logs/README.txt` — the decode leg was later relaunched with MTP for
  experiment 06, overwriting its no-MTP log. The no-MTP transfer evidence is in `prefill.log` + the
  bench result.
