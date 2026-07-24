# dmabuf GPU-mem investigation — pack-up (2026-07-24)

## What this is

An offline, hardware-level investigation of what `ibv_reg_dmabuf_mr`
(GPUDirect dma-buf RDMA memory registration) does to GPU memory accounting on
the **ionic (AMD Pensando RoCE) + MI355X (gfx950)** platform.

Pure MVP: small C++/HIP programs that malloc GPU memory, register it via ionic
verbs, and measure the effect with **`hipMalloc` / registration success/failure
as ground truth** (all gauges were validated against known-true controls first).

## The two registration paths, and why we are forced onto dmabuf

There are two ways to register GPU memory for RDMA GPU-direct:

1. **`ibv_reg_mr`** (plain) — for GPU memory this REQUIRES a **peermem kernel
   module** (`ib_peer_mem` / amdgpu-peermem, the nvidia-peermem analogue). It
   pins via the peermem path. **Without peermem loaded, `ibv_reg_mr` cannot
   register GPU memory at all.**
2. **`ibv_reg_dmabuf_mr`** (dma-buf) — exports the GPU allocation as a dma-buf
   fd and registers that. This is the path available when there is no peermem,
   and is the modern/preferred path.

So when peermem is absent, **`ibv_reg_dmabuf_mr` is the ONLY option** — that's
why this whole investigation is about the dmabuf path.

## The core finding (corrected)

**On a NIC without ODP (On-Demand Paging), `ibv_reg_dmabuf_mr` ALWAYS pins the
GPU memory, and KFD then DOUBLE-COUNTS that externally-pinned memory — so the
reported available GPU memory shrinks by a whole pool's worth, even though no
new physical VRAM was allocated.**

Break it down:
- **ionic has no ODP** (its `*_odp_caps` are all zero) → dynamic-attach /
  move-notifier dma-buf is impossible → registration is **forced down the
  `ib_umem_dmabuf_get_pinned` path**, which pins the buffer for the MR lifetime.
- **KFD double-counts the pin.** The pool was already counted once as the
  original allocation. When it is pinned again via the external dma-buf import,
  KFD counts it a SECOND time against available memory. Result: the KFD/HIP
  available figure (`hipMemGetInfo` free) drops by ~pool-size at registration.
- **Physically there is no duplication** — the amdgpu TTM buddy allocator
  (real physical VRAM) does not move at registration. The shrink is a
  **double-count in KFD's accounting**, not a second physical copy. But it is a
  REAL functional loss: the double-counted memory genuinely cannot be allocated
  while the MR is registered, so subsequent allocations OOM.
- **Reversible / no leak:** dereg removes the pin and the double-count; the
  available figure returns. 20× reg/dereg cycles leave zero residue.

## The decisive experiment (allocate-first-then-register)

`logs/ro_dmabuf.log` + `scripts/joint_reorder.sh`: occupy 150 GiB FIRST (leaving
only ~36 GiB physical free per TTM), THEN register the 100 GiB pool.

```
P2 occupier 150G : hip_free=37.22  TTM_free=35.80   (physical nearly full)
P3 register 100G : registered 100/100 SUCCESS
                   TTM_free=35.80 UNCHANGED  hip_free=0.00
```

Two facts, together, nail the mechanism:
- **Registration SUCCEEDS with only 36 GiB physical free, and TTM physical does
  not move** → registration needs NO new physical memory (no 2× physical copy).
- **`hip_free` nonetheless collapses 37 → 0** → KFD double-counts the pinned
  100 GiB against available memory, so the reported free drops below what
  physically remains.

That is exactly "pin + KFD double-count → available shrinks," proven without
trusting any single gauge — it rests on "registration succeeded / TTM unmoved /
hip_free collapsed."

## How to navigate

- `REPRODUCE.md` — exact ordered steps to re-run everything.
- `environment.md` — node, GPU, NIC, kernel, driver, provider, peermem/ODP state.
- `notes.md` — the full arc, every wrong turn and how it was caught, and the
  instrument-reliability table (the most re-read file).
- `src/dmabuf_verdict_mvp.cpp` — the final (reordered) MVP.
- `scripts/check_nopin_capability.sh` — **one-click** 4-layer check (NIC ODP /
  peermem / kernel / rdma-core) with a PASS/FAIL verdict on whether no-pin is
  possible. `results/nopin_verdict_chi2798.txt` = its live output here.
- `scripts/` — also the handshake-synced host/container harnesses and the
  original split diagnostics (`nopin_diag.sh`, `peermem_chk.sh`).
- `logs/` — raw evidence: `ro_dmabuf.log` (decisive, allocate-then-reg),
  `hs_dmabuf.log` (handshake VRAM/GTT), `vmm_dmabuf.log` (TTM buddy timeline),
  `dmesg_dmabuf.log` (KFD dmesg — silent, see notes).

## Scope

This pack-up closes the **GPU-memory-accounting** question for `ibv_reg_dmabuf_mr`
on ionic. It does not cover the separate cross-node dma-buf transfer
access-violation (task #23).
