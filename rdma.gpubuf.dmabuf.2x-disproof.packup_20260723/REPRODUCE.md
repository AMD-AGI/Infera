# Reproduction kit — ionic dmabuf "2×" disproof

Goal: show that `ibv_reg_dmabuf_mr(VRAM)` adds **zero** real device VRAM, and that
the apparent 2× is only in `hipMemGetInfo`. ~2 minutes, single node, no partner,
no mooncake.

## 0. Prerequisites

- **Machine:** one AMD MI355X (gfx950) node with **new ionic** RDMA stack and an
  **IDLE GPU 0** (a foreign job on the GPU pollutes absolute VRAM readings — see
  notes.md; we hit this on the first run). This run used `chi2879` via jump host
  `root@149.28.124.225` (ProxyJump to chiXXXX).
- **Secrets:** cluster SSH only (ProxyJump). No registry login needed.
- **External deps (absolute paths, not in repo):**
  - A ROCm+RDMA container image with `libibverbs` + `libamdhip64`. We used
    `infera/engine-sglang:pd-final` (any ROCm image works).
  - **Host libionic matching the kernel driver ABI.** The 26.03 ionic kernel
    driver requires libionic **ABI 4** (`54.0-187`). Images shipping libionic
    `54.0-149` (ABI 1) are rejected → 0 RDMA devices. Inject the host file
    `/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187` into the container (all
    the `run_*.sh` here do this automatically).
  - Shared FS `/mnt/vast` (holds the scripts). Not strictly required — you can
    `docker cp` the scripts instead.
- **No git repo state needed** — the probe is standalone ctypes.

## 1. Confirm the ionic stack + kernel dma-buf support

    bash scripts/kern_dmabuf.sh        # expect: ionic 26.x, CONFIG_DMABUF_MOVE_NOTIFY=y, CONFIG_PCI_P2PDMA=y
    # (run.sh in original_probe_from_colleague/ prints driver/fw/libionic versions too)

## 2. The colleague's original VERDICT probe (uses hipMemGetInfo only)

    # copy the probe onto a shared path first, then:
    bash scripts/run_probe2.sh         # runs original_probe_from_colleague/ionic_vram_test.py
    # -> prints: bare ibv_reg_mr(VRAM)=OK ; dma-buf reg(VRAM)=OK_2x   (the artifact)

## 3. THE DISPROOF — three-meter cross-check on an idle GPU

    bash scripts/run_hold2.sh          # 4 GiB buffer
    bash scripts/run_hold_big.sh       # 64 GiB buffer (PD-scale)

Each launches `hold_probe.py` inside the container (registers a dmabuf MR, pauses
at each phase) while the HOST samples amdgpu's true VRAM meter
(`/sys/class/drm/card*/device/mem_info_vram_used`, summed over all cards).

Read the two columns side by side per phase:
- `hipMemGetInfo` free: drops at malloc AND again at reg  → looks 2×.
- **amdgpu true VRAM: +buffer at malloc, +0 at reg** → NOT 2×.

## 4. Deeper ablations (optional, prove it's not a usage bug)

    bash scripts/run_debug.sh          # dmabuf_debug.py: HIP vs HSA-v1 vs HSA-v2-NONE vs HSA-v2-PCIE
                                       # -> all identical; export API/mapping flag is NOT the lever
    # reg_variants.py  -> access-flag sweep (all 2x in hipMemGetInfo) + double-register same buffer (2nd = +0)
    # whatis2x.py      -> 4 distinct buffers each show the artifact; proves it's per-import in hipMemGetInfo view

## Expected output

`run_hold_big.sh` prints, at the reg phase:

    t=18s amdgpu_vram_all=66.22GiB | PHASE=after_reg hipfree=159.37 mr=OK

amdgpu true VRAM = 66 GiB (2.22 baseline + 64 buffer), **not 128**. That single
line is the disproof. `hipfree` shows 159 (=287−64−64) — the artifact.

## Cleanup / RDMA discipline

Every `run_*.sh` removes its container at the end (`docker rm -f`), which releases
the MR (dereg), closes the dmabuf fd, and frees the buffer. Confirm the node is
back to baseline before leaving:

    cat /sys/class/drm/card*/device/mem_info_vram_used | awk '{s+=$1} END{printf "%.2fG\n", s/1073741824}'   # ~2.2G idle
    rocm-smi --showpids    # No KFD PIDs

## If it doesn't reproduce

See `notes.md`. Most likely: (a) GPU not idle → absolute VRAM readings polluted
by a foreign job (we saw card0 sitting at 251 GiB); pick an idle card. (b) 0 RDMA
devices → libionic ABI mismatch, inject the host `54.0-187`. (c) `rocm-smi
--showmeminfo vram` absolute is unreliable on this stack (didn't even track
hipMalloc) — use the amdgpu sysfs meter, not rocm-smi.
