# Environment

Captured 2026-07-23 on chi2879 (the node the disproof ran on).

```
### chi2879 ###
GPU: gfx950 x 8
ROCm: 7.2.0
kernel: 6.8.0-124-generic
ionic driver: 26.03.3.001
ionic_rdma: 26.03.3.001
NIC fw: 1.117.5-a-77
host libionic: 54.0-187-1
kernel dmabuf cfg: CONFIG_PCI_P2PDMA=y CONFIG_HSA_AMD_P2P=y CONFIG_DMABUF_MOVE_NOTIFY=y 
ib_peer_mem: 2 (used by ionic_rdma)
```

## Notes on the stack

- **Our (new) ionic:** driver/ionic_rdma 26.03.3.001, NIC fw 1.117.5-a-77,
  kernel 6.8.0-124, host libionic 54.0-187 (ABI 4).
- **Colleague's (old) ionic baseline** (from their probe README, for contrast):
  driver 25.08.4.004, fw 1.117.1-a-63, libionic 54.0-149, kernel 6.8.0-107 →
  `bare ibv_reg_mr = FAIL_14`, `dmabuf = OK_2x`.
- **Container image:** infera/engine-sglang:pd-final (any ROCm+RDMA image works).
  It ships libionic 54.0-149 (ABI 1) which the 26.03 kernel driver REJECTS —
  the host 54.0-187 must be injected (see REPRODUCE §0 / the run_*.sh).
- **VRAM meters:**
  - `amdgpu mem_info_vram_used` (sysfs) = the authoritative driver VRAM meter used here.
  - `hipMemGetInfo` = HIP process/context free view; reflects P2P/BAR mapping → source of the "2×" artifact.
  - `rocm-smi --showmeminfo vram` (absolute) = UNRELIABLE on this stack (did not track hipMalloc); do not use.
- **Model/weights:** none — the probe is synthetic (hipMalloc), no model needed.
- **Secrets:** cluster SSH via ProxyJump only.
