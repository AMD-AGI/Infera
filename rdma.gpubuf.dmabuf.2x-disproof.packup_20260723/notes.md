# Notes — the debug narrative, hypotheses, and the source-backed mechanism

## The claim we were testing

Colleague's probe (`original_probe_from_colleague/`) reported, on their OLD ionic
(25.08): `bare ibv_reg_mr(VRAM) = FAIL_14 (EFAULT)` and
`ibv_reg_dmabuf_mr(VRAM) = OK but 2× VRAM (shadow copy)`. Their probe measures
VRAM with `hipMemGetInfo` only. The task: verify on our new ionic + decide if the
2× is real. Prior belief (correct, as it turned out): a driver bug that blatant
wouldn't ship → suspect measurement/usage.

## What / why / how / context of the conclusion

**WHAT:** The dmabuf "2×" is not real device-VRAM duplication. It is an artifact
of `hipMemGetInfo`'s "free" number, which drops again when the dmabuf is
imported/pinned for P2P (the P2PDMA/BAR mapping shows up as consumed in HIP's
view) even though no second physical VRAM allocation happens.

**WHY it looked real:** `hipMemGetInfo` was the *only* meter the probe used. It
genuinely shows −buffer at hipMalloc AND −buffer again at reg. With one meter you
can't tell "the reg allocated 4 GB more" from "the reg mapped the existing 4 GB
and HIP's free-accounting reflects the mapping".

**HOW we proved it:** cross-checked against the **amdgpu driver's own VRAM
bookkeeping** (`/sys/class/drm/card*/device/mem_info_vram_used`, the same source
rocm-smi reads). On an **idle** GPU, true VRAM = +buffer at hipMalloc, **+0 at
ibv_reg_dmabuf_mr**, at both 4 GiB and 64 GiB. A real 2× at 64 GiB would show
128 GiB; it showed 66. (See results/.)

**CONTEXT (source-backed mechanism):** dma-buf peer import is zero-copy *by
design* — the exporter (amdgpu) hands the importer (ionic) a physical-address
`sg_table` via `map_dma_buf`; the NIC DMAs VRAM directly, no copy. The only known
mechanism that WOULD double VRAM is amdgpu migrating the buffer to GTT when a peer
can't do PCIe P2P — but that path is gated on `!CONFIG_DMABUF_MOVE_NOTIFY`, and
this kernel has `CONFIG_DMABUF_MOVE_NOTIFY=y` + `CONFIG_PCI_P2PDMA=y`, so amdgpu
keeps it in VRAM. Consistent with the measured +0.
Refs: OpenFabrics "RDMA with GPU memory via dma-buf" (Xiong, Intel); kernel
dma-buf docs; drm/amdgpu "Don't pin VRAM without DMABUF_MOVE_NOTIFY" patch.

## Hypotheses raised and RULED OUT (all with evidence)

1. **Export API too old (v1) → use v2** — DEAD. HIP export, HSA v1, HSA v2/NONE,
   HSA v2/**PCIE** (`HSA_AMD_DMABUF_MAPPING_TYPE_PCIE`) all give identical reg cost.
2. **Missing PCIE mapping flag** — DEAD (covered above).
3. **Wrong access flags** — DEAD. LOCAL / +REMOTE_WRITE / +REMOTE_READ / ALL all same.
4. **malloc+reg merged into one allocation** — DEAD. Export cost is +0; the two
   −buffer drops are hipMalloc and reg separately.
5. **ionic driver bounce-copies per allocation** (the "real 2×" hypothesis) — DEAD.
   True amdgpu VRAM +0 at reg. The earlier memory note claiming "real per-alloc
   duplicate" was WRONG — it trusted hipMemGetInfo; retracted.

## `ibv_reg_mr` vs `ibv_reg_dmabuf_mr` (clarification — both zero-copy)

Common misconception: "`ibv_reg_mr` needs CPU pin/map, only dmabuf is GPU-direct."
FALSE. Both achieve GPUDirect (NIC DMAs VRAM directly). Difference is how the
kernel obtains GPU physical addresses:
- `ibv_reg_mr(gpu_ptr)` → vendor **peer-memory** callback (`get_pages`). Here that
  is amdgpu peermem + `ib_peer_mem` (loaded, used by ionic_rdma). Out-of-tree.
- `ibv_reg_dmabuf_mr(fd)` → standard **dma-buf** framework (`dma_buf_attach` /
  `map_attachment` → sg_table). Upstream since kernel 5.12.
On new ionic both work at 1×. On old ionic the peer-memory path was broken
(FAIL_14) so only dmabuf worked — that, not the "2×", was the colleague's real
result.

## Gotchas for the next person

- **Idle-GPU requirement:** absolute VRAM meters are meaningless if a foreign job
  is on the card. Our first run had card0 at 251 GiB (someone else) and the 4 GB
  signal was lost in noise + a coincidental 250 GB drain. Re-ran on an idle card.
- **Don't trust one VRAM meter.** `hipMemGetInfo` (process view) ≠ amdgpu true
  VRAM. And `rocm-smi --showmeminfo vram` absolute was outright broken here (read
  0.28 G even after a 64 GB hipMalloc). Use `mem_info_vram_used` sysfs.
- **libionic ABI:** new ionic 26.03 kernel driver wants libionic ABI 4 (54.0-187).
  Container images with 54.0-149 (ABI 1) see 0 RDMA devices ("does not support
  kernel ABI 1, supports 4-4"). Inject the host 54.0-187 .so + the
  `libibverbs/libionic-rdmav34.so` provider symlink.
- **RDMA reset discipline:** each run_*.sh removes its container at the end
  (dereg MR, close fd, free buffer). Verify VRAM back to ~2.2 G idle before next.

## Bottom line for infera / Phase 2

There is **no VRAM penalty** from dmabuf on new ionic. Both bare ibv_reg_mr
(peermem) and dmabuf register at 1×. The "don't use dmabuf, it costs 2×" concern
is retracted. Separately, a real infera-side item surfaced: the sglang image ships
libionic 54.0-149 (ABI 1); it should be bumped to 187 to match the 26.03 driver
(the vllm image already bakes 187).
