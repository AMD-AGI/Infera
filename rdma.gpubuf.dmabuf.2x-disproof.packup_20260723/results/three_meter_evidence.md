# Results — captured evidence

## 1. Colleague probe VERDICT on our new ionic (run_probe2.sh)
```
===== ENV =====
driver ionic      : 26.03.3.001
driver ionic_rdma : 26.03.3.001
firmware fw_ver   : 1.117.5-a-77
kernel            : 6.8.0-124-generic
RDMA devices visible: ['ionic_0'..'ionic_7']

=== TEST B: ibv_reg_dmabuf_mr on 4.0 GiB VRAM ===
  free VRAM: before alloc=287.37  after alloc=283.37  after reg=279.37 (GiB)
  buffer=4.00  alloc_cost=+4.00  reg_extra_cost=+4.00
  ibv_reg_dmabuf_mr: OK   ->  2x SHADOW (VRAM doubled on registration!)
=== TEST A: plain ibv_reg_mr on 256 MiB VRAM ===
  ibv_reg_mr(VRAM): OK  -> Mooncake works out-of-box, no code change

===================== VERDICT =====================
  bare ibv_reg_mr(VRAM) : OK
  dma-buf reg(VRAM)     : OK_2x     <-- the artifact (hipMemGetInfo only)
```

## 2. Export API / mapping-flag ablation (run_debug.sh -> dmabuf_debug.py)
All four export paths behave identically: export=+0, reg=+size. The v1/v2/PCIE
mapping flag is NOT the lever.
```
--- export_mode=hip ---         cost: malloc=+4.00 export=+0.00 reg=+4.00 -> 2x SHADOW
--- export_mode=hsa_v1 ---      cost: malloc=+4.00 export=+0.00 reg=+4.00 -> 2x SHADOW
--- export_mode=hsa_v2_none --- cost: malloc=+4.00 export=+0.00 reg=+4.00 -> 2x SHADOW
--- export_mode=hsa_v2_pcie --- cost: malloc=+4.00 export=+0.00 reg=+4.00 -> 2x SHADOW
```
access-flag sweep (reg_variants.py): LOCAL / +REMOTE_WRITE / +REMOTE_READ / ALL — all 2x (hipMemGetInfo).
double-register same buffer: 1st reg +2.00, 2nd reg (same alloc) +0.00.

## 3. THE DISPROOF — three-meter cross-check, idle GPU0 (run_hold2.sh, 4 GiB)
```
                          hipMemGetInfo free   amdgpu true VRAM (mem_info_vram_used, all cards)
PHASE=baseline            287.37               2.22
PHASE=after_malloc(4G)    283.37  (-4)         6.22  (+4)   <- real alloc, both agree
PHASE=after_export        283.37               6.22  (+0)
PHASE=after_reg (dmabuf)  279.37  (-4 again)   6.22  (+0)   <- NO extra VRAM. artifact exposed.
PHASE=after_dereg         283.37               6.22
PHASE=after_free          287.37               2.22         <- clean release
```

## 4. Confirm at PD scale — 64 GiB (run_hold_big.sh)
```
                          hipMemGetInfo free   amdgpu true VRAM
PHASE=baseline            287.37               2.22
PHASE=after_malloc(64G)   223.37  (-64)        66.22 (+64)
PHASE=after_reg (dmabuf)  159.37  (-64 again)  66.22 (+0)   <- true VRAM stays 66, NOT 128
PHASE=after_free          287.37               2.22
```
If the 2x were real, a 64 GiB registration would consume 128 GiB. It consumes 64.
Disproof complete.

## 5. Kernel dma-buf support (kern_dmabuf.sh)
```
CONFIG_PCI_P2PDMA=y   CONFIG_HSA_AMD_P2P=y   CONFIG_DMABUF_MOVE_NOTIFY=y
dmesg: amdgpu ... added peer-to-peer DMA memory 0x... (per GPU)
```
move_notify + P2PDMA both enabled -> amdgpu keeps the buffer in VRAM for P2P and
does NOT migrate to GTT, consistent with the +0 true-VRAM at registration.
