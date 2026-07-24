# Environment (pinned)

## Hardware / node
- **Node:** chi2798 (via jump host root@149.28.124.225). k8s partition.
- **GPU used:** card1 = PCI `0000:75:00.0`, gfx950 (MI355X), device id `1002:75a3`.
  - VRAM total: 309220868096 B (~288 GiB). KFD node partition size 294896 MiB.
  - **BAR0 = 512 GiB, 64-bit prefetchable (Resizable/Large-BAR)** — covers all
    288 GiB VRAM. This is why P2P into VRAM is possible without GTT bounce.
  - Card selection: `HIP_VISIBLE_DEVICES=0` → HIP dev0 → PCI 75:00.0 → card1.
    (card4 was occupied by another user's `mlperf_gptoss` container — avoided.)
- **NIC:** ionic_0 = PCI `0000:79:00.0`, **AMD Pensando RoCE HCA** (vendor
  0x1dd8). BAR0=32KiB, BAR2=2MiB, BAR4=64MiB (doorbell). RoCEv2 / Ethernet
  link-layer. GID index 1 (idx0 is link-local).
  - **ODP caps: `NO SUPPORT`** — `ibv_devinfo -d ionic_0 -v` prints
    `rc_odp_caps: NO SUPPORT` (and uc/ud/xrc the same; general empty). The
    driver actively declares no On-Demand Paging → dynamic-attach / move-notifier
    dma-buf impossible → **`ibv_reg_dmabuf_mr` is FORCED to the pin path**
    (`ib_umem_dmabuf_get_pinned`), and KFD double-counts that pin.

## peermem state (why ibv_reg_mr is not an option)
- **No peermem module loaded** (`lsmod | grep peermem` empty; none packaged
  under `/lib/modules/$(uname -r)`). Plain `ibv_reg_mr` GPU-direct requires a
  peermem module (`ib_peer_mem`/amdgpu-peermem) — absent here, so `ibv_reg_mr`
  cannot register GPU memory. dmesg shows `amdgpu: PeerDirect support was
  initialized successfully` (the dma-buf/PeerDirect side is up), so the ONLY
  usable GPU-direct registration path is `ibv_reg_dmabuf_mr`.
  Check script: `scripts/peermem_chk.sh`.

## Software (host)
- **Kernel:** `6.8.0-107-generic`.
- **amdgpu:** DKMS out-of-tree, **version 6.16.13**, srcversion
  `A6F143BEC60C0AFC3263226`, file
  `/lib/modules/6.8.0-107-generic/updates/dkms/amdgpu.ko.zst`.
  dmesg: `amdgpu version: 6.16.13` + `PeerDirect support was initialized
  successfully`.
- **Kernel config (verified 3 ways: /boot/config, kallsyms symbol, dmesg):**
  - `CONFIG_DMABUF_MOVE_NOTIFY=y` (symbol `dma_buf_move_notify` present)
  - `CONFIG_PCI_P2PDMA=y` (symbol `pci_p2pdma` present)
  - `CONFIG_HSA_AMD_P2P=y`, `CONFIG_DMABUF_HEAPS=y`
- **rdma-core / libibverbs:** `/lib/x86_64-linux-gnu/libibverbs.so.1`
  (→ .1.14.50.0), exports `ibv_reg_dmabuf_mr@@IBVERBS_1.12`.
- **ionic provider:** `libionic-rdmav34.so → libionic.so.1.1.54.0-187`
  (Feb 6 install). Container matched host EXACTLY (same symlink/version).

## Software (container — where the MVP runs)
- **Image:** `rocm/jax-training:maxtext-v26.5` (the only ROCm image on chi2798).
- **ROCm:** 7.2.0 (`/opt/rocm-7.2.0`, `libamdhip64.so.7.2.70200`).
- **Build tools:** hipcc (gfx950), g++, hsa headers, verbs.h all present.
- **Container run (kill/run each round, NEVER stop/restart):**
  ```
  docker run -d --name dmabuf_probe --network host --ipc host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --group-add render -v /mnt/vast:/mnt/vast \
    rocm/jax-training:maxtext-v26.5 sleep infinity
  ```
  `--device=/dev/infiniband` is REQUIRED (without it: `ionic:0`, `no pd`).
  Container's ionic provider + libibverbs matched host bit-for-bit (verified).

## Instruments used (and their reliability — see notes.md)
- **`hipMemGetInfo` free (hip_free):** RELIABLE. Tracks the HIP *allocatable
  budget*. Every phase matched the malloc ground truth.
- **amdgpu TTM buddy free** (`/sys/kernel/debug/dri/0000:75:00.0/amdgpu_vram_mm`,
  line with `total:/free:`): RELIABLE for *physical* VRAM, **host-only**
  (debugfs not mounted in container). ~settle lag on free.
- **sysfs `mem_info_vram_used` / `mem_info_gtt_used`:** BO-counter. Tracks
  hipMalloc, but **does NOT track dma-buf pin** (blind to the reg path) and has
  ~3 s down-edge settle lag. Do NOT use its "no change" to argue physical cost.
- **rocm-smi `--showmeminfo vram`:** UNRELIABLE in-container (pid-ns global
  counter, did not move on hipMalloc). Discarded.
- **dmesg KFD/pin/reserve:** SILENT during registration — pin does not emit
  printk. Inconclusive as an instrument for this path (not evidence of absence).

## External dependencies / secrets
- Shared fs: `/mnt/vast` (world-writable; our home base
  `/mnt/vast/c_huggingface`). Compute nodes cannot see `/tmp` — stage via
  `/mnt/vast`.
- Cluster access: SSH ProxyJump via root@149.28.124.225 (jump host = chi2866).
  No secret values are stored here.
