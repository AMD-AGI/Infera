<!-- 草稿，未提交。Mooncake 第一个 PR（分配器 + 经典传输引擎 + TENT 拒绝 GEM 显存）。
     标题：[TransferEngine][ROCm] Add HipGemAllocator so pinned KV caches are not charged twice by KFD
     分支 llying-001:feat/hip-gem-allocator（本地 80a3cab0，基于 main 504f2ae4）；21 个文件 +852/−12，
     按 CONTRIBUTING 口径（不算 kernel/data/config/test）566 行，其中 42 行是三个新文件的许可证头。
     11-16 加了 GEM 显存设备号修正（crusoe 发现 HIP 把导入显存报成 GPU 0）。11-17 在 n02-25（6.19.14）上跑了 81/83/84/86/90/91；92 门禁 + 网卡计数器已填；[待填] 只剩 RFC issue、第二个 PR、SGLang PR 的编号。 -->

## Description

RFC: #[待填]. TENT support follows in #[待填], stacked on this PR. Companion SGLang PR: sgl-project/sglang#[待填].

On amdgpu-dkms 6.18.4 through at least 7.1.9, and on kernels without torvalds/linux [9131cc85d0de](https://github.com/torvalds/linux/commit/9131cc85d0dedd8ba4cae60dbe5985d081551f30) ("drm/amdkfd: correct vram_available calculation", v7.4-rc1), the KFD allocation limit (`amdgpu_amdkfd_reserve_mem_limit()`) subtracts `vram_pin_size` without crediting KFD's own pins back. A `hipMalloc` buffer that a NIC pins is then counted twice: once in `kfd.vram_used` and once as pinned. When a prefill or decode server registers a KV cache larger than the VRAM left after allocating it, every later device allocation of the process fails; in SGLang the scheduler segfaults in `c10::hip::HIPStream::stream()` at the first read of a new stream after the KV registration (ROCm 10 reports `hipErrorOutOfMemory` there instead). This happens with dma-buf and with peer-memory registration alike.

The KFD pin anchor (`MC_HIP_KFD_PIN_FIX`, #4657) does not help on these drivers: it raises `kfd.vram_pinned`, which `hipMemGetInfo` credits but the allocation limit no longer does, so the reported free VRAM looks right while allocations still fail.

This PR adds `hip_gem_allocator.so`, a `torch.cuda.memory.CUDAPluggableAllocator` backend exposed as `mooncake.allocator.HipGemAllocator` (the `NVLinkAllocator` pattern). Each buffer is an amdgpu GEM BO created on the GPU's render node and imported with `hipImportExternalMemory`. A dma-buf import is not reserved in `kfd.vram_used`, so the pinned BO is counted once, on every driver: the upstream fix's `amdgpu_amdkfd_account_vram_pin()` skips BOs without a `kfd_bo`. A one-page dma-buf MR pins the BO at allocation, so `hipMemGetInfo` covers it from the start and it cannot be evicted before registration.

HIP reports no address range for imported memory, ROCr's `hsa_amd_portable_export_dmabuf` cannot export it, and HIP has no IPC handle for it, so the transfer engine looks a buffer up in the allocator (`mc_hip_gem_lookup()`, found with `dlsym`; the Python wrapper loads the library `RTLD_GLOBAL`) and:

- registers it through the BO's own dma-buf whatever `MOONCAKE_DISABLE_HIP_DMABUF` says, failing if the kernel lacks dma-buf P2P (`isKernelDmabufSupported()` is split so the kernel probe can be asked on its own);
- never through `ibv_reg_mr()`: the amdgpu peer-memory client's `amd_get_pages()` dereferences `bo->kfd_bo`, which is NULL for a GEM BO. Registration refuses that path for these buffers and the KFD pin anchor skips them;
- leaves it out of the HIP transport (the existing "skip non-device memory" path), so with no `hip` buffer published for its range, same-host transfers to it are routed over RDMA;
- takes its GPU from the allocator: `hipPointerGetAttributes` reports memory imported with `hipImportExternalMemory` as device 0's whichever device imported it, so `mc_hip_gem_lookup()` also returns the device a buffer was allocated on. `getMemoryLocation()` uses it for the location the buffer is registered under, which picks the NICs on both ends of a transfer; otherwise every GPU's KV cache is located on GPU 0 and goes through GPU 0's NIC. The HIP and TCP transports use it for the device they copy from;
- in TENT, which cannot export these buffers yet, makes the `ibv_reg_mr()` fallback and the MR warm-up refuse them, so registering one under `MC_USE_TENT` fails instead of reaching the peer-memory client. The follow-up PR adds the dma-buf export.

Docs: the allocator and `MC_HIP_GEM_PIN_NIC`, and where the KFD pin anchor does not help (the current text says amdgpu-dkms 6.19.14 counts a dma-buf pin once; its source and our measurements say otherwise).

## Suggested Merge Priority

- [ ] P0 — Urgent: Critical security vulnerability, data loss/corruption, or widespread production outage requiring immediate action
- [ ] P1 — High: Significant correctness or stability issue in existing functionality, or a regression blocking users without a practical workaround
- [x] P2 — Normal: Routine bug fix, validated performance/compatibility improvement, or new feature for a confirmed use case
- [ ] P3 — Low: Experimental feature, optional enhancement, refactoring, or cleanup with no urgent user impact

## Module

- [x] Transfer Engine (`mooncake-transfer-engine`)
- [x] Integration (`mooncake-integration`)
- [x] Docs

## Type of Change

- [x] Bug fix (works around a driver accounting bug)
- [x] New feature (`HipGemAllocator`)
- [x] Documentation update

## How Has This Been Tested?

AMD MI355X (288 GB), ionic RNICs, amdgpu-dkms 6.19.14.31400100 (`n04-29`) and 6.16.13 (`n04-33`). Mooncake built inside `lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261109` the way that image builds it (`-DUSE_HIP=ON -DUSE_HIP_DMABUF=ON`, no etcd, no store).

**Test commands:**
```bash
cmake .. -DUSE_HIP=ON -DUSE_HIP_DMABUF=ON -DUSE_ETCD=OFF -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF \
    -DBUILD_UNIT_TESTS=ON -DUSE_TENT=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 -GNinja
ninja engine build_hip_gem_allocator hip_gem_allocator_test rdma_hip_kfd_pin_test tent
export ROCR_VISIBLE_DEVICES=0,1 MC_TEST_NICS=ionic_0 MC_GID_INDEX=1
./mooncake-transfer-engine/tests/hip_gem_allocator_test
./mooncake-transfer-engine/tests/rdma_hip_kfd_pin_test
```

**Test results:**
- [x] Unit tests pass
- [x] Integration tests pass (loopback and end to end, below)
- [x] Manual testing done (described below)

- Driver accounting, without Mooncake (HIP, HSA and libibverbs only): after registering 40 GiB, the largest allocation the driver still accepts drops by 80 GiB on 6.19.14 for dma-buf and for peer memory, and on 6.16.13 for dma-buf (40 GiB for peer memory). With 150 GiB registered through peer memory, a new HIP stream fails on 6.19.14 and works on 6.16.13. The same 40 / 150 / 200 GiB allocated as GEM BOs drop it by 40 / 150 / 200 GiB on both drivers.
- `hip_gem_allocator_test` on 6.19.14, two GPUs visible: 4/4 pass. In `LocationNamesTheAllocatingGpu`, `hipPointerGetAttributes` reports device 0 for a buffer allocated on GPU 1, and its location is `hip:1`. `rdma_hip_kfd_pin_test`: 2/2 pass.
- TENT builds with the change (`-DUSE_TENT=ON`).
- torch + Mooncake loopback (`batch_register_memory` of 6 GiB KV buffers, then a new `torch.cuda.Stream(priority=0).cuda_stream`, then an RDMA write across a 1 GiB MR boundary, verified), HIP transport on, `MOONCAKE_DISABLE_HIP_DMABUF=1`, 6.19.14:

  | KV | Result |
  |---|---|
  | 150 GiB, `HipGemAllocator` | charged once at allocation (KFD free VRAM 287.4 → 137.2 GiB), unchanged by registration; the new stream and a further 8 GiB allocation succeed; write verified, 16.0 GB/s |
  | 198 GiB, `HipGemAllocator` | 287.4 → 89.2 GiB, unchanged by registration; the new stream and 8 GiB more succeed; write verified, 15.6 GB/s |
  | 150 GiB, `hipMalloc`, peer memory | `hipMemGetInfo` reports 887.2 GiB free after registration; segfault at the new stream |
  | 150 GiB, `hipMalloc`, dma-buf + `MC_HIP_KFD_PIN_FIX=1` | `hipMemGetInfo` reports 137.2 GiB free (the anchor's correction); segfault at the new stream |

  Same-pair, same-NIC RDMA write bandwidth is the same for `hipMalloc` + peer memory, `hipMalloc` + dma-buf and GEM + dma-buf (15.4 GB/s on one node, 28.0–28.3 GB/s on the other, three rounds each); GPU copy and read bandwidth of GEM buffers match `hipMalloc` buffers.
- End to end, GLM-5.2 with SGLang (main `44ad9d69f`) PD disaggregation, one-node 4+4 on 6.19.14, `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM` (companion SGLang PR): starts and serves with 121.77 GB of KV per prefill GPU and 141.74 GB per decode GPU registered over 8 RNICs; smoke 6/6, GSM8K probe 197/200, no garbled outputs, 16k-token needle 32/32, MTP acceptance 3.64. That run predates the device fix. With it, on another 6.19.14 node with the same image and settings: smoke 6/6, GSM8K probe 196/200, no garbled outputs, 16k-token needle 31/32 (one runaway) and GSM8K 30/32, MTP acceptance 3.58; the KV went over the NICs of the four prefill GPUs, 14–15 GB each (`MC_ENABLE_DEST_DEVICE_AFFINITY` makes the loopback's receiving NIC the sending one). The same gate on the same node with the engine from before the fix (every rank's KV located on GPU 0) gave the same accuracy (probe 198/200, needle 32/32, GSM8K 31/32, acceptance 3.57), but the KV went over two NICs only and took about 18 times the packets for the same requests (265 million against 15 million), with 256–476 thousand duplicate responses per NIC against 7–10 thousand, and 6 completion errors: retransmissions. On another cluster, where the receiving NIC drops frames, the same mislocation put 122 TB of a 1P1D run on one NIC and ended in a prefill out-of-memory as transfers piled up; with an equivalent fix in a preloaded allocator the 8 NICs carried 15–22 TB each. Without the GEM pool, the same nightly image with its own Mooncake failed at prefill start-up on another 6.19.14 node in every try: in the three runs whose prefill logs were kept, all four prefill ranks segfaulted in `HIPStream::stream()` right after the KV registration (two more runs, with its Mooncake or rdma-core swapped, did the same).
- `clang-format` 20.1.8 (`scripts/code_format.sh --check --changed-lines`) and `typos` 1.30.2 pass on the changed files.

## Checklist

- [x] Every commit I contribute includes my `Signed-off-by` trailer certifying the [DCO](https://github.com/kvcache-ai/Mooncake/blob/main/DCO)
- [x] I have performed a self-review of my own code
- [x] I have formatted my code using `./scripts/code_format.sh`
- [ ] I have run pre-commit on the files changed in this PR and all hooks pass
- [x] I have updated the documentation (if applicable)
- [x] I have added tests to prove my changes are effective
- [x] For changes >500 LOC: I have filed an RFC issue (#[待填]; 566 lines by the CONTRIBUTING count, 42 of them license headers)

## AI Assistance Disclosure

- [ ] No AI tools were used
- [x] AI tools were used (specify below)

AI coding assistants helped trace the driver accounting, write the allocator, the transfer-engine changes, the test and the docs, and run the builds and hardware checks above. I reviewed the change.
