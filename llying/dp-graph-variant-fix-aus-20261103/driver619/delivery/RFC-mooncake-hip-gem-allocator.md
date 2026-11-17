<!-- 已提交：https://github.com/kvcache-ai/Mooncake/issues/5583（11-17）。Mooncake issue，用 RFC 模板（标题前缀 [RFC]:，标签 RFC）。提完马上提第一个 PR，不等回复。
     标题：[RFC]: HipGemAllocator: KV caches on AMD GPUs without the KFD double charge of amdgpu 6.18+ -->

### Changes proposed

#### Motivation

On amdgpu-dkms 6.18.4 through at least 7.1.9 (every release checked in between), and on kernels without torvalds/linux [9131cc85d0de](https://github.com/torvalds/linux/commit/9131cc85d0dedd8ba4cae60dbe5985d081551f30) ("drm/amdkfd: correct vram_available calculation", in mainline from v7.4-rc1), the KFD allocation limit `amdgpu_amdkfd_reserve_mem_limit()` is

```
vram_size - pt - ras - vram_pin_size
```

where 6.16 had `... - vram_pin_size + kfd.vram_pinned`. A `hipMalloc` buffer is already in `kfd.vram_used`; once a NIC pins it (dma-buf or the amdgpu peer-memory client), it is also in `vram_pin_size`, so it is charged twice. `hipMemGetInfo` (`amdgpu_amdkfd_get_available_memory()`) still credits `kfd.vram_pinned`, so what it reports and what allocations get disagree.

For PD disaggregation this breaks start-up whenever the registered KV cache is larger than the VRAM left after allocating it, which is the normal case (KV takes most of HBM). On MI355X with 6.19.14, GLM-5.2 prefill servers segfault in `c10::hip::HIPStream::stream()` right after Mooncake registers the KV cache (the first new stream needs VRAM for its queue); ROCm 10 reports `hipErrorOutOfMemory` instead. Without Mooncake (HIP, HSA and libibverbs only), registering 40 GiB lowers the largest accepted allocation by 80 GiB on 6.19.14, for dma-buf and for peer memory.

The upstream fix is not in any released amdgpu-dkms, and clusters are being upgraded to these drivers now. The KFD pin anchor (`MC_HIP_KFD_PIN_FIX`, #4657) only helps on drivers whose limit credits `kfd.vram_pinned` back; on 6.18.4+ it fixes what `hipMemGetInfo` reports but not what allocations get.

#### Proposal

Allocate the buffers a NIC will pin as amdgpu GEM BOs instead of KFD BOs. A BO created with `DRM_IOCTL_AMDGPU_GEM_CREATE` on the GPU's render node and imported into HIP with `hipImportExternalMemory` (dma-buf fd) is not reserved in `kfd.vram_used`, so pinning it charges it once on every driver; the upstream fix's `amdgpu_amdkfd_account_vram_pin()` skips BOs without a `kfd_bo` too.

1. `hip_gem_allocator.so`, a `torch.cuda.memory.CUDAPluggableAllocator` backend exposed as `mooncake.allocator.HipGemAllocator` (the `NVLinkAllocator` pattern), so frameworks put the KV cache in a `torch.cuda.MemPool`. A one-page dma-buf MR pins each BO at allocation, so `hipMemGetInfo` covers it from the start and it is not evicted before registration. It exports `mc_hip_gem_lookup()` for the transfer engine.
2. Transfer engine handling of these buffers, which HIP reports no address range or IPC handle for and ROCr cannot export:
   - register through the BO's own dma-buf (`ibv_reg_dmabuf_mr`), whatever `MOONCAKE_DISABLE_HIP_DMABUF` says;
   - never through `ibv_reg_mr()`: the amdgpu peer-memory client's `amd_get_pages()` dereferences the GEM BO's NULL `kfd_bo`;
   - leave them out of the HIP transport, so same-host transfers to them go over RDMA;
   - take their GPU from the allocator: HIP reports imported memory as device 0's whichever device imported it, and the buffer's location (`hip:<device>`) picks the NICs on both ends of a transfer.
3. TENT: refuse these buffers first (PR 1), then register them through the same dma-buf (PR 2).
4. SGLang: `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM` puts the KV cache in this pool (separate SGLang PR).

#### Alternatives considered

- Backport the driver fix: correct, and we have a backport that builds for amdgpu-dkms 6.19.14, but it needs every cluster to rebuild its kernel module until AMD ships a release with it.
- `hipMalloc` with a smaller KV cache: the double charge scales with the registered size, so it costs about half the KV capacity.
- Extend the KFD pin anchor: on 6.18.4+ nothing a pin can do raises the allocation limit.

#### Plan

- PR 1 (#5584): allocator, classic transfer engine, TENT refusal, test (`hip_gem_allocator_test`), docs.
- PR 2: TENT dma-buf export, tested with the same test under `MC_USE_TENT` (#5587, a draft until PR 1 merges).
- SGLang PR (sgl-project/sglang#50938): the `AMD_GEM` memory pool option, opened alongside PR 1 and merged once a Mooncake release has PR 1.

Opt-in: nothing changes unless a caller uses `HipGemAllocator`.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues and read the [documentation](https://kvcache-ai.github.io/Mooncake/)
