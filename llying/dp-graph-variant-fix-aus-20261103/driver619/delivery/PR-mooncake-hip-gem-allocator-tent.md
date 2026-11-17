<!-- 已以 draft 提交：https://github.com/kvcache-ai/Mooncake/pull/5587（11-17），#5584 合入后变基到 main 再转为正式。Mooncake 第二个 PR（TENT 通过 dma-buf 注册 GEM 显存），等第一个 PR 合入后以 main 为基。
     标题：[TENT][ROCm] Register HipGemAllocator buffers through their dma-buf
     分支 llying-001:feat/hip-gem-allocator-tent（本地 be8a8047，叠在 80a3cab0 上）；4 个文件 +40/−3。
     11-17 在 n02-25 上编译、单测通过（81、91）；第一个 PR #5584、RFC #5583 已填；分支已推送，等 #5584 合入后以 main 为基提交。 -->

## Description

**Depends on #5584** (RFC #5583), which adds `mooncake.allocator.HipGemAllocator` and makes TENT refuse its buffers. Draft until #5584 merges: this branch is stacked on it, so the diff here also shows #5584's commits. This PR's own change is the last commit, [d64161e4](https://github.com/llying-001/Mooncake/commit/d64161e4b48468c99cc15504d07ba807737129ca) (4 files, +41/−3). Once #5584 merges I will rebase onto main and mark this ready for review.

A `HipGemAllocator` buffer is an amdgpu GEM BO imported into HIP. ROCr cannot export it as a dma-buf, and `ibv_reg_mr()` must never see it (the amdgpu peer-memory client dereferences the GEM BO's NULL `kfd_bo`). `RocmPlatform::exportDmabuf()` now looks the buffer up in `hip_gem_allocator.so` (`mc_hip_gem_lookup()`) and returns a new descriptor of the BO's dma-buf and the buffer's offset in it, so `RdmaContext` registers it with `ibv_reg_dmabuf_mr()`. Ordinary HIP memory keeps the default (`NotImplemented`, registered with `ibv_reg_mr()`). HIP reports imported memory as device 0's, so `RocmPlatform::getLocation()` and `getPointerDeviceId()` take a GEM buffer's device from the allocator as well, as the classic transfer engine does in #5584; they run per transfer, so they only use an allocator that registration already found. The docs line for the allocator now says TENT registers these buffers.

## Suggested Merge Priority

- [ ] P0 — Urgent: Critical security vulnerability, data loss/corruption, or widespread production outage requiring immediate action
- [ ] P1 — High: Significant correctness or stability issue in existing functionality, or a regression blocking users without a practical workaround
- [x] P2 — Normal: Routine bug fix, validated performance/compatibility improvement, or new feature for a confirmed use case
- [ ] P3 — Low: Experimental feature, optional enhancement, refactoring, or cleanup with no urgent user impact

## Module

- [x] Transfer Engine (`mooncake-transfer-engine`)
- [x] Docs

## Type of Change

- [x] New feature

## How Has This Been Tested?

AMD MI355X, ionic RNICs, amdgpu-dkms 6.19.14.31400100, built inside `lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261109` as in #5584.

**Test commands:**
```bash
cmake .. -DUSE_HIP=ON -DUSE_HIP_DMABUF=ON -DUSE_ETCD=OFF -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF \
    -DBUILD_UNIT_TESTS=ON -DUSE_TENT=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 -GNinja
ninja build_hip_gem_allocator hip_gem_allocator_test tent
# TENT with RDMA only, so a buffer TENT cannot register through its dma-buf fails instead of going over TCP
export ROCR_VISIBLE_DEVICES=0,1 MC_TEST_NICS=ionic_0 MC_GID_INDEX=1 MC_USE_TENT=1 MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 \
    MC_TENT_CONF='{"transports":{"tcp":{"enable":false},"io_uring":{"enable":false}}}'
./mooncake-transfer-engine/tests/hip_gem_allocator_test
```

**Test results:**
- [x] Unit tests pass: `hip_gem_allocator_test` through TENT, 4/4. It registers a 4 GiB `HipGemAllocator` buffer without lowering KFD free VRAM, creates a stream afterwards, and an RDMA write past the buffer's first 1 GiB lands, and the buffer allocated on GPU 1 is located on GPU 1.
- [ ] Integration tests pass (if applicable)
- [ ] Manual testing done (describe below)

## Checklist

- [x] Every commit I contribute includes my `Signed-off-by` trailer certifying the [DCO](https://github.com/kvcache-ai/Mooncake/blob/main/DCO)
- [x] I have performed a self-review of my own code
- [x] I have formatted my code using `./scripts/code_format.sh`
- [ ] I have run pre-commit on the files changed in this PR and all hooks pass
- [x] I have updated the documentation (if applicable)
- [x] I have added tests to prove my changes are effective (the existing `hip_gem_allocator_test`, run under `MC_USE_TENT`)
- [ ] For changes >500 LOC: I have filed an RFC issue (43 lines)

## AI Assistance Disclosure

- [ ] No AI tools were used
- [x] AI tools were used (specify below)

AI coding assistants helped write the change and run the build and the test above. I reviewed the change.
