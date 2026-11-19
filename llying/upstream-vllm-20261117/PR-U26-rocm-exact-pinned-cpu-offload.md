# PR U26：ROCm 上 CPU offload 张量按精确大小锁页

- 目标：vllm-project/vllm main。分支（仓库外 `/home/liyingli/bench_agentx/upstream-20261119/vllm`）：`u26-rocm-exact-pin-main`
  （基于 main `f0c592bfd`，一个提交），`u26-rocm-exact-pin`（同一改动基于 v0.34.0，P4D4 用）。
- 只影响 ROCm（11-19 按用户要求改）：精确锁页只在 `is_rocm()` 时启用；`CudaRTLibrary.cudaHostAlloc` / `cudaFreeHost` 用到时才解析，
  不进 `exported_functions`，CUDA 进程不会查找这两个符号。
- 状态：代码 + 单测完成；单测在 v0.34.0 与 main 上通过；主机内存对比（J1 `u26-hostmem`）与 P4D4 端到端（J3）待补。**尚未提交到 GitHub。**

## 验证数据（都是单次，除非注明）

| 项 | 结果 | 出处 |
|---|---|---|
| 单测，v0.34.0 镜像 | `tests/v1/kv_offload/cpu`：PR 226 通过 / 基线 223（多出的 3 个是新测试），无新增失败 | J1 `v034-u26b`（作业 298936，v3-057） |
| 单测，main `f0c592bfd`（nightly `9c1d45c2` 镜像 + 同步 main 的 Python 代码） | PR 254 通过 / 基线 251，无新增失败 | J1 `main-u26` |
| 2 × 20 GB 主机内存（torch 锁页 vs 本 PR） | 待测（J1 `u26-hostmem`）；此前覆盖层的同类测试：torch +64 GiB，`hipHostAlloc` +37 GiB，分配 6.3 s 对 14.4 s | `../vllm-pd-tools-20261117/scripts/exact_pin_test2.py` |
| P4D4 3600 s，CPU 层 1.9 TB（覆盖层版本，同一 HIP 锁页分配） | f1 48,173 / f2 44,691 tok/s/GPU，两轮无崩溃 / 卡死；每个 PP worker 锁页 474.5–474.9 GB，`Shmem` 1,805 GiB | 主线 P4D4 README 1t（两台节点各一次） |
| P4D4 3600 s，本 PR 代码 | 待测（J3） | — |
| 对照：torch 锁页分配器 | 1.25 TB 实际 1,360 GiB（+17%）；1.6 TB 涨到 2.6 TB 以上，主机 OOM（m8b） | 主线 P4D4 README 1o |

## PR description (English)

**Title:** [ROCm][KV Offload] Allocate private CPU offload tensors at their exact pinned size

### Problem

On ROCm, `CPUOffloadingSpec._uses_shared_region()` is False, so the CPU offload tier is one private pinned tensor per KV
cache tensor, created with `torch.zeros(..., pin_memory=True)`. PyTorch's caching host allocator rounds every pinned
allocation up to the next power of two, so the host memory actually pinned is up to twice `cpu_bytes_to_use`, and it
jumps whenever the per-tensor size crosses a power of two.

GLM-5.2-MXFP4 prefill, TP1 x PP4 on MI355X (24-27 KV tensors per worker):

| `cpu_bytes_to_use` | per tensor | pinned (4 workers) |
|---|---|---|
| 1.25 TB | ~15.6 GB -> 16 GiB | 1,360 GiB (1.46 TB, +17%) |
| 1.6 TB | ~20 GB -> 32 GiB | grew past 2.6 TB; the host (2.75 TB) ran out of memory and a worker was killed |

CUDA is not affected: it uses the shared mmap region, which `pin_mmap_region` registers at its exact size.

### Change

- `CudaRTLibrary`: `cudaHostAlloc` / `cudaFreeHost` (HIP: `hipHostMalloc` / `hipHostFree`), returning raw error codes
  like `cudaHostRegister`, and resolved on use (not in `exported_functions`), so other platforms never look them up.
- `host_register.pinned_host_tensor(num_rows, row_bytes)`: on ROCm, allocate exactly `num_rows * row_bytes` pinned bytes
  with the runtime (the allocation underneath torch's pinned allocator, without the rounding) and view them as an int8
  tensor. The memory is freed when the last view is released (a `weakref.finalize` on the buffer that
  `torch.frombuffer` keeps alive). Returns None off ROCm or on failure (the pending error is drained).
- `CPUOffloadingWorker`: on ROCm, private tensors use it, falling back to `torch.zeros(pin_memory=True)`. The handlers
  share and clear the `cpu_tensors` list, so the worker keeps its own references and drops them only after every
  transfer direction has stopped (and after a device sync if a handler failed).
- `VLLM_KV_OFFLOAD_EXACT_PIN` (default 1) in `envs.py`; 0 restores the old allocation. Ignored for compile hashing.

### Tests

- `test_pinned_host_tensor_is_exact_and_freed_with_its_last_view` (ROCm): the runtime is asked for exactly 15,000 bytes
  (torch would round to 16 KiB), the tensor is pinned, and the memory is freed only when the last view goes away.
- `test_worker_private_cpu_tensors_use_exact_pinned_memory[True/False]` (ROCm): private tensors take the exact path
  (or not, with the switch off), stay alive after a handler's shutdown clears the shared list, and are freed by the
  worker's shutdown.
- `test_transfer` now asserts that its private-tensor cases run on exact-size memory on ROCm, so its GPU <-> CPU round
  trips cover the new path.
- Results: MI355X, `tests/v1/kv_offload/cpu` 254 passed (main 251), no new failures.

### Validation (MI355X)

- Host memory, 2 x 20 GB tensors: <TBD from u26-hostmem: torch vs this PR, Shmem delta, allocation time>.
- GLM-5.2 PD (prefill TP1 x PP4) with a 1.9 TB CPU tier: each worker pins 474.5-474.9 GB, 1.90 TB in total, where torch's
  allocator OOMs the host above 1.25 TB; two 3600 s AgentX runs completed without errors. <TBD: run with this PR's code>.

### Notes

- Pinned memory from `hipHostMalloc` is accounted as `Shmem`, like torch's pinned memory, and is released at shutdown
  (no allocator cache keeps it).
- Alternative considered: an anonymous mmap registered with `hipHostRegister` (as `pin_mmap_region` does for the shared
  region). It also sizes exactly, but uses registered user memory, which the driver handles differently from its own
  pinned allocations; the runtime allocation keeps the mechanism torch already uses.
