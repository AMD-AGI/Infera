# PR U26：ROCm 上 CPU offload 张量按精确大小锁页

- 目标：vllm-project/vllm main。分支（仓库外 `/home/liyingli/bench_agentx/upstream-20261119/vllm`）：`u26-rocm-exact-pin-main`
  （基于 main `f0c592bfd`，一个提交 `31d3966af`），`u26-rocm-exact-pin`（同一改动回移到 v0.34.0，只给 P4D4 用，不作上游依据）。
- 只影响 ROCm：精确锁页只在 `is_rocm()` 时启用；`CudaRTLibrary.cudaHostAlloc` / `cudaFreeHost` 用到时才解析，
  不进 `exported_functions`，CUDA 进程不会查找这两个符号。不需要 CUDA 测试（PR-PLAN 跨平台表）。
- 状态：代码 + 单测完成，main / v0.34.0 单测通过，主机内存对比与 P4D4（v0.34.0 回移版）都已测。**尚未提交到 GitHub。**
- 提交前：上游 main 已比 `f0c592bfd` 新 13 个提交（11-19 14:50 查，`79e937641`），其中改了 `vllm/envs.py`（本 PR 也改），rebase 可能有小冲突；
  挪到新 main 后在对应 nightly 上重跑 `main-u26`。提交信息里的 `Co-authored-by: Cursor` 保留（用户 11-19 定）。

## 验证数据（都是单次，除非注明）

| 项 | 结果 | 出处 |
|---|---|---|
| 单测，main `f0c592bfd`（nightly `9c1d45c2` 镜像 + 同步被测树的 Python 代码） | `tests/v1/kv_offload/cpu`：PR 254 通过 / 基线 251（多出的 3 个是新测试），无新增失败 | J1 `main-u26`（作业 298936，v3-057） |
| 单测，v0.34.0 镜像（回移版） | PR 226 / 基线 223，无新增失败 | J1 `v034-u26b` |
| 2 × 20 GB 主机内存（请求 37.25 GiB） | torch 锁页 `Shmem` +64.0 GiB、13.5 s；本 PR +37.25 GiB、7.8 s；`is_pinned()` 为真、GPU 往返正确；释放后 `Shmem` 回到起点 +0.7 GiB 以内 | J1 `u26-hostmem`（v0.34.0 镜像 + 回移版文件；`host_register.py` / `cuda_wrapper.py` 的改动与 main 版逐行相同） |
| P4D4 3600 s，本 PR 代码（v0.34.0 回移版），CPU 层 1.9 TB | 每个 PP worker 精确锁页：PP0 27 个张量 474.85 GB、PP1 25 / 474.53、PP2 25 / 474.53、PP3 24 / 474.51，合计 1,898.4 GB；运行中 `Shmem` 稳定在 1,805 GiB；无 OOM / `evict_process_worker` / 崩溃；43,269 tok/s/GPU（新节点，吞吐不归因于本 PR） | J3（作业 298999，v3-191），`results/pr-298999/pr-checks.txt` |
| P4D4，main + 本 PR（+ U12 / U13） | 排队中 | J3b（作业 299112） |
| P4D4 3600 s，覆盖层版本（同一 `hipHostMalloc` 分配） | f1 48,173 / f2 44,691 tok/s/GPU，两轮无崩溃 / 卡死；每 worker 474.5–474.9 GB | 主线 P4D4 README 1t（两台节点各一次） |
| 对照：torch 锁页分配器 | 1.25 TB 实际 1,360 GiB（+17%）；1.6 TB 时 4 个 worker 持有 2,481 GiB，主机（MemTotal 2,752 GiB）OOM（m8b） | 主线 P4D4 README 1o |

- 推算（非实测）：J3 的 101 个张量每个 17.6–19.8 GB，经 torch 分配器都会取整到 32 GiB，合计 3,232 GiB，超过主机 2,752 GiB。
- 注意（不影响 PR，但要知道）：J3 拆除容器（`docker rm -f`，即进程被强杀）30 s 后 `Shmem` 仍有 1,547 GiB（`meminfo-after.txt`），之后没有读数。
  主线 m8b 见过 OOM 后约 1.3 TB 在 2.5 min 内逐步释放到 8 GB（README 1o），所以更可能是内核 / KFD 释放慢，**未验证**。正常关停路径的释放由单测覆盖
  （`hipHostFree` 随最后一个视图调用）。J3b 结束时 `meminfo-after.txt` 会再有一次读数。

## PR description (English)

**Title:** [ROCm][KV Offload] Allocate private CPU offload tensors at their exact pinned size

### Problem

On ROCm, `CPUOffloadingSpec._uses_shared_region()` is False, so the CPU offload tier is one private pinned tensor per KV
cache tensor, created with `torch.zeros(..., pin_memory=True)`. PyTorch's caching host allocator rounds every pinned
allocation up to the next power of two, so the host memory actually pinned is up to twice `cpu_bytes_to_use`, and it
jumps whenever the per-tensor size crosses a power of two.

GLM-5.2-MXFP4 prefill, TP1 x PP4 on MI355X (24-27 KV tensors per worker, host with 2,752 GiB of memory):

| `cpu_bytes_to_use` | per tensor | pinned (4 workers) |
|---|---|---|
| 1.25 TB | ~15.6 GB -> 16 GiB | 1,360 GiB (1.46 TB, +17%) |
| 1.6 TB | ~20 GB -> 32 GiB | 2,481 GiB when the host ran out of memory and a worker was killed |

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
- MI355X, `tests/v1/kv_offload/cpu`: 254 passed with this PR, 251 on main (the difference is the new tests), no new
  failures.

### Validation (MI355X)

- Host memory for two 20 GB tensors (37.25 GiB requested), measured from `/proc/meminfo` `Shmem`:
  `torch.zeros(pin_memory=True)` +64.0 GiB in 13.5 s; `pinned_host_tensor` +37.25 GiB in 7.8 s, `is_pinned()` true,
  GPU round trip correct, and `Shmem` back to its starting value (within 0.7 GiB) once the tensors are dropped.
- GLM-5.2 PD, prefill TP1 x PP4 with `cpu_bytes_to_use=1.9e12`, one 3600 s AgentX run with this change: the 4 workers
  pinned 474.85 / 474.53 / 474.53 / 474.51 GB (101 tensors of 17.6-19.8 GB, 1,898 GB in total), host `Shmem` stayed at
  1,805 GiB for the whole run, and there was no OOM or crash. Through torch's allocator each of those tensors would be
  rounded to 32 GiB, 3,232 GiB in total, more than the host has.

Both measurements ran on a v0.34.0 build with this change applied (the change is identical); unit tests ran on main.

### Notes

- Pinned memory from `hipHostMalloc` is accounted as `Shmem`, like torch's pinned memory, and is released at shutdown
  (no allocator cache keeps it).
- Alternative considered: an anonymous mmap registered with `hipHostRegister` (as `pin_mmap_region` does for the shared
  region). It also sizes exactly, but uses registered user memory, which the driver handles differently from its own
  pinned allocations; the runtime allocation keeps the mechanism torch already uses.
