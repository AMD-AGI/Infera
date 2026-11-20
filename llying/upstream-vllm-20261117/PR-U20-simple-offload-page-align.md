# PR U20：Simple CPU offload 切不开时按页对齐层起点——与上游 #66594 重复，不开 PR

- 分支（仓库外）：`u20-simple-offload-page-align-main`（基于 main `f0c592bfd`，`83d93e2a5`）、`u20-simple-offload-page-align`（v0.34.0 回移版）。
- **11-19 查重结论：不开 PR。** 上游 open PR [#66594](https://github.com/vllm-project/vllm/pull/66594)（@cjackal，11-09 起，11-17 更新）
  "[Bugfix][KV Offload] Fix SimpleCPUOffload pinning for host pools over 256 GiB" 修的是同一个起栈失败
  （`No block and page boundary fits the registration chunk limit`，#60130 引入的回归），做法几乎相同：`host_registration_ranges` 加
  `tensor_offsets` 参数，`host_tensor_offsets()` 在一个层会跨过注册分块时把它的起点挪到页边界（本分支是"打包布局切不开时每层起点都页对齐"）。
  #66594 只在需要的地方补齐，填充更少。按 vLLM `AGENTS.md` 的查重规定不再开。
- 可在 #66594 下补的数据（明天 review 后由用户定）：GLM-5.2 MXFP4 PP4、MI355X、Simple offload 自动 `num_cpu_blocks=392,382` 时起栈失败，
  `389,120` 可以（绕法）；B 线 3.5 节的 P4D4 3600 s（Simple lazy = 28,696 tok/s/GPU）。
- **第三批里的"nightly 上 offload 测试清理阶段随机中止"已由上游修好，不提 issue（11-20 查）**：我们 J1 `main-u20` 的整组运行里 PR 和基线两边都
  `Fatal Python error: Aborted`，中止点在 conftest 的 `cleanup_dist_env_and_memory` → `torch.accelerator.empty_cache` / `empty_host_cache`，
  当时前面测试留下的 `copy_backend.py:_copy_loop` 线程还活着（PR 那次 1 个、基线 13 个），中止的位置每次不同。上游
  [#68710](https://github.com/vllm-project/vllm/pull/68710)（@bogdan-petkovic，11-19 22:52 合并）报的是同一个签名（还多一行
  `rocclr/device/device.cpp:362 : Memobj map does not have ptr: 0x0`）：`SimpleCPUOffloadConnector` 没有 `shutdown()`，引擎在进程内拆掉后拷贝线程不停、
  锁页的 CPU 缓存不取消注册，同一进程里后来的注册落到同一地址就让 ROCm 运行时中止；修法是加 `SimpleCPUOffloadWorker.shutdown()` 并由 connector 调用。
  它进 nightly 之后，`main-*` 步骤就不再需要 `PER_FILE=1`。
- 本分支的测试数据保留作素材：main `f0c592bfd` 上 `tests/utils_/test_host_memory.py` + `tests/v1/simple_kv_offload/test_worker_layout.py`
  PR 282 / 基线 275（逐文件进程，因 main 原有的清理阶段崩溃，见第三批的 nightly 清理崩溃 issue）。

## 给上游 #66594 的留言稿（英文，待用户 review，**未发**）

> 依据：B 线 3.5 节的起栈失败与绕法（v0.34.0，单次）。没有用 #66594 的代码测过，所以只说"看起来是同一根因"，并提出可以帮测。

Another instance of this start-up failure, on ROCm: GLM-5.2-MXFP4 at TP1 x PP4 on MI355X with
`SimpleCPUOffloadConnector` (vLLM v0.34.0). With the automatic `num_cpu_blocks=392382` start-up failed with
`No block and page boundary fits the registration chunk limit`; `num_cpu_blocks=389120` worked, which is the workaround
we used. It looks like the same root cause (layers packed after one whose size is not a page multiple leave no block
boundary on a page boundary). We had a local fix along the same lines (start the layers on a page when the packed
layout cannot be split) and would rather see this PR land; happy to check this branch on our configuration.
