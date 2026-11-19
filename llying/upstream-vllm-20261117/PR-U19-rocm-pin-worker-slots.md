# PR U19：ROCm 上共享 offload 区域只注册本 worker 的槽

- 目标：vllm-project/vllm main。分支（仓库外 `/home/liyingli/bench_agentx/upstream-20261119/vllm`）：`u19-rocm-pin-worker-slots-main`
  （基于 main `f0c592bfd`，一个提交 `651ea418b`），`u19-rocm-pin-worker-slots`（v0.34.0 回移版；P4D4 现在用 `CPUOffloadingSpec`，用不到）。
- 只影响 ROCm：新门槛只在 `current_platform.is_rocm()` 时生效，其他平台保持"每行省 ≥ 3 MiB 才只注册本 worker 的槽"；原有测试
  `test_pin_mmap_region_selects_by_savings_not_row_count` 固定为非 ROCm。不需要 CUDA 测试。
- 适用场景：多个 worker 共用一块 `SharedOffloadRegion`（ROCm 上来自 `TieringOffloadingSpec`），每行由各 worker 的槽拼成。
  GLM-5.2 PP4 的实际场景还需要 U11（否则 PP > 1 + `TieringOffloadingSpec` 起不来），但本 PR 不依赖 U11：TP / DP 多 worker 的共享区同样走这段逻辑。
- main 上 ROCm 仍是同步锁页：main 新增的后台锁页（`async_init`）只在 CUDA 上可用（`cpu/spec.py`），`pin_mmap_region` 的选择规则与 v0.34.0 相同，
  所以在 v0.34.0 镜像里做的测量对 main 适用。
- 状态：代码 + 单测完成，main 单测通过，1 TB 区域 4 worker 的锁页测量完成。**尚未提交到 GitHub。**
- 提交前：上游 main 比 `f0c592bfd` 新 13 个提交，未碰本 PR 的文件；挪到新 main 后重跑 `main-u19`。
- review 时请留意：新规则每行一次注册调用，行越小调用越多（见下面"代价"），3 MiB 门槛原本就是为了摊薄这部分开销。

## 验证数据（都是单次，除非注明）

| 项 | 结果 | 出处 |
|---|---|---|
| 单测，main `f0c592bfd`（nightly `9c1d45c2` + 同步被测树的 Python 代码） | `tests/v1/kv_offload/cpu/test_shared_offload_region.py` + `tests/v1/kv_offload/tiering`：PR 814 通过 / 基线 812（多出的 2 个是新测试），无新增失败 | J2a `main-u19`（作业 299106，v3-015，2 卡非独占） |
| 单测，v0.34.0（回移版） | PR 772 / 基线 770（不作上游依据） | J2a `v034-u19` |
| 锁页测量：1 TB 区域，GLM-5.2 PP4 布局（行 3,186,688 B = 4 个 796,416 B 的槽按页取整），4 个 worker 进程同时调 vLLM 自己的 `pin_mmap_region(worker_slot_only=True)` | 现规则（每个 worker 注册整块）：rank 0 / 3 在 687.18 GB 处 `cudaHostRegister failed (code=1)`、整块退回可分页，rank 1 / 2 成功但用了 608.0 / 616.2 s；本 PR（只注册本 worker 的槽）：4/4 成功，每个 worker 313,805 次调用、38.5–41.2 s | J1b `u19-pin-measure`（作业 298969，v3-006，v0.34.0 镜像），`results/u19-pin-measure/measure.jsonl` |
| 实际部署中的失败（B 线，P4D4 + `TieringOffloadingSpec` 1.25 TB + U11 覆盖层） | `host_register failed for rank=0 at 549.74 of 1250.00 GB`、rank 2 在 755.90 GB 失败，这两个 stage 的 CPU 层退成可分页内存 | B 线 README 2 节（b2-r1，v3-258） |

- `u19-pin-measure` 的 rc=2 是两组测量都结束后，步骤脚本在运行中被改动导致 "Stale file handle"（HANDOFF 第 4 节记过的问题），测量数据完整，不影响结论。
- 代价（由上表推算）：本 PR 下每个 worker 约 7,600–8,200 次注册 / s（4 个 worker 同时）。调用次数 = 区域大小 / 行大小：
  1 TB、3.2 MB 行是 31 万次（约 40 s）；若行只有 0.5 MB，同样 1 TB 是 200 万次，按这个速率约 4 min。整块注册在 1 TB 时本身就要 10 min，
  所以这里的布局下新规则更快，但对行很小的模型会变慢。可选的改法（PR 讨论时提）：ROCm 上按"worker 数 × 区域大小"或注册调用次数选规则。

## PR description (English)

**Title:** [ROCm][KV Offload] Register only each worker's slots of a shared offload region

### Problem

When several workers share one `SharedOffloadRegion` (each row is the workers' slots side by side),
`pin_mmap_region(worker_slot_only=True)` registers only the worker's own slots if that saves at least
`MIN_HOST_REGISTER_SAVED_BYTES` (3 MiB) per row, and otherwise registers the whole region. With
`TieringOffloadingSpec` and GLM-5.2 at PP4 on MI355X a row is 3.19 MB and a worker's slot 0.80 MB, so only 2.4 MB per row
would be saved and each of the 4 workers registers the whole region: 4x the region's size in registrations.

On ROCm the driver does not accept that. With a 1.25 TB region two of the four workers failed part-way
(`host_register failed for rank=0 at 549.74 of 1250.00 GB; the offload region stays pageable`, rank 2 at 755.90 GB),
leaving those pipeline stages on slow pageable copies.

### Change

On ROCm, register only the worker's slots whenever that saves at least a page per row (`min_savings = page`). Other
platforms keep the 3 MiB rule.

### Tests

- `test_pin_mmap_region_rocm_registers_own_slots_whenever_a_page_is_saved[2/4]`: with the GLM-5.2 PP4 slot size
  (2.4 MB saved per row, under 3 MiB) on ROCm, each worker registers exactly its own page-rounded slots, one range per
  row, each smaller than a row.
- `test_pin_mmap_region_selects_by_savings_not_row_count` now pins the platform to non-ROCm, so it keeps covering the
  3 MiB rule.
- MI355X, `tests/v1/kv_offload/cpu/test_shared_offload_region.py` and `tests/v1/kv_offload/tiering`: 814 passed, 812 on
  main (the difference is the new cases), no new failures.

### Validation (MI355X)

A 1 TB region with the GLM-5.2 PP4 layout (3,186,688-byte rows of four 796,416-byte slots), joined by 4 worker
processes (one GPU each) that call `pin_mmap_region(worker_slot_only=True)` at the same time:

| | workers pinned | per worker |
|---|---|---|
| whole region per worker (current rule) | 2 of 4: ranks 0 and 3 failed at 687.18 GB and stayed pageable | 608.0 / 616.2 s for the two that succeeded (15 calls) |
| own slots only (this PR) | 4 of 4 | 313,805 calls, 38.5-41.2 s |

(Single run, on a v0.34.0 build; `pin_mmap_region`'s selection rule is the same on main, and on ROCm main still pins
synchronously since `async_init` is CUDA-only.)

### Notes

- Cost: one registration call per row. Here that is ~8,000 calls/s per worker, so a 1 TB region of 3.2 MB rows takes
  ~40 s; a region with much smaller rows takes proportionally more calls (e.g. 0.5 MB rows: ~2 M calls, ~4 min at this
  rate), though registering the whole region 4 times over is not cheap either (~10 min per worker for 1 TB here). If
  reviewers prefer, the ROCm rule could instead switch on the total registered size (workers x region size) or on the
  number of calls.
