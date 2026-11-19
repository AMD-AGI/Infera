# PR U19：ROCm 上共享 offload 区域只注册本 worker 的槽

- 目标：vllm-project/vllm main。分支（仓库外 `/home/liyingli/bench_agentx/upstream-20261119/vllm`，fork `llying-001/vllm`）：
  `u19-rocm-pin-worker-slots-main`，11-19 15:05 rebase 到 main `8828834fb`（一个提交 `8bf00c65f`，无冲突）；rebase 前测过的版本留在
  `u19-rocm-pin-worker-slots-main-f0c592bfd`（`651ea418b`）。`u19-rocm-pin-worker-slots` 是 v0.34.0 回移版（P4D4 现在用 `CPUOffloadingSpec`，用不到）。
- 提交方式（用户 11-19 定）：先在 fork 里开 PR，用户明天在那里 review，review 后再开到上游。
  **fork PR：[llying-001/vllm#2](https://github.com/llying-001/vllm/pull/2)**（11-19 15:40 开，base = fork 的 main，已同步到上游 `9a2e60358`）。
- 只影响 ROCm：新门槛只在 `current_platform.is_rocm()` 时生效，其他平台保持"每行省 ≥ 3 MiB 才只注册本 worker 的槽"；原有测试
  `test_pin_mmap_region_selects_by_savings_not_row_count` 固定为非 ROCm。不需要 CUDA 测试。
- 适用场景：多个 worker 共用一块 `SharedOffloadRegion`（ROCm 上来自 `TieringOffloadingSpec`），每行由各 worker 的槽拼成。我们遇到它的
  GLM-5.2 PP4 部署另外用了 U11 的修复（PP > 1 + `TieringOffloadingSpec` 否则起不来），但本 PR 不依赖 U11，TP 多 worker 的共享区同样走这段逻辑。
- main 上 ROCm 仍是同步锁页：main 的后台锁页（`async_init`）只在 CUDA 上可用（`cpu/spec.py`），`pin_mmap_region` 的选择规则与 v0.34.0 相同，
  所以在 v0.34.0 镜像里做的测量对 main 适用。
- 查重（11-19）：上游没有改这个门槛的 open PR。相关的有 [#52296](https://github.com/vllm-project/vllm/pull/52296)（draft，协调多个 worker 的
  注册结果、失败时整组回退）——它让失败的结果一致，本 PR 是在 ROCm 上避免失败，两者互补。
- 状态：代码 + 单测完成；新 main 上整组单测、pre-commit 都通过，新测试在 main 的代码上失败、在 PR 上通过（都在 J4）。
- 开到上游前要确认：正文最后一条 "the submitter reviewed every changed line and the measurements" 属实。
- review 时请留意：新规则每行一次注册调用，行越小调用越多（见"代价"），3 MiB 门槛原本就是为了摊薄这部分开销。

## 验证数据（都是单次，除非注明）

| 项 | 结果 | 出处 |
|---|---|---|
| 单测，main `8828834fb` + PR `8bf00c65f`（nightly `9c1d45c2` + 同步被测树的 Python 代码） | PR `814 passed, 1 skipped, 18 warnings in 390.20s`，main `812 passed, 1 skipped … in 446.73s`，无新增失败；新增 2 个用例各 0.34 s | J4 `main2-u19`（作业 299174，v3-221） |
| 新测试跑在 main 的代码上 | `test_pin_mmap_region_rocm_registers_own_slots_whenever_a_page_is_saved[2]` / `[4]` 失败（main 整块注册：2 个 worker 时一次注册 4,778,496 B，而不是 3 个 798,720 B 的槽），`test_pin_mmap_region_selects_by_savings_not_row_count` 通过：`2 failed, 3 passed, 17 warnings in 3.03s` | J4 `newtests-on-base` |
| pre-commit（2 个文件） | 17 个适用的 hook 全过（ruff、typos、mypy、测试接入 CI、docstring 等） | J4 `precommit-u13u19` |
| 单测，main `f0c592bfd`（rebase 前） | `tests/v1/kv_offload/cpu/test_shared_offload_region.py` + `tests/v1/kv_offload/tiering`：PR 814 / 基线 812，无新增失败 | J2a `main-u19`（作业 299106，v3-015） |
| 锁页测量：1 TB 区域，GLM-5.2 PP4 布局（行 3,186,688 B = 4 个 796,416 B 的槽按页取整），4 个 worker 进程同时调 vLLM 自己的 `pin_mmap_region(worker_slot_only=True)` | 现规则（每个 worker 注册整块）：rank 0 / 3 在 687.18 GB 处 `cudaHostRegister failed (code=1)`、整块退回可分页，rank 1 / 2 成功但用了 608.0 / 616.2 s；只注册本 worker 的槽：4/4 成功，每个 worker 313,805 次调用、38.5–41.2 s | J1b `u19-pin-measure`（作业 298969，v3-006，v0.34.0 镜像；"只注册槽"用 `MIN_HOST_REGISTER_SAVED_BYTES = 0` 实现，即门槛 = 一页，与本 PR 在 ROCm 上的规则相同） |
| 实际部署中的失败（B 线，P4D4 + `TieringOffloadingSpec` 1.25 TB + U11 覆盖层） | `host_register failed for rank=0 at 549.74 of 1250.00 GB`、rank 2 在 755.90 GB 失败，这两个 stage 的 CPU 层退成可分页内存 | B 线 README 2 节（b2-r1，v3-258） |

- `u19-pin-measure` 的 rc=2 是两组测量都结束后，步骤脚本在运行中被改动导致 "Stale file handle"（HANDOFF 第 4 节），测量数据完整。
- 代价（由上表推算）：每个 worker 约 7,600–8,200 次注册 / s（4 个 worker 同时）。1 TB、3.2 MB 行是 31 万次（约 40 s）；行只有 0.5 MB 时是
  200 万次，按这个速率约 4 min。整块注册在 1 TB 时本身要 10 min，所以这个布局下新规则更快，但对行很小的模型会变慢。

## PR description (English, vLLM template)

**Title:** [ROCm][KV Offload] Register only each worker's slots of a shared offload region

<!-- markdownlint-disable -->

## Overview

When several workers share one CPU offload region, `pin_mmap_region(worker_slot_only=True)` registers the whole region
in every worker unless registering only the worker's slots saves at least 3 MiB per row. On ROCm the driver refuses part
of those registrations (4 PP workers x 1.25 TB), and the affected stages fall back to pageable copies. On ROCm, this PR
registers only the worker's slots whenever that saves at least a page per row.

## Claims

- ROCm: each worker registers only its own slots of a shared offload region whenever that saves a page per row. Other
  platforms keep the 3 MiB rule.
- MI355X, 1 TB region with the GLM-5.2 PP4 layout, 4 workers pinning at once: **4/4 workers pinned in 38.5-41.2 s**,
  where the current rule pinned 2/4 (the other two failed at 687 GB and stayed pageable) and took 608-616 s.
- This is the failure we hit serving GLM-5.2 with `TieringOffloadingSpec` at PP4 (1.25 TB region): ranks 0 and 2 failed
  at 549.74 and 755.90 GB.

## Validation

**Unit tests** on MI355X, main `8828834fb` + this PR (`8bf00c65f`), in `vllm/vllm-openai-rocm:nightly-9c1d45c2` with the
tree's Python code installed over the image:

```bash
pytest -v tests/v1/kv_offload/cpu/test_shared_offload_region.py tests/v1/kv_offload/tiering
```

```
PR:   =========== 814 passed, 1 skipped, 18 warnings in 390.20s (0:06:30) ============
main: =========== 812 passed, 1 skipped, 18 warnings in 446.73s (0:07:26) ============
```

The 2 new cases take 0.34 s each; the modified `test_pin_mmap_region_selects_by_savings_not_row_count` (3 cases)
0.34-0.39 s each. Run against main's code, the new test fails, as expected (main registers each row in full):

```
FAILED tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_pin_mmap_region_rocm_registers_own_slots_whenever_a_page_is_saved[2]
FAILED tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_pin_mmap_region_rocm_registers_own_slots_whenever_a_page_is_saved[4]
E       AssertionError: assert [(1048576, 4778496)] == [(1843200, 79...9888, 798720)]
=================== 2 failed, 3 passed, 17 warnings in 3.03s ===================
```

`pre-commit run --files tests/v1/kv_offload/cpu/test_shared_offload_region.py vllm/v1/kv_offload/cpu/gpu_worker.py`
passes.

**Registration on MI355X.** A 1 TB `SharedOffloadRegion` with the GLM-5.2 PP4 layout under `TieringOffloadingSpec`
(3,186,688-byte rows of four 796,416-byte slots), joined by 4 worker processes (one GPU each) that call vLLM's
`pin_mmap_region(worker_slot_only=True)` at the same time. Single run, on a v0.34.0 build: `pin_mmap_region`'s
selection rule is the same on main, and the "own slots" case was run with `MIN_HOST_REGISTER_SAVED_BYTES = 0`, i.e. the
page threshold this PR applies on ROCm. On ROCm main still pins synchronously (`async_init` is CUDA-only).

| | workers pinned | per worker |
|---|---|---|
| whole region per worker (current rule) | 2 of 4: ranks 0 and 3 failed at 687.18 GB and stayed pageable | 608.0 / 616.2 s for the two that succeeded (15 calls) |
| own slots only (this PR's rule) | 4 of 4 | 313,805 calls, 38.5-41.2 s |

<details>
<summary>Raw output</summary>

```
== whole region (current rule)
cudaHostRegister failed (code=1) — transfers will still work but may be slower (unpinned DMA)
host_register failed for rank=0 at 687.18 of 1000.00 GB; the offload region stays pageable
cudaHostRegister failed (code=1) — transfers will still work but may be slower (unpinned DMA)
host_register failed for rank=3 at 687.18 of 1000.00 GB; the offload region stays pageable
{"mode": "full", "rank": 0, "pinned": false, "registrations": 0, "seconds": 590.1, "region_gb": 1000.0}
{"mode": "full", "rank": 1, "pinned": true, "registrations": 15, "seconds": 608.0, "region_gb": 1000.0}
{"mode": "full", "rank": 2, "pinned": true, "registrations": 15, "seconds": 616.2, "region_gb": 1000.0}
{"mode": "full", "rank": 3, "pinned": false, "registrations": 0, "seconds": 577.7, "region_gb": 1000.0}
== own slots only
{"mode": "slots", "rank": 0, "pinned": true, "registrations": 313805, "seconds": 38.5, "region_gb": 1000.0}
{"mode": "slots", "rank": 1, "pinned": true, "registrations": 313805, "seconds": 39.0, "region_gb": 1000.0}
{"mode": "slots", "rank": 2, "pinned": true, "registrations": 313805, "seconds": 41.2, "region_gb": 1000.0}
{"mode": "slots", "rank": 3, "pinned": true, "registrations": 313805, "seconds": 39.4, "region_gb": 1000.0}
```

</details>

## Details

- **Root cause.** With 4 workers, a GLM-5.2 PP4 row is 3,186,688 bytes and a worker's slot 796,416 bytes, so
  registering only the slot saves 2.39 MB per row, under `MIN_HOST_REGISTER_SAVED_BYTES` (3 MiB). Each worker then
  registers the whole region, i.e. the same memory is registered 4 times, and on ROCm the driver stops accepting the
  registrations part-way.
- **Fix.** On ROCm the threshold is one page, so a worker registers its page-rounded slots (one call per row) whenever
  that saves anything. Other platforms are unchanged.
- **Tradeoff.** The cost is one registration call per row: ~8,000 calls/s per worker here, so ~40 s for a 1 TB region of
  3.2 MB rows. Smaller rows mean more calls (0.5 MB rows: ~2 M calls, ~4 min at this rate), but registering the whole
  region 4 times over was slower still (~10 min per worker for 1 TB). If reviewers prefer, the ROCm rule could instead
  switch on the total size registered (workers x region size).
- **Related.** #52296 (draft) coordinates registration outcomes across the workers of a region so that a failure leaves
  all of them consistently unpinned; this PR avoids the failure on ROCm. The two are complementary.
- **CI.** `tests/v1/kv_offload` runs on AMD CI. The new test stubs `host_register` and the platform check: the real
  failure needs several TB of registrations, which the measurement above covers.
- **AI assistance** was used to write this change, its tests and this description; the submitter reviewed every changed
  line and the measurements.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
