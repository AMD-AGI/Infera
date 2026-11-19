# PR U11：PP 下 KV offload 按最大 stage 统一每个 worker 的槽宽

- 目标：vllm-project/vllm main。分支（仓库外 `/home/liyingli/bench_agentx/upstream-20261119/vllm`，fork `llying-001/vllm`）：
  `u11-offload-pp-worker-bytes-main`，11-19 15:58 rebase 到 main `8828834fb`（一个提交 `de267df1a`，无冲突）；rebase 前测过的版本留在
  `u11-offload-pp-worker-bytes-main-f0c592bfd`（`754162ed3`）。`u11-offload-pp-worker-bytes` 是 v0.34.0 回移版。
- 提交方式（用户 11-19 定）：先在 fork 里开 PR，明天 review 后再决定上游怎么提。
- **查重（11-19）：与上游 open PR 部分重叠**，正文写明关系，由用户明天决定：
  - [#50653](https://github.com/vllm-project/vllm/pull/50653)（@StellaH-ter，08-01 起，"Fix CPU offload block count mismatch across PP ranks"，
    复现同样是 GLM-5.2 PP4）：把各 PP stage 的 CPU chunk 数统一成最小值（经 `CacheConfig.num_cpu_blocks`），修的是私有张量路径
    （`CPUOffloadingSpec`）上 scheduler 分出超出某个 worker 缓冲区的块号。它**不**统一每个 worker 的槽宽，所以 `TieringOffloadingSpec`
    的共享区（各 worker 的槽拼成一行）在各 stage 之间的区域大小 / 槽偏移仍然不一致——这是本 PR 修的部分。两者都改 `build_offloading_config`，会冲突。
  - [#68020](https://github.com/vllm-project/vllm/pull/68020)（@xsxszab，Inferact，draft）：给 canonical layout 加 PP 支持；本 PR 改的是非 canonical（直接）布局。
- **影响 CUDA（无法只限 ROCm）**：这是通用的正确性问题（CUDA 上 PP > 1 + `TieringOffloadingSpec` 同样起不来），修复会让较小 stage 的槽宽变成
  最大 stage 的值（GLM-5.2 PP4 多用 2–7% 主机内存）。新字段 `compare=False`，否则十几处 `KVCacheConfig` 相等断言会失败。**需要 CUDA 测试**
  （PR-PLAN 跨平台表）：CUDA 上 PP2 + `TieringOffloadingSpec` 起栈与读回、PP2 + `CPUOffloadingSpec` 读回，`tests/v1/kv_connector/unit/offloading_connector/`、`tests/v1/core/`。
- 状态：代码 + 单测完成；pre-commit 通过、新测试在 main 的代码上按预期失败（J5）；新 main 上的整组单测待 J5 `main2-u11`；端到端数据来自 v0.34.0（B 线）。

## 验证数据（都是单次）

| 项 | 结果 | 出处 |
|---|---|---|
| 单测，main `8828834fb` + PR `de267df1a` | <待 J5 `main2-u11`> | J5（作业 299267，v3-221） |
| 新测试跑在 main 的代码上 | `test_kv_cache_configs_record_largest_worker_bytes_per_block`（`AttributeError: 'KVCacheConfig' object has no attribute 'max_worker_bytes_per_block'`）与 `test_worker_slots_sized_by_largest_pipeline_stage[24-24]`（`assert 16 == 24`）失败，`[None-16]` / `[8-16]` 通过：`2 failed, 2 passed, 17 warnings in 11.42s` | J5 `newtests-on-base-u11` |
| pre-commit（5 个文件） | 通过 | J5 `precommit-vllm` |
| 单测，main `f0c592bfd`（rebase 前） | PR 1,710 / 基线 1,706；1 个失败两边都有（`test_finalize_extensible_kv_cache_auto_max_model_len_warns_when_short[True]`） | J1 `main-u11`（作业 298936） |
| 每块 KV 字节数（GLM-5.2 MXFP4 PP4，64 token / 块，FP8 + indexer + MTP） | PP0 796,416、PP1 / PP2 779,520、PP3 742,656 | B 线 README 2 节 |
| 聚合 PP4 复现（v0.34.0，GPU 800 块 × 64 token、CPU 层 2 GB、FS 层本地 NVMe，6 条约 2.4 万 token 带暗号的 prompt 再重复前 3 条） | 不修：PP0 建了 2.00 GB 的区域，PP3 等文件长到 1,998,323,712 字节，30 s 超时，引擎退出；修后：起栈 101 s，9/9 正确，FS 层命中 1,461 个 chunk、读回 4.66 GB；`CPUOffloadingSpec` + 修复回归 9/9 | B 线 README 2 节、`results/b2/agg/` |
| P4D4（v0.34.0 + 修复，`TieringOffloadingSpec` 1.25 TB + FS 层，3600 s） | 4 个 PP worker 加入同一块 1,250.00 GB 区域，无超时（这轮吞吐因 U19 的锁页失败不可比） | B 线 README 2 节（b2-r1） |
| 原始故障（v0.34.0，s6） | PP1 / PP2 等 `/dev/shm/vllm_offload_*.mmap` 长到 1,249,999,527,936 字节超时，PP0 已建好 1250.00 GB 的文件 | 主线 P4D4 README 1e |

## PR description (English, vLLM template)

**Title:** [Bugfix][KV Offload] Size every worker's slot by the largest pipeline stage

<!-- markdownlint-disable -->

## Overview

With pipeline parallelism, KV offload sizes each worker's slot of a CPU chunk from that worker's own `KVCacheConfig`,
but pipeline stages hold different layers. The workers of one engine (and the scheduler, which uses a copy of the first
stage's config) then disagree on the slot width and on the size of the shared CPU region, and `TieringOffloadingSpec`
fails to start with PP > 1. This records the largest KV bytes per block over all workers and sizes every slot by it.

## Claims

- `TieringOffloadingSpec` starts and round-trips correctly with PP > 1 (GLM-5.2 PP4: previously the last stage timed out
  after 30 s waiting for the shared region).
- Every worker and the scheduler agree on one chunk layout (slot width, chunk count, region size) under PP.
- Cost: smaller stages use slots as wide as the largest stage's (2-7% more host memory per smaller stage for GLM-5.2 PP4).

## Validation

**Unit tests** on MI355X, main `8828834fb` + this PR (`de267df1a`), in `vllm/vllm-openai-rocm:nightly-9c1d45c2` with the
tree's Python code installed over the image:

```bash
pytest -v tests/v1/core/test_kv_cache_utils.py tests/v1/core/test_kv_cache_planning.py \
  tests/v1/kv_connector/unit/offloading_connector tests/v1/kv_offload/tiering
```

```
<TBD from J5 main2-u11: PR and main summary lines>
```

The new tests run against main's code fail, as expected:

```
FAILED tests/v1/core/test_kv_cache_utils.py::test_kv_cache_configs_record_largest_worker_bytes_per_block
  E       AttributeError: 'KVCacheConfig' object has no attribute 'max_worker_bytes_per_block'
FAILED tests/v1/kv_connector/unit/offloading_connector/test_config.py::test_worker_slots_sized_by_largest_pipeline_stage[24-24]
  E       AssertionError: assert 16 == 24
================== 2 failed, 2 passed, 17 warnings in 11.42s ===================
```

`pre-commit run --files <the 5 changed files>` passes.

**End to end** (vLLM v0.34.0 with the same fix applied as a patch; the sizing code it fixes is the same on main).
GLM-5.2-MXFP4, TP1 x PP4 on
MI355X, KV bytes per block per stage: 796,416 / 779,520 / 779,520 / 742,656.

- Aggregated PP4 server, `TieringOffloadingSpec` with a 2 GB CPU tier and an FS tier on local NVMe, 800 GPU blocks of 64
  tokens; 6 prompts of ~24K tokens with a hidden passphrase, then the first 3 again:
  - main: stage 0 creates a 2.00 GB region; stage 3 waits for the file to reach 1,998,323,712 bytes and times out after
    30 s; the engine exits.
  - this PR: starts in 101 s; 9/9 answers correct; the FS tier serves 1,461 chunk hits (4.66 GB read back).
  - `CPUOffloadingSpec` with this PR: 9/9 correct.
- PD deployment, prefill PP4 with `TieringOffloadingSpec` (1.25 TB region + FS tier), 3600 s: the 4 workers join one
  1,250.00 GB region (on v0.34.0 without the fix, stages 1 and 2 timed out waiting for it).

## Details

- **Root cause.** `build_offloading_config` computes `worker_kv_bytes_per_block` as the worker's KV tensor size over
  `num_blocks`. Under PP each worker's config holds only its stage's layers, so the value differs per stage, and so do
  `kv_bytes_per_chunk` (world size x slot), `num_chunks` and each worker's slot offset in a row. With
  `TieringOffloadingSpec` the region's creator and the joining workers expect different file sizes; even when the sizes
  happened to match, the slot offsets would differ.
- **Fix.** `get_kv_cache_configs` already unifies `num_blocks` (min) and `needs_kv_cache_zeroing` (or) across workers;
  it now also records the largest per-worker KV bytes per block in `KVCacheConfig.max_worker_bytes_per_block`, which the
  scheduler config inherits. `build_offloading_config` sizes the slot by it (never below the worker's own value) and logs
  when it differs. The field is `compare=False` because it is derived from the configs.
- **Relation to #50653.** #50653 unifies the CPU chunk count across PP stages (minimum, via `CacheConfig.num_cpu_blocks`),
  which fixes out-of-range block ids on the private-tensor path (`CPUOffloadingSpec`). It does not unify the slot width,
  so the shared region of `TieringOffloadingSpec` still disagrees across stages; this PR unifies the slot width, which
  also makes the chunk count agree. The two touch `build_offloading_config` and would need reconciling; happy to fold
  this into #50653 or rebase on it. #68020 adds PP support to the canonical layout; this PR is about the direct layout.
- **Platforms.** This is not ROCm specific: CUDA with PP > 1 and `TieringOffloadingSpec` hits the same path. We could
  only validate on MI355X; CUDA validation (PP2 + `TieringOffloadingSpec` start-up and read-back, PP2 +
  `CPUOffloadingSpec` read-back) would be welcome. The unit tests are CPU-only and run in CI.
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
