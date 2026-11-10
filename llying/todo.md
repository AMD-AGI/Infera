# TODO（llying）

## 1. GLM-5.2 在 SGLang main 上：ROCm 的 MLA KV 写回改走 staged 路径，可能比 E10 的 `lf_pf`@16 慢（11-10 记，未实测）

### 背景

- **E10**（`prefill-opt-20261027/README.zh-CN.md` 10-29 07:30 / 13:50 节）：驱逐写回用的 `transfer_kv_all_layer_mla_lf_pf` 从 2 个 workgroup 改成 16 个，
  KV 每次驱逐 278 → 125 ms（`results/test_e10_backup_patch-1029T0701.txt`），C104 吞吐 +3.3%。
- **E10 的镜像带 B5 补丁**（`deploy/docker/patches/sglang_rocm/patch_hicache_rocm_staged_write_back.py`，标记 `GLM52_ROCM_STAGED_WRITE_BACK`）。
  它在 ROCm 上把 MLA pool 的 `can_use_write_back_jit` 关掉，所以 MLA KV 走的是 `lf_pf`。`test_e10_backup_patch.py` 也写死了 `can_use_write_back_jit=False`，
  从来没有测过 staged 路径。
- **SGLang main（`ed068770`，11-10）和原样 v0.5.22 不需要 B5**：MLA pool 的门控是 `_is_cuda or _is_hip`；controller 按 pool 分别安排 indices
  （`HostPoolGroup.supports_per_pool_backup_indices`、`HybridCacheController._move_write_operation`），v0.5.16 上那个崩溃走不到
  （`inferencex-publish-plan-20261027/README.zh-CN.md` B5）。所以 ROCm 上 MLA KV 写回走 staged JIT（`jit_transfer_hicache_all_layer_mla_staged_lf_pf`），
  DSA indexer 仍走 `lf_pf`（它的门控只在 CUDA 上开）。
- **staged 在 ROCm 上怎么拷**：先用 relayout kernel 把 64 页收集进显存里的 staging buffer，再逐页 `hipMemcpyAsync` 到 host。
  `staged_write_back.cuh` 的 `try_copy_page_first_pages_batch` 在 `USE_ROCM` 下直接返回 false，没有 CUDA 那样的批量拷贝；这条路径不受 `block_quota` 影响。
  GLM-5.2 一页是 64 token × 20 层 × 576 B ≈ 737 KB，一次 6.14 GB 的驱逐约 8,300 次 `hipMemcpyAsync`。
- **推算**：sglang#28534 的 MI355X 表里，与 GLM-5.2 最接近的是 24 层 × 512 B（每页约 786 KB）：staged 30.86 GiB/s，`lf_pf`（quota 2）24.34 GiB/s。
  staged 的带宽随每页字节数上升（每页约 10 MB 时约 50 GiB/s），符合"逐页调用有固定开销"。E10 实测的 `lf_pf`@16 是 6.14 GB / 125 ms ≈ 45.7 GiB/s。
  粗估 main 上一次驱逐的 KV 部分约 185 ms（E10 是 125 ms）。这是拿别人的 microbench（bf16、每次 64 页）类推的。
- **影响**：换到 main 或原样镜像（去掉 B5）后，E10 的收益可能丢掉一部分；sglang#48042 合入也补不回来，因为它只管 `lf_pf`。
  按 E10 的弹性（四级驱逐之和少 0.41 s 换来 +3.3%）线性折算，每级多 60 ms 约等于 −2%。

### 要做的

- [ ] **microbench**（空闲 GPU，数据采集，非性能测试）：生产形状（KV 20 层 × 576 B，64 token/页，一次驱逐 75 个节点 ≈ 6.14 GB），
  比较 staged JIT、`lf_pf`@2、`lf_pf`@16，并逐字节核对。以 `prefill-opt-20261027/scripts/test_e10_backup_patch.py` 为基础；
  staged 那一组要设 `can_use_write_back_jit=True`、`dst_indices` 放在 CPU、配 staging buffer（`_WRITE_BACK_STAGING_PAGE_CHUNK = 64` 页）。
  - 预测（开跑前写下）：staged 30–35 GiB/s，`lf_pf`@16 约 46 GiB/s，`lf_pf`@2 约 21 GiB/s。
  - 若 staged ≥ `lf_pf`@16 的 90%，这一项关闭，不再做后面的步骤。
- [ ] **端到端**：E10 同一工作点（1 prefill TP1 × PP4 + 1 decode TP4/DP4，AgentX C104，同一对节点），各计分一次，并记录驱逐 metrics。
  - A = 当前 E10（B5 + quota 16）；B = 去掉 B5（MLA 走 staged，indexer 走 `lf_pf`@16）。
  - 预测：B 的每级驱逐均值比 A 高约 60 ms，吞吐低约 2%。
  - 判据：B 的 PP1–PP3 驱逐均值比 A 高 ≥ 40 ms，或吞吐低 ≥ 1%，就认定 staged 在这个形状上更慢。
- [ ] **若 staged 更慢**，候选修法（每项先离线估收益）：
  1. 改进 staged 的 ROCm 实现：把 host 目标地址连续的页合并成一次 `hipMemcpyAsync`；查 ROCm 有没有批量拷贝接口。
  2. ROCm 上这种小页形状让 MLA 走 `lf_pf`@16。要考虑 `write_through` / 后台写回时与 forward 争 CU，以及 sglang#35043 提到的
     mmap 注册内存在 GPU 上地址不同的问题。
  3. 过渡：上游修好之前，Infera 镜像继续带 B5。#48042 按 reviewer 的建议合入后（all-layer MLA op 在 ROCm 上默认 `_default_mla_block_quota()` = 16），
     不设环境变量也是 16。
- [ ] 结果回帖到 sglang#48042，或单独开 issue / PR。

### 相关

- [sglang#48042](https://github.com/sgl-project/sglang/pull/48042)（xiezhq-hermann 11-10 的 review：问 MLA KV 用的是哪个 kernel；建议复用 `SGLANG_HICACHE_BLOCK_QUOTA`）
- [sglang#28534](https://github.com/sgl-project/sglang/pull/28534)（ROCm 打开 MLA / MHA 的 staged 写回，含 MI355X 对比表，对照组是 `lf_pf`@2）
- [sglang#30350](https://github.com/sgl-project/sglang/pull/30350)（打开其余三个 pool 的 ROCm 门控，已关闭未合入）
- [sglang#35043](https://github.com/sgl-project/sglang/pull/35043)（DeepSeek-V4 pool 和 DSA indexer 的 ROCm staged 写回，OPEN）
