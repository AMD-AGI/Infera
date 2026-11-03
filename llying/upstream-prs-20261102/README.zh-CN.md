# 当前最佳 recipe 的 4 个 patch：上游状态（2026-11-03 更新）

| # | patch | 上游状态 | 结论 |
|---|---|---|---|
| 1 | `deepseek_v2.py`：PP 中间级读 `embed_tokens.embedding_dim` 崩 | **已合入 SGLang**：[#21194](https://github.com/sgl-project/sglang/pull/21194)（10-29 合并，用 `getattr`）。main 现在是 `getattr(self.embed_tokens, "embedding_dim", None) == 7168`。#28827、#34465 已关闭 | **不需要 PR**。GLM-5.2 hidden size 是 6144，`== 7168` 的分支在所有 stage 上本来就不进，行为与我们的 `config.hidden_size == 7168` 补丁一致。v0.5.22 镜像没有这个修复，所以我们还要挂 patch；换更新的镜像后可去掉 |
| 2 | `pp-chunk-wait`（v3）：PP 中间块 KV 发送前等 forward | 同一竞态的 [#45524](https://github.com/sgl-project/sglang/pull/45524)，**open**，`REVIEW_REQUIRED`；17 项失败的检查全是汇总 gate（`pr-gate`、`pr-test-*finish`），不是具体测试 | **不另提 PR**。我们 patch 的第一处与 #45524 逻辑相同（在 `process_prefill_chunk` 里 `pp_size > 1` 时先 `record` 一个 event 到 forward stream，再 `send_kv_chunk`；#45524 用 `self.device_module.Event()`，我们用 `torch.cuda.Event()`，ROCm 上等价）。第二处（`maybe_send_cached_prefix_chunk` 里 `enable_overlap or pp_size > 1`）**不在 #45524 里**：作者 10-29 的评论说他们测过——PP 下该早发在 `run_batch` 的 `forward_stream_ctx` 内执行，页索引拷贝已经排在 forward 之后，72/72 对，所以删掉了这一处。→ 我们的第二处很可能不需要（我们没有单独验证过），后续可改成只用 #45524 的 8 行 |
| 3 | `evict-headroom` | 已提交 [#48047](https://github.com/sgl-project/sglang/pull/48047) | 环境变量改名 `SGLANG_HICACHE_EVICT_HEADROOM_TOKENS`；只在 `write_back` 生效；含单测（5 项，在 v0.5.22 镜像里对 main 的代码跑过，测试工具用桩） |
| 4 | `hicache-backup-quota` | 已提交 [#48042](https://github.com/sgl-project/sglang/pull/48042) | 环境变量 `SGLANG_HICACHE_BACKUP_BLOCK_QUOTA`；未设置时不传参，默认行为不变；含单测 |

提交人是 `llying-001`（fork `llying-001/sglang`）；PR 描述里写明了：测试在 v0.5.22 构建上做，main 上只跑了单测和 import 检查，没有跑 main 的端到端。
没有给别人的 PR（#45524）发评论；如果要补我们的实测数据（PP4 + async 下长上下文捞针 13/64 → 64/64），需要你确认。

本目录里的 `PR-*.md` 和 `*.diff` 是提交前的草稿（对 v0.5.22 文件），以上面的 PR 为准。
