# 上游 PR 草稿（未提交）

2026-11-02。当前最佳 recipe（E10）里我写的两个 patch 的 PR 草稿；只在本目录，没有向任何仓库提交。

| 草稿 | diff | 说明 |
|---|---|---|
| `PR-evict-headroom.md` | `evict-headroom.diff` | 在 patch 本体上加了 `is_write_back` 条件（语法检查通过，**加条件后的版本没有跑过**；测过的版本只有环境变量） |
| `PR-hicache-backup-quota.md` | `hicache-backup-quota.diff` | `block_quota` 参数化，默认 2 不变 |

提交前要做的：rebase 到 SGLang main 并确认文件还在；环境变量挪进 `sglang.srt.environ`；在 main 上跑一遍相关单测；是否对外提交由用户决定。
第三个（PP 中间块 KV 竞态，`pp-chunk-wait`）不另提 PR：SGLang 已有同一竞态的 open PR #45524，先逐行比对再决定是评论补数据还是补充修改，见主报告。
