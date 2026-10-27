---
review: false
checks:
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
---
## 任务描述

- 目标：（要把哪个版本或哪个已接受的改动做成可交付的代码：kit、overlay 补丁、router 改动、kernel 配置表……）
- 输入：（例如 @close/r<N> 的 `stack.json`，或某个实验任务）
- 代码基线：（仓库、分支；新开发分支的名字）

## 约束

- 补丁照 `deploy/docker/patches/` 的约定（anchor、幂等、README 和 patch.upstream.status.md 各加一行）。
- 交付前复测一次，确认收益还在、质量门通过。push、开 PR、发布先问我。

## 交付

- `report.md`：以"## 结论"开头，写改了什么、怎么验证的、分支和 commit。
