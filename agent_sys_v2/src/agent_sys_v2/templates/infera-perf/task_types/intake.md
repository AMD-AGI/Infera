---
review: true
checks:
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - '"$AGENT_SYS_PYTHON" -m agent_sys_v2.perfcheck intake "$AGENT_SYS_TASK"'
---
## 任务描述

- 第 N 轮的增量问卷：按项目的 skill infera-round 的 `questionnaire_rN.md`，用上一轮 close 的建议预填（上一轮：@close/r<N−1>）。

## 授权

本任务可以写本任务工作区以外的这些文件（其余照约定只写本任务工作区）：
- 项目的 `project.yaml`：`round`、`goal`、`machines`、`ledger.working_point`，其余键不动；写完确认它还能解析；
- 项目的 `rules/project.md` 中有变化的行，和 `skills/cluster-<名>/`；
- 起草本轮任务的草稿 `tasks/<类型>/r<N>….md`（不要开始执行），并在要上机的任务 frontmatter 写 `machines`。

## 要做的事

- 预填问卷写成 `questions.md`，结束这一轮等我回答；拿到回答后按问卷"拿到回答后"一节起草本轮任务、更新项目文件。

## 交付

- `report.md`：以"## 结论"开头：一页 brief（本轮方向、预算、起草了哪些任务）和改了哪些项目文件。
