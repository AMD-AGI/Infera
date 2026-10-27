---
review: false
checks:
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - 'test ! -d "$AGENT_SYS_WORKSPACE/results" || "$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger check "$AGENT_SYS_WORKSPACE/results"'
  - '"$AGENT_SYS_PYTHON" -m agent_sys_v2.perfcheck close "$AGENT_SYS_TASK"'
---
## 任务描述

- 第 N 轮收尾：集成、封版、归档、复盘，起草下一轮的 intake。本轮的实验见上游 plan 的子任务。

## 授权

本任务可以写本任务工作区以外的这些文件：
- 项目的 `project.yaml`：封版时写 `version: v<N>` 和 `ledger.baseline_series: v<N>`，其余键不动；写完确认它还能解析；
- 起草下一轮的 intake 任务（类型 intake，名为 r<N+1>；草稿，不要开始执行）。

## 要做的事

- 集成：把本轮判为接受（以及"待集成"）的改动叠成一个镜像，过正确性门：全量 1319 题 GSM8K（`ledger gsm8k --series v<N>`），和上一版比差值 95% 区间下界不低于 `-quality.gsm8k_margin`（默认 0.02，`perfcheck close` 复算）；测整条曲线（L5），结果写 `series: v<N>`、`round: <N>`、`role: baseline`；工作点在同一节点测两次，下一轮的噪声底就用它（不测则沿用上一版的复测噪声）。
- 判定封版：在工作点和 X 的 2–3 个水平上 Y 都不劣于 v<N−1>，且工作点 ΔY 超过噪声（台账的工作点 Δ）。通过则封版 v<N>；不通过或没有接受任何改动，版本不变，只做复盘。
- `stack.json`：版本清单（格式见 skill infera-round），`sealed` 写是否封版。
- 复盘：快速验证的预测与集成实测对照（偏差大的写进 lessons）；下一轮建议（是否重做对标和 profile、候选排序、是否交付）。
- 起草下一轮的 intake（类型 intake，名为 r<N+1>）：frontmatter 写 `depends_on: [<本任务>]`、`review: true`、`agent: {model: sonnet, budget_usd: 20}`（验收检查在纳入托管时按类型补上），正文从 intake 类型的骨架（`task_types/intake.md`）开始，写明用 `questionnaire_rN.md` 和本报告的建议预填。

## 交付

- `report.md`：以"## 结论"开头：是否封版、版本说明、曲线和相对上一版与对标的 Δ、预测与实测对照、下一轮建议。
