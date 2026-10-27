---
review: true
checks:
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - '"$AGENT_SYS_PYTHON" -m agent_sys_v2.perfcheck plan "$AGENT_SYS_TASK"'
---
## 任务描述

- 第 N 轮的优化计划。候选来源：本轮 profile 的 `candidates.yaml`、本轮 survey 的配置差异表、上一轮剩下的候选、问卷里我提的想法。

## 要做的事

- 读台账和已完成任务的报告，去掉已经否掉的方向；配置和拓扑类先做，代码类后做；按"预期收益 / 节点·小时"排序。
- 给出本轮的 GPU 预算和机器分配：实验的并行度等于空闲节点数，不等于 agent 数。
- 起草实验子任务（草稿，不要开始执行）：`tasks/experiment/r<N>-<名>.md`，一个子任务对应一个大杠杆，或一组同层、同验证方法的小杠杆；frontmatter 写 `parent`、`summary`、`machines`、`expect`（预期 ΔY、快速验证级别、证伪条件，格式见 skill infera-round）。
- 起草本轮的收尾任务（类型 close，名为 r<N>，不是子任务）：`depends_on: [{task: <本任务>, kind: children}]`，写明集成用的机器。
- 起草完就把本任务标为完成，不要等子任务：等待由 close 的 children 依赖负责。

## 交付

- `report.md`：以"## 结论"开头：候选排序表（来源、预期区间、成本、验证级别）、本轮预算和机器分配、起草的任务。
