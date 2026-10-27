---
review: true
checks:
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - test -s "$AGENT_SYS_WORKSPACE/candidates.yaml"
  - test -s "$AGENT_SYS_WORKSPACE/profile.json"
---
## 任务描述

- 对象：本轮的基线（例如 @baseline/r<N> 的 v0 曲线，或 @close/r<N−1> 的 vN−1 曲线）
- 模式：standard（默认，不用 ultracode）；onboarding、deep 要先经我批准，并在任务配置里打开 `agent: {ultracode: true}`

## 要做的事

- 按 skill infera-profile 的 P0–P6：先读引擎的 playbook（`playbooks/<引擎>.md`）；从便宜的层做起（时间账、资源利用率），只对瓶颈层做 kernel 级或 CPU 级分析；关键数字由核实子 agent 独立重算。
- 读台账和已完成任务的报告，不重复已经否掉的方向。
- 不起草实验任务：候选交给 plan。

## 交付

- `report.md`：以"## 结论"开头：瓶颈、时延税、容量风险、前 3 个候选及预期区间、是否建议 deep 及理由。
- `candidates.yaml`（infera.candidates/v1）、`profile.json`、`verify.md`。
