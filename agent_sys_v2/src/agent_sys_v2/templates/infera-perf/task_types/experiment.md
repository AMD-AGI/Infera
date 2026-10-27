---
review: false
checks:
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - '"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger check "$AGENT_SYS_WORKSPACE/results" --require-verdict'
---
## 任务描述

- 假设：（瓶颈是什么，改哪个变量或哪一组同层的变量，预期机理和幅度）
- 对照：本轮的基线曲线（project.yaml 的 `ledger.baseline_series`）
- 机器：在任务配置里选

## 要做的事

- 按 skill infera-experiment：dry-run 对比生效配置，证明杠杆生效；用 frontmatter `expect.level` 那一级快速验证迭代；新镜像过正确性门（GSM8K、真实 MTP 接受率）。
- 最后只测工作点单点（L4，AgentX 正式，同节点先测 base 再测变体）：变体写 `series: r<N>-<名>`、`round: <N>`；base 是本轮基线系列（`ledger.baseline_series`）的一次复测，写 `role: baseline`、`series` 为基线系列、`label` 带 `-base`（skill infera-experiment）。整条曲线留给 close。
- 用 `ab_judge.py` 判定，写进结果的 verdict；预期收益低于 L4 噪声的会判 noise，加 `--note 待集成`；作废的也要写。

## 交付

- `report.md`：以"## 结论"开头：接受 / 拒绝 / 噪声 / 待集成，数值和机理证据，与 `expect` 的预测对照，改动的 diff 或命令。
