---
review: false
checks:
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - '"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger verify "$AGENT_SYS_WORKSPACE/results"'
---
## 任务描述

- 假设：（瓶颈是什么，改哪个变量或哪一组同层的变量，预期机理和幅度）
- 对照：本轮的基线曲线（project.yaml 的 `ledger.baseline_series`）
- 机器：在任务配置里选

## 要做的事

- 按 skill infera-experiment：dry-run 对比生效配置，证明杠杆生效；用 frontmatter `expect.level` 那一级快速验证迭代；候选栈跑 200 题 GSM8K（真实 MTP 接受率，`ledger gsm8k --series r<N>-<名>` 转换），判 accept 必须有它，且和本版本同题比不能显著变差。
- 最后只测工作点单点（L4，AgentX 正式，时长不低于 `ledger.min_duration_s`；同一节点上先测 base 再测变体，窗口不重叠，相邻两次间隔不超过 `ledger.max_arm_gap_min`），每个点用 `ledger agentx` 从 AgentX 输出转换（不手写结果，运行目录原地保留：验收要从它复算）：变体写 `series: r<N>-<名>`、`round: <N>`；base 是本轮基线系列（`ledger.baseline_series`）的一次复测，写 `role: baseline`、`series` 为基线系列、`label` 带 `-base`（skill infera-experiment），要复现封版任务在这个并发测的点（差 `ledger.reproduce_pct` 以内），否则判 rerun。fast 和更短的 AgentX 点进台账时标"筛选"，不作判定输入。整条曲线留给 close。
- 用 `ab_judge.py --write` 判定：只看 AgentX 正式测量的数字，同一并发的有效正式结果全部参与；预期收益低于 L4 噪声的会判 noise，加 `--note 待集成`；噪声底高于接受线时自动判"判不了 · 待集成"，不必重测。作废只有两类：失败请求 > 10%（`--void failed-requests`，由 AgentX 计数算）和其他（`--void other`，写原因和证据文件）。验收检查 `ledger verify` 从原始输出重新转换、重新判定，对不上就不通过。

## 交付

- `report.md`：以"## 结论"开头：接受 / 拒绝 / 噪声 / 待集成和 AgentX 数值，机理证据（解释，不是判定条件），与 `expect` 的预测对照，改动的 diff 或命令。
