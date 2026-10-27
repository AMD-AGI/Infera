---
review: false
checks:
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - '"$AGENT_SYS_PYTHON" -m agent_sys_v2.perfcheck survey "$AGENT_SYS_TASK"'
  - 'test ! -d "$AGENT_SYS_WORKSPACE/results" || "$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger check "$AGENT_SYS_WORKSPACE/results"'
---
## 任务描述

- 对标调研：InferenceX 上与本项目同模型、同精度、同硬件（再加最好的 NVIDIA 条目）的曲线和配方；未合入的 IX PR 也看。不占 GPU。
- 上一轮的对标：（例如 @survey/r<N−1>，没有就删掉这一行）

## 要做的事

- 参考曲线写进台账：每个点一个 `results/<标签>.json`，`role: reference`，`series: ix:<条目>`，`round: <N>`，指标名用 InferenceX API 的（`input_tput_per_gpu`、`p90_intvty`、`p90_e2e_norm_intvty`、`p90_ttft`），`evidence` 写数据来源（API 地址或 run_url）和抓取时间。
- 没有对应条目时按顺序回退：同模型同硬件换引擎 → 换精度或相近硬件（只作量级参考）→ 同架构的姊妹模型 → 上游官方 recipe（标"非官方"）；都没有就在报告里写"无公开对标"。
- 配置差异表：对方配置与我们当前版本的每一项差异（我们的值、对方的值、预期影响），每一项都是 plan 的一个"采纳 X"候选；合规清单（如果要提交 IX）。
- 只读：不跑 GPU，不提交作业。

## 交付

- `report.md`：以"## 结论"开头：对标曲线与我们当前版本在工作点的差距、配置差异表、合规清单。
