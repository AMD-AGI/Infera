---
review: false
checks:
  - test -s "$AGENT_SYS_WORKSPACE/report.md"
  - '"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger check "$AGENT_SYS_WORKSPACE/results"'
  - '"$AGENT_SYS_PYTHON" -m agent_sys_v2.perfcheck baseline "$AGENT_SYS_TASK"'
  - test -s "$AGENT_SYS_WORKSPACE/stack.json"
---
## 任务描述

- 部署：按项目规则（`rules/project.md` 的部署形态和代码），或写明这次不同的地方
- 负载：AgentX，并发点按项目规则；机器在任务配置里选

## 要做的事

- 环境预检（节点、镜像、权重、RDMA / CPU KV 层探针）→ 质量门（skill infera-quality-gate：全量 1319 题 GSM8K，用 `ledger gsm8k --series v0` 转换；smoke、真实 MTP 接受率排查用）→ 工作点先跑 fast 冒烟再跑正式 → 工作点在同一节点至少测 `ledger.min_repeats`（默认 2）次正式 AgentX，由此算出的复测噪声要低于 `ledger.accept_pct`（`perfcheck baseline` 检查）；跨节点再测一次 → 其余并发点从高往低扫（约 ×0.7），KV 悬崖附近加密。
- 复测噪声不低于接受线时，这套测量判不了 A/B：不要带着它进入实验，把状态改为阻塞，交给我决定（每臂多测几次，或经我同意提高接受线）。
- 每个点用 `"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger agentx <run 目录> --out "$AGENT_SYS_WORKSPACE/results" --series v0 --round <N> --role baseline` 写进台账（skill infera-bench）；复测的点加 `--label v0-rep<k>`，同一系列同一并发的结果重名时转换会报错，不会覆盖。
- 同时产出：`stack.json`（v0 的栈清单，`sealed: true`）；常开的系统采样（GPU busy、功耗、时钟、NIC 字节、pidstat），profile 会用；可选 `calibration.json`（fast 对正式的偏移、dstep 每步 1 ms 对应的 ΔY，只给实验迭代筛选用，格式见 skill infera-round）。
- 对齐门槛：同配置对 IX 参考 ±10% 以内或能解释差异；本地复测 ≤ 2%；正式窗口内无 JIT、0 次 KV 传输失败、失败请求 < 10%。

## 交付

- `report.md`：以"## 结论"开头：基线曲线、工作点数值和噪声、与对标的差距、生效配置和复现命令。
