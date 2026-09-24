# 问题记录

每个问题记录现象、日志片段、原因、处理、状态。8P4D 阶段的问题（DPA eager、rail 隔离、DCP KV staging、会话亲和、LMCache、
新驱动显存、抢占等）见 [../../8p4d-atom-infera/.record/issues.md](../../8p4d-atom-infera/.record/issues.md)。

## 1. C80 正式测试被抢占中断

- 现象：06:19 SSH 到 n04-21、n10-29 被 pam_slurm_adopt 拒绝。`sacct -D -j 31690`：02:18:56-06:00:48 运行，状态 `PREEMPTED`，之后重新排队（`Reason=Resources`）。sweep 日志停在 05:55:40（测量第 24:45），AgentX 容器继续运行到 06:02，其间 06:01:54 起无法连通路由。
- 影响：C80 测量只完成约 30 分钟，没有 InferenceX 聚合 JSON。部分指标见 `results/README.md`。
- 遗留：n04-21 上的 `glm52-8p8d-atom-{decode,router,etcd,agentx}` 与 n10-29 上的 `glm52-8p8d-atom-prefill`（持有约 1.25 TiB pinned 主机内存）可能仍在运行，节点已无法访问。
- 状态：等待作业重新运行后重测完整的 3600 s。
