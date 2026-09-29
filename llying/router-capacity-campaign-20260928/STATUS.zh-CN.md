# 当前状态：资源阻塞，计划尚未全部完成

更新：2026-09-29。Slurm会计记录确认，重试请求32076与32077在00:42:28被root取消，均未分配GPU；没有给出取消原因。目前本用户没有运行或排队的GPU作业。等待进程与allocation watcher已停止，没有绕过资源管理操作继续重提。

## 已完成并保存

- **B1 新P首次评分**：输出吞吐−2.45%，平均TTFT+9.82%，不纳入后续默认组合，代码保留为默认off。
- **B2 D radix**：输出吞吐−1.04%，平均TTFT−6.45%，D本地前缀复用约23.62%；完整结果及warmup/边界限制见b2/RESULT.zh-CN.md。
- **B3 D radix＋D亲和**：客户端完整发压，10,308完成/9取消/0错误；导出与末尾诊断被抢占中断，恢复指标单独标注。其token吞吐+3.06%伴随平均实际输出+2.93%，不能当作请求处理能力同比提升。D原生匹配子集复用约91.8%，实测RDMA接收平均速率约下降68.5%。
- **RB 新节点完整基线**：10,186完成/8取消/0错误，2661.35 output token/s，TTFT均值6.536s、p95 19.107s。请求配对与D前缀诊断覆盖100%，引擎身份稳定。
- **B4 Triton完整测试**：10,728完成/11取消/0错误；同RB节点，输出吞吐+3.38%、完成请求数+5.32%，TTFT均值−28.79%、p95−37.29%。匹配请求P排队−47.75%、P前向观测窗口−37.07%。后续保留Triton。Fusion始终开启、IndexShare关闭。

统一数据表和图：[overview/COMPARISON.zh-CN.md](overview/COMPARISON.zh-CN.md)。B4解读：[b4/RESULT.zh-CN.md](b4/RESULT.zh-CN.md)。B3恢复数据和限制：[b3-interrupted/RESULT.zh-CN.md](b3-interrupted/RESULT.zh-CN.md)。

## 尚未完成

**B6 2P8＋1D8、B7 4P4＋1D8、B5 P8＋D4的C80性能数据尚未取得。** B6首次启动时，内部DP事件端口25539–25546与对外KV端口25541冲突；还未进入真实答案或性能测量。随后基础32054与第三台32056在23:57:35被同时抢占。

端口范围和完整注册就绪检查已经修复，测试可复现旧冲突并验证新规则。重试使用独立目录，但其排队请求又被root取消，尚未实机确认修复后的完整启动。原失败日志、节点分配和取消记录全部保留，见[b6-startup-failure/RESULT.zh-CN.md](b6-startup-failure/RESULT.zh-CN.md)与[RESOURCE_BLOCK.json](RESOURCE_BLOCK.json)。

## 代码复核与恢复

已复核并修正：首次评分预约/释放逻辑、跨worker诊断身份、真实rank覆盖、HIP/SMI PCI映射、GPU吞吐分母、SSH准入等待、固定服务端口避让、完整注册就绪、分配登记并发和资源收尾。既有Router的292单元/26 HTTP/4 ZMQ/14 render检查通过；新增布局、GPU映射、SSH及端口复现检查通过。具体范围和限制见[CODE-REVIEW.zh-CN.md](CODE-REVIEW.zh-CN.md)。

继续需要新的可用GPU分配。优先沿用P=n04-33、D=n05-21、额外P=n05-29，以保留同节点比较；若改节点，需要重新确定比较基线。取消的32076/32077不能直接复用。更新分配ID与准备记录后，才能移除runtime的STOP_AUTORUN并启动重试。执行入口见[RUNBOOK.zh-CN.md](RUNBOOK.zh-CN.md)。

Runtime：`/perf_apps/liyingli/bench_agentx/router-capacity-20260928`。原始请求、trace、采样及失败记录保留在各runs目录；没有GPU重置或停止无关容器。没有创建“全部实验完成”标志。
