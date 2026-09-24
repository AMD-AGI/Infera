# R2-on正式实验执行状态

2026-09-24 12:51 UTC 已开始启动 P/D 模型服务，尚未完成性能窗口。

用户已明确批准清理 n01-21 上的现存 K3 服务。四个已核实容器已停止，8张GPU恢复空闲、约0.28GiB/卡；清理记录见events。该节点已补齐与P相同的基线镜像。

本轮只运行R2-on/C80/P-D实际均4K，Prefill HiCache开启，R1/R3/R4关闭，逐候选日志关闭。历史完整A0作参考，不重跑基线。配置批准及release哈希已绑定。

P=n10-29/job31705，D=n01-21/job31706。两个allocation独立核验；两端无GPU设备的Docker清理监控均已验证持续心跳，避免依赖SSH子进程存活。

发压前检查实际server info、容量、镜像、随机种子、请求/缓存reset及基础smoke。完成发压后先保存结束身份，立即停止Prefill启动HiCache释放，再做离线分析；失败路径也清理本任务容器。

实时状态和日志：`/perf_apps/liyingli/bench_agentx/r2-4k-31705-31706-20260924`。LAUNCHING不代表smoke或性能已经通过。
