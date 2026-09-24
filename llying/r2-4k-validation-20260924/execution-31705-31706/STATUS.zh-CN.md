# R2-on正式实验执行状态

2026-09-24 13:06 UTC：n01-21 的四个 K3 服务已按用户指示停止，allocation 保留。清理后8张GPU空闲、约0.28GiB/卡。

第一次启动的实际配置校验、8条真实P/D配对smoke、全部rank缓存reset均通过。客户端环境检查因n01-21主机rocm-smi只报告“AMD Radeon Graphics”而失败，尚未进入warmup/正式性能窗口。13:03:52首先请求停止P，13:04:35所有本轮服务停止；HiCache/VRAM释放仍需单独观察。

已修复GPU型号检测：主机工具无型号时使用同一实验镜像内rocm-smi，双节点验证结果均为MI355X；没有硬编码型号或修改实验配置。失败目录保留为runs/r2-on-4k-31705-31706-failed-hardware-preflight，容器和AITER缓存也已保留。

13:06 UTC已重新启动驱动（P主机PID4014518），先等待旧P释放，再启动原定R2-on/C80/P-D实际4K实验。Prefill HiCache开；R1/R3/R4关；候选诊断日志关。只和历史完整A0比较，不补跑基线。

P=n10-29/job31705，D=n01-21/job31706。双端allocation watchdog已重启并验证；结束或失败先停P，再离线分析。实时目录：/perf_apps/liyingli/bench_agentx/r2-4k-31705-31706-20260924。
