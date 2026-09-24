# R2-on正式实验执行状态

2026-09-24 13:06 UTC：n01-21 的四个 K3 服务已按用户指示停止，allocation 保留。清理后8张GPU空闲、约0.28GiB/卡。

第一次启动的实际配置校验、8条真实P/D配对smoke、全部rank缓存reset均通过。客户端环境检查因n01-21主机rocm-smi只报告“AMD Radeon Graphics”而失败，尚未进入warmup/正式性能窗口。13:03:52首先请求停止P，13:04:35所有本轮服务停止；HiCache/VRAM释放仍需单独观察。

已修复GPU型号检测：主机工具无型号时使用同一实验镜像内rocm-smi，双节点验证结果均为MI355X；没有硬编码型号或修改实验配置。失败目录保留为runs/r2-on-4k-31705-31706-failed-hardware-preflight，容器和AITER缓存也已保留。

13:06 UTC已重新启动驱动（P主机PID4014518），先等待旧P释放，再启动原定R2-on/C80/P-D实际4K实验。Prefill HiCache开；R1/R3/R4关；候选诊断日志关。只和历史完整A0比较，不补跑基线。

P=n10-29/job31705，D=n01-21/job31706。双端allocation watchdog已重启并验证；结束或失败先停P，再离线分析。实时目录：/perf_apps/liyingli/bench_agentx/r2-4k-31705-31706-20260924。

13:30 UTC左右：用户指出辅助检查不应中断实验。调整脚本：硬件元信息检测失败记unknown/告警；采样preflight失败告警；ownership遥测查询失败不再终止client（实际foreign GPU占用仍保留处理）；测量脚本异常保留服务，标记NEEDS_REVIEW供人工检查。正常完成仍先停P，allocation丢失仍由独立watchdog处理。已运行的启动shell仍可能持有旧ERR trap，新启动脚本已修正；后续exec的测量脚本使用新逻辑。配置和router算法不变。语法检查通过。

13:45 UTC：13:31:58两端显存全部降至0.096%；13:33重启模型，13:37服务就绪。正在执行的shell曾因原位修改文件报EOF，重启启动脚本解决（没有重启机器或GPU驱动）。smoke检查混入前次同名rid，已按本次capture-start过滤诊断记录，8条配对全部通过，服务未因此停止。客户端脚本的DIR变量被历史配置覆盖，导致调用旧版工具；改用独立R2_HARNESS_DIR，13:41开始正式客户端启动，目前依赖安装中，尚未进入性能计时。正常使用原定R2/C80/4K，未补跑其他短时实验。上次遗留的两个采样进程已停止，失败处理补上仅采样进程清理。
