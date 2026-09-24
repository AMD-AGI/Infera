# R1+R4自主实验进度

用户授权：自主申请两台batch节点，排除n04-29；排查3条预热空内容错误、离线复核R1+R4；机器可用后短时无HiCache smoke，再做C80/4K正式性能实验。用户休息约10小时后review，不需重复确认已授权配置。

作业31719，partition=Compute-DCPT，qos=batch，2节点独占，24小时，排除smci355-ccs-aus-n04-29。目前PENDING(Resources)，排队预估开始21:40:30 UTC，仅为调度预测。持续监测，未启动模型。

离线复核276项测试串行全部通过，包含新增R1+R4同时释放P记账而保留D记账的组合测试。首次并行运行274通过、1项日志过滤测试失败（tracing subscriber/callsite并行环境）；完整串行复跑包含新增测试，276通过。没有修改生产调度算法，正式沿用已验证release二进制0cf48cbf96df48caf8c03949fc5edce1c97d26bebb293c71d5409dbe77cafe6d。

A0/G0关键引擎配置（chunk、TP/DP、种子、HiCache、mem fraction、MTP等）一致，见review/a0-g0-settings.json。正式用历史A0与G0，不补跑基线。

运行目录：/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924。
