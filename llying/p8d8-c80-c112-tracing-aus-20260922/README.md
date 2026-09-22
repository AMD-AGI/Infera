# AUS P8D8 C80→C112 诊断

2026-09-22 用户授权适配并执行，持续保存现场、生成报告、阶段性提交代码。
目标：定位 C80 的吞吐限制、C112 首个退化阶段，以及 P/D 路由、负载均衡和 admission 优化空间。

沿用 AUS aligned 配方、跨 rank 路由、HiCache 与 MTP simulation 3.61。
Prefill n01-33，Decode n02-21。只管理本实验服务，不干扰 Decode 的 xiaoming-dev。

运行根目录：`/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922`。
基础 harness：`/perf_apps/liyingli/bench_agentx/glm52-c80-20260922/scripts/bench-harness`。

当前状态见 `analysis/PROGRESS.md`；完整原始数据留运行目录，报告/汇总与校验清单回收本目录。
诊断补丁不增加 GPU synchronization；host_load_ack 是已有同步完成后的CPU观测时间，不等于纯DMA耗时。
