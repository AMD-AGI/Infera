# R1+R4夜间工作已完成

更新：2026-09-25 00:08 UTC。完整60分钟性能窗口已完成并通过验收，模型服务和显存释放均已核实，自行申请的31719已释放。

**结论：本case不建议启用当前R4。** 相比历史G0（R1-only），R1+R4输出吞吐下降31.32%，平均TTFT从7.58秒升到26.93秒；未命中比例从4.55%升到10.92%，P排队明显增加。建议保留R1，先校准R4的成本模型。

完成内容：276项离线测试通过；无HiCache smoke完成；空内容响应机制实机复现；R1+R4 C80/4K完整实验及A0/G0、匹配请求、缓存长尾分析完成。正式请求7167条成功、错误0，结束边界取消49；884条预热中1条空内容解析错误已保留。没有追加基线或其他策略实验。

- [最终报告与结论](results/REPORT.zh-CN.md)
- [配置与代码复核](REVIEW.zh-CN.md)
- [smoke和空内容复现](smoke-results/REPORT.zh-CN.md)
- [完整执行历史](HISTORY.zh-CN.md)
- [最终资源状态](results/resource-release.json)

原始运行目录：`/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924/runs/r1r4-31719-performance-attempt4`。
