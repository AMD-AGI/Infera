# 执行状态

本轮已因Slurm抢占中断，正式窗口约43分45秒，未完成60分钟。8条smoke通过；884条预热结束，原始导出含3条空内容解析错误。前43分钟6987条成功、错误0。详见[部分结果报告](results-partial/REPORT.zh-CN.md)。Prefill停止调用报错，后续SSH拒绝，最终资源释放未核实。

进一步对照发现R2改善D瞬时KV均衡和分配长尾，但未改善主要P侧等待；详见[bottleneck-analysis](bottleneck-analysis/REPORT.zh-CN.md)。
