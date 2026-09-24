# R1+R4夜间工作进度

更新：2026-09-24 22:10 UTC。作业31719已获得n05-21/n05-29，第四次正式尝试已于22:10:01开始884条预热；qos=batch，C80/4K、R1+R4、P HiCache保持不变。此前3次尝试均被抢占于warmup。当前持续监控，预热后进入3600秒正式窗口。

| 工作 | 结果 |
|---|---|
| R1+R4离线复核 | 完成；276项默认并行测试通过，增加组合生命周期测试并隔离日志过滤单测的并发干扰；生产算法未改 |
| 无HiCache短时smoke | 完成，约10分35秒；80/80 C80功能请求、24/24 AIPerf请求通过；146个P决策符合R4最小评分，46次与原策略不同；R1完成释放实际触发 |
| 3条预热空内容错误排查 | 在同镜像实机复现单独<think>生成后HTTP200/usage=1但正文为空；同版本AIPerf解析为None。历史3条具体token未保存，不能逐条直接归因 |
| 正式性能测试 | 未完成；3次尝试均在warmup期间被Slurm抢占，没有有效profiling窗口，不给出R1+R4吞吐结论 |
| 后续 | 等待更合适的运行窗口；不重复完整smoke，不重跑A0/G0基线，只恢复同配置正式实验 |

正式配置：C80，P/D有效chunk均4K，P HiCache开，R1=completion，R4=on，R2/R3=off，884条预热、3600秒窗口；镜像与二进制沿用已验证版本。

- [离线复核与配置](REVIEW.zh-CN.md)
- [smoke与原始响应复现](smoke-results/REPORT.zh-CN.md)
- [执行历史与抢占记录](HISTORY.zh-CN.md)
- [机器可读当前状态](CURRENT.json)

完整运行目录：`/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924`。`events/queue-monitor.jsonl`、`events/scheduling-plans.jsonl`保存周期性监测；大文件、原始日志和镜像归档留在共享目录。
