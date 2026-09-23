# A0 完成：新分配的 4K C80 对照

2026-09-23 17:33 UTC 完成审计。节点 P=n10-29、D=n02-21，job 31644。完整 profiling 3600 秒，统一 warmup，4K，C80；路由候选二进制处于 decode 控制模式。

- 成功 9,727 / 发出 9,740；结束边界取消 13；profiling 导出错误 0；9,727 条全部 P/D 配对。
- Total tokens/s/GPU **20,166.30**；Output tokens/s/GPU **161.66**（集群 output 2,586.52 tokens/s）。
- TTFT mean 9.85979 s、p50 5.50831 s、p90 22.24993 s；ITL mean 0.01414 s。
- 相比旧分配的新启动 4K：total -2.12%、output -0.85%；平均输入长度 -1.58%。节点不同，不归因于单一参数。G0 使用本次 A0 为直接控制。
- 所有配置/缓存初始化/进程身份/模型元数据/所有权检查通过。profiling P scrape 缺口 12/1791、D 14/1791；不补零。
- 逐请求缓存记录：输入 1,161,686,094、device 命中 1,097,429,312、host 命中 6,266,432、miss 57,990,350。AIPerf 聚合 GPU hit rate 0.82803 与此口径不一致，不用于宣称缓存收益；优先逐请求 cohort。
- 32 条有直接 D KV allocation 阻塞，allocation mean 25.89 ms。仍不足以优先投入 D 扩容。

P 完成观测与 9,727 条 profiling 请求全部关联；client end − P HTTP body drain 的 mean 12.561 s、p50 5.621 s、正值占 93.01%。这只是控制组 P 记账可能多持有时间的代理量，不是 GPU 计算时间、distinct-block 占用或收益估算。warmup 多为负值，保留原值；P HTTP 尾部结束可能晚于 D 客户端结束。

G0 已通过重置与模式校验，只改变 P guard 释放时机。待完整结果后决定回滚 A1 或裁剪此分支。不能把机制变化直接当成性能收益。

数据摘要与完整审计见 `a0-final/`；较大原始证据保留于共享目录，路径、大小和 SHA256 见该目录 `large-evidence-manifest.json`。采集器已增加互斥锁、原子发布、固定容器 ID 和完成封存检查；本轮最终完整配对和审计通过。
