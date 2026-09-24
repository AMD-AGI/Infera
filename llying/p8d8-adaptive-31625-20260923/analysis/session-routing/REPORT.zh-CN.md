# Session / turn 的 Prefill rank 分布与 KV 重复测量边界

已用完成的 A0、G0、C1 profiling 数据生成图表。C1G1 仍在运行，将在完成并审计后加入，避免混用部分窗口。

- [交互查看全部 session/conversation](session-routing.html)：选择实验与分组，查看每个请求的turn、P rank、输入、device/host命中、miss及TTFT。
- [路由分散总览](routing-overview.png)（[PDF](routing-overview.pdf)）。
- [相同conversation的turn→rank热图](conversation-turn-ranks.png)（[PDF](conversation-turn-ranks.pdf)）：按三轮总请求数选择最多的40个共同conversation，原始turn_index作横轴。完整列表在heatmap-selection.json。
- 全量逐请求CSV：A0-request-routing.csv、G0-request-routing.csv、C1-request-routing.csv。聚合统计与源文件SHA256在routing-summary.json。

## 当前结果

| 指标 | A0 | G0（guard优化） | C1（容量扩展） |
|---|---:|---:|---:|
| 完成并关联的正式请求 | 9,727 | 10,033 | 9,774 |
| root_correlation_id分组数 | 155 | 155 | 155 |
| conversation数 | 1,842 | 1,848 | 1,819 |
| 至少2条请求的conversation数 | 479 | 482 | 476 |
| 上述conversation使用多个P rank的比例 | 60.33% | 58.09% | 59.45% |
| 上述conversation平均使用rank数 | 2.086 | 1.807 | 2.013 |
| 连续turn换P rank比例 | 14.71% | 10.49% | 14.10% |
| 请求加权的conversation主rank占比 | 85.92% | 88.79% | 86.81% |

root session按root_correlation_id分组，包括并行分支；不是80个并发lane，也不能把分支分散当成同一conversation迁移。连续turn换rank只统计同一conversation中原始turn_index=t和t+1均存在且各有唯一完成记录的相邻对，跳号/重复turn不算。

各轮闭环推进不同，所以上述全量集合不同。进一步要求相邻对的两个端点均有相同source_trace、相同实际输出长度，且输入差≤8 tokens及≤0.1%：

| 对照 | 可比相邻对数 | 左侧换rank比例 | 右侧換rank比例 |
|---|---:|---:|---:|
| A0 → G0 | 7,116 | 14.28% | 10.27% |
| A0 → C1 | 6,850 | 14.01% | 13.75% |
| G0 → C1 | 7,195 | 10.23% | 13.65% |

相同集合也支持G0路由更集中；C1相对A0变化较小。仍是已完成请求交集，有选择效应，不能据此单独估计吞吐因果效应，也未排除热身和时间偏差。

## 为什么这还不是重复 KV 比率

换rank后，原rank可能已驱逐前缀；同rank内不同分支可能只共享较短前缀。不同session也可能共享系统提示词。因此session分散既不是同时驻留重复块的直接计量，也不是可回收显存比例。

当前aus_diag只记录request级缓存token计数，没有完整block hash/创建/驱逐历史。仅有最终选中rank的路由日志，也无法重建当时其他7个候选rank的缓存视图。

若取得同一时刻各P rank的**GPU缓存块集合** B_r(t)，且块使用相同模型/兼容键、链式前缀hash和page size，可计算：

```
重复副本比例 D(t) = [sum_r |B_r(t)| - |union_r B_r(t)|] / sum_r |B_r(t)|
两rank的Jaccard = |B_i(t) ∩ B_j(t)| / |B_i(t) ∪ B_j(t)|
额外副本tokens = [sum_r |B_r(t)| - |union_r B_r(t)|] × page_size
```

空集合分母为0时结果应为未定义而非0。变长block须按有效tokens/bytes加权。host要单独测，不与GPU混合；只统计8个DP rank，不把同rank的TP分片误当冗余副本。

这些是内容去重的理论空间，不等于可无代价回收的容量：并发可能需要多个副本，删除副本会引入路由热点或传输成本。需要完整、按rank和tier标注的Stored/Removed/Cleared流，从已知清空状态开始，或可靠的engine驻留快照；必须记录event序号、丢失、快照时间偏差、freshness，并用cache gauges校验。事件镜像快照只能代表其索引视图，不能未经验证称为物理显存真实驻留。当前没有追溯补造该比率。

## weight和当前调度

当前文本请求、无额外retention放大时，router对每个P DP rank计算：

```
cost(rank) = active_distinct_blocks(rank) + recent_blocks(rank)
             - W_prefill × cached_prefix_blocks(rank)
```

选择cost最小的rank，cost相同优先load更小。prefix hits按连续最长前缀计算，遇第一个缺失块即停止。recent每次pick乘0.97，再向被选rank计入本次miss块数（未知块信息时记1）。active是正在途请求引用的不同block hash数，不是物理resident KV容量。

| “不设weight”的具体操作 | 实际行为 |
|---|---|
| 沿用本benchmark脚本，仅不写P/D weight环境变量 | 脚本仍默认P=20、D=2，并显式传给router |
| 直接启动router，不传P专用weight | P继承全局kv-overlap-weight |
| 直接启动router，所有weight参数均省略 | 全局默认1.0；P/D均继承1.0 |
| 显式P weight=0，仍用kv-aware | 去掉直接prefix-hit评分项，按active+recent选rank；recent仍按miss记账，不能说所有缓存信息都被禁用 |
| router-policy=round-robin | 换为另一种轮转策略；不是weight=0 |

源码证据：`rust/router/src/policy.rs`（KvEventAwarePolicy::pick/load_of/record_dispatch）、`rust/router/src/config.rs`（weight默认）、`infera/server/args.py`、`bench/glm5p2_pd/config.sh`。当前运行日志实际w_overlap=20，未改变。

## 是否调高 P prefix-hit 权重

调高可能减少迁移/重算，却可能让高命中rank积压。分散图可以找候选问题，不能单独证明提高权重有净收益。更适合先记录每个pick的8个候选rank：prefix hits、active、recent、score、队列、可用/可驱逐容量，以及request/session身份；当前winner-only日志缺少这些反事实信息。

对于高命中rank H和低负载rank L，如果H多命中Δhits但多负载Δload，H胜出条件为 `W > Δload/Δhits`（Δhits>0，忽略其他项）。这个阈值是路由记账单位，不是已校准的秒级收益。当前W=20已是较强缓存偏好，且G0在W不变时已降低换rank比例，优先修正guard负载记账并观察候选分数，比直接上调更有证据。

未来若候选日志显示频繁为小幅load优势放弃大段prefix，并伴随重复驻留/重算，且高命中rank并未严重排队，再做固定服务状态下的单变量提高权重筛查。最终仍以miss、排队、TTFT、output throughput及尾延迟共同判断，并保留热身混杂限制。
