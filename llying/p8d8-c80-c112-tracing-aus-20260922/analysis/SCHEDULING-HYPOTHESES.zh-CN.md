# 调度与负载均衡：运行前假设

以下是当前源码/既有AUS配置揭示的候选问题，不是本轮实测结论。

1. Router独立选择P/D。当前kv-aware cost为 `-overlap_weight * prefix_hits + active_distinct_blocks + recent_blocks`；Prefill overlap weight=20，Decode=2。它不是基于实际GPU KV余量、scheduler queue或计算时间的调度器。
2. P/D的ActiveGuard共同在Decode响应结束时释放。因此Prefill已完成但Decode尚未完成的请求仍影响Prefill的active load，可能使估计负载滞后于实际计算负载。
3. active load按不同block的集合去重。现有AUS Decode `disable_radix_cache=true`、`disaggregation_decode_enable_radix_cache=false`；并发请求即使共享输入prefix，Decode仍可能各自分配KV。distinct blocks因此不能直接代表Decode实际KV需求。
4. recent load在每次pick时乘0.97，而非按墙钟衰减。两端pick共享此策略对象；业务到达率变化会改变这项记忆的实际时间尺度。
5. Decode preallocation按队列遍历，在request pool、metadata pool或KV预算不足时break。一个大请求是否阻碍后面可容纳的小请求，需要结合队列时间线验证，而不是直接修改排序。

验证方法：

- 以实际P/D rank和request id重建生命周期；比较router pick统计与真实P waiting/forward、D allocation/generation状态。
- 比较每rank累计与60秒窗口的请求数、miss tokens、host tokens、input/output tokens及stage latency，识别均值掩盖的短时偏斜。
- 对allocation阻塞同时检查其他rank KV余量、running/waiting和可分配slot。其他rank有空闲并不自动说明能接纳该请求，还要满足其KV需求及TP/DP协同约束。
- Prefill cache affinity的优化必须同时评估miss增加和队列下降；不能仅追求请求数平均。
- 若观测明确指出某一估计偏差，再做单变量策略对照。未做对照时只报告优化候选，不声称收益。

对应源码：`rust/router/src/policy.rs`、`rust/router/src/disagg.rs`；Decode分配分支已在诊断镜像加入按原因限频记录。历史server-info来自AUS aligned C80，正式运行后重新保存live配置。
