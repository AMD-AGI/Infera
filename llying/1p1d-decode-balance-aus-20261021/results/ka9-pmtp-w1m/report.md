**A. 客户端结果**（AgentX；interactivity = 1/ITL 分位数）

| 轮次 | tok/s/GPU | 输出 tok/s/GPU | 完成请求 | 平均输入 | P50 / P90 intv | ITL p50 / p90 (ms) | TTFT p50 / p90 / 均值 (s) | GPU / host 命中（理论） |
|---|---|---|---|---|---|---|---|---|
| ka9-pmtp-w1m C88 | 27,186 | 229.4 | 5,184 | 114.2k | 86.5 / 79.6 | 11.6 / 12.6 | 5.6 / 12.0 / 9.6 | 40.1% / 9.2%（96.4%） |

**B. 瓶颈链**（服务端为测量窗口均值；首 token 前在途 / decode 有效并发为 AIPerf 的 effective prefill / decode concurrency）

| 轮次 | prefill 缓存份额 | prefill 新算 tok/s | prefill 排队（实例均值） | 首 token 前在途 | decode 有效并发 | decode 每 rank 在跑 | decode KV 使用率 | KV>90% 时间 | 等预分配 / rank | 等 KV 传输 / rank |
|---|---|---|---|---|---|---|---|---|---|---|
| ka9-pmtp-w1m C88 | — | — | — | 27.1 | 30.4 | — | — | — | — | — |

**C. prefill 实例均衡**（新算 tok/s / 缓存份额 / 排队）

| 轮次 |
|---|
| ka9-pmtp-w1m C88 |

**D. decode rank 均衡**（8 个 DP rank 的最小 – 最大）

| 轮次 | 在跑请求 | KV 使用率 | KV>90% 时间 | 等预分配最多的 rank | 等 KV 传输 |
|---|---|---|---|---|---|
| ka9-pmtp-w1m C88 | — | — | — | — | — |

**E. 引擎节奏（判断栈是否漂移）与 router 会话**：decode 每步 ms 按同一 3 s 内最忙 rank 的在跑数分组；prefill 为排队时满 8192-token chunk 的耗时中位数 (ms)

| 轮次 | decode @6 | @8 | @10 | @12 | @14 |  | 会话命中 / 新绑定 |
|---|---|---|---|---|---|---|---|
| ka9-pmtp-w1m C88 | 40.8 | 43.5 | — | — | — |  | — |

