**A. 客户端结果**（AgentX；interactivity = 1/ITL 分位数）

| 轮次 | tok/s/GPU | 输出 tok/s/GPU | 完成请求 | 平均输入 | P50 / P90 intv | ITL p50 / p90 (ms) | TTFT p50 / p90 / 均值 (s) | GPU / host 命中（理论） |
|---|---|---|---|---|---|---|---|---|
| ka9-pmtp-w0 C88 | 26,462 | 227.9 | 5,160 | 111.5k | 75.8 / 55.7 | 13.2 / 17.9 | 5.5 / 11.1 / 8.3 | 50.8% / 12.7%（96.2%） |

**B. 瓶颈链**（服务端为测量窗口均值；首 token 前在途 / decode 有效并发为 AIPerf 的 effective prefill / decode concurrency）

| 轮次 | prefill 缓存份额 | prefill 新算 tok/s | prefill 排队（实例均值） | 首 token 前在途 | decode 有效并发 | decode 每 rank 在跑 | decode KV 使用率 | KV>90% 时间 | 等预分配 / rank | 等 KV 传输 / rank |
|---|---|---|---|---|---|---|---|---|---|---|
| ka9-pmtp-w0 C88 | — | — | — | 23.3 | 35.9 | — | — | — | — | — |

**C. prefill 实例均衡**（新算 tok/s / 缓存份额 / 排队）

| 轮次 |
|---|
| ka9-pmtp-w0 C88 |

**D. decode rank 均衡**（8 个 DP rank 的最小 – 最大）

| 轮次 | 在跑请求 | KV 使用率 | KV>90% 时间 | 等预分配最多的 rank | 等 KV 传输 |
|---|---|---|---|---|---|
| ka9-pmtp-w0 C88 | — | — | — | — | — |

**E. 引擎节奏（判断栈是否漂移）与 router 会话**：decode 每步 ms 按同一 3 s 内最忙 rank 的在跑数分组；prefill 为排队时满 8192-token chunk 的耗时中位数 (ms)

| 轮次 | decode @6 | @8 | @10 | @12 | @14 |  | 会话命中 / 新绑定 |
|---|---|---|---|---|---|---|---|
| ka9-pmtp-w0 C88 | 40.5 | 43.6 | 52.4 | 65.2 | 68.6 |  | — |

