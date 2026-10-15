# 3P(TP4)1D(TP8) on sglang v0.5.21：结果

> 基线扫描已完成（C112–C176），复盘见 `REVIEW.zh-CN.md`；调优结果见 §3（阶段 1 已完成，持续更新）。数据来源：`results/<round>/c<CONC>/agentx_conc*.json`（汇总 `results/summary.csv`、`results/summary.md`），
> 过程记录 `operations/STATUS.zh-CN.md`。每卡吞吐按 20 张 GPU（3×TP4 prefill + 1×TP8 decode）计。

## 0. 结论摘要（2026-10-14）

**目前最高：C176 = 30,562 tok/s/GPU**（t4d-tok4：R1 + P 会话亲和 + R2；prefill mem 0.88 + HiCache 400/200 GB/rank；decode
`--tokenizer-worker-num 4`），P50 / P90 interactivity 48.9 / 44.7，TTFT p50 3.1 s。相对 r2 基线最高点（C112 20,545）+49%。
同配置 C144 25,269、C160 28,479（P50 49.4 / P90 44.7）。三个点的 P50 interactivity 都在 47–49，略低于 50。

| 步骤 | 关键数字 | 说明 |
|---|---|---|
| decode 预热 | r2 C144 18,001 → t1-base 19,503 | r2 C144 是 decode 重启后首轮，每步慢 13–17%，属测量问题 |
| R1（prefill 完成即释放记账） | 19,308（噪声） | 保留；TTFT −6~8% |
| P 会话亲和 | **21,639（+12.1%）** | 缓存份额 88% → 94%，prefill 新算 −42% |
| R2（decode 按 token 需求选 rank） | **22,340（+3.2%）** | decode KV 拉平，KV>90% 6% → 1% |
| 2a（prefill mem 0.88 + HiCache 扩容） | C144 22,238（持平）、C160 21,161 → **22,279（+5.3%）** | 平台拓宽到 C144–C160，峰值未抬高 |
| **decode tokenizer worker ×4** | C144 22,238 → **25,269（+13.6%）**，C160 **28,479**，C176 **30,562** | 此前吞吐平台的真正原因：decode 单进程 TokenizerManager 饱和（CPU 100%，decode TTFT 30 s 而 prefill 3 s 已完成） |

瓶颈演变：prefill 缓存局部性（会话漂移）→ decode TokenizerManager（单 Python 线程承担全部流式输出）→ decode GPU（每 rank 约 10–11
个请求时 ITL ≈ 20 ms；每 rank 输出约 650–720 tok/s 后随 batch 不再增长；C176 时 decode KV>90% 的时间占 38%、开始出现等预分配，KV 也接近约束）。prefill 计算忙碌约 35–41%、缓存份额
约 96%（理论 96.6%），有明显余量。

interactivity 口径：aiperf Full-Response ITL（每请求平均 token 间隔）取 P50 / P90 后取倒数；P50 ≥ 50 ⇔ ITL p50 ≤ 20 ms。
C160 的 P50 为 49.4（略低于 50）；作对照，P8D4 最优（23,051）的 P50 为 47.4。

可靠性问题（详见 `operations/STATUS.zh-CN.md`）：只重启 decode 后部分 prefill→decode 传输通道失效（需同时重启 prefill）；
prefill-2 三次 GPU 侧挂起（对调 GPU 后仍挂在 prefill-2，跟着实例走，根因未明，§4.12）；共享 `/home` 写满导致编排中断；Mooncake 偶发分片传输失败。

讨论中形成的专题分析（指标口径、tokenizer 瓶颈原理、decode 容量模型更正、prefill 时延分解、write_back 理论分析等）见 §4。
与 InferenceX 官方 AgentX 的一致性检查见 §5：客户端与官方等价，tokenizer 提升在同类部署架构下可以带到提交里；但官方要求每个并发点重新部署，
我们的数字是在复用的栈上测的，提交前需要按官方协议复测。

## 1. 基线扫描（r2：yaocheng 2P1D 配置 + v0.5.21 必需项 + Mooncake 跨 rank 修复，grouped topk 关闭）

| CONC | tok/s/GPU | 总 tok/s | P50 / P90 interactivity | ITL p50 (ms) | TTFT p50 / p90 (s) | 成功 / error-dropped | GPU 缓存命中 |
|---|---|---|---|---|---|---|---|
| **112** | **20,545** | 410,901 | 71.6 / 41.6 | 14.0 | 6.0 / 16.1 | 12,319 / 4 | 91.1% |
| 128 | 19,758 | 395,168 | 71.3 / 40.5 | 14.0 | 7.8 / 30.4 | 12,376 / 5 | 86.6% |
| 144 | 18,001 | 360,023 | 52.9 / 34.8 | 18.9 | 10.2 / 55.1 | 11,393 / 4 | 74.8% |
| 160 | 17,754 | 355,080 | 75.4 / 39.5 | 13.3 | 19.6 / 85.2 | 11,079 / 3 | 75.6% |
| 176 | 16,298 | 325,962 | 96.9 / 49.4 | 10.3 | 38.5 / 127.2 | 10,009 / 11 | 62.1% |

decode 负载（每个测量窗口内各 DP rank 的均值，`results/r2/c*/decode_balance.txt`）：

| CONC | 每 rank 在跑请求 | KV 使用率 | KV>90% 时间 | 等预分配 | 等 KV 传输 |
|---|---|---|---|---|---|
| 112 | 5.87 | 0.41 | 1.9% | 0.04 | 2.14 |
| 128 | 6.06 | 0.50 | 6.2% | 0.21 | 3.86 |
| 144 | 7.01 | 0.63 | 19.7% | 1.60 | 5.61 |
| 160 | 5.03 | 0.68 | 28.3% | 2.97 | 8.84 |
| 176 | 3.75 | 0.76 | 41.4% | 5.60 | 11.94 |

并发升高时，decode 显存越来越多地被"已预分配、仍在等 prefill 传 KV"的请求占住，真正在 decode 的请求反而减少：
ITL 变低（interactivity 变高），但吞吐和 TTFT 都变差。各 rank 之间也不均衡（C144 时 DP3 曾 99% 且 33 个请求等预分配），
因为 decode 不上报 KV 事件，router 选 D rank 实际按请求个数轮转。

## 2. 正确性

见 `operations/STATUS.zh-CN.md` 18:52–19:19：r1（grouped topk=1）已知答案门禁 0/48；r2（grouped topk=0）简单集 48/48、困难集
38 正确 + 10 检索错位 + 0 编造。

## 3. 调优（计划 `PLAN-TUNING.zh-CN.md`，过程 `operations/STATUS.zh-CN.md`）

协议：同一个 r2 栈（引擎不重启，decode 已预热），每轮只换 router 开关。顺序是：重启 router → 对 3 个 prefill `/flush_cache`
→ 核对 12 个 rank 的首批 `#cached-token` 均为 0 → AgentX full（warmup 10/lane，3600 s，seed 42）。客户端每轮都带会话 header
（router 只在开会话亲和时读取）。表 E 的引擎节奏用来判断栈是否漂移。下表由 `scripts/tune_report.py` 生成，每轮结束后重跑更新。

<!-- tune-tables:begin -->

**A. 客户端结果**（AgentX；interactivity = 1/ITL 分位数）

| 轮次 | tok/s/GPU | 输出 tok/s/GPU | 完成请求 | 平均输入 | P50 / P90 intv | ITL p50 / p90 (ms) | TTFT p50 / p90 / 均值 (s) | GPU / host 命中（理论） |
|---|---|---|---|---|---|---|---|---|
| r2 C144 | 18,001 | 157.1 | 11,393 | 113.7k | 52.9 / 34.8 | 18.9 / 28.8 | 10.2 / 55.1 / 21.7 | 74.8% / 4.5%（96.2%） |
| t1-base C144 | 19,503 | 168.0 | 11,953 | 117.4k | 80.6 / 37.9 | 12.4 / 26.4 | 12.8 / 52.1 / 22.3 | 81.8% / 5.8%（96.3%） |
| t1-r1 C144 | 19,308 | 170.8 | 12,149 | 114.3k | 72.3 / 39.7 | 13.8 / 25.2 | 12.0 / 48.5 / 20.6 | 80.7% / 6.4%（96.2%） |
| t1-r1aff C144 | 21,639 | 185.0 | 13,192 | 118.0k | 71.4 / 33.1 | 14.0 / 30.2 | 11.2 / 24.4 / 14.8 | 80.9% / 11.7%（96.4%） |
| t1b-r2 C144 | 22,340 | 189.4 | 13,628 | 118.0k | 63.1 / 31.0 | 15.8 / 32.2 | 10.1 / 19.2 / 12.0 | 81.6% / 11.7%（96.5%） |
| t1c-best C128 | 21,554 | 186.0 | 13,337 | 116.2k | 61.9 / 35.3 | 16.2 / 28.3 | 6.3 / 13.3 / 7.6 | 87.2% / 6.8%（96.4%） |
| t1c-best C160 | 21,161 | 177.6 | 12,812 | 118.8k | 71.7 / 30.5 | 14.0 / 32.8 | 19.7 / 34.4 / 21.6 | 74.1% / 17.6%（96.5%） |
| t2a-best C144 | 22,238 | 189.0 | 13,520 | 118.4k | 63.1 / 31.0 | 15.9 / 32.3 | 11.4 / 20.4 / 12.4 | 84.8% / 10.2%（96.5%） |
| t2a-best C160 | 22,279 | 185.2 | 13,355 | 120.0k | 66.9 / 29.4 | 14.9 / 34.0 | 18.2 / 28.0 / 18.5 | 77.7% / 16.2%（96.6%） |
| t2a-best C176 | 21,218 | 169.8 | 12,393 | 123.3k | 79.8 / 30.9 | 12.5 / 32.3 | 31.5 / 41.8 / 30.8 | 67.1% / 26.4%（96.6%） |
| t4d-tok4 C144 | 25,269 | 208.9 | 15,177 | 119.9k | 47.4 / 39.6 | 21.1 / 25.3 | 2.1 / 5.4 / 3.2 | 86.5% / 8.2%（96.6%） |
| t4d-tok4 C160 | 28,479 | 223.5 | 16,293 | 125.9k | 49.4 / 44.7 | 20.2 / 22.4 | 2.5 / 7.1 / 4.0 | 80.5% / 13.9%（96.8%） |
| t4d-tok4 C176 | 30,562 | 232.5 | 17,044 | 129.2k | 48.9 / 44.7 | 20.5 / 22.4 | 3.1 / 11.1 / 5.5 | 75.7% / 18.6%（96.8%） |
| t5b-2p1d C176 | 28,900 | 230.7 | 13,575 | 122.6k | 54.8 / 49.2 | 18.2 / 20.3 | 8.7 / 46.1 / 19.9 | 50.3% / 40.1%（96.6%） |

**B. 瓶颈链**（服务端为测量窗口均值；首 token 前在途 / decode 有效并发为 AIPerf 的 effective prefill / decode concurrency）

| 轮次 | prefill 缓存份额 | prefill 新算 tok/s | prefill 排队（实例均值） | 首 token 前在途 | decode 有效并发 | decode 每 rank 在跑 | decode KV 使用率 | KV>90% 时间 | 等预分配 / rank | 等 KV 传输 / rank |
|---|---|---|---|---|---|---|---|---|---|---|
| r2 C144 | 87.9% | 43,645 | 3.48 | 68.0 | 58.9 | 7.01 | 0.63 | 19.7% | 1.60 | 5.61 |
| t1-base C144 | 88.5% | 44,851 | 3.96 | 73.2 | 47.7 | 5.53 | 0.59 | 15.3% | 0.83 | 6.38 |
| t1-r1 C144 | 87.8% | 47,001 | 2.90 | 68.7 | 50.3 | 5.87 | 0.61 | 16.9% | 1.19 | 5.82 |
| t1-r1aff C144 | 94.0% | 26,019 | 1.36 | 53.5 | 59.4 | 7.04 | 0.51 | 6.0% | 0.51 | 2.49 |
| t1b-r2 C144 | 94.7% | 23,628 | 1.04 | 44.8 | 65.2 | 7.74 | 0.51 | 1.1% | 0.00 | 1.88 |
| t1c-best C128 | 95.5% | 19,513 | 0.46 | 27.8 | 62.8 | 7.53 | 0.45 | 0.1% | 0.00 | 1.14 |
| t1c-best C160 | 93.2% | 28,548 | 1.64 | 76.2 | 58.1 | 6.82 | 0.52 | 4.4% | 0.03 | 2.85 |
| t2a-best C144 | 95.9% | 18,489 | 0.44 | 45.9 | 66.1 | 7.97 | 0.48 | 0.2% | 0.00 | 1.22 |
| t2a-best C160 | 95.7% | 18,954 | 0.50 | 67.9 | 64.3 | 7.66 | 0.48 | 0.7% | 0.00 | 1.14 |
| t2a-best C176 | 95.6% | 18,852 | 0.63 | 104.8 | 53.7 | 6.41 | 0.44 | 0.7% | 0.00 | 1.30 |
| t4d-tok4 C144 | 95.9% | 20,613 | 0.40 | 13.2 | 86.3 | 10.32 | 0.62 | 2.8% | 0.00 | 1.33 |
| t4d-tok4 C160 | 95.9% | 23,299 | 0.56 | 17.9 | 87.5 | 10.42 | 0.68 | 11.7% | 0.04 | 1.79 |
| t4d-tok4 C176 | 95.8% | 25,624 | 0.82 | 25.8 | 92.1 | 10.98 | 0.76 | 37.7% | 0.52 | 2.20 |
| t5b-2p1d C176 | 93.6% | 29,710 | 5.37 | 74.2 | 63.3 | 7.64 | 0.82 | 61.8% | 1.94 | 6.95 |

**C. prefill 实例均衡**（新算 tok/s / 缓存份额 / 排队）

| 轮次 | prefill-0（135） | prefill-1（137） | prefill-2（137） |
|---|---|---|---|
| r2 C144 | 14.4k / 88.5% / 3.3 | 14.9k / 87.9% / 3.6 | 14.3k / 87.2% / 3.6 |
| t1-base C144 | 15.2k / 88.0% / 4.9 | 15.2k / 88.5% / 3.8 | 14.5k / 88.9% / 3.2 |
| t1-r1 C144 | 15.9k / 88.2% / 2.7 | 15.3k / 87.9% / 3.0 | 15.8k / 87.3% / 3.0 |
| t1-r1aff C144 | 9.8k / 92.2% / 1.9 | 8.1k / 94.9% / 0.9 | 8.2k / 94.5% / 1.2 |
| t1b-r2 C144 | 8.2k / 93.6% / 1.1 | 8.8k / 94.2% / 1.5 | 6.6k / 96.0% / 0.5 |
| t1c-best C128 | 6.6k / 95.3% / 0.4 | 6.5k / 95.4% / 0.5 | 6.4k / 95.7% / 0.5 |
| t1c-best C160 | 9.2k / 93.8% / 1.6 | 9.8k / 92.7% / 1.9 | 9.5k / 93.2% / 1.4 |
| t2a-best C144 | 5.6k / 96.6% / 0.4 | 6.6k / 94.9% / 0.6 | 6.2k / 95.9% / 0.3 |
| t2a-best C160 | 6.4k / 95.8% / 0.4 | 6.5k / 95.4% / 0.5 | 6.1k / 96.0% / 0.5 |
| t2a-best C176 | 5.5k / 96.5% / 0.5 | 6.8k / 94.7% / 0.7 | 6.5k / 95.3% / 0.7 |
| t4d-tok4 C144 | 6.9k / 95.7% / 0.3 | 7.1k / 95.9% / 0.5 | 6.7k / 96.2% / 0.4 |
| t4d-tok4 C160 | 7.5k / 96.2% / 0.5 | 7.8k / 95.7% / 0.6 | 8.0k / 95.8% / 0.6 |
| t4d-tok4 C176 | 7.8k / 96.4% / 0.6 | 9.0k / 95.6% / 1.1 | 8.9k / 95.4% / 0.8 |
| t5b-2p1d C176 | 13.4k / 95.6% / 2.7 | 16.3k / 89.5% / 8.1 | — |

**D. decode rank 均衡**（8 个 DP rank 的最小 – 最大）

| 轮次 | 在跑请求 | KV 使用率 | KV>90% 时间 | 等预分配最多的 rank | 等 KV 传输 |
|---|---|---|---|---|---|
| r2 C144 | 6.28 – 7.57 | 0.56 – 0.69 | 8.0% – 30.8% | DP6 4.12 | 5.12 – 5.97 |
| t1-base C144 | 4.96 – 6.00 | 0.54 – 0.64 | 5.0% – 22.8% | DP6 1.32 | 5.97 – 6.73 |
| t1-r1 C144 | 5.43 – 6.69 | 0.54 – 0.68 | 8.7% – 27.7% | DP4 2.91 | 5.39 – 6.16 |
| t1-r1aff C144 | 6.82 – 7.21 | 0.46 – 0.53 | 3.5% – 8.8% | DP3 2.28 | 2.23 – 2.75 |
| t1b-r2 C144 | 7.17 – 8.67 | 0.49 – 0.52 | 0.3% – 2.5% | DP0 0.00 | 1.79 – 2.08 |
| t1c-best C128 | 7.16 – 7.97 | 0.44 – 0.45 | 0.0% – 0.5% | DP0 0.00 | 1.01 – 1.23 |
| t1c-best C160 | 6.49 – 7.48 | 0.49 – 0.54 | 2.7% – 5.4% | DP2 0.04 | 2.51 – 3.23 |
| t2a-best C144 | 7.48 – 8.38 | 0.47 – 0.50 | 0.1% – 0.5% | DP0 0.00 | 1.12 – 1.42 |
| t2a-best C160 | 7.07 – 8.08 | 0.46 – 0.49 | 0.5% – 1.4% | DP2 0.01 | 1.04 – 1.20 |
| t2a-best C176 | 5.99 – 6.95 | 0.43 – 0.46 | 0.4% – 1.0% | DP0 0.00 | 1.12 – 1.49 |
| t4d-tok4 C144 | 10.08 – 10.61 | 0.61 – 0.62 | 2.1% – 3.4% | DP0 0.01 | 1.23 – 1.45 |
| t4d-tok4 C160 | 9.57 – 11.31 | 0.67 – 0.68 | 10.2% – 13.4% | DP5 0.06 | 1.63 – 1.99 |
| t4d-tok4 C176 | 10.37 – 11.58 | 0.75 – 0.77 | 33.6% – 41.1% | DP1 0.61 | 2.08 – 2.43 |
| t5b-2p1d C176 | 7.13 – 9.02 | 0.81 – 0.83 | 59.4% – 63.4% | DP6 2.05 | 6.60 – 7.45 |

**E. 引擎节奏（判断栈是否漂移）与 router 会话**：decode 每步 ms 按同一 3 s 内最忙 rank 的在跑数分组；prefill 为排队时满 8192-token chunk 的耗时中位数 (ms)

| 轮次 | decode @6 | @8 | @10 | @12 | prefill-0 / 1 / 2 | 会话命中 / 新绑定 |
|---|---|---|---|---|---|---|
| r2 C144 | 41.8 | 47.3 | 61.3 | 72.4 | 1207 / 1216 / 1313 | — |
| t1-base C144 | 36.5 | 40.0 | 52.1 | 63.3 | 1227 / 1274 / 1228 | 0 / 0 |
| t1-r1 C144 | 36.3 | 40.0 | 51.9 | 63.8 | 1277 / 1305 / 1357 | 0 / 0 |
| t1-r1aff C144 | 36.0 | 40.0 | 52.8 | 64.8 | 1155 / 1040 / 1106 | 12,400 / 2,414 |
| t1b-r2 C144 | 35.3 | 40.1 | 52.5 | 63.6 | 1143 / 1141 / 943 | 12,809 / 2,444 |
| t1c-best C128 | 35.3 | 40.3 | 52.9 | 63.8 | 988 / 972 / 962 | 12,232 / 2,542 |
| t1c-best C160 | 35.7 | 40.4 | 52.2 | 63.6 | 1139 / 1131 / 1083 | 12,451 / 2,169 |
| t2a-best C144 | 35.5 | 39.7 | 53.1 | 63.7 | 973 / 1005 / 911 | 12,697 / 2,446 |
| t2a-best C160 | 35.4 | 40.1 | 52.4 | 63.7 | 990 / 1023 / 945 | 12,916 / 2,231 |
| t2a-best C176 | 36.2 | 40.7 | 52.2 | 63.0 | 977 / 970 / 980 | 12,342 / 2,026 |
| t4d-tok4 C144 | 38.5 | 41.5 | 55.4 | 67.5 | 982 / 1009 / 982 | 14,180 / 2,618 |
| t4d-tok4 C160 | 35.8 | 38.6 | 53.0 | 66.0 | 1039 / 1022 / 1002 | 15,637 / 2,473 |
| t4d-tok4 C176 | 35.5 | 38.4 | — | 66.0 | 1045 / 1088 / 1025 | 16,611 / 2,446 |
| t5b-2p1d C176 | 36.4 | 40.7 | 52.1 | 63.6 | 1217 / 1565 / — | 13,527 / 2,051 |

<!-- tune-tables:end -->

### 3.1 阶段 1 结论（C144）

1. **r2 C144 偏低是因为 decode 刚重启**。同配置的 t1-base 为 19,503 tok/s/GPU（+8.3%）。decode 在最忙 rank 跑 8 / 10 / 12 个请求时，
   每步耗时从 47.3 / 61.3 / 72.4 ms 回到 40.0 / 52.1 / 63.3 ms，与预热后的 r2 C112/C128 一致。预热后 C144 与 r2 C128（19,758）、
   C112（20,545）处在同一平台。
2. **R1（prefill 完成即释放记账）：吞吐差在噪声内，保留**。总吞吐 −1.0%（平均输入短 2.6%），输出与完成请求 +1.6%~1.7%；
   TTFT p50/p90/均值 −6%/−7%/−8%，首 token 前在途 73.2 → 68.7，decode 有效并发 47.7 → 50.3。省下的 TTFT 把更多请求送进了 decode，
   但 C144 下 decode 已接近饱和（KV>90% 时间 15–17%）：ITL p50 12.4 → 13.8 ms，与并发增加相抵。历史 P8D8 C80 的 +5.4% 是在
   decode 不饱和、瓶颈在 prefill 记账时测得的。
3. **P 会话亲和：+12.1%，接受**。21,639 tok/s/GPU，输出 +8.3%，完成请求 +8.6%，TTFT p90 48.5 → 24.4 s。代价是 ITL p90
   25.2 → 30.2 ms，P90 interactivity 39.7 → 33.1；P50 为 71.4，仍在 50–100。

4. **R2（decode 按在途 token 需求选 rank）：+3.2%，接受**（阶段 1b）。22,340 tok/s/GPU，完成请求 +3.3%，平均输入相同（118.0k）；
   输出 +2.4%。TTFT p50/p90 −10%/−21%，首 token 前在途 53.5 → 44.8，decode 有效并发 59.4 → 65.2，KV>90% 时间 6.0% → 1.1%，
   等预分配 0.51 → 0，等传输 2.49 → 1.88。代价是 ITL p50 14.0 → 15.8 ms（P50 interactivity 71.4 → 63.1，仍在 50–100）。
   R2 把 KV（显存）拉平了（各 rank 0.49–0.52，KV>90% 时间 0.3%–2.5%），但各 rank 在跑请求数的差距变大（7.17–8.67，亲和后为 6.82–7.21）：
   短请求多的 rank 在跑请求更多。DP attention 下全局每步跟着最忙的 rank 走，这是 ITL 上升的原因之一。

5. **最优配置推到 C160：21,161 tok/s/GPU，比 C144 低 5.3%**（完成请求 −6.0%，输出 −6.2%）。引擎节奏与 C144 相同，不是栈漂移。
   多出来的 16 个会话让 prefill GPU KV 装不下工作集：客户端 GPU 命中 81.6% → 74.1%，host 命中 11.7% → 17.6%（更多前缀要从 host 读回），
   prefill 新算 23.6k → 28.5k tok/s，TTFT p50 10.1 → 19.7 s，首 token 前在途 44.8 → 76.2。decode 反而更闲：有效并发 65.2 → 58.1，
   每 rank 在跑 7.74 → 6.82。所以 C160 的约束回到 prefill 侧，具体是 prefill GPU KV 容量（首 token 路径变长），这正是 2a 要解决的。
   目前峰值在 C144；12:05 起测 C128 以确定峰值左侧。
6. **最优配置 C128：21,554 tok/s/GPU，比 C144 低 3.5%**（完成请求 −2.1%，输出 −1.8%，平均输入短 1.5%）。TTFT p50/p90 6.3/13.3 s，
   GPU 命中 87.2%，首 token 前在途 27.8，decode KV>90% 时间 0.1%，两侧都有余量但并发不够。**峰值区间为 C128–C144，C144 最高**；
   各配置的 P50 interactivity 都在 50–100（C144 为 63.1）。

7. **2a（prefill mem 0.88 + HiCache 135=400 / 137=200 GB/rank，只重启 prefill）**：GPU KV 2.27M → 2.43M tokens/rank（+7.2%）。
   - C144：22,238 tok/s/GPU，与 t1b-r2（22,340）持平。prefill 缓存份额 94.7% → 95.9%，新算 −22%，排队 0.44，但 C144 的约束在 decode，吞吐不变。
   - C160：**22,279 tok/s/GPU，比 2a 前的 C160（21,161）+5.3%**，完成请求 +4.2%，输出 +4.3%。GPU 命中 74.1% → 77.7%，prefill 新算
     28.5k → 19.0k，TTFT p90 34.4 → 28.0 s，首 token 前在途 76.2 → 67.9，decode 有效并发 58.1 → 64.3，KV>90% 时间 4.4% → 0.7%。
   - 结论：2a 把平台从 C128–C144 拓宽到 C144–C160（约 22.3k），没有抬高峰值；平台上的约束是 decode 每步耗时。
     prefill 卡显存最低空闲 135 为 8.8 GiB、137 为 11.1 GiB，mem 0.88 安全。

8. **decode `--tokenizer-worker-num 4`（只重启 decode；之后为消除传输通道失效又重启了 prefill）**：
   - C144：**25,269 tok/s/GPU（+13.6%）**，TTFT p50/p90 2.1/5.4 s（11.4/20.4），P50 / P90 interactivity 47.4 / 39.6；decode 有效并发 66 → 86，
     每 rank 在跑 8.0 → 10.3。这一轮是重启后首轮，decode 每步比预热状态慢 3–8%，数字偏保守。
   - C160：**28,479 tok/s/GPU（+12.7%）**，输出 +7.0%，完成请求 +7.4%（平均输入 +5%），P50 / P90 interactivity 49.4 / 44.7，KV 0.68。
   - C176：**30,562 tok/s/GPU（+7.3%）**，输出 +4.0%，完成请求 +4.6%（平均输入 +2.6%），P50 / P90 interactivity 48.9 / 44.7，TTFT p50/p90 3.1/11.1 s；
     decode 有效并发 92.1、每 rank 在跑 11.0、KV 0.76、KV>90% 37.7%、等预分配 0.52（decode 分配 KV 等待 0.97 s），prefill 到达→完成 5.6 s。
     吞吐仍在上升，但 decode KV 开始成为约束；09:32–09:33 有一次分片失败引起的成串传输失败（约 10 条）。
   - TTFT 分段（sampler）：修复前 C176 prefill 到达→完成 5.9 s、decode TTFT 34.4 s，decode 主进程 CPU 101%；修复后 C144 分别为 3.1 / 2.9 s，
     4 个 tokenizer worker 合计 CPU 106%（最忙 29%）。

### 3.2 瓶颈分析

> **更正（10-14）**：下文"亲和后约束在 decode 每步耗时"的判断有误。按 TTFT 分段与进程 CPU 复核，亲和之后到 2a 的吞吐平台实际来自 decode 的
> 单进程 HTTP / TokenizerManager 饱和（见 3.1 第 8 条、`operations/STATUS.zh-CN.md` 02:30）。decode GPU 成为约束是在 tokenizer worker ×4 之后。

- **亲和前（t1-base / t1-r1）瓶颈在 prefill 一侧**：
  1. 会话的后续轮次被 kv-aware 按负载分到别的 rank，缓存份额只有 88%（理论 96%），prefill 每秒新算约 45k token、排队 3–4；
  2. 首 token 前在途 68–73 个；
  3. 已在 decode 预分配 KV 的请求等传输（每 rank 5.8–6.4 个），占住 decode 显存（KV>90% 时间 15–17%）；
  4. decode 实际在跑的请求只有每 rank 5.5–5.9 个。
- **亲和后 prefill 不再是瓶颈**：
  1. 会话固定在同一 prefill rank，缓存份额 94.0%（接近理论 96.4%），新算量 −42%（26.0k tok/s），排队 1.4；
  2. 首 token 前在途 53.5；
  3. 等传输降到每 rank 2.5，KV>90% 时间 6.0%；
  4. decode 每 rank 在跑 7.04、有效并发 59.4。
- **现在的约束在 decode 每步耗时**：每步耗时随 batch 增大（最忙 rank 在跑 10 时约 52.8 ms/步，MTP 平均接受 3.61 → 约 14.6 ms/token），
  ITL p90 升到 30 ms。prefill 算力有富余，ITL p50 14 ms 距 P50 interactivity 50 对应的 20 ms 还有空间，所以下一步是提高并发。
- **decode 有效并发要结合 decode 速度看**（Little 定律：在途数 = 吞吐 × 停留时间）。r2 C144 的 58.9 是 decode 慢、请求停留久造成的；
  t1-base 的 47.7 是 decode 快的结果；t1-r1aff 在同样的 decode 速度下到 59.4，才是 decode 承载真实增加。

### 3.3 负载均衡

- **prefill 三个实例各轮都均衡**：新算量差异在 ±10% 以内，135 上 4 卡 + 7 网卡的实例与 137 上两个实例相当。亲和后 prefill-0
  新算量略高（9.8k vs 8.1k）、缓存份额略低（92.2% vs 94.5–94.9%），差距不大。
- **decode 8 个 rank**：亲和前，每 rank 在跑 4.96–6.00 个，KV>90% 时间 5%–28%，等预分配集中在个别 rank（t1-r1 DP4 2.91）。
  亲和后，在跑 6.82–7.21 个，KV>90% 时间 3.5%–8.8%，KV 使用率 0.46–0.53。
  decode 的不均衡主要来自"等传输的预分配请求"在个别 rank 上堆积；传输等待变少后，即使 router 仍按请求个数选 decode rank，各 rank 也趋于均衡。

### 3.4 进行中与下一步（10-15 更新）

最优配置 = R1 + P 会话亲和 + R2 + 2a + decode tokenizer worker ×4，C176 30,562 tok/s/GPU。之后的尝试：3P1D C192 因 prefill-2 挂起无有效结果
（部分数据未超过 C176，§4.11）；2P(TP4)1D C176 28,900（−5.4%）；3P1D + decode mem 0.92 的预热轮因 prefill-2 第三次挂起作废。
栈状态和下一步见 `HANDOFF.zh-CN.md`；提交 InferenceX 前还需要按官方协议复测（§5.3）。

## 4. 专题分析（讨论中形成的结论，10-13 ~ 10-15）

数据来源：sampler（`runtime/runs/r2/tune/samples.jsonl`，`scripts/sample_window.py` 按测量窗口取差分）、`results/<round>/c*/decode_balance.txt`、
`engine_pace.txt`、客户端 `agentx_conc*.json`。

### 4.1 指标口径

- **ITL / interactivity**：aiperf 的 Full-Response ITL，每个请求 =（最后一个 chunk − 第一个有内容的 chunk）÷（输出 token − 1），是请求级平均，
  和 TPOT 同类。MTP 一次多个 token 会被平均掉；输出链路上的延迟（例如 tokenizer 进程排队）也算在里面。P50 / P90 interactivity = 1 / 所有请求
  ITL 的 P50 / P90。表里的"服务端 ITL"是 sglang decode 的直方图，口径不同，只作参考。
- **TTFT**：只有 decode 服务记录 `time_to_first_token`（decode 收到请求 → 发出首 token）。router 同时把请求发给 prefill 和 decode，所以它约等于
  客户端 TTFT 减去 router 和网络开销。prefill 不流式输出，没有 TTFT，只记录到达 → 完成（e2e）及其中的排队、bootstrap、传输。
- **decode 在跑 vs 有效并发**：在跑（`#running-req`）是 decode GPU batch 里的请求数；有效并发是 aiperf 从客户端看的"已出首 token、未完成"的请求数，
  按 Little 定律（在途 = 速率 × 停留时间）计算，包含输出链路上的排队。有效并发高可能是 decode 慢、请求停留久，要结合速度看。
- **与 P8D4 对照**：P8D4 最优 23,051（C80）ITL p50 21.1 ms、P50 interactivity 47.4，每 decode rank 约 11.9 个；3P1D tok4 C144 为 25,269、
  21.1 ms、47.4，每 rank 约 10.8 个。两者都说明每个 decode rank 在 ITL ≈ 20 ms 时承载约 10–11 个请求。P8D4 满足 P50 ≥ 50 的是 21,081（50.3）。

### 4.2 decode tokenizer 瓶颈：原理与适用范围

- sglang 每个服务有一个 HTTP + TokenizerManager 进程（asyncio 单线程，受 GIL 限制，最多用一个核），每个 DP rank 一个 scheduler，一个 detokenizer。
  入口：解析请求 JSON、套聊天模板、分词（decode 也要对完整 prompt 做一遍）。出口：从 detokenizer 收每一步的增量，组装 SSE chunk、JSON 序列化、写回连接。
- decode 承担全部流式输出：60–90 个流，每 40–70 ms 一步，`stream_interval=1` 时每个在跑请求每步一个输出事件，每秒上千个；同时每秒接入 4–5 个长 prompt。
  单线程打满（实测 101%）后事件在 asyncio / ZMQ 里排队，首 token 和后续 token 都被推迟。AgentX 每条 lane 拿到响应才发下一轮，系统于是停在平台上，
  而 prefill 和 decode 的 GPU 都不忙。prefill 不流式输出、请求又分到 3 个实例，主进程只用 30–48%。
- `--tokenizer-worker-num 4` 启动 4 个独立进程分摊连接。C144 / C160 / C176 合计 106% / 117% / 125%，最忙一个 29% / 35% / 36%，余量充足。
- 适用范围：只有那个单进程接近饱和时才有收益（负载 ∝ 请求速率 × 同时在流的请求数），没有副作用；对 prefill 瓶颈本身没有作用。阶段 1 里的
  "decode 饱和"实际主要是这个进程饱和；同条件下的组间比较仍有效，但 R1 等项在 tok4 下的表现可能不同。是否依赖 AgentX 的闭环模式见 §5.2。

### 4.3 prefill HiCache 扩容（2a）为什么 C144 没有吞吐收益

| | C144 2a 前 | C144 2a 后 | C160 2a 前 | C160 2a 后 |
|---|---|---|---|---|
| 客户端 GPU 命中 | 81.6% | 84.8% | 74.1% | 77.7% |
| 总命中（理论上限约 96.5%） | 93.3% | 95.0% | 91.7% | 93.9% |
| prefill 新算 tok/s | 23.6k | 18.5k（−22%） | 28.5k | 19.0k（−33%） |
| tok/s/GPU | 22,340 | 22,238 | 21,161 | 22,279（+5.3%） |

命中率提高了，但 C144 的约束在 decode 一侧（当时以为是 GPU 每步耗时，后来证实是 decode tokenizer），省下的 prefill 算力换不成吞吐。C160 的约束正好在
prefill GPU KV，所以有 +5.3%。另外可提升的空间本身就小：对话里每轮新增的内容必须新算，理论上限约 96.5%，亲和后已到 93.3%。

### 4.4 上限的算法

- **decode**：日志每 rank 每 40 步一行，相邻两行时间差 ÷ 40 = 每步耗时。DP 同步执行时每步由最忙 rank 决定，所以按"同一 3 s 内最忙 rank 的在跑数"分组
  取中位数（表 E）。每 rank 每秒产出 ≈ 在跑数 × 3.61 ÷ 每步耗时：在跑 8–14 个时都在 650–720 tok/s，基本走平（长上下文下每步耗时与 batch 内上下文
  总量近似成正比）。局限：这是最忙 rank 的产出，平均只有它的约 75%；接受长度是固定模拟值；"走平"不等于总吞吐完全不能涨（C160 靠各 rank 更满，输出 +7%）。
- **prefill 计算**：粗算一个满 chunk（8192 token）约 1.1 s → 每 rank 约 7.4k tok/s，12 个 rank 约 89k，C144 用了约 23%（偏乐观，没算小 batch 开销和长前缀
  chunk 更慢）。按日志中 batch 执行时间占比：C144 约 35–36%、C160 约 40–41%（各 rank 32–49%）。余量约 2–2.5 倍。
- **prefill 显存**（mem 0.88）：权重 119.8 GiB + KV 池 125.1 GiB（2.43M token/rank），预留 33.4 GiB；满载高水位下每卡最少还空 8.7–11 GiB，
  不够再提到 0.90。

### 4.5 decode 容量模型更正：理论约 150，实测约 90

| | 当初估算（`REVIEW.zh-CN.md` §6） | C176 实测 |
|---|---|---|
| 每个在跑请求占的 KV | 约 12.5 万 token（平均输入 + 生成预留） | 20.7 万 token |
| 已占 KV 但没开始生成的请求 | 没算 | 等 KV 传输每 rank 2.2 个，占 KV 11.9% |
| 能同时在跑的请求 | 约 190（考虑不均衡约 150） | 约 88（每 rank 11 个） |

- **20.7 万的算法**：测量窗口内 decode 日志所有行的 `#token` 之和 ÷ `#running-req` 之和。这是**按请求在 decode 里停留的步数加权**（停得越久，在采样里出现越多），
  不是完成请求的平均输入。验证：C144 / C160 / C176 的平均输入 125k / 130k / 134k，按输出长度加权 156k / 166k / 172k，实测 178k / 195k / 207k。
  长输入的请求输出也更长、在 decode 里停得更久，解释了大部分差距；剩下的（172k → 207k）可能来自已生成的 token 和分配预留，没有单独核实。
- **等预分配（`#prealloc-req`）不占 KV**：它是还没分到 KV、在排队的请求，是 KV 不够的症状。占着 KV 却没开始生成的是**等 KV 传输**（`#transfer-req`；early
  策略下请求一到 decode 就分配 KV，一直占到 prefill 算完并传完）：

| | 等 KV 传输 / rank | 其占 KV 比例 | 等预分配 / rank | 在跑请求占 KV |
|---|---|---|---|---|
| C144 | 1.33 | 7.1% | 0.00 | 62% |
| C160 | 1.79 | 9.4% | 0.04 | 68% |
| C176 | 2.20 | 11.9% | 0.52 | 76% |

- 等传输的数量 ≈ 每 rank 到达率 × prefill 侧（排队 + 计算 + 传输）时间，所以和 prefill 时延直接相关；它只在 decode 显存紧张时（C176）才真正挤占在跑请求。
- 重算：每 rank 2.99M ×（可用约 90% − 12%）÷ 20.7 万 ≈ 11 个，8 个 rank 约 90 个，与实测一致。ITL 约束（每 rank 10–11 个时 ITL ≈ 20 ms）也在同一处，
  两个约束在 C176 同时触顶。

### 4.6 C144 → C176，prefill 到达 → 完成从 3.1 s 涨到 5.6 s

| 平均每请求（s） | C144 | C160 | C176 | C144 → C176 |
|---|---|---|---|---|
| 排队 | 0.58 | 0.83 | 1.23 | +0.65 |
| bootstrap（等 decode 分配 KV） | 0.04 | 0.11 | 0.97 | **+0.93** |
| KV 传输 | 0.58 | 0.72 | 0.80 | +0.22 |
| 计算及其他（差值） | 1.91 | 2.37 | 2.60 | +0.69 |
| **到达 → 完成** | 3.11 | 4.03 | 5.60 | +2.49 |
| 参照：decode 分配 KV 等待 | 0.01 | 0.08 | 0.97 | |

- 最大一项是 bootstrap：C176 时 decode KV>90% 的时间占 38%，decode 分配 KV 平均要等 0.97 s，prefill 侧 bootstrap 等待同步涨到 0.97 s，
  也就是 **decode 显存不足反压到了 prefill**。
- 排队 +0.65 s：平均忙碌只有 35–45%，但服务时间长尾很重（有 30 万 token 级的请求），会话亲和又把请求固定在 rank 上，单 rank 的排队随负载非线性增长。
- 计算 +0.69 s：GPU 命中 86.5% → 75.7%，host 读回和新算增多（新算 20.6k → 25.6k tok/s），单请求计算更久。
- 传输 +0.22 s：每请求 KV 6.36 → 6.84 GB（输入更长），同时在传的也更多。
- 所以"prefill 计算只忙 35–45%"和"prefill 时延变长"不矛盾：增长里约 37% 来自 decode 反压，其余来自排队长尾和命中下降，不是算力打满。

### 4.7 HiCache write_through → write_back 与 prefill_complete（理论分析，未实测）

- **为什么绑定**：v0.5.21 的 `_validate_prefill_complete`（`srt/arg_groups/pd_disaggregation_hook.py`）对 prefill 的要求是 HiCache 只能是不带 storage
  backend 的 L2 write_back，同时也不支持 decode radix / decode HiCache 等；报错原文是 "does not yet support"，属于实现尚未覆盖，不是原理冲突。
  prefill_complete 走 optimistic prefill 路径：不等 decode 分配就开算，算完 `mark_prefill_complete()`，再等 decode 分到 KV 后发送；代码里对 write_through
  还有一段额外的 SWA 分支保护判断。推测是 write_through 的异步回写和"算完后在 prefill GPU 上保留 KV 等 decode"这段流程还没适配（推断，未核实）。
- **两种策略的差别**：

| | write_through（现用） | write_back |
|---|---|---|
| 何时写 host | 新算的 KV 插入 radix 后立即异步 D2H | 只在 GPU 淘汰时 D2H |
| GPU 淘汰 | host 已有副本，直接释放 | 先回写再释放，可能落在新请求分配显存的关键路径上 |
| 有效缓存容量 | ≈ host（GPU 上的内容在 host 都有副本） | ≈ GPU + host |
| 持续 D2H 流量 | 与新算量成正比 | 与淘汰量成正比，通常更少 |

- **对链路各环节的影响**：
  1. router / 到达：无影响。
  2. prefill 排队与计算：省掉持续 D2H；但 GPU 池满时淘汰要先回写，可能拖慢新请求开算。host 命中后的 H2D 回读不变。
  3. prefill → decode 传输：机制不变，PCIe 上少了持续 D2H，竞争略降。
  4. 缓存容量：每 token KV 约 53 KB（每请求传输 6.4–6.8 GB ÷ 12–13 万 token）。每 rank GPU 2.43M token，host 200 GB ≈ 3.8M、400 GB ≈ 7.5M，
     write_back 让总容量 137 约 +65%、135 约 +30%。但总命中已是 94–95%（理论 96.6–96.8%），最多再补 1.5–2.5 个百分点。
  5. decode（来自 prefill_complete，不是 write_back 本身）：decode 只在 prefill 完成后才分配 KV，占 KV 的时间从"prefill 排队 + 计算 + 传输"（3–5.6 s）缩到
     传输本身（0.6–0.8 s），等传输占的 KV 从 7–12% 降到约 2%。代价是 prefill 要把算完的 KV 留在 GPU 上，直到 decode 分到显存；decode 紧张时会挤占
     prefill 的缓存。
- **适合开启的场景**：decode KV 是约束，且有相当比例被等传输的请求占着（prefill 时延相对 decode 停留时间长）；prefill GPU 有余量保留算完的 KV；
  缓存容量（而不是 GPU 命中）在限制命中率。
- **本形态的预期**：C144–C160 decode KV 不紧，预期约 0。C176 / C192 等传输占 12%，释放后相当于再多约 10% decode KV，作用和 decode mem 0.85 → 0.92
  （+12.4%）同类；但每 rank 产出已在 650–720 tok/s 走平、ITL 已在 20 ms，收益上限受 decode 计算限制。因此先用 mem 0.92 检验"多给 decode KV 能不能涨吞吐"：
  没有收益就不必做 prefill_complete，有收益再叠加。

### 4.8 decode mem 与 cuda graph 的取舍

- mem 0.85 → 0.92：每 rank KV 2.99M → 3.36M（+12.4%），每卡多约 19 GiB。所有 cuda graph 合计约 6.4 GiB/卡（主模型 4.8 + MTP 1.6）。
- graph 每 rank 最大 batch 32 → 16 只省约 3 GiB（约 +2% KV），而 C176 时有 11.9% 的时间某 rank batch 超过 16，要退回非 graph；32 → 24 省约 1.5 GiB，
  超出的时间 0.3%。省出的显存比 mem 提升小一个数量级，为了归因清楚这次只改 mem。
- 没直接上 0.95：KV 池以外满载实际用量约 12.6 GiB/卡，0.92 余量约 9 GiB、0.94 约 4 GiB、0.95 约 1 GiB，容易在测量中途 OOM（一次作废约 1.5 小时）。
  计划先在 0.92 下测满载高水位，再决定是否到 0.94（届时顺带把 graph 降到 24）。

### 4.9 重启后首轮偏慢；C160 的 interactivity 为什么比 C144 高

- 按"最忙 rank 在跑 6 / 8 / 10 / 12 个"对齐 decode 每步耗时：t4d C144（重启后第一轮正式测量）38.5 / 41.5 / 55.4 / 67.5 ms，t4d C160 35.8 / 38.6 / 53.0 / 66.0，
  预热参照 t2a C144 35.5 / 39.7 / 53.1 / 63.7。首轮慢 3–8%；r2 C144 首轮慢 13–17%（`REVIEW.zh-CN.md` §8）。yihou 在 v0.5.21 也记录过"新栈第一次跑 decode
  每步慢 7–12%，flush 不能消除，跑过一轮后消失"。
- 所以 tok4 C144 的 25,269 / P50 47.4 偏保守，C160 的 P50 49.4 更高并不是"并发越高越快"；两轮的 decode 负载、rank 均衡、输出长度几乎一样，
  多出的 16 个并发变成了首 token 前在途（13 → 18）、等传输（1.33 → 1.79）和 KV 占用（0.62 → 0.68）。
- 这一点和 InferenceX "每个并发点重新部署"的规则直接相关，见 §5.3。

### 4.10 Mooncake "Process failed for slice"

`Process failed for slice (opcode: 1, length: 65536, …): transport retry counter exceeded`：Mooncake 把 KV 切成 64 KB 分片，用 RDMA WRITE 写到 decode 显存。
这条报错是网卡在 RC 层重传次数用完（`MC_IB_TIMEOUT=18`，约 1 s 乘以重试次数内收不到确认），该分片作废，Mooncake 会重新提交（`max_retry_cnt 9`）。
C160 期间 2 条分片失败，请求级传输失败为 0，说明重试成功。属于 RoCE 链路偶发拥塞 / 丢包，过去两天零星出现，以 135 的 prefill-0 → decode 居多，作为巡检观察项。
C176 的 09:32–09:33 出现过一次成串（约 10 条请求级失败），对结果影响小。这和"只重启 decode 后传输通道持续失效"不是同一个问题。

### 4.11 3P1D C192 与 16 卡形态

- **C192**：11:31 prefill-2 第二次挂起，中止，无 AgentX 结果。挂起前 26 分钟的服务端数据与 C176 同期对比：请求速率 5.03 vs 5.19 req/s，输出约 4,690 vs 4,840 tok/s
  （−3%），decode TTFT 10.2 vs 5.0 s。仅作参考：3P1D 当前配置大概在 C176 到顶，限制是 decode KV 与计算。
- **2P(TP4)1D C176**：28,900 tok/s/GPU（−5.4%）。瓶颈转到 prefill（GPU 命中 50.3%、prefill 排队 10.2 s），两个实例还因 bootstrap 积压加会话亲和而失衡。
- **1P(TP8)1D**（未测）：TP8 权重减半，每 rank GPU KV 约 3.1M（prefill 总 GPU KV 比 2P(TP4) 多约 29%），一个实例内没有实例间失衡，部署是 137 整机 + 138；
  TP8 通信更多，但 prefill 不是计算瓶颈。只能放在 137 上。

### 4.12 prefill-2 挂起

- 三次：10-14 05:30、11:31（137 card4–7），10-15 约 02:29（与 prefill-1 对调到 card0–3 后）。签名相同：4 个 scheduler 主线程都卡在
  `process_batch_result_disagg_prefill` → `stream.synchronize()`，GPU 100%，日志没有错误；第三次 1 小时后看门狗超时退出（exit 137）。prefill-0、prefill-1 从未挂起。
- 对调 GPU 后仍挂在 prefill-2：**不是 card4–7 的硬件问题**。prefill-2 与 prefill-1 的差别：同节点第二个实例、`PID_OFFSET` 60000（prefill-1 为 30000）、
  端口、启动顺序、按实例名分开的 triton / HF 缓存目录。

**10-15 排查（进行中）**

已排除：
- **ROCr IPC socket 撞名**：ROCr 默认 IPC 在宿主机共享网络命名空间里绑定 `\0hsa<容器内 pid>`，所以要用 PID_OFFSET 错开。实测只有 scheduler 进程绑定
  （prefill-1 为 `@hsa30696` 等 4 个）。prefill-1 运行 12 小时后 PID 计数器才到 34141，碰不到 prefill-2 的 60000 段，两个实例的 scheduler 不会撞名。
- **KV 池打满**：第三次挂起时 DP1 的 token usage 为 0.98，但前两次挂起前只有 0.01–0.72，不是共同条件。
- **新 GEMM 形状的首次编译**：第一、三次挂起前日志里有 aiter 的新形状提示，第二次没有。

现场特征（三次一致）：4 个 scheduler 主线程都停在 `prefill.py:910` 的 `copy_done.synchronize()`（等前向结果拷回主机的 event），Mooncake
`transfer_worker` 也在等 event，4 张卡 GPU 100%，日志没有错误。也就是前向在 GPU 上没有执行完，同时有 kernel 在自旋。py-spy 只能看到 Python 线程，
ROCr 内部线程看不到。

主要假设：**prefill 打开了 scratch 回收（`HSA_NO_SCRATCH_RECLAIM=0`），在显存紧张时 scratch 分配进入等待，与集合通信互锁。**
- ROCr 7.2 `GpuAgent::AcquireQueueMainScratch`（`amd_gpu_agent.cpp` 约 1952 行）：scratch 分配失败时，只要本卡还有队列持有一次性大 scratch，
  请求方队列就进入 retry，一直等到那块 scratch 释放。如果持有者是在等对端 GPU 的通信 kernel，而对端在等这个被挂起的队列跑到集合通信，
  就会形成循环等待：GPU 一直自旋，无报错。`HSA_NO_SCRATCH_RECLAIM=1` 时不走这条等待路径（scratch 常驻在队列上，分配失败直接报错）。
- prefill 设成 0 是为了规避早期的 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`（`scripts/bench-harness/issue.md` §3.4），decode 一直是 1，从未挂起。
- 时间线：mem 0.85 时（每轮结束每卡剩约 20 GiB）三个实例共跑了十几个小时，没有挂起；三次挂起都发生在 2a 把 prefill mem 提到 0.88 之后
  （剩约 9–13 GiB，PyTorch 缓存分配器占住其余显存，ROCr 的 scratch 在 PyTorch 之外分配）。但 2a 同时加大了 HiCache，时间线不能单独归因。
- 显存余量各实例相近，三次都落在 prefill-2 也可能是概率问题；目前没有找到 prefill-2 特有的触发条件。

验证方案（A/B，10-15 05:15 起）：只给 prefill-2 加 `HSA_NO_SCRATCH_RECLAIM=1`（`config/stage-mem092-nsr.sh`），prefill-0 / prefill-1 保持 0 作对照，
继续跑 3P1D + decode mem 0.92 的预热、C176、C192。同时启用 `scripts/hang_watch.py`：按引擎自己的计数器（`realtime_tokens_total`、
`kv_transfer_latency_ms_count`）判断有请求但 10 分钟没有进展即为挂起，自动调用 `scripts/hang_capture.sh` 抓 py-spy 和 rocgdb
（各 HSA 队列读写指针、活跃 dispatch、全部线程栈），用来拿到挂起时 GPU 侧的直接证据。

## 5. 与 InferenceX 官方 AgentX 的一致性检查（2026-10-15）

问题：AgentX 每条 lane 拿到响应才发下一轮，tokenizer worker 的提升是否依赖这种压测模式？我们的压测和 InferenceX 是否一致，提交时能否拿到同样的提升？

核对对象：我们用的 InferenceX `918524ff`（2026-08-27，aiperf 子模块 `754356e`，2026-08-12）与 InferenceX main `3babaeed`（2026-10-15）。main 已重组为
`inferencex-e2e/`，AgentX 客户端改为 PyPI 包 `agentx-harness==1.0.9`（内含 aiperf，默认 `agentx` 场景）。

### 5.1 客户端：等价

| 项 | 我们（`results/t4d-tok4/c176/benchmark_command.txt`） | 官方 `agentx` preset |
|---|---|---|
| 场景约束 | `inferencex-agentx-mvp` | `agentx` 是 mvp 的复制加 CLI 默认值，约束逐行相同（强制流式、ignore_eos、首轮前缀 cache bust、全系统闲置上限 10 s、最短 900 s） |
| 发压模型 | `--concurrency 176`，176 条闭环 lane，保留 trace 中轮次间隔（单轨迹闲置上限 300 s） | 相同 |
| 数据集 | `semianalysis_cc_traces_weka_062126`，393 条 | 相同（GLM-5.2 用不截断语料） |
| 时长 / 预热 | 3600 s；每 lane 预热 10 个；轨迹起点 25–75%；grace 1800 s | 相同 |
| 其他参数 | streaming、chat、seed 42、失败阈值 0.10、server token count、slice 1.0、TCP_USER_TIMEOUT 900000 | 相同 |
| 会话 header | `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true` | 官方文档列为 DP attention 场景的可选项，大量官方 recipe 在用 |
| 吞吐口径 | （输入 + 输出 token）÷ 窗口时长 ÷（prefill + decode GPU 数） | 相同 |
| MTP 接受长度 | 模拟 3.61 | 官方 golden AL：GLM-5.2 MTP 5 = 3.61，PR 检查单要求用该值 |

两版 aiperf 只有 15 个文件不同。与发压有关的是 `agentic_replay.py` / `branch_orchestrator.py` / `request_rate.py`：基准时长到点停止发送时，
正在延迟等待的子 agent 轮次会被正确取消，让会话树收尾（之前要等到 grace 超时）。另有实时统计少算 sweep 曲线的性能改动。两者都不改变测量窗口内的发压节奏。

### 5.2 tokenizer 提升不是闭环压测的产物，但依赖部署架构

- 闭环下服务端时延的任何下降都会直接变成吞吐；开环（固定到达率）下，同样的单进程饱和会表现为队列无限增长，能维持的最大到达率被它卡住。tokenizer worker
  提高的是服务端的真实容量，闭环只是把它直接体现为吞吐。官方 AgentX 是同样的闭环 lane 模型，所以只要部署架构相同，提升就能带到提交里。
- 它依赖的是**谁在做流式输出**：
  - **sglang 自己的 HTTP 服务对外流式输出**：我们（infera router → decode HTTP），以及官方 AMD MI355X 用 sglang-router 的 recipe（例如 DSv4
    `multi_node/srt-slurm-recipes/dsv4/sglang/mi355x-fp4/agentx/disagg-umbp-dspark.yaml`）。decode 的 TokenizerManager 在流式路径上，瓶颈存在。
    这个官方 recipe 本身就给两个角色都设了 `tokenizer-worker-num: 8` 和 `stream-interval: 20`。
  - **Dynamo 前端**：官方 NVIDIA 的 GLM-5.2 sglang AgentX 多机 recipe 都是这样（例如 GB300 `disagg-2p1d-p-dep8-d-dep16-c192-mtp.yaml`）。decode 设
    `skip-tokenizer-init: true` 和 `stream-interval: 30`，分词、反分词、SSE 由 Rust 前端（可多实例）完成，sglang 的 TokenizerManager 不在流式路径上，
    这个瓶颈不存在，`--tokenizer-worker-num` 也不起作用。
- 提交时需要：前端用同类架构，并在 recipe 中显式写上 decode 的 `--tokenizer-worker-num 4`（或更多）。

### 5.3 可能导致提交结果与我们不同的差异

1. **测试协议（影响最大）**：官方要求每个并发点重新部署服务，不 flush、不复用在线服务（`agentx-standalone.md` 明确写出，`infx/bench/agentic/run.py`
   拒绝一个部署测多个 CONC）。我们调优时在同一个栈上连续测多轮，每轮重启 router 并 flush。按 §4.9，重启后首轮 decode 每步慢 3–17%，而 30,562 是 tok4 重启后的
   第三轮。按官方协议测，C176 可能偏低，需要实测：整栈重启 → 直接测 C176，预热只用 AgentX 自带的每 lane 10 个。
2. **前端 / router**：R1、P 会话亲和、R2 来自 infera rust router（相对各自基线 +12%、+3%）。官方 AMD recipe 用 sglang-router（consistent_hashing + dp-aware，
   decode round_robin），NVIDIA 用 Dynamo + nginx 按 `X-Dynamo-Session-ID` 亲和。提交时能否使用 infera router、镜像里的 Mooncake 补丁能否接受，需要按
   InferenceX 规则确认（main 已要求 recipe 依赖的上游 PR 先合入，#2210）。
3. **`stream-interval`**：不是官方规定，由各 recipe 自选。InferenceX main 的 137 个 AgentX recipe 文件里，61 个不设（sglang 默认 1），设了的取值有
   1、10、20、30、32、50、60；规则文档、PR 检查单和 AgentX 校验代码都没有对它的要求。GLM-5.2 的 sglang 多机 recipe 用 30（B200/GB200/GB300，Dynamo 前端）
   或 60（H200）。我们是 1。v0.5.21 在 `stream_interval > 1` 时按 `len(output_ids) % k == 1` 发送，首 token 立即发出，TTFT 不受影响；ITL 是请求级平均，
   也不受影响。每秒输出事件数会降到约 1/k 以下（MTP 一步多个 token，实际更少），decode tokenizer、router、客户端的 CPU 都会下降。值得测。
4. **harness 版本**：用 `agentx-harness==1.0.9` 复跑可以消除版本差，预期影响很小。

## 6. PD 分离与 mixed（aggregated）在 AgentX 下的取舍（10-15）

背景：InferenceX 上 B300 的 GLM-5.2 用 4 卡 aggregated 实例（TP4 + DP4 attention）在 C48 跑到 42,047 tok/s/GPU
（`../inferencex-glm52-b300-dynamo-agg-20261015/README.zh-CN.md`）；同为 MI355X 的 ATOM 4 卡 TP4 aggregated 在 C32 / C48 为
29,924 / 34,195（P50 交互 55.5 / 35.1），与我们 20 卡 3P1D 的 C176 = 30,562（48.9）同一量级。用户决定先把 PD 做到极致，mixed 之后再做。

### 6.1 PD 分离最初的动机

DistServe、Splitwise、Mooncake 等工作的出发点：
1. **两个阶段的特性相反**：prefill 一次处理成千上万个 token，受计算限制；decode 每步每个请求只出一两个 token，受显存带宽和 KV 容量限制。
   合在同一批 GPU 上，两边都难以跑在最优点。
2. **互相干扰**：mixed 下一个长 prefill 块插进来，同批所有 decode 请求这一步都要等它，ITL 出现尖峰；decode 批也会拖慢 prefill，抬高 TTFT。
   分离后 decode 步速稳定，TTFT 与 ITL 可以分别保证，按"同时满足两项 SLO 的吞吐"计更划算。
3. **两边可独立配置与扩缩**：并行方式（我们 prefill TP4、decode TP8）、batch、CUDA graph、显存分配、chunk 都分开调；P:D 配比按负载调整，
   也可以用不同硬件。

### 6.2 AgentX 下 PD 的代价被放大

AgentX 是"很多长会话、每轮只追加少量内容"：缓存命中约 95%，平均输入 12–13 万 token，输出约 1000 token，吞吐把命中缓存的输入也计入（输入占 99% 以上）。
- **prefill 真正要算的很少**：每轮新增约 5%，为 prefill 单独留的算力用不满（3P1D 的 prefill 只忙 35%–45%，§4.4）。
- **同一会话的 KV 存两份、每轮重传**：prefill 缓存着前缀；每轮还要把整段上下文 KV（6.4–6.8 GB）传给 decode，decode 再存一份完整副本
  （每个在跑请求约 20.7 万 token，§4.5），且轮与轮之间不复用。mixed 下会话 KV 只有一份，在跑请求的 KV 就是下一轮要命中的前缀。
- **干扰变轻**：每轮新算 token 少，prefill 块短，插进 decode 步的代价不大。

### 6.3 PD 在我们这里仍有的优势

- **交互速度更稳**：decode 步不被 prefill 打断。同为 MI355X，ATOM aggregated 在 C48 的 P50 交互只有 35.1；3P1D 在 C176 为 48.9。
  若坚持 P50 ≥ 50 的约束，高负载区间 PD 更容易满足。
- **冷启动和超长请求被隔离**：预热阶段动辄几十万 token 的冷请求在 PD 里只占 prefill；aggregated + DP attention 下各 rank 每步同步，
  一个 rank 算长 prefill，同实例所有 rank 的 decode 都要等。
- **配比与配置可单独调**：decode 加显存、prefill 加 HiCache 互不影响。

### 6.4 在 PD 内部借用 mixed 的长处

PD 在 AgentX 下最大的损失是"每轮重传整段 KV、decode 不跨轮复用"，可以在 PD 框架内部分补回：
- **decode radix cache + decode 侧会话亲和**（PR #184 移植到 v0.5.21）：同一会话后续轮次落到同一 decode rank，prefill 只传新增部分，
  decode 里的 KV 跨轮复用。历史数据：P4D4 C40 +3.0%，P8D8 +0.2% / +0.6%（当时 decode 不是瓶颈，且 decode 显存紧时复用率低）。
  代价是缓存的前缀与在跑请求争用 decode GPU KV，需要配合扩容（§6.5）或较低的缓存占比。
- **prefill_complete**：decode 只在 prefill 完成后分配 KV，释放"等 KV 传输"请求白占的 7%–12% decode KV（§4.7）。

### 6.5 decode 侧能否用 HiCache（write_back）扩容

**不能增加同时在跑的请求数**：decode 每一步都要读每个在跑请求的完整上下文，这部分 KV 必须在 HBM 里（单个请求约 20.7 万 token、约 11 GB，
经 PCIe 从主机内存读回比 HBM 慢一到两个数量级）。主机内存只能存"暂时不参与 decode 的 KV"：
1. **跨轮复用的会话前缀**：配合 decode radix cache 和 decode 侧会话亲和（§6.4），已结束轮次的 KV 先留在 GPU 的 radix 树，满了用
   `write_back` 淘汰到主机，下一轮落到同一 rank 时从本机主机内存读回（PCIe），代替从 prefill 经 RDMA 重传整段。这能减少传输量、
   让 decode 的 GPU KV 主要留给在跑请求，但不增加在跑容量。历史上单开 decode HiCache 曾使每步多 1.7 ms、吞吐 −4.5%（REVIEW §7），需在新条件下复测。
2. **等待 KV 传输的请求**：v0.5.21 有 `--disaggregation-decode-host-receive-threshold`（GPU KV 紧张时把传入的 KV 先收进主机内存），
   正对应 C176 时这 7%–12% 的占用；但参数说明写明"需要 dense MHA，且传输后端支持主机目标，目前没有内置后端支持"，GLM-5.2（MLA/DSA）
   + Mooncake 用不了。这一块目前只能靠 prefill_complete。
3. **被抢占（retract）的请求**：`--disaggregation-decode-retraction-backup host_pool` 可把被抢占请求的 KV 备份到主机，避免重算；
   C176 时 retract 为 0，暂无价值。

扩大 decode 在跑容量的可用手段仍是：decode mem fraction（0.92 进行中）、prefill_complete、降低 cuda graph 占用、更紧凑的 KV 格式
（ATOM 在 MI355X 上用 fp4 的 indexer cache，sglang 里对应选项 `enable_deepseek_v4_fp4_indexer` 看名字只针对 DSv4，GLM-5.2 是否可用待查）。
