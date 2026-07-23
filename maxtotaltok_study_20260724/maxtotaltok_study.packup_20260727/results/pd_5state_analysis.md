# PD 5-state 队列分析（用 llm-pd-bottleneck-finder 提取）

每条 scheduler stat line 是 per-DP-rank。用 `extract_pd_stats.py` 对各配置 decode/prefill 日志跑
5 态队列分析（steady-state，滤掉 warmup）。

## 5 态映射（SGLang counter → pipeline 位置）

| state | counter | side | 含义 |
|-------|---------|------|------|
| 1 prefill-input | `#queue-req` | prefill | 等 prefill 算力的请求 |
| 2 prefill-outbound | `#inflight-req` | prefill | KV 算完、正经 RDMA 写给 decode（transfer 交叉验证点）|
| 3 decode-admission | `#prealloc-req` | decode | 到了 decode、等 KV slot/running 座位 |
| 4 decode-transfer-in | `#transfer-req` | decode | slot 已留、KV 正传入 |
| 5 decode-running | `#running-req` | decode | 正在生成 token |
| mem | `#retracted-req` | decode | 因 KV 不足被驱逐（DSv4 = NotImplementedError 崩）|

## decode 侧实测（per rank）

| config | 池 | max_run/rank | **running(5)** | **transfer-in(4)** | admission(3) | retract | occupancy | 结果 |
|--------|-----|-----|------|------|------|------|------|------|
| baseline | 3.9M | 64 | mean15.3 max16 | 0.8 | 0 | 0 | **24%** | 稳 29.4K |
| R2 | 163840 | 64 | mean15.2 max16 | 0.9 | 0(max2) | 0 | 24% | 稳 29.6K |
| G2 | 131072 | **8** | **mean2.8 max5** | **mean9.2 max13** | 0.8 | 0 | 35% | **塌** |
| G3(conc512) | 3.9M | 64 | mean12.8 max25 | mean10.2 max27 | 0 | 0 | **20%** | decode健康,router熔断 |

## prefill 侧实测（per rank，真实流量）

| config | 池 | full usage峰值 | swa峰值 | running | **queue(1)** | **inflight(2)** |
|--------|-----|------|------|------|------|------|
| prefill(conc流量) | 3,441,920 | **0.02(2%)** | 0.13 | 0 | **20** | 6 |

## 判定（apply verdict rule）

**8k/1k @ 1P1D = prefill-bound**：
- state1 prefill-input `#queue-req` **backed up (max 20)** ← 请求堆在 prefill 门口
- state2 prefill-outbound `#inflight-req` **shallow (6)** ← transfer 不是瓶颈（KV 一算完就传走）
- decode running **只 20-24% cap** ← decode 巨大余量，KV 远非瓶颈
- 即使 conc=512（4× 过载）decode running 仍只 20% cap → 过载表现为 router circuit-breaker 拒绝（prefill 跟不上），**不是 decode KV 崩**

**G2 吞吐塌的真相**（非 KV 不足、非 wire 慢）：
- max_running=8/rank 把 running 硬砍到比自然 running(16) 还小 → transfer 管道(9.2 在途)预留 slot 速度 > 8-cap 消化速度 → 池被 transfer-waiting 塞满、running 饿死到 2.8 → 吞吐塌。
- **本质：max_running < 自然 running → 自己掐死 decode**。

**transfer 结论**：state2 `#inflight-req` 在所有配置所有负载都 shallow(≤6)。若 RDMA/mooncake 是瓶颈，state2 会堆积——从未堆积。**transfer 被豁免**。
