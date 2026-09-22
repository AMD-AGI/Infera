# P8D8 C144 全量动态 RCA 与首轮 partial 交叉验证

## 1. 结论

Replacement full run 已完整完成：

- Warmup：1598/1598，4494.15 秒；
- Profiling：目标 3600 秒，phase wall time 3640 秒；
- 成功 profiling records：6697；
- AgentX exit code：0；
- 所有配置、逐 rank metrics、CUDA graph counter、GPU/HCA counter 均保存。

全量结果与首轮 partial RCA 的主结论一致，并显著提高了置信度：

1. **C144 的主问题是耦合 queue/capacity 反压，而不是 compute graph 或网络。**
   - Prefill queue p50/p90：135/181；
   - Decode prealloc queue p50/p90：11/27；
   - Decode transfer queue p50/p90：146/193；
   - Decode KV usage p50/p90：约 90.4%/95.0%；
   - HiCache host pool 典型使用率约 99.96%。

2. **`max_running_requests` 未触顶。**
   - 全局 cap 256、每 rank cap 32；
   - Prefill/Decode running max 20/65；
   - Decode 单 rank running 未达到 32。

3. **Decode CUDA graph 不是限制。**
   - `decode_cuda_graph=472,683` passes；
   - `decode_none=0`；
   - counter-derived coverage 100%。

4. **没有持续 Prefill compute 不均。**
   - 每 rank cache-miss/input compute tokens max/min 1.037x；
   - CV 1.17%；
   - workload/cache-hit tier 有差异，但实际 Prefill compute 极均衡。

5. **没有 NIC 硬瓶颈证据。**
   - 每 rail 长窗口均值 1.01–1.54 GB/s；
   - 5 秒区间 p99 12.03 GB/s，最大 19.31 GB/s；
   - 单 rail 标称约 50 GB/s；
   - retransmit、retry exhausted、ACK timeout 增量均为 0。

6. **Actual cache hit 仍显著低于 theoretical。**
   - GPU hit 72.38%；
   - CPU/HiCache hit 12.22%；
   - overall hit 84.60%；
   - theoretical hit 95.85%。

全量窗口还补充了首轮 partial 未稳定暴露的一点：

> Decode prealloc queue 确实存在持续 allocation wait，但更大的积压仍位于 allocation
> 之后的 transfer queue。也就是说，Decode KV capacity 同时限制新请求分配，并让
> 已预留 KV 的请求长期等待 Prefill input/transfer/done signal。

---

## 2. 实验口径

- Run：`c144-resume-138-136-20260922T085229Z`
- 复用服务：`c144-138-136-20260922T071057Z`
- Prefill：crsuse2-m2m-138
- Decode：crsuse2-m2m-136
- P8D8、TP8/DP8/DPA
- Prefill HiCache on，Decode HiCache off
- C144、warmup 10/lane、profiling 3600 秒
- `max_running_requests=256`
- graph max BS=256
- PD rank affinity on；rank i → ionic_i
- simulated acceptance 3.61

该 run 使用首轮已热 service/cache，重新执行完整 warmup 和 profiling。它不是
fresh-launch A/B，但适合验证 C144 机制是否在完整窗口持续成立。

结构化数据：

- `runs/c144-resume-138-136-20260922T085229Z/analysis/full-run-data.json`
- `runs/c144-resume-138-136-20260922T085229Z/analysis/runtime-summary.json`
- `runs/c144-resume-138-136-20260922T085229Z/bench/agentx_conc144.json`

---

## 3. Headline 结果

### 3.1 Throughput 与请求量

| 指标 | 全量值 |
|---|---:|
| successful profiling requests | 6697 |
| warmup records | 1598 |
| records error dropped | 28 |
| QPS mean | 1.846 |
| input throughput | 190,508.7 token/s |
| output throughput | 1,740.0 token/s |
| total throughput | 192,248.6 token/s |
| total throughput/chip | 12,015.5 token/s/chip |

28 个 error record 均为 warmup `InvalidInferenceResultError`。Profiling 的已导出
成功记录为 6697；phase 结束时另有 99 条在 grace-period timeout 中取消，不进入
最终成功 request distribution。

### 3.2 Request latency

| 指标 | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| TTFT | 54.00 s | 116.70 s | 198.49 s | 255.86 s | 422.61 s |
| E2E | 68.72 s | 133.82 s | 218.23 s | 277.17 s | 442.74 s |
| Full-response ITL | 14.53 ms | 17.48 ms | 20.54 ms | 23.62 ms | 42.04 ms |
| ISL | 64,735 | 122,702 | 241,089 | 347,694 | 669,638 |
| OSL | 409 | 977 | 1,990 | 3,437 | 8,549 |

TTFT 与 ITL 再次解耦：

- 请求在首 token 前等待几十到数百秒；
- 一旦进入 generation，典型 ITL 仍约 14.5 ms。

### 3.3 Effective concurrency

由 request timestamps 重建：

| 指标 | avg | p50 | p90 | max |
|---|---:|---:|---:|---:|
| Effective total | 178.2 | 188 | 238 | 251 |
| Effective Prefill | 153.1 | 164 | 220 | 243 |
| Effective Decode | 25.1 | 24 | 41 | 67 |

典型墙钟状态中，绝大多数 in-flight request 位于首 token 前；Decode generation
并发远低于 Prefill phase。

---

## 4. 与历史 sweep-of-record C144 的复现度

| 指标 | 历史 C144 | 本次 full run |
|---|---:|---:|
| throughput/chip | 12,555 | 12,016 |
| TTFT p50 | 52.45 s | 54.00 s |
| ITL p50 | 14.40 ms | 14.53 ms |
| overall cache hit | 85.20% | 84.60% |
| Effective total p50 | 197 | 188 |
| Effective Prefill p50 | 169 | 164 |
| Effective Decode p50 | 25 | 24 |
| successful requests | 6862 | 6697 |

除了 throughput/chip 低约 4.3%，其余关键指标高度接近。考虑 Prefill 节点由 137
换成 138、service cache history 不同，本次已成功复现历史 C144 的容量与延迟形态。

---

## 5. Warmup

- duration：4494.15 秒（74.90 分钟）
- 25 个 `KVPoll.WaitingForInput` 1800 秒 timeout
- rank 分布：`{0:5, 1:2, 2:2, 3:2, 4:3, 5:3, 6:3, 7:5}`
- 28 个 raw `InvalidInferenceResultError`

与首轮：

- 4571.12 秒；
- 29 个 WaitingForInput timeout；
- 39 个 InvalidInferenceResultError。

两次均出现“长平台 → 1800 秒 timeout → 继续推进”，证明该机制可复现且不是
单次随机故障。

---

## 6. Profiling queue 与 Decode admission

### 6.1 Prefill

跨 8 rank 求和：

| 指标 | p50 | p90 | max |
|---|---:|---:|---:|
| running | 10 | 13 | 20 |
| waiting queue | 135 | 181 | 206 |
| active token_usage | 0.51 | 0.92 | 1.94 |

Prefill running 低、waiting queue 高。Active token usage 低不表示 GPU prefix
cache 为空：绝大多数 GPU slots 由 `kv_evictable_tokens` 中的历史 prefix 占据。

### 6.2 Decode

| 指标 | p50 | p90 | max |
|---|---:|---:|---:|
| running | 26 | 42 | 65 |
| prealloc queue | 11 | 27 | 46 |
| transfer queue | 146 | 193 | 210 |
| token_usage（8 rank 求和） | 7.23 | 7.60 | 7.84 |

换算成平均每 rank Decode KV usage：

- p50：约 90.4%
- p90：约 95.0%
- max 同时状态：约 98.0%

SGLang Decode pipeline：

```text
prealloc queue
  -> 等待 request slot / metadata buffer / GPU KV pages
  -> 分配 destination pages，向 Prefill 发布地址
transfer queue
  -> destination 已预留
  -> 等待 Prefill input、RDMA transfer、done signal
scheduler waiting/running
  -> KV 到齐后进入 generation
```

Full run 同时看到 prealloc=11 和 transfer=146，说明：

- Decode KV capacity 已开始阻止一部分请求完成 allocation；
- 更主要的积压发生在 allocation 之后的 transfer/handoff 阶段；
- 大量 preallocated KV 被尚未进入 running 的请求占据，继续压缩可分配空间。

这正是 Decode admission backpressure。

---

## 7. Cache hierarchy

### 7.1 AgentX headline

| 层级 | hit rate |
|---|---:|
| GPU/device | 72.38% |
| CPU/HiCache | 12.22% |
| overall | 84.60% |
| theoretical | 95.85% |

实际 miss 约 15.40%，理论 miss 约 4.15%。按 miss token fraction 粗略估算，
实际需要计算的比例约为理想无限缓存口径的 3.7 倍。

### 7.2 Effective-token counter

Sampler profiling window delta：

- device hit：503.80M
- host hit：104.62M
- input/miss：87.51M

该 counter 的 effective-token 口径得到约 87.4% hit，高于 AgentX headline
84.6%。两者分母和 phase-boundary accounting 不完全相同；容量判断使用 AgentX
headline，rank 间实际 compute 比较使用同源 counter。

### 7.3 HiCache capacity

- host total：37.7247M tokens
- host used p50：37.7095M
- usage：约 99.96%

GPU Radix prefix cache 与 host pool 均接近满载。HiCache 贡献约 12.2% headline
host hit，但没有恢复到 theoretical hit。

---

## 8. Rank 均衡

### 8.1 Prefill compute

每 rank `prefill_effective_tokens_total{mode="input"}`：

| rank | actual compute tokens |
|---:|---:|
| 0 | 10.94M |
| 1 | 10.79M |
| 2 | 10.75M |
| 3 | 11.00M |
| 4 | 11.00M |
| 5 | 11.15M |
| 6 | 10.83M |
| 7 | 11.04M |

- max/min：1.037x
- CV：1.17%

全量窗口比 partial 的 CV 5.1% 更均衡。可以高置信排除持续 Prefill compute
rank imbalance。

### 8.2 Evictable GPU prefix

每 rank `kv_evictable_tokens` p50 约 2.967M–3.043M：

- cross-rank CV p50：6.4%
- max/min p50：1.22x
- max/min p90：1.65x

存在短时 cache-state 差异，但没有 rank 长期显著空闲或显著独占容量。

---

## 9. CUDA graph

Profiling counter delta：

- Decode graph passes：472,683
- Decode eager/none passes：0
- coverage：100%

所有 rank 启动时捕获到 request batch size 32，而实际单 rank running 未达到该边界。
因此 graph boundary/eager fallback 不是 C144 queue 或 TTFT 的原因。

Prefill graph 被配置明确禁用，`prefill_none=21,710` 属于预期路径。

---

## 10. Rail

使用 HCA 硬件 counter：

```text
tx_rdma_ucast_bytes
rx_rdma_ucast_bytes
```

结果：

- 每 rail 长窗口平均：1.01–1.54 GB/s
- 5 秒 interval p99：12.03 GB/s
- 5 秒 interval max：19.31 GB/s
- 单 rail 标称：约 50 GB/s
- retransmit bytes/packets：0
- retry exhausted：0
- ACK timeout：0

Prefill TX 与 Decode RX 一致。网络链路没有持续饱和、重传或硬错误证据。

---

## 11. Full run 与 partial run 是否一致

### 一致的结论

1. Warmup 都有跨全部 rank 的 1800 秒 WaitingForInput timeout。
2. Prefill queue 长，Effective Prefill concurrency 占主导。
3. Decode transfer queue 极高，Decode KV 接近满载。
4. HiCache host pool 接近 100%。
5. Actual hit 显著低于 theoretical。
6. `max_running_requests` 未触顶。
7. Decode graph 100% 覆盖。
8. Prefill compute 在 rank 间均衡。
9. Rail 无硬瓶颈证据。

### Full run 对 partial 的修正

1. Partial 只有约 1002 秒，throughput/chip 仅 8470；full run 为 12,016。
2. Partial TTFT p50 18.2 秒，受早期完成样本偏置；full run 稳定到 54.0 秒。
3. Partial prealloc queue p50 为 0，未覆盖稳态；full run p50 为 11。
4. Partial rank compute CV 5.1%；full run 收敛到 1.17%。
5. Partial 适合发现机制，full run 才适合 headline throughput/latency。

Full run 没有推翻 partial RCA，而是确认主机制并补出了稳态 allocation wait。

---

## 12. 根因分层

### 直接证实

1. Prefill waiting queue 与 Decode prealloc/transfer queue 同时积压。
2. Decode KV capacity 典型达到 90%–95%。
3. GPU/host prefix hierarchy 均处于高容量压力。
4. Actual cache hit 比 theoretical 低约 11.3pp。
5. 25 个 warmup WaitingForInput timeout 跨全部 rank。
6. Max-running、CUDA graph、compute rank balance、NIC hard fault 均不是主因。

### 高置信度机制

1. C144 arrival/fan-out 拉长 Prefill queue 与 prefix reuse distance。
2. Cache miss/host restore 增加 Prefill service time，进一步放大 queue。
3. Decode KV 被 running 与 preallocated transfer requests 占用。
4. 后续请求同时在 prealloc 与 transfer 阶段等待。
5. 一部分 warmup request 超过 1800 秒，靠 timeout 释放后继续推进。

### 仍需 request-level tracing

1. prealloc queue 中 request slot、metadata slot、KV token budget 各自等待多久；
2. transfer queue 中等待 Prefill input、RDMA bytes、done signal 各多久；
3. Prefill enqueue、GPU/host lookup、compute、transfer submit 的逐请求时间；
4. C80→C112 的第一处拐点究竟从 cache locality、Prefill compute 还是 Decode
   admission 开始。

---

## 13. 下一步选择

C144 已完成“放大并确认故障域”的任务。继续无差别复现 C144 的信息增益较低。

为了回答最初的“为什么 HiCache on 的峰值仍在 C80”，下一步应：

| 指标 | C80 | C112 |
|---|---:|---:|
| throughput/chip | 19,513 | 19,014 |
| TTFT p50 | 5.69 s | 11.30 s |
| TTFT p90 | 22.12 s | 64.47 s |
| actual cache hit | 94.81% | 91.62% |
| Effective Prefill p50 | 27 | 69 |
| Effective Decode p50 | 37 | 40 |

C80 是吞吐峰值的边界基线；C112 已出现吞吐回落、TTFT/cache hit/Prefill
concurrency 恶化，但尚未进入 C144 的全面正反馈。成对比较比继续观察已经饱和的
C144 更能定位第一个变坏阶段。

1. 增加 request-level phase timestamps；
2. fresh launch，清空旧 cache history；
3. 同一 deployment 按原顺序运行 C80 → C112；
4. 对比每阶段 dwell time、cache mode、rank、KV allocation 与 transfer completion。

两个点都应记录：

```text
router dispatch / selected rank
decode bootstrap start
prealloc queue enter/exit
KV allocation done / destination published
prefill enqueue/start
GPU/host/storage cache lookup result
prefill compute start/end
transfer submit/start/done
done signal emit/receive
decode transfer queue exit
decode waiting/running start
```

对应脚本已准备在：

`bench/glm5p2_pd/results/p8d8-c80-c112-tracing-20260922/`

脚本使用 SGLang 现有 ReqTimeStats + OTLP `request,mooncake` spans，并额外给
per-request log/trace 增加 device/host/storage/miss cache-tier 信息。实验脚本带
`--confirm-exclusive` 硬门禁，不会自动启动。

如果 C112 没有足够 transfer timeout 样本，再补带 tracing 的 C144，用于放大
WaitingForInput 路径。
