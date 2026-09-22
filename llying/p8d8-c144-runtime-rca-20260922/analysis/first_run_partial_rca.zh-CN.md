# P8D8 C144 首轮动态 RCA：Prefill 排队、Decode KV 反压与指标倒挂

## 1. 结论

首轮动态复现已经足以回答本次最核心的问题：

> C144 的异常不是由全局 `max_running_requests`、单个 DP rank 触顶、CUDA graph
> 回退或单 rail 硬故障造成。主瓶颈是首 token 前的耦合排队：Prefill 服务队列很长，
> 同时 Decode KV 接近满载、Decode transfer queue 大量积压；HiCache host pool
> 也已经接近 100%。三者形成正反馈，使请求长期停留在 Prefill/KV handoff 域。

直接证据如下。

1. **`max_running_requests` 没有触顶。**
   - 配置为全局 256，即 DP8 每 rank request pool 上限 32。
   - partial profiling 中 Prefill/Decode 聚合 running 最大仅 18/51。
   - Prefill 单 rank running 最大 9，Decode 单 rank最大 18，均远低于 32。

2. **Decode CUDA graph 没有回退。**
   - partial profiling 内 `decode_cuda_graph=121,099` passes。
   - `decode_none=0`，counter 口径覆盖率为 100%。
   - 启动时八个 rank 均捕获到 request batch size 32，完整覆盖实际最大 running
     batch 18/rank。

3. **不是单 rank 或单 rail 故障。**
   - Warmup 的 29 次 `KVPoll.WaitingForInput` 1800 秒 timeout 分布在全部八个
     Decode rank，每 rank 2–5 次。
   - profiling 中 queue 确有 rank 间差异，但没有一个 rank 单独解释全局停顿。
   - 8 条 rail 无新增 `req_tx_retry_excd_err` 或 `tx_rdma_ack_timeout`。
   - profiling 平均每 rail 约 0.51–1.39 GB/s，远低于单条 400 Gb/s 链路标称带宽；
     不能排除微秒级长尾，但排除了持续链路带宽打满和已计数硬故障。

4. **Prefill 与 Decode capacity/handoff 同时形成反压。**
   - Prefill queue：p50 `141`、p90 `159`、max `179`。
   - Decode transfer queue：p50 `147`、p90 `163`、max `178`。
   - Decode 每 rank KV usage p50 `88%~96%`，八 rank 聚合 p50 等价于约 `93%`，
     p90 约 `96.4%`。
   - Prefill GPU KV pool 在 8 个 rank 上合计约 25.15M token slots。典型采样时：
     - 当前正在处理的请求锁住约 1.29M slots（`kv_used_tokens`，约 5.1%）；
     - 已完成请求留下、仍可能复用的历史 prefix 占约 23.85M slots
       （`kv_evictable_tokens`，约 94.8%）；
     - 真正尚未存放任何 KV 的空闲位置只剩约 16K slots
       （`kv_available_tokens`，约 0.064%）。
   - `token_usage` 只近似描述第一类“当前请求正在使用、不能驱逐”的 KV，不包含
     第二类“暂时没有请求使用、但仍留在 GPU 等待复用”的 Radix prefix。因此
     `token_usage≈5.1%` 不能读成“整个 GPU prefix cache 只用了 5.1%”；准确读法是
     active request 占用较少，但 GPU prefix cache 几乎被历史 prefix 填满，host
     HiCache 也接近满载。

5. **HiCache 增加了容量，但没有消除 working-set cliff。**
   - profiling 中 host pool 平均使用 `37,709,937 / 37,724,672 tokens`
     （约 `99.96%`）。
   - 实际 effective prefix hit：
     - device hit `92,678,592`
     - host hit `21,820,352`
     - miss/input `23,524,587`
     - overall hit `82.96%`
   - 同期 trace 理论 prefix hit 约 `95.5%`，相差约 `12.5` 个百分点。
   - HiCache 提供了约 `15.8%` 的 host hit，但 pool 已满，仍不能兑现理论局部性。

因此最合理的机制不是“Prefill 或 Decode 某一端孤立变慢”，而是：

```text
C144 并发与 fan-out
  -> Prefill queue 增长
  -> prefix 等待期间 GPU/host working set 被轮转
  -> 实际 cache hit 低于理论值，Prefill 实算量增加
  -> 大量已启动 Decode bootstrap 的请求等待 KV
  -> Decode KV 接近满载，transfer queue 无法及时排空
  -> WaitingForInput 超过 1800 秒，timeout 释放部分请求
  -> 系统以 timeout/crawl 的方式继续推进
```

这也解释了原 sweep 中为什么高并发时 Effective Prefill Concurrency 和 TTFT
暴涨，而 Effective Decode Concurrency 与聚合 Decode throughput 下降。

---

## 2. 实验口径

### 2.1 配置

- Prefill：`crsuse2-m2m-138` / `10.245.157.237`
- Decode：`crsuse2-m2m-136` / `10.245.154.168`
- P8D8，TP8/DP8/DPA，GPU 0–7
- Prefill HiCache on，ratio 1.5，write-through
- Decode HiCache off（Decode MTP 与 HiCache 不兼容）
- `max_running_requests=256`
- Prefill/Decode graph max BS=256
- AgentX C144，warmup 10/lane，目标 profiling 3600 秒
- PD rank affinity on
- GPU rank i → ionic_i；destination affinity on
- MTP simulated acceptance 3.61
- `index_share_for_mtp_iteration=false`

Live assertion 保存于：

- `runs/c144-138-136-20260922T071057Z/live-config.json`
- `runs/c144-138-136-20260922T071057Z/snapshot/`

这次只把原 sweep 的 Prefill 节点从 137 换为 138。镜像 digest、Decode 节点、
计算配置、HiCache、rank/rail affinity、warmup 和运行参数均保持一致。

### 2.2 采样

- Engine/Router Prometheus：目标 2 秒间隔，保留全部 label，不预先跨 rank 聚合。
- GPU/HCA：目标 5 秒间隔。
- Server logs：Prefill、Decode、Router 持续保存。
- Request records：AIPerf `profile_export.jsonl`。

结构化分析数据：

- `runs/c144-138-136-20260922T071057Z/analysis/partial-run-data.json`
- `runs/c144-138-136-20260922T071057Z/analysis/runtime-summary.json`

对应分析脚本：

- `scripts/analyze_partial_run.py`
- `scripts/analyze_runtime.py`

---

## 3. 中止口径：不是性能故障，也不是真正的 resume

### 3.1 发生了什么

原 client/wrapper 在 `2026-09-22 08:50:49 UTC` 收到外部终止：

```text
Terminated
exit_code: unknown
```

中止前：

- Warmup 已完整结束；
- profiling 已运行约 1006 秒；
- Client 实时统计仍在推进；
- 无 AIPerf traceback、failed-request threshold 或 worker fatal；
- 四个服务容器仍健康，三个 health endpoint 均为 HTTP 200。

目录中没有 `bench-exit-code.txt`、`completed-at.txt` 和最终
`agentx_conc144.json`，证明 wrapper 没有走到正常收尾路径。终止来自运行 wrapper
的外部任务生命周期，不是 benchmark 自身判定失败。

### 3.2 原 client 是否被中止

是。AIPerf client 进程随 wrapper 被终止，未发现残留 `aiperf profile` 进程。
服务端没有重启或停止。

### 3.3 “resume” 的准确含义

后续启动的 `c144-resume-138-136-20260922T085229Z` 不是从 profiling 第 16 分钟
继续计时，也不会把两个窗口在 AIPerf 内拼接。

它实际执行：

1. 复用同一套仍存活的服务进程；
2. 保留首轮产生的 GPU/HiCache/Radix cache 状态；
3. 创建全新的 AgentX client；
4. 重新执行 warmup 10/lane；
5. 重新执行完整 3600 秒 profiling。

因此更准确的名字是 **replacement full run against a retained warm service**。

它适合：

- 补齐完整 3600 秒 profiling；
- 验证首轮机制是否在长窗口稳定存在；
- 获取最终 AgentX aggregate。

它不适合：

- 冒充首轮窗口的无缝 continuation；
- 与 fresh-launch C144 做严格 cache-state A/B；
- 与首轮 request percentile 直接合并。

报告会把两次 run 分开呈现。

---

## 4. 数据完整性与限制

### 4.1 完整部分

Warmup 完整完成：

- 目标/完成：`1598 / 1598`
- wall time：`4571.12 s`（76.19 分钟）
- exported warmup records：1598
- raw error records：39 个 `InvalidInferenceResultError`

Server、rank、rail 和 HiCache 采样覆盖整个 warmup。

### 4.2 Partial profiling

- 计划窗口：3600 秒
- 已保存窗口：约 `1002.33 s` request span
- phase wall-clock：`1006.37 s`
- 覆盖计划窗口约 `27.9%`
- 已完成 profiling records：1580
- 导出 profiling error/cancel record：0

“0 error”只适用于已完成并导出的 1580 条 profiling records。外部强杀时仍在途的
请求没有正常收尾，不能据此声称整个窗口零失败。

### 4.3 Metrics scrape gaps

Prometheus endpoint 在大 burst 或 scheduler 忙时偶发超过 1.5 秒 scrape timeout：

- Warmup：Prefill/Decode 各 82/53 次 timeout，约占 2255 次尝试的 3.6%/2.4%；
- partial profiling：14/12 次，约占 497 次尝试的 2.8%/2.4%；
- GPU/HCA sampler：0 error。

缺口被显式记录，没有被填零。剩余 97% 左右的逐 rank 采样足以判断持续队列、
capacity、graph coverage 和 rail 趋势，但不适合声称捕获了每个瞬时尖峰。

---

## 5. 时间线

### 5.1 启动

- 07:10:57：wrapper 启动；
- 07:12:27–07:13:05：Prefill/Decode 主模型权重加载，约 31–35 秒；
- 07:13:15：Decode target-verify graph capture 开始；
- 07:15:10：target-verify graph capture 完成，约 113–114 秒；
- 07:15 左右：服务 healthy；
- 07:17:52：AgentX warmup 开始。

### 5.2 Warmup

- 07:17:52：首批 158 trajectory credits；
- 07:37 左右：返回停在约 146–147，余下请求长时间不动；
- 07:49:35：第一条 `WaitingForInput` 1800 秒 timeout；
- 07:49–07:53：第一组 timeout 释放请求；
- 07:53 后：warmup 再次快速推进；
- 08:04：返回到 1450 后再次进入 plateau；
- 08:23–08:26：第二组 1800 秒 timeout；
- 08:34:03：warmup 1598/1598 完成。

这种“长平台 → 到 1800 秒整点 timeout → 突然继续”的形态直接证明 timeout 是
系统推进机制的一部分，而不只是无关日志。

### 5.3 Partial profiling

- 08:34:03：profiling 开始；
- 08:50:49：外部终止；
- 实际保存约 16.8 分钟。

---

## 6. Warmup：Prefill crawl 与 WaitingForInput timeout

### 6.1 29 个 timeout 分布在全部 rank

Decode 日志中的 1800 秒 `KVPoll.WaitingForInput` timeout：

| Decode rank | timeout 数 |
|---:|---:|
| 0 | 3 |
| 1 | 5 |
| 2 | 4 |
| 3 | 3 |
| 4 | 2 |
| 5 | 4 |
| 6 | 3 |
| 7 | 5 |

没有一个 rank 独占故障。它是全局负载/协议状态问题，而不是某张卡、某个 rank 或
某条 rail 永久坏掉。

### 6.2 Client raw error 与进度行口径不同

AIPerf progress 行一直显示 `errors=0`，但 raw JSONL 中有 39 个 warmup
`InvalidInferenceResultError`，内容均为：

> No responses with actual content were received from the server.

这两个值不能互相替代：

- phase runner 的 `errors=0` 表示 warmup barrier/control path 没有把它们计成
  phase-level error；
- raw record 的 39 条 error 才是 request-level 结果。

29 个 server timeout 与 39 个 invalid inference record 数量不等，说明至少还有
10 条空响应来自另一条路径，或者一个 server failure 与 client record 并非一一对应。
这需要 request ID/phase timestamp 关联，不能强行配对。

### 6.3 Warmup queue/capacity

跨 rank 聚合：

| 指标 | p50 | p90 | max |
|---|---:|---:|---:|
| Prefill running | 4 | 9 | 20 |
| Prefill queue | 0 | 70 | 146 |
| Prefill active token usage（8 rank 求和） | 0.12 | 1.11 | 1.78 |
| HiCache host used tokens | 31.24M | 37.71M | 37.71M |
| Decode running | 0 | 0 | 0 |

Warmup 使用 one-token output，不能拿 Decode running=0 推导正式 generation 性能；
但它能证明 1800 秒 timeout 发生时问题位于生成前的 bootstrap/KV handoff。

---

## 7. Partial profiling：请求级结果

### 7.1 Throughput

在约 1002.33 秒 request span 内：

- successful completed records：1580
- QPS：`1.576`
- input throughput：`133,993 token/s`
- output throughput：`1,526.8 token/s`
- total throughput：`135,519.8 token/s`
- per-chip total throughput：`8,470.0 token/s/chip`

这一值不能替代完整 3600 秒结果：窗口只有计划的 27.9%，并且终止时未完成请求被
截断。作为方向性对照，它低于历史完整 C144 的 12,555 token/s/chip。

### 7.2 Request-level latency

| 指标 | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| TTFT | 18.20 s | 79.56 s | 227.41 s | 315.76 s | 520.18 s |
| E2E | 32.82 s | 102.73 s | 245.79 s | 333.42 s | 555.11 s |
| Full-response ITL | 13.01 ms | 15.55 ms | 18.40 ms | 20.42 ms | 37.65 ms |
| ISL | 58,370 | 101,240 | 176,585 | 284,869 | 603,291 |
| OSL | 400 | 950 | 1,727 | 3,277 | 8,764 |

典型形态仍然是：

- TTFT 有数十到数百秒长尾；
- 已进入 generation 的请求 ITL 仍较好。

这与“首 token 前拥塞、Decode survivors 较快”的 phase-conditioned inversion
完全一致。

### 7.3 Effective concurrency

由每请求时间戳重建：

| 指标 | avg | p50 | p90 | max |
|---|---:|---:|---:|---:|
| Effective total concurrency | 131.1 | 169 | 193 | 206 |
| Effective Prefill concurrency | 110.8 | 150 | 177 | 184 |
| Effective Decode concurrency | 20.3 | 20 | 36 | 55 |

注意：

- C144 是 144 条 trajectory lane，不是 144 个 flat request；
- fan-out 使 Effective total max 达到 206；
- 三个 percentile 独立计算，不能严格相加。

但量级已经非常明确：典型墙钟状态下约 150 个请求在首 token 前，只有约 20 个在
generation。

---

## 8. Server queue 与 KV capacity

### 8.1 Prefill

跨 8 rank 求和：

| 指标 | p50 | p90 | max |
|---|---:|---:|---:|
| running | 9 | 12 | 18 |
| waiting queue | 141 | 159 | 179 |
| active token usage | 0.41 | 0.828 | 1.27 |
| active `kv_used_tokens` | 1.29M | 2.61M | 4.04M |
| cached `kv_evictable_tokens` | 23.85M | 24.49M | 25.13M |
| free `kv_available_tokens` | 16.2K | 19.8K | 24.2K |

Prefill GPU KV pool 总容量为 25.15M tokens（每 rank 3.14M）。
`token_usage`/`kv_used_tokens` 描述正在被 active request 使用、暂时不能驱逐的部分；
`kv_evictable_tokens` 描述仍驻留在 GPU RadixCache、但在容量压力下可以驱逐的 prefix。

可以把一个 rank 的 GPU KV pool 想成一间仓库，token slot 是货架位置：

- **used/active**：正在生产线上使用的箱子。当前请求还没结束，不能搬走；
- **evictable/cached**：生产已经结束，但箱子先留在快速仓库中，因为下一轮可能继续
  使用；有新货没位置时可以搬到 CPU 仓库或丢弃；
- **available/free**：真正空着、可以直接放新 KV 的货架。

一次 Prefill 请求大致经历：

```text
请求到达
  -> 在 GPU RadixCache 查找已有 prefix
  -> 命中的 prefix 在请求执行期间受到保护，属于 active working set
  -> 未命中的 prompt tail 需要新建 KV；优先占 free slot
  -> free slot 不足时，驱逐较旧的 evictable prefix
  -> 请求完成后，其 KV 不再 active，但通常转成 evictable prefix 留待下一轮复用
```

这里必须强调：**`kv_used_tokens` 不是“本轮需要重新计算的 token 数”**，而是
“当前被 active request 锁住、暂时不能驱逐的 GPU KV slots”。它同时包含：

1. **device-hit prefix**：KV 已经在 GPU，不需要重新计算，也不分配第二份；
   scheduler 只是给原有 Radix node 增加 lock reference。它在请求执行期间从
   “evictable”口径转为“used/protected”口径；
2. **host-hit prefix**：不需要模型重新计算，但必须先从 CPU 恢复到 GPU slot，
   然后同样被 active request 锁住；
3. **cache-miss / 新 prompt tail**：需要分配 GPU slot 并实际运行 Prefill 生成 KV；
4. chunked Prefill 中尚未完成、以及等待 handoff 释放的 active KV。

因此一个 prompt 即使 100% 命中 GPU prefix、几乎不做 Prefill compute，它在被调度
执行期间仍会让对应 slots 进入 `kv_used_tokens`。区别是：

```text
命中 prefix：复用并锁住已有 KV，不重新算
未命中 tail：分配并锁住新 slot，同时执行模型计算生成 KV
```

Live SGLang 的统计实现也直接使用：

```text
kv_used_tokens =
    max_total_num_tokens
    - (kv_available_tokens + kv_evictable_tokens)
```

即“既不是 free、也不能立即 eviction 的 slots”，而不是 FLOP 或 uncached-token
counter。真正判断 Prefill 实算量应看
`prefill_effective_tokens_total{mode="input"}`；device/host hit 则分别看
`mode="device_hit"` / `mode="host_hit"`。

例如某 rank 只有 100 个 slots，调度前是：

```text
free=20, evictable prefix=80, used=0
```

一个请求命中 60 个 GPU prefix tokens，另有 10 个 token miss：

```text
命中的 60：不重新计算、不复制第二份，但从 evictable 转为 locked/used
miss 的 10：从 free 分配新 slots，并实际执行 Prefill

调度期间约为：free=10, evictable=20, used=70
```

此时 `kv_used_tokens=70`，但真正需要模型计算的只有 10。请求结束并解除 lock 后，
这批 KV 若被缓存，通常又从 used 转回 evictable。

所以“当前生产线只用了 5%”与“仓库已经几乎没有空货架”可以同时成立。本实验就是
这种状态：同一时刻 active request 不多，但过去请求留下的可复用 prefix 占满了
其余 GPU slot。

`token_usage` 是每 rank active ratio 的和，因此除以 8 后：

- p50 平均每 rank 约 `5.1%`
- p90 约 `10.4%`
- max 同时状态约 `15.9%`

但 p50 时：

- active KV：约 1.29M（5.1%）
- evictable GPU prefix：约 23.85M（94.8%）
- free slot：仅约 16.2K（0.064%）

三个 p50 是对各自时间序列独立取中位数，并非强制来自完全同一个采样瞬间，因此
相加会有约千分之几的偏差；这里用于判断容量构成，不应当作逐 token 会计恒等式。

所以准确状态是：

> Prefill active batch 很小，但 GPU prefix cache 几乎填满；不是 device KV
> “很空”。大量历史 prefix 以 evictable RadixCache 的形式占据 GPU，新的 KV
> 写入会驱逐旧 prefix 到 host tier，或者在两层都无法保留时最终重算。

低 active usage 说明 Prefill 不是被“同时运行请求需要的不可驱逐 KV”占满；
接近零的 free slot 则证明 prefix working set 已经吃满 GPU cache。两者并不矛盾。

### 8.2 Decode

跨 8 rank 求和：

| 指标 | p50 | p90 | max |
|---|---:|---:|---:|
| running | 22 | 37 | 51 |
| transfer queue | 147 | 163 | 178 |
| token usage | 7.44 | 7.71 | 7.84 |

平均每 rank Decode KV usage：

- p50 约 `93.0%`
- p90 约 `96.4%`
- max 同时状态约 `98.0%`

这说明 profiling 阶段 Decode 不是“完全空闲”；它有约 20–37 个 generation
request，但同时有约 147–163 个 request 堆在 transfer queue。Decode KV capacity
已经成为 transfer admission 的直接约束。

### 8.3 P:D 配比的准确表述

数据不支持简单说“只需要加 Prefill”或“只需要加 Decode”：

- Prefill queue 极高，说明 Prefill 服务率不足以跟上到达；
- Decode KV 几乎满，说明即使 Prefill 完成，KV 也不能及时进入 Decode；
- Decode compute batch 只有 p50 22，远低于全局 256，说明 Decode 算力没有被
  请求池填满；
- 但 Decode KV capacity 与 transfer admission 已经饱和。

因此当前 P8D8 的问题是：

> Prefill 有效服务率不足 + Decode KV capacity 不足的耦合，而不是某一端 GPU 数
> 的单变量结论。

增加 Prefill 可能缩短排队并恢复 prefix locality，但也会更快把 KV 推向已经接近满载
的 Decode；若并发不降，最终还需要增加 Decode KV capacity 或改变 KV 生命周期。

---

## 9. HiCache：有效，但已满且不足以兑现理论 hit

### 9.1 容量

profiling 中：

- host total：37,724,672 tokens
- host used mean：37,709,937 tokens
- mean usage：约 `99.96%`

所有 rank 都接近自身 4,715,584 token host pool 上限，不是单 rank 热点。

同时 GPU Radix prefix cache 也接近满载：每 rank 约 3.0M token 处于
`kv_evictable_tokens`，free slot 通常只有 1K–3K。HiCache 的 write-through
策略会把 GPU 中可复用 prefix 写入 host tier；当下一轮请求到达时：

1. prefix 仍在 GPU：device hit；
2. 已从 GPU 驱逐、仍在 host：host hit，需要 CPU→GPU restore/promotion；
3. GPU 与 host 都已替换该 prefix：miss，需要重新 Prefill。

GPU 3.14M/rank 与 host 4.72M/rank 不能简单相加成 7.86M unique prefix：
write-through 会让两层保留部分重复副本。Host tier 扩大了可恢复窗口，但不是把
device active usage 按比例变成同等大小的独立唯一容量。

因此“active GPU KV usage 低”不意味着请求不该访问 CPU。Active KV 和缓存驻留
prefix 是两个不同集合；本实验中前者低、后者几乎占满整个 GPU pool。

### 9.2 Effective token 分解

profiling counter delta：

| 路径 | tokens | 占 effective input |
|---|---:|---:|
| device hit | 92,678,592 | 67.1% |
| host hit | 21,820,352 | 15.8% |
| miss / input compute | 23,524,587 | 17.0% |
| storage hit | 0 | 0% |

实际 overall hit 为 `82.96%`，与实时 server prefix hit 约 83% 相符。

理论 prefix hit 约 95.5%，理论 miss 约 4.5%；实际 miss 17.0%，即实际需要计算的
比例约为理论 miss 的 3.8 倍。

因此：

- HiCache 不是无效：它贡献了约 15.8% host hit；
- 但它不是无限容量：profiling 几乎全程打满；
- 它没有把实际 hit 恢复到理论 95.5%；
- 额外容量未转化为 C144 的健康运行，因为 Prefill/Decode queue 和 working-set
  churn 同时存在。

`hicache_dropped_tokens_total` 在两个已知 reason 上增量为 0，只能说明没有由这两个
counter 报告 drop；不能反证 eviction/locality loss 不存在。实际 hit 与 theoretical
hit 的 12.5pp 差距是更直接的结果指标。

### 9.3 Queue 与 Prefill 实算量是正反馈，不是单向因果

“Prefill 实算量增加是否才是 queue 增长的原因”不能简单回答为是或否。更准确的
排队模型是：

```text
utilization ρ ≈ request arrival rate λ × mean service time E[S]
```

C144 首先同时增加：

- live trajectory 数；
- subagent fan-out 后的 HTTP request 数；
- 同时驻留的 session/prefix working set；
- 相邻两轮之间插入的其他 session 数量（reuse distance）。

当到达率接近 Prefill 服务率时，queue 先开始增长。Queue 增长又延长同一 session
两次访问之间的墙钟时间和插入条目数，使 prefix 更可能从 GPU、再从 host 被替换。
Miss 增加后，单请求 Prefill service time 变长，进一步降低服务率并扩大 queue：

```text
并发/到达率上升
  -> queue 与 reuse distance 上升
  -> GPU/host prefix eviction 增加
  -> miss/recompute 与 host restore 增加
  -> E[S] 上升
  -> utilization ρ 进一步上升
  -> queue 更长
```

本次 profiling 中理论 miss 约 `4.5%`，实际 miss 约 `17.0%`。如果仅按需要执行
Prefill compute 的 token 比例估算，实际计算 token 比理想无限缓存口径约高
`17.0 / 4.5 = 3.8x`。这不是精确 wall-time 倍数，因为 host restore、chunking
和不同长度请求的成本不同，但足以说明 miss amplification 对服务率有一阶影响。

旧 P:D 配比实验提供了因果方向更强的干预证据：

- 1P1D C128：Prefill queue 约 95–120，actual hit 0.691，总吞吐 85,729 token/s；
- 2P1D C128：Prefill queue 降至约 35，actual hit 回到 0.937，总吞吐
  348,884 token/s；
- 两者 theoretical hit 均约 0.957。

增加 Prefill 服务能力同时缩短 queue、恢复 actual hit，并把吞吐提高 4.07 倍。
这支持“queue/reuse distance 导致 hit 下降”，而 hit 下降后的重算又继续放大 queue。
它们是正反馈环的两个方向，不能只选其中一个作为唯一原因。

---

## 10. `max_running_requests` 不是限制

配置：

```text
global max_running_requests = 256
attention DP = 8
per-rank request pool ceiling = 32
```

profiling 实测：

- Prefill running：总 max 18，单 rank max 9；
- Decode running：总 max 51，单 rank max 18。

没有 rank 达到 32，也没有总 running 接近 256。

大量请求位于 Prefill waiting queue 和 Decode transfer queue，而不是 running pool。
所以提高 `max_running_requests` 不会直接释放当前队列；它只会扩大 pool/graph/内存
预算，并可能进一步增加 Decode KV 压力。

---

## 11. CUDA graph 不是限制

### 11.1 启动 capture

Decode 八个 rank 均捕获：

```text
bs=[1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,24,26,28,30,32]
```

这正是 `256 / DP8 = 32` 的 per-rank request pool 范围。

### 11.2 profiling counter

- `decode_cuda_graph`: 121,099 passes
- `decode_none`: 0 passes
- counter-derived coverage：100%

各 rank 都有 graph pass 增量，最低 7,362、最高 22,519；差异来自分配到各 rank
的 workload 数量，而不是某 rank graph 缺失。

`sglang:is_cuda_graph` 瞬时 gauge 在采样中长期为 0，与 cumulative counter 冲突。
该 gauge 只反映最后一次被写入的瞬时状态，可能被 idle/prefill 状态覆盖；在这种冲突
下应以带 mode 的单调 counter 为准。

Prefill graph 按启动日志明确被禁用，因此 `prefill_none=5,973` 是配置预期，
不是 C144 跨过 graph boundary。

---

## 12. Rank 与 rail 不均衡

### 12.1 存在 workload skew，但不是单点故障

profiling 中：

- Prefill queue 较高 rank：2、7；
- Decode transfer queue 较高 rank：2、7；
- Decode running 较高 rank：4、7；
- 29 个 timeout 遍布全部 rank。

Rank affinity 确认开启：

- `PD_DP_RANK_AFFINITY=1`
- Router live env：`INFERA_PD_DP_RANK_AFFINITY=true`
- rank i → GPU i → ionic_i
- destination affinity on

因此 queue skew 是 workload/routing/key-locality 差异，而不是 affinity 忘记开启。
它可能放大局部尾延迟，但不能独立解释所有 rank 的系统性 timeout。

### 12.2 GPU prefix 驻留与 Prefill 实算量是否均衡

这需要看两组不同指标：

1. `kv_evictable_tokens`：每个 rank 当时保留了多少可驱逐 GPU prefix；
2. `prefill_effective_tokens_total{mode="input"}`：每个 rank 在 profiling 窗口
   实际需要计算的 cache-miss tokens。

#### GPU evictable prefix

每 rank 的 `kv_evictable_tokens` p50：

| rank | p50 tokens |
|---:|---:|
| 0 | 3.060M |
| 1 | 3.020M |
| 2 | 3.000M |
| 3 | 3.016M |
| 4 | 3.050M |
| 5 | 2.978M |
| 6 | 3.054M |
| 7 | 3.050M |

持续状态非常接近：

- p50 横跨 rank 的 CV：4.7%
- p90 CV：12.0%
- p50 max/min：1.16x
- p90 max/min：1.47x

个别 burst 中 max/min 曾短暂达到 3.47x，但不是持续状态。所有 rank 的 GPU
prefix pool 在典型时刻都接近满载，没有一个 rank 长期“特别空”或“特别满”。

#### 实际 cache-miss / Prefill compute tokens

partial profiling 的每 rank counter delta：

| rank | device hit | host hit | input/miss（实际计算） | miss rate |
|---:|---:|---:|---:|---:|
| 0 | 11.38M | 6.13M | 3.08M | 15.0% |
| 1 | 9.29M | 1.08M | 2.69M | 20.6% |
| 2 | 6.07M | 0.14M | 3.04M | 32.9% |
| 3 | 10.50M | 3.28M | 3.11M | 18.4% |
| 4 | 12.58M | 3.27M | 3.06M | 16.2% |
| 5 | 8.29M | 2.49M | 2.82M | 20.7% |
| 6 | 19.38M | 3.07M | 2.78M | 11.0% |
| 7 | 15.20M | 2.37M | 2.95M | 14.4% |

虽然每 rank 收到的 logical/effective input 与 cache-hit tier 差异很大，但真正需要
模型计算的 `input/miss` 很均衡：

- min：2.69M
- max：3.11M
- max/min：1.16x
- CV：5.1%

因此当前数据不支持“某一个 Prefill rank 承担了绝大多数实算，拖慢全局”的解释。
更准确的是：

- routing/cache locality 在 rank 间不同；
- rank 2 miss rate 高达 32.9%，rank 6 只有 11.0%；
- 但 rank 6 承接更多高命中 logical tokens，最终八个 rank 的 miss/compute token
  总量反而被摊得较均衡。

所以存在 cache locality 和 queue skew，但没有显著、持续的 Prefill compute
负载不均。系统问题仍以所有 rank 共同面对的 cache capacity、Prefill queue 和
Decode transfer/KV 反压为主。

### 12.3 Rail

partial profiling：

- Prefill aggregate RDMA TX：约 7.57 GB/s
- Decode aggregate RDMA RX：约 7.57 GB/s
- 单 rail：约 0.51–1.39 GB/s
- retry exhausted delta：0
- ACK timeout delta：0

吞吐存在约 2.7 倍 rank/rail workload skew，但最高 rail 仍远低于标称带宽；
高 queue rank 与最高流量 rail 也不是固定一一对应。因此网络硬带宽/硬错误不是当前
主因。

这些数值来自每个 ionic HCA 的单调硬件 counter：

```text
/sys/class/infiniband/ionic_N/ports/1/hw_counters/tx_rdma_ucast_bytes
/sys/class/infiniband/ionic_N/ports/1/hw_counters/rx_rdma_ucast_bytes
```

对 profiling 窗口做：

```text
average GB/s =
    (counter_at_end - counter_at_start)
    / elapsed_seconds
    / 1e9
```

Prefill TX 与 Decode RX 的总量和逐 rail 结果吻合，说明计数路径自洽。

为避免长窗口平均掩盖 burst，又对约 5 秒采样间隔做相邻 counter 差分：

- partial run：全 rail interval p99 `8.77 GB/s`，最大 `18.34 GB/s`
- full replacement run：p99 `12.03 GB/s`，最大 `19.31 GB/s`
- 单条 rail 标称 400 Gb/s，约 `50 GB/s`

即使最高 5 秒窗口也只有标称带宽约 38.6%。同时两个 profiling 窗口中：

- `tx_rdma_retx_bytes/packets` 增量均为 0
- `req_tx_retry_excd_err` 增量为 0
- `tx_rdma_ack_timeout` 增量为 0

所以目前没有持续带宽饱和、重传或硬 timeout 证据。0.51–1.39 GB/s 的差异主要是
rank 承接的 logical KV 流量不同；在完整 run 中单 rail 长窗口平均收敛到
1.01–1.54 GB/s，max/min 仅约 1.53x，不是固定坏 rail。

尚不能排除：

- transfer done signal 的软件长尾；
- bootstrap 生命周期竞态；
- 小于 5 秒采样分辨率的微突发；
- 小比例请求的控制面丢失；
- 没有进入硬错误 counter 的短时 stall。

---

## 13. 与历史 C144 的关系

历史 sweep-of-record C144：

- total throughput/chip：12,555 token/s/chip
- TTFT p50/p90：52.45 / 209.33 s
- ITL p50/p90：14.40 / 19.88 ms
- Effective total/prefill/decode p50：197 / 169 / 25
- overall server cache hit：85.20%

本次 partial profiling：

- total throughput/chip：8,470 token/s/chip
- TTFT p50/p90：18.20 / 227.41 s
- ITL p50/p90：13.01 / 18.40 ms
- Effective total/prefill/decode p50：169 / 150 / 20
- effective server cache hit：82.96%

方向高度一致：

- Prefill phase 占大多数 in-flight；
- Decode phase 较小；
- TTFT 长尾极重；
- ITL 相对正常；
- actual hit 明显低于 theoretical。

绝对吞吐和 TTFT p50 不能做严格复测比较，因为本次只有 1002 秒，且结束时对在途请求
做了外部截断；数据组成尚未稳定到完整 3600 秒窗口。

---

## 14. 根因分层

### 已被当前动态数据直接证实

1. Warmup 有 29 个跨全部 rank 的 1800 秒 `WaitingForInput` timeout。
2. profiling 中 Prefill queue 与 Decode transfer queue 同时大幅积压。
3. Decode KV 每 rank 典型使用率约 88%–96%，接近容量上限。
4. HiCache host pool 几乎 100% 使用。
5. actual prefix hit 约 83%，显著低于 theoretical 95.5%。
6. global/per-rank max-running pool 都未触顶。
7. Decode runtime passes 100% 走已捕获 graph，没有 eager fallback。
8. rail 无新增 retry exhausted/ACK timeout，吞吐未接近硬带宽。

### 高置信度机制

1. Prefill 排队破坏 prefix temporal locality，使 miss/recompute 比例上升。
2. 大量 request 完成/进行 Prefill 后，在 Decode KV admission/transfer queue 堆积。
3. Decode bootstrap timeout 从请求进入 waiting 状态开始计时；部分请求等不到 KV
   handoff 完成，1800 秒后失败。
4. Timeout 释放请求后系统继续推进，形成 crawl，而不是永久死锁。
5. Prefill 服务率、HiCache 容量和 Decode KV capacity 形成耦合正反馈。

### 尚未唯一判定

1. 单个 timeout request 的 1800 秒中，Prefill compute、HiCache read、KV copy 和
   done-signal wait 各占多少；
2. 39 个 InvalidInferenceResult 与 29 个 WaitingForInput timeout 的逐请求对应；
3. Decode prealloc queue 中等待 request/metadata/KV slot allocation 的时间，与
   已完成 allocation 后在 transfer queue 中等待 Prefill 输入、RDMA transfer 或
   done signal 的时间分别占多少；
4. Router/session key 分布为何使 rank 2/7 queue 较高；
5. 增加 Prefill、增加 Decode KV capacity、缩短/延后 bootstrap timer 三种干预的
   相对收益。

---

## 15. 当前工程判断

在没有进一步 A/B 前，不建议：

- 继续提高 `max_running_requests`；
- 扩大 CUDA graph；
- 把问题归因于单 NIC；
- 只增加 Prefill 而不观察 Decode KV；
- 用降低 ITL 证明系统更健康。

建议的下一层验证优先级：

1. 为每个 request 记录
   `decode bootstrap start → prefill start → prefill done → transfer submit →
   transfer done → decode running`；
2. 分别记录 Decode prealloc queue 的 allocation wait，以及 transfer queue 的
   input/transfer/done-signal wait；
3. 在 C144 做单变量：
   - 增加 Decode KV capacity/worker；
   - 增加 Prefill worker；
   - 修改 bootstrap timer 起点或 timeout；
4. 再回到 C112，验证同一机制是否已在 knee 附近出现，只是幅度更小。

---

## 16. Replacement run 状态

`c144-resume-138-136-20260922T085229Z` 已启动，当前：

- 使用原服务进程和已热 cache；
- 新 client；
- 重新执行 warmup 10/lane；
- 目标重新执行完整 3600 秒 profiling；
- 独立采样和独立输出目录；
- 不与首轮 request samples 合并。

完整 run 完成后，本报告将增加完整窗口结果，并检查本文的 queue/capacity/graph
结论是否在稳态下保持。
