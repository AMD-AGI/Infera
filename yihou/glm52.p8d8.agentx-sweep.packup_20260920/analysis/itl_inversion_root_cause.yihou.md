# P8D8 concurrency sweep 中 ITL 反转的静态分析

## 1. 结论

这次 `conc=112 -> 144 -> 192 -> 256` 时 ITL 逐步降低，并不表示高并发下系统变快了。它是 **PD（prefill/decode 分离）系统进入饱和后，端到端拥塞位置与 ITL 统计区间发生分离** 的结果：

1. 请求大量堆积在首 token 之前的 prefill、缓存读取/重算、KV 传输及 decode 接入链路；
2. 真正拿到首 token、进入 generation 阶段的并发请求反而越来越少；
3. decode 侧的小 batch 降低了成功请求的每 token 间隔；
4. AgentX 汇总的 ITL 是成功请求的 request-level 中位数，并且定义中明确减掉 TTFT，所以它看不见首 token 前累积的数十到数百秒等待；
5. 高并发下成功样本还向较短请求偏移，进一步强化了 survivor/sample-composition bias。

最直接的证据是：

- `conc=112 -> 144 -> 192 -> 256`：
  - Full-Response ITL p50：`15.92 -> 14.40 -> 11.23 -> 10.18 ms`
  - Effective Decode Concurrency p50：`40 -> 25 -> 10 -> 7`
  - Effective Prefill Concurrency p50：`69 -> 169 -> 243 -> 305`
  - TTFT p50：`11.30 -> 52.45 -> 132.82 -> 159.51 s`
- `conc=256` 相比 `conc=80`：
  - ITL p50 下降 `31.6%`，但 TTFT p50 上升约 `28.0x`；
  - 每 chip 总吞吐从 `19,513.2` 降到 `5,328.9 token/s/chip`，下降 `72.7%`；
  - 成功完成请求从 `9,483` 降到 `3,659`，下降 `61.4%`；
  - Active Decode Throughput 从 `2,528.07` 降到 `966.39 token/s`；
  - `conc=256` 还出现 `487` 个 `InvalidInferenceResultError`，占全部记录的 `7.43%`。

因此，本次反转应解释为：

> 高并发把瓶颈推到了首 token 之前，decode 被“饿瘦”；少量进入 decode 的成功请求以小 batch 运行，所以它们的 ITL 更低。这个 ITL 是条件化在“成功且已经进入 decode”上的 survivor metric，不是系统整体变快的证据。

静态材料足以确认上述主因，不需要复现。现有材料不足以进一步唯一判定首 token 前瓶颈中，prefill 重算、GPU/CPU HiCache I/O、KV 网络传输、控制面等待或 DP rank 不均衡各占多少；若要完成这一层归因，需要另行获准后做动态复现和逐 rank 观测。

---



## 2. 分析范围与数据口径

本分析只使用 packup 中已有实验记录和运行该实验所对应的代码，没有启动服务、回放 trace 或执行动态 debug。

主要实验材料：

- `results/results.csv`
- `results/sweep_results.yihou.md`
- `results/c{080,112,144,192,256}/agentx_conc*.json`
- `results/c{080,112,144,192,256}/profile_export_aiperf.csv`
- `results/c{080,112,144,192,256}/server_metrics_export_aiperf.csv`
- `results/c080/stdout.txt`
- `results/c080/stderr.txt`
- `run-logs/p8d8_agentx_sweep.nohup.txt`
- `notes.md`
- `analysis/max_running_cap.yihou.md`
- packup 中的 `scripts/bench-harness/`

代码口径：

- InferenceX / AgentX 父仓 commit：
`918524ff94045b3f091115f1051c22a8588edf2b`
- AIPerf submodule commit：
`754356e9a39acc6cc6afb242d123bb57c3fb6f75`
- SGLang patchwork：
`sglang/.patchwork/sgl519/`

`results.csv` 的 `Median ITL` 实际取自 AgentX JSON 中的
`request_metrics.latency.full_response_itl.p50`。packup 的
`scripts/bench-harness/tools/collect_agentx.py` 明确读取
`latency["full_response_itl"]` 并写入 `Median ITL (s)`。

---



## 3. How to read 指标

本节说明本文使用的指标来自哪个统计总体、如何计算，以及数值升降应该怎样解读。最重要的前提是：本文同时使用了 **request-level 分布**、**时间加权 sweep-line 分布** 和 **server counter/gauge** 三类口径；它们的 p50/avg 不能互相当成同一种统计量。

### 3.1 先区分三类统计口径

| 口径 | 本文中的例子 | 样本/权重 | p50 的含义 |
|---|---|---|---|
| Request-level | TTFT、ITL、ISL、OSL | 每条成功 profiling 请求等权 | 50% 的请求不高于该值 |
| Time-weighted sweep-line | Effective Concurrency、Active Throughput | 每段墙钟时间按持续时长加权 | 50% 的墙钟时间不高于该值 |
| Server counter/gauge | cache hit、cache usage、prompt tokens | server metrics 的计数器增量或 gauge 汇总 | 由具体 server metric 决定，不一定有 request-level 含义 |

Request-level percentile 由 AgentX `stats_for()` 对每请求标量排序后做线性插值。它不是按 token 数加权：一个输出 100 token 的请求和一个输出 5,000 token 的请求，在 ITL p50 中都只贡献一个样本。

Time-weighted percentile 则先把并发或速率构造成阶梯函数。例如并发值 10 持续 100 秒、值 100 持续 1 秒，p50 会更接近 10，而不是把两个状态等权后取中值。

不同曲线的 p50 是各自独立计算的，因此：

```text
p50(Effective Prefill Conc) + p50(Effective Decode Conc)
```

不等于某个时刻的总并发，也不要求等于 `p50(Effective Conc)`。

### 3.2 单位与文件位置

| 指标 | AgentX JSON | AIPerf CSV | `results.csv` |
|---|---|---|---|
| TTFT/E2EL | 秒 | 毫秒 | TTFT 为秒 |
| ITL/Full-Response ITL | 秒 | 毫秒 | ITL 为毫秒 |
| Throughput | token/s | token/s | token/s 或 token/s/chip |
| Cache hit/usage | 0~1 fraction | 常显示为百分比 | theoretical cache hit 为 0~1 |

因此 JSON 中 `full_response_itl.p50=0.01018` 表示 `10.18 ms`，不是
`0.01018 ms`。本文表格已经统一转换到列名标出的单位。

主要读取路径：

- Headline TTFT：
  `request_metrics.latency.ttft.{p50,p90}`
- Headline ITL：
  `request_metrics.latency.full_response_itl.{p50,p90}`
- Logical throughput：
  `request_metrics.throughput.*`
- ISL/OSL：
  `request_metrics.tokens.*`
- 记录计数：
  `request_accounting.*`
- Server cache：
  `server_metrics.cache.*` 与 `server_metrics.kv_cache.*`
- Effective/Active 指标：
  `profile_export_aiperf.csv`

### 3.3 `CONC`、QPS 与 in-flight request

AgentX 的 `--concurrency=N` 表示 N 条持续存活的 trajectory-tree lane：

- 它不是每秒发送 N 个请求；
- 它不是 N 个 flat HTTP connection；
- 它也不是服务端 running batch size；
- 一个 trajectory 内的 subagent fan-out 可让瞬时 HTTP in-flight request 超过 N。

所以 `conc=256` 只能读成“256 条 agent session lane”，不能直接读成“decode
同时运行 256 个请求”。

`QPS` 是成功请求 completion timestamp 按 1 秒窗口计数得到的完成速率。粗略阅读时：

```text
QPS ≈ records_profiled / profiling duration
```

它回答“每秒成功完成多少请求”，而 CONC 回答“客户端维持多少条 trajectory
lane”；二者不是同一个负载旋钮。

### 3.4 延迟时间线

可以用以下时间点理解本文的延迟指标：

```text
t_credit -> t_start -> t_first_content -> t_last_content -> t_request_end
```

- `t_credit`：replay lane 获得发送 credit 的时间；
- `t_start`：请求真正开始；
- `t_first_content`：客户端收到第一个非空 content chunk；
- `t_last_content`：收到最后一个非空 content chunk；
- `t_request_end`：HTTP 请求明确完成。

对应公式：

```text
Credit-to-Start = t_start - t_credit
TTFT            = t_first_content - t_start
Request Latency = t_last_content - t_start
Decode Duration = t_last_content - t_first_content
Full Decode Duration
                = t_request_end - t_first_content
```

#### TTFT

TTFT 越小越好。它包含请求开始后到第一个 content chunk 之间的全部客户端可见时间，包括 server queue、prefix-cache 查找/读取、prefill 或重算、PD KV handoff、decode 接入和首 token 生成。

所以 TTFT 能说明“首 token 前很慢”，但不能仅凭一个 TTFT 数值把耗时唯一归因给 prefill compute 或 KV 网络。

#### E2EL / Request Latency

本文 JSON 中的 `e2el` 对应 `Request Latency`，终点是最后一个非空 content
chunk，不包含其后的 usage-only / `[DONE]` 等尾部完成开销。越小越好。

#### Credit-to-Start

公式为 `request_start_ns - credit_issued_ns`。它度量 lane 已获得发送资格后，在
client credit queue 中等到真正发起请求的时间。该值很小，说明分钟级 TTFT
不是由这一段 client-side credit wait 造成。

#### HTTP Blocked

公式为：

```text
connection_pool_wait_end - connection_pool_wait_start
```

它只度量等待空闲 TCP connection slot 的时间。值为 0 表示没有 connection
pool wait；不能据此证明整个 client/event loop 完全无负载，但可以排除连接池排队是主要延迟来源。

### 3.5 ITL、Full-Response ITL、TPOT 与 Interactivity

对单条成功流式请求，设 `OSL` 为 server 报告的 output token 数：

```text
ITL =
    (t_last_content - t_first_content)
    / (OSL - 1)

Full-Response ITL =
    (t_request_end - t_first_content)
    / (OSL - 1)
```

实验命令使用 `--use-server-token-count`，所以分母来自 server usage 中的
completion token count，而不是客户端重新 tokenize。

本文的 headline `Median ITL` 实际取 **Full-Response ITL p50**。它比普通 ITL
多包含最后 content 到请求明确结束的尾部开销；本次两者非常接近。

正确读法：

- 越小表示“已拿到首 token 且成功完成”的请求，平均每个后续 token 占用的 decode
  墙钟时间越短；
- 它不含 TTFT；
- 它是每请求先求平均、再对请求等权取 percentile；
- 它不是所有真实相邻 token 间隔的 percentile；
- 在 MTP 下多个 token 可由一个 verify step 成批产出，因此它更接近平均 TPOT，
  不等于 SSE chunk 间隔。

AgentX 聚合层的 `TPOT` 直接复用 ITL 样本列表，所以在本报告中 TPOT 与 ITL
不是两项独立证据。

`Interactivity` 也不是独立测量，而是同名 ITL 统计量的倒数：

```text
Interactivity p50 = 1 / ITL p50
```

单位近似为 `token/s/user`，越大越好。它会完整继承 ITL 的成功样本条件和
survivor bias；ITL 下降时 interactivity 必然上升。

### 3.6 普通 percentile 与 error-adjusted percentile

普通 AgentX headline latency 只使用：

```text
profiling phase AND error is null AND metric is finite/positive
```

的记录。

AIPerf 对带 `PERCENTILE_INCLUDES_FAILED_REQUESTS` flag 的 latency 另建
`error-adjusted` 分布：

```text
adjusted samples = successful samples + [∞] * error_count
```

因此：

- 普通 ITL 回答“成功请求有多快”；
- error-adjusted ITL 回答“把失败视作无限慢后，服务总体 percentile 如何”；
- 当失败比例低于 50% 时，adjusted p50 仍可能是有限值；
- adjusted p50 与普通 p50 接近，不代表失败不重要，只表示失败尚未越过中位数门槛。

### 3.7 Effective Concurrency

AIPerf 用请求时间戳构造精确 sweep line：

```text
Effective Concurrency:
    [t_start, t_request_end)

Effective Prefill Concurrency:
    [t_start, t_first_content)

Effective Decode Concurrency:
    [t_first_content, t_request_end)
```

每条处于区间内的请求贡献 1，然后对整段 profiling window 做时间加权统计。

所以：

- Effective Prefill Concurrency 高：很多有合法 phase boundary 的请求同时处于首 token 前；
- Effective Decode Concurrency 高：很多有合法 generation boundary 的请求同时处于 generation；
- p50=7：代表至少一半 profiling 墙钟时间里，该 phase 的并发不高于 7；
- 它不是 server scheduler 内部 `num_running_reqs`，而是由客户端请求时间线推导的 phase concurrency；
- sweep 首先按 profiling phase 选记录，不会统一预先删除全部 error record；
- 某条记录是否进入具体曲线取决于所需时间戳是否有效：有 start/end 的 error
  record 仍可能进入 Effective Concurrency；缺 first-content/TTFT 的记录不会进入
  Prefill/Decode phase curve。

本文正是用 “Prefill Concurrency 上升、Decode Concurrency 下降” 判断并发从
首 token 后迁移到了首 token 前。

### 3.8 Effective/Active Decode Throughput

AIPerf 有两条 decode throughput curve 构造路径：

1. 若 metrics backend 保留了可逐请求回放的 Inter-Chunk Latency（ICL），则优先按
   SSE chunk 时间边界构造速率；为保持总 token 数一致，会把每个请求的
   `OSL-1` 均摊到该请求的非零 ICL interval 上。
2. 若没有可回放的 per-record ICL，则使用 request-level fallback，假设 generation
   区间内均匀产出：

```text
per-request decode rate_i =
    (OSL_i - 1)
    / (t_request_end_i - t_first_content_i)

aggregate decode rate(t) =
    sum(rate_i for requests active at t)
```

两条路径都会得到 aggregate decode rate 的阶梯曲线，再做时间加权统计。

- `Effective Decode Throughput`：在整个 profiling window 上统计，decode idle
  时间按 0 计入；
- `Active Decode Throughput`：只在至少一个请求处于 decode 的时间段统计，idle
  时间剔除；
- `Active Decode Throughput Per User`：

```text
aggregate decode rate(t) / decode concurrency(t)
```

再对 active 时间段做时间加权。

这里的 “user” 实际是当时处于 generation 的 request，不是 AgentX trajectory
lane。该吞吐是根据请求首尾时间和 token 数推导的均匀速率，不是 GPU/server
逐时刻硬件计数器。

读法：

- aggregate Active Decode Throughput 越高，decode 阶段总体产出越高；
- Per User 越高，活跃 decode 请求平均分到的 token rate 越高；
- aggregate 下降而 Per User 上升，通常意味着同时 decode 的请求减少、小 batch
  中单请求更快，而不是系统总能力提升。

### 3.9 Logical Throughput 与 Token Throughput/Chip

AgentX 对成功 profiling 请求计算：

```text
duration =
    max(successful request end)
    - min(successful request start)

Input Throughput =
    sum(ISL) / duration

Output Throughput =
    sum(actual OSL) / duration

Total Throughput =
    (sum(ISL) + sum(actual OSL)) / duration

Token Throughput/Chip =
    Total Throughput / (num_prefill_gpu + num_decode_gpu)
```

本次 P8D8 的分母是 `8 + 8 = 16` 张 GPU。

这里的 input throughput 是 **logical prompt token throughput**：完整 ISL 都计入，
包括 cache hit 的 prompt token；它不是实际执行 prefill compute 的 token/s。
因此：

- Total Token Throughput/Chip 适合比较同一 workload 下系统完成了多少逻辑工作；
- 它不能直接当作 FLOP/s 或实际 prefill compute rate；
- 本 workload 的 ISL 很大，Total Throughput 主要由 input token 主导；
- decode 容量应同时看 Output Throughput 和 Active Decode Throughput。

### 3.10 Records、ISL 与 OSL

记录计数的关系：

```text
records_total:
    profile_export.jsonl 中全部记录

records_warmup_dropped:
    benchmark_phase != profiling

records_error_dropped:
    error 字段非空

records_profiled:
    profiling phase 且 error 为空

records_dropped_total =
    records_total - records_profiled
```

同一条记录可以既是 warmup 又带 error，所以：

```text
records_dropped_total
```

不一定等于 `records_warmup_dropped + records_error_dropped`。

Token 指标：

- `ISL`：Input Sequence Length，单请求输入 token 数；
- `actual OSL`：server 实际报告的输出 token 数；
- `expected OSL`：trace 元数据中的预期输出长度，用于 workload/结果校验，不进入
  ITL 分母；
- 表中的 mean/p50/p90 都是成功 profiling 请求的 request-level 分布。

比较不同并发点的 ITL 前，应同时看 ISL/OSL 分布是否变化；否则可能把成功样本
组成变化误认为纯系统性能变化。

### 3.11 Cache 指标

`Theoretical Prefix Cache Hit` 来自 trace loader 预先计算的 block metadata：

```text
theoretical hit rate =
    sum(theoretical reusable prefix blocks)
    / sum(theoretical total prefix blocks)
```

它假设无限容量、按 trace 结构判断“理论上可复用多少”，不包含真实容量、
eviction、CPU offload I/O 和运行时调度影响。

SGLang server cache hit 使用 server counter：

```text
GPU hit rate =
    device cached prompt tokens / prompt_total

CPU/HiCache hit rate =
    host cached prompt tokens / prompt_total

Overall hit rate =
    all cached prompt tokens / prompt_total
```

所以：

- theoretical 高、actual 低：workload 本来有复用机会，但实际容量/eviction/调度没有保住；
- GPU hit 降、CPU hit 升：更多命中从快层转移到 host HiCache；
- overall 也下降：不仅是层级迁移，还有更多 miss/recompute。

`GPU/CPU cache usage` 是 gauge/high-watermark 风格汇总，不是每个 rank 的平均驻留率。
接近 100% 说明至少观测到 cache 接近满载，但不能单独推出每个时刻、每个 rank
都处于 100%。

### 3.12 其它诊断项

- `Spec Accept`：本实验强制 `SGLANG_SIMULATE_ACC_LEN=3.61`，主要是配置/性能口径；
  它不提供 correctness 健康信息。
- `Rail faults before/after`：只统计指定 RDMA fault counter 是否增长；`0/0`
  能排除已计数的 rail fault，不能证明网络没有带宽饱和或长尾。
- `duration`：成功 profiling 请求从最早 start 到最晚 end 的时间跨度；各点约
  3,627~3,630 秒。
- `p50` 看典型请求/典型墙钟状态，`p90/p95` 看尾部；判断 saturation 时不能只看
  p50。

### 3.13 本报告建议的阅读顺序

判断系统是否饱和时，按以下顺序读：

1. `Token Throughput/Chip`、Output Throughput、`records_profiled` 是否停止增长或下降；
2. TTFT p50/p90 是否恶化，错误记录是否增加；
3. Effective Prefill/Decode Concurrency 判断拥塞在首 token 前还是后；
4. aggregate 与 per-user Active Decode Throughput 判断是总能力增长还是 batch 变小；
5. theoretical/actual cache hit 和 cache usage 判断 cache pressure；
6. 最后再读 ITL，并明确它只描述成功、已进入 generation 的请求。

对本次异常，不能使用 “ITL 更小，所以系统更快” 的读法；必须把它与
TTFT、throughput、completion count 和 phase concurrency 联合解释。

---

## 4. 异常曲线与同时发生的阶段迁移



### 4.1 ITL 与 decode 并发同步下降


| conc | Full-Response ITL p50 (ms) | ITL p90 (ms) | Effective Decode Conc p50 | Active Decode Tput avg (token/s) | Active Decode Tput/User avg |
| ---- | -------------------------- | ------------ | ------------------------- | -------------------------------- | --------------------------- |
| 80   | 14.89                      | 21.28        | 37                        | 2,528.07                         | 72.16                       |
| 112  | 15.92                      | 21.75        | 40                        | 2,502.30                         | 69.75                       |
| 144  | 14.40                      | 19.88        | 25                        | 1,817.49                         | 78.50                       |
| 192  | 11.23                      | 15.75        | 10                        | 1,171.04                         | 107.37                      |
| 256  | 10.18                      | 13.82        | 7                         | 966.39                           | 118.98                      |


数据来源：

- ITL：各 `agentx_conc*.json`
- Effective Decode Concurrency 与 Active Decode Throughput：
各 `profile_export_aiperf.csv`

这组数据呈现两个阶段：

1. `conc=80 -> 112`：decode p50 并发由 `37` 增至 `40`，ITL 由
  `14.89` 增至 `15.92 ms`，符合 batch 变大后单用户 token 间隔变差的正常趋势。
2. `conc=112 -> 256`：decode p50 并发由 `40` 降至 `7`，ITL 同步由
  `15.92` 降至 `10.18 ms`。这不是 decode 能力增强，而是进入 decode
   阶段的请求变少。

Active Decode Throughput/User 的均值从 `69.75` 上升到
`118.98 token/s/user`，而聚合 Active Decode Throughput 从
`2,502.30` 下降到 `966.39 token/s`。这正是“小 batch 中单请求更快、系统总产出更低”的形态。

ITL 的倒数也给出相同直觉：

- `15.92 ms/token` 约等于 `62.8 token/s/user`
- `10.18 ms/token` 约等于 `98.2 token/s/user`

它们和 Active Decode Throughput/User 的上升方向一致。

### 4.2 总并发转移到首 token 之前


| conc | Effective Conc p50 | Effective Prefill Conc p50 | Effective Decode Conc p50 | TTFT p50 (s) | TTFT p90 (s) |
| ---- | ------------------ | -------------------------- | ------------------------- | ------------ | ------------ |
| 80   | 67                 | 27                         | 37                        | 5.69         | 22.12        |
| 112  | 114                | 69                         | 40                        | 11.30        | 64.47        |
| 144  | 197                | 169                        | 25                        | 52.45        | 209.33       |
| 192  | 267                | 243                        | 10                        | 132.82       | 443.83       |
| 256  | 357                | 305                        | 7                         | 159.51       | 510.12       |


注意：这些列分别是各自时间加权分布的 p50，不能把三列 p50 当作同一时刻的精确加减分解。但趋势非常清楚：

- 总体 in-flight 请求继续增长；
- 首 token 前的 prefill-phase 并发暴涨；
- generation-phase 并发从 `conc=112` 开始反向收缩；
- TTFT 的中位数和尾延迟同时恶化。

也就是说，高并发没有把更多请求持续送进 decode，而是把更多请求积压在首 token 之前。

---



## 5. 为什么 ITL 看不到这次拥塞



### 5.1 AIPerf 的 ITL 明确排除了 TTFT

AIPerf commit `754356e` 中
`src/aiperf/metrics/types/inter_token_latency_metric.py` 定义：

```text
Inter Token Latency =
    (Request Latency - Time to First Token)
    / (Output Sequence Length - 1)
```

Full-Response ITL 定义为：

```text
Full-Response ITL =
    Full Decode Duration
    / (Output Sequence Length - 1)
```

两者都只描述首 token 之后的平均 token 间隔。请求在 prefill、prefix cache
miss、HiCache 读取、KV transfer 或 decode 接入之前等多久，主要进入 TTFT，
不会进入 ITL 的分子。

因此在 PD 系统中完全可能同时出现：

- TTFT 从几十秒恶化到几百秒；
- 到达 decode 的并发越来越低；
- 成功请求的 ITL 反而下降。

这不是公式计算错误，而是指标条件域与系统瓶颈位置不同。

### 5.2 Effective Decode Concurrency 的代码含义

AIPerf 的 `src/aiperf/analysis/sweepline.py` 使用：

```text
[generation_start_ns, request_end_ns)
```

构建 generation concurrency sweep line，再对整段 profiling window 计算时间加权统计。也就是说，Effective Decode Concurrency 统计的是已经拿到首 token、仍处于 generation 阶段的请求，不包含在首 token 前等待的请求。

同一文件的 Active Decode Throughput/User 又用聚合 decode throughput 除以 generation concurrency。实验中 decode concurrency 下降、单用户 decode throughput 上升、聚合 decode throughput 下降，三者在定义和观测上相互一致。

### 5.3 PD 队列结构允许这种阶段解耦

SGLang patchwork 的
`srt/disaggregation/decode.py` 描述了 decode 侧从 pre-allocation、transfer、
waiting 到 running batch 的流水线。`staging_handler.py` 和 `conn.py` 中还会使用
`SGLANG_DISAGGREGATION_WAITING_TIMEOUT` 等待传输完成；本次
`scripts/bench-harness/engine.sh` 将该值配置为 `1800 s`。

这说明“请求已经进入系统”和“请求已经进入 decode running batch”不是同一个状态。上游 prefill/cache/KV handoff 变慢时，decode 可以同时表现为：

- running batch 较小；
- 已经进入 generation 的请求 ITL 较好；
- 系统外部观察到的 TTFT、完成量和吞吐很差。

`results/sweep_results.yihou.md` 还保留了 `conc=144` warmup 卡住时的直接诊断：

- prefill 仍在工作，多个 rank 有队列；
- decode 八个 rank 的 `num_running_reqs` 都为 `0`；
- 请求在 `KVPoll.WaitingForInput` 等满 `1800 s` 后触发 `KVTransferError`；
- 节点健康检查为 200，没有 rank abort，RDMA rail fault 为 0。

因此，“prefill 忙、decode 饥饿”在 saturation cliff 的 `conc=144` 上有直接日志证据，不只是根据 ITL 曲线推断。局限是：packup 没有保存 `conc=192/256` 正式 profiling 阶段同等级的 decode `num_running_reqs`、batch size 和 queue depth 时间序列。后两点的 decode starvation 主要由 Effective Decode Concurrency、TTFT、吞吐和完成量的共同趋势支持。

现有 packup 也没有导出足以逐请求分解 KV transfer wait 的原始指标，所以不能把首 token 前的全部时间都归因于网络传输。

---



## 6. 首 token 前压力的直接证据



### 6.1 Prefix cache / HiCache 命中率持续坍塌


| conc | GPU cache read hit | CPU cache read hit | Overall server cache hit | Theoretical prefix hit |
| ---- | ------------------ | ------------------ | ------------------------ | ---------------------- |
| 80   | 94.25%             | 0.56%              | 94.81%                   | 96.45%                 |
| 112  | 89.23%             | 2.39%              | 91.62%                   | 96.45%                 |
| 144  | 73.08%             | 12.12%             | 85.20%                   | 95.85%                 |
| 192  | 45.42%             | 28.70%             | 74.12%                   | 95.53%                 |
| 256  | 34.41%             | 31.45%             | 65.85%                   | 94.76%                 |


数据来自各 `agentx_conc*.json` 的 `server_metrics` 和
`request_metrics.cache_hit.theoretical_prefix_hit_rate`。

关键点不是 trace 理论可复用性发生了同等幅度变化：理论 prefix hit 只从
`96.45%` 小幅降至 `94.76%`；实际 server cache hit 却从 `94.81%`
降至 `65.85%`。同时：

- GPU cache usage 几乎一直在 `99%~100%`；
- CPU cache usage 为 `100%`；
- 命中从 GPU 大量转移到 CPU，最终总命中也明显下降。

这表明并发增加扩大 working set 后，GPU cache 容量、CPU HiCache 和 miss
重算路径承受了明显压力。更多 prompt token 需要慢层读取或重新 prefill，
与 Effective Prefill Concurrency 和 TTFT 暴涨相吻合。

cache collapse 是“为什么请求堵在首 token 前”的重要证据，但仅凭现有汇总不能把 cache read、重算与 KV 网络传输的耗时比例分开。

此外五个点在同一 deployment 中顺序执行，cache 状态并非每点独立冷启动；因此命中率是与并发共同变化的观测量，不是经过独立控制的单变量。它与容量压力的方向一致，但不能单独作为因果归因。

### 6.2 系统吞吐和完成量证明整体没有变快


| conc | Total Token Tput/Chip | Successful requests | Input Tput (token/s) | Output Tput (token/s) |
| ---- | --------------------- | ------------------- | -------------------- | --------------------- |
| 80   | 19,513.23             | 9,483               | 309,689.01           | 2,522.60              |
| 112  | 19,013.53             | 9,536               | 301,715.78           | 2,500.76              |
| 144  | 12,554.95             | 6,862               | 199,063.89           | 1,815.25              |
| 192  | 7,409.20              | 4,339               | 117,401.67           | 1,145.51              |
| 256  | 5,328.88              | 3,659               | 84,379.95            | 882.06                |


从 `conc=112` 开始，ITL 越来越“好”，但：

- 每 chip 总吞吐持续下降；
- 成功完成请求数持续下降；
- 输入和输出吞吐都持续下降；
- TTFT 持续上升。

因此 ITL 曲线与系统容量曲线已经脱钩。用于寻找 saturation knee 时，应相信吞吐、TTFT、完成量及错误率，而不能把成功请求 ITL 单独当作容量信号。

---



## 7. 成功样本选择进一步强化了反转



### 7.1 AgentX 汇总只保留 profiling phase 的无错误记录

InferenceX commit `918524ff` 的
`utils/agentic/aggregation/request_metrics.py` 中，
`load_records_with_accounting()` 会：

1. 丢弃非 profiling phase 的 warmup 记录；
2. 丢弃带 `error` 的记录；
3. 只把剩余记录传给 `compute_latency_stats()`；
4. 对每条成功记录的 Full-Response ITL 做等权 percentile。

所以 `results.csv` 中的 Median ITL 是：

> 在 profiling phase 内、成功完成且有合法 ITL 的请求集合上，计算 request-level p50。

它不是：

- 所有提交请求的端到端 token latency；
- 按 token 数加权的总体平均；
- 包含失败请求的 service-level latency；
- 首 token 前和首 token 后的统一延迟。



### 7.2 高并发下成功样本的输入显著变短


| conc | Profiled successes | Mean ISL | ISL p50 | Mean actual OSL | Actual OSL p50 |
| ---- | ------------------ | -------- | ------- | --------------- | -------------- |
| 80   | 9,483              | 118,523  | 74,210  | 965.44          | 447            |
| 112  | 9,536              | 114,842  | 80,497  | 951.86          | 423            |
| 144  | 6,862              | 105,248  | 65,436  | 959.75          | 422            |
| 192  | 4,339              | 98,206   | 62,545  | 958.22          | 420            |
| 256  | 3,659              | 83,649   | 55,774  | 874.42          | 405            |


`conc=256` 的成功样本相对 `conc=80`：

- Mean ISL 低约 `29.4%`；
- ISL p50 低约 `24.8%`；
- Actual OSL p50 只低约 `9.4%`；
- 成功请求数低 `61.4%`。

失败样本也不是随机子集。在 AIPerf 能统计出 input length 的失败记录中：

- `conc=192`：失败子集 `n=18`，Mean ISL `492,627`；成功样本 Mean ISL `98,206`；
- `conc=256`：失败子集 `n=51`，Mean ISL `396,554`；成功样本 Mean ISL `83,649`。

也就是说，能观察到长度的失败子集明显偏向超长 prompt。固定时长运行加上失败过滤，会使成功集合偏向输入更短、能进入 generation 的请求，形成 sample-composition bias。

不过 actual OSL 在 `conc=80~192` 基本稳定，`conc=256` 也只下降约 9%，所以“输出变短”不能主要解释 32% 的 ITL 改善。主因仍是 generation 阶段并发收缩；样本选择是放大因素。

### 7.3 失败过滤不是 p50 反转的充分解释

失败记录随并发显著增加：


| conc | records_error_dropped | errors / records_total |
| ---- | --------------------- | ---------------------- |
| 80   | 3                     | 0.03%                  |
| 112  | 5                     | 0.05%                  |
| 144  | 32                    | 0.38%                  |
| 192  | 185                   | 2.85%                  |
| 256  | 487                   | 7.43%                  |


错误类别均为 `InvalidInferenceResultError`。现有 packup 没有保存足够的 per-request error payload 来区分空 content、输出校验或其它触发路径；运行记录也明确把低并发点的同类错误列为尚未调查的独立故障路径。因此这些错误证明 `conc=256` 不是健康工作点，但不能静态断言它们全部由基础设施过载直接造成。

`records_warmup_dropped` 与 `records_error_dropped` 是分别计数的，记录可以同时属于两类，因此上表比例只表示带 error 标记的记录占全部记录的比例，不能直接等同于正式 profiling 请求失败率。

AIPerf 对带 `PERCENTILE_INCLUDES_FAILED_REQUESTS` flag 的 ITL 另外生成
error-adjusted 分布。`derived_latency.py` 的实现把失败记录作为 `+inf`
追加到成功样本，并生成独立的 `adj_*` 指标；它不会改写普通 success-only ITL。

在 `conc=256`：

- 普通 Full-Response ITL p50：`10.18 ms`
- error-adjusted Full-Response ITL p50：`10.21 ms`
- 普通 p90：`13.82 ms`（AIPerf CSV 口径）
- error-adjusted p90：`14.09 ms`

因此，即使把失败请求按最差延迟加入，p50 仍然很低。失败过滤会造成服务质量口径偏乐观，但不能单独解释从 `15.92` 到 `10.18 ms` 的反转；真正改变成功请求 ITL 的仍是进入 decode 的并发显著减少。

### 7.4 warmup dropped 数量不是根因

`config.full.sh` 配置
`AGENTX_WARMUP_REQUESTS_PER_LANE=10`。因此并发越高，warmup 记录本来就越多：

- `conc=80`：`884`
- `conc=256`：`2,845`

这个增长大体符合“每 lane 固定 warmup 数”的设计，不能把它当成高并发故障证据。warmup 记录不会进入 profiling percentile；各点正式 profiling duration 都约为 `3,627~3,630 s`，所以 ITL 反转也不是测试时长不同造成的。

---



## 8. 配置边界如何影响现象



### 8.1 `max-running-requests=256` 是全局预算

`analysis/max_running_cap.yihou.md` 已结合 SGLang 代码确认：

- 配置值 `256` 会按 DP size `8` 划分；
- 每个 DP rank 的上限为 `32`；
- `conc=80/112/144/192/256` 的简单 lane/DP 比值为
`10/14/18/24/32`。

但 AgentX 的 `--concurrency=N` 表示 N 条存活的 trajectory-tree lane，不是 N
个 flat HTTP 请求，也不是服务端 running batch size。subagent fan-out 还可使瞬时
HTTP 请求数高于 N。因此 `conc=256` 与 `max-running-requests=256` 数值相等，
并不能证明 scheduler 恰好只在该点触顶；简单 lane/DP 比值超过 CUDA graph max
batch size `16/rank` 也不等于实际 running batch 超过 16。

这些 scheduler/CUDA graph 边界可能增加 queueing、eager fallback 或容量压力，
使首 token 前拥塞更陡，但现有数据不能定量分离其贡献。

但它们不能被解释为“decode 在高并发下更高效”：

- `conc=256` 的 Effective Decode Concurrency p50 只有全局 `7`；
- 该值远低于 decode 的 global running cap；
- 低 ITL 发生时，decode 的有效 batch 实际更小，而不是更满。

因此 cap 最多是形成上游排队和阶段迁移的候选放大因素，不是 ITL 改善本身，也不能由客户端 CONC 数值直接判定已命中。

### 8.2 simulated acceptance 不是曲线反转的变量

本实验设置 `SGLANG_SIMULATE_ACC_LEN=3.61`，各点 Spec Accept 约为
`3.603~3.606`，基本恒定。它意味着本实验是在模拟固定 speculative
acceptance 的性能口径下进行，不是 correctness run。

固定 acceptance 会影响 ITL 的绝对值，但因为各并发点没有系统性变化，不能解释
`conc=112 -> 256` 的相对反转。

AIPerf 的分子是首个 content chunk 到最后 content/request completion 的墙钟区间，
分母是 server 报告的 output token 数。在 MTP 下多个 token 可由一个 verify step
产出，因此这里的 ITL 是“完整 decode 时间 / token 数”的平均 TPOT，不等于每个
SSE chunk 之间的实际间隔。缺少原始 per-chunk 联合分布，无法判断 chunk packing
是否还有额外影响；固定的 simulated acceptance 只能说明它不是当前反转的已证实变量。

---



## 9. 已排除或不受支持的替代解释



### 9.1 不是客户端 HTTP connection pool 阻塞

各点 AIPerf `HTTP Blocked` 均为 `0 ms`。Credit-to-Start Latency p50：

- `conc=80`：`0.69 ms`
- `conc=256`：`0.96 ms`

虽然 `conc=256` 的尾部有增加，但量级远小于 `159.51 s` 的 TTFT p50。因此主要等待发生在请求启动之后，而不是客户端拿不到 HTTP connection。

### 9.2 不是高并发测试时间更短

五个点 duration 都约一小时，没有足以制造该单调 ITL 趋势的时长差异。

### 9.3 不是 decode 总吞吐提高

Active Decode Throughput 从 `conc=112` 的 `2,502.30` 降至
`conc=256` 的 `966.39 token/s`。只有 per-user decode speed 上升，总产出并未上升。

### 9.4 不能仅凭现有材料断言是 KV 网络瓶颈

现象与 decode 等 KV、decode starvation 一致，但本次没有可用的逐请求
KV transfer latency、各 DP rank queue depth、网络吞吐/拥塞和
`WaitingForInput` 驻留时间。因此“上游 prefill/cache/KV handoff 是瓶颈域”
有充分证据，“其中网络是唯一根因”则没有。

### 9.5 不是 OOM、rank crash 或已记录的 rail fault

最终五个点的 rail fault 均为 `0/0`；`conc=144` 卡住时节点健康检查仍为
200、无 rank abort，GPU 也没有 OOM。它们不能排除网络带宽或长尾停顿，但排除了
“ITL 反转是由显式 rank crash/OOM/rail fault 计数触发”的解释。

---



## 10. 根因分层与置信度



### 已被现有记录和代码直接证实

1. `results.csv` 的 ITL 是成功记录的 Full-Response ITL request-level p50。
2. ITL 公式排除了 TTFT，只覆盖首 token 后。
3. `conc=112` 以后 Effective Decode Concurrency 与 ITL 同步下降。
4. 同时 Effective Prefill Concurrency 和 TTFT 大幅上升。
5. decode 聚合吞吐、每 chip 总吞吐和成功完成量均显著下降。
6. prefix cache 实际命中率坍塌，而理论 trace prefix hit 基本不变。
7. `conc=256` 有大量错误，成功样本的 ISL 显著偏短、OSL 略短。
8. `conc=144` warmup 有 “prefill busy / decode 全 rank idle” 的直接日志。



### 高置信度推断

1. 饱和瓶颈已迁移到首 token 前的 prefill/cache/KV handoff 域。
2. `conc=192/256` 的 decode 因到达不足而形成小 batch，导致成功请求的 per-user token rate 提升、ITL 下降；这两点缺少像 c144 warmup 那样的逐 rank profile 日志，但多项汇总指标一致支持该判断。
3. 成功请求过滤和 workload composition shift 进一步让 headline ITL 偏向 survivors。



### 必须动态复现/调试才能区分

1. prefill 重算、GPU->CPU HiCache I/O 与 KV transfer 各自的时间占比；
2. 是否有特定 DP rank、节点或 NIC 成为热点；
3. decode `WaitingQueue/WaitingForInput` 的驻留分布；
4. `InvalidInferenceResultError` 的具体触发路径及其与过载的关系；
5. `max-running-requests`、CUDA graph boundary 对 TTFT 拐点的定量贡献。

本次遵照要求没有执行这些动态操作。

---



## 11. 指标使用建议



### 11.1 saturation knee 不应由 ITL 单独判断

P8D8 这组数据的合理容量结论是：

- 已测点中每 chip throughput 峰值在 `conc=80`；由于没有更低并发点，真实峰值只能确定为 `conc<=80` 或接近 80；
- `conc=112` 比 c80 低 `2.6%`，已出现 TTFT 明显恶化；
- `conc=144` 已经越过健康 knee，吞吐下降且 TTFT 尾部明显恶化；
- `conc=192/256` 是严重首 token 前拥塞区；
- `conc=256` 的低 ITL 不应被用于证明高并发配置更优。

选择工作点时至少联合观察：

1. Total Token Throughput/Chip；
2. TTFT p50/p90；
3. successful requests / completion rate；
4. failed rate；
5. Effective Prefill/Decode Concurrency；
6. cache hit 与 cache usage；
7. ITL p50/p90。



### 11.2 报告 ITL 时增加条件说明

建议把 headline 改写成类似：

> Full-Response ITL p50 over successful profiling requests,
> conditioned on reaching generation.

并同时展示：

- success-only ITL；
- error-adjusted ITL；
- Effective Decode Concurrency；
- TTFT；
- completion/error count；
- 成功样本的 ISL/OSL 分布。



### 11.3 若获准做最小动态验证

若后续需要把首 token 前瓶颈继续拆分，最小验证不必重跑全部 sweep，可优先选择
`conc=112/192/256`，采集：

- 每 rank prefill/running/waiting queue depth；
- decode `WaitingForInput` 驻留时间；
- KV transfer submit/start/done latency；
- 每 rank GPU/CPU cache read、promotion、eviction、miss/recompute；
- NIC bytes、bandwidth 和 retry/stall；
- scheduler running batch size 与 CUDA graph/eager 命中；
- 每个成功/失败请求的 phase timestamps。

在获得明确许可前不应执行该复现。

---



## 12. 最终判断

这次 ITL 反转不是 benchmark 算错，也不是 `conc=256` 的系统性能好于
`conc=80`。它是一个典型的 **phase-conditioned metric inversion**：

```text
并发上升
  -> prefix working set 超出快层缓存能力，首 token 前积压加重
  -> prefill/cache/KV handoff 中的请求增多
  -> 能进入 generation 的同时请求数减少
  -> decode batch 变小
  -> survivors 的 per-token interval 下降
  -> success-only ITL 看起来更好
```

与 ITL 下降同时发生的吞吐坍塌、TTFT 暴涨、完成量下降和错误增加，才是系统真实进入严重饱和的信号。