# 阶段 F：decode DP rank 由 Infera 路由选（B0 / B1 设计）

2026-11-27。代码在 `Infera-router-dp`（分支 `llying/router-decode-dp-rank`，基于 AMD-AGI/Infera main `3528a242`）。本文没有新的测量数据。

## 1. 现在 vLLM DP rank 怎么注册、怎么指定

- 注册：`infera/engine/vllm/args.py` 对 `--data-parallel-size > 1` 的引擎登记 `dp_size`，内部均衡（一个进程管 4 个 rank）不登记 `dp_rank`。
  路由把这种 worker 视为"按 rank 复用"，`pool::expand_targets` 展开成 4 个目标，键为 `<worker>#dp0..3`。
- 指定：`dp::header_rank` 给 vLLM / sglang 发 `X-Data-Parallel-Rank`（ATOM 写进请求体 `data_parallel_rank`）。
  vLLM 读 `X-data-parallel-rank`（`vllm/entrypoints/generate/base/serving.py:211`，nightly `de7cf7f8`；HTTP 头不分大小写），指定后不走 vLLM 自己的 DP 均衡。
  PD 的 decode 腿（`open_decode`、串行路径 `serial_decode`、NATS `leg_payload`）都带这个头；recipe 分支上的 vLLM 串行模式（`INFERA_VLLM_MOONCAKE_SERIAL=1`）复用 `serial_decode`。
- `INFERA_VLLM_DP_ENGINE_LB=1`（#250）让 decode 不登记 DP 身份，路由只看到 1 个目标、不发头，由 vLLM（+ fork #7）选 rank。
- 所以 **B0 不需要新代码**：去掉 `INFERA_VLLM_DP_ENGINE_LB=1`，路由已经按 rank 展开、记账、发头。m10（不开 #250）就是这样跑的。

## 2. m13 为什么崩

- m13 = m10 配置 + 路由环境变量 `INFERA_DECODE_REQUEST_WEIGHT=100000`（crusoe README 1s，`results/m13/engine-config.env`）。
- 崩溃签名：decode 一个 rank 在 hipBLASLt GEMM 上 `HIPBLAS_STATUS_INTERNAL_ERROR` → `HSA_STATUS_ERROR_OUT_OF_RESOURCES`，之前有一段生成速度掉到 30–67 tok/s，
  同节点 `evict_process_worker` 计数 64 → 128。与 m9b（不同路由配置）签名相同，m12 卡死前也有同样的变慢；三次都在 decode 显存目标 0.95 + aiter AG/RS 时（U29）。
- 推断：与路由选 rank 无关，是 decode 显存余量不够；J3、f1/f2（0.93）没有再出现。未确认。
- 另一处疑点：m13 的路由环境里**没有** `INFERA_DECODE_INPUT_DEMAND`。在现在的 main 上，只设请求权重会在启动时报错；m13 用的旧源码（`baseline/Infera-vllm-glm52`）已不在，
  那时请求权重是否生效无法确认。如果没生效，m13 实际就是 m10 配置，更说明崩溃与路由无关。重跑时要同时给 `--decode-input-demand on`。

## 3. B0：路由现有的输入需求 + 请求权重

- 打分（每个 rank 一个目标）：`已登记的输入 token + 本请求输入 token + W × 该 rank 持有的请求数`，选最小。
  在选中时登记，decode 流结束（EOF、出错、客户端断开）时释放（`decode_demand::Reservation` 的 Drop）。所以"请求结束时释放"已经有了。
- 开法：decode 不设 `INFERA_VLLM_DP_ENGINE_LB`；路由 `--decode-input-demand on --decode-request-weight W`。fork #7 的 `W` 此时不生效（rank 已指定）。
- 新增单测：一个 `dp_size=4` 的 worker，8 个相同请求按 2/2/2/2 分到 4 个 rank，且每个都带 rank 头。
- 已知局限：
  - 选 rank 在 prefill 之前（串行模式也是），登记从选中开始算，prefill 期间就占着；
  - 只算输入，不算输出增长（B1 解决）；
  - decode KV 事件关着，decode 侧 overlap 恒为 0；decode 选择几乎只由需求决定。
  - kv-aware 的"连续零命中"告警会把 decode 的选择也算进去，prefill 有命中时会清零，只是日志。

## 4. B1：输出增长记账（`--decode-output-demand on`，默认关）

- 做法：decode 流式响应的每个块经过 `GuardedStream` / `GuardedBody`（HTTP 和 NATS 两条路径）时，`ActiveGuard::observe_output` 把块交给这次的 `Reservation`；
  `OutputMeter` 按 SSE 行解析（跨块的半行留到下一块），只取生成部分（`choices[].delta` / `choices[].text`，或 Responses 的 `*.delta` 事件），按 4 字节 / token 估算，
  增量加到该 rank 的已登记 token 上；释放时连同输出一起减掉。打分公式不变，只是"已登记的 token"现在 = 输入 + 已输出。
- 为什么估算而不是数事件：decode 用 `--stream-interval 30`，加上 MTP，一个事件含多个 token；vLLM 只在请求带 `continuous_usage_stats` 时才逐块给 token 数，路由不改客户端请求。
- 非流式响应不增长（结束时一次释放）。
- 关着时的开销：每块一次 `Option` 判断。开着时每个 SSE 事件一次 JSON 解析和一次账本加锁。
- 与 Dynamo 的差别：Dynamo 按块追加输出、并扣缓存命中；这里按估算 token 追加，不扣缓存命中。
- 缓存命中抵扣（未实现）：现在 recipe 的 decode 开了 prefix caching 但关了 KV 事件，路由看不到 decode 各 rank 的缓存，抵扣无从算起。要做需先开 decode KV 事件（见第 5 节）。
- 新增单测：SSE 解析跨块、只计生成部分；登记随输出增长、释放归零；未开时不增长；开了以后长输出的目标在下一次选择中被避开。

## 5. KV 事件与 DP rank

- 按 rank 复用的 vLLM worker：vLLM 每个 rank 的发布端口 = 配置端口 + rank（`offset_endpoint_port`），路由按 `base + r` 订阅 4 路（`kv_event.rs`）。
  问题：Infera 只为它预留 1 个端口（U7），`base+1..3` 可能和别的端口撞；KV 元数据的块大小登记成 16，实际 64（U7）。#196 修的是外部均衡（每 rank 一个进程）的端口偏移，与这里不同。
- 现在 recipe 的 decode `--no-enable-kv-events --kv-events off`，路由不订阅 decode，B0 / B1 都不依赖它，保持关。只有做 decode 缓存命中抵扣时才需要先修上面两处。

## 6. 要比的东西（同一镜像，C152，fast 调通后 full）

| 配置 | decode 环境 | 路由参数（在 kit 的 `ROUTER_ARGS` 基础上） |
|---|---|---|
| A（现状） | `INFERA_VLLM_DP_ENGINE_LB=1` + fork #7 `W=14` | 现有 |
| B0 | 不设 `INFERA_VLLM_DP_ENGINE_LB` | `--decode-request-weight W` |
| B1 | 同 B0 | `--decode-request-weight W --decode-output-demand on` |

- 每轮记录：最忙 / 平均请求数、上下文不均（`scripts/dp_imbalance.py`）、rank KV > 95% 的时间占比、吞吐、ITL、TTFT；路由 `/metrics` 的 `infera_router_decode_demand_tokens/requests{target}`。
- `W` 用 decode 日志拟合（步长对最忙 rank 的请求数和 token 数做最小二乘，同 #210 的方法）；B1 下"已登记 token"包含输出，拟合值要按 B1 的口径重做。
- B 比 A 差时依次看：rank 定得太早（prefill 前）、输出估算偏差、`W` 不合适。
