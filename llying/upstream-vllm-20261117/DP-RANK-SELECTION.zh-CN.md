# decode DP rank 由谁选、怎么选：现状、各家做法与待定事项

2026-11-27 记录。**结论：暂不改架构，也暂不决定上游怎么提；经过 e2e 验证后再定。** 在那之前，现有实现（Infera #250 + vLLM fork #7）保留不动。

## 1. 问题

PD 的 decode 用 DP（vLLM DP4、sglang DP attention）时，每一步都要等最慢的 rank。一个 rank 的步长同时随请求数和它持有的上下文（KV）增长，所以新请求分到哪个 rank，直接影响 ITL 和吞吐。

## 2. 我们现在的做法（vLLM recipe）

两层配合，缺一不可：

| 层 | 改动 | 作用 | 位置 | 实测（单次） |
|---|---|---|---|---|
| 谁选 | Infera #250，`INFERA_VLLM_DP_ENGINE_LB=1` | decode 注册成单个 worker，路由不再指定 rank，交给 vLLM | Infera，ready 等 review | a3：+1.1%（最忙 / 平均 1.28 → 1.13） |
| 怎么选 | vLLM fork #7，`--data-parallel-lb-kv-weight W` | vLLM 均衡器打分 = 等待 + 在跑 + `W × KV 用量`（真实用量） | `llying-001/vllm` draft，未上游 | a5：+1.2%（和 GEMM 表一起测，未单独对照） |

- fork #7 依赖 #250：路由一旦指定 rank，vLLM 不走自己的均衡器，`W` 那一项不生效。
- #250 可以单独用：vLLM 用原有规则（只看请求数）选 rank。
- `W` 依赖模型、kernel 和硬件：a3 拟合约 14；之后 kernel 变快，同一拟合里上下文项缩小约 5 倍，a43 试 `W = 5` 因撤队列作废，没有结论。

## 3. 各家做法对比（11-26 / 27 读源码：sglang main `8703754`、Dynamo main `32f91bc`）

| | 谁选 rank | KV 信息 | 请求数 | 备注 |
|---|---|---|---|---|
| Infera + sglang（现 sglang recipe） | 路由（`--decode-input-demand` + `--decode-request-weight`） | 路由按已登记的输入 token 估算，不算输出增长 | 路由记账，加权 | `--decode-request-weight` 照 Dynamo 做 |
| Dynamo + vLLM | 路由；rank 经请求传给 vLLM 的 `data_parallel_rank` | 路由记账：decode 块数（有追加输出块的接口）减去缓存重叠抵扣 | 路由记账，加权 | 引擎上报的真实 KV 用量只用于过载判断，不参与打分；vLLM 内部均衡时会打日志建议改用 hybrid / external |
| sglang 引擎 `--load-balance-method total_tokens` | 引擎 | 真实 KV 用量 + 排队 token | 只用于打破平局 | 路由一旦指定 rank 就被跳过；PD decode 默认 `round_robin` |
| 我们的 vLLM recipe | 引擎 | 真实 KV 用量 | 引擎实时值，与 KV 加权求和 | 见第 2 节 |

- Infera + sglang 与 Dynamo 大方向一致（都是路由选、路由记账、加请求数权重），差别是 Dynamo 计入输出增长并扣缓存命中。Dynamo 的输出追加只看到接口，**没有跟完调用链**，未确认在 decode 选择里实际生效。
- 两家路由方式都**不用真实 KV 用量打分**；用真实用量的只有 sglang 引擎内的 `total_tokens` 和我们的 fork #7。
- m10 的分析认为 Infera 路由选 rank 不均的原因是"不跟踪输出增长和请求结束"（最忙 rank 在跑数是平均的 1.28 倍，60% 的时间有 rank KV > 95%）。

## 4. 候选方向

| 方向 | 要改哪里 | 好处 | 风险 / 代价 |
|---|---|---|---|
| A. 保持现状：引擎选（#250 + fork #7） | Infera + vLLM | 有数据；请求到 decode 时才选，时机最晚 | 依赖 vLLM 上游合入 #7（InferenceX 不允许引擎补丁）；和 sglang 的控制方式不一致 |
| B. 统一成路由选（照 Dynamo，补输出增长和缓存命中抵扣） | 只改 Infera | 不依赖 vLLM 上游；sglang / vLLM 都由路由参数控制 | 没有数据（m13 唯一一次在 vLLM 上试路由请求数权重时 decode 崩溃作废）；路由在 prefill 前就定 rank，时机早；decode 要重新按 rank 展开，#250 不能用，要连带修 KV 事件端口（#196 和"只预留 1 个端口"） |
| B'. 路由选 + 引擎上报真实 KV 用量参与打分 | Infera（worker 上报 + 路由打分） | 同 B，且用真实用量 | 两家都没有先例；改动最大；上报有延迟 |
| C. 统一成引擎选（sglang 用 `total_tokens`） | sglang 只改配置；vLLM 同 A | sglang 侧改动小 | sglang 的 `total_tokens` 请求数只打破平局；没有测过 |

## 5. e2e 要回答的问题（决定前）

1. fork #7 在 vLLM main 上的单独效果：同一部署 `W = 0` 对 `W > 0` 成对比较，并用当前 kernel 重新拟合 `W`。
2. 路由选 rank 在 vLLM 上能到什么水平：至少重跑 m13（路由 `--decode-request-weight`），与 #250 + fork #7 成对比较；若要比 B，先在路由记账里加输出增长。
3. 可选：sglang 的 `--load-balance-method total_tokens` 与现有路由方式成对比较。
4. 记录每轮的最忙 / 平均请求数、上下文不均（`scripts/dp_imbalance.py`）、KV > 95% 的时间占比，不只看吞吐。

## 6. 相关
- vLLM fork #7：https://github.com/llying-001/vllm/pull/7 （正文列了上游相关 PR：#58070、#43279、#69910、#52464、#43483、#66962、已关闭的 #40795）
- Infera #250：https://github.com/AMD-AGI/Infera/pull/250 （正文待更新 a3 / a5 数据，并注明可配合 fork #7）
- Infera #196（DP rank 的 KV 事件端口）、U7"KV 事件端口只留 1 个"：走方向 B 时要一起解决。
