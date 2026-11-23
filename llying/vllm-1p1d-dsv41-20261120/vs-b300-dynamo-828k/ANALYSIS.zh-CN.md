# DeepSeek-V4.1-Flash：MI355X Infera+vLLM 1P1D 基线 vs InferenceX B300 Dynamo-vLLM 828.2k 配置对比

日期 2026-11-23。

- **基线**：本实验目录的 c768 点（`../raw/pd1p1d-c768-hyper-1121T1745/`），459,183 tok/s/GPU，实验与分析报告见 `../EXPERIMENT.zh-CN.md`、`../ANALYSIS.zh-CN.md`；部署代码在 `llying/example/vllm_1p1d_dsv41_flash` 的 `examples/vllm_1p1d_dsv41_flash/`。
- **对照**：InferenceX B300 dynamo-vllm 的最高点，828,209 tok/s/GPU。本目录的 `inferencex_dsv41flash_b300_dynamo_vllm.json` 是公开接口返回的全部 B300 dynamo-vllm 行（2026-11-23 抓取）。

第 3.2 节和第 7 节里的通信占比与 decode 上限是根据日志和计数器推出来的估算，前提写在原处，需要用 profiler 证实（第 9 节第 1 步）。

---

## 0. 结论

1. **828k 和我们不是同一规模的点。** 它是 5P1D：5 个 DEP2 prefill worker（10 卡）加 1 个 DEP8 decode worker（8 卡），共 18 卡、3 个节点，并发 3584；我们是 2+2 卡，并发 768。1.80 倍的差距可以拆成两部分：每卡请求速率高 1.66 倍，每个请求的 token 数多 1.09 倍（c3584 时 trace 的平均输入更长）。
2. **差距不来自缓存命中、KV 传输或 router。** 把 InferenceX 混进去的 D 侧口径扣掉以后，B300 prefill 的重算比例约 4.7%，我们是 4.2%，两边接近。我们的 Mooncake 传输 p90 约 40 ms，router 加引擎前端约 20 ms，都不是瓶颈。
3. **差距来自每卡算得快。** prefill 每卡每秒新算的 token 数，B300 是我们的 1.79 倍（约 69k vs 38.7k）；decode 每卡输出是我们的 2.0 倍（13.9k vs 6.9k）。原因分三层：
   - **规模与拓扑**（2+2 里复制不了）：DEP8 decode 每卡只放 1/8 的专家，每步要读的专家权重和留给 KV 的空间都比 2 路切分好得多；prefill worker 随并发从 2 个加到 5 个。
   - **互联**：MI355X 任意两卡之间只有一条 xGMI，峰值约 76.8 GB/s/方向；B300 走 NVSwitch，约 900 GB/s/方向。我们 2 卡引擎的 MoE 用 AllGather+ReduceScatter，全部走这一条链路。按 bf16 估算，P 每算 1 个 token 要单向传约 0.82 MB，大约占 P 每 token 耗时的 40–50%。
   - **Blackwell 专属的融合算子**：FlashMLA MegaAttention、DeepGEMM MegaMoE、DeepSelect top-k。ROCm 上只有 MegaMoE 有对应物，即 `aiter_mega_moe`。
4. **以现在的 kernel，D 已经接近上限。** 用 c384 和 c768 两点拟合，D 每步耗时 ≈ 23 ms + 0.37 ms × 每步请求数。KV 用满时每 rank 吞吐只能再多约 5%，把 GMU 提到 0.95 也只到约 +9%。所以只优化 P，系统吞吐最多涨 5–9%（约 480–500k）。要明显超过 500k，D 每步的效率必须一起提升。
5. **只改配置就能落地的：**
   - D 的 CUDA graph 要覆盖实际批大小。现在每步超过 168 条就跑 eager，c768 时大部分步都超过。
   - GMU 从 0.9 提到 0.95。
   - P 去掉用不上的 decode graph，把显存还给 KV。
   - UMBP 开 lazy offload。
   - D 的 `prefill-schedule-interval` 从 11 改为 2。
   - 显式绑定 NUMA。
6. **最值得做的结构实验**：P 和 D 改成 DEP2 + `aiter_mega_moe` + MoRI，对标 B300 的 DEP2 + DeepGEMM MegaMoE，目的是压缩并隐藏 xGMI 上的 MoE 通信。其次是 indexer top-k 后端的 A/B。
7. **不适用或无效的 B300 设置：**
   - Blackwell 专属 kernel、NIXL/EFA、EPLB 的 symm_mem 通信。
   - `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP`：该镜像里根本没有这个变量，是空设置。
   - `VLLM_USE_V2_MODEL_RUNNER`：已经是默认。
   - `prefix-cache-retention-interval`：DSv4.1 开 SWA bounded replay 时预计无效。
   - P 侧的 `prefill-schedule-interval`：纯 prefill 引擎上基本不起作用。

---

## 1. 对比对象与口径

### 1.1 两个点

| | 我们（基线） | B300 828.2k |
|---|---|---|
| 来源 | `../raw/pd1p1d-c768-hyper-1121T1745/` | InferenceX API 行 `448173`（2026-11-17）；[Actions run 43769338888](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/43769338888/attempts/2)，来自 [PR #4585](https://github.com/SemiAnalysisAI/InferenceX/pull/4585)（head `a358d7d`） |
| 配方 | `examples/vllm_1p1d_dsv41_flash/engine/leg.sh`、`common.sh` | `inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/dsv41flash/vllm/b300-fp4/agentx/disagg-variants.yaml` 的 `override_5p1d_dep2_dep8_c3584`（跑分时的版本与 main 当前版本逐字一致） |
| 硬件 | 1 个 MI355X 节点，用其中 4 卡 | 3 个 B300 节点：P 用 gpu-01 的 8 卡和 gpu-02 的 2 卡，D 用 gpu-03 的 8 卡，共 18 卡 |
| 拓扑 | 1P1D：P 是 DP2（GPU 0,1），D 是 DP2（GPU 4,5） | 5P1D：P 是 5 × DEP2，D 是 1 × DEP8 |
| 并发 | 768（每卡 192） | 3584（每卡 199） |
| 镜像 | `vllm-openai-rocm:nightly-9c1d45c2`（vLLM 2026-11-19） | `vllm-openai:cu134-nightly-96710245`（vLLM 2026-11-10）。我们的版本比它新 755 个提交，并包含它 |
| 前端 / router | Infera Rust router：kv-aware + prefill 会话亲和 | Dynamo 1.6.0.dev20261010 frontend：`router-mode kv` + 会话亲和 |
| KV 传输 | Mooncake，RDMA 回环（P、D 在同一节点） | NIXL push 模式，LIBFABRIC over EFA（跨节点） |
| 主机 KV 层 | UMBP standalone，449 GB，两个 P rank 共享，eager 写入 | vLLM `SimpleCPUOffloadConnector`，每个 P rank 70 GiB，lazy 写入 |
| 投机解码 | DSpark K=5，计分时 golden 接受长度 3.51 | 同上。InferenceX 的 golden acceptance planner 会改写所有 srt 任务（`infx/launch/drivers/srt/submit.py`） |

### 1.2 口径注意

1. InferenceX 的 `tput_per_gpu` 是总 token 除以全部 GPU；`input_tput_per_gpu` 按 P 卡数归一，`output_tput_per_gpu` 按 D 卡数归一。
2. **InferenceX 公布的命中率混进了 D 侧。** `infx/results/agentic/backends/vllm.py` 把所有被抓取 server 的 `vllm:prompt_tokens_by_source` 加在一起。disagg 时，D 从 P 收到的 KV 被记为 `external_kv_transfer`。B300 的 5 个点 external 命中都 ≥ 0.5000，其中 c256 恰好是 0.5000、c768 是 0.5002、c1280 是 0.5003，说明这一组点把 D 计入了。其他硬件的一些点 < 0.5，说明并非所有跑分都计入了 D，所以只对 B300 做还原。还原后 B300 c3584 的 P 侧：
   - GPU 命中 ≈ 2 × 0.4657 = 93.1%
   - 主机层命中 ≈ 2 × (0.5108 − 0.5) = 2.2%
   - 重算 ≈ 4.7%（理论未命中为 3.1%）
3. 我们 P 侧的命中直接取自引擎计数器 `vllm:prompt_tokens_by_source`：GPU 59.1%，UMBP 36.7%，重算 4.2%（理论未命中为 3.4%）。
4. B300 decode 设了 `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP=4`。这个变量只存在于 vLLM 的 dev 镜像（vLLM recipe 里有说明），`96710245` 和我们的 `9c1d45c2` 源码里都没有。InferenceX 的 changelog 在别的 disagg 配方里也已把它当作 unused 删掉了。所以 828k 这次跑分里它不生效。

---

## 2. 总体指标与差距拆解

| | 我们 1P1D c768 | B300 5P1D c3584 | 比值 |
|---|---|---|---|
| GPU（P+D） | 2+2 | 10+8 | |
| 总吞吐 tok/s/GPU | 459,183 | 828,209 | 1.80 |
| 请求速率 req/s（括号内为每卡） | 14.75（3.69） | 109.9（6.11） | 1.66 |
| 平均输入 / 输出 token | 123,237 / 940 | 134,602 / 1,012 | 1.09（两者之和的比值） |
| 每张 P 卡的输入 tok/s（含命中） | 908,869 | 1,479,650 | 1.63 |
| 每张 P 卡每秒新算 token | 38,737 | ≈69,400 | ≈1.79 |
| 每个请求新算 token | 5,217 | ≈6,300 | |
| P 侧 GPU 命中 / 主机层命中 / 重算 | 59.1% / 36.7% / 4.2% | ≈93.1% / 2.2% / 4.7% | |
| 每张 D 卡的输出 tok/s | 6,930 | 13,908 | 2.01 |
| 每个 D rank 每步的请求数 | ~190（running 均值 ~210） | ~156（上限 168） | |
| D 每步耗时 | ~93 ms（均值） | ~40 ms（3.51 ÷ 交互性中位数） | |
| TTFT p50 / p90 | 16.96 / 31.75 s | 8.22 / 13.25 s | |
| ITL p50 / 交互性 p50 | 30.5 ms / 32.8 tok/s | 11.5 ms / 86.8 tok/s | |

这张表的读法：

- 总吞吐 = 请求速率 × 每请求 (输入 + 输出) ÷ GPU 数，所以 1.80 ≈ 1.66 × 1.09。输入里 95% 以上是缓存命中，所以这个"吞吐"本质上衡量的是每卡每秒能完成多少个长上下文请求。
- 两边每个请求在 P 上新算的 token 差不多（5.2k vs 6.3k）。B300 的 P 每卡新算速度是我们的 1.79 倍，D 每卡输出是我们的 2.0 倍。差距在"每卡算得快"，不在"需要算得少"。
- B300 的 P 侧 GPU 命中约 93%，主机层只补约 2%；我们的 GPU 只命中 59%，另有 37% 要从 UMBP 搬回 GPU。

D 固定为 DEP8 时，B300 的吞吐随 prefill worker 数的变化：

| 点 | P worker | GPU（P+D） | 并发 | tok/s/GPU | 每张 D 卡输出 | TTFT p50 |
|---|---|---|---|---|---|---|
| 2P1D | 2 | 4+8 | 256 | 154,022 | 1,543 | 0.22 s |
| 2P1D | 2 | 4+8 | 768 | 420,185 | 4,420 | 0.63 s |
| 3P1D | 3 | 6+8 | 1280 | 576,607 | 7,081 | 0.75 s |
| 4P1D | 4 | 8+8 | 2048 | 738,119 | 10,527 | 0.92 s |
| 5P1D | 5 | 10+8 | 3584 | 828,209 | 13,908 | 8.22 s |

每张 D 卡的输出从 1.5k 一路涨到 13.9k：prefill worker 加到 5 个、并发加到 3584，才把 DEP8 decode 喂满。最后一个点 TTFT 中位数跳到 8.2 s，说明 5 个 P 也开始排队了。B300 的最高点本质上是"一个大 decode 池，按需增加 prefill"，这种结构在 2+2 里做不出来。

---

## 3. 并行策略

### 3.1 对比

| | 我们 P | 我们 D | B300 P（每个 worker） | B300 D |
|---|---|---|---|---|
| attention | DP2 | DP2 | DP2 | DP8 |
| MoE | 专家按 TP 切到 2 个 rank，无 EP；通信用 AllGather+ReduceScatter | 同左 | EP2，DeepGEMM MegaMoE 内部完成 dispatch 和 combine | EP8 + EPLB（16 个冗余专家，symm_mem 通信，异步） |
| 每卡专家份额 | 1/2 | 1/2 | 1/2 | 1/8（另加冗余） |
| 每 rank 权重 | 150.82 GiB（日志） | 同左 | 未公布 | 未公布（明显更少） |
| 每 rank KV | 60.96 GiB，29.9M token | 同左 | — | — |
| 卡间链路 | 1 条 xGMI，峰值 ~76.8 GB/s/方向 | 同左 | NVLink5，~900 GB/s/方向 | 同左 |
| 实例数 | 1 | 1 | 5 | 1 |

### 3.2 分析

**Attention 两边都用 DP。** DSv4.1 的 KV 是单头的压缩 latent。如果 attention 做 TP，每个 rank 都要存一份完整的 KV，KV 容量会按 TP 度数下降。所以吞吐场景下两边都用 DP attention，这一点没有差异。

**MoE 的切法。** 只有 2 张卡时，TP-MoE 和 EP2 每卡都存一半专家，显存占用一样。区别只在通信形态，以及能不能和计算融合：

- **TP-MoE（我们，AgRs）**：每层先把两个 rank 全部 token 的 hidden state（bf16）all-gather 过来，算完再 reduce-scatter 回去。vLLM 日志里有：`Detected DP deployment with no --enable-expert-parallel. Falling back to AllGather+ReduceScatter dispatch/combine.`
- **EP2（B300 的 P）**：token 只发给持有它所需专家的 rank。top-6 路由下，2 个 rank 时几乎每个 token 都要跨卡，传输量和 all-gather 相当。但 dispatch 可以用 FP8，而且 MegaMoE 把 dispatch、GEMM、combine 融合进少数几个 kernel，通信能和计算重叠。
- **EP8（B300 的 D）**：每卡只放 1/8 的专家。decode 每步都要把本卡的专家权重读一遍；我们 D 拟合出的固定开销约 23 ms，和读一遍本 rank 约 150 GiB 权重的时间同一量级。EP8 把这部分读取降到 2 路切分时的 1/4，省下的显存给 KV。代价是需要高带宽的 all-to-all，以及用 EPLB 平衡专家负载。

**互联是 MI355X 上 2 卡引擎的硬约束。** 下面是估算，前提有三个：AgRs 用 bf16（日志里专家显示为 `AiterMxfp4Experts [w=mxfp4, a=none]`，进入专家前激活没有量化）；hidden 为 5120、共 40 层（来自日志里 aiter MoE 的 shape 键和 `layers.[0-39]`）；通信不与计算重叠。

- 每算 1 个 token，每层 all-gather 约 10 KB、reduce-scatter 约 10 KB，40 层合计约 0.82 MB/方向。DSpark 的 3 个草稿层另算。
- **P**：每个 rank 实测每秒新算 38.7k token，即每个 token 约 25.8 µs。0.82 MB 在 60–77 GB/s 上需要 10.7–13.7 µs，**约占 40–50%**。
- **D**：每条序列每步有 6 个 token（1 + 5 个草稿），约 4.9 MB，需要 64–82 µs，约占每条序列每步成本（0.37 ms，见 7.2）的 17–22%。
- 同样的数据量在 NVLink5 上只需要约 1/12 的时间，所以 B300 用 2 卡的 DEP2 做 prefill worker 不会遇到这个问题。

这是本报告中最重要、也最需要实测确认的判断。验证方法有两步：先用 rocprof 或 vLLM profiler 抓 P 稳态下 20 步，看 RCCL all_gather / reduce_scatter 的时间占比；再用 rccl-tests 在 GPU0↔GPU1 上测 168 MB 消息的实际带宽（满 16k token 的一步，每层 all-gather 正好约这么大）。

**P:D 配比。** 在这个 workload（输入/输出约 130:1）上，B300 最高点用的是 10:8。我们 2:2 时的情况：

- P 已饱和：每 rank 排队约 116 个、在处理约 4 个，每张 P 卡 7.4 req/s。
- D 的每 rank KV 使用率均值 81%、p90 99.7%。按 7.2 的拟合，D 的上限约 7.4–7.7k tok/s/rank，相当于每卡 7.9–8.2 req/s。

所以当前 1:1 基本平衡，P 略紧。2+2 固定时没有调整配比的余地，只能提高每卡效率。

### 3.3 各并行方案的适用场景

| 方案 | 适用 | 不适用 / 代价 |
|---|---|---|
| DP attention + TP-MoE（AgRs），2 卡 | 不依赖 EP kernel，最稳；token 少或卡间带宽高时开销小 | 通信量等于全部 token × hidden × 2 个方向；在 MI355X 单条 xGMI 上，大的 prefill 批代价很高；通信无法与 GEMM 融合 |
| DEP2（DP + EP2）+ MegaMoE | 每 worker 2 卡、可以横向复制的 prefill 池（B300 的 P）；需要用融合 MoE 隐藏通信 | 显存与 TP-MoE 相同；需要 MoRI / DeepEP 级别的 all-to-all；ROCm 的 `aiter_mega_moe` 要求 gfx950、TP=1、不开 EPLB |
| DEP8 / DEP16 decode + EPLB | 高并发的大 decode 池：权重读取被摊薄，KV 空间更多 | 需要 8 张以上的卡和高带宽全互联；EP 度数大时必须开 EPLB |
| TP4 / TP8（含 monokernel） | 低并发、高交互性（B300 agg 的 TP4 c1–c16；MI355X TP4 monokernel） | MLA 的 KV 在 TP rank 间重复，吞吐差；MI355X 的 monokernel 不支持 DP |
| 多个小 P worker + 一个大 D | 输入远多于输出、命中率高的 agentic 负载，P 可以随并发增减 | 需要 router 在多个 worker 之间做 KV 亲和；卡数多 |

---

## 4. Router 策略

| | 我们（Infera Rust router） | B300（Dynamo frontend 1.6） |
|---|---|---|
| 选 prefill | kv-aware，`cost = -w × 命中块数 + 在途负载`，w=20；选择粒度到 P 的 DP rank | `router-mode kv`：由 KV 事件驱动，按前缀重叠打分加活跃块负载，用默认权重；共 10 个 DP rank 可选 |
| 会话亲和 | `--session-affinity prefill`，会话 ID 来自 AIPerf 的 `x-session-id` | `router-session-affinity-ttl-secs 14400`，会话 ID 来自 `x-dynamo-session-id` |
| 随机性 | 取 argmin | `DYN_ROUTER_TEMPERATURE=0`，同样是 argmin |
| router 侧排队 / 拒绝 | 无 | 关闭（`DYN_ROUTER_QUEUE_THRESHOLD=None`，各 active 阈值都为 None）：请求立即下发，由引擎排队 |
| 选 decode | 只有 1 个 D worker，具体 rank 由 vLLM 的 DP 负载均衡决定（`INFERA_VLLM_DP_ENGINE_LB=1`），w=2 | 只有 1 个 D worker（DEP8），由 Dynamo 选 rank |
| P→D 下发 | 串行：P 回复之后才发 D（PATCHES #11） | 先 prefill 后 decode（Dynamo disagg 的标准流程），KV 用 NIXL push |
| 分词 | router 内部渲染并分词（DSv4.1 编码器，PATCHES #12） | `DYN_TOKENIZER=fastokens`，8 GB 分词缓存 |
| 开销 | router + 两个引擎前端约 20 ms；客户端到 router 约 0.2 s | 未公布 |

分析：

- **两者原理一致。** 都是高 overlap 权重加会话亲和，把同一会话的多轮请求钉在持有其前缀的 prefill rank 上，这是 agentic 多轮场景的正确做法。
- **1P1D 下 router 可选的余地很小。** 只有 2 个 P rank 可选，两个 rank 的负载也基本均衡（排队 110 vs 121，prompt 吞吐 988k vs 850k tok/s）。UMBP 由两个 rank 共享，即使选错 rank 也能从 UMBP 命中。
- **router 不是 1P1D 的吞吐杠杆**，不建议为了吞吐去改它。有两个小问题可以顺手处理：
  - prefill 日志里有 `UMBP KV event tracking limit reached; suppressing new BlockStored events`。UMBP 的 `kv_event_max_objects` 默认是 65536，超过之后 router 看到的主机层视图就不完整了。
  - 全程出现过 7 次 `UMBP standalone lookup ... no reply within 1s`。超时按未命中处理，但次数很少。
- **各项设置的适用场景**：
  - router 侧的排队和阈值，适合在 worker 很多时做准入控制。
  - 温度 > 0 适合 worker 很多、负载打分经常持平的情况，可以避免请求扎堆。
  - 会话 TTL 要覆盖 agentic 会话两轮之间的思考间隔（B300 设为 4 h）。

---

## 5. 融合算子与 kernel 后端

| 部位 | 我们（MI355X） | B300 | ROCm 上能否对齐 |
|---|---|---|---|
| sparse MLA attention | `ROCM_FLASHMLA_SPARSE_DSV41`（AITER，`VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True`），KV 为逐张量 `fp8_e4m3` | `FLASHMLA_MEGA_ATTN_DSV41`：一个 kernel 完成 Q RoPE、sparse attention、输出逆 RoPE 和 FP8 量化，结果直接写进 `wo_a` 的输入。SM100 专属 | 没有对应物，属于 kernel 工作 |
| MoE | `aiter`：FlyDSL 的 a8w4 两段式 kernel（`flydsl_moe1_afp8_wfp4…`），外加 AgRs 通信 | `deep_gemm_mega_moe`：把 EP dispatch、专家 GEMM、combine 融合在一起 | **有**：`aiter_mega_moe`，要求 gfx950、DP+EP、MoRI、不开 EPLB（[vllm#59685](https://github.com/vllm-project/vllm/pull/59685)）。PR 里 DSv4-Pro DEP8 跑随机 1k/1k 时，输出吞吐比不融合的 aiter EP 路径高 2.6–3.5 倍。但那个基线本身很慢（c1 时 ITL 114 ms），不能直接外推到我们的 TP-MoE 基线 |
| indexer top-k | 默认后端（ROCm 上是 `per_row`；日志里也出现了 FlyDSL radix top-k） | `deep_select`（SM100a / SM103a） | 可以 A/B：`aiter`（gfx950）、`persistent` |
| indexer | MXFP4 paged indexer + sparse logits | 同左 | 已对齐 |
| 线性层 | `RocmDotScaledMxfp8LinearKernel` | FlashInfer（`enable_flashinfer_autotune`） | — |
| mHC | AITER `mega_mhc_stream_h5120` | — | 已在用 |
| Engram 表 | 放主机内存，`dp_shared_memory=true`：经 /dev/shm 在 DP 副本间共享，每个引擎 2 × 94.42 GiB | 放主机内存，`embedding_across_dp=true`：按 DP rank 切分 | 单节点 DP2 时我们的做法与之等价，而且没有每步的 DP 集合通信，保持现状 |
| decode 专用 kernel | 不用 monokernel（mono decode 不支持 DP） | 不用 | — |
| CUDA graph | breakable FULL_AND_PIECEWISE，PIECEWISE 最大 1024 token；FULL decode graph 最多 168 条 | P：FULL_AND_PIECEWISE 最大 2046，外加最大 2048 的 decoder replay graph；D：FULL_DECODE_ONLY，最大 6 × max-num-seqs | 见 6.2 和 A1、A3 |
| torch.compile | `CompilationMode.NONE` | `mode: VLLM_COMPILE` | 两个版本里 DSv4.1 模型都没有 `@support_torch_compile`，B300 实际上也只用了 graph，这不是杠杆 |
| EPLB | 无 | D 开启：16 个冗余专家，`symm_mem` 通信，异步 | EP2 不需要，`aiter_mega_moe` 也不支持 |

各类融合算子的适用场景：

- **MegaAttention / MegaMoE**：把一层里的多个算子和通信合并成一两个 kernel。收益来自三方面：kernel 数量减少、中间结果的读写减少、通信被藏进计算里。token 越多、EP 度数越大，越划算。B300 的 P 和 D 都依赖它们。
- **DeepSelect 等 top-k 后端**：决定 decode 时每条序列的 indexer 成本，上下文越长越重要。这个 workload 平均上下文有 12–13 万 token。
- **monokernel**：适合 TP 下每步不超过 48 行的小 decode，用于交互性点，不适合我们的吞吐点。
- **EPLB**：只在 EP 度数较大（8 及以上）时有意义。

---

## 6. 其他 vLLM 参数逐项对比

"语义"一列都在对应版本的 vLLM 源码里核对过。

### 6.1 Prefill 侧

| 参数 | 我们 | B300 | 语义 | 对我们 |
|---|---|---|---|---|
| `gpu-memory-utilization` | 0.9 | 0.95 | — | 可用，见 A2 |
| `max-num-batched-tokens` | 16384 | 16384（GB300 用 32768） | 每步的 token 预算 | 先保持，B1 之后再测 32768（B4） |
| `max-num-seqs` | 默认 1024 | 64 | 每个 rank 同时运行的请求上限 | 我们每 rank 只跑约 4 个，没有影响 |
| `long-prefill-token-threshold` | 0 | 8192 | 有多个请求时，单个请求每步最多算 8192 token；只剩一个请求时不限 | 主要改善 TTFT 的公平性，吞吐基本不变，优先级低 |
| `prefill-schedule-interval` | 不设 | 2 | 只有当有 decode 请求在跑时才推迟 prefill，并且引擎饱和时自动失效（见 `scheduler.py` 的 `defer_prefills`、`prefill_capacity_bound`） | 在纯 prefill 引擎上基本无效。示例 README 里"在纯 prefill 引擎上 N 步里空转 N−1 步"的说法，与这个版本的代码不符，可以更正 |
| `prefix-cache-retention-interval` | 0 | 32768 | 只作用于参与前缀缓存的 sliding-window / Mamba 组 | DSv4.1 开了 `swa_bounded_replay`（两边都开）时，SWA 组不参与前缀缓存，压缩器状态又是 CircularBuffer，所以预计无效 |
| KV dtype | `fp8_e4m3` | `fp8`（即 `fp8_ds_mla`） | ROCm 的 sparse MLA 需要逐张量 fp8（vllm#66036） | 保持 |
| 主机 KV 层 | UMBP standalone 449 GB，eager 写入 | SimpleCPUOffload，每 rank 70 GiB，`lazy_offload` | lazy：只把快要被逐出的 GPU 块写到主机 | 可用，见 A4 |
| `kv_load_failure_policy` | recompute | fail | 加载失败时是重算还是报错 | 保持 recompute |
| CUDA graph | 83 个 PIECEWISE + 80 个 FULL decode + 80 个 dspark + 15 个 decoder replay，每 rank 约 12.7 GiB | FULL_AND_PIECEWISE，最大 2046 | P 每个请求只出 1 个 token；排队时每步约 16k token，超过 1024 一律跑 eager | 可用，见 A3 |
| `enable-ep-weight-filter` | — | true | 开 EP 时每个 rank 只加载自己的专家，加快加载 | 只在做 B1 时有用 |
| `numa-bind` | 自动绑定失败，日志为 `Automatic NUMA binding unavailable on this host; starting unbound` | 显式开启 | — | 可用，见 A6 |
| `cpu-distributed-timeout-seconds`、`VLLM_RPC_TIMEOUT` 等 | — | 调大 | 启动和 Engram 加载的超时 | 与性能无关 |

### 6.2 Decode 侧

| 参数 | 我们 | B300（c3584） | 语义 | 对我们 |
|---|---|---|---|---|
| `max-num-seqs` | 默认 1024（实际被 KV 限制在每步约 190–210 条） | 每 rank 168 | — | 不建议照搬这个上限，改为扩大 graph 覆盖，见 A1 |
| `max-num-batched-tokens` | 16384 | 1680（= 10 × max-num-seqs） | — | 没有影响 |
| CUDA graph | FULL decode graph 最多 168 条/步（= 1024 ÷ 6）；PIECEWISE 最多 1024 token | FULL_DECODE_ONLY，从 6 到 1008，步长 6 | c768 时 D 每步 p50 有 220–225 条（约 1320 token），**超过了两种 graph 的上限，跑 eager**；c384 时每步约 98 条，在 graph 范围内 | 可用，见 A1 |
| `prefill-schedule-interval` | 11 | 2 | 在 D 上控制"KV 已经到达的请求"多久被收进 batch，最多要等 N−1 步（11 × ~93 ms ≈ 1 s） | 可用，见 A5 |
| GMU | 0.9 | 0.95 | — | 见 A2 |
| EPLB | 无 | 16 个冗余专家 | — | EP2 不需要 |
| `numa-bind-nodes` | 未绑定 | `[0,0,0,0,1,1,1,1]` | — | 见 A6 |
| `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` | — | 4 | 该镜像里不存在，是空设置。对应的 CLI 参数是 `--max-async-kv-loads-per-step`（两个版本都有，B300 没设） | 不需要 |

### 6.3 环境变量

| 变量 | B300 | 说明 | 对我们 |
|---|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER=1` | 设了 | Model Runner V1 已经删除，设成 1 只会打印 "redundant" | 不需要 |
| `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` | 设了 | DP 中空闲 rank 的 dummy batch 改用随机输入 | 影响很小，可以不管 |
| `VLLM_ALLREDUCE_USE_SYMM_MEM=0`、`NCCL_CUMEM_ENABLE=1`，以及一组 UCX / FI / EFA 变量 | 设了 | CUDA / EFA 专属 | 不适用 |
| 我们的 `VLLM_ROCM_*`、`VLLM_USE_BREAKABLE_CUDAGRAPH=1`、`MC_*` | — | ROCm、AITER、Mooncake 必需 | 保持 |

---

## 7. 我们的瓶颈画像

### 7.1 Prefill

- **已饱和。** 每 rank 排队约 110–121 个、运行约 4 个，排队原因几乎都是 `capacity`。TTFT 均值中 83% 是在 P 上排队。
- **被每步的算力卡住，而不是被并发数或 KV 卡住。** 每个请求平均新算 5,217 token（p50 约 1.8k，p90 约 8.4k）。16384 的预算一步大约能装 3 个请求，和"运行约 4 个"对得上。P 的 KV 使用率只有约 9%。
- **每 token 的耗时里，通信可能占一半。** 每 rank 每秒新算 38.7k token，即每个 token 约 25.8 µs；如果每步都是满的 16k token，平均每步不超过 0.42 s。按 3.2 的估算，其中 AgRs 通信约占 40–50%。
- **GPU 前缀缓存偏小。** GPU 只命中 59%，UMBP 补了 37%。
- **UMBP 写入量异常大。** UMBP 是 eager 写入，两个 rank 合计每小时写 5.95 TB，而加载只有 3.58 TB/h。按加载时的每 token 字节数折算，写入量约等于把"所有 GPU 命中的 token + 新算的 token"再写一遍，是新算 KV 量的十几倍。

### 7.2 Decode

- **两点拟合。** 每步请求数用"每步生成 token ÷ 3.51"算：c384 时每步约 98 条、耗时 59.5 ms；c768 时约 189 条、93.5 ms。由此得到 **每步耗时 ≈ 23 ms + 0.37 ms × 每步条数**。截距对应每步读一遍权重之类的固定成本；斜率对应每条序列的 attention、indexer、通信等成本。
- **以现有 kernel，D 接近上限。** 按此拟合，每 rank 吞吐 = 3.51n ÷ (23 + 0.37n)：
  - 现在约 7.1k tok/s；
  - KV 用满（n ≈ 233）时约 7.4k，即 +5%；
  - GMU 提到 0.95（n ≈ 285）时约 7.7k，即 +9%；
  - 理论渐近值约 9.4k。
- **斜率里可能混有 eager 开销。** c768 时大部分步超出了 graph 上限、跑的是 eager，而 c384 的步都在 graph 内。扩大 graph 覆盖之后，斜率可能下降（A1）。
- **和 B300 对比。** 按"每步耗时 ÷ 每步条数"算，B300 的 D 每 rank 约 156 条、约 40 ms，每条约 0.26 ms；我们每条约 0.49 ms，大约是 B300 的 1.9 倍。B300 这个数字里还包含了 EP8 摊薄权重读取带来的好处。

### 7.3 推论

- 只改善 P：P 变快之后 D 会先到顶，系统吞吐最多涨约 5–9%（约 480–500k）。
- 要明显超过 500k，D 每条序列的成本（0.37 ms）和每步的固定成本（23 ms）必须一起降下来。手段是 graph 覆盖、MoE 通信（EP + MegaMoE）、indexer 和 attention 的 kernel。
- c768 已经饱和（TTFT p50 17 s）。在容量提升之前，单纯加并发没有意义；每一轮改动之后都要重新扫并发找拐点。

---

## 8. 可落地的改动

### A. 只改配置（低风险，可以合并成一轮）

| # | 改动 | 依据 | 预期 | 风险 / 怎么验证 |
|---|---|---|---|---|
| A1 | D 改用 FULL_DECODE_ONLY，并把 decode graph 的覆盖扩到 KV 允许的最大并发（约 260–290 条）。具体做法：去掉 D 上的 `--max-cudagraph-capture-size 1024`，改为 `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":1728}'`（1728 = 6 × 288），必要时再配合 `max_decode_cudagraph_capture_requests` | c768 时每步 p50 有 220–225 条，超出了 FULL graph 的 168 条和 PIECEWISE 的 1024 token。decode graph 的条数上限是 `max_cudagraph_capture_size ÷ 6`（见 `compilation.py`）。B300 的 D 用 FULL_DECODE_ONLY，覆盖到 6 × max-num-seqs。MI300X / MI325X 的 DSv4.1 配方也用 FULL_DECODE_ONLY | 降低 D 每步耗时；去掉 PIECEWISE 也能释放一部分 graph 内存。幅度要实测 | vLLM 文档说明：DeepSeek indexer 的 FULL graph 会占用 KV 内存。验证时看日志里的 `Available KV cache memory` 和 D 每步耗时。准入步（每 N 步一次的混合批）仍会走 eager |
| A2 | P、D 都改 `--gpu-memory-utilization 0.95`（可以先从 0.93 开始） | 现在每 rank KV 为 60.96 GiB；B300 用 0.95；多出约 14 GiB，即约 +23% | D 能多容纳约 20% 的序列（按 7.2 拟合，吞吐约 +4%）；P 的 GPU 前缀缓存变大，UMBP 加载减少 | 峰值激活可能 OOM；用 GEM 分配器时要确认实际可用量；每次改动都要跑 smoke 和 accuracy |
| A3 | P 去掉用不上的 graph：至少设 `decode_cudagraph_capture_request_sizes: []`（文档：`[]` 表示不捕获），PIECEWISE 只保留小尺寸，同时做一组 `--enforce-eager` 对照 | P 每个请求只出 1 个 token，排队时每步约 16k token，全是 eager；而 graph 池每 rank 约 12.7 GiB，是从 KV 里扣的 | P 的 GPU KV 增加约 20%，GPU 命中率上升 | 小批步和 decoder replay 段可能变慢。验证时对比 P 每 token 耗时和 GPU 命中率 |
| A4 | P 的 UMBP connector 加 `"lazy_offload": true`，同时把 `kv_event_max_objects` 调大 | UMBP 每小时写入 5.95 TB，约为新算 KV 的十几倍；B300 的主机层也是 lazy；对应功能 vLLM #61797（11-11 合入）已在我们的镜像里 | 降低 D2H、PCIe 和 UMBP server 的压力，减少主机层抖动。对命中率和 P 耗时的影响要实测 | 这是新功能，先跑 `umbp_datapath_check` 和 accuracy |
| A5 | D 改 `--prefill-schedule-interval 2`（现在是 11） | 在 D 上，KV 已经到达的请求最多要等 N−1 步才能进 batch，约 1 s；B300 的 D 用 2 | 主要降低 TTFT（D 排队均值 1.4–2.1 s），D 的空位也能更快被填上 | 混合步变多，eager 步也随之变多。验证时看 D 每步耗时 |
| A6 | P、D 都显式绑定 NUMA：`--numa-bind --numa-bind-nodes …`，按 GPU 所在 socket 设置（P 的 GPU0/1、D 的 GPU4/5），并确认镜像里有 `numactl` | 两边日志都写着"自动绑定不可用，未绑定"；B300 配方的注释说，未绑定的主机线程在高并发下会拖慢 decode | 小幅降低 decode 的主机侧开销 | 风险低；先用 `rocm-smi --showtopo` 确认每张 GPU 属于哪个 NUMA 节点 |

### B. 结构性实验（潜在收益更大，需要验证）

| # | 改动 | 依据 | 注意 |
|---|---|---|---|
| B0 | 先做 profile：P、D 在稳态下各抓约 20 步；再用 rccl-tests 测 GPU0↔GPU1 的 all_gather / reduce_scatter 带宽 | 3.2 里的通信占比只是估算，而它决定了 B1 值不值得做 | 必做，成本最低 |
| B1 | 先 P 后 D：`--enable-expert-parallel --all2all-backend mori --moe-backend aiter_mega_moe` | 对标 B300 的 DEP2 + DeepGEMM MegaMoE。目标是用 FP8 dispatch 加融合重叠，减少单条 xGMI 上的 MoE 通信 | InferenceX 的 MI355X DP2 配方是特意去掉 EP 的，2 卡下 EP 是否更快没有公开证据。MoRI 的对称堆默认每 rank 8 GiB，会从 KV 里扣。还要验证它与 Mooncake / UMBP connector、`canonical_kv`、DSpark 草稿层以及 golden 模式的兼容性和正确性 |
| B2 | D 上用 `--kernel-config '{"sparse_indexer_topk_backend":"aiter"}'`，和默认后端、`persistent` 做对比 | 我们的 decode 每条序列成本约为 B300 的 1.9 倍，并随上下文变长而增长；B300 用的是 `deep_select` | 显式指定的后端如果不满足约束，启动时会直接报错 |
| B3 | DBO（`--enable-dbo`，dispatch 和 combine 重叠） | 如果 B1 做不通，这是另一种隐藏 MoE 通信的方式；AMD 的 DSv4.1 模型代码里已经有 ubatch 相关处理 | 可用性没有验证过，属于探索 |
| B4 | B1 之后，在 P 上再测 `--max-num-batched-tokens 32768` 和 `--long-prefill-token-threshold 8192` | GB300 的 P 用 32768：GEMM 更大、步数更少 | 激活显存增加会挤占 KV；long-prefill 主要影响 TTFT |
| B5 | kernel 侧的两个问题：DSpark LM head 的 bf16 GEMM（N=129280）在 aiter 里没有调优配置（日志：`not found tuned config … using torch solution`）；运行中出现过 FlyDSL radix top-k 和 `_sparse_mla_fp8` 的 JIT 编译 | 补调优表、扩大 warmup 覆盖 | 交给 kernel 组；`enable_rocm_aiter_gemm_warmup` 可以解决一部分首批请求的卡顿 |

### C. 不适用或预期无效

| B300 设置 | 原因 |
|---|---|
| 5P1D、DEP8 decode、按并发增加 prefill worker | 需要超过 4 张卡，1P1D 2+2 无法复现 |
| `FLASHMLA_MEGA_ATTN_DSV41`、`deep_gemm_mega_moe`、`deep_select`、FlashInfer autotune | Blackwell 专属；MoE 部分用 `aiter_mega_moe` 对标（B1） |
| EPLB（symm_mem） | EP2 不需要；`aiter_mega_moe` 也不支持 |
| NIXL + LIBFABRIC/EFA、`enable-cumem-allocator false`、UCX / FI 环境变量 | 平台专属；我们的 Mooncake 传输 p90 约 40 ms，不是瓶颈 |
| `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` | 该镜像里不存在 |
| `VLLM_USE_V2_MODEL_RUNNER=1` | 已经是默认 |
| `prefix-cache-retention-interval 32768` | DSv4.1 开 bounded replay 时预计无效 |
| P 侧的 `prefill-schedule-interval 2`、`max-num-seqs 64` | 纯 prefill 引擎、且已饱和时不起作用 |
| `engram embedding_across_dp` | 单节点 DP2 用 `dp_shared_memory` 与之等价，而且没有每步的集合通信 |
| `mode: VLLM_COMPILE` | DSv4.1 模型没有 torch.compile 支持 |
| Dynamo router 本身 | 原理与我们相同，1P1D 下不是瓶颈 |

---

## 9. 建议的实验顺序

1. **B0：profile**（约半天）。确认 P 上 AgRs 通信的占比，以及 D 每步的构成。如果 P 的通信占比低于 20%，就下调 B1 的优先级，把重点转向 kernel。
2. **第一轮：配置改动 A1 + A2 + A5 + A6 一起上。** 这几项都只影响容量和调度，不改数值路径。跑 smoke、accuracy 和 c768 AgentX，与基线对比 D 每步耗时、D 的 KV、P 的 GPU 命中率和吞吐。
3. **第二轮：A3、A4 分别单独测。** A4 改动了主机层的数据路径，要单独验证正确性。
4. **第三轮：B1。** 先 P 后 D，每一步都要过 accuracy 门禁。
5. **第四轮：B2。** 做 top-k 后端的 A/B。
6. 每轮结束后，在 c640 / c768 / c896 附近重新扫一遍并发，找新的饱和点。

---

## 附录：证据位置

- **我们的启动参数**：`../raw/pd1p1d-c768-hyper-1121T1745/prefill.cmd`、`decode.cmd`。
- **权重、KV、graph、MoE 通信、NUMA**：见 `../raw/pd1p1d-c768-hyper-1121T1745/` 下的 `decode-175921.log.gz` 和 `prefill.log.gz`，关键行如下：
  - `Model loading took 150.82 GiB`
  - `Available KV cache memory: 60.96 GiB`
  - `GPU KV cache size: 29,919,641 tokens`
  - `Graph capturing finished … took 12.69 GiB`
  - `decode_cudagraph_capture_requests': 168`
  - `Falling back to AllGather+ReduceScatter`
  - `Automatic NUMA binding unavailable`
- **引擎计数器**：`../raw/pd1p1d-c768-hyper-1121T1745/agentx/server_metrics_export.csv`，包括 `prompt_tokens_by_source`、`umbp_transfer_bytes`、`iteration_tokens_total`、`num_requests_running/waiting`、`kv_cache_usage_perc`。c384 对应 `../raw/pd1p1d-c384-1120T0606/agentx/` 下的同名文件。
- **B300 那一行**：本目录 `inferencex_dsv41flash_b300_dynamo_vllm.json` 中 id 为 448173 的行（来源 `https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4.1-Flash`）；第 2 节的 2P–5P 曲线取自同一文件。
- **InferenceX 命中率的汇总方式**：`inferencex-e2e/infx/results/agentic/backends/vllm.py`。
- **vLLM 参数语义**（`9c1d45c2` 与 `96710245` 两个版本）：
  - `vllm/config/scheduler.py`：`prefill_schedule_interval`、`long_prefill_token_threshold`、`max_async_kv_loads_per_step`
  - `vllm/v1/core/sched/scheduler.py`：`defer_prefills`
  - `vllm/config/cache.py`：`prefix_cache_retention_interval`、`swa_bounded_replay`
  - `vllm/config/kernel.py`：`moe_backend`、`sparse_indexer_topk_backend`
  - `vllm/models/deepseek_v41/amd/model.py`：MegaMoE 要求开 EP
  - `vllm/distributed/kv_transfer/kv_connector/v1/umbp/scheduler.py`：`lazy_offload`
  - `vllm/config/compilation.py`：decode graph 的条数上限
