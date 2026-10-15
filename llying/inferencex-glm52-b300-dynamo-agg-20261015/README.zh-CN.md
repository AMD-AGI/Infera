# InferenceX：GLM-5.2 NVFP4 B300 Dynamo + SGLang（aggregated TP4）AgentX 为什么能到 42k tok/s/GPU

核对日期：2026-10-15。数据来自 InferenceX 公开 API 和该次 CI 上传的服务端日志，没有在我们的硬件上复现。
可复核的材料在 [`evidence/`](evidence/)，来源链接见 §7。

## 结论

**先更正对象：** InferenceX 仪表板上没有 GLM-5.3 在 B300 上的数据（GLM-5.3 只有 MI355X 的 tilert 结果）。
这次刷出 42k 的是 **GLM-5.2** NVFP4、B300、`dynamo-sglang`、MTP、aggregated 这组，对应
[InferenceX PR #3953](https://github.com/SemiAnalysisAI/InferenceX/pull/3953)，仪表板日期 2026-10-14。

**42k 是并发 48 时的 42,047 tok/s/GPU**，用 4 张 B300（一个实例）。其中输入 41,749、输出 298 tok/s/GPU；
AgentX 吞吐把命中缓存的输入 token 也计入，所以这个数主要反映“能留住并命中多少长前缀、能同时服务多少长会话”，
而不是算力。这一点的 TTFT p50 / p90 为 3.6 / 13.2 s，交互速度 p50 64.5 tok/s，已经是延迟变差的一端。

它快的主要原因按影响排序：

1. **4 卡一个 aggregated 实例，高并发时 TP4 + DP4 attention。** GLM-5.2 是 MLA 类模型，普通 TP 下每个 rank 都存完整
   的每 token KV；DP attention 让 4 个 rank 各存各的请求，能缓存的会话前缀约为 TP 时的 4 倍。
2. **会话亲和落到 DP rank 级别。** Dynamo 前端把这个实例注册成 4 个 `(worker, dp_rank)` 目标，并开启会话亲和
   （TTL 3600 s，最多 65,536 条）。并发 24 时 GPU 缓存命中 94.7%。
3. **很大的主机内存缓存（HiCache，`write_back`）。** 每 rank 270–295 GB。并发 48 时 GPU 命中只剩 52.3%，主机内存再命中
   42.5%，合计约 95%，接近理论上限 97.3%。
4. **高并发专用的调度参数**：chunk 32768–65536、`enable-prefill-delayer`、`enable-dp-lm-head`、max-running 128。

**一个与配置名称不符的事实：** 引擎**没有发布 KV 事件**（`use_kv_events=False`），Dynamo 启动时就警告缓存感知路由
缺少 KV 事件；4,888 次路由全部是 `overlap_blocks=0`。所以 `router-mode: kv` 实际看不到缓存，起作用的是会话亲和。
副作用是 4 个 rank 的请求分布不均（最多的 rank 占 33.2%，最少的 18.9%），这部分还有提升空间。

## 1. 数据点

B300、GLM-5.2、FP4、MTP、AgentX。每卡吞吐的分母是该点实际使用的 GPU 数（Dynamo 这组为 4，普通 SGLang 为 8）。

| 并发 | Dynamo + SGLang（4 卡）tok/s/GPU | TTFT p50 / p90 (s) | 交互 p50 (tok/s) | 普通 SGLang（8 卡）tok/s/GPU | TTFT p50 / p90 (s) |
|---:|---:|---|---:|---:|---|
| 1 | 5,207 | 0.32 / 0.86 | 318.5 | 2,620 | 0.77 / 1.33 |
| 8 | 13,666 | 0.22 / 0.71 | 225.2 | 7,210 | 0.35 / 1.14 |
| 16 | 22,862 | 0.29 / 2.50 | 151.1 | 12,936 | 0.44 / 1.74 |
| 20 | 26,052 | 0.48 / 6.84 | 119.9 | 15,900 | 0.55 / 2.06 |
| 24 | 29,726 | 1.30 / 2.92 | 94.1 | **18,357** | 0.60 / 2.30 |
| 28 | — | — | — | 18,352 | 0.99 / 5.86 |
| 32 | 37,390 | 1.41 / 3.60 | 86.6 | 16,714 | 3.33 / 15.63 |
| 40 | — | — | — | 13,372 | 15.79 / 35.91 |
| 48 | **42,047** | 3.60 / 13.17 | 64.5 | 8,537 | 43.48 / 130.52 |

缓存命中（`server_gpu_cache_hit_rate` / `server_cpu_cache_hit_rate` / 理论值）：

| 点 | GPU | 主机内存 | 合计 | 理论上限 |
|---|---:|---:|---:|---:|
| Dynamo c24 | 94.7% | 1.3% | 96.0% | 97.5% |
| Dynamo c48 | 52.3% | 42.5% | 94.8% | 97.3% |
| 普通 SGLang c24 | 75.8% | 20.3% | 96.1% | 97.7% |

并发 48 这一点：总吞吐 168,187 tok/s，3,626 s 内 4,349 个请求，平均输入 139k token（中位数 94k），分配主机内存 1,499 GB。
仪表板的数字和产物里的 AIPerf 结果一致：测量时长都是 3,625.61 s、TTFT p50 都是 3.60 s
（输出吞吐 1,190.3 与 1,185.6 tok/s 的小差别来自两边口径）。

## 2. 部署形态与生效参数

镜像 `lmsysorg/sglang:nightly-dev-cu13-20260901-07c8f729`，Dynamo `1.5.0.dev20260903`。一个节点 8 张 B300，每个数据点只用
其中 4 张跑一个 aggregated 实例（不是 PD 分离），etcd 和 NATS 在单独的 infra 节点。

**前端（[渲染后的 c48 配置](evidence/c48/config_c48.yaml)）：**

```yaml
frontend:
  type: dynamo
  args:
    router-mode: kv
    router-session-affinity-ttl-secs: 3600
    active-decode-blocks-threshold: None        # 关闭“worker 太忙就拒绝/转发”
    active-prefill-tokens-threshold: None
    active-prefill-tokens-threshold-frac: None
```

压测端设置 `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true`，让 AIPerf 给每个会话带 `x-dynamo-session-id`。

**引擎：** 公共部分为 NVFP4（`modelopt_fp4`）、KV `fp8_e4m3`、`bf16-gemm-backend: cutedsl`；EAGLE 用模型自带 NextN 头，
5 步 / topk 1 / 6 个 draft token，**接受长度由 launcher 固定为 3.61**（GLM-5.2 thinking-on 的标准值，与我们实验的模拟
接受口径相同），所以输出吞吐和 ITL 是模拟的理想值。HiCache `write_back`、`io-backend direct`、
`mem-layout page_first_direct`。各并发点的差别：

| 参数 | c1–c20 | c24 | c32 | c48 |
|---|---|---|---|---|
| 并行 | TP4 | TP4 + DP4 attention | TP4 + DP4 attention | TP4 + DP4 attention |
| `enable-dp-lm-head` | — | 开 | — | 开 |
| `enable-prefill-delayer` | — | 开 | 开 | 开，`max-delay-passes 60` |
| `chunked-prefill-size` | 8192 | 65536 | 32768 | 32768 |
| `mem-fraction-static` | 0.83 | 0.85 | 0.83 | 0.85 |
| `hicache-size`（GB/rank） | 270 | 295 | 270 | 270 |
| `max-running-requests` / `cuda-graph-max-bs` | 8–64 / 8–64 | 128 / 64 | 64 / 64 | 128 / 64 |
| `enable-symm-mem` | — | — | 开 | — |

c48 实际生效的 SGLang 参数（[`server_args.json`](evidence/c48/server_args.json)）与上表一致，另有
`skip_tokenizer_init=False`、`tokenizer_worker_num=1`、`stream_interval=1`、`schedule_policy=fcfs`。

运行时的 recipe（PR 分支 `caa2199e`）和合入后 main 上的版本，引擎与前端参数完全相同，差别只在压测框架元数据
（模型和镜像标识、并发点列表、压测命令）。两份 recipe 都存在 [`evidence/recipes/`](evidence/recipes/)。

## 3. 为什么快：逐项分析

### 3.1 DP attention 解决了 MLA 的 KV 重复

recipe 注释写明：“MLA-family: every TP rank holds complete per-token KV”。普通 TP 下每张卡各存一份完整 KV，卡数增加
并不增加能缓存的会话；DP attention 让 4 个 rank 分别服务不同请求、各存各的 KV，MoE 仍由 4 卡 TP 计算。
AgentX 是“很多个十几万 token 的长会话”，能留住多少前缀直接决定命中率和能撑住的并发。

对照组（普通 SGLang，8 卡一个 TP8 实例，无 DP attention）并发 24 时合计命中率同样约 96%，但峰值只有 18.4k/GPU，
并发 32 以后 TTFT 急剧上升（c48 时 p50 43 s），吞吐下降。差距主要不在命中率，而在同一份显存能同时留住的会话数。

### 3.2 会话亲和落到 DP rank

前端日志（[`routing-summary.txt`](evidence/c48/routing-summary.txt)）：

- `Adding worker WorkerWithDpRank { worker_id: …, dp_rank: 0..3 }`：同一个实例被注册成 4 个路由目标；
- `session affinity enabled ttl_secs=3600 max_entries=65536`：限制与我们 #186 的默认值相同。

4,888 次路由（含预热）中，真正进入选择逻辑（`Selected worker`，带打分）的只有 531 次：预热期间 97 次，正式测量期间
434 次。其余约 89% 的请求没有经过选择，据此推断它们由会话亲和直接定位到已绑定的 rank，只有新会话或绑定失效时才重新
选择。这和并发 24 时 94.7% 的 GPU 命中率一致。

### 3.3 KV 路由实际看不到缓存

worker 日志：`use_kv_events=False from kv_events_config=None`。前端启动时报
`KV-event publishing is disabled for a role that requires KV events … requirement="cache_aware_routing" … dp_ranks=0,1,2,3`。
4,888 次路由（含预热）全部 `overlap_blocks=0`。

后果是 rank 间请求分布不均，而且会话粘住后不会再纠正：

| DP rank | 请求数 | 占比 |
|---|---:|---:|
| 0 | 1,160 | 23.7% |
| 1 | 1,178 | 24.1% |
| 2 | 926 | 18.9% |
| 3 | 1,624 | 33.2% |

DP attention 下各 rank 每步同步执行，最忙的 rank 决定整体步速，所以这组结果还有提升空间
（思路类似我们 R2 的“按 token 需求选 rank”）。

### 3.4 主机内存缓存接住溢出

每 rank 270–295 GB，4 个 rank 合计 1.1–1.2 TB，`write_back`。并发 24 时几乎全在 GPU 命中；并发 48 时 GPU 只命中 52.3%，
主机内存补上 42.5%。没有这一层，c48 的大量前缀需要重算。

### 3.5 高并发调度参数

- chunk 从 8192 提到 32768–65536：新增输入少切几块，prefill 打断 decode 的次数更少。
- `enable-prefill-delayer`：按 SGLang 这个开关的设计用意，让各 DP rank 的 prefill 尽量凑在同一步，减少 DP 同步时
  其他 rank 空等。
- `enable-dp-lm-head`、max-running 128、mem 0.85、c32 的 `enable-symm-mem`。

这些参数只在 DP attention 的点上打开，单独贡献多少没有对照数据。

### 3.6 分词：没有绕开 SGLang 的 TokenizerManager

引擎 `skip_tokenizer_init=False`、`tokenizer_worker_num=1`，仍是单个 TokenizerManager；Dynamo 前端另有一份分词器
（日志：`wrapping tokenizer in L1 prefix cache cache_bytes=67108864`）。但每个实例只承载 48 路并发，而我们 3P1D 的 decode
单实例承载 176 路。这和我们的发现一致：单实例的流式输出量决定 TokenizerManager 会不会成为瓶颈，小实例横向扩展
天然避开了它。

## 4. 与我们 3P1D 当前最佳配置的对照

我们的数据见 [`../3p1d-tp4-tp8-v0521-20261012/RESULT.zh-CN.md`](../3p1d-tp4-tp8-v0521-20261012/RESULT.zh-CN.md)：
MI355X、GLM-5.2-MXFP4、20 卡，C176 = 30,562 tok/s/GPU（P50 交互 48.9，TTFT p50 3.1 s）。两边硬件、量化格式、每卡分母
（他们 4 卡没有单独的 decode 卡）、测量协议（他们每点重新部署）都不同，只能比较做法，不能直接比较绝对值。

| 项 | InferenceX B300 | 我们 3P1D | 判断 |
|---|---|---|---|
| 形态 | aggregated，4 卡 TP4 + DP4 attention | PD：3P（TP4/DP4/DPA）+ 1D（TP8/DP8/DPA） | 值得 A/B：小规模 aggregated 实例 |
| chunk | 32768–65536 | prefill 32768 | 已对齐 |
| `enable-prefill-delayer` / `enable-dp-lm-head` | 开 | 未开 | 值得单独试 |
| HiCache | `write_back`、`direct`、`page_first_direct`，270–295 GB/rank | `write_through`、`kernel`，400 / 200 GB/rank | `write_back` 我们已有理论分析（RESULT §4），可做实测 |
| 会话亲和 | Dynamo，DP rank 级 | P 会话亲和（#186），+12.1% | 已对齐 |
| rank 选择 | 没有 KV 事件，只靠亲和，rank 不均 | kv-aware（有 KV 事件）+ R2 按需求选 rank | 我们这块更完整 |
| MTP | EAGLE 5/1/6，模拟接受 3.61 | 相同 | 已对齐 |
| 分词 | 每实例 48 路，1 个 TokenizerManager | decode 176 路，4 个 tokenizer worker | 结论一致 |

## 5. 可以直接跟进的实验

1. 在 prefill（以及 aggregated 实例）上打开 `--enable-prefill-delayer` 和 `--enable-dp-lm-head`，其余不变做 A/B。
2. HiCache `write_back` + `direct` 与现在的 `write_through` + `kernel` 对比。
3. 用同样的卡数，比较“多个 TP4 + DP4 attention 的 aggregated 实例”与 3P1D，并保留 R2 和会话亲和。
   InferenceX 这组的 rank 不均说明 aggregated 形态下按需求选 rank 仍有价值。

## 6. 局限

- 没有复现；§3.2 中“未经过选择的请求由会话亲和直接定位”是从日志计数推断的，日志本身没有逐请求标出亲和命中。
- 输出吞吐和 ITL 基于固定接受长度 3.61。
- 各调度参数只在高并发点打开，没有逐项消融数据，§3.5 的作用说明来自参数设计用意。
- c48 的服务端日志来自该次运行的 attempt 1，被 attempt 2 沿用；指标已与仪表板核对一致（§1）。

## 7. 来源

| 内容 | 链接 / ID |
|---|---|
| 配置 PR | [InferenceX #3953](https://github.com/SemiAnalysisAI/InferenceX/pull/3953)，合入提交 `708cb17b` |
| CI 运行 | [run 38356572105 attempt 2](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38356572105/attempts/2)，PR 分支提交 `caa2199e` |
| c48 服务端日志 | 产物 `multinode_server_logs_…conc48…4ad8e439…`（ID 11798041008，约 41 MB，未入库） |
| 运行时 recipe | [`agg-tp4-variants.yaml@caa2199e`](https://github.com/SemiAnalysisAI/InferenceX/blob/caa2199eb65d4ef05d010f34c88b1c8614a0d125/inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/glm5.2/sglang/b300-fp4/agentx/agg-tp4-variants.yaml) |
| 对照组 recipe / 运行 | [`b300-fp4-mtp/agentic.yaml@67ad5a49`](https://github.com/SemiAnalysisAI/InferenceX/blob/67ad5a4958cbc5932085c449e475a573696bc2e4/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/glm5.2/sglang/b300-fp4-mtp/agentic.yaml)，[run 38021505485](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38021505485/attempts/1) |
| 数据 API | `https://inferencex.semianalysis.com/api/v1/benchmarks?model=GLM-5.2`，2026-10-15 06:15 UTC 获取 |
| InferenceX main（核对时） | `b301e682` |

`evidence/` 内容：

| 文件 | 内容 |
|---|---|
| `api-glm52-b300-rows.json` | API 快照中 B300 的 26 个数据点（原样） |
| `c48/config_c48.yaml` | c48 渲染后的完整配置 |
| `c48/server_args.json` | c48 SGLang 实际生效的全部参数 |
| `c48/routing-summary.txt` | 前端 / worker 日志中路由、会话亲和、KV 事件相关的行和计数 |
| `recipes/` | 运行时 recipe 与对照组 recipe |
