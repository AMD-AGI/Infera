[English](README.md) | 简体中文

# InferaSim — 服务模拟器与投影器

**InferaSim** 无需实际部署配置，就能回答“这种服务配置会有怎样的表现？”。
它可以为某个服务方案投影 TTFT、词元间延迟、吞吐量和 KV 缓存占用，
并在到达驱动的负载下模拟一组引擎——这一切都能在笔记本电脑上完成，
默认路径无需 GPU。

其目的在于降低搜索成本。先在模拟中筛选数千种候选部署，
再只把 GPU 时间花在入围方案上。

核心思想是**稀疏测量，解析迁移**：在单张 GPU 上对一个低成本的小规模锚点
进行基准测试，然后据此投影其他所有方案（TP/EP/PP、批大小、并发数、数据类型），
而不是逐一重新测量。

- [安装](#安装)
- [两分钟了解概念](#两分钟了解概念)
- [快速开始](#快速开始第一次投影)
- [操作指南](#操作指南)
- [环境变量](#环境变量)
- [故障排查](#故障排查)

若要了解各组件的内部工作原理——成本内核、调度器模型、锚点与区间，
以及本工具有意不建模的内容——请参阅
[ARCHITECTURE.zh-CN.md](ARCHITECTURE.zh-CN.md)。

## 安装

```bash
pip install ".[projection]"          # projection + simulation
pip install ".[projection-tuning]"   # + the DSPy tuning agent
```

`torch` 和服务引擎（`vllm`）来自引擎基础镜像，而非 pip。
无 GPU 路径不需要二者。

安装后会提供两个控制台脚本：

| 命令               | 用途                         |
| ------------------ | ---------------------------- |
| `inferasim`        | 投影 + 离散事件模拟          |
| `inferasim-tune`   | LLM 驱动的方案搜索           |

`infera-projection` 和 `infera-tuning` 仍作为别名保留。如果不想安装这些脚本，
下文中的所有命令也可以通过 `python -m infera.projection.cli` 运行。

## 两分钟了解概念

**四项信息描述一次运行。** *模型*（架构预设）、*硬件*（GPU 架构和 HBM 预算）、
*方案*（并行形态、数据类型、并发数——即要搜索的对象）以及*工作负载*
（输入/输出长度、到达模式、前缀复用）。

**两个引擎回答不同的问题。**

| 引擎                         | 回答的问题                                                                | 如何启用                                  |
| ---------------------------- | ------------------------------------------------------------------------- | ----------------------------------------- |
| 解析投影器                   | 稳态均值：TTFT、ITL/TPOT、吞吐量、内存、可行性                            | 默认                                      |
| 离散事件模拟器（DES）        | 分布（p50/p90/p99）、给定负载下的排队、集群行为                            | 添加 `--arrival-model poisson` 或跟踪文件 |

它们共享同一个成本模型，因此为其中一个加载的实测锚点也会被另一个采用。
DES 报告会在解析报告之外额外打印。

**三种保真度来源**，通过 `--profiling-mode` 选择：

| 模式          | 需要 GPU | 含义                                           |
| ------------- | -------- | ---------------------------------------------- |
| `simulate`    | 否       | 解析内核模型（扫描时的默认选项）               |
| `benchmark`   | 是       | 在真实硬件上测量，然后从测量结果投影           |
| `both`        | 是       | 分别运行两种模式并并排报告                     |

`benchmark` 会实际运行模型服务来测量，因此还需要服务引擎和 `--bench-model`
（结构配置指定的是架构，而不是检查点）。它绝不会在无法测量时悄悄降级为
`simulate`：如果无法测量，它会明确说明并停止。

二者之间由**锚点**衔接：锚点是一次低成本实测运行保存下来的产物，
用于校准解析路径。请参阅[使用 GPU 校准（锚点）](#使用-gpu-校准锚点)。

## 快速开始：第一次投影

无需 GPU。

```bash
INFERASIM_MODEL=gpt_oss_120B INFERASIM_TP=2 INFERASIM_EP=2 \
inferasim inference \
  --config infera/projection/examples/exp_pretrain.yaml \
  --inference-mode performance --serving-model continuous \
  --input-len 1024 --output-len 1024 --max-concurrency 32 \
  --gpu-arch mi355x --hbm-capacity-gb 288 \
  --profiling-mode simulate
```

报告末尾如下：

```
[inferasim:Inference] Performance Projection
  Workload: input=1024 tok, output=1024 tok, batch=1
  Serving model: CONTINUOUS BATCHING (concurrency=32)
  Profiling source: SIMULATION
  Max sustainable concurrency: 4808  (HBM=288 GB via --hbm-capacity-gb)
  Concurrency used: 32
  TTFT (time to first token):      42.32 ms
  ITL / TPOT (per token):          14.49 ms
  Interactivity (per user):        69.0 tok/s/user
  Decode step latency (pure):      13.68 ms  | mixed: 38.90 ms
    Mixed-step fraction:           3.12%  → TPOT pollution: 8.4%
  End-to-end request latency:      14870.59 ms
  Per-request decode throughput:   69.0 tok/s
  Aggregate decode throughput:     2209.8 tok/s
  Decode throughput / GPU:         1104.9 tok/s/gpu
  Prefill throughput:              40576.9 tok/s
  Replica GPUs (TP×PP):            2
  Communication breakdown (exposed ms/forward):
    prefill:  TP-AR 3.22 | EP-A2A 14.85 | PP-P2P 0.00 | total 18.07
    decode:   TP-AR 0.00 | EP-A2A 0.01 | PP-P2P 0.00 | total 0.02
```

如何解读：

- **最大可持续并发数**是加载权重后 HBM 中能够容纳的请求数。它是容量上限，
  不是建议值——在该上限运行会使吞吐量最大化，也会严重损害延迟。
- **混合步骤占比 / TPOT 污染**是连续批处理带来的代价：同时承载预填充块和
  解码的步骤会更慢，该指标表示这会将每词元延迟抬高多少。如果将预填充与
  解码解耦，这个数字就会消失。
- **每 GPU 解码吞吐量**是比较 GPU 数量不同的方案时应采用的维度；
  单看聚合吞吐量总会偏向使用更多 GPU 的方案。传入 `--gpu-cost-per-hour`
  可用成本进行同样的比较——请参阅[为投影计价](#为投影计价)。
- **通信明细**显示的是*暴露的*（未被重叠隐藏的）集合通信时间，
  因而在改变并行形态之前，就能看出方案受计算还是通信所限。

## 操作指南

### 选择模型和硬件

`INFERASIM_MODEL` 用于选择从
`configs/models/<framework>/<model>.yaml` 解析出的架构预设（随附 65 个预设，
包括 `gpt_oss_120B`、`deepseek_v3`、`llama3.1_405B`、`kimi_k2`、`qwen3_*`）。
硬件由 `--gpu-arch` 和 `--hbm-capacity-gb` 指定；HBM 数值限定可容纳的内容，
进而决定最大并发数。

```bash
ls infera/projection/configs/models/megatron/     # available presets
```

### 设置服务方案

并行配置来自环境变量，因此扫描可以逐点改变它：

```bash
INFERASIM_TP=8 INFERASIM_EP=8 INFERASIM_PP=1 inferasim inference ...
```

一个副本占用 **TP × PP** 张 GPU。专家并行部署在张量并行 GPU *内部*，
因此提高 EP 不会增加 GPU 数量。数据类型由 `--weight-dtype` 和
`--kv-cache-dtype` 指定。

### 数据并行注意力（MLA）

```bash
--attention-dp-size 8       # split requests across the 8 tensor-parallel ranks
```

注意力以数据并行方式运行，而 MLP 和专家仍采用张量/专家并行：
每个 rank 持有一部分正在处理的请求及其完整 KV 缓存。该大小必须能整除 TP；
它是对 TP 进行细分，而非在 TP 之外增加规模，因此副本的 GPU 数量不变。

这正是 MLA 模型的实际服务方式。张量并行会对 GQA 模型的缓存进行分片，
因为它对 KV 头进行了分片；但 MLA 缓存的是所有头都会读取的单个压缩隐变量，
所以 TP 会复制它——在 TP=8 时，DeepSeek-R1 会将同一份缓存存储八次。
改为按请求拆分后只需存储一次，由此释放的容量完全决定长上下文或智能体场景的
容量规划结果：MI355X 节点上的并发序列数可从 277 提升至 2216。
当某个 rank 的注意力输出属于自身时，也无需执行全归约，因此每层的 TP
集合通信次数也会随之减少。

对于 GQA 模型，这个维度几乎不起作用——TP 已经对各个头完成了分片。

### 控制并发数

这是最常导致结果令人困惑的地方。**`--max-concurrency` 决定运行点。**
如果不设置，投影器会按 HBM 能容纳的最大并发数运行——对于大显存 GPU 上的
小模型，这可能是数千个请求，从而产生巨大的 TTFT 和接近于零的单用户交互速度；
这些看似错误，实际只是系统饱和的表现。

```bash
--max-concurrency 32        # project at 32 in-flight requests
```

`--inference-batch-size` 描述工作负载的批形态，并不限制服务并发数。
如果报告中的 `Concurrency used:` 后跟着一个远大于预期的数字，原因就在这里。

### 为投影计价

向投影器提供 GPU 每小时价格，它就会用服务预算所采用的单位报告吞吐量：

```bash
--gpu-cost-per-hour 2.50
```

```
  Cost basis:                      $2.5/GPU-h x 2 GPU
  Cost / 1M output tokens:         $0.629
  Cost / 1M in+out tokens:         $0.314
```

计费覆盖整个副本，因此通过增加 GPU 来换取吞吐量的方案也必须为这些 GPU
付费。这正是该指标的意义：tokens/s 按速度为方案排序，但更快的方案不一定
更便宜。`in+out` 一行按工作负载自身的输入输出比例混合计算——在
3072 输入 / 1024 输出的情况下，每生成一个词元，就有四个词元参与计费，
所以该数值是纯输出数值的四分之一。（上例中输入 1024 / 输出 1024，
因此是二分之一。）

没有价格就不会计价。省略该参数后，成本行会直接缺席，而不会采用默认值，
因为虚构的 GPU 每小时价格可能会被误认为实测价格。

### 使用 GPU 校准（锚点）

在任何装有服务引擎的 ROCm 主机上采集一次即可。`--model` 是要运行服务的
检查点，因此应为 HF id 或本地路径，而不是预设名称：

```bash
inferasim anchor --model openai/gpt-oss-120b --benchmark-gpus 1 --save anchor.json
```

`--serving-backend {vllm,sglang,atom}` 用于选择引擎。引擎会通过平台实际服务时
所用的同一组适配器启动，因此锚点描述的是部署状态下的引擎，而不是测试工具
碰巧采用的启动方式。锚点覆盖一个引擎：跨副本路由由模拟完成，而非实测，
因此通过路由器测量会将该层重复计算两次。

之后即可在无 GPU 的环境中据此投影任意方案：

```bash
inferasim inference ... --load-benchmark anchor.json
```

如果希望一步完成测量和投影，而不是先保存锚点，可让投影本身执行测量。
这使用的是同一个测试工具，只是由配置驱动：

```bash
inferasim inference ... --profiling-mode benchmark \
  --bench-model openai/gpt-oss-120b --save-benchmark anchor.json
```

涉及多个方案时，建议采用两步形式：测量是目前成本最高的环节，
而 `--load-benchmark` 可以让共享同一区间的所有投影复用一次测量。

一次运行覆盖的不只是请求的单点：它会覆盖直到 `--max-concurrency` 的
CUDA 图捕获阶梯，因为批大小是投影迁移所沿用的维度，而单个测量点在该维度上
只能保持为常数。随后查找解码数据时，会把批大小填充到最近的已测大小，
与引擎填充到最近的已捕获大小的做法一致。

服务式锚点使用 vLLM 的默认形态作为该阶梯，因为引擎在另一个进程中运行，
无法从这里读取其真实捕获列表；默认启动会捕获默认阶梯。离线锚点（`--offline`）
则会从构建好的引擎读取该列表，还会测量不同上下文下的解码，从而让投影器能够
拟合注意力 KV 项，而不是假设解码时间不随上下文变化。无论使用哪条路径，
通过 `--batches` 明确指定批大小都会覆盖该阶梯。

测量只校准*延迟*。内存投影始终采用解析方法——权重、KV 和激活工作集都根据
模型形态和并行布局计算，`--profiling-mode benchmark` 不会改变任何内存数值。

锚点 JSON 与引擎无关（包含一个 `"backend"` 字段以及按批大小记录的
解码/预填充测量值），因此可以添加不同的采集器而无需改动投影器。

**一次预热会在多少张 GPU 上测量。** 单 GPU 锚点无法观测跨 GPU 通信
（TP 全归约、EP 全对全），因此预热需要不止一张 GPU——但只需增加到一定程度。
规则只有一行：`min(tp, 4)`，并向下调整为能整除 `tp` 的并行度。
这里没有扫描，也没有需要逐级攀升的阶梯；只测量一个锚点，投影器会据此还原
所有目标配置。

四张 GPU 是实测结论，而非假设。将 TP1/2/4/8 扫描中的每个并行度分别作为
其*未测量*并行度的锚点进行评分后，四 GPU 锚点以 6.6% 的误差成为最佳单一选择，
优于整节点八 GPU 锚点的 7.6%；这是因为它可以双向插值，而端点只能外推。
第二个锚点在每个目标上都只能打平或不如它，因此多锚点路径
（`--load-benchmark-scaling`）是应急手段，而非默认路径。上限设为四也意味着
预热永远无需等待完整节点。

### 模拟模型前缀复用

智能体和多轮流量会共享长前缀——系统提示、工具模式定义、对话历史——
引擎通过自动前缀缓存让这些内容常驻。在解析路径上，可直接声明复用率：

```bash
inferasim inference ... --input-len 4096 --prefix-cache-hit-rate 0.8
```

缓存前缀（`R × input_len` 个词元）会跳过预填充计算；剩余后缀仍会关注完整上下文。
因此，**TTFT 和连续批处理污染中的预填充占比会随 `(1 - R)` 缩放**，
而解码和 KV 容量规划不变。`R = 0` 表示冷缓存。始终至少会预填充一个词元，
所以 `R` 会被限制在小于 1 的范围内。

这在解析模式和锚点校准模式下都适用。可在 YAML 的 `inference:` 块中按工作负载
设置（`prefix_cache_hit_rate: 0.8`），使搜索中的每个方案都在相同假设下评分。

对于由工作负载内容自然*产生*而非直接指定的复用，请使用下文的 DES 块缓存。

### 模拟负载并获取百分位数

解析路径给出固定并发数下的均值。若要模拟请求按各自时间表到达、排队和尾延迟，
请运行 DES：

```bash
inferasim inference ... \
  --request-rate 1.5 --arrival-model poisson --des-num-requests 120
```

```
[inferasim:Inference] Discrete-Event Simulation (arrival-driven)
  Arrivals: poisson @ 1.5 req/s offered  (achieved 1.20 req/s, utilization 99%)
  Simulated: 120 requests over 100.20 s  → system throughput 1226 tok/s
  metric                        mean         p50         p90         p99
  TTFT (from admit)          31.22 ms     31.49 ms     32.48 ms     35.72 ms
    queue wait                4.59 ms      4.42 ms      7.85 ms     21.02 ms
    TTFT (from arrival)      35.80 ms     35.31 ms     39.89 ms     53.52 ms
  TPOT (per token)            9.11 ms      9.42 ms     10.25 ms     10.40 ms
  ITL (inter-token)           9.14 ms      9.30 ms     10.29 ms     29.94 ms
  End-to-end latency       9360.36 ms   9669.08 ms  10526.00 ms  10673.39 ms
  Batch packing: avg batch 10.5 (max 19) | avg prefill/decode reqs 0.0/10.5 | ...
```

如何解读：

- **两行 TTFT。** *从准入开始*表示请求开始运行后的引擎行为；
  *从到达开始*还会加上排队等待时间，这才是客户端实际感受到的延迟。
  在高负载下二者会显著分化；把前者当作后者引用，是让饱和系统显得健康的
  典型方式。
- 到达信息行中的 **`[SATURATED]`** 表示给定负载超出容量。
  此时延迟反映的是无界增长的队列，不能作为有意义的运行点——应将
  `--request-rate` 降至报告的最大可持续速率以下。
- **批组装**显示调度器实际组装出的内容，可据此判断延迟问题源于批处理
  还是排队。

到达选项：`--arrival-model` 可取 `closed`（无队列，仅稳态）、`poisson`
或 `deterministic`；`--des-burstiness` 使到达服从伽马分布（1.0 = Poisson，
值越低突发性越强）；`--des-range-ratio` 让每个请求的长度在配置的 ISL/OSL
附近分散。添加 `--des-sweep` 可扫描给定负载并输出吞吐量—延迟曲线，
而不是单个点。

### 模拟集群

在路由器后放置多个引擎副本，每个副本都带有真实的内容寻址 KV 块缓存：

```bash
inferasim inference ... \
  --request-rate 6 --arrival-model poisson --des-num-requests 200 \
  --des-instances 4 --des-routing kv \
  --des-num-prefixes 8 --des-prefix-len 2048 --des-block-size 512
```

```
  Fleet: 4 instance(s), routing=kv | prefix pool: 8 prefixes
  KV block cache: block_size=512 tok, capacity/instance=unbounded, evictions=0, block-reuse=44.8%
  Prefix-cache hit rate: 89.5% of requests (avg 1833 cached tok/req; per-instance 88–90%)
```

一个提示是由块哈希 id 构成的有序序列，命中是指已经驻留的块中最长的
*连续前导序列*。缓存容量有限，并会在压力下按 LRU 淘汰，因此**命中率是内容、
容量和路由共同作用后自然产生的属性**，而不是由用户直接提供的数值。

路由策略（`--des-routing`）：

| 策略                         | 行为                                                                                                                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kv`                         | 服务路由器自身的策略：最小化 `--des-overlap-weight` × 未命中块数 + 副本负载。该权重用于权衡复用与均衡（0 = 纯负载均衡）。                               |
| `prefix_aware`               | 对前导块进行一致性哈希并映射到主副本，使相同前缀的请求位于同一处。未命中数 ≈ 前缀数，与集群大小无关。                                                  |
| `round_robin` / `random`     | 忽略局部性，因此每个副本都会重新预热每个前缀。未命中数 ≈ 前缀数 × 副本数。                                                                             |

这种权衡在两个方向上都真实存在：追求局部性的策略可以最大化复用，但也可能将一个
热门前缀过度集中到某个副本，提高其解码压力并延长完工时间。报告会同时显示汇总延迟
和各副本命中率的分布范围，以便观察权衡的两面。由于 `kv` 的评分函数与部署路由器
相同，扫描 `--des-overlap-weight` 可以在生产环境实际调整该权重前，
投影调整会带来的成本。

设置 `--des-instances 1` 可将单个引擎的自动前缀缓存作为流中的时间复用来研究。
添加 `--des-kv-blocks N` 可限制每个副本的容量并研究淘汰压力；
`--des-prefix-zipf` 会使前缀热门程度呈偏斜分布，以模拟少数热门系统提示
主导真实流量的情况。

### 重放真实跟踪

```bash
inferasim inference ... \
  --des-mooncake-trace trace.jsonl \
  --des-instances 4 --des-routing kv --des-block-size 512
```

Mooncake 跟踪是 JSONL/JSON，其中包含 `timestamp`（毫秒）、`input_length`、
`output_length` 和 `hash_ids`（块哈希序列）。共享系统提示的请求会共享前导
`hash_ids`，从而驱动真正的内容寻址复用，而不是使用假定的复用率。
跟踪会提供自己的到达时间，因此不需要 `--request-rate`。

`--des-workload-file` 可重放不含哈希 id 的简单 JSON/CSV 工作负载。

### 模拟解耦式服务

为预填充和解码分别设置资源池及并行形态，并计入二者之间的 KV 传输成本：

```bash
inferasim inference ... --disaggregate \
  --prefill-tp 4 --prefill-ep 4 --prefill-replicas 1 \
  --decode-tp 8  --decode-ep 8  --decode-replicas 2 \
  --kv-transfer-bw-gbps 400
```

在这里，`--decode-admission-steps` 最为重要：预填充离开关键路径后，
TTFT 中仍然可见的是已完成预填充的请求等待加入解码批次的时长。

### 扫描配置空间

在 Python 中进行脚本化搜索：

```python
from infera.projection.core.projection.inference_projection.sweep import sweep

res = sweep(
    "gpt_oss_120B",
    tp=[1, 2, 4, 8], ep=[1, 2, 4, 8], pp=[1],
    concurrency=[1, 8, 32, 128],
    isl=1024, osl=1024,
    gpu_arch="mi355x", hbm_gb=288.0,
    valid=lambda tp, ep, pp: ep <= tp,      # your own legality rules
)

for p in res.points:
    if p.feasible:
        print(p.tp, p.ep, p.concurrency, round(p.ttft_ms, 1),
              round(p.tpot_ms, 2), round(p.decode_tps_per_gpu, 1))
```

扫描会强制使用 `--profiling-mode simulate`，因此需要 **零张 GPU**。
不可行的点会被*保留*并通过 `p.reason` 标注，而不是被丢弃，
这样就能区分“无法容纳”与“从未尝试”。传入 `workload=` 可根据自己的
实验配置进行投影，而不是使用随附的默认配置。

### 使用调优代理搜索

```bash
inferasim-tune --workload <workload.yaml> --target-cluster <cluster.yaml>
```

分为两个阶段：先进行确定性的种子扫描以实现热启动，再由 LLM 驱动的搜索
从热启动后的当前最优方案继续，提出方案并通过投影器评分。默认路径无需 GPU。
使用 `--seed-only` 可只运行确定性阶段。配置方法请参阅
`agents/tuning_agent/`。

## 环境变量

所有环境变量均以 `INFERASIM_*` 开头。实际会设置的变量如下：

| 变量                                                            | 含义                                      |
| --------------------------------------------------------------- | ----------------------------------------- |
| `INFERASIM_MODEL`                                               | 架构预设（例如 `gpt_oss_120B`）           |
| `INFERASIM_TP` / `_EP` / `_PP` / `_CP` / `_VP`                  | 并行形态                                  |
| `INFERASIM_GPU_ARCH`                                            | 目标 GPU 架构                             |
| `INFERASIM_ROOT`                                                | 仓库/配置根目录覆盖值                     |
| `INFERASIM_ANCHOR_STORE`                                        | 实测锚点目录                              |
| `INFERASIM_SEQ_LENGTH`                                          | 默认序列长度                              |
| `INFERASIM_TEAM` / `_USER` / `_EXP_NAME` / `_WORKSPACE`         | 启动器身份字段                            |

还有更多 `INFERASIM_*` 变量用于内核模型和基准测试内部配置
（`INFERASIM_GEMM_BACKEND`、`INFERASIM_BENCH_*`、`INFERASIM_MOE_*`、
`INFERASIM_DEBUG_*` 等）。它们是高级覆盖项，其读取位置附有文档说明。

## 故障排查

**并发数远高于我的要求。** 请传入 `--max-concurrency`。
`--inference-batch-size` 不会限制服务并发数；请参阅
[控制并发数](#控制并发数)。

**所有扫描点都不可行。** 请检查 `p.reason`。常见原因是该并行形态下
HBM 预算不足以容纳权重，或者 `valid=` 谓词拒绝了所有配置。

**DES 显示 `[SATURATED]`。** 给定负载超出容量，因此队列会无界增长，
延迟百分位数描述的是积压，而不是运行点。请将 `--request-rate` 降至报告的
最大可持续速率以下，或通过 `--des-instances` 添加副本。

**DES 中的前缀命中率为 0%。** 请求之间没有共享内容。请提供
`--des-num-prefixes`/`--des-prefix-len`，或提供包含 `hash_ids` 的跟踪文件；
`--prefix-cache-hit-rate` 是解析模型参数，不会驱动块缓存。

**更改数据类型或后端后，锚点结果看起来不正确。** 锚点只对自身的执行区间
提供保证。更改数据类型、内核后端或图模式后，之前的锚点便无法继续迁移——
请重新采集。

## 仓库布局

| 路径                                    | 内容                                      |
| --------------------------------------- | ----------------------------------------- |
| `cli.py`                                | `inferasim` 入口（投影 + DES）            |
| `core/`, `modules/`, `platforms/`       | 投影引擎和模拟器                          |
| `agents/tuning_agent/`                  | 方案搜索代理（`inferasim-tune`）          |
| `configs/`                              | 运行时解析的模型和预设树                  |
| `examples/`                             | 示例实验配置和工作负载                    |
| `ARCHITECTURE.md`                       | 工作原理以及未建模的内容                  |

其中有意**不包含** Megatron/训练闭包；推理路径永远不需要它。
