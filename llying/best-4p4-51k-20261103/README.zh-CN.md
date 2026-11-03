# GLM-5.2 AgentX PD 4+4 最高吞吐 51k：配置、性能、遗留问题与后续建议（11-03）

整理自 `../prefill-opt-20261027/`（README、`results/`、`config/`）、yihou 的 `../../yihou/glm52.llrepro-c104-tail.packup_20261028/OPT_LEDGER.md`、
`../2p1d-v0522-20261020/`、`../prefill-car-20261023/`，以及 138 上 11-02 night30–34 的队列日志（`$L/night3{0,1,3,4}-1102.out`，`prefill-opt` 的 README 还没收录）。
第一至五节是 11-03 03:00 的版本，没有新跑实验；第五节的数据是从已有运行日志和镜像里读出来的。第六节是 11-03 03:30 起的补充问答；
正确性修复实验在 141 上进行，记录在 `../car-copyin-141-20261103/`。

## 结论

1. **最高吞吐是 E10**：yihou 的 llr35 栈 + 3 个 prefill 补丁（v3、E1、E10），AgentX C104、计分 3600 s。
   138→140 上 **51,374 tok/s/GPU**，decode 换到 135 后 51,658；比同节点 v3 基线 48,360 高 6.2%。
2. **瓶颈已经从 prefill 转到 decode**（decode 每 token 时间和 decode KV 池容量）。
3. **正确性**：低频"开头坏"仍在最佳配置里。11-02 关掉 decode custom all-reduce（CAR）后三次限池门禁都是 0 乱码、0 开头坏，CAR 与它有关。
   `prefill-opt` README 11-02 09:50 节"现有证据不支持 CAR 是原因"已过时。
4. **建议**（第四节）：
   - 正确性应该修，先试 10-23 已验证过的 all-reduce 捕获拷入补丁（保留 CAR）；同一类修法也可能让 E11（MoE 合并改 reduce-scatter）恢复正确。
   - 4+4 现栈近期的优化空间约 +4–8%（约 53.5–55.5k），主要在 decode；先做通信、调度、DP 均衡这些便宜项，单个 kernel 的收益多在 1–2%。
   - PD 配比：4+4 两侧都接近饱和，往任一侧加卡都不划算。唯一值得测的是 2P1D（decode TP8/DP8），靠每卡 MoE 权重减半、每卡 KV 多 32% 提高效率，
     估计 0 到 +10%，排在 CAR 修复之后。
5. **11-03 更新**（第六节、`../car-copyin-141-20261103/`）：
   - **E11 乱码的根因找到并修好**：每个 DP rank 按自己的 batch 选 dense / sparse decode graph，同一步不同 rank 回放不同的 graph，
     注册了 graph 池输入的 aiter collective 读到对端另一套 graph 的旧缓冲。拷入（26 / 200 → 0 / 600）或把所有 rank 固定到 sparse（→ 0 / 200）都能修好。
   - E11 的乱码里有和"开头坏"同样的形状，E10 的"开头坏"很可能是同一机制的低频版；141 上 E10 对照本身不出现，要跨机验证，脚本已备好未启动。
   - 固定 sparse 还去掉了 decode 每步最大的一处同步（`select()` 里的 `.item()`），预计省 1–2 ms / 步（6.5 节）。
   - 关 CAR 实测只掉 1.4%（50,670 对 51,374，另一会话 night35 测），比我原先估的 2–2.5% 小，可作正确配置兜底。
   - dense 层在线 FP8 的改动量和收益估计见 6.6 节：核心改动约几十行，预计 +2–4%（用户 11-03：放到后面）。
6. **11-03 08:26 起：通用修法与 138 / 140 全面验证**（`../dp-graph-variant-fix-20261103/`，用户要求不用"固定 sparse"这种定制做法）：
   - 读代码发现 dense / sparse 只在 DECODE 模式下改变计算，开 MTP 时两套 verify graph 完全相同。修法分两部分：
     (1) TARGET_VERIFY 只捕获一套 graph；(2) DECODE 模式下各 rank 通过 DP attention 已有的 MLP sync 表对齐长度上界，用同一个 variant，不读设备。
   - 补丁、单元测试、验证队列（对照 / 修法交替的限池门禁、E11 与不开 MTP 的正对照、审计、生产全套 + lm-eval、计分）和开跑前写定的判据都已备好。
   - 08:38 起登录节点到 135–141 全部不可达（之后主机名也解析不了），队列由登录节点上的等待进程在两台机器恢复且空闲后自动启动。

## 一、配置

公共部分：模型 GLM-5.2-MXFP4，KV cache fp8_e4m3；镜像 `infera-sglang:v0522-fdbafb60`（sha `509b8427`，ROCm 7.2）；
Infera router + etcd，PD 之间 Mooncake RDMA（ionic_0–7）；共 8 张 MI355X。负载是 AgentX trace 回放，104 个并发 lane，计分 3600 s。

| 项 | Prefill（crsuse2-m2m-138，GPU 4–7） | Decode（crsuse2-m2m-140，GPU 4–7） |
|---|---|---|
| 并行 | TP1 × PP4，层切分 20/20/20/18 | TP4 + DP4 attention，EP1 |
| 显存 | mem-fraction 0.88 | mem-fraction 0.95 |
| 调度 | chunk 8192；shortest-prefill-first；`--pp-async-batch-depth 1`；`SGLANG_PP_COMM_OVERLAP=1` | cuda graph 最大 bs 256；stream-interval 30；custom all-reduce 开 |
| 缓存 | HiCache write_back，每级主机层 250 GB，kernel IO，page_first（在 InferenceX DRAM 预算内，约占 82%） | — |
| MTP | EAGLE 3 步 / topk 1 / 4 个 draft token（prefill 也要带，才能传 draft KV） | 同左；计分用模拟接受长度 2.99 |
| kernel | DSA prefill 用 triton | yihou 的 KF overlay（sparse MLA / indexer / MoE）、FP8 draft MoE（`SGLANG_GLM_NEXTN_MOE_PTPC=1`）、调优过的 GEMM CSV |

prefill 上的 3 个补丁（decode 侧只挂 overlay，不改源码）：

| 补丁 | 改动 | 作用 |
|---|---|---|
| v3 `pp-chunk-wait` | PP 下发送中间块 KV 前，先在 forward stream 上记 event | 修复开 async、PP 通信重叠时的长上下文乱码，性能不变（48,360） |
| E1 `evict-headroom` | write_back 驱逐每次至少腾出 512K token | 驱逐从每步 1.16 次降到 0.11 次，+2.8% |
| E10 `hicache-backup-quota` | 驱逐写回的 D2H kernel 从 2 个 workgroup 改为 16 个（约 25 → 49 GB/s） | 每次驱逐耗时 −28%，在 E1 上再 +3.3% |

配置入口是 `../prefill-opt-20261027/config/site-llr35-v3-evict512k-bq16.env`，它依次 source `site-llr35-4p4.env`、`site-llr35-v3.env`、`site-llr35-v3-evict512k.env`；
kit 是 `2p1d-kit/llr35`（`engine/worker.sh` 里写死了 HiCache、并行方式、MTP 等参数）。复现（138 上，kit 目录下）：

```bash
P=/home/liyingli/bench_agentx/baseline/Infera/llying
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
L=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/llr
cd $KIT && bash $P/prefill-car-20261023/scripts/perf_plan.sh $KIT $L/perf $P/prefill-opt-20261027/config/plan-night20-1029.txt
cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh llr37-v3-evict512k-bq16 104
```

## 二、性能

演进：yihou llr35 C96 47.8–48.0k → C104 48.2–48.4k → v3 在 138/140 上 48,360 → E1 两次平均 49,719 → **E10 51,374**。

| 指标 | **C104（最高吞吐点）** | C72（11-02 测的 Pareto 点） |
|---|---|---|
| tok/s/GPU | **51,374** | 43,627 |
| 输出 tok/s/GPU | 424.1 | 341.4 |
| 请求/s；完成 / 丢弃 | 3.397；12,325 / 2 | 2.737；9,929 / 4 |
| 每用户 tok/s p50 / p90 | 63.6 / 56.1 | 73.4 / 62.6 |
| ITL p50 / p90 | 15.7 / 17.8 ms | 13.6 / 16.0 ms |
| TTFT p50 / p90 / 均值 | 4.34 / 10.42 / 5.95 s | 2.23 / 4.38 / 2.55 s |
| E2E p90 | 42.5 s | 33.0 s |
| 前缀缓存命中 | 96.0% | 96.6% |

结果文件：`../prefill-opt-20261027/results/run-llr37-v3-evict512k-bq16-c104.txt`、`run-llr37-135-e10-c104.txt`、`run-llr37-140-e10-c72-c72.txt`。

C104 下的内部状态：
- **Prefill**：满块节拍中位 312 ms、p90 683 ms。59% 的时间有排队，新算约 16.3k tok/s。每小时驱逐 1,239 次，四级平均耗时 358 / 195 / 259 / 225 ms，主机层 21.7M token 用满。
- **Decode**：步长约 44 ms，接受长度 2.99，折合每个输出 token 15.3 ms。同时在跑 53 个请求，最忙的 rank 有 17 个（p90 26 个）。
  平均 4.2 个请求在等 KV 预分配，15.9 个在等 KV 传输。

## 三、仍存在的问题

### 正确性

1. **"开头坏"/乱码 bug 在最佳配置里仍然存在**。表现是第一个 token 对，第二个起就乱。
   - 11-02 E10 生产配置（不限池）的 200 题探测：正确率 0.960，乱码 6 条，开头坏 5 条；历史上 1,319 题里有 4–6 条，约 0.4%。
   - 关掉 decode CAR 后，连续 3 次限池门禁都是 0 乱码、0 开头坏；开着 CAR 的 8 次门禁里，有 6 次出现 3–8 条。
     按事先定的判据，可以认定它与 CAR 有关（巧合概率约 1.6%），但根因还没定位（数据见第五节）。
   - 关 CAR 的代价：yihou 测过 decode 步长 +7%（llr45，40.5 → 43.4 ms）。我按报告的弹性粗估吞吐会掉 2–2.5%；
     **11-03 实测（另一会话 night35，138 → 140）是 50,670，−1.4%**，最忙 rank 17 个请求时步长 45 → 46 ms。
2. E1 和 E10 的正确性证据只来自限池门禁。生产配置的正确性测试触发不到驱逐，覆盖不到这两条改动的代码路径。
3. 计分用的是模拟接受长度 2.99，分数本身不反映输出内容是否正确，正确性要另外用真实接受率测。

### 性能瓶颈

1. **服务端能优化的空间有限**。每个 lane 的时间大约这样分：50% 是 trace 自带的工具/思考间隔（服务端改变不了），34% 是 decode，16% 是 TTFT。
   所以 E9（后台预写回）让 TTFT 降了 14%，吞吐却只涨 0.7%，已暂停。
2. **decode 每 token 时间是主要瓶颈**。步长 ≈ 25.8 ms 固定开销 + 1.09 ms × 最忙 rank 的请求数。verify 部分的拆解（C104 负载下 55.6 ms 的一步）：
   MoE GEMM 17 ms（31%，只用到 HBM 带宽峰值的 65%）、dense GEMM 8 ms、通信 8 ms、sparse MLA 7 ms、indexer 3.6 ms、draft 4.7 ms、GPU 空闲 3 ms。
3. **通信优化被正确性问题卡住**。E11（`SGLANG_USE_ROCM700A=0`，MoE 合并改走 aiter reduce-scatter）预计每步省 1.3–2.4 ms，但实测乱码：探测正确率 0.670、乱码 44 条。
4. **DP 不均衡**。最忙 rank 的请求数是平均的 1.31 倍，完全均衡的理论上限约省 10% 步长（吞吐约 +3%），但 KV 池常满，做均衡有风险。
5. **decode KV 池总容量不够**。每个 rank 使用率均值 0.82，25% 的时间 ≥ 0.97，预分配占池的 17%；瓶颈是总容量，不是 rank 之间不均。
   这也是并发加不上去的原因：yihou 测过 C108 −2.2%、C112 −6.3%。不过这个 C104 峰值是在 E1/E10 之前测的，在 E10 上没有重新扫。
6. **prefill 仍有余量但收益小**。prefill GPU 只忙约 50%，每次驱逐里约 46% 的时间仍在同步写回；E9 把驱逐耗时降了 80%，端到端看不到收益。
7. **测量噪声**。单次测量波动约 ±1%，节点之间差约 3%，很多结论都是单次运行得出的。

### 工程与落地

1. **补丁都还没进上游**。最佳配置依赖 3 个挂载进容器的 prefill 补丁，加上 kit 里的 `deepseek_v2.py` 修改和 decode 侧的 KF overlay。
   InferenceX 维护者已在 [#4629](https://github.com/SemiAnalysisAI/InferenceX/pull/4629) 拒绝给引擎打补丁，必须先合进 SGLang。
   E1 和 E10 的 PR 草稿在 `../upstream-prs-20261102/`，还没提交（E1 加了 `is_write_back` 条件的版本没跑过）；v3 对应上游还开着的 [sglang#45524](https://github.com/sgl-project/sglang/pull/45524)，需要逐行比对。
2. **和 CI recipe 差距大**。按 InferenceX 流程在 AUS 上跑 #4629 的 4+4 C80 是 37,346 tok/s/GPU；51k 这套栈只在 138/140 的 kit 上跑过。
3. **Mooncake RDMA 内存注册偶发失败**（140 上 1 次，AUS 上 2 次 ENOMEM），导致起栈失败，根因未知。
4. **HiCache 容量受 InferenceX DRAM 预算限制**，只能用每级 250 GB；HC500 + E1 能到 50,574，但超预算。

## 四、后续建议

### 4.1 正确性：应该修，先试 all-reduce 捕获拷入

理由：
- 51k 这个分数是在模拟接受率下测的，真实接受率下这套栈会给 0.4%–3% 的请求输出乱码。GSM8K 只抽最后的数字，看不出这类错误
  （51k 栈的 1,319 题探测正确率 0.963，与 AUS 上 #4629 配置的 eval 0.9727 同一水平）；但带着已知乱码的配置去发布有风险，真实部署也不能接受。
- 修复很可能不贵。10-23 在 mixed 部署里查清过同一类问题：aiter custom all-reduce 在 CUDA graph 捕获时注册 graph 池里的输入张量，对端回放时读到旧数据；
  改成捕获时拷进预注册的 IPC 池就修好了，CAR 保留（`../prefill-car-20261023/README.zh-CN.md`，V1 乱码 0）。补丁是 `../prefill-car-20261023/patches/custom_all_reduce.py`
  （`SGLANG_AITER_AR_CAPTURE_COPY_IN=1`），**E10 的 decode 没有带它**。镜像里该文件与补丁基线逐字节一致（第五节），可以直接挂载。
- 镜像源码（`parallel_state.py:87-91`）对 all-gather / reduce-scatter 的同一条路径写明："the registered path makes peers read stale data over IPC on replay
  (garbage rows / NaN under DP attention + TP MoE on MI355X)"，开关 `SGLANG_AITER_CAPTURE_COPY_IN` 默认关。E11 改成 MAX_LEN 后，graph 内的 all-gather / reduce-scatter
  正好走这条注册路径，乱码与注释描述的情形一致。所以同一类修法可能同时解锁 E11（+1–1.5%）。
- 尚未解释的一点：PD decode 里 PREBUILT 不进 forward、IDLE 也走 graph（`decode_schedule_batch_mixin.py:31`、`forward_batch_info.py:274`），
  不一定有 mixed 那种 eager 步与 graph 回放交替；为什么偏偏坏在新请求的第一个 decode 步，还没有机制解释。所以这是"最便宜的直接检验"，不是已确认的根因。

实验顺序（每步先写预测、按门禁判定，与现有方法一致）：

| 步 | 内容 | 机时 | 判定 |
|---|---|---|---|
| 1 | E10 + decode AR 捕获拷入（CAR 开），限池门禁 × 3（同 night31 / 33） | 约 1.3 h | 3 次开头坏合计 ≤ 1 视为修好；否则转第 4 步 |
| 2 | 同配置 C104 计分 1 次 | 约 1.5 h | 拷入每层多一次小拷贝，预计 < 1%；与 E10 的 51,374 比 |
| 3 | E11 + `SGLANG_AITER_CAPTURE_COPY_IN=1` + AR 拷入：smoke + 200 题探测 + 16k × 32 | 约 0.5 h | 探测 ≥ 0.97、乱码 0；通过后 C104 计分（预测 ≥ 52,000） |
| 4 | 若第 1 步不成立：E10 关 CAR 的 C104 计分 1 次，量出代价 | 约 1.5 h | 用来决定是否以关 CAR 作为发布配置（11-03 已由 night35 测出：−1.4%） |

站点配置写法（叠在 E10 上）：

```bash
. /home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027/config/site-llr35-v3-evict512k-bq16.env
CA=/sgl-workspace/sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py
export DECODE_MOUNT_EXTRA="$DECODE_MOUNT_EXTRA /home/liyingli/bench_agentx/baseline/Infera/llying/prefill-car-20261023/patches/custom_all_reduce.py:$CA:ro"
export DECODE_ENV_EXTRA="$DECODE_ENV_EXTRA SGLANG_AITER_AR_CAPTURE_COPY_IN=1"
unset CA
```

### 4.2 优化空间与 decode kernel

上限：服务端时间（decode 34% + TTFT 16%）只占 lane 的一半。C104 下即使 decode 和 TTFT 都减半，吞吐也只有约 +33%，现实做不到。
报告实测的弹性是 decode 每快 10% ≈ 吞吐 +3.4%；TTFT 的弹性更低（E9：TTFT −14% 只换来 +0.7%，因为 2/3 的请求是并行子 agent，不在关键路径上）。

decode 侧各方向（每步占比按 C104 负载下 55.6 ms 的 trace，端到端按 0.34 折算）：

| 方向 | 现状 | 每步可省 | 端到端 | 前提 / 代价 |
|---|---|---|---|---|
| E11：MoE 合并改 aiter reduce-scatter | 通信 8 ms；微基准每层省 17–30 µs | 1.3–2.4 ms（3–4%） | +1–1.5% | 先修 graph 内乱码（4.1 第 3 步） |
| graph 之间的 GPU 空闲 | 3 ms（CPU 准备、D2H 同步、偶发 39 ms 的 zmq 停顿） | 1–2 ms（2–4%） | +0.7–1.3% | 调度层改动，不是 kernel |
| decode DP 均衡 | 最忙 rank 是均值的 1.31 倍；完全均衡上限 −10% | 现实约 3–5% | +1–1.7% | KV 池常满，按请求数均衡有风险 |
| MoE GEMM 带宽利用率 65% → 80% | 17 ms，带宽受限，与 batch 基本无关 | 约 3 ms（5%） | +1.7% | KF kernel 已深度调过，难度高 |
| sparse MLA + indexer | 10.6 ms | 10–20%，即 1–2 ms | +0.7–1.3% | kernel 工作 |
| dense GEMM BF16 → FP8 | 8 ms；o_proj 每层约 45 µs，按 BF16 权重推算约 4.5 TB/s，接近带宽上限 | 约 4 ms（7%，推算） | +2.4% | **改精度**，按规则只测不采用，由用户决定；InferenceX 上的 ATOM recipe 已经这么做（6.3 节） |
| prefill 侧（E9、关 KV events 等） | TTFT 只占 lane 16% | — | 各 < 1% | E9 已暂停 |

判断：
- 不改精度的几项都做成，decode 步长约 −10~15%，端到端约 +3.5~5%；decode 变快后 KV 释放更快，并发峰值可能上移一点。合计约 **+4~8%（53.5–55.5k）**，收益递减明显。
- **decode 值得继续优化，但先做便宜的非 kernel 项**：拷入修复带出的 E11 通信优化、graph 之间的空闲、DP 均衡，每项 1–1.5%，改动小。
- **纯 kernel 工作放在后面**：MoE GEMM、sparse MLA、indexer 每项 1–2%，KF 已经调过，投入产出比低。
- 每卡 MoE 开销最大的杠杆不是 kernel，而是并行方式：MoE 每步读全部 257 个专家（TP4 下每卡约 91 GB 的切分权重，与 17 ms × 5.3 TB/s 对得上），
  decode 改 TP8 后每卡只读一半。见 4.3。

### 4.3 PD 配比

现在的 4+4 两侧都接近饱和：prefill 59% 的时间有排队，decode KV 池 25% 的时间 ≥ 0.97、平均 4.2 个请求等预分配。
往任一侧加卡，另一侧马上成为瓶颈，而每卡吞吐的分母变大：
- **多 decode（如 4+8，12 卡）**：要每卡不降，总吞吐需要涨 50%，单个 prefill 撑不住。#210 栈上 4+8 C88 均衡是 31,088 / GPU，4+4 C80 是 37,310。
- **多 prefill（如 3P1D、8+4）**：decode 已是主瓶颈，只会更差。

历史上 2P1D C176 是最高吞吐点，但只比 4+4 高 3%（同一 #210 栈，K5、SIM 3.61、无 yihou 选项）：

| 配置（#210 栈） | 卡 | tok/s/GPU | 每用户 tok/s p90 |
|---|---|---|---|
| 2P1D C176（2 × PP4 prefill + TP8/DP8 decode） | 16 | 38,392 | 43.2 |
| 4+4 C80 | 8 | 37,310 | 44.3 |
| 4+8 C88 均衡 | 12 | 31,088 | 71.7 |
| **对照：现在的 4+4 E10 C104** | 8 | **51,374** | **56.1** |

现在的 4+4 在吞吐和交互性上都支配旧的 2P1D C176。但 2P1D 的 decode 是 TP8/DP8，在当前瓶颈下有两个结构性优势（第五节数据）：
1. **每卡 MoE 权重减半**：TP4 每卡权重 120 GB，TP8 是 74 GB；两者联立，按 TP 切分的权重共约 365 GB（主体是专家），每卡复制的部分约 29 GB。
   MoE GEMM 每步约 17 ms，按带宽推算可降到约 8.7 ms。
2. **每卡 KV 多 32%**（mem 相同时约 38%）：TP4 每卡 304 万 token，TP8 每卡 403 万 token（mem 0.92）。KV 池总容量正是现在限制并发的因素。

代价：
- DP8 的 gather / combine 行数翻倍、rank 数翻倍，每步通信可能从 8 ms 涨到 12–16 ms（只有 4 rank 的微基准，未测 8 rank）。
- 8 个 rank 锁步，最忙 rank 的放大更大（C176 时 1.37）。
- 10-20 观察到 TP8 decode 步长随负载跳变，形态像 RCCL 在消息变大时换协议（`../2p1d-v0522-20261020/README.zh-CN.md`）。
- 两个 prefill 实例的跨实例缓存丢失略多（已发前缀未命中 0.32–0.41%，1P 是 0.24%）。
- KF 的 MoE kernel 是在 TP4 形状上验证的，TP8 下专家切分不同，要重新验证正确性。

估计：每卡 0 到 +10%（点估计 +3~5%，约 53–54k），高度依赖 TP8 下的通信。建议：
- 排在 4.1 之后：TP8 更依赖集合通信，CAR 和 E11 修好对它的帮助比对 TP4 更大。
- 先做一次离线准备：8 rank 的 `dp_comm_bench.py` 微基准、KF MoE 在 TP8 形状上的正确性检查。如果 8 rank 通信 ≤ 12 ms/步，再上机计分。
- 计分从 C208（每 prefill 实例 104 个 lane，与现在相同）起，再看 C192 / C224。
- 资源：要占满 138 和 140 的 16 张卡。两个 prefill 的 HiCache 共约 2.46 TB 主机内存（138 有 2.75 TB；InferenceX 预算按每 4 卡 1,499 GB 算是 2,998 GB，需要再核对规则）。
  每次计分约 2.5–3 小时（两个 HiCache prefill 起栈和拆栈后显存释放都慢）。

## 五、本次新取的数据

**11-02 正确性检查（138 → 140，`$L/night3{0,1,3}-1102.out`；200 题探测，长上下文"捞针 / 长前缀 GSM8K"）**：

| 部署 | 探测 正确 / 乱码 / 开头坏 | 长上下文 |
|---|---|---|
| E10 生产配置（不限池，CAR 开），11:00 | 0.960 / 6 / 5 | 16k × 32：16/16、15/16 |
| E10 限池门禁 + 关 CAR，第 1 次（09:44） | 0.985 / 0 / 0 | 两遍 16k × 128：64/64、61/64；64/64、62/64 |
| 同上，第 2 次（11:10） | 0.980 / 0 / 0 | 64/64、63/64；62/64、63/64 |
| 同上，第 3 次（11:34） | 0.980 / 0 / 0 | 64/64、61/64；64/64、62/64 |
| 对照：E10 限池门禁 + CAR 开（此前 8 次） | 6 次开头坏 3–8 条，2 次 0 条 | — |
| E10 + 关 graph packet capture（只用于采 trace，不限池） | 0.930 / 6 / 0 | 16k × 32：16/16、15/16 |
| E11（`SGLANG_USE_ROCM700A=0`，不限池） | 0.670 / 44 / 11（smoke 4/6） | 16k × 32：16/16、15/16 |

**decode 显存（运行日志 `Load weight end`、`KV Cache is allocated`）**：

| decode | 运行 | 每卡权重（主模型 + NextN） | mem-fraction | 每卡 KV 池 |
|---|---|---|---|---|
| TP4 / DP4（E10） | `c104-1029T1152` | 119.8 + 6.9 GB | 0.95 | 3,043,840 token（135.2 GB） |
| TP8 / DP8（2P1D） | `c176-1024T0241` | 74.2 + 4.6 GB | 0.92 | 4,029,184 token（179.0 GB） |

**补丁兼容性**：E10 镜像（sha `509b8427`）里的 `srt/distributed/device_communicators/custom_all_reduce.py` md5 是 `03643d4f…`，
与 `../prefill-car-20261023/patches/custom_all_reduce.py.orig` 相同，补丁版可以直接挂载。

**机器状态（11-03 03:00）**：138 GPU 4–7 显存为 0，11-02 13:25 C72 跑完后没有排新任务。

## 六、补充问答（11-03 03:30 起）

### 6.1 custom all-reduce：为什么先判 mixed 有问题、PD 没问题，现在又说可能是同一个问题

mixed 的根因结论没有变；变的是对 PD 的判断，因为当时的检验看不到"开头坏"。按时间：

| 时间 | 判断 | 依据 | 回头看 |
|---|---|---|---|
| 10-19 至 10-23 | mixed（聚合 TP4 + DP4 attention）开 CAR 乱码，根因是 aiter custom all-reduce 在 graph 捕获时注册 graph 池输入张量 | 乱码条数与 EXTEND 步数相当；关 CAR 或关 graph 都干净；all-reduce 捕获拷入后 0.975、乱码 0（`../prefill-car-20261023/`） | 成立 |
| 10-23 | PD decode 开 CAR 正确 | 机制推理：PD decode 只有 decode 步，没有 eager 的 EXTEND 与 graph 回放交替；实测 4+8（decode TP8/DP8，K5，无 KF）smoke 6/6、GSM8K 0.9659、200 题探测乱码 0、长上下文 64/64 | 当时的探测只看 `garbled` 标记，"开头坏"（最终答案常仍对）检测不到；检测脚本 `probe_badstart.py` 是 10-27 才写的 |
| 10-27 | 发现"开头坏"；200 题里只在开 CAR 的变体出现（2–6 条），关 CAR 的 6 个变体都是 0 | 二分（关 HiCache） | 当时已指向 CAR |
| 10-27 | 改判"巧合" | 1,319 题复测 clean-k3 + CAR、b3 + CAR 都是 0 | 不限池时出现率只有约 0.4%，0 / 1,319 的样本不足以排除 |
| 11-02 | 与 CAR 有关 | 限池门禁关 CAR 连续 3 次 0 / 0；开 CAR 的 8 次里 6 次 3–8 条 | 当前结论 |

11-03 用现在的检测脚本补查 10-23 的 PD 探测（`car-correct-1023T1802.probe.jsonl`）：开头坏 0 / 200。
如果出现率有限池门禁那么高（约 3%），200 题全为 0 的概率约 0.2%，可以排除；但若是生产配置的 0.4%，全为 0 的概率约 45%，排除不了。
而且那次配置（TP8、K5、无 KF、不限池）和 E10 不同。所以"PD 没问题"的结论是机制推理加上一个看不到这类错误的检验给出的，下得过早。

"可能是同一个问题"是推断，依据有三条：
- 缺陷在 CAR 的实现层面（注册 graph 池张量），mixed 和 PD 用的是同一份代码；
- 关 CAR 后开头坏消失；
- 镜像对 all-gather / reduce-scatter 同一条路径的注释写的是"回放时（on replay）对端读到旧数据"，并不要求 eager 步与 graph 交替。

尚未解释的是 PD 里的触发条件：为什么偏偏坏在新请求的第一个 decode 步，出现率又低得多。
（11-03 05:20 补充：141 同机部署上开着 CAR 也复现不了，而 E11 的乱码确实被拷入修好；"开头坏"更可能与跨机 RDMA 传输有关，见 6.4。）

### 6.2 graph 之间每步约 3 ms 的 GPU 空闲是什么

decode 每步按顺序回放 3 个 CUDA graph（draft、verify、draft_extend），graph 之间还有一些 eager 小 kernel（DSA 元数据、输入整理）。
"空闲"是 GPU 时间线上没有任何 kernel 在跑的时间。下面拆的是 C104 负载下 DP0 的 40 步（trace `decode-161450`，关 packet capture，
脚本 `../prefill-opt-20261027/scripts/ana/dgaps.py`、`dgaps2.py`）。

按位置：空闲落在 eager kernel 之前 2.30 ms/步、verify graph 之前 0.59、draft graph 之前 0.12、draft_extend 之前 0.01，合计 3.02 ms/步。

| 来源 | ms/步 | 说明 |
|---|---|---|
| 偶发长停顿 | 1.03 | 40 步里 1 次约 39 ms（另 1 次约 2 ms）。调度线程停在 verify 里，同时有线程在 zmq `recv_multipart`（推测是收新请求时的 GIL 争用，未证实）。GPU 全空 |
| graph runner 准备输入时的同步 D2H | 约 0.78 | 每步都有：`decode_cuda_graph_runner.execute` 里一次阻塞的 `hipMemcpyWithStream`，CPU 要等 GPU 跑完前面的工作才能继续准备、拷入 graph 的静态输入缓冲，这段时间 GPU 空着。平均 0.4 ms，最长 1.8 ms。具体是哪一行还没定位 |
| DP rank 之间的控制面广播 | 0.34 | CPU 侧 `torch.distributed.broadcast`，40 步里 4 次、每次约 1 ms |
| verify graph 开头 | 0.26 | graph 启动到第一个 RCCL all_gatherv 之间 |
| DSA decode 元数据之后的同步 D2H | 0.12 | 每步一次 `hipMemcpyWithStream` |
| 其他零散间隙 | 约 0.45 | |

（11-03 06:30 更正：按调用栈逐条核对（`../car-copyin-141-20261103/scripts/dgap_stacks.py`），表中"偶发长停顿"和"graph runner 准备输入时的同步 D2H"
是同一个调用点：`graph_variants.py` `DsaGraphVariants.select()` 里的 `seq_lens.max().item()`。那次 39 ms 是调度线程卡在这个 `.item()` 里，
同时另一个线程在 zmq 收消息。另外几次 2–6 ms 的空闲来自 DP attention 每轮的 CPU 同步。见 6.5 节。）

DP2 只有 1.62 ms/步。DP0 多出来的主要是那次 39 ms 停顿：DP attention 每层都锁步，DP0 停住时其他 rank 在集合通信里自旋等它，
在它们的 trace 里显示为"通信 kernel 在跑"而不是空闲，所以这次停顿实际上让 4 个 rank 一起慢了。

可做的事：
- 把每步那两处阻塞 D2H 改成异步拷贝加 event，或挪到下一个 graph 已经发出之后；
- 查清 39 ms 停顿的线程归属。yihou 试过 DP controller 阻塞 poll、`--enable-dp-attention-local-control-broadcast`，都在噪声内。

现实估计每步省 1–2 ms，端到端 +0.7–1.3%（4.2 节）。这是调度层的工作，不是 kernel。

### 6.3 dense GEMM 的精度：ATOM、Dynamo SGLang、我们

| | 注意力投影（q_a / kv_a / q_b / kv_b / o_proj）、indexer | 前 3 层 dense MLP | shared expert | 路由专家 | MTP 层（第 78 层） |
|---|---|---|---|---|---|
| AMD `GLM-5.2-MXFP4` checkpoint（quark，`exclude` 1,319 项） | BF16 | BF16 | MXFP4 | MXFP4 | BF16（整层排除） |
| **我们**（SGLang + Infera，E10） | **BF16**（hipBLASLt / triton a16w16） | BF16 | MXFP4 | MXFP4 | BF16；draft MoE 在线转 FP8（`SGLANG_GLM_NEXTN_MOE_PTPC=1`） |
| **ATOM**（InferenceX `glm5.2/atom/mi355x-fp4-mtp/agentic.yaml`） | **在线 `ptpc_fp8`** | 在线 `ptpc_fp8` | MXFP4（`*expert*` 排除在在线量化外） | MXFP4 | BF16（`model.layers.78.*` 排除） |
| NVIDIA `GLM-5.2-NVFP4` checkpoint（modelopt，`exclude_modules` 156 项） | BF16（`self_attn*`） | BF16（层 0–2） | BF16（`shared_experts*`） | NVFP4 | BF16（层 78） |
| **Dynamo SGLang**（InferenceX GB300 / B300 recipe，`nvidia/GLM-5.2-NVFP4`） | **BF16**（只设 `quantization: modelopt_fp4`，没有在线量化） | BF16 | BF16 | NVFP4 | BF16 |

三家的 KV cache 都是 FP8。结论：
- 我们的 dense GEMM 精度与 Dynamo SGLang 相同，都是 checkpoint 原生的 BF16。
- ATOM 对除专家、gate、lm_head、embedding、MTP 层之外的线性层在线做 per-token / per-channel FP8。InferenceX changelog 原文："only the dense attention/MLP Linears are re-quantized to ptpc_fp8"。
- ATOM 的这个做法已经在 InferenceX 发布的 recipe 里被接受。所以 4.2 节的"dense GEMM 改 FP8"有先例，不是独创的改精度，但仍要做精度验证，按规则由用户决定是否采用。
- 镜像里的 SGLang 能否对 quark MXFP4 checkpoint 的非专家层做在线 FP8，没有核实（yihou 的对比分析把它列为 P23，未实现）。

### 6.4 141 上的修复实验（11-03 03:40–05:20）

目录 `../car-copyin-141-20261103/`（配置、队列脚本、预测与结果）。141 是同机 PD（prefill GPU 0–3、decode GPU 4–7），E10 配方，真实接受率。

| 部署 | HiCache | 探测 正确 / 乱码 / 开头坏 | 结论 |
|---|---|---|---|
| E10 对照（CAR 开） | 关 | 0.978 / 0 / 0（400 题） | 开头坏不复现 |
| E10 对照（CAR 开） | 开 | 0.910 / 3 / 0（成功的 387 条，另 13 条同机传输 500） | 仍不复现 |
| E11 对照（`SGLANG_USE_ROCM700A=0`） | 关 | 0.810 / **26** / 5（200 题），smoke 4/6 | 乱码复现 |
| E11 + 拷入（all-gather / reduce-scatter + all-reduce） | 关 | 0.978 / **0** / **0**（成功的 593 条，另 7 条同机传输 500），smoke 6/6，长上下文 32/32、31/32 | **修好** |
| E11 + 固定 sparse graph，不拷入 | 关 | 0.995 / **0** / **0**（成功的 191 条，另 9 条同机传输 500），smoke 6/6 | **也修好**：确认根因 |
| E10 + all-reduce 拷入（CAR 开） | 开 | 0.970 / **0** / 0（成功的 599 条），生成到上限 2.0%（对照 4.7%），长上下文同对照 | 开 HiCache 时的"长生成后退化"3 → 0，有利但样本小 |

- **E11**：乱码来自 graph 内 aiter collective 的注册路径，拷入后与对照同水平；decode 每步约快 1–2 ms（探测负载下粗测），与预测一致。下一步是跨机 C104 计分。
- **"开头坏"**：同机复现不了，开不开 HiCache 都一样。跨机（138 → 140）同条件 8 次门禁有 6 次 3–8 / 200，两边的差别只在 KV 传输路径（同机 XGMI，跨机 RDMA）。
  05:20 我据此推断它与跨机 RDMA 有关、6.1 节的推断变弱；**06:40 找到 E11 的根因后，这个推断要撤回一半**：
  E11 的乱码里有和"开头坏"同样形状的输出（首 token 对、紧接一串乱码、后面回到正确答案），而 E10 graph 内唯一走注册路径的 collective 就是 CAR 的 all-reduce。
  所以"开头坏"很可能就是同一机制的低频版；141 上为什么不出现还没有解释。
- **根因（06:14 实验确认）**：镜像在 HIP 上为 DSA 模型捕获 dense（`max_kv_len ≤ 2048`）和 sparse 两套 decode graph，每个 DP rank 按自己 batch 的最长上下文选
  （`graph_variants.py` `DsaGraphVariants.select`，idle rank 一律 sparse）。同一步不同 rank 回放不同 graph 时，注册了 graph 池输入的 collective 就读到对端另一套 graph 的缓冲。
  E11 不拷入、只把所有 rank 固定到 sparse：smoke 6/6、200 题乱码 0、开头坏 0（对照 4/6、26、5）。
  这也解释了为什么 AgentX 计分从来不受影响：请求上下文都远超 2048，所有 rank 一直是 sparse；而 GSM8K 这类短 prompt（包括 InferenceX CI 的 eval）会触发。
- 跨机后续脚本 `../car-copyin-141-20261103/scripts/q140_1103.sh` 已备好、**未启动**：138 / 140 正在跑另一个会话的 `night35_1103.sh`
  （E10 关 CAR 计分已完成，正在扫 Pareto 点），不能并行。内容：E10 + all-reduce 拷入、E10 + 固定 sparse 的限池门禁各 2 次（看开头坏是否消失），
  E11 + 固定 sparse 的门禁 1 次，再计分 E10 + 固定 sparse（预测 ≥ 51,900）和 E11 + 固定 sparse（预测 ≥ 52,400）。
- 同机部署踩到两个坑，已在 kit 副本 `2p1d-kit/llr35-141` 里修好：RCCL 共享内存按进程号命名，同机两个容器撞名；Mooncake 偶发注册失败，改为重试。
  开 HiCache 时还有偶发的 aux 数据发送失败。

### 6.5 graph 之间的空闲：具体怎么省

按调用栈逐条核对（10-30 trace `decode-161450` DP0，40 步，`../car-copyin-141-20261103/scripts/dgap_stacks.py`），GPU 空闲 2.78 ms/步：

| 来源 | ms/步 | 做法 | 预计可省 |
|---|---|---|---|
| `DsaGraphVariants.select()` 里的 `seq_lens.max().item()`：每步 2–3 次同步 D2H，CPU 等 GPU 跑完才继续准备 verify graph 的输入；那次 39 ms 停顿也卡在这里 | 约 1.0 + 偶发 | 见下 | 1–2 ms |
| DP attention 每轮的 CPU 同步（`prepare_mlp_sync_batch` → `exchange_sync_rows` → gloo `all_gather`），偶发 2–6 ms | 约 0.4 | 难：要把同步与 GPU 计算重叠或降频；yihou 试过 `--enable-dp-attention-local-control-broadcast`，在噪声内 | 0–0.3 ms |
| 其余小于 0.5 ms 的零散间隙（graph 启动边界、元数据小 kernel） | 约 0.5 | 不做 | — |

第一项的做法：
1. **快速版**：把所有 rank 固定到 sparse graph（`../car-copyin-141-20261103/patches/dsa-force-sparse/`，5 行，环境变量开关），`select()` 不再读 GPU。
   AgentX 的请求上下文都远超 2048，本来就全是 sparse，不多算；同时修掉 6.4 节的 graph 错配。代价是短上下文（< 2048）的 batch 也走 sparse，会多算 indexer。
2. **通用版**：`max_kv_len` 改用 CPU 侧的长度（plain decode 已有 `seq_lens_cpu` 镜像，spec 的 TARGET_VERIFY 路径没有，要补上），
   并把它放进 DP attention 每轮已经在交换的同步行里，所有 rank 用全局最大值选同一套 graph。不加新的同步，估计 30–50 行。
3. 验证：采一次 decode trace，确认 `select` 不再出现在空闲里；C104 计分，预测 E10 + 固定 sparse ≥ 51,900（+1%，即每步省约 1.5 ms 按 0.34 折算）。

39 ms 停顿里有多少能省掉不确定：它也可能是 GIL 被收消息的线程占着，去掉 `.item()` 后换个地方等。所以写 1–2 ms 而不是 2 ms。

### 6.6 dense 层在线 FP8：改动量与预估效果

**要量化的层**（与 ATOM 的 `exclude_layer` 对齐，主干 78 层）：`fused_qkv_a_proj_with_mqa`（6144 → 2624）、`q_b_proj`（2048 → 16384）、`o_proj`（16384 → 6144）、
indexer 的 `wq_b` / `wk`、前 3 层 dense MLP。**保持 BF16**：`kv_b_proj`（decode 走吸收后的 bmm，已是 a8w8）、indexer `weights_proj`（`dsa_indexer.py:557`
直接拿权重做 `torch.mm`）、router gate、lm_head、embedding、第 78 层（MTP draft）。

**代码改动（读镜像源码估计）**：
1. `layers/quantization/quark/quark.py` 的 `get_quant_method`：被 `exclude` 命中的线性层现在走 `UnquantizedLinearMethod`；已有一个分支在设置了
   `excluded_fp8_config` 时改走 `Fp8LinearMethod`。加一个环境变量和层名白名单，命中时返回 `Fp8LinearMethod(Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"))`。约 30–50 行。
2. `Fp8LinearMethod` 本身已支持在线量化：加载后量化权重；在 aiter 上设现成的 `SGLANG_USE_AITER_FP8_PER_TOKEN=1`，就是"权重按通道、激活按 token 动态"，
   即 ATOM 的 `ptpc_fp8`，并做 bpreshuffle。`fp8.py` 里还有 RMSNorm + 量化的融合路径，q_a / kv_a 的输入量化可以并进前面的 RMSNorm。不用改。
3. GEMM 调优：decode 的 M 很小（每 rank 约 50–100 行），aiter a8w8 没调过的形状可能只有带宽的六七成。要用 aiter 的 tuner 调这 4 种 (N, K) × decode 的 M 桶，
   ATOM 已有一部分调好的行可以借（yihou 记录的 `patch_aiter_glm52_decode_gemm_rows.py`，M 144–240）。约半天。
4. 风险点：模型代码里如果还有其他绕过线性层、直接用 BF16 权重的融合 kernel，要逐个排除；已查到的只有 `weights_proj`。

合计：代码半天，GEMM 调优半天，正确性（smoke、1,319 题、16k / 100k 长上下文）和一次 C104 计分半天。正确性可以先在 141 上做。

**预估效果（decode）**：
- 带宽：这些权重每卡约 24 GB（DP attention 下注意力权重每个 rank 一份），每步 verify 都要读一遍，按约 5 TB/s 约 4.8 ms；FP8 减半，省约 2.4 ms，
  扣掉 o_proj 输入的单独量化（约 0.3 ms）和小 M 下 GEMM 效率，**每步省 1.5–2.5 ms，端到端约 +1.2–1.9%**。
- 显存：每卡省约 12 GB，按每 token 44.4 KB，KV 池每 rank 多约 27 万 token（+9%）。decode KV 池现在是并发的瓶颈之一，这部分估计再 +0.5–2%（不确定性大）。
- 合计 **+2–4%（约 52.4–53.4k）**。prefill 也可以开（M = 8192 时 FP8 算力约是 BF16 的两倍，dense GEMM 占 prefill 步长约 15%），但 prefill 不是端到端瓶颈，收益 < 1%，先只开 decode。
- 精度：ATOM 用同样的方案，PR209 报告 GSM8K 0.975（200 题），没有同口径的长上下文数据。我们要自己跑 16k / 100k 捞针和长前缀 GSM8K。
