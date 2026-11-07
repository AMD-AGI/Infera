# AUS 验证与优化：结果与分析（持续更新）

方法、环境与预测见 `README.zh-CN.md`。原始结果在 `results/`（`verify.out`、`judge.txt`、每次部署的 `*.review.txt` 与压缩的探测原始输出）。

## 0. 摘要

- **`judge.py`：0 FAIL（全部判据满足）**，`results/judge.txt`。
- **新最高吞吐：E11 + 修法 53,609 tok/s/GPU**（C104，AUS n06-25 → n06-33），同对 E10 + 修法 51,477、E10 原样 50,797（+5.5%）；E11 此前因乱码不可用，修法使它正确（第 3 节）。
- **修法在跨机部署上有效（第 1 节）**：对照组 A 两次门禁复现了问题（开头坏 0 + 6、乱码 9 + 11，共 26 条，判据 ≥ 3）；修法组 B 三次门禁 600 题
  开头坏 0、乱码 0；E11 对照 C0 乱码 53（≥ 10），E11 + 修法 C 两次 400 题乱码 0、开头坏 0；不开 MTP 的对照 N0 乱码 33，修法 N1 / N1a 各 400 题 0 / 0；
  B 生产全套 1,319 题 0.966、乱码 0、开头坏 0，lm-eval strict 0.971 / flexible 0.973，16k / 100k 长上下文全部过线。
- **修法组剩下的错题是模型本身（第 2 节）**：同一批 1,319 题在去掉我们全部优化的原版 sglang 上（MXFP4 与官方 FP8 权重各一组）同样错，
  27 道题在 4 种配置里全错（含官方 FP8 权重的原版配置）；我们的栈与"原版 + 官方 FP8 权重"的正确率 0.966 对 0.967，逐题互有对错 12 / 12，分不出差别；
  同一原版配置重跑一遍就从 0.973 变成 0.966（temperature 0 下 batch 组成不同导致临界题翻转），所有配置都落在这个范围内。
  错题的推理都连贯，没有乱码 / 开头坏；配置之间的差异是临界题在数值差异下的翻转。

## 1. 验证（`results/verify.out`、`results/judge.txt`）

| 组 | 部署 | 探测（正确 / 开头坏 / 乱码 / 撞上限） | 长上下文 | 核对 |
|---|---|---|---|---|
| unit | — | 26/26 | — | 镜像 n06-25 `1a49bf20`，preflight 两台 11/11 基线一致 |
| B1（E10 限池 + 修法） | 13:31 | 198/200 / 0 / 0 / 1 | 64/64、62/64；64/64、63/64 | fix=6、recipe=5、dual=0 |
| N1a（不开 MTP + 修法 + 审计） | 13:47 | 390/400 / 0 / 0 / 5 | 16/16、… | fix=5 debug=1、dual=4 |
| A1（E10 限池对照） | 13:59 | 187/200 / 0 / **9** / 13 | | orig=6 |
| B2 | 14:16 | 196/200 / 0 / 0 / 2 | | |
| A2 | 14:32 | 185/200 / **6** / **11** / 14 | | |
| B3 | 14:49 | 192/200 / 0 / 0 / 5 | | |
| C0（E11 对照） | 15:03 | 131/200 / **13** / **53** / 45 | | |
| C1、C2（E11 限池 + 修法） | 15:19、15:32 | 197/200、197/200 / 0 / 0 / 3、2 | | |
| N0（不开 MTP 对照） | 15:47 | 145/200 / 3 / **33** / 48 | | |
| N1（不开 MTP + 修法） | 16:00 | 390/400 / 0 / 0 / 6 | | |
| B 生产 | 16:12 | 1274/1319（0.966）/ 0 / 0 / 16 | 16k：64/64、63/64；100k：16/16、15/16 | lm-eval 0.9712 / 0.9727 |
| C 生产 | 17:03 | 1278/1319（0.969）/ 0 / 0 / 17 | 16k：64/64、63/64；100k：16/16、15/16 | lm-eval 0.9674 / 0.9697 |

（长上下文与审计的逐项数字见 `results/judge.txt`；上表空格处为待补。）

对照 A 的被标记样本是这个 bug 的典型形状：开头坏 `Theutzer1-case55500The $500 + 800 = 1300. …`、`The了多少1" a0200 students …`，
长生成后退化成 `7=-5-9=15=7==-=17=-==y.76=7Ty-7…` 或无尽重复（`results/verify/arm-A-gate-*.review.txt`）。修法组没有一条这样的输出。

## 2. 错题：模型本身还是我们的栈（11-03 16:40–17:30）

用户 16:26 问 B1 的 2 道错题与长前缀的错题是什么、是否是我们的问题。B1 的错题：第 12 题（柠檬树"几年后开始赚钱"，模型答回本的 12 年，标准答案 13）；
第 119 题（工资题，反复推敲"30 percent higher"的含义直到 16,384 token 上限，无最终答案）。长前缀：第 12 题同上；Meredith"2/5 times more"
（模型理解为 5 × 2/5 = 2 篇 → 44，标准答案按 5 + 2 = 7 篇 → 104）；Charlie 蜡烛答 `#### 20.00`，打分脚本按字符串比较判错（实际正确）。

参照组（`README` 第 4 节，同题、temperature 0、max_tokens 16384、并发 64、同一探测脚本）：

| 配置 | 1,319 题正确（打分脚本） | 按数值相等 | 撞上限 | 开头坏 / 乱码 |
|---|---|---|---|---|
| 原版 sglang，MXFP4，KV bf16（`ref-mxfp4-stock`） | 0.973 | 35 错 | 12 | 0 / 0 |
| 原版 sglang，**官方 FP8 权重**，KV bf16（`ref-fp8-stock`） | 0.967 | 44 错 | 18 | 0 / 0 |
| 原版 sglang，MXFP4，**KV fp8_e4m3**（`ref-mxfp4-fp8kv`） | 0.965 | 46 错 | 20 | 0 / 0 |
| 原版 sglang，MXFP4，KV bf16，**原样重跑**（`ref-mxfp4-stock-r2`） | 0.966 | 45 错 | 13 | 0 / 1（误标：第 213 题答案正确、推理连贯） |
| **我们**：E10 + 修法，PD 4+4 生产配置（`arm-B-prod`） | 0.966 | 44 错 | 16 | 0 / 0 |
| **我们**：E11 + 修法，PD 4+4 生产配置（`arm-C-prod`） | 0.969 | | 17 | 0 / 0 |

- **27 道题在 4 种配置里全错**（第 119、306、368、403、454、505 等），包括官方 FP8 权重的原版配置：这是模型（或题目标注）的问题。
- 两两比较"你错我对 / 我错你对"：我们对官方 FP8 原版 12 / 12，对 MXFP4 + FP8 KV 原版 12 / 10，都对称；只在我们这里错的 6 题里 5 题是撞上限，
  而每种配置都各有约 6 道"只它错"的题。MXFP4 + KV bf16 第一遍错得最少（对其余三组 6 / 15、4 / 15、6 / 15）；**同一配置原样重跑一遍是 0.966、45 错，
  与第一遍互有对错 5 / 15**，和它与其他配置之间的不对称一样大：第一遍只是偏好的一次。两遍逐字相同的输出只有 74 / 1,319 条——temperature 0 下
  batch 组成不同、kernel 的归约顺序随之不同，临界题就会翻转。**同一原版配置两次就跨了 0.966–0.973，所有配置（0.965–0.973）都在这个范围里**。
- 长前缀 GSM8K（16k 填充，64 题）：原版 MXFP4 62/64（与我们的 62–63/64 同样错柠檬树和 Meredith 两题），**原版 + 官方 FP8 权重 64/64**（柠檬树答 13、
  Meredith 答 104）：这两题的倾向来自 MXFP4 量化的权重，不是我们的栈。
- 常被问到的几题在各配置下的答案：第 119 题三组原版全部撞上限；第 12 题 MXFP4 两组答 12、官方 FP8 答 13（修法组里也有一次答 13）；第 45、100、93 题在
  原版组之间也在两种理解或撞上限之间翻转。
- 我们的栈在数值上与原版的差异（FP8 KV、MXFP4 MoE 的 KF kernel、MTP、DP attention、PD 传输）合起来，正确率落在原版几组的范围内；
  FP8 KV 是 InferenceX 上三家（ATOM、Dynamo SGLang、我们）的共同选择。
- 结论：**可以接受**。反复出错的题是模型能力 / 题目歧义；修法组没有任何开头坏 / 乱码，正确率与原版 + 官方 FP8 权重相同，lm-eval 0.971 / 0.973 与此前 AUS 上
  #4629 配置的 eval 0.9727 同一水平。

## 3. 计分（AgentX C104，3600 s，SIM 2.99，`results/score-vfix-*.txt`）

| 组 | tok/s/GPU | 输出 tok/s/GPU | req/s | 完成 / 丢弃 | 交互 p50 / p90 | ITL p50 / p90 ms | TTFT p50 / p90 / 均值 s | 缓存命中 |
|---|---|---|---|---|---|---|---|---|
| crusoe 138 → 140 E10（参考，不同节点） | 51,374 | 424.1 | 3.397 | 12,325 / 2 | 63.6 / 56.1 | 15.7 / 17.8 | 4.34 / 10.42 / 5.95 | 96.0% |
| **scoreB：E10 + 修法**（11-03 18:02–19:02） | **51,477** | 425.9 | 3.411 | 12,373 / 6 | 63.9 / 56.2 | 15.7 / 17.8 | 4.29 / 10.93 / 6.04 | 96.0% |
| **scoreC：E11 + 修法**（19:31–20:31） | **53,609** | 438.9 | 3.510 | 12,732 / 3 | 69.1 / 60.9 | 14.5 / 16.4 | 4.01 / 9.36 / 5.53 | 96.1% |
| **E10：同对基线，无修法**（21:00–22:00，优化队列 30） | **50,797** | 419.1 | 3.367 | 12,209 / 5 | 63.7 / 55.7 | 15.7 / 17.9 | 4.55 / 11.41 / 6.46 | 96.0% |

- scoreB 的 decode：在跑中位 54、最忙 rank 中位 17 / p90 25，预分配等待 4.7、传输等待 15.4（crusoe E10：53、17 / 26、4.2、15.9）；
  步长按最忙 rank 请求数：@17 45 ms、@25 52 ms（crusoe E10 约 44 ms @17）。**与 crusoe E10 在噪声内相同**；
  **同对 E10 基线 50,797：修法 +1.3%**（预测 ≥ +1%，成立；单次测量 ±1%，方向一致但幅度接近噪声）。E10 的预分配等待 5.1、传输等待 16.6、
  TTFT 均值 6.46 s，B 为 4.7、15.4、6.04 s；步长 @17 都是 45 ms：省下的主要在 verify 前的同步与 graph 数量（显存），不在每步 kernel 时间。
- **scoreC 53,609：同对上比 scoreB +4.1%、比 E10 基线 +5.5%**（预测 C ≥ B + 1%、≥ 52,400，成立）。decode 在跑中位 51、最忙 rank 16 / p90 24、预分配等待 3.1；
  步长 @17 42 ms（B 45 ms，−7%）、@25 49 ms（B 52 ms）：E11（MoE 合并走 graph 内 aiter reduce-scatter）每步约省 3 ms，比 4.2 节估计的 1.3–2.4 ms 多。
  交互 p90 56.2 → 60.9，ITL 中位 15.7 → 14.5 ms，TTFT 均值 6.04 → 5.53 s。**这是目前最高吞吐（crusoe 51,374 → AUS 53,609）**（注：草稿 MoE 处在 4.3 节的 PTPC bug 下，计分口径不受影响；真实接受率下应关 PTPC 或带 #45800，C + PTPC 关 52,830），且正确性已验证（C 门禁 400 题、C 生产 1,319 题、lm-eval 0.967 / 0.970）。

## 4. 优化实验

### 4.1 decode trace（E11 + 修法，C104 负载，11-03 23:16 / 23:21 两次 40 步，`results/opt/opt.out`、trace 在 `/perf_apps/liyingli/vfix-logs/opt/trace/`）

`ana/dgaps.py`（GPU 空闲按下一个 kernel 的归属分）与 `ana/dgap_stacks.py`（空闲期间的阻塞 runtime 调用与 Python 栈）：

| rank | 第一次 ms/步（→draft / →verify / →draft_extend / →eager / 合计） | 第二次合计 |
|---|---|---|
| DP0 | 0.11 / **0.03** / 0.01 / 1.72 / **1.86** | 1.76 |
| DP1 | 0.11 / 0.03 / 0.01 / 0.61 / 0.75 | 1.76 |
| DP2 | 0.10 / 0.03 / 0.01 / 0.61 / 0.75 | 0.60 |
| DP3 | 0.11 / 0.03 / 0.01 / 0.58 / 0.73 | 0.53 |
| 10-30 E10（修法前，crusoe）DP0 / DP2 | 0.12 / **0.59** / 0.01 / 2.30 / **3.02** | DP2 1.62 |

- **verify graph 前的空闲 0.59 → 0.03 ms/步**：`select()` 的 `.item()` 没了（预测成立）；每步空闲最多 1.9 ms（预测 1.5–2 ms）。
- **00:20 更正**：DP0（第二次也有 DP1）多出来的那段约 34–47 ms 空闲在 trace 开头（profile 启动后的第一步：首个 `hipGraphLaunch` 6.9 ms，随后
  `process_batch_result_decode` 的 `hipEventSynchronize` 等它），是采集本身的启动开销，不是每 40 步一次的停顿；该段时间里 GPU 其余时候一直有 kernel 在跑。
  去掉 trace 开头 60 ms 后，**8 份 trace 每步空闲 0.52–0.83 ms**（DP0 0.83 / 0.59，DP1 0.74 / 0.59，DP2 0.74 / 0.58，DP3 0.71 / 0.52），约为步长的 1.3–2%。
  （10-30 trace 里那次 39 ms 停顿是否也是同一种启动开销，trace 不在本集群，无法复核。）
- 剩下的空闲主要是 DP attention 每轮的 CPU 同步：`dp_attn.exchange_sync_rows`（gloo gather）期间 GPU 断档，40 步里 0–5 次、每次约 3 ms（约 0.25–0.4 ms/步）。
  调度线程在这个 gather 里一次要待 39–50 ms（整整一步：各 rank 在此锁步等最慢的一个），多数时候与上一步的 GPU 工作重叠，只有 GPU 先做完时才露出空闲。
- **判断**：graph 间空闲这一方向的收益已由修法拿到；剩余全去掉也只省约 0.5 ms/步（端到端约 +0.4%），不再投入。
- **verify graph 按类别**（`ana/dtrace_cat.py`，第二次 trace，trace 模式下）：DP0 / DP2 每步 42.6 ms = MoE GEMM 16.6 / 16.7、dense GEMM 7.9、sparse MLA 6.8、
  indexer 2.6 / 3.4、MoE 杂项 1.1、norm 1.1、其他 6.4 / 5.4（其中 aiter `allgather_vec` + `reduce_scatter_split_first_dim` 每层各约 27 µs，78 层合计约 4.3 ms）。
  对照 10-30 E10 trace（55.6 ms 步：MoE 17、dense 8、通信 8、sparse MLA 7、indexer 3.6）：**E11 把通信从约 8 ms 降到约 4.3 ms**，与步长 @17 45 → 42 ms 一致。
  MoE GEMM 仍占 39%，是 2P1D（TP8 每卡 MoE 权重减半）要攻的那块；`33b` 在 2P1D 计分后采 TP8 的同样拆解作对比。

### 4.2 decode DP 均衡（router 按请求数加权，11-03 23:58–00:58，`results/opt/score-Cdpbal.txt`）

| | tok/s/GPU | 交互 p50 / p90 | ITL p50 / p90 ms | TTFT p50 / p90 / 均值 s | decode 在跑中位 / 最忙 rank 中位 / p90 | 预分配等待 / 传输等待 |
|---|---|---|---|---|---|---|
| scoreC（权重 0） | 53,609 | 69.1 / 60.9 | 14.5 / 16.4 | 4.01 / 9.36 / 5.53 | 51 / 16 / 24 | 3.1 / 15.9 |
| **Cdpbal（权重 1,000,000）** | **53,107（−0.9%）** | **73.9 / 65.6（+7% / +7.7%）** | 13.5 / 15.2 | 4.50 / 12.99 / 6.77 | 48 / **14 / 20** | **6.1** / 16.7 |

- router 核对：容器内二进制 md5 与重编的一致（`6ba18caf…`）、`INFERA_DECODE_REQUEST_WEIGHT=1000000`、新指标 `infera_router_decode_demand_requests` 在报（00:46 四个 rank 11 / 11 / 12 / 14）。
- **预测（吞吐 +1–3%）不成立**：吞吐在噪声内持平。机制与 10-21 1P1D 相同——最忙 rank 请求数中位 16 → 14、p90 24 → 20，decode 步长变短、交互 p90 +7.7%；
  代价是请求被派到 KV 余量更少的 rank，预分配等待 3.1 → 6.1、TTFT 均值 +22%、p90 +39%（事先写下的风险成立）。C104 下 lane 周期 = TTFT + decode，两者抵消。
- 用处：同吞吐下交互 +7.7%，可能让更高并发仍满足交互要求；E11 使 decode 变快后最优并发也可能上移。追加 C112 的两组（C、C + 均衡），排在 2P1D 之后。

### 4.3 真实 MTP 接受长度只有约 1.2：原因是 FP8 草稿 MoE（`SGLANG_GLM_NEXTN_MOE_PTPC=1`）（11-04 01:15 发现，01:42 定位）

decode 日志 `accept len` 的均值（每 DP rank 每 40 步一行，探测 / smoke 负载，真实接受率）：

| 部署 | CAR | 修法 | 接受长度均值（行数） |
|---|---|---|---|
| A 门禁（E10 原样） | 开 | 无 | **1.13**（2,354） |
| B1 门禁 / B 生产 | 开 | 有 | 1.31（967）/ 1.22（4,708） |
| C1 门禁 / C 生产（E11） | 开 | 有 | 1.25（1,566）/ 1.23（4,782） |
| 2P1D 门禁（TP8 decode） | 开 | 有 | 1.35（434，进行中） |
| 参考：#210 栈 4+8 / 2P1D（K5，CAR 开，10-23 / 10-20） | | | 中位 4.55–4.64 |

- K3（3 步、4 个 draft token）下上限是 4，按 InferenceX 的 golden 值应约 3（计分用的模拟值 2.99）。现在每步只多接受约 0.2 个 token（接受率 0.09–0.14）：
  **草稿几乎全被拒**。输出仍正确（verify 精确，修法组无乱码 / 开头坏、正确率与原版同），计分用模拟接受长度也不受影响；但真实流量下的吞吐会远低于计分，
  2.99 这个模拟值对这套栈也没有依据。**修法前（A）就是这样，与修法无关**；draft 的 graph runner 不建 variant（`attention_graph_variants = None`），也排除了 variant 错配。
- 与 yihou 09-18 的记录同症状（"decode 上的 custom all-reduce 破坏推测路径……MTP 接受长度约 1.25"）。E10 相对 #210 栈的不同还有：FP8 草稿 MoE
  （`SGLANG_GLM_NEXTN_MOE_PTPC=1`）、`index_share_for_mtp_iteration=true`、KF kernel、K3。
- **二分**（队列 32b–32e，C 生产配置各改一项，smoke + 200 题，读接受长度；预测写于 01:20）：关 CAR → 若 ≥ 2.5 则 CAR 是原因（最可能，同 yihou）；
  关 PTPC、`index_share` 改回 false、关 KF：各自若 ≥ 2.5 则是该项。都不变则是 K3 / 草稿 KV 传输等其他原因，再查。

二分结果（C 生产配置各改一项，smoke + 200 题，真实接受率；`results/opt/opt.out` 的 `accept` 行）：

| 改动 | 接受长度均值（行数） | 200 题 正确 / 开头坏 / 乱码 |
|---|---|---|
| 关 CAR（`DECODE_CUSTOM_AR=0`） | 1.22（1,119） | 196 / 0 / 0 |
| **关 FP8 草稿 MoE（`SGLANG_GLM_NEXTN_MOE_PTPC=0`）** | **3.61**（189） | 197 / 0 / 0 |
| `index_share_for_mtp_iteration=false`（两侧） | 1.20（818） | 195 / 0 / 0 |
| 关 KF（`YIHOU_KF_*=0`，PTPC 仍开） | 1.18（883） | 197 / 0 / 0 |

- **原因是 PTPC 草稿 MoE**：MTP 层（第 78 层）的路由专家在 checkpoint 里是 BF16（quark 排除），`SGLANG_GLM_NEXTN_MOE_PTPC=1` 在加载时把它们转成
  per-channel FP8、激活按 token 动态量化（`models/glm4_moe.py` 的 `should_apply_glm_nextn_moe_ptpc` 与 `layer_quant_config["model.decoder.mlp.experts"]`）。
  这样算出的草稿几乎全被拒。关掉后接受长度 3.61（上限 4，高于计分用的 2.99），同样 200 题 decode 日志行数 1,119 → 189，decode 快约 3 倍。
- 关 CAR 无变化：与 yihou 09-18 的症状相同但原因不同（当时的 CAR 问题已由别的改动修掉）。
- **KF 关、PTPC 开仍是 1.18：错在 PTPC 路径本身**（加载时的 FP8 转换或 `QuarkW8A8FP8MoE` 的按 token 动态量化执行），与 KF MoE overlay 无关；
  `index_share` 也无关。已排除：shared expert 被拼进 FP8 专家组（v0522 的 `shared_experts_fusion_disable_reason` 在 PTPC 下关掉了融合，日志
  "GLM NextN PTPC does not support shared experts fusion … disabled"）；加载时的转换本身（每输出通道 amax / 448、e4m3fn，读代码无误）。
  剩下的嫌疑在执行：per-channel scale 的加载 / TP 切分、aiter FP8 MoE 对这种 scheme 的支持。下一步：单卡上取 MTP 层 MoE，同一输入比 BF16 与 PTPC 的输出（计分空档里做）。
- 计分 C + PTPC 关（`32f`，C104，SIM 2.99，02:05 起）量出它在计分口径下的代价（草稿 MoE 回到 BF16）。
- 含义：现在所有计分都用模拟接受长度，不受这个 bug 影响；但带着 PTPC 的配置在真实流量下每步只多出约 0.2 个 token，真实吞吐远低于计分。**发布配置应关 PTPC
  （或先修好 PTPC 路径）**。

**03:10 根因（上游已修）**：PTPC 功能本身是 SGLang [#39155](https://github.com/sgl-project/sglang/pull/39155)（09-16 合入）。它的 bug 由
[#45800](https://github.com/sgl-project/sglang/pull/45800)（10-29 合入，**晚于 v0.5.22**）修掉：GLM NextN 把 checkpoint 里逐专家的排除项映射成运行时名字，
Quark 把这些叶子排除当作排除整个融合专家模块，于是 **PTPC 的量化方法从未被选上，而加载器照样把专家权重转成了 FP8**。FP8 权重进了按 BF16 执行的 MoE，
草稿是错的。日志里那行 "MTP fused MoE … runs as PTPC-FP8" 只说明配置写进去了。我们的镜像（fdbafb60，基于 10-18 的 v0.5.22）没有 #45800。

- 计分口径下的影响（C104，SIM 2.99）：C + PTPC 关 **52,830**，C（PTPC 开但是坏的）53,609，差 −1.5%（单次 ±1%）；步长 @17 43 对 42 ms。
  两者执行的都是 BF16 MoE，差别大半应是噪声。yihou 10-26 记录的 PTPC +1.4%（llr26 45,066）同样是在这个 bug 下、模拟接受率测的。
- 下一步（队列 32g / 32h）：把 #45800 回移到镜像的 `models/glm4_moe.py`（`opt/ptpc-fix/`，hunk 偏移 5 行），C + PTPC 开 + #45800：先门禁（真实接受长度、200 题、
  16k × 64），接受长度 ≥ 3 且干净才计分，量出真正的 FP8 草稿 MoE 的收益。
- InferenceX 规则（`docs/PR_REVIEW_CHECKLIST.md`）：agentic 的推测解码必须用 golden AL 的模拟接受长度（草稿质量不进分数）；更低精度的同样 FLOPs 允许，
  前提是通过 private eval，且"只用对准确性有要求的客户在生产中会用的优化"；不许给引擎打补丁（镜像原样运行）。所以 PTPC 只有在 #45800 进了所用镜像、
  且真实接受率正常时才站得住。ATOM（InferenceX recipe `glm5.2_fp4_mi355x_atom_mtp.sh`）：`--online_quant_config '{"global_quant_config":"ptpc_fp8",
  "exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}'`——排除的是所有专家（主干专家本来就是 MXFP4），MTP 层的专家因此保持 BF16；
  **05:40 更正**：recipe 并没有点名排除 `model.layers.78.*`（best-4p4 6.3 节的写法不准确），MTP 层非专家线性层是否也被在线量化取决于 ATOM 对草稿模型是否套用同一配置，未核实。
  Dynamo SGLang（NVFP4）MTP 层是 BF16。两家都没有量化草稿 MoE。ATOM 的 `--online_quant_config` 是 ATOM 引擎自己的功能；SGLang main 对 quark MXFP4
  checkpoint 的 BF16 排除层没有现成的在线 FP8（`quark.py` 的 `excluded_fp8_config` 只用于 checkpoint 里本就是 FP8 的排除层），要做需自己实现（best-4p4 6.6 节估约 30–50 行）。

- **03:43 #45800 回移验证**（`arm-C-prod-ptpcfix`，C + PTPC 开 + 回移的 `glm4_moe.py`）：**真实接受长度 3.60**（334 行；坏的 PTPC 1.2、关 PTPC 3.61），
  200 题 198 / 0 / 0、撞上限 1，16k × 64 捞针 32/32、长前缀 30/32。**#45800 修好了 PTPC**：草稿 MoE 现在真正以 FP8 执行且草稿正确。
  计分（`32h`，C104，SIM 2.99）03:43 起。

- **05:12 C + PTPC + #45800 计分：53,666**（C104，SIM 2.99）：比 PTPC 关（52,830）+1.6%，与坏的 PTPC（53,609）持平；交互 p50 / p90 70.8 / 61.9、
  预分配等待 2.6（关 PTPC 3.7）。**这是目前吞吐最高、且草稿真实可用（接受长度 3.60）的配置。**
- 显存 / KV 池（decode `max_total_num_tokens`）：坏的 PTPC 3,043,456、PTPC 关 3,043,072、**PTPC + #45800 3,093,376（+1.6%）**（顺带：关 CAR 3,191,552，+4.9%）。
  坏 PTPC 与关 PTPC 容量相同，所以两者 1.5% 的差不是容量所致：两者 MoE 都按 BF16 执行，区别是坏 PTPC 关掉了草稿的 shared expert 融合（单独算），
  关 PTPC 时草稿 shared expert 融进 MoE（257 个专家）；步长 @17 42 对 43 ms。单次 ±1%，多半是噪声加这一点路径差。真 FP8 才有容量与带宽收益。
- PTPC 量化的范围：只有 MTP 层（第 78 层）的**路由专家**（gate / up / down），权重按输出通道静态 FP8（加载时算），激活按 token 动态 FP8；shared expert
  与 MTP 层的注意力、dense 部分仍是 BF16。来历：SGLang [#39155](https://github.com/sgl-project/sglang/pull/39155)（AMD，09-16 合入，默认关），
  动机是 MXFP4 checkpoint 里 MTP 层的专家是 BF16（每个 71.7 MB，MXFP4 层 19.0 MB），每个草稿步都要读。我们（yihou llr26）只是打开了开关。
  InferenceX recipe、SGLang 文档 / 示例、Infera 示例里都没有打开它的推荐配置。
- 与"dense 层在线 FP8"（交接第 4 项、best-4p4 6.6 节）的区别：后者是主干 78 层的非专家线性层（q_a / kv_a / q_b / o_proj、indexer wq_b / wk、前 3 层 dense MLP）
  做在线 per-token / per-channel FP8，**与 ATOM 的 `ptpc_fp8` 是同一件事**；ATOM 恰好把 MTP 层排除在外。两者只是用了同一种量化方式（PTPC），作用的层不同。

### 4.4 2P1D（2 × PP4 prefill + TP8 / DP8 decode）

- 门禁（C + PTPC 开坏的版本，01:03）：smoke 6/6、200 题 198 / 0 / 0、16k × 128 捞针 63/64、长前缀 61/64；真实接受长度 1.35（PTPC bug）。KF kernel 在 TP8 形状下结果正确。
- **门禁（C + PTPC + #45800，05:13）**：smoke 6/6、200 题 196 / 0 / 0（撞上限 1）、16k × 64 捞针 32/32、长前缀 31/32；**真实接受长度 3.43**（TP4 为 3.60）。
  decode KV 池 4,069,696 token / rank（TP4 3,093,376，+32%，与 best-4p4 5 节的估计一致）。C208 计分 05:22 起。

### 4.5 DP 均衡权重的依据（05:40）

`ana/decode_step_fit.py`（10-21 的脚本，加了 `DP_RANKS`）：每 3 s 一个窗口，取各 rank 的在跑请求数与上下文 token，拟合
`步长 = a + b·最忙 rank 请求数 + c·最忙 rank 上下文(M) + d·总请求 + e·总上下文(M)`，b / c 就是"最忙 rank 上 1 个请求相当于多少 token"——router 的代价是 token，
权重应取这个换算值。

| 部署 | R² | 拟合 | 1 个请求 ≈ token | 最忙 / 均值 请求数（中位 / p90） |
|---|---|---|---|---|
| scoreC（权重 0） | 0.878 | 21.3 + 0.70·max_req + 1.66·max_tok + 0.096·sum_req − 0.089·sum_tok | **423,308** | 1.28 / 1.52 |
| Cdpbal（权重 1,000,000） | 0.925 | 21.4 + 0.55·max_req + 1.54·max_tok + 0.116·sum_req + 0.026·sum_tok | 355,716 | 1.20 / 1.38 |

- 10-21 的 1,000,000 是在 v0.5.22 **1P1D C88** 上拟合的（那里 1 个请求 ≈ 105 万 token）；在本配置（4+4、K3、E11）上换算值约 **42 万**，1,000,000 偏大一倍多，
  过度按请求数分配、把长上下文请求压到 KV 余量少的 rank，这与预分配等待翻倍、TTFT +22% 一致。
- 拟合只看步长，不含 KV 池压力；KV 常满时最优权重可能比换算值更小。队列 35 改为 **400,000**（原计划 300,000）。
- **C208 计分（05:44–06:44，`results/opt/score-2p1d.txt`）：54,094 tok/s/GPU**（16 卡），4+4 最佳正确配置 53,666（8 卡）：**+0.8%（噪声内）**；
  交互 p50 / p90 77.2 / 65.7（4+4：70.8 / 61.9，+9% / +6%），ITL p50 12.9 ms（14.1），TTFT p90 8.01 s（9.28）、均值 6.31 s（5.70）。预测 0 到 +10%（点估计 +3–5%）：落在下沿。
- **瓶颈转到 prefill**：两个 prefill 92% 的时间有排队（4+4 为 66%），排队均值 6.1 / 6.2（3.45）；未命中缓存的吞吐 17,639 / 18,952 tok/s（4+4 一个实例 16,735），
  两实例大致均衡（块数 7,343 / 7,920）。decode 有余量：预分配等待 0.2（4+4 2.6，TP8 每 rank KV 多 32% 解除了容量瓶颈），在跑中位 86（两套 4+4 约 98），
  传输等待 39.1（17.0）——请求在等 prefill。TP8 步长在同样的最忙 rank 请求数下快约 3 ms（@4 22 对 25 ms，@15 37 对 40 ms），低负载固定开销下降，与每卡 MoE 权重减半一致。
- 判断：按每卡吞吐 2P1D 与 4+4 持平，但交互度更高，在 Pareto 上同吞吐点支配 4+4 C104。要让 2P1D 吞吐再上去，需要提高 prefill 能力
  （decode 已有余量）：prefill 侧的候选见 STATUS 之后的计划。
- **2P1D 提吞吐的 prefill 侧候选（07:10 整理）**：
  1. prefill chunk 8192 → 16384：yihou llr38 已测 −2.2%（prefill 批次 1.93 → 1.34/s、TTFT 与 KV 传输延迟上升），不做。
  2. prefill BF16 GEMM 调优：10-28 已做（`../prefill-opt-20261027/gemm-tune/`），调出与默认启发式同一族 kernel，每级每步省 < 1 ms（< 0.3%），不做。
  3. **E9 后台预写回**：4+4（decode 受限）下 TTFT −14%、吞吐 +0.7%；2P1D 受 prefill 限，可能真正转化为吞吐。补丁（`2p1d-kit/llr35/patches/pre-write-back/`）
     在 crusoe 本地、未进仓库，本集群没有，需要按 `../prefill-opt-20261027/` README 10-29 的设计重写（v5：受害叶子同步写回 3.2%、未命中率 4.83%）。
  4. **dense 层在线 FP8**（ATOM 式 `ptpc_fp8`，交接第 4 项的最后一项）：prefill 侧 dense GEMM 约占 prefill 步长 15%，FP8 约省一半 → prefill 步长约 −7%；decode 侧
     best-4p4 6.6 节估 +1.2–1.9%。改精度、需自己实现（约 30–50 行 + GEMM 调优）、需通过 eval。
  5. 并发：受 prefill 限时加 lane 只会加排队；降到 C192 可能同吞吐下 TTFT 更好（Pareto 上的另一点），不是提吞吐。
- **07:19 C + PTPC + #45800 生产全套正确性**（`33a`）：smoke 6/6；1,319 题 **0.971**、开头坏 0、乱码 0、撞上限 16；16k × 128 捞针 64/64、长前缀 62/64；
  100k × 32 捞针 16/16、长前缀 16/16；**lm-eval strict 0.9697 / flexible 0.9689**；真实接受长度 3.62（1,705 行）。全部达到生产判据，可作为发布候选配置
  （InferenceX 合规还需：带 #45800 的 SGLang 镜像，及 #48302 合入）。

## 5. Roofline 与剩余空间（11-04 07:40）

**基准层面**：AgentX 是闭环回放，每卡吞吐 ∝ 并发 lane 数 /（卡数 × lane 周期），lane 周期 = trace 自带的工具 / 思考间隔（约 50%，服务端改变不了）+ TTFT（约 16%）
+ decode（约 34%）（10-30 E10 拆分）。同样并发下服务端时间全部归零，吞吐的上限也只有约 2 倍；能加多少并发又受 decode KV 池（4+4 均值 0.82、常满）与 prefill 能力
（4+4 66%、2P1D 92% 时间排队）限制。所以 roofline 要分三层看：周期结构、容量（KV、prefill 吞吐）、每步硬件效率。

**decode 每步**（步长拟合 `ana/decode_step_fit.py`；带宽按可达约 6.5 TB/s 估，MI355X 峰值 8 TB/s）：

| | TP4（4+4，scoreC） | TP8（2P1D C208） |
|---|---|---|
| 拟合 | 21.3 + 0.70·最忙请求 + 1.66·最忙上下文(M) + … | 17.8 + 0.80·最忙请求 + 0.64·最忙上下文(M) + … |
| 每卡权重（主模型 + NextN） | 113 + 6.9 GB | 74.2 + 4.6 GB |
| 每步读量（verify 1 次 + 草稿 3 步与 draft_extend 共 4 次 NextN） | 约 140 GB | 约 88 GB |
| 带宽下限 | 约 21.6 ms | 约 13.6 ms |
| 固定开销（截距）与下限之差 | **≈ 0（已在带宽 roofline 上）** | **约 +4 ms（TP8 的 kernel / 通信没吃到权重减半的好处）** |

- TP4 的固定部分已在带宽 roofline 上：要再降只能少读字节（dense 层 FP8：每卡约 24 GB → 12 GB，约 −1.8 ms；PTPC 真 FP8 已把草稿专家减半，−1.4 ms 左右，与 +1.6% 一致）
  或换并行方式（TP8）。随负载增长的部分（最忙 rank 16 个请求时约 11–14 ms：注意力 / indexer / MoE 计算 / 通信）才是 kernel 优化的对象。
- verify 分项（4.1 节）：MoE GEMM 16.6 ms（约 91 GB → 5.5 TB/s，峰值的 69%）、dense GEMM 7.9 ms（6.6 节估读完这些权重约 4.8 ms，效率约 60%）、sparse MLA 6.8、
  indexer 2.6–3.4、集合通信约 4.3、norm / 杂项约 2–3、GPU 空闲 0.5–0.8 ms。
- TP8：截距只比 TP4 低 3.5 ms（按字节应低约 8 ms）。可能原因：KF 的 MoE kernel 是按 TP4 形状（每 rank 中间维 512）调的，TP8 每 rank 只有 256，GEMM 更瘦；
  8 rank 的 all-gather / reduce-scatter 更慢。TP8 trace（队列 `33b`，暂停中）能定位。2P1D 现在受 prefill 限，decode 变快只改善交互与 TTFT，要先解开 prefill 才转成吞吐。

### 5.1 剩余优化项与预期收益（11-04 07:40，端到端吞吐；单次 ±1%；各项不能简单相加——decode 变快后瓶颈会转到 prefill）

| 优化项 | 每步 / 机制 | 预期 | 改精度 | InferenceX | 工作量 |
|---|---|---|---|---|---|
| dense 层在线 FP8（ATOM 式，目标模型 decode + prefill，**排除 MTP 层**） | dense GEMM 7.9 → 约 4–5 ms，KV +9%；prefill dense GEMM 约占步长 15% | 4+4 +2–4%；2P1D 的 prefill 也受益 | 是 | 允许（目标模型量化，需过 eval） | 30–50 行 + GEMM 调优 |
| MoE GEMM 带宽利用率 69% → 约 80% | −2–3 ms | +1.5–2.5% | 否 | 需进 aiter / SGLang 上游 | 难（KF 已深调） |
| sparse MLA + indexer | −1.5–3 ms（共 9.4–10.2 ms） | +1–2.5% | 否 | 同上 | kernel |
| graph 内通信（与 norm 融合 / one-shot） | 约 −1 ms | 约 +0.8% | 否 | 同上 | 中 |
| 小 kernel 融合（norm、MoE 杂项） | −0.5–1 ms | +0.4–0.8% | 否 | 同上 | 中 |
| DP 均衡（拟合权重 40 万） | 最忙 / 均值 1.28 → 约 1.2 | 吞吐 ±1%，交互更高 | 否 | Infera #213 已合入 | 已备好 |
| 4+4 C112 | — | −2% 到 +3% | 否 | — | 在跑 |
| graph 间空闲 | < 0.5 ms | < 0.4% | — | — | 不再做 |
| 2P1D：E9 后台预写回 | 4+4 下 TTFT −14% | +1–3% | 否 | 需上游 | 重写约半天 |
| 2P1D：TP8 decode kernel（MoE 的 TP8 形状、8 rank 通信） | 截距 −2 到 −4 ms | 交互提升；解开 prefill 后才转为吞吐 | 否 | 需上游 | TP8 trace 定位中 |
| 2P1D：prefill 调度改 `hrrn` | 长请求不再饿死（6.2 节） | −2% 到 +3% | 否 | 上游已有（v0.5.22 自带） | 只改参数 |
| 2P1D → 1P(PP8)1D(TP8) | 单一缓存，免两实例重复（6.2 节） | −3% 到 +5% | 否 | 普通参数 | 一次门禁 + 计分 |

4+4 现实目标约 +5–8%（约 56–58k），前提是 prefill 不先成为新瓶颈。

### 5.2 第二版 Pareto 的建议（07:40）

- 现在这套栈**不能原样提交 InferenceX**（第 6 节清单），所以现在扫的曲线只能作内部决策 / 预期；正式曲线要在上游合入、换新镜像后扫。
- 做法：先 TP8 trace 与 C112（07:58 起在跑）；再做轻量内部 v2：4+4（提交方向的配置，见 6.1）补 C32 / C64 / C88，2P1D 补 C144 / C176（已有 C208），约 5 个点、7–8 小时。

## 6. 提交 InferenceX：现状与差距（11-04 08:10）

**规则更正（08:05，InferenceX `.github/codeowner-signoff-verify-prompt.md` Check 13、`CONTRIBUTING.md` "Draft-model precision"）**：草稿必须"按发布时的样子"运行——
对草稿权重 / 激活 / 计算做任何在线或离线量化都判 FAIL（eval 通过、接受长度达标都不豁免），`SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE` 被点名禁止。
**所以 `SGLANG_GLM_NEXTN_MOE_PTPC=1` 在 InferenceX 上不能用**（`amd/GLM-5.2-MXFP4` 的第 78 层按 BF16 发布）。4.3 节"更低精度允许"只对目标模型成立，对草稿不成立。
目标模型量化（如 ATOM 式 dense 层 `ptpc_fp8`）仍允许，前提是过 eval 且不顺带量化草稿。

### 6.1 当前最佳配置（C + PTPC + #45800，53,666）里各项的上游状态

| 项 | 位置 | 上游 / 规则状态 | 提交方向 |
|---|---|---|---|
| 修法（graph variant 跨 rank 一致）6 个文件 | decode 挂载 | [sglang#48302](https://github.com/sgl-project/sglang/pull/48302) OPEN（11-04） | 必须合入（E11、CAR 的正确性依赖它） |
| E11：`SGLANG_USE_ROCM700A=0` | decode 环境变量 | 普通开关 | 可用（依赖 #48302） |
| decode custom all-reduce 开 | kit 去掉 `--disable-custom-all-reduce` | 普通参数 | 可用（依赖 #48302） |
| PTPC FP8 草稿 MoE + #45800 回移 | decode 环境变量 + 挂载 `glm4_moe.py` | #45800 已合入；**PTPC 违反 Check 13** | **不能用**（关掉：52,830） |
| KF kernel overlay（sparse MLA / indexer / MoE） | decode `PYTHONPATH` + `sitecustomize` 运行时替换函数 | 禁止 monkey-patching；**没有上游 PR** | 不能用；要进 aiter / SGLang 上游 |
| 两个 aiter GEMM 调优 CSV | 挂进 aiter 安装目录 | 改镜像文件 | 先合进 aiter，或写 waiver |
| `index_share_for_mtp_iteration=true` | 两侧 model override | 规则：不接受减少模型 FLOPs 的优化，评审可能问 | 待定（#210 用 false） |
| prefill v3 pp-chunk-wait（PP 下发送非末块前等事件） | prefill 挂载 `disaggregation/prefill.py` | [#45524](https://github.com/sgl-project/sglang/pull/45524)（他人）OPEN | 必须（`SGLANG_PP_COMM_OVERLAP=1` 的正确性依赖它） |
| E1 evict-headroom | prefill 挂载 `unified_radix_cache.py` | [#48047](https://github.com/sgl-project/sglang/pull/48047) OPEN（我们的） | 合入后可用 |
| E10 hicache-backup-quota | prefill 挂载 `pool_host/{mla,dsa}.py` | [#48042](https://github.com/sgl-project/sglang/pull/48042) OPEN（我们的） | 合入后可用 |
| `deepseek_v2.py`（PP 非首级 hidden size） | prefill 挂载 | [#47184](https://github.com/sgl-project/sglang/pull/47184) OPEN（我们的）/ 第三方 #21194 | 必须（PP4 prefill） |
| 镜像烘焙的 B1 / B2 / B6 / B7 / B8（Infera Dockerfile） | 镜像 | #33059、#46830、#46820、#47184/#21194、issue #46829（`../inferencex-publish-plan-20261027/` 5.1 节） | 即第一版被卡住的"5 个 patch"；需合入或 waiver |
| `--pp-async-batch-depth 1`、`--stream-interval 30`、mem 0.95、`SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`、`SGLANG_OPT_USE_TOPK_V2=false` | 参数 | 普通参数 | 可用，写进 cookbook |
| Infera router / kit | router 镜像、`2p1d-kit/llr35` 引擎脚本 | router 需 Infera release；kit 与 #210 的 example 有差异 | 迁到 Infera example + srt-slurm infera frontend |

结论：53,666 里有三块在 InferenceX 上用不了——PTPC（已知 −1.6%）、KF overlay（未量化，需一次"关 KF"计分）、GEMM CSV（未量化）；
`index_share=true` 待评审。其余依赖 7 个 OPEN 的 SGLang PR（含第一版的 5 个）与 Infera 发版。

### 6.2 2P1D 的 prefill：缓存命中、均衡、调度策略、PP8（11-04 08:10，C208 计分窗口 05:49–06:49）

- **前缀缓存命中**（prefill 日志 `Prefill batch` 的 new / cached token 累计）：prefill0 **95.96%**（新算 254.0M / 命中 6,032.8M），prefill1 **95.59%**（273.0M / 5,916.3M）；
  合计未命中占比 4.22%，4+4 单实例约 3.89%（命中 96.11%）——**两个实例的新算量多约 8.5%**（同一份前缀要在两边各缓存、跨实例命不中）。
- **均衡**：prefill1 的请求条目多 9.6%（81,434 / 74,310）、新算 token 多 7.5%，两实例都 92% 的时间有排队，属轻度不均衡（session affinity 按会话固定实例）。
- **SPF 的长请求饿死**（`ana/ttft_by_len.py`，aiperf 逐请求，按未命中 token 分档）：

  | 未命中 token | 2P1D C208：n / TTFT p50 / p90 / p99 / 最长（s） | 4+4 C104 |
  |---|---|---|
  | < 1K | 7,524 / 3.9 / 6.3 / 9.6 / 63 | 3,976 / 3.9 / 7.3 / 15.5 / 44 |
  | 16–32K | 628 / 9.9 / 34.5 / 66 / 76 | 274 / 8.3 / 21.0 / 48 / 75 |
  | 64–128K | 160 / 27.1 / 79.6 / 139 / 151 | 77 / 20.2 / 94.1 / 261 / 261 |
  | > 128K | **131 / 75.3 / 271.7 / 1,017 / 1,524** | 48 / 37.8 / 251.8 / 484 / 484 |

  一条 74 万 token 的冷请求纯算约 44 s，实测 TTFT 1,524 s：SPF 下不断有更短的请求插队。这类请求只占 0.6%，但每条占住一个 lane（常连着子 agent 树）。
  SPF 是 10-20 为解决 FCFS 的队头阻塞引入的（`../2p1d-aus-v0521-20261019/`：FCFS 下 prefill 只忙约 65% 却排队 10.9 s、49% 的忙碌时间被单个长冷请求独占；SPF +15.4%）。
- **其他策略**：镜像（v0.5.22）自带 `hrrn`（highest response ratio next，按 token 计的老化：比值 = 1 + 等待期间处理的 token / 本请求未命中 token），
  计数器在 `scheduler.run_batch` 累加、PD prefill 入队时打点，kit 的 `prefill.py` 补丁保留了这几行——PP 分离 prefill 下可用。它介于 SPF 与 FCFS 之间：短请求仍优先，
  长请求等得越久越靠前。未测；预期长请求尾部大幅下降、短请求 TTFT 略升，吞吐 −2% 到 +3%。
- **2 × PP4 → 1 × PP8**（未测）：好处是单一 radix 树 + HiCache（8 × 250 GB），免掉两实例的重复缓存（上面的 +8.5% 新算若能收回，prefill 受限时就是吞吐）；
  每卡权重减半（约 51 GB），prefill 设备 KV 池约翻倍，回载更少。代价：流水 8 级，要 ≥ 8 个在途块才不空泡（92% 时间排队时基本满）；单请求 TTFT 的流水延迟变长；
  每块 7 次级间激活传输（8,192 × 6,144 × 2 B 约 100 MB / 次，XGMI 上约 1–2 ms，对约 160 ms 的每级计算可忽略）；PP8 在本栈从未跑过（PP 相关补丁都按 PP4 验证）。预期 −3% 到 +5%。
- 第一版（PR 210 栈）的 2P(PP4)+1D C176 是 38,392（`../2p1d-v0522-20261020/` 698 行），当时同样 prefill 到顶（单 PP4 约 3.0 req/s）。

### 4.6 TP8 decode trace（2P1D C208，11-04 08:33 / 08:38，`/perf_apps/liyingli/vfix-logs/opt/trace/opt-trace2p1d-c208/`）

- GPU 空闲 0.9–2.5 ms/步（含采集起始那一步）。verify 按类别（`ana/dtrace_cat.py` + 按 kernel 名）：

  | 分项（ms/步） | TP4（C104，4.1 节） | TP8（C208） |
  |---|---|---|
  | MoE GEMM | 16.6（KF `gemm1_a4w4_port_fp4_h6144_i512_…_glm` + `gemm2_a4w4_port_…`） | **12.5**：gemm1 **aiter 原版** `mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x64x256` 6.88、gemm2 `gemm2_a4w4_port_hmax8192_imax256_…` 4.59、`moe_reduction` 1.03 |
  | 集合通信（all-gather + reduce-scatter） | 5.15（下方精确值；4.1 节的约 4.3 是粗估） | 5.39（下方精确值） |
  | sparse MLA / indexer | 6.8 / 2.6–3.4 | 8.0 / 3.7–3.9 |
  | dense GEMM | 7.9 | 8.1（DP attention 下每 rank 一份，与 TP 无关） |
  | MoE 排序 / 杂项 | 1.1 | 2.1 |

- **深入拆解（09:20，`ana/dtrace_deep.py`：8 / 4 个 rank 全部、每个 kernel 归类；集合通信按"第 i 次调用在各 rank 的最短时长 ≈ 传输，其余 ≈ 等最慢的 rank"拆）**：

  | ms/步（verify） | TP4（C104） | TP8（C208） | 差 |
  |---|---|---|---|
  | MoE gemm1 | 10.96（KF `gemm1_a4w4_port_…_i512_…_glm`） | 6.91（aiter 原版 `mfma_moe1_…`） | −4.05 |
  | MoE gemm2 | 5.66（`…_nt_atomic…`） | 4.63（`…_nt_reduce…`） | −1.03 |
  | MoE 单独 reduction | 0 | 1.03（reduce 版 gemm2 要另一次归约） | +1.03 |
  | MoE 排序 / 量化 | 1.11（含 KF `glm52_moe_sort_fill`） | 2.50（aiter `opus_moe_sorting` 等） | +1.39 |
  | **MoE 合计** | **17.7**（GEMM 5.81 TB/s，峰值 73%） | **15.1**（GEMM 4.18 TB/s，52%） | −2.7 |
  | all-gather（传输 + 等待） | 2.83（1.96 + 0.87） | 2.65（2.03 + 0.63） | −0.18 |
  | reduce-scatter（传输 + 等待） | 2.32（1.97 + 0.34） | 2.74（2.39 + 0.35） | +0.42 |
  | sparse MLA / indexer | 6.82 / 3.36 | 8.05 / 3.85 | +1.72 |
  | dense GEMM / norm | 7.89 / 1.19 | 8.12 / 1.15 | +0.19 |
  | 草稿 graph / draft_extend（另计） | 2.1 / 2.2 | 1.5 / 1.8 | −1.0 |

- **更正 08:55 的说法**：通信不是 TP8 的问题——按 kernel 精确统计，TP4 5.37、TP8 5.54 ms，几乎相同（之前拿 TP4 的粗估 4.3 ms 比，得出"多 1.7 ms"是错的）；
  其中等最慢 rank 的部分 TP8 约 1.0、TP4 约 1.2 ms。
- **TP8 的问题集中在 MoE**：MoE 三个部件都没用上 KF（gemm1 回退原版、gemm2 用 reduce 版要多一次归约、排序是 aiter 通用版）。按 TP4 的效率，
  TP8 的 MoE GEMM 应约 8.3 ms、加排序约 9.4 ms，实测 15.1 ms：**约 5.6 ms 空间（verify 步长的 13%）**，与截距高出带宽下限约 4 ms 对得上（第 5 节）。
- sparse MLA / indexer 在 TP8 下慢 1.7 ms，但各 rank 耗时几乎相同（8.0–8.2 ms），可能是 graph 按最大 batch 补齐的开销（TP8 最忙 / 均值 1.43，TP4 1.28），未证实。
- 对应的优化：KF 的 gemm1、atomic 版 gemm2、融合排序移植 / 调优到 i256（每 rank 中间维 256）。TP8 decode 变快在 2P1D 受 prefill 限时先体现为交互度。

### 4.7 KF overlay 与 yihou 的上游 PR（11-04 08:40）

| KF 开关 | 替换的调用点 | yihou（dorado269）的 SGLang PR |
|---|---|---|
| `YIHOU_KF_SPARSE` | `triton_sparse_mla.triton_sparse_mla_fwd` | [#47278](https://github.com/sgl-project/sglang/pull/47278) gfx950 split-K decode plan for 64-head fp8 sparse MLA（同一文件）OPEN |
| `YIHOU_KF_INDEXER` | `dsa_indexer.aiter_paged_mqa_logits` | [#47897](https://github.com/sgl-project/sglang/pull/47897) in-tree gfx950 paged MQA logits for decode / MTP-verify rows OPEN |
| `YIHOU_KF_INDEXER` | `DSATopKBackend.topk_transform` | [#47872](https://github.com/sgl-project/sglang/pull/47872) opt-in single-pass Triton top-k for long-context decode rows OPEN |
| `YIHOU_KF_FUSED_INDEXER`（本配置未开） | `gfx950_fused_indexer`（每 rank ≤ 48 行） | [#47276](https://github.com/sgl-project/sglang/pull/47276) faster gfx950 fused indexer logits and top-k for MTP verify rows OPEN |
| **`YIHOU_KF_MOE`** | **`moe_runner.aiter.AiterRunnerCore.run`（a4w4 路由 MoE）** | **没有 PR**（占 decode 步 39% 的那块） |

- 4 个 PR 覆盖 sparse MLA、indexer logits、top-k（另加一个本配置没开的融合 indexer），**MoE kernel 没有覆盖**。PR 内容与 overlay 是否逐行等价未比对
  （overlay 文件头写明基于 v0.5.21 `11bfd868` 与 aiter `e7d2453`）。
- `index_share_for_mtp_iteration`：`GLM-5.2-MXFP4/config.json` 里就是 `true`（SGLang 代码里的 `False` 只是缺省值），kit 基础参数显式覆盖为 `false`，
  E10 配方改回 `true`——即发布时的默认，不是减少 FLOPs 的改动。
- Check 13 原文：[CONTRIBUTING.md "Draft-model precision"](https://github.com/SemiAnalysisAI/InferenceX/blob/57116fff418d7bb651567e722907253a12b75a67/CONTRIBUTING.md#draft-model-precision)、
  [`.github/codeowner-signoff-verify-prompt.md` Check 13](https://github.com/SemiAnalysisAI/InferenceX/blob/57116fff418d7bb651567e722907253a12b75a67/.github/codeowner-signoff-verify-prompt.md#L608-L668)，
  由 [InferenceX#3283](https://github.com/SemiAnalysisAI/InferenceX/pull/3283)（09-19 合入）引入。

### 4.8 4+4 加并发：C112（C + PTPC/#45800，09:21–10:21，`results/opt/score-Cfix-c112.txt`）

| | C104 | **C112** |
|---|---|---|
| tok/s/GPU | 53,666 | **52,478（−2.2%）** |
| 交互 p50 / p90 | 70.8 / 61.9 | 70.1 / 61.8 |
| TTFT p50 / p90 / 均值 | 4.12 / 9.28 / 5.70 s | 5.53 / 14.74 / 8.19 s |
| prefill 有排队的时间 / 排队均值 | 66% / 3.45 | **86% / 5.19** |
| decode 在跑中位 / 最忙 / 预分配等待 / 传输等待 | 49 / 16 / 2.6 / 17.0 | 49 / 16 / **7.4** / 19.8 |

- 预测（−2% 到 +3%）落在下沿。多出的 8 个 lane 没有变成 decode 在跑的请求（中位仍 49），而是堆在 prefill 排队（86%）和 decode 预分配等待（2.6 → 7.4）里：
  4+4 在 C104 时两侧都已到顶，**C104 仍是 4+4 的最高吞吐点**（与 yihou 旧栈的 C108 −2.2%、C112 −6.3% 同方向）。

## 7. 11-04 下午至 11-05 的结论（问答原文见 QA-20261105.zh-CN.md，时间线见 STATUS.zh-CN.md）

所有吞吐为 tok/s/GPU，AgentX C 值、3600 s 测量、K3 时 `SIM_ACCEPT_LEN=2.99`（K5 时 3.61），单节点 4+4 = prefill GPU 0-3 + decode
GPU 4-7 同机。

### 7.1 单节点 4+4 = 两节点 4+4（根因：同机 Mooncake HIP 传输）

| 配置 | 吞吐 | 交互 p90 | TTFT p90 | prefill 满块 | 每级 KV 传输 | 等 decode 预分配 |
|---|---|---|---|---|---|---|
| 两节点 4+4（n06-25 → n06-33） | 53,666 | 61.9 | 9.28 s | 322 ms | — | — |
| 单节点，HIP 传输（kit 原样） | 40,807 | 67.3 | 26.52 s | 572 ms | 3.14 s / 0.99 GB/s | 6.93 s |
| 单节点，`MC_DISABLE_HIP=1` | **53,209** | 61.0 | 9.64 s | 331 ms | — | — |

- 同机时 Mooncake（faae8dd4）路由优先选 HIP 传输（xGMI GPU 间拷贝，负载下约 1 GB/s 一次），不走 RDMA；kit 里的
  `MC_DISABLE_HIP_TRANSPORT=1` 只在打了 Infera `transfer_engine_impl.diff` 的 vLLM / ATOM 镜像里有效，SGLang 镜像要用
  `MC_DISABLE_HIP=1`（RDMA 同网卡回环；同主机两块不同网卡之间不通）。方案最早见 `yihou/samenode-kv-transfer-slow.brief.yihou.md`。
- `MC_DISABLE_HIP=1` 下生产全套正确性通过（0.970、lm-eval 0.9682 / 0.9689、真实接受长度 3.62）。
- 排除项：GPU 频率 / 功耗、CPU、GPU 位置（对调后 41,395）、v2 镜像（同节点 pkg kit 40,807）。
- 节点间差异：同配置 n02-33 53,209、n04-33 53,057（0.3%）。

### 7.2 dense 层在线 FP8（target 模型 BF16 线性层，ROCm + 环境变量开启，不影响 CUDA）

| 配置（单节点 C104，no-HIP） | 吞吐 | 交互 p90 | TTFT p90 |
|---|---|---|---|
| 基线 | 53,209（n04-33 53,057） | 61.0 | 9.64 s |
| 两侧 dense FP8 | 53,176 | 55.4 | 7.36 s |
| 只 prefill dense FP8（n04-33） | 52,992 | 61.2 | 10.38 s |

- 正确性通过（0.969、lm-eval 0.9704 / 0.9727、接受长度 3.63）；decode 权重 -10.9 GB / 卡，KV +7.7%。
- 现形态没有收益。decode trace（每步）：FP8 GEMM 6.06 ms 对 BF16 约 6.6 ms（小 M 下 FP8 带宽利用率低，约 1.8 TB/s 对 3.3），
  激活量化 1.08 ms（Triton），KF 融合 indexer 只覆盖 6 的倍数且 ≤ 96 行的 batch（负载高时回退，+0.67 ms），草稿 MoE 换成 aiter
  per-token FP8 内核反而 -0.8 ms。改进方向：小 M FP8 GEMM 内核、量化融合进 RMSNorm 或改 block-scale、扩展 KF fused2；三项都做好
  估计同交互 +8%-12%（推算）。HIP 瓶颈下的单节点曾显示 +5.3%（KV 容量能转化为吞吐），不代表正常部署。
- 两处补丁：`quark.py`（排除层改走现有 `Fp8LinearMethod`），`deepseek_common/utils.py`（在线逐通道 FP8 的 `[1, N]` scale 被
  `_is_block_scale_fp8` 误判，带标记的层返回 False）。GEMM 调优 143 行（aiter a8w8 bpreshuffle 调优工具，去掉与镜像自带表重复的 10 行）。

### 7.3 DP 均衡 40 万（单节点 C104，同节点对照）

52,813 对 53,209（-0.7%）；交互 p50 / p90 73.4 / 65.0 对 70.3 / 61.0（+4.4% / +6.6%）；TTFT p50 / p90 4.61 / 12.39 对 4.25 / 9.64 s
（+8% / +29%）。帕累托（交互 p90 对吞吐）上右移；TTFT p90 代价大于预期。权重 40 万来自步长拟合
`21.3 + 0.70·最忙请求 + 1.66·最忙上下文(百万)`（1 个请求 ≈ 42 万 token）。

### 7.4 MTP K 按负载选

| 配置（单节点 C64） | 吞吐 | 交互 p50 / p90 | TTFT p90 |
|---|---|---|---|
| K3（2.99） | 41,557 | 83.1 / 76.2 | 3.90 s |
| K5（3.61） | 42,356 | 85.8 / 75.5 | 3.83 s |

同负载步长比 K5 / K3：@1-3 1.09-1.13、@6-12 1.13-1.19、@14-18 1.21-1.23（盈亏线 1.207）→ 交叉点约在最忙 decode rank 14-16 个请求。
C64 及以下用 K5，C104 及以上用 K3，C88 两种都测。

K5 C88 在 `mem-fraction-static 0.95` 下 3/3 崩溃（HSA OUT_OF_RESOURCES），原因是 decode 显存余量不够：K5 捕获完 graph 后每卡只剩 2.1 GB（K3 6.0），
高负载时运行中还要多占 3.6-4.9 GB。改 0.93 后跑完（1800 s）：45,081 / 交互 p90 57.8，不如 K3 C88（50,316 / 63.9），C88 用 K3。
K5 高负载建议 0.92 或更低（QA-20261106 第五节第 8 小节）。

### 7.4a v2 单节点 4+4 Pareto（v2 kit，`MC_DISABLE_HIP=1`，3600 s；11-07 04:54 更新）

| 点 | 吞吐 | 交互 p50 / p90 | TTFT p90 | 前沿 |
|---|---|---|---|---|
| K5 C32 | 23,369 | 118.0 / 101.5 | 2.26 s | 被 DP 均衡 K5 C32 取代 |
| **DP 均衡 40 万 + K5 C32** | 23,530 | 121.6 / 106.8 | 2.39 s | 是 |
| K5 C48 | 32,384 | 103.9 / 90.5 | 2.76 s | 被 DP 均衡 K5 C48 取代 |
| **DP 均衡 40 万 + K5 C48** | 32,819 | 108.0 / 97.8 | 2.76 s | 是 |
| K5 C64 | 42,284 | 86.3 / 74.0 | 3.86 s | 被 DP 均衡 K5 C64 取代 |
| **DP 均衡 40 万 + K5 C64** | 43,245 | 91.3 / 81.6 | 3.92 s | 是 |
| K3 C88 | 50,316 | 73.3 / 63.9 | 5.61 s | 否（DP 均衡 K3 C88 待测，26c） |
| K3 C104 | 52,409 | 70.5 / 60.6 | 11.00 s | 被 DP 均衡 K3 C104 取代 |
| K3 C112 | 53,026 | 69.0 / 60.9 | 13.63 s | 否 |
| **DP 均衡 40 万 + K3 C104** | 53,460 | 73.0 / 65.4 | 11.27 s | 是 |

16 卡形态：1P(PP8) + 1D C208 56,593 / 70.9（RESULT 7.5，pkg kit），2P1D C208 54,094 / 65.7；12 卡 1P2D 不进前沿（7.6a）。

### 7.5 prefill 侧

- 2P1D C208 + hrrn 51,804（SPF 54,094，不同节点对）：> 128K 请求 TTFT p99 1,017 → 302 s（饿死解除），但短请求 TTFT 约翻倍，
  整体 TTFT p90 8.0 → 17.1 s。计分保留 SPF。
- E9 重写：未做。
- **1P(PP8) + 1D（TP8 / DP8 decode）C208 56,593**（11-06 23:36，n04-25 → n04-29，16 卡，pkg kit + fdbafb60，K3 / 2.99）：目前最高。

| 配置（C208，16 卡） | 吞吐 | 交互 p50 / p90 | TTFT p50 / p90 | prefill 现算 / 排队均值 | decode 最忙 rank 中位 / p90 |
|---|---|---|---|---|---|
| 2P1D（2 × PP4 prefill，SPF） | 54,094 | 77.2 / 65.7 | 4.16 / 8.01 s | — / 6, 6 | — |
| 1P(PP8) + 1D | **56,593** | 86.0 / 70.9 | 7.50 / 14.47 s | 20,453 tok/s / 52.6 | 12 / 17 |

  吞吐 +4.6%、交互 p90 +7.9%，TTFT p90 +81%（单个 PP8 prefill 92% 时间有排队）。同交互下比 4+4 前沿（插值约 49,990）高约 13%。
  PP8 gate（11-05 23:44）：smoke 6/6、探针 0.985、16k 32/32 + 31/32、接受长度 3.60。

### 7.6 v2 kit（Infera `feat/sglang-pd-glm5.2-agentx`，main 44a294f8 镜像）

- 镜像 11 个被替换文件 md5 与 BASE_MD5SUMS 一致；单节点生产全套正确性与参照持平。
- `MC_DISABLE_HIP=1` 已提交（347b9b38，README note 12）；C104 52,409（同节点基线 53,057，-1.2%，单轮）。
- `MTP_STEPS`（K3 / K4 / K5，sweep 按 golden 表设接受长度）、`DECODE_WORKERS=2`（1P2D）已提交（bf61a7c7，11-06 21:35）。
- v2 两节点 4+4 C104 53,055（11-07，n02-33 → n04-33）：pkg kit 两节点 53,666（-1.1%）、v2 单节点 52,409（+1.2%），v2 在两节点上无倒退。
- 待跑：单节点 Pareto（K5 C32 / C48 / C64 / C88、K3 C88 / C112、DP 均衡 C64 / C104），两节点 1P2D C96 / C128、v2 两节点 4+4。

### 7.6a 1P2D（v2 kit `DECODE_WORKERS=2`，11-06，n04-25 PP4 prefill → n04-29 两个 TP4 / DP4 decode，12 卡，K3 / 2.99）

- gate：smoke 6/6、GSM8K 探针 193/200（乱码 0、开头坏 0）、16k 捞针 32/32、真实接受长度 3.69。

| 点 | 吞吐 | 交互 p50 / p90 | TTFT p50 / p90 | req/s | 4+4 前沿同交互（插值） |
|---|---|---|---|---|---|
| C96 | 38,094 | 93.8 / 84.3 | 3.28 / 6.07 s | 3.569 | 约 39,950（K5 C48 32,384 @ 90.5 与 DP 均衡 K5 C64 43,245 @ 81.6 之间） |
| C128（11-07 重跑，n02-33 → n04-33） | 37,999 | 88.5 / 78.2 | 3.93 / 7.22 s | 3.912 | 约 45,400（同上两点之间） |

- 第一次 C128（11-06）预热末尾 1 个请求挂住、看门狗停止，无结果；重跑正常。
- **1P2D 被单个 PP4 prefill 限住**：C96 → C128 req/s +9.6%，每 GPU 吞吐持平（38,094 → 37,999）。总吞吐约 456K tok/s，4+4 C104 约 428K，多 4 张卡只多约 7%。
  同交互比 4+4 前沿低 5%（C96）/ 16%（C128），不进前沿。用 K5 也改变不了 prefill 上限，不再测。

### 7.7 调度经验

- 作业时限越短越早开始（`sbatch --test-only`：两节点 6 h 估当天 12:32，24 h 估次日 05:31）；两节点 6 h、单节点 8 h。
- 抢占先发 SIGTERM、约 5 分钟后杀作业；guard 步骤在 SIGTERM 时删掉我们的容器，条目收到 SIGTERM 后停住、下次分配重跑。
- 多个单节点作业共用一个队列：`running/<name>.job` 记录所属作业，时间不够的条目放回队列（`opt_item5.sh`）。
- 分到的节点上别人的 GPU 容器起栈前 `docker stop`（`clear_foreign_containers.sh`）。
