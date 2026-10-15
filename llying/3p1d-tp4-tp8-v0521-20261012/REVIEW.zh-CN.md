# 复盘：3P(TP4)1D(TP8) on sglang v0.5.21（2026-10-12 17:50 → 10-13 02:50 UTC）

数据：`results/r2/c<CONC>/`（AgentX JSON、`decode_balance.txt`、`prefill_balance.txt`），汇总 `results/summary.csv`；
过程：`operations/STATUS.zh-CN.md`。每卡吞吐按 20 张 GPU 计。所有点在同一个栈（r2）上按 144→160→176→128→112 顺序测。

## 1. 结果

| CONC | tok/s/GPU | 输出 tok/s/GPU | 完成请求 | 平均输入 | P50 / P90 intv | TTFT p50 / p90 (s) | GPU 缓存命中 | decode 有效并发 | 首 token 前在途 |
|---|---|---|---|---|---|---|---|---|---|
| **112** | **20,545** | **174.3** | 12,319 | 120.0k | 71.6 / 41.6 | 6.0 / 16.1 | 91.1% | 50.3 | 28.4 |
| 128 | 19,758 | 173.4 | 12,376 | 114.9k | 71.3 / 40.5 | 7.8 / 30.4 | 86.6% | 50.5 | 44.7 |
| 144 | 18,001 | 157.1 | 11,393 | 113.7k | 52.9 / 34.8 | 10.2 / 55.1 | 74.8% | 59.0 | 68.0 |
| 160 | 17,754 | 153.1 | 11,079 | 115.3k | 75.4 / 39.5 | 19.6 / 85.2 | 75.6% | 43.2 | 104.6 |
| 176 | 16,298 | 136.1 | 10,009 | 117.2k | 96.9 / 49.4 | 38.5 / 127.2 | 62.1% | 30.8 | 147.2 |

- 吞吐最高的是 **C112（20,545）**；但 C112 与 C128 完成请求数、输出吞吐几乎相同，总吞吐 +4% 主要来自平均输入更长（+4.5%），
  两者应视为同一平台。**峰值区间是 C112–C128**；更低的 C96 没测，峰值是否继续向左未确认。
- 原定的 C144/C160/C176 都在峰值右侧，单调下降（C176 比 C112 低 21%）。
- interactivity：P50 全部在 50–100；P90 全部 < 50（InferenceX 官方口径下无一达标）。
- 每点 error-dropped 3–11 个，测量期内真正的传输失败只有 1 次（C176 23:18 的 aux 发送失败）；窗口结束时的取消会在 decode
  日志里留下 KVTransferError，属预期。

## 2. 为什么并发越高吞吐越低（数据链）

| CONC | prefill 缓存份额 | prefill 新算 tok/s | prefill 交付总 token（窗口内） | prefill 排队 | decode 每 rank 在跑 | decode 等预分配 | decode 等传输 | decode KV>90% |
|---|---|---|---|---|---|---|---|---|
| 112 | 93.5% | 26.6k | 14.8 亿 | 1.4 | 5.87 | 0.04 | 2.1 | 1.9% |
| 128 | 91.0% | 35.5k | 14.2 亿 | 2.5 | 6.06 | 0.21 | 3.9 | 6.2% |
| 144 | 87.9% | 43.6k | 13.0 亿 | 3.5 | 7.01 | 1.60 | 5.6 | 19.7% |
| 160 | 86.1% | 49.2k | 12.8 亿 | 5.3 | 5.03 | 2.97 | 8.8 | 28.3% |
| 176 | 83.4% | 53.9k | 11.7 亿 | 7.1 | 3.75 | 5.60 | 11.9 | 41.4% |

1. **前缀缓存容量先到顶**：并发越高，同时活跃的会话越多，工作集超过 prefill 的前缀缓存（GPU KV + HiCache，HiCache 全程 ~100% 占用），
   缓存命中份额 93.5% → 83.4%。
2. **prefill 做得更多、交付得更少**：新算 token 速率翻倍（26.6k → 53.9k/s），交付的总 token 却从 14.8 亿降到 11.7 亿；prefill 排队 1.4 → 7.1，
   TTFT p90 16 s → 127 s。（AgentX 的总吞吐 ≈ 99% 是输入 token，基本等于 prefill 交付的总 token。）
3. **decode 被拖累**：首 token 慢，已在 decode 预分配 KV 的请求等传输的时间变长，占住 decode 显存（KV>90% 时间 1.9% → 41.4%），
   decode 实际在跑的请求反而减少（有效并发 C144 59 → C176 31）。另外 decode 不上报 KV 事件，router 选 D rank 按请求数轮转，各 rank
   KV 不均衡（C144 时 DP3 99%、33 个等预分配，其余 48–80%）。
4. 3 个 prefill 实例负载均衡（135 上 GPU 0,4,5,6 + 7 张网卡的实例与 137 上两个实例相当），不是单点瓶颈。

与 yaocheng 2P1D（v0.5.19，24 卡）相比：本轮峰值 20,545/GPU @C112 对其 18,807/GPU @C192（+9%），C144 时 18,001 对 18,519（−3%）。
本轮峰值出现在更低并发，与前缀缓存容量一致：本轮 12 个 prefill rank × 153 GB HiCache ≈ 1.8 TB、GPU KV 12 × 2.27M ≈ 27M tokens；
2P1D 为 16 × ~212 GB ≈ 3.4 TB、16 × 3.14M ≈ 50M tokens，约 1.85 倍。（镜像、卡数、实例形态都不同，只作方向性参考。）

## 3. 正确性

- **r1（grouped topk=1，即 yaocheng 配置）不正确**：已知答案门禁 0/48；18 tokens 的 prompt 正确，1.7k–2.7k tokens 约 40% 正确；
  prefill 首 token 分布异常（`Z` 概率 ~0.4）。改 `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0`（config.sh 默认、yihou v0.5.21 取值）后修复。
  这也说明此前 v0.5.21 的吞吐实验都只在模拟接受下跑，没人在真实接受下验证过正确性。
- **r2（当前配置）**：简单文档（1.7k/2.7k tokens）95/96；41k 文档 65/80；123k 文档 7/16。非正确答案全是文档内真实内容：
  另一条记录的代码（相邻 ±1/±10，或编号相近如 187→1871），或数字正确但缺开头 "Z"（2 次，仅长文档）；0 个编造/乱码。
- **未闭合**：没有已知正确部署在同一批题上的正确率做参照，长文档上的检索错位与偶发缺 "Z" 是否模型（DSA top-2048）本身极限，
  还是 v0.5.21 长上下文的细微问题（fused DSA indexer、HiCache 读回的 KV 等），尚不能下结论。吞吐测试用模拟接受（输出按设计为乱码），不检查正确性。

## 4. 过程中的问题与教训

| 时间 | 问题 | 影响 | 教训 / 处理 |
|---|---|---|---|
| 18:05 | dryrun 漏了 `DRY_RUN=1`，实际起栈 | ~20 min（转用作真机验证与 JIT 预热） | 已给 launch.sh 加 `DRY_RUN=1` 模式；高代价命令执行前复核参数 |
| 18:26 | 预检循环 / 同节点启动门里的 ssh 读走了 `while read` 的 stdin：只检查了 135；decode 那一行被吞没启动 | decode 漏起 | 循环内 ssh 一律 `</dev/null` 或 `-n`；yihou 同节点只有两行所以没暴露 |
| 18:33 | 在登录节点 mv 替换正在 137 上运行的 NFS 脚本 → Stale file handle | launch 中断 | 运行期间不改在用脚本（mv 也不行），要改就另存新文件 |
| 18:33 | TP8 decode 8 rank 同时编译撑满 8g tmpfs | decode 起不来 | `ENGINE_TMPFS_SIZE=32g` |
| 18:52 | 门禁 0/48 | 排查约 15 min | 先用短/长、同/跨 rank、首 token logprob 逐层排除 → grouped topk |
| 19:16 | 困难集 95% 门槛定得过严（无基线） | 误判失败 | 门禁改为两层；但基线问题仍未闭合（见 §3） |
| 19:24 | 容器以 root 建的缓存目录，本用户 mkdir 失败，三点秒败 | ~6 min | 种子缓存后 chown；bench 循环加"无结果即停" |
| 02:44 | `pgrep -f` 匹配到执行命令的 shell 自身 | 无（只误杀自身） | 不在包含模式串的命令行里用 `pgrep -f`/`pkill -f` |

做得对的：Mooncake 四个补丁先离线验证可叠加、构建内断言；镜像三台 ID 一致；正式起栈前真机验证 + JIT 种子；流水线脚本化、
在控制节点 setsid 运行；结果按点自动发布；按"峰值在右侧就向左补点"的规则自动加测 C128/C112。

## 5. 待决与建议（按优先级）

1. **补 C96**：确认峰值是否在 C112 以左（同一栈即可，约 75 min）。
2. **闭合长上下文正确性**：固定种子的同一题集，对照已知正确部署（v0.5.19 镜像 fd7220a5，b4 门禁曾通过）或非 PD 部署；必要时 v0.5.21 关 fused indexer
   对照；对缺 "Z" 的题取首 token logprob。
3. **调优方向（针对上面的数据链）**：
   - 前缀缓存容量：137 上 8 个 prefill rank 共用 2.7 TB 内存，HiCache 可从 153 GB/rank 提到 ~200 GB/rank（TP8 时的水平）；135 只有 1 个实例（4 rank），
     可到 ~450 GB/rank（需 `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512`）。
   - decode KV：`--disaggregation-decode-allocation-policy prefill_complete`（v0.5.21，避免等传输的请求占 decode 显存），decode mem 0.85 → 0.95。
   - decode 均衡：按 token 的 D rank 选择（b4 的 R2），需 router 改动，不属于"router 与 yaocheng 一致"的范围。
4. 单点只测了一轮，没有重复测量；不同点的平均输入长度相差最多 ~5%，几个百分点的差异要谨慎解读。

## 6. decode 理论容量与下一步方案（2026-10-13 03:15 讨论）

**更正**：§1 说"峰值区间 C112–C128"依据的是两点完成请求数持平（12,319 / 12,376）；但 C112 是已测最左点，左侧没有更低的点，严格说峰值未夹住。
吞吐已走平，C96 大概率更低，属确认性测量，优先级低。

> **更正（10-15）**：下表 KV 上限高估了约一倍。实测每个在跑请求占 KV 约 18–21 万 token（按 decode 停留步数加权，不是平均输入 12.5 万），
> 等 KV 传输的请求另占 7–12% 的 KV（等预分配的请求不占 KV）。按实测口径，mem 0.85 时 decode 约容纳 90 个在跑请求，与 C176 实测 88 个一致。
> 详见 `RESULT.zh-CN.md` §4.5。

**decode（8 卡 TP8/DP8，mem 0.85）两个上限**

| 上限 | 计算 | 结果 |
|---|---|---|
| KV 容量 | 2,992,512 tokens/rank × 8 = 23.94M；每请求 ≈ 数据集平均输入 121,385（中位 85,546，p95 389,970）+ 生成预留 ~4k ≈ 125k | ~190 驻留（~24/rank）；考虑长请求不均/碎片约 150；mem 0.95（+~535k tokens/rank）约 225 |
| ITL（P50 interactivity ≥ 50 → ITL p50 ≤ 20 ms） | 实测 ITL p50 vs 每 rank 在跑：3.75→10.3、5.03→13.3、5.87→14.0、7.01→18.9 ms | 每 rank ~7.5–10 → ~60–80 同时 decode |

换算到 AgentX：每条会话一个周期 = 空闲（C112 实测 ~9.7 s）+ TTFT + decode（≈ 输出 ~1,000 token × ITL）；
CONC ≈ decode 驻留 × 周期 / decode 时长。ITL 20 ms（decode ~20 s）、TTFT ~4 s 时，60–80 驻留对应 CONC ≈ 100–135，吞吐 ≈ 1.8–2.4 万 tok/s/GPU。

实测：C112/C128 decode 有效并发 ~50（3.39–3.41 req/s，TTFT 均值 8.4 / 13.2 s）；C144 升到 59 但完成速率降到 3.14 req/s；C160/C176 降到 43/31。
与 ITL 上限差 10–30 个，与 KV 上限差更多；加并发后 decode 驻留反而下降，原因在 prefill 侧（缓存命中下降、TTFT 上升）。

**方案评价**：顺序合理（固定 decode → 理论上限 → 对比 → 解决 prefill 瓶颈 → 推高吞吐 → 减卡/加速）。补充：
1. 先定约束（如 P50 ≥ 50），它决定 decode 上限是 ITL 约束的 60–80 还是 KV 约束的 ~190；
2. 理论值要扣除生成预留、已预分配待传输的请求（实测每 rank 2–12）和 rank 不均衡；
3. 根因是前缀缓存容量（不是 prefill 算力），加 prefill 卡前先试：HiCache 加大（137 ~200 GB/rank、135 ~450 GB/rank）、`prefill_complete`、decode 均衡；
   加一个 TP4 prefill = 卡数 +20%，需吞吐 +20% 以上每卡才不降；135 只剩 GPU 2,3,7 空闲，放不下 TP4，136 被占。

**优化点位**：以 C128（拐点）与 C144（开始变差）为成对评估点；达标 = C144 吞吐超过 C128 基线 19,758，decode 有效并发向 65–75 靠近，
prefill 缓存份额保持 ~90%，P50 interactivity ≥ 50；达标后再推 C160。ITL vs 每 rank batch 的曲线建议单独测 decode step 时间以确认。

## 7. 调优项优先级（2026-10-13 04:10，结合历史 router 调优数据）

历史数据来源：`../experiment-summary-20261008/REPORT.zh-CN.md`（P8D8 C80，v0.5.19）、`../b4-align-crsuse2-20261008/P8D4-OPTIMIZATION.zh-CN.md`（P8D4 C80）、
`../decode-kv-aware-mtp-radix-20260924/README.zh-CN.md`（P4D4 C40）、`../session-affinity-20260928/`。均为单轮测量，形态与本次 3P1D 不同，需在本形态复测。

**落地成本**：本次 v0.5.21 镜像的 router 源自 Infera `28121cb4`，已包含 `INFERA_PD_PREFILL_GUARD_RELEASE`（R1）、`INFERA_SESSION_AFFINITY`、
`INFERA_R2_DECODE_DEMAND`、`INFERA_R3_CACHE_TIERS/R3_HOST_WEIGHT`、`INFERA_R4_PREFILL_WORK`、`INFERA_P_DYNAMO_SCORE`（二进制 strings 已核对；
`R2_DECODE_HIT_CREDIT` 不在）。本轮"router 与 yaocheng 一致"即这些开关全关。router 类调优只需重启 router，不重启引擎；会话亲和还需客户端
发 `X-Dynamo-Session-ID`（InferenceX 918524ff 的 aiperf 支持 `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1`，需在 harness 的 runtime.env 加上）。

| 优先级 | 调优项 | 针对本次哪个瓶颈 | 历史证据 | 本形态的成本 |
|---|---|---|---|---|
| 1 | **R1：P 完成即释放路由记账** | P 排队 / TTFT；3 个 P 实例间的负载估计（原行为把 P 预约留到 D 结束，长 decode 时高估 P 负载） | P8D8 C80：Total +5.44%，TTFT p50/p90 −17%/−30%（同节点同进程）；扩容后再测 +3.8%、TTFT p90 −41% | router 环境变量 |
| 2 | **P 会话亲和（prefill 模式）** | 前缀缓存局部性：缓存分散在 3 实例 × 4 rank = 12 个 rank，命中份额随并发 93.5%→83.4% | 在 R1 之上：Total +1.9%，TTFT p50/p90 −10.5%/−21%，ITL +5.5%（P8D8 C80，D 换了节点） | router 环境变量 + 客户端 header |
| 3 | **加大 prefill HiCache** | 前缀缓存容量（本次总 HiCache ~1.8 TB，是 2P1D 的约一半） | 本次 r2 数据直接指向；但 P8D8 C80 单独把 P GPU KV +8%（host 不变）未见收益（不同形态/负载） | 引擎重启；137 ~200 GB/rank、135 ~450 GB/rank（需 CHUNK_GB=512） |
| 4 | **`prefill_complete` 分配策略** | decode 被等传输的请求占显存（C144→C176 每 rank 在跑 7.0→3.8） | 无历史数据（v0.5.21 新参数） | 引擎重启 + 正确性门禁 |
| 5 | **R2：decode 按 token 均衡** | decode rank 间 KV 不均（C144 DP3 99%、33 个等预分配） | P8D4（D 容量受限）+9.3%，热冷 rank 差 42→14 pp；加命中加分无额外收益 | router 环境变量；在推到 C144+ 时价值更大 |
| 6 | R3：分层缓存评分 | HiCache 命中对路由的可见性 | 未测 | router 环境变量，探索性 |
| 7 | decode mem 0.95 / decode graph bs 64 | decode KV 容量（目前不是约束） | P8D4 上 mem 0.90 + graph 64 → KV +12.6% | 引擎重启，小 |
| 8 | D radix（+ D 绑定） | P→D 传输量、decode 预分配等待 | P4D4 C40 +3.0%；P8D8 B2 +0.2%、B3 +0.6%（输出 +3.1%），D 本地复用 92%；D 显存紧时复用很低 | 需把 MTP 下开 D radix 的引擎补丁（PR #184）移植到 v0.5.21，且长前缀正确性未验证 |
| 不做 | D HiCache；R4；Dynamo 式首次评分；8K chunk | — | D HiCache −4.5%（每步 +1.7 ms）；R4 −34%；Dynamo 首次评分 −1.9%；8K chunk −4.5% | — |

建议节奏：1、2（及 5、6）都是 router 开关，可在当前 r2 栈上只重启 router 依次测（C128 与 C144），代价低；先单独测 R1，再叠加 P 会话亲和，
保留归因。router 项确定后再做引擎重启类的 3、4（必要时 7）。若仍是 prefill 瓶颈，再考虑加 prefill 实例（需另找节点）。

## 8. 更正：r2 C144 受"decode 刚重启后的第一次跑"偏慢影响（2026-10-13 04:20）

按同一时刻**最忙 rank** 的在跑请求数对齐 decode 每步耗时（`results/r2/decode_step_matched_load.txt`；DP attention 下各 rank 同步，步长取决于最忙 rank）：

| 最忙 rank 在跑 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|
| C144（decode 19:24 重启后的第一个点） | 41.8 ms | 47.3 | 61.3 | 72.4 |
| C160 / C176 / C128 / C112 | 36.0–37.4 | 40.4–41.2 | 51.1–53.1 | 60.4–63.9 |

后四个点彼此一致（±2–3%），C144 每档慢 13–17%，与 yihou 在 v0.5.21 记录的"新栈第一次跑 decode step 慢 7–12%"（`/flush_cache` 不能消除，跑过一轮后消失）同类。
因此 **r2 C144 的 18,001 偏低，C128 与 C144 的比较有偏**；C160/C176 的下降仍成立（均在 warm 状态）。峰值是否其实在 C144，需要在 warm 栈上复测 C144。
r2 各点之间没有 flush，前一轮的前缀缓存会带入下一轮（每轮有 15–25 分钟 warmup，影响应较小但存在）。
