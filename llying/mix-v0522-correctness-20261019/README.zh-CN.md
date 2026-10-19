# GLM-5.2 aggregated TP4 + DP4 attention：并发下输出乱码（10-19）

## 现象

在升级 sglang v0.5.22 的 example（`examples/sglang_mix_glm5.2`，分支 `feat/sglang-mix-glm5.2-tp4-v0522`）做端到端验证时，
GSM8K（lm_eval 0.4.13，5-shot、200 题，经 router，真实 MTP 接受）只有 0.69 / 0.74，远低于 v0.5.19 PD 时的 0.97。
答错的题里很多是乱码：上万字符、夹 `</think>`、反复出现其他题目的数字片段，像读到了别的请求的 KV。

| 镜像 / 设置 | 并发 | 题数 | strict / flexible | 乱码题（>8000 字符或含 `</think>`） |
|---|---|---|---|---|
| v0.5.22（`infera-sglang:v0522-llying-mix-1019b`），example 配方，GPU 0-3 | 64 | 200 | 0.74 / 0.69 | 63 |
| v0.5.22，同上 | 1 | 24 | 0.958 / 0.958 | 0 |
| v0.5.21（`infera-sglang:v0521-llying-1012-mc4340-destpin`），同配方但 HiCache 64 GB、直连引擎，GPU 4-7 | 64 | 200 | 0.745 / 0.690 | 59 |

结论：**不是 v0.5.22 引入的回归**，v0.5.21 上同样存在；只在并发（批处理）时出现。smoke（顺序 6 个请求）和此前的
负载下 decode 检查（`decode_check.py`，答案只有 4-5 个 token）都没有暴露它。性能测试用模拟接受长度、输出长度由 trace 决定，
吞吐数据不受影响，但这个配方直接上线会输出错误内容。

## example 性能对齐（10-19，138 GPU 0-3）

按 example README 手动起：`SIM_ACCEPT_LEN=3.61 bash engine/up.sh`，`CONC=48 bash engine/agentx/agentx.sh`，`bash engine/down.sh`。
计分窗口 20:21:48-21:21:48；其中 20:26-20:43 GPU 4-7 在跑 topk-torch-200 排查（之后暂停）。产物在 `results/example-c48/`，
指标用 `agg-tp4dpa-138-v0521-20261015/scripts/plot_agg_rounds.py` 的同一套 `measured()` 算。

| | a3 C48（v0.5.21 镜像，138，10-15） | example C48（v0.5.22 镜像，138，10-19） |
|---|---|---|
| tok/s/GPU | 31,605 | **32,809**（+3.8%） |
| 完成请求 / 每秒 | 3,409 / 0.947 | 3,494 / 0.964（0 错误） |
| interactivity p50 / p90 | 40.4 / 20.8 | 40.8 / **22.5** |
| TTFT p50 / p90 (s) | 4.8 / 14.5 | 4.5 / 14.0 |
| ITL p50 / p90 (ms) | 24.8 / 48.1 | 24.5 / 44.5 |
| decode ms/token | 30.6 | 30.3 |
| AIPerf 计入命中 / trace 上限 | 96.1% / 97.1% | 96.3% / 97.2% |

性能对齐（略好）。v0.5.22 的服务端命中计数口径变了：AgentX 实时日志的 `srv prefix_cache_hit` 显示 48.1%、引擎计数器拆分为
GPU 40.8% / 主机 7.4%，但 AIPerf 按 API `cached_tokens` 算的命中是 96.3%，与 a3 相同，吞吐也对得上，所以是计数口径而非命中下降。

## 二分

`scripts/engine_bisect.sh NAME [FUSE_TOPK=0] [HICACHE=0] [MTP=0] [DPA=0] [GATHERV=0] [DELAYER=0] [FUSION=0]`：在 138 上起
etcd + 引擎（example 配方去掉对应特性，HiCache 64 GB），64 道 GSM8K 同时发、直连引擎。结果在
`/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/bisect/<NAME>/runs/<NAME>/`（138 本地盘）。

### 更正：HiCache 不是必要条件

前几轮用 lm-eval 的回答数"乱码"（>8000 字符或含 `</think>`），据此得出"关 HiCache 为 0、需要 HiCache 和 DP attention
同时开"。**这个结论是错的**：漏数了空回答。模型一直不闭合 thinking、生成到 `max_gen_toks=16384` 时，glm45 reasoning
parser 把全部文本放进 `reasoning_content`，lm-eval 看到的 `content` 是空串。把空回答算上（"坏"= 长/含 `</think>` + 空）：

| 变体 | 关掉的特性 | strict | 长或含 `</think>` | 空 | 坏 / 64 |
|---|---|---|---|---|---|
| base | 无（example 配方，HiCache 64 GB） | 0.859 | 8 | 2 | 10 |
| hicache0 | HiCache | 0.891 | 0 | 7 | 7 |
| **dpa0** | **DP attention（纯 TP4，HiCache 开）** | **0.953** | 0 | 2 | **2** |
| mtp0 | EAGLE MTP | 0.891 | 2 | 4 | 6 |
| fusetopk0 | DSA fused top-k（`SGLANG_DSA_FUSE_TOPK=0`） | 0.812 | 6 | 6 | 12 |
| wb | HiCache 改 `write_back` | 0.844 | 7 | 3 | 10 |
| directio | HiCache I/O 改 `direct` + `page_first_direct` | 0.906 | 2 | 4 | 6 |
| b300like | `write_back` + `direct` + `page_first_direct`，不覆盖 `index_share` | 0.859 | 1 | 8 | 9 |
| idxshare1 | 去掉 `index_share_for_mtp_iteration=false` 覆盖 | 0.859 | 3 | 6 | 9 |
| gtt | `HSA_USERPTR_FOR_PAGED_MEM=0`（HiCache 主机池用 GTT） | 0.906 | 1 | 4 | 5 |
| gatherv0 | `SGLANG_DP_USE_GATHERV=0` | 0.906 | 3 | 5 | 8 |
| delayer0 | prefill delayer | 0.859 | 3 | 6 | 9 |
| **stock** | **上游镜像 `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`，`sglang.launch_server`，不打任何 Infera 补丁** | 0.891 | 0 | 7 | **7** |

空回答是不是乱码，看引擎日志：收尾阶段只剩 1 个请求、已超过 6000 token 的 decode 行里，hicache0 四个 DP rank 的 MTP
平均接受长度 1.03–1.36、一直跑到 17k token（上限），stock 的 DP2 也是 1.31；正常 decode 是 3.5–4.5。接受长度掉到 1
说明 target 输出的已经不是 draft 能猜的正常文本——就是乱码，只是写在 thinking 里。dpa0 收尾时 2 个请求合计 5.5k token、
接受长度 4.4，没有这种长尾（它的 2 个空回答另查）。

所以：
- **需要的是 DP attention，不是 HiCache。** 只有 dpa0 没有乱码长尾；关 HiCache、换任何 HiCache 组合都还在。
- **上游无补丁镜像同样出现**，不是我们的 DSA 补丁集或 HiCache 补丁引入的；v0.5.21 也有（见上表）。
- 之前读 HiCache 代码排除的几点（delayer 否决在 `init_load_back` 之前、无载回时 `start_loading()` 返回 -1、HiCache
  同步用 attention TP 组）仍然成立，只是不再相关。上游 sglang#30321（GLM-5.2 + MTP + HiCache 乱码）对方是纯 TP4，与这里不同。
- 上游 CUDA 上 GLM-5.2-NVFP4 TP4/DP4 + EAGLE 的 GSM8K 是 0.948（sglang#32196 的验证），这里同构型只有 0.86–0.89，
  更可能是 ROCm 上 DP attention 路径（aiter / triton DSA / MXFP4）特有的问题。

### 换判据：`scripts/gsm8k_probe.py`

lm-eval 只看 `content`，判不了这个问题。`gsm8k_probe.py` 把 lm-eval 样本里的 GSM8K prompt 原样直连引擎重放（temperature 0，
`max_tokens` 16384），每个请求记 `finish_reason`、生成 token 数、`reasoning_content` 与 `content`，给两个数：
- **garbled**：正文（thinking + 回答）里出现 `</think>`、`</arg_value>` 等模板标签的字面文本。真的特殊 token 会被 parser
  剥掉，只有解码坏了才会把它们当普通文本吐出来；hicache0-n200 的 60 条长回答全部命中，dpa0 一条不中。
- **runaway**：被 `max_tokens` 截断。干净的模型也会在个别有歧义的题上反复推敲到上限（dpa0 的 2 条即此，逐字看过是连贯推理），
  所以只作参考。

**200 题、并发 64 比 64 题同时发灵敏得多。** 同样关 HiCache，64 题同时发只有 7 条空回答，200 题持续发有 60 条乱码。
64 题同时发时 prefill 全挤在开头，之后是纯 decode；持续发时新请求不断进来，总有"一个 DP rank 在 prefill、其余在 decode"
的混合步——这一步全局不能走 CUDA graph，decode 的 rank 走 eager，DP gather 用 `SUM_LEN` 布局（`dp_attention.py`
`get_dp_padding_mode`），纯 decode 步则是 `MAX_LEN` + CUDA graph。之后都用 200 题、并发 64（prompt 取 hicache0-n200 的样本）。

| 变体 | 设置 | 题数 | correct | garbled | runaway |
|---|---|---|---|---|---|
| base-p | 基线，64 题同时发 | 64 | 0.859 | 9 | 8 |
| **dpa0-200** | 关 DP attention（纯 TP4，HiCache + MTP 开） | 200 | **0.980** | **0** | 2（连贯的过度推敲） |
| cg0-200 | 基线 + `--disable-cuda-graph` | 200 | — | — | 引擎第一个 decode 步即崩溃 |
| **mtp0-200** | **关 MTP** | 200 | 0.685 | **61** | 59 |
| fusion0-200 | 关 `--enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion` | 200 | 0.710 | 56 | 59 |
| mtp0cg0-200 | 关 MTP + `--disable-cuda-graph`（全部 eager，避开上面的 draft 崩溃） | 200 | 手动停止，无汇总 | | 见下 |
| lmhead0-200 | 关 `--enable-dp-lm-head` | 200 | 0.675 | 55 | 60 |
| dec-tilelang-200 | DSA decode 后端 `triton` → `tilelang` | 200 | 0.670 | 55 | 1 |
| topk-torch-200 | `--dsa-topk-backend torch --speculative-dsa-topk-backend torch` + `SGLANG_DSA_FUSE_TOPK=0`（DSA top-k 改用 `torch.topk`；torch 后端不支持 fused top-k，不关会在 graph 捕获时报错） | 152 / 200 | 135 / 152 | 12 | 8 |

**MTP 也不是必要条件。** 关 MTP 后同样 61 / 200 乱码。至此 HiCache、MTP、prefill delayer、GatherV、融合算子、
DP lm-head、HiCache 各种写法都排除，**唯一必要的是 DP attention**，且要在持续并发下（有混合步）。

**关 CUDA graph 后长尾少了一个数量级。** mtp0cg0-200 全部 eager，decode 每请求只有约 8 tok/s，跑满 16384 token 要半小时以上，
为了把 GPU 0-3 让给 example 的性能对齐，healthy 后 14 分钟手动停掉（`verdict.txt` 有记录）。停时只剩 6 个请求在跑
（各 rank 合计 3.6k-12.8k token）；对照开 graph 的 mtp0-200，healthy 后 4 分钟每个 rank 还有 13-16 个请求、平均各 1 万
token 以上。这 6 个是不是乱码没来得及看（之后用逐条落盘的新版 `gsm8k_probe.py` 补测）。

v0.5.19 的 3P1D PD 部署 GSM8K 是 0.97，那里 decode 节点同样是 DP attention + CUDA graph，但 decode 节点从不做 prefill，
没有混合步。合起来的工作假设：**聚合部署里 eager 的混合步（DP gather `SUM_LEN`）与 CUDA graph 的纯 decode 步
（`MAX_LEN`）交替时，有状态没有对齐**，graph 回放读到了错位的数据。尚未定位到具体代码。

**cg0-200 的崩溃。** 全部走 eager 后，第一个 decode 步在 MTP draft 的 `_draft_extend_for_decode` → NextN 层 DSA indexer →
`fast_topk_transform_fused` 报 `RuntimeError: lengths must have one entry per score row`，随后 GPU memory access fault。
即 DP attention 下 eager 的 draft-extend 路径，打分矩阵的行数与 `lengths` 条目数不一致（纯 decode 步 eager 时 DP gather 是
`MAX_LEN`，行数按全局最大 token 数填充）。平时纯 decode 走 CUDA graph 绕开了这里；混合步也走 eager，但 `SUM_LEN` 不填充，
行数凑巧对得上。draft 算错只会拉低接受率、不会让输出乱码（target 逐个验证），所以它不是乱码的直接原因，但说明
**DP attention 下 eager 路径的 DSA 元数据与 DP 填充对不齐**；target 的 verify / decode 若在同类路径上行错位而不报错，
就会按错的 top-k 去读 KV，正是乱码的样子。

**同节点两个引擎会撞 RCCL 共享内存名。** 两个容器都 `--ipc host`，容器内 PID 都从很小的数开始，后起的那个在
`ncclCommInitRank` 报 `HIP failure: 'invalid argument'` 退出（hicache0 前两次即此）。并行跑两个排查栈时，第二个先空转把 PID
推高（`PID_OFFSET=30000`，与 mixed 套件的 `PID_OFFSET_mixed_0` 同理）。

**换 DSA decode 后端（tilelang）同样乱码，但形态变了。** 55 条乱码里只有 1 条跑到上限，其余都以 `stop` 结束（500-4.8k token），
内容是成串的 `!!!!`（0 号 token）夹模板标签。`!` 连串是 logits 全 NaN / 全 0 时 argmax 的典型结果；triton 后端下则是数字和
标签碎片。两个后端读到的像是同一批无效数据，只是容错不同——它们共用的上游是 DSA indexer 给出的 top-k 索引，所以下一项把
top-k 换成 `torch.topk`（`--dsa-topk-backend torch`）。

**top-k 换成 `torch.topk` 也乱码。** topk-torch-200 跑到 152 题时已有 12 题乱码（另 48 题未完，多为长尾），为了不和 example 的性能
对齐（GPU 0-3，计分窗口 20:22-21:22）重叠太久，20:43 手动停掉（`verdict.txt` 记部分结果；20:26-20:43 与该窗口重叠，性能结果
里注明）。top-k kernel、DSA decode 后端都排除后，剩下的是 indexer 打分及更上游、DP gather/scatter 与 MoE、以及 overlap 调度。
example 计分窗口内 GPU 4-7 不再跑排查。之后：`--disable-overlap-schedule`；上游 v0.5.20 / v0.5.19 镜像的聚合 DP attention
（找出从哪个版本开始出现）。

**乱码是 decode 中途逐步出现的，不是 prefill 写坏。** mtp0-200 的 61 条乱码里，第一个模板标签出现在回答的第 721 / 2,049 / 8,033
个字符（p10 / p50 / p90），开头 200 字符内出现的只有 1 条。逐条看，开头都是连贯、正确的长推理，然后先零星冒出单个错 token——
`50,000 \u200b\u200b000`（零宽空格）、`17!02`、`7辩论`、`I's`——之后越来越密，最后崩成成串的数字、`!` 和模板标签。也就是说，
某些 decode 步里某一行的 logits 突然是错的，一个错 token 让模型偏离，错误再累积。`!` 是 0 号 token，通常意味着那一行那一步的
hidden state 是全 0 / NaN，而 DP 填充行正是用 0 填的：像是某些步里真实请求的行在 gather / scatter 或 graph 输入输出的切片上
错了位，读到了填充行。

相关的上游记录：sglang#20404（MI355X，PD + DP attention，DeepSeek-R1 输出退化成重复/空串，关 DP attention 94%；2026-03，
v0.5.9，仍 open）；sglang#37192 第 3 点（DP 下 batch 先补到全局 token 数，graph runner 再补到捕获桶，第二次补齐不进 DP 缓冲区
长度——说的是 prefill graph，但与这里怀疑的"补齐与切片错位"同类）。

**大块 HiCache 栈撤下后，同节点新引擎会卡住。** example 栈（4 × 270 GB 锁页主机池）21:27 `down.sh` 后，21:28 在 GPU 4-7 起的
stock0520 引擎进程停在 `D` 状态、没有任何日志，直到 21:42 GPU 0-3 的显存读数归零才继续——驱动回收前一个栈期间，新进程的 KFD
初始化被挡住。64 GB 主机池的排查栈之间没有这个现象。

### 上游旧版本：v0.5.19、v0.5.20 也有

为了找出从哪个版本开始，用上游镜像（`STOCK=1`，不打任何补丁）跑同一配方，关 MTP、关 HiCache（避开旧版缺 DSA 补丁时的崩溃；
v0.5.22 上这两项单独关掉都还乱码）：

| 变体 | 镜像 | correct | 模板标签乱码 | 截断（runaway） |
|---|---|---|---|---|
| stock0520 | `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260924` | 0.830 | 0 | 21 |
| stock0519 | `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917` | 0.870 | 0 | 26 |
| 对照 dpa0-200 | v0.5.22（我们的镜像），关 DP attention | 0.980 | 0 | 2（连贯） |

两个旧版本没有模板标签，但截断的回答逐条看都是坏的：v0.5.20 是 `tsetsetset…`、`0.0.0.0…`、`1111…`、`magic magic magic…`、
`field field field…`；v0.5.19 是 `0.8 * 60 = 48.` 无限重复、`,0,0,0…`、`1008 years8 years8…`，还有一条模型自己写
"I will not include the weird glitching text in the final output"。所以**这不是 v0.5.19 之后的回退**，ROCm 上聚合 DP
attention 一直有；版本之间只是表现不同（v0.5.21 起多了模板标签外泄）。判据上，`garbled`（标签）只抓到新版本的形态，
**`runaway` 在干净配置下只有 2 条连贯的过度推敲，作为第二指标更通用**（乱码时 20-60 条）。

注意部分统计会偏乐观：长尾总是最后结束，stock0519 跑到 174 题时还是 0 / 0，剩下的 26 题全是截断。

v0.5.19 的 3P1D PD 部署 GSM8K 0.97，那里 decode 节点同样是 DP attention + CUDA graph，区别是 decode 节点从不做 prefill。
下一步直接检验"混合步"：`--prefill-delayer-max-delay-passes 2000`（混合时最多等 2000 步，持续负载下各 rank 基本都等到
"全部可 prefill"再一起做，GSM8K 的 prompt 不分块，混合步基本消失）；同时测 `--disable-overlap-schedule`。

### 混合步不是触发条件

| 变体 | 设置 | correct | 模板标签乱码 | runaway |
|---|---|---|---|---|
| dly2000-200 | 基线 + `--prefill-delayer-max-delay-passes 2000` | 0.705 | 61 | 60 |

把引擎日志的 `Prefill batch` 行按 30 ms 间隔分组（同一前向步内各 rank 的日志相差几毫秒），看每个 prefill 步有几个 rank 在做：
mtp0-200 是 51 步里 17 步只有 1 个 rank（其余 rank 在 decode 或空转），dly2000-200 是 40 步里 36 步 4 个 rank 一起、只有 4 步不齐。
**混合步基本没了，乱码一条没少**，前面"eager 混合步与 graph 纯 decode 步交替"的工作假设不成立。

`--disable-overlap-schedule` 与 prefill delayer 不能同时开（`AssertionError: To use PrefillDelayer, disable_overlap_schedule must
be False`），ovl0-200 改为同时关 delayer 重跑（delayer0 已证明 delayer 本身不是原因）。

目前站得住的：只要 DP attention；关 CUDA graph 后长尾少一个数量级；持续到达（200 题 / 并发 64）比一次发完（64 题）严重得多。
上游在 v0.5.22 的 `models/deepseek_common/moe_transport.py` 里已经写明 "Pad rows are not refreshed during graph replay. Stale
bf16-max values can overflow the MoE and contaminate the combine output"，并在 HIP + DP attention + TP MoE 时把 DP gather 前的
填充行清零（`SGLANG_DP_MASK_PAD_ROWS`，默认开；`decode_cuda_graph_runner._set_dp_real_num_tokens` 每次回放写真实行数）。
清零后的填充行全部路由到同一组专家，MoE 负载极度倾斜——下一项把 `AITER_USE_FLYDSL_MOE_SORTING` 关掉（docker 对重复的
`-e` 取最后一个，容器内已确认为 0）。

| 变体 | 设置 | correct | 模板标签乱码 | runaway |
|---|---|---|---|---|
| flydsl0-200 | `AITER_USE_FLYDSL_MOE_SORTING=0` | 0.700 | 55 | 60 |
| ovl0-200 | `--disable-overlap-schedule` + 关 delayer | 200 题 | 53 | 53 |
| **mtp0cg0b-200** | **关 MTP + `--disable-cuda-graph`（重跑，逐条落盘）** | **197 / 199** | **0** | **0** |
| maskpad0-200 | `SGLANG_DP_MASK_PAD_ROWS=false`（关掉上游的填充行清零） | 0.705 | 52 | 61 |
| breakable-200 | `--cuda-graph-backend-decode breakable` | 起不来：DSA 不支持（`'NoneType' object has no attribute 'get_page_table_64'`） | | |

MoE 排序实现、overlap 调度也排除。一个规律：乱码条数与 EXTEND 步数大致相当——base-p（64 题一次发完）6 个 prefill 步、9 条乱码；
mtp0-200 51 步、61 条；dly2000-200 40 步、61 条——像是每个 EXTEND 步之后坏掉约一个请求，而全 eager 时不坏。EXTEND 步本身两种
模式下一样（总是 eager），不同的只是之后的 decode 走 graph 还是 eager；PD 的 decode 节点只有 decode 步，所以没事。工作假设改为：
**eager 的 EXTEND 步与随后的 decode graph 回放之间，有共享状态没对齐**（v0.5.22 的 decode 输入缓冲区注册表与 ForwardBatch 共享
存储，见 `cuda_graph_buffer_registry.py`）。

**关掉 decode CUDA graph 就干净。** mtp0cg0b-200（关 MTP、`--disable-cuda-graph`，其余同基线：DP attention、HiCache、delayer、
融合算子都开）跑到 199 题：197 对、0 乱码、0 截断（22:41 手动停，剩 1 题在 eager 下以约 8 tok/s 慢慢跑，未到 7k token）。
对照同配置开 graph 的 mtp0-200：61 乱码、59 截断。加上 dpa0（关 DP attention，开 graph）也干净，结论是：
**DP attention 下的 decode CUDA graph 路径出错**，与 MTP、HiCache 无关；eager decode 太慢，不能当绕过办法。
下一项 `--cuda-graph-backend-decode breakable`：graph 分段捕获、注意力层 eager——干净则问题在 graph 内的注意力（DSA indexer /
注意力对 DP 填充行的处理），仍乱码则在 graph 内的 MoE / DP 通信；若干净且性能可接受，也是一个绕过办法。

上游的填充行清零开关（`SGLANG_DP_MASK_PAD_ROWS`）开或关结果一样，不是它；`breakable` graph 后端不支持 DSA，没法用来区分注意力
与 MoE；元数据 glue graph（`SGLANG_ENABLE_METADATA_GLUE_GRAPH`）默认就是关的。下一项用 v0.5.22 自带的 KV canary
（`--kv-canary log --kv-canary-real-data PARTIAL --kv-canary-sweep-interval 50`）：写 KV 时记内容哈希、读和定期巡检时校验。
有违规说明 graph decode 里有写坏别人 KV 的（如填充行），没有则问题在 hidden state / logits 一侧。配置简化为关 MTP、关 HiCache
（v0.5.19 / v0.5.20 上这个配置同样乱码），并跑同配置不开 canary 的对照 mtp0hc0-200。
