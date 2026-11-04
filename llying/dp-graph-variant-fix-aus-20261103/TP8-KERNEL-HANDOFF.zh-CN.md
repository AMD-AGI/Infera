# 交接：2P1D 的 TP8 decode kernel 优化（11-04 10:45）

背景、数据和判断都来自本目录的 `RESULT.zh-CN.md`（第 4.4、4.6、4.7、5 节）。本文件只列出接手需要的东西。

## 1. 要解决的问题

2P1D（2 × PP4 prefill + 1 × TP8 / DP8 decode）的 decode 每步固定开销 17.8 ms，比按权重字节算的带宽下限（约 13.6 ms）高约 4 ms。
TP4（4+4）的固定开销则正好在下限上（21.3 对 21.6 ms）。trace 定位到 **MoE**：TP8 下 MoE 的三个部件都没用上 yihou 的 KF kernel，
合计约有 **5.6 ms/步的空间（verify 步长的 13%）**；通信与 TP4 基本相同，不是问题。

## 2. 部署与配置

| 项 | 值 |
|---|---|
| 节点 | AUS `smci355-ccs-aus-n06-25`（prefill，GPU 0–3 / 4–7 各一个 PP4）→ `smci355-ccs-aus-n06-33`（decode，GPU 0–7，TP8 / DP8 attention，EP1） |
| 镜像 | `infera-sglang:v0522-fdbafb60`（两台都有；tar 在 `/perf_apps/liyingli/images/infera-sglang-v0522-fdbafb60.tar`，65 GB） |
| trace 用的配置 | `/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/config/arm-C-2p1d-ptpcfix-trace.env`（= `arm-C-2p1d-ptpcfix.env` + `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0`） |
| 配置链 | `arm-C-2p1d-ptpcfix.env` → `arm-C-prod-ptpcfix.env`（PTPC + #45800 回移）→ `arm-C-prod.env`（E11 `SGLANG_USE_ROCM700A=0` + 修法 6 个文件）→ `recipe-e10.env` |
| decode 侧关键设置 | KF overlay `PYTHONPATH=/yihou_overlay`、`YIHOU_KF_SPARSE=1 YIHOU_KF_INDEXER=1 YIHOU_KF_MOE=1`、调优 GEMM CSV、custom all-reduce 开、MTP EAGLE 3 / 1 / 4、mem 0.95、cuda graph 最大 bs 256 |
| KF overlay 源码 | `/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/overlay/yihou_kf/`（`kf_moe.py` 1,227 行；`__init__.py` 里是挂钩与形状门控） |
| kit | `/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/`（`engine/worker.sh`、`up.sh`） |
| 负载 | AgentX C208（每个 prefill 104 lane），模拟接受长度 2.99；trace 时 AgentX 跑 1,500 s，预热结束 600 s 后采两次，每次 40 步，所有 DP rank |

模型形状（`GLM-5.2-MXFP4/config.json`）：hidden 6144，`moe_intermediate_size` 2048 → **每 rank 中间维 TP4 是 512、TP8 是 256**；
256 个路由专家加 1 个 shared，融合后 `ne257`，top-8 + shared = 9；共 78 层，前 3 层 dense，75 层 MoE；a4w4（MXFP4，`per_1x32`）。
C208 下 decode 在跑约 86 个请求，verify 每个请求 4 个 token，MoE 的 M 约 344（8 个 rank all-gather 之后）。

## 3. 数据位置

| 内容 | 路径 |
|---|---|
| **TP8 trace**（2 次 × 8 rank，每个约 10 MB） | `/perf_apps/liyingli/vfix-logs/opt/trace/opt-trace2p1d-c208/traces/decode-083302/`、`…/decode-083802/`（`decode-*-TP-{0..7}-DP-{0..7}.trace.json.gz`） |
| TP4 对照 trace（4+4 C104，2 次 × 4 rank） | `/perf_apps/liyingli/vfix-logs/opt/trace/opt-traceC-c104/traces/decode-231636/`、`…/decode-232136/` |
| TP8 trace 那次部署的引擎日志（含 `[yihou-kf]` 决策行） | `/perf_apps/liyingli/vfix-logs/runs-n06-25/dprof-c208-1104T0758/logs/glm52_2p1d-decode.log` |
| 2P1D C208 计分那次的引擎日志（步长拟合用） | `/perf_apps/liyingli/vfix-logs/runs-n06-25/c208-1104T0523/logs/` |
| TP4 计分的引擎日志 | `/perf_apps/liyingli/vfix-logs/runs-n06-25/c104-1103T1909/logs/`（scoreC） |
| 计分汇总 | `/perf_apps/liyingli/vfix-logs/opt/score-2p1d.txt`（54,094）、`score-Cptpcfix.txt`（4+4 53,666） |

## 4. 初步分析（`RESULT` 4.6 节）

verify graph（ms/步，取最忙 rank 约 16 个请求时；`ana/dtrace_deep.py` 统计全部 rank、每个 kernel 都归了类）：

| 分项 | TP4（C104） | TP8（C208） | 说明 |
|---|---|---|---|
| MoE gemm1 | 10.96：KF `gemm1_a4w4_port_fp4_h6144_i512_ne257_bm16_…_glm` | **6.91：aiter 原版 `mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x64x256_…`** | KF 的 gemm1 只为 i512 做了 |
| MoE gemm2 | 5.66：`gemm2_a4w4_port_hmax8192_imax512_bm16_bn128_bk256_nt_atomic_…` | 4.63：`gemm2_a4w4_port_hmax8192_imax256_bm64_bn128_bk256_nt_reduce_tk9_…` | TP8 用的是 reduce 版 |
| MoE reduction | 0 | **1.03**：`moe_reduction_bf16_bf16_t9_n6144_…` | reduce 版 gemm2 需要单独的归约 |
| MoE 排序 / 量化 | 1.11（含 KF `glm52_moe_sort_fill`） | **2.50**（aiter `opus_moe_sorting` P23 / P0_v2、`mxfp4_moe_sort_kernel`、`topk_reg`） | |
| **MoE 合计** | **17.7**，GEMM 读 96.6 GB → 5.81 TB/s（峰值 73%） | **15.1**，GEMM 读 48.3 GB → **4.18 TB/s（52%）** | 按 TP4 的效率算，TP8 约 9.4 ms → **约 5.6 ms 空间** |
| 集合通信（传输 + 等最慢的 rank） | 5.37（all-gather 1.96 + 0.87，reduce-scatter 1.97 + 0.34） | 5.54（2.03 + 0.63，2.39 + 0.35） | 基本相同 |
| sparse MLA / indexer | 6.82 / 3.36 | 8.05 / 3.85 | TP8 多 1.7 ms，各 rank 耗时几乎相同，可能是 graph 按最大 batch 补齐（未证实） |
| dense GEMM / norm | 7.89 / 1.19 | 8.12 / 1.15 | DP attention 下每 rank 一份，与 TP 无关 |
| 草稿 graph / draft_extend | 2.1 / 2.2 | 1.5 / 1.8 | |
| GPU 空闲 | 0.5–0.8 | 0.9–2.5（含采集起始那一步） | 去掉 trace 开头约 60 ms 再统计 |

- KF 的 MoE 挂钩在 `moe_runner.aiter.AiterRunnerCore.run`，**只在 M = 128–448 时走 KF**（TP4 是 112–448），其余调用走原版。决策见日志里的
  `[yihou-kf] … moe M=… -> KF/orig`。另外，即使走了 KF，TP8 的 gemm1 用的也是 aiter 原版 kernel（见 trace 里的 kernel 名）。
- 步长拟合（`ana/decode_step_fit.py`，`DP_RANKS=8`）：TP8 是 `17.8 + 0.80·最忙请求 + 0.64·最忙上下文(M)`；TP4 是 `21.3 + 0.70·… + 1.66·…`。

**建议方向**：把 KF 的 gemm1（`gemm1_a4w4_port…`）、atomic 版 gemm2（省掉单独的 reduction）、融合排序（`glm52_moe_sort_fill`）移植并调优到 **i256**，
同时把 M 的门控上限放宽到 448 以上（C208 高峰时会超过）。上限约 5.6 ms/步。另一个可查的方向是 TP8 下 sparse MLA / indexer 多出的 1.7 ms 是不是补齐造成的。

## 5. 工具（CPU 即可，登录节点就能跑）

| 脚本（`/perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/ana/`） | 用途 |
|---|---|
| `dtrace_deep.py [--moe-gb 48.3] TRACE_RANK0 … TRACE_RANKn` | 每 rank 三个 graph 的 GPU 时间、verify 按类别（列出每类耗时最多的 kernel）、通信拆成传输和等待、MoE 实际带宽 |
| `dtrace_cat.py TRACE…` | 早期的分类版（类别正则漏掉了 `mfma_moe1`，会归到 other） |
| `dgaps.py` / `dgap_stacks.py` | GPU 空闲按位置统计 / 空闲期间的阻塞调用与 Python 栈 |
| `decode_step_fit.py`（`DP_RANKS=8`） | decode 步长线性拟合 |

## 6. 约束

- **InferenceX 不允许 overlay / monkey-patching**（`sitecustomize` 运行时替换函数），KF 的 MoE 目前**没有上游 PR**（yihou 的 #47276 / #47278 / #47897 / #47872
  覆盖 sparse MLA、indexer 和 top-k，不含 MoE）。所以成果最终要以 aiter / SGLang PR 的形式交付。
- 正确性：任何 kernel 改动都先过门禁，即 `verify_suite.sh`：smoke 6/6、200 题无开头坏和乱码、16k 长上下文，并在**真实接受率**下看 decode 日志的 `accept len`。
  对照数据：2P1D 门禁 196/200、0 / 0、接受长度 3.43。
- 节点：holder 作业 35103（到 11-04 22:44）正在按队列跑 2P1D + hrrn → 4+4 DP 均衡 → PP8，**两台节点的 GPU 都在用**。要上机请先协调，
  或者把实验做成 `/apps/liyingli/holder/queue/` 的队列项（写法见 `opt/scripts/opt_item2.sh`；trace 用 `trace` 类型）。
- 2P1D 现在受 prefill 限制（两个 prefill 92% 的时间有排队），decode 变快先体现为交互度和 TTFT；要转化为吞吐，还需要 prefill 侧的改进（见 RESULT 6.2）。
