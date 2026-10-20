# GLM-5.2-MXFP4 2P(TP4)1D(TP8)，sglang v0.5.22，138 + 140（10-20 起）

目标：在 v0.5.22 镜像上做 2P1D 的 example（`examples/sglang_2p1d_glm5.2`，分支 `feat/sglang-mix-glm5.2-tp4-v0522`），
端到端验证正确性，用融合配置测性能，再测 C176 以外的并发点，为 InferenceX 上 infera + sglang 的 Pareto 线提供数据。
每卡吞吐按 16 张 GPU 计。

## 部署

| 实例 | 节点 | GPU |
|---|---|---|
| prefill 0（TP4 / DP4 attention） | crsuse2-m2m-138（10.245.157.237） | 0–3 |
| prefill 1（TP4 / DP4 attention，PID 偏移 30000） | 138 | 4–7 |
| decode（TP8 / DP8 attention，EAGLE MTP） | crsuse2-m2m-140（10.245.159.30） | 0–7 |
| etcd / router / AgentX、GSM8K 客户端 | 138 | — |

镜像 `infera-sglang:v0522-fdbafb60`（合并 main 后的分支 `fdbafb60` 按 `Dockerfile.sglang` 构建，ID `sha256:509b8427…`，
138 构建后 `docker save | load` 到 140，两边 ID 相同）。运行配置 138 `/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/site.env`，
运行目录 `.../2p1d/work/runs/`。

## 融合配置（example 默认值）

以 AUS 2P1D 最高吞吐 a1-spf（`../2p1d-aus-v0521-20261019/`，v0.5.21，C176 = 35,377）为底，叠加今天 mixed 的 A/B 结论
（`../mix-v0522-correctness-20261019/`）：

| 项 | 取值 | 来源 |
|---|---|---|
| prefill | mem 0.88，chunk 32768，max running 256，HiCache 250 GB/rank `write_back`（ratio 1.5、kernel、page_first），`--schedule-policy shortest-prefill-first`，`HSA_NO_SCRATCH_RECLAIM=1` | a1-spf |
| decode | mem 0.92，decode graph 到 256，max running 256，`--tokenizer-worker-num 4`，EAGLE 5 / 1 / 6 | a1-spf |
| 两侧 | `--disable-custom-all-reduce`（同时关掉 QuickReduce，覆盖 a1-spf 的 prefill `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`） | mixed：custom AR 导致乱码 |
| 两侧 | `SGLANG_OPT_USE_TOPK_V2=true` | mixed：+8.6% 吞吐、正确性不变 |
| 两侧 | `index_share_for_mtp_iteration=false` | mixed：默认值无收益 |
| router | kv-aware，prefill / decode 重叠权重 20 / 2，`--session-affinity prefill`，`--pd-prefill-guard-release completion`（R1），`--decode-input-demand on`（R2） | 3P1D / a1-spf（R1、R2 现为正式参数） |
| Mooncake | 8 条 ionic，GID 1，`MOONCAKE_DISABLE_HIP_DMABUF=0`，dest-local-rail + `MC_IB_TIMEOUT=18`，dest device affinity | crsuse2 3P1D（AUS 的 DMABUF=1、NodePort 等不适用） |
| 分配策略 | early（默认） | a1-spf（prefill_complete 在 AUS 上更差） |

## 计划

1. main 合入分支（无文本冲突），lint / unit / 镜像内 unit / engine 检查。
2. 写 example（无自动门禁，手动起停），GSM8K 与 InferenceX 一致（`run_eval`），AgentX 沿用 InferenceX `918524ff`。
3. 正确性：真实 MTP 接受率起栈，smoke + GSM8K 全量 1,319 题、并发 64。
4. 性能：C176（模拟接受率 3.61），对照 AUS 35,377。
5. 并发点 C128 → C96 → C64 → C32，每点按 InferenceX 方式整栈重启（down → 等显存释放 → up → 预热 10/lane → 3,600 s）。
   每点开测前检查两台节点有无他人 GPU 进程 / 显存 / CPU 负载。

## 并发点的选择（任务 4）

InferenceX 的 Pareto 图横轴是每用户交互速度（tok/s/user），纵轴是每卡吞吐；一套配置在不同并发下给出一条曲线，最终的线是
各配置曲线的上包络。2P1D（16 卡）在这条线上的位置：

- **C176 以上没有意义**：AUS 上 C192 比 C176 低 4.7%（decode KV 被等传输的请求占满，`../2p1d-aus-v0521-20261019/` §2.10），
  吞吐已过峰值，交互也不会更好。
- **C176 是吞吐峰值**，但它的优势不只在吞吐：AUS a1-spf C176 的 interactivity p50 / p90 = 50.0 / 46.0，而今天 mixed（TP4 单机
  4 卡）C48 吞吐 34,662 时 p90 只有 22.7。2P1D 的 decode 独占 8 卡、没有 prefill 抢占，尾部交互远好于 mixed，所以它在"同等吞吐、
  更高交互"的区间最可能进入上包络；往低并发走，交互继续提高、每卡吞吐下降，正是 Pareto 线需要的那一段。
- **下限 C32**：decode 8 个 DP rank，C32 时每 rank 约 4 个会话，已接近 decode 每步耗时的平台；再往下（C16 及以下）16 卡的每卡
  吞吐会很低，会被单机 TP4 / TP8 配置（InferenceX MI355X 单机 conc 1–16 的点）压在下面，不值得占两台机器。
- **中间取 C128、C96、C64**：与 C176、C32 大致等比（每档约 ×0.7），曲线上间距均匀；InferenceX GB300 2P1D 高吞吐配置也测
  128 / 192，低延迟的 1PnD 测 45–48，覆盖范围相当。
- 顺序从高到低：先拿到峰值附近的点，若中途机器被占用或出问题，已测的点更有价值。

## 进度

**15:20 main 合入**（`fdbafb60`，已推送）：无文本冲突；PR #187 在 main 上的最终内容与分支已有的完全一致（合并后 `deploy/` 无差异）。
合并后检查：unit（python:3.10）2,789 passed / 149 skipped；镜像内 unit 2,579 passed / 91 skipped；engine（sglang，8 个文件）全部通过。
lint 只有 `ruff format` 失败一处：`infera/projection/core/projection/module_profilers/moe_mlp.py`，与 origin/main 完全相同，
来自 main 的 PR #205，不是本分支改动，未在本分支修改（会与 main 产生分歧）。140 上没有 `python:3.10` 且 Docker Hub 凭据过期，lint 改在 138 跑。

**15:40 第一次起栈（正确性，真实 MTP 接受率）**：prefill 1 启动失败——`port_base at 27436 is not available`。DP attention 下
sglang 从 `--port + 233` 起占 11 个端口（DP4），两个 prefill 端口只差 1（27201 / 27202）时两段重叠。改为 prefill i 用
`PREFILL_PORT + 256 i`（与 AUS harness 的 stride 256 一致），只删掉并重启 prefill 1（prefill 0 与 decode 在冷启动中，不重来），
其余照 `up.sh` 继续。KV 事件内部端口由 infera 随机挑选，两个 prefill 不冲突（17471 / 17123）。

**15:50 起栈失败：decode 2 个 rank 的 KV 缓冲区注册失败 → `MC_MAX_MR_SIZE=1073741824`**。两个 prefill 健康后，infera 启动时的
PD 探测（每个 DP rank 发一个请求走一遍 KV 传输）在 decode DP3 上连续 3 次失败（prefill 侧 `Failed to send kv chunk`、decode 侧
`Failed to get kvcache from prefill instance`），两个 prefill 都退出。decode 日志：DP3、DP5 各有一个缓冲区
`Failed to register memory 0x77ffc0000000 / 0x7e58c0000000: Invalid argument [22]`（chunk 0，context 0）。decode 每 rank 79 个
KV 缓冲区，每个 2,320,846,848 字节，超过 ionic 的 `max_mr_size`（2 GiB），Mooncake（#2644 自动分块）切成 2 GiB + 余量两段，
用 dma-buf 注册。632 个缓冲区里**恰好只有这 2 个起始地址按 1 GiB 对齐，正是失败的 2 个**。8 条 ionic 链路 IPv6 互 ping 全通。

独立复现（`scripts/mr_align_test*.py`，镜像内用 Mooncake Python 接口，138 GPU 0，8 条 ionic）：分配 131 个同尺寸张量，其中起始
1 GiB 对齐的那个整块注册失败（同样的 EINVAL），4 个不对齐的对照都成功；在大分配**中间**的 1 GiB 对齐地址注册 2 GiB 成功
（失败条件是分配本身从 1 GiB 对齐处开始、首段 2 GiB）；同一对齐分配注册前 1 GiB / 512 MiB / 2 MiB、或跳过前 2 MiB 都成功。
`MC_MAX_MR_SIZE=1073741824`（Mooncake 读的环境变量，分块上限 1 GiB）后对齐分配切成 3 段注册成功。ionic `max_mr` 16384，
decode 每网卡约 1,900 个 MR，够用。3P1D 的 REPRODUCE 里"只重启 decode 后部分传输通道失效、某个 decode rank 收不到 KV、
重启后好了"应是同一个原因（地址对齐是随机的）。example 的 `worker.sh` 已加，README note 8。值得报给 Mooncake / ionic。

**16:02 第二次起栈（正确性）**，kit v3（端口 stride 256 + `MC_MAX_MR_SIZE`）。约 4 分钟三个引擎健康，decode 无注册失败，
prefill 启动探测"verified real KV transfer for 8 DP rank pair(s)"，router 列出 2 prefill + 1 decode。

**正确性结果（真实 MTP 接受率，产物 `results/correct-v2/`）**：

| 项 | 结果 |
|---|---|
| smoke（`smoke.sh`） | **6/6**：PD 注册 `decode prefill prefill`、completion、chat、stream、66k token 长前缀冷 40.0 s / 复用 1.9 s |
| GSM8K（`eval.sh` = InferenceX `run_eval`，全量 1,319 题，并发 64，`max_tokens` 16384） | **strict-match 0.9697 ± 0.0047，flexible-extract 0.9704**，用时约 4 分钟 |
| 乱码扫描（lm-eval 样本） | 0 条模板标签泄漏、0 条空回答；16 条回答超过 1 万字符（15 条答错），均为连贯的反复推敲直到 16384 token 上限（人体金字塔、勺子、邀请人数等有歧义的题），不是乱码 |
| decode MTP 平均接受长度 | 4.18（失控长回答在重复文本上接受 ~5.97，拉高均值） |

发现：up.sh 原先在 router 健康 10 s 后就打印 worker 列表，此时 prefill 还没通过启动探测、尚未注册（只列出 decode）。改为等到
router 列出 2 prefill + 1 decode（prefill 容器中途退出则报错），kit v4。

**16:22 性能串行开始**：`scripts/pd_perf_chain.sh`（kit v4，138 后台），C176 → C128 → C96 → C64 → C32，每点：等两节点显存释放 →
记录干扰检查 `perf/c<C>.env-check` → `SIM_ACCEPT_LEN=3.61 up.sh` → `agentx.sh`（3,600 s，预热 10/lane）→ `down.sh`。

## 性能结果（每卡按 16 张计，产物 `results/c<C>/`）

| 点 | tok/s/GPU | 输出 tok/s/GPU | 每秒请求 | 完成 / 无效 | intvty p50 / p90 | ITL p50 / p90 (ms) | TTFT p50 / p90 / 均值 (s) | E2E p90 (s) | API 命中（理论） |
|---|---|---|---|---|---|---|---|---|---|
| **C176** | **35,637** | 265.7 | 4.375 | 15,881 / 11 | 40.4 / 35.7 | 24.8 / 28.0 | 4.28 / 8.08 / 5.99 | 62.2 | 95.85%（96.83%） |
| 参照：AUS a1-spf C176（v0.5.21） | 35,377 | 265.3 | — | 15,657 / — | 50.0 / 46.0 | 20.0 / 21.7 | 7.3 / 15.0 / 11.0 | — | — |

**C176（16:45 起栈，预热 25 分钟，计分 17:16–18:16）**：干扰检查无他人 GPU 进程，他人容器 CPU ≈ 0（138 负载 5.6 来自刚结束的显存
回收 kworker）。**吞吐与 AUS 最高持平（+0.7%）**，延迟分布不同：TTFT p90 15.0 → 8.1 s（−46%），ITL p50 20.0 → 24.8 ms（+24%），
interactivity p50 50.0 → 40.4。decode 每 token 变慢的主要嫌疑是 `--disable-custom-all-reduce`（decode 集合通信走 RCCL；mixed
上同一改动使 interactivity p90 −15%），这是正确性必需的代价；TTFT 改善来自 crsuse2 的 decode 不慢于 AUS（AUS decode 每步比
crsuse2 慢约 20%，见 AUS §2.2）以及 decode 释放更快。11 条 `InvalidInferenceResultError`（0.07%）。
每点下线后 138 显存回收约 20 分钟（`kfd_process_wq` / `ttm` kworker，D 状态），140 一两分钟。

**C176 decode 每步耗时**（decode 日志每 40 步一行，同一 DP rank 相邻两行间隔 / 40，按该 rank 在跑请求数分组，计分窗口内；
平均接受长度 3.61）：

| 本 rank 在跑 | 2 | 4 | 6 | **7** | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 每步中位数 (ms) | 38.4 | 47.1 | 53.0 | **79.1** | 80.4 | 81.6 | 82.9 | 86.6 | 89.5 | 92.0 | 93.2 |

6 → 7 个在跑请求时每步耗时跳 +49%，之后进入平台。AUS a1-spf（按最忙 rank 分组，custom all-reduce 开）为 @8 38.7、@10 53.4、
@12 66.4、@14 69.1 ms；同等负载下本次每步约慢 25–30%，与 ITL +24% 对得上。跳变形态像 RCCL 在消息变大时换协议：decode TP8 每步
对 8 个 rank 做 gather / reduce-scatter，`--disable-custom-all-reduce` 后全走 RCCL，代价比 mixed（TP4）大。可能的收回办法是
之前搁置的窄修复（只关 custom reduce-scatter + all-gather、保留 custom all-reduce，需先验正确性），本轮按用户决定不做。

**干扰检查（15:36）**：两节点 GPU 显存为 0；138 上他人容器 `dev_xb`、`yihou-expert8-tools-138` CPU ≈ 0；138 有一个
12 天前启动的 root python 进程（pid 2906078）持有 GPU 句柄、显存 0，之前所有性能测试期间都在，不动它。140 只有系统 `gpuagent`。
