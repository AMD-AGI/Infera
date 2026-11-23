# DeepSeek-V4.1-Flash 1P1D：crusoe 复现与提升吞吐实验记录

日期：2026-11-23 起（UTC）　负责人：liyingli

AUS 上的基线与并发扫描见 `../vllm-1p1d-dsv41-20261120/`；与 B300 Dynamo-vLLM 828k 配置的对比与优化方向见
`../vllm-1p1d-dsv41-20261120/vs-b300-dynamo-828k/ANALYSIS.zh-CN.md`。本目录记录两件事：

1. 在 crusoe 上复现 AUS 的 1P1D 结果（同一镜像、同一配方）；
2. 在 P2D2（4 卡）不变的前提下推高 AgentX 吞吐的各轮实验。

## 1. 环境

| 项 | 值 |
|---|---|
| 集群 | crusoe spur，分区 `amd-spur-v2`，账户 / QoS `amd-hyperloom` / `amd-hyperloom-qos`（每人同时 1 个作业） |
| 节点 | `crsuse2-m2m-v3-*`：MI355X × 8 的虚拟机，amdgpu-dkms 6.16.13，8 × Pensando ionic 400G，2.42 TiB 内存，本地 NVMe |
| 申请方式 | 半个节点：`--gres gpu:4 -c 118 --mem 1200G`（不独占）；P 用分到的两张低号卡，D 用两张高号卡 |
| 镜像 | `infera-vllm-dsv41:054985cc1e03-nightly-9c1d45c2`：kit `054985cc1` 在节点上构建，与 AUS 用的 `c33de6336` 内容相同；tar 在 `/shared_nfs/liyingli/images/` |
| kit | 分支 `llying/example/vllm_1p1d_dsv41_flash`，`cluster/cluster.crusoe.sh`（第 2 节） |
| 模型 | `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4.1-Flash`（revision `314113f8`），每次作业先拷到本地 NVMe（510 GB，约 190 s） |
| 压测 | InferenceX `bd334b4f`（AUS c768 跑分前最后一个 main 提交），agentx-harness 1.0.14，`cc-traces-weka-062126`，正式 3600 s、每 lane 10 个预热请求，decode golden 3.51 |

## 2. 与 AUS 的差异（都在 `cluster/cluster.crusoe.sh` 与 kit 提交 `3731453b8` 里）

| 项 | AUS | crusoe | 原因 |
|---|---|---|---|
| KV 显存 | GEM | GEM | 6.16.13 对 dma-buf 注册的显存计两次，且本集群没有 peer memory |
| KV 走的网卡 | 每张 GPU 用自己 PCIe 交换机上的网卡 | 两条 leg 的所有 GPU 都用 D 第一张卡对应的网卡（如 `ionic_2`），本机 RDMA 回环 | 每张 ionic 是一条独立路由的 rail，`ionic_i` 只能到 `ionic_i`；任意 P rank 都可能发给任意 D rank，而镜像里的 Mooncake（PyPI 版）不支持按目的端选本地网卡 |
| 宿主机 libionic | 不挂 | 不挂 | 宿主机库要 GLIBC 2.38，镜像是 2.35；镜像自带的驱动可用（容器内 9 个 RDMA 口都在） |
| UMBP 服务端可见的 GPU | 全部 8 张 | 只有 P 的两张（`UMBP_GPUS`） | P rank 按自己的 HIP 下标报告 KV 所在设备；P 不是 0 号起的卡时，服务端会映射到错误的物理 GPU，并在别人的卡上建上下文 |
| 端口 | 默认 | etcd 22379、router 28700、P 31300、D 31301 | 半节点会和其他作业共机，30000 与 sglang 默认端口相同 |
| GSM8K | 运行时从 HF 下载 | 本地 parquet 转成 JSONL（同一份 test 集） | 不依赖外网 |

## 3. 复现结果

| | crusoe c640 | AUS c640（重跑） | AUS c768 |
|---|---|---|---|
| run ID | `pd1p1d-c640-crusoe-1123T1307` | `pd1p1d-c640-1123T0714` | `pd1p1d-c768-hyper-1121T1745` |
| 节点 / 作业 | v3-105 / 306372（4 卡，节点上无其他作业） | n02-25（整机） | n02-25（整机） |
| 正确性 | GSM8K 0.980，needle 16k/64k/128k 32/32，kv-aware 8/8 | GSM8K 0.980，32/32，8/8 | GSM8K 0.975，32/32，8/8 |
| 总吞吐 tok/s/GPU | **463,531** | 471,698 | 459,183 |
| 输出吞吐 tok/s/GPU | 3,488 | 3,549 | 3,475 |
| 正式计分请求 / 错误 | 53,730 / 0 | 54,356 / 0 | 53,690 / 0 |
| TTFT p50 / p90（s） | 4.71 / 13.28 | 5.20 / 15.88 | 16.96 / 31.75 |
| ITL p50（ms） / 交互性 p50 | 30.7 / 32.6 | 29.1 / 34.4 | 30.5 / 32.8 |
| P 每 rank 排队 / 运行 | 35 / 3.6 | 43 / 3.8 | 116 / 3.9 |
| P GPU 命中 / UMBP / 重算 | 73.3% / 22.9% / 3.9% | 73.2% / 22.9% / 3.9% | 59.1% / 36.7% / 4.2% |
| P 每 rank 现算 tok/s | 35,975 | — | 38,737 |
| D 每 rank 运行（均值） / KV 使用率 | 211 / 0.81 | 202 / 0.78 | 210 / 0.81 |
| D 每步（迭代计数口径） | 每步 ~188 条 / 93 ms，与 AUS 拟合式 23.3 + 0.371 × 条数 相差 ±0.4% | — | ~189 条 / 93.5 ms |
| Mooncake 传输 均值 / p90 | 25 / 58 ms（单网卡回环） | — | 16 / 40 ms |

结论：crusoe c640 与 AUS c640 相差 −1.7%，在单轮波动内；P 侧各项与 AUS c640 几乎相同，D 侧与 AUS 拟合式吻合。
复现成立，本集群可以用作后续优化实验的平台。

## 4. 过程中的问题

| 时间（UTC） | 问题 | 原因 | 处理 |
|---|---|---|---|
| 10:00 | 作业 306107 准备阶段失败 | HF 缓存里的 GSM8K parquet 是指向 `blobs/` 的符号链接，只挂快照目录时容器内找不到 | 用 `realpath` 挂实际文件 |
| 13:01 | v3-157 上 docker 不可用（`docker=000`） | 该节点上本账号没有 `docker` / `video` / `render` 组 | 节点体检显式检查 `docker` 组；v3-157 加入排除名单 |
| 16:55 | c704（作业 306623，v3-180）直接以 golden 起栈时 decode 两个 rank 的 KV 为 30.3M / 31.4M token | 只在"直接以 golden 起栈"时出现，此前所有"先 block 再重启到 golden"的 run（这里与 AUS）都是 29,919,641 | 基线与各轮统一用 `GATE=1`（先 block 起栈过门禁再重启到 golden），D 的 KV 大小一致 |
| 17:37 | c704（作业 306623，v3-180）预热 30 分钟后失败：0 个正式请求，AIPerf 报 648 个 `ServerDisconnected` | **预热第一波没在宽限期内跑完，不是推理服务故障**。AIPerf 的预热先给每条 lane 发一个冷的长 prompt（平均约 12.8 万 token，长尾到几十万），要等这一波几乎全部返回才继续回放：c640 时 685 个第一波请求用了约 26 分钟（`sent` 一直停在 685），c704 时 755 个到 1800 s（`AGENTIC_WARMUP_GRACE_PERIOD`）还有 18 个没返回，AIPerf 拆掉预热并判失败。v3-180 上同时跑着 10 个不属于任何 spur 作业的容器（其他用户的 kernel agent，起始负载约 70），prefill 被拖慢 | ① 节点体检（`scripts/node_check.sh`）拒绝"没有其他作业却有外来容器"或启动前负载已高的节点，自动排除并重交（最多 4 次），v3-180 已排除；② c768 起预热宽限期用 3600 s（`WARMUP_GRACE`，InferenceX 自己的配方用 1800 或 3600，计分窗口不变） |
| 19:30 | R1 第一次（作业 306919，v3-032）起服务正常（KV 见 5.2 节），smoke 第一个请求时 decode 两个 rank 同时卡在 `execute_dummy_batch`，300 s 后 fatal | **decode DP2 死锁第 3 次**，这次两个 rank 都在空批次：请求 19:30:31 到达，prefill 19:30:33 算完；decode 在等 KV 期间两个 rank 以约 150 次/s 陪跑空批次（截至最后一次落盘 19:30:48.5，两边都是 2,290 次 FULL 6/1，完全对齐），约 19:30:52 起都不再返回；P→D 的 RDMA 握手 19:30:58 才完成（基线在请求后 1 s 内完成）。数据：`raw/pd1p1d-c768-r1-capacity-1123T1904/` | 三次死锁的共同点：decode 有请求正在等 P 的 KV、两个 rank 在频繁空转。同一 arm 重跑（R1b），流水线改为：起服务后立即启动 `hang_watch`（不再等 golden smoke），smoke 期间 decode 死掉则保留现场、重新部署一次；逐步记录改为每 1 s 落盘 |
| 20:32 | R1b（作业 307146，v3-021，同机另有一个 4 卡作业）起服务 50 分钟未完成，作业被外部取消（`CANCELLED 0:9`，`Reason=None`） | **加载权重极慢**：P、D 都在同一段 shard（约第 14–20 个）停住，D 停 10 分钟、P 停 24 分钟，之后每个 shard 约 1 s；P 又在第 47 个（101.5 GB 的 Engram 表文件）停住。停住期间节点 NVMe 读为 0，P worker 在 `copy_` 里、CPU 约 200%。加载耗时与时段相关：13:00–17:00 两次 P 加载 55–91 s，17:47 以后三次 446 s、581 s、>2,000 s，与 arm、流水线版本、是否有同机作业都对不上，像是节点主机侧（页缓存 / 锁页内存）的间歇问题。取消发生在我用 `srun --overlap` 进作业查看进程后 30–60 s；spur 会因作业步异常结束整个作业，此后不再 `srun` 进运行中的实验作业 | 改用 `pipeline4.sh`：在 `pipeline3.sh` 基础上，AgentX 期间 decode 死锁时保留现场（`ax-try1/`）、以 golden 重新部署并重跑一次 AgentX（剩余时间够 3.5 h 才重跑） |
| 19:00 | c704 基线重跑（作业 306918，v3-086，同机另有一个 4 卡作业）预热中 decode 引擎 fatal：`engine core reported fatal failure` → `LLM became permanently unhealthy` | **decode DP2 死锁复现**（AUS 4 次中 1 次，见 `../vllm-1p1d-dsv41-20261120/DECODE_HANG.zh-CN.md`）。发生在预热第一波刚结束、第二波大量请求开始进入 decode 时；逐步记录显示卡住前一步：rank1 收进 4 个 KV 刚到的新请求（24 token，FULL graph 24/4），rank0 用同尺寸的空批次陪跑；随后两边都跑 1 个请求的空批次，下一步（rank0 一个 332,907 token 上下文的新请求，rank1 空批次）两个 worker 都没有返回，300 s 后超时 | 原因未定位。kit 新增只采集的 `engine/hang_watch.sh`（crusoe wrapper 在每个 AgentX 点自动启动），下次出现时抓两个 decode worker 的 py-spy 栈，确认各自停在哪个集合通信里 |

## 5. 提升吞吐的各轮实验

方法（与 review 时讨论的一致）：

- 在已饱和的并发上做 A/B，用分项指标归因，总吞吐作最终确认：λ（P 每秒完成请求数）、P 每 rank 现算 tok/s 与命中构成、
  D 每步耗时与每步条数。有收益的改动再重新扫并发，比较峰值对峰值。
- 每轮都在作业里记录逐步数据（`instr/vpd_steplog.py`，每个 engine 的每一步：耗时、token、请求数、graph 模式），并在计分窗口
  3300 s 处对 P（5 s）、D（3 s）各抓一次 torch profile；分析在节点上完成（`scripts/steplog_summary.py`、`trace_cat.py`）。
- 改了数值路径的轮次（graph 模式、EP / MegaMoE、top-k 后端）过完整的精度门禁。

### 5.1 prefill 每步耗时（基线，冷 prefill）

c704 基线预热第一波全是冷 prompt，P 的逐步记录给出纯 prefill 的单步成本（`raw/pd1p1d-c704-base-1123T1746/steplog-p/`）：

| 每 rank | 满步（16,384 token） | 小步（约 110 token） |
|---|---|---|
| 步数 | 5,783 / 5,807 | 92 / 103 |
| 单步耗时 p50 | 381 ms | 17–24 ms |
| 新算吞吐 | 42.0k / 41.6k tok/s | — |
| 上下文（ctx_max p50） | 11.2 万 / 10.7 万 token | — |

即每步约 20 ms 固定成本（与每步把本 rank 约 150 GiB 权重读一遍同量级）加每 token 约 22 µs。按 B300 对比报告 3.2 节的估算，
每 token 的 MoE AllGather + ReduceScatter 在两卡之间那一条 xGMI 上约 11–14 µs，若不重叠，约占一半；R1 的 profile 用来核实。

### 5.2 各轮

| 轮次 | arm | 改动 | 状态 |
|---|---|---|---|
| 基线 | `arms/base.sh` | AUS 配方原样，c704 | 两次未完成（第 4 节：v3-180 预热超时、v3-086 decode 死锁） |
| R1 | `arms/r1-capacity.sh` | P/D GMU 0.95；P 去掉 FULL decode graph；D FULL_DECODE_ONLY 覆盖到 288 条；D prefill-schedule-interval 2；c768 | 运行中（作业 306919，v3-032）。KV 容量：P 29.9M → 38.1M token/rank（+27%），D 29.9M → 40.7M（+36%）；graph 显存 P 12.7 → 9.1–10.8 GiB、D 12.7 → 4.6–4.9 GiB |
| R2 | `arms/r2-megamoe.sh`（起不来则 `arms/r2-lazy.sh`） | R1 + 两条 leg EP2 + `aiter_mega_moe`（MoRI） | 排队（作业 307071） |

## 6. 复现

```bash
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123      # 工作目录：scripts/、arms/、instr/、logs/
# 一次 run = 一个作业：节点体检 -> 拷模型 / 取镜像 -> pipeline3.sh（起栈、门禁、AgentX、分析、打包）
sbatch -J lyl-dsv41-base704 -x "$(paste -sd, $W/exclude_nodes.txt)" \
    --export=ALL,ARM=arms/base.sh,CONC=704,GATE=1,PROFILE_AT=3300 $W/run_opt.sbatch
bash $W/scripts/monitor.sh        # 登录节点：新步骤、作业结束、长时间无进展
```

本目录 `scripts/`、`arms/`、`instr/` 是工作目录的副本；c640 用的是 `run_once.sbatch` + `pipeline.sh`（之后统一为
`run_opt.sbatch` + `pipeline3.sh`）。`raw/<run>/` 由 `scripts/pack_raw.py` 生成，格式与 AUS 目录相同。完整原始数据在
`$W/logs/<run>/`。
