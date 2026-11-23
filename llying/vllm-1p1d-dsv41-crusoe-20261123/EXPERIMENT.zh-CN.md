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
| 17:37 | c704 预热 30 分钟后失败：0 个正式请求，AIPerf 报 648 个 `ServerDisconnected` | **压测客户端卡住，不是推理服务问题**：v3-180 上有 10 个不属于任何 spur 作业的容器（其他用户的 kernel agent，运行 2–3 小时，起始负载约 70）。AIPerf 启动时 5 个 worker 的回应被当成"不认识的服务"丢弃（c640 为 0 个），依赖这些请求完成的 704 条回放 lane 全部停在 `pending join`，30 分钟内没有再发出任何请求；到 1800 s 预热宽限期 AIPerf 拆掉预热阶段并判失败。引擎与 router 全程无报错 | 节点体检（`scripts/node_check.sh`）拒绝"没有其他作业却有外来容器"或启动前负载已高的节点，自动加入排除名单并重交（最多 4 次）；v3-180 已排除 |

## 5. 提升吞吐的各轮实验

方法（与 review 时讨论的一致）：

- 在已饱和的并发上做 A/B，用分项指标归因，总吞吐作最终确认：λ（P 每秒完成请求数）、P 每 rank 现算 tok/s 与命中构成、
  D 每步耗时与每步条数。有收益的改动再重新扫并发，比较峰值对峰值。
- 每轮都在作业里记录逐步数据（`instr/vpd_steplog.py`，每个 engine 的每一步：耗时、token、请求数、graph 模式），并在计分窗口
  3300 s 处对 P（5 s）、D（3 s）各抓一次 torch profile；分析在节点上完成（`scripts/steplog_summary.py`、`trace_cat.py`）。
- 改了数值路径的轮次（graph 模式、EP / MegaMoE、top-k 后端）过完整的精度门禁。

| 轮次 | arm | 改动 | 状态 |
|---|---|---|---|
| 基线 | `arms/base.sh` | AUS 配方原样，c704 | 运行中（作业 306918） |
| R1 | `arms/r1-capacity.sh` | P/D GMU 0.95；P 去掉 FULL decode graph；D FULL_DECODE_ONLY 覆盖到 288 条；D prefill-schedule-interval 2；c768 | 排队（作业 306919） |

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
