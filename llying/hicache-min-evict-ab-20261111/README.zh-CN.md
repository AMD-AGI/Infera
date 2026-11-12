# sglang#48047（`SGLANG_HICACHE_WRITE_BACK_MIN_EVICT_TOKENS`）MI355X GLM-5.2 PD 实测（2026-11-11 → 11-12）

对象：[sgl-project/sglang#48047](https://github.com/sgl-project/sglang/pull/48047)。reviewer（xiezhq-hermann）要的是
"post-#46930 main 上 MI355X GLM-5.2 PD 的 prefill 数据，device / host 命中分开报告"（非阻塞）。

## 结论

1. **在我们的配置下，这个开关没有端到端收益，也不损失命中。** RDMA 传输，prefill TP1×PP4，sglang main，三种配置
   （生产 chunk 8192 + async、chunk 4096、关 async）下，16384 与关闭的第 2 轮吞吐 / TTFT / ITL 都在噪声内；
   驱逐次数降一半多、驱逐耗时降 40–65%，device / host 命中、丢弃 token 都不变。
2. **值太大有害**：524288（单次约 53 万 token）比关闭慢约 4%（两次），TTFT 中位 +9%：单次驱逐约 0.4 s 超过一步 forward，
   PP3 出现 0.3–0.55 s 停顿；多踢掉的前缀变成 host 命中，load-back 耗时翻倍。
3. **原因（PP0 trace 实测）**：main 上 prefill 每步 GPU 计算忙 91–94%、完全空闲 < 2%；调度线程待在驱逐函数里的时间
   （关闭 65% 墙钟）里 GPU 95% 在算——驱逐的耗时主要是写栅栏在等 GPU 正在跑的 forward。开关省下的是这段 CPU 等待，
   不是 GPU 时间，所以节拍（≈ 8192 token / 0.245 s）和吞吐不变。
4. **与 10 月 +2.8%（E1）的区别**：E1 在 v0.5.22（#46930 之前，每个 victim 单独等一次；驱逐经 `_pp_sync` 跨级同步）上，
   当时每步 GPU 忙约 240 ms、空闲约 200 ms，驱逐同步在锁步环的关键路径上（E1 节拍 380 → 328 ms）。
   main 合入 #46930 后一批 victim 只等一次，GPU 已基本跑满，这个开关在我们的栈上没有可省的东西。
   reviewer 在 B200 单卡 Qwen2.5-1.5B 上测到的收益（forward 很短、驱逐固定开销占比高）与此不矛盾。
5. **TCP 下测到的 "+16%" 不成立**：前一半实验的容器打不开 RDMA 网卡，KV 走 TCP；那个差异来自 TCP 传输积压，不是驱逐（§5）。

## 1. 环境与方法

- **栈**：sglang main `daa17184` + #48047 + #46830、#46820（已合入）+ #46918、#45524（未合入，PP prefill 需要），
  源码树 `44ad9d69f`，挂载在 11-09 SGLang ROCm nightly 镜像上；镜像加 Infera 0.3.3 与 Infera 的 host libionic 入口
  （`scripts/image.Dockerfile`，标签 `e1ab-sglang:1112`）。kit 为 Infera v2 kit `a1f49449` 的副本，改动见 `scripts/kit-vs-infera-v2.diff`
  （去掉 md5 检查、挂载 main 源码、decode 安全配置、A/B 用的环境变量与诊断开关）。
- **部署**：单节点 4+4。prefill TP1×PP4（GPU 0–3），`--pp-async-batch-depth 1`，chunk 8192，HiCache `write_back`
  （kernel I/O、page_first，host 21.7M token），显存池 10.73M token；decode TP4 + DP attention（GPU 4–7），EAGLE MTP 3 步 topk 1；
  Mooncake RDMA 本机回环（`MC_DISABLE_HIP=1`）。每个点都确认 prefill 4 个 rank `Found 8 HCAs`。
- **节点**：holder 35609 的 n01-29、n02-29，amdgpu 6.16.13（6.19 驱动节点会让 RDMA 下的 prefill 段错误，§5）。
- **负载**（`python -m sglang.benchmark.serving`，generated-shared-prefix，每次部署两轮，`scripts/e1ab_item.sh` 的 `gsp`）：
  - 第 1 轮（不计分）：1024 个 11k token 系统提示词各发一次，共约 11.3M token，超过显存池，最早的被 write_back 驱逐到 host。
  - 第 2 轮（计分）：同一种子（同一批系统提示词），zipf α=0.8 抽组，1024 个请求，每个新增 8k token、输出 64，并发 64。
    命中的系统提示词分 device / host，新问题持续申请显存触发驱逐。prefill 全程排队约 51 个请求（prefill 计算瓶颈）。
  - 指标：第 2 轮前后的 prefill `/metrics` 差值（pp_rank 0），加压测端的 TTFT / ITL / TPOT。`scripts/gsp_report.py` 生成表格。
- **AgentX 没用上**：一个 fast 点约 60 分钟，而 AUS batch QoS 只有 30 分钟抢占豁免，35861 在第 30 分钟被抢占。

## 2. 结果（RDMA，第 2 轮，完整表在 `data/round2-metrics.md`）

### 2.1 生产配置（chunk 8192，async 1），各两次（n01-29 / n02-29，顺序错开）

| MIN_EVICT | 时长 s | 新计算 tok/s | device / host 命中 | 驱逐次数 | 单次驱逐 token | 驱逐耗时 s（PP0） | load-back 耗时 s | TTFT 中位 ms | TPOT ms |
|---|---|---|---|---|---|---|---|---|---|
| 0（关） | 269.5 / 265.8 | 32,546 / 32,999 | 45.0% / 12.2% | 1,288 / 1,289 | 8.7k | 190 / 182 | 0.77 / 0.78 | 2,541 / 2,478 | 9.2 / 9.3 |
| 16384 | 269.0 / 266.0 | 32,605 / 32,974 | 45.0% / 12.2% | 582 / 580 | 19.3k | 106 / 104 | 1.09 / 1.07 | 2,517 / 2,502 | 9.2 / 9.1 |
| 524288 | 282.7 / 274.6 | 31,025 / 31,942 | 44.6–44.7% / 12.4–12.5% | 22 / 22 | 528k | 8.9 / 9.1 | 1.68 / 1.65 | 2,749 / 2,716 | 9.1 / 9.2 |

KV 传输平均 1.24–1.29 s（无积压）；丢弃 token 全为 0。

### 2.2 chunk 4096（async 1）与关 async（chunk 8192）

| 配置 | MIN_EVICT | 时长 s | 新计算 tok/s | 驱逐次数 | 驱逐耗时 s（PP0） | TTFT 中位 ms |
|---|---|---|---|---|---|---|
| chunk 4096 | 0 | 300.9 / 296.3 | 29,153 / 29,601 | 2,354 / 2,352 | 155 / 148 | 1,711 / 1,713 |
| chunk 4096 | 16384 | 302.1 / 305.4 | 29,034 / 28,723 | 624 / 626 | 54 / 56 | 1,727 / 1,748 |
| async 0 | 0 | 280.9 | 31,222 | 1,288 | 76（四级最大 148） | 2,266 |
| async 0 | 16384 | 278.7 | 31,477 | 585 | 40（四级最大 90） | 2,238 |

命中率全部 44.9% / 12.2%。关 async 两档都比开 async 慢约 5%（与 llr35 记录的 async +2.5% 同量级）。

### 2.3 TTFT / ITL

- 16384 与关闭：三种配置下 TTFT 中位差 0–2%、P90 差 < 3%，方向不一致；ITL / TPOT 差 1–3 ms，方向不一致。
- 524288：TTFT 中位 +9%、均值 +4%，原因同吞吐。
- P99 TTFT 约 240 s（各档相同）：prefill 用 shortest-prefill-first，长请求在约 50 个请求的队列里一直被往后排；与开关无关。
- ITL 中位约 250 ms 而 TPOT 约 9 ms：decode `stream_interval=30`，ITL 是两次推送的间隔，看生成速度用 TPOT。
- 局限：负载是固定并发、prefill 持续排队，TTFT 主要是排队；低负载下没测。

## 3. 原因

### 3.1 节拍不随驱逐变（批次日志，`gsp_ts.py` 与批次间隔统计）

每个批次都是满的 8192 token，第 2 轮约 1071 批，批次间隔中位：关闭 0.244 s、16384 0.247 s、524288 0.240 s
（524288 的 PP3 P90 / P99 升到 0.33 / 0.55 s）。吞吐 ≈ 8192 / 节拍。关闭比 16384 每批多约 78 ms 驱逐，节拍不变
→ 这段驱逐时间不在关键路径上。

### 3.2 PP0 trace（`trace_evict.py`，生产配置，各 31 批）

| | 关闭 | 16384 |
|---|---|---|
| 每批节拍 | 248 ms | 247 ms |
| GPU 计算忙 | 94% | 91% |
| GPU 只在跑 RCCL | 0.34 s | 0.51 s |
| GPU 完全空闲 | 0.12 s（1.6%） | 0.13 s（1.8%） |
| 驱逐调用 / 驱逐内墙钟 | 36 次 / 4.81 s（窗口 65%） | 15 次 / 2.34 s（32%） |
| 驱逐期间 GPU 在算 | 95% | 89% |
| 驱逐期间 GPU 空闲 | 0.23 s | 0.25 s |

write_back 驱逐对调度线程是同步的：`_demote_backed_up` → `writing_check(write_back=True)` → `start_writing`
（D2H 流先 `wait_stream(forward_stream)`，即写栅栏）→ `ack.finish_event.synchronize()`。CPU 等的是 GPU 正在跑的 forward，
这期间 GPU 不空闲；开关减少的是这段等待的次数，不改变 GPU 的忙闲。关 async 时驱逐那一刻 forward 多半已跑完，
写栅栏几乎不等（PP0 驱逐 76 s vs 开 async 190 s），同样不占 GPU 时间。

trace 局限：带调用栈的 Python 采集让第 2 轮从约 267 s 拖到约 530 s，PP1–PP3 的批次间隔被拉长到 2.5–7 s、GPU 统计失真，只用 PP0
（节拍 248 ms 与不采集时的 244 ms 一致）。不采集时四级的批次间隔分布相同，推断其余三级同样饱和，未直接证实。

### 3.3 和 E1（v0.5.22，+2.8%）的区别

| | E1 时（v0.5.22，`prefill-opt-20261027`） | 这次（main） |
|---|---|---|
| 写回等待 | 每个 victim 各等一次（#46930 前） | 一批 victim 只等一次 |
| 每步节拍 / GPU 忙 / GPU 空闲 | 约 380 ms / 236–247 ms / 188–222 ms | 约 245 ms / 91–94% / < 2% |
| 驱逐在关键路径上 | 是：锁步环里 PP0 的驱逐同步推迟下发，下游表现为"等上游" 60–120 ms；E1 节拍 −14% | 否（§3.1、§3.2） |
| 负载 | AgentX trace C104，两节点 4+4，async 1，512K | 合成负载，单节点 4+4，async 1 |

收益条件不是"驱逐比 GPU 时间长"，而是"GPU 每步有空等，且驱逐是造成空等的串行段之一"。E1 时满足，main 上不满足。

## 4. 时间线与作业

| 时间（UTC） | 内容 |
|---|---|
| 11-11 17:07–17:38 | AgentX 单点 35861 在第 30 分钟被抢占 → 改 `bench_serving` 方案 |
| 11-11 18:31–11-12 06:00 | 配置 A / B 的 TCP 下数据（§5），节点 n10-29 / n04-29 / n06-25 / n06-33 / n04-33 等 |
| 11-12 06:39–08:20 | 传输线程计时、碎片计数诊断 |
| 11-12 08:20 | 发现 KV 走 TCP（`Found 0 HCAs`） |
| 11-12 08:34–09:14 | 加 libionic 入口后 RDMA 段错误排查（全在 n04-25，6.19 驱动） |
| 11-12 09:42–12:38 | holder 35609（n01-29 / n02-29，6.16.13）：RDMA A/B、chunk 4096、async 0、trace |

## 5. 遇到的问题与处理

1. **AgentX 跑不完**：AUS batch QoS 抢占豁免 30 分钟，一个 fast 点约 60 分钟 → 改用 `bench_serving` 两轮负载（单作业约 24 分钟），
   之后改在 holder 上用 `srun --overlap` 跑。
2. **压测容器导入错**：工作目录 `/sgl-workspace` 下的 `sglang/` 被当成命名空间包，`No module named sglang.benchmark.serving`
   → 容器加 `-w /tmp`。
3. **KV 走了 TCP（11-11 到 11-12 08:20 的数据全部受影响）**：nightly 镜像里的 libionic 与宿主机 ionic 驱动 ABI 不匹配，
   Mooncake `No RDMA devices found` / `Found 0 HCAs`；kit 又设了 `MC_DISABLE_HIP=1`（该镜像的 Mooncake 认这个变量），只剩 TCP。
   TCP 下传输线程每块 150–320 ms 的 CPU 时间、每线程 0.6–1 GB/s，与调度器抢 CPU，于是出现"16384 +16%、524288 −18%"
   与 15–20 s 的传输积压——都是 TCP 特有的，换 RDMA 后消失。处理：镜像加 Infera 的 `infera_inject_host_ionic.sh` 入口；
   `e1ab_item.sh` 在部署后检查 HCA 数，`Found 0 HCAs` 就停止。PR 描述里一度发出的 TCP 数据已撤下。
   - 与 10 月单节点 P4+D4 比两节点慢的问题不同：那次是 Mooncake 对同主机目标优先走 HIP 传输（kit README 第 12 条，`MC_DISABLE_HIP=1` 解决）。
     Mooncake #5005（HIP 拷贝流高优先级）针对的是 HIP 传输，对 RDMA 回环与这次的 TCP 退化都不适用。
4. **RDMA 下 prefill 段错误**（`c10::hip::HIPStream::stream()`，调度器事件循环第一步）：依次排除了 dma-buf 注册方式、Mooncake 版本
   （换 Infera 镜像的 Mooncake）、RDMA 用户态库（两镜像 rdma-core 均 39.0-1）、`RDMAV_FORK_SAFE`。另一会话查明 amdgpu 6.19.x 节点会崩，
   崩溃全部发生在 n04-25（6.19）；换到 6.16.13 的 holder 节点后正常。`e1ab_gsp.sh` / `e1ab_holder_chain.sh` 遇 6.19 直接退出。
   在 6.19 的 n04-29 上 TCP 能跑，6.19 是否只在 RDMA 下出问题未单独确认。
5. **两个被否定的解释**：
   - prefill 页碎片导致传输慢（`scripts/diag-kv-contiguity.diff`）：三档每传输块页数 37.4 / 38.5 / 45.1，与传输快慢无关。
   - "驱逐被 async 流水线藏住"：关 async 后两档仍相同；trace 显示真正原因是驱逐时间本身大多在等 GPU（§3.2）。
6. **其他**：
   - 作业被 root 取消（35968、36024，未重排，原因未查）。
   - Mooncake 传输线程按目标端口和对 4 取余分片，4 个 decode DP 会话常撞到同一线程（如端口 15620/16439/16655/16993 → 0/3/3/1，一个线程空闲）。
     RDMA 下传输无积压，对本实验无影响；端口每次部署随机，未见与档位相关。是否值得上游改进待定。
   - n02-29 的 `/data` 100% 满，经用户同意 `docker image prune -a` 删除未被容器使用的镜像（913.5 GB）。

## 6. 文件

- `scripts/e1ab_item.sh`（`gsp` / `upcheck` 分支）、`e1ab_gsp.sh`（单点作业）、`e1ab_holder_chain.sh`（holder 内串行）、`image.Dockerfile`
- `scripts/gsp_report.py`（第 2 轮表格）、`gsp_ts.py`（10 s 时间序列）、`trace_evict.py`（trace 分析）、
  `kvdiag.py` / `kvwdiag.py` 与两个诊断 diff（只在诊断作业里挂载，A/B 源码未改）
- `data/round2-metrics.md`：全部运行的第 2 轮指标（`gspbr` RDMA 生产、`gspc4k`、`gspd0`、`gspb-*` 为 TCP 下的配置 B，仅供参考）
- 原始日志：`/perf_apps/liyingli/bench_agentx/e1ab/logs/`（`gsp/<run>/` 指标与 trace，`runs/<run>/logs/` 引擎日志，`e1ab.out` 总日志）
