# DeepSeek-V4.1-Flash 1P1D · crusoe 复现与 P2D2 提升吞吐实验

## 结论（截至 2026-11-24 05:50 UTC）

- **当前最好**：R1 + P 的 MoE 用 FP4 激活（`arms/r4-a4w4p.sh`），c640 **486,737 tok/s/GPU**，同集群基线 c640 463,531（+5.0%），
  AUS 最好的 c640 471,698（+3.2%）；TTFT p50 4.71 → 3.40 s。正确性门禁与基线相同（gsm8k 0.965，needle 全对）。
  同配置 c768 为 477,212（R1 同并发 439,432，+8.6%；AUS c768 459,183，+3.9%），低于 c640，峰值仍在 c640 附近。
- **复现**：crusoe c640 463,531 tok/s/GPU，与 AUS c640 471,698 一致（−1.7%）。
- **瓶颈**：P2D2 在 c640–c768 时由 P 的现算速度决定。P 两个 rank 一直跑 16k token 满步（389 ms/步，每 rank 40.9k tok/s），
  GPU 时间里 32–38% 是 MoE 的 AllGather + ReduceScatter，走两卡之间唯一一条 xGMI，已到链路带宽（约 54 GB/s/方向）。
  前缀命中（95.8%）已贴近 trace 理论上限（96.0%），加缓存不减少计算；D 有余量。
- **试过的现成配置**：加 KV 容量 + D 全 FULL graph（R1，c768 439k，与 AUS c768 459k 持平偏低，decode 变快但吞吐不变）；
  EP2 + MegaMoE（R2，P 满步慢 35%）；RCCL 通道 / 协议（无变化）；DBO（vLLM 对 Engram 模型禁用）；
  **P 的 MoE 改 a4w4（R4/R5，P 满步 −10%，有效）**。
- **再往上需要改代码**：MoE 通信与计算重叠、fp8 dispatch、P 改 TP2 + QuickReduce、把短 prefill 分给 D（`EXPERIMENT.zh-CN.md` 5.4 节）。
- **稳定性（11-24 19:30 更新，EXPERIMENT 4.4–4.6 节）**：所谓"DP2 死锁"和整机 GPU 停顿已经定位，根因是 **KFD 撤队列**。
  内存规整迁移了注册给 GPU 的主机内存（userptr）里的页，KFD 就撤下每个注册了这段内存的进程的全部 GPU 队列；恢复前要重新
  校验整段内存，Engram 表每条 leg 有 189 GiB，所以要 45–200 s，超过 300 s 就被 vLLM 判为死锁。现成开关
  `VLLM_ROCM_ENGRAM_HIP_VMM=1`（`arms/r4-hipvmm.sh`）把 Engram 表放到驱动分配的内存里，复现中停顿从每个 cycle 几十秒降到 0，
  起栈也从 25–36 分钟降到 10 分钟。UMBP 主机池（418 GiB）仍会被撤队列，期间 UMBP 缓存失效；彻底修复要管理员预留
  hugetlb 大页。R9（HIP VMM，c640）过了精度门禁，第一波期间 0 次停顿。
- **预热切换时的连接重置（11-24 20:15 定位，EXPERIMENT 4.2 节末尾）**：router 的 listen backlog 只有 128，切换时约 1,000 个
  新连接 1 s 内到达，accept / SYN 队列溢出，内核改用 SYN cookie，cookie 校验失败的连接被重置（R9：`SyncookiesFailed` +58，
  与 58 个重置完全相等）。修复：LD_PRELOAD 把 router 的 backlog 提到 4096（`instr/listen_backlog.c`，kit 的
  `ROUTER_PRELOAD`，`pipeline9.sh` 默认启用）。以下是之前的记录。
- **稳定性（旧）**：DP2 lockstep 死锁在 crusoe 上一天出现 4 次（decode 3 次、MegaMoE 的 prefill 1 次），都在一边真实批次（常是刚拿到
  KV 的新请求）一边空批次时；R5 抓到现场：两边 CPU 都已发完这一步，两张 GPU 都 100% 忙在等对方（4.1 节）。流水线会保留现场并
  重新部署一次。另一类失败是 AgentX 预热切换到第二阶段（所有 lane 同时发请求）时有连接被对端重置、AIPerf 判根请求失败中止
（已 3 次，4.2 节；之前误判为第一波 1,800 s 超时），来源尚未定位，`pipeline7.sh` 开始记录宿主机 TCP 计数器。

## 文件

| 文件 / 目录 | 内容 |
|---|---|
| `HANDOFF.zh-CN.md` | 交接说明：用户要求与约束、路径与脚本、当前状态、下一步 |
| `EXPERIMENT.zh-CN.md` | 环境、与 AUS 的差异、复现结果、问题与处理（第 4 节时间线）、各轮实验与瓶颈分析（第 5 节）、复现方法 |
| `summary.txt` | 每个 run 每一步一行的时间线（工作目录 `logs/summary.log` 的副本；仓库忽略 `*.log`） |
| `scripts/` | 作业脚本（`run_opt3.sbatch`）、流水线（`pipeline5.sh`）、节点体检、准备、监控、死锁现场采集（`hang_watch2.sh`）、分析与打包工具、RCCL 微基准 |
| `arms/` | 每轮实验的配置覆盖（`base.sh` 为基线） |
| `instr/` | 逐步记录器（每个 engine 的每一步写一行 CSV，只采集、不改变行为） |
| `raw/<run>/` | 每个 run 的精简数据，格式同 `../vllm-1p1d-dsv41-20261120/raw/`；优化轮次另有 `analysis/`（指标对比、逐步汇总、profile 分类）、`steplog-*/` |

部署代码在分支 `llying/example/vllm_1p1d_dsv41_flash` 的 `examples/vllm_1p1d_dsv41_flash/`（crusoe 用 `cluster/cluster.crusoe.sh`）。
