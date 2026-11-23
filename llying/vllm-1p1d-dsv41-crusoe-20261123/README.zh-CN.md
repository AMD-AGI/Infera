# DeepSeek-V4.1-Flash 1P1D · crusoe 复现与 P2D2 提升吞吐实验

## 结论（截至 2026-11-24 00:00 UTC）

- **复现**：crusoe c640 463,531 tok/s/GPU，与 AUS c640 471,698 一致（−1.7%）。
- **瓶颈**：P2D2 在 c640–c768 时由 P 的现算速度决定。P 两个 rank 一直跑 16k token 满步（389 ms/步，每 rank 40.9k tok/s），
  GPU 时间里 32–38% 是 MoE 的 AllGather + ReduceScatter，走两卡之间唯一一条 xGMI，已到链路带宽（约 54 GB/s/方向）。
  前缀命中（95.8%）已贴近 trace 理论上限（96.0%），加缓存不减少计算；D 有余量。
- **试过的现成配置**：加 KV 容量 + D 全 FULL graph（R1，c768 439k，与 AUS c768 459k 持平偏低，decode 变快但吞吐不变）；
  EP2 + MegaMoE（R2，P 满步慢 35%）；RCCL 通道 / 协议（无变化）；DBO（vLLM 对 Engram 模型禁用）。
- **再往上需要改代码**：MoE 通信与计算重叠、fp8 dispatch、P 改 TP2 + QuickReduce、把短 prefill 分给 D（`EXPERIMENT.zh-CN.md` 5.4 节）。
- **稳定性**：DP2 lockstep 死锁在 crusoe 上一天出现 3 次（decode 2 次、MegaMoE 的 prefill 1 次），都在空闲后第一个请求、
  一边真实批次一边空批次时；流水线现在会保留现场（py-spy）并重新部署一次。

## 文件

| 文件 / 目录 | 内容 |
|---|---|
| `EXPERIMENT.zh-CN.md` | 环境、与 AUS 的差异、复现结果、问题与处理（第 4 节时间线）、各轮实验与瓶颈分析（第 5 节）、复现方法 |
| `summary.log` | 每个 run 每一步一行的时间线 |
| `scripts/` | 作业脚本（`run_opt3.sbatch`）、流水线（`pipeline5.sh`）、节点体检、准备、监控、死锁现场采集（`hang_watch2.sh`）、分析与打包工具、RCCL 微基准 |
| `arms/` | 每轮实验的配置覆盖（`base.sh` 为基线） |
| `instr/` | 逐步记录器（每个 engine 的每一步写一行 CSV，只采集、不改变行为） |
| `raw/<run>/` | 每个 run 的精简数据，格式同 `../vllm-1p1d-dsv41-20261120/raw/`；优化轮次另有 `analysis/`（指标对比、逐步汇总、profile 分类）、`steplog-*/` |

部署代码在分支 `llying/example/vllm_1p1d_dsv41_flash` 的 `examples/vllm_1p1d_dsv41_flash/`（crusoe 用 `cluster/cluster.crusoe.sh`）。
