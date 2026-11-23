# DeepSeek-V4.1-Flash 1P1D · crusoe 复现与 P2D2 提升吞吐实验

| 文件 / 目录 | 内容 |
|---|---|
| `EXPERIMENT.zh-CN.md` | 环境、与 AUS 的差异、复现结果、问题与处理、各轮优化实验、复现方法 |
| `summary.log` | 每个 run 每一步一行的时间线 |
| `scripts/` | 作业脚本（`run_opt.sbatch`）、流水线（`pipeline3.sh`）、节点体检、准备、监控、分析与打包工具 |
| `arms/` | 每轮实验的配置覆盖（`base.sh` 为基线） |
| `instr/` | 逐步记录器（每个 engine 的每一步写一行 CSV，只采集、不改变行为） |
| `raw/<run>/` | 每个 run 的精简数据，格式同 `../vllm-1p1d-dsv41-20261120/raw/`；优化轮次另有 `analysis/`、`steplog-*/` |

部署代码在分支 `llying/example/vllm_1p1d_dsv41_flash` 的 `examples/vllm_1p1d_dsv41_flash/`（crusoe 用 `cluster/cluster.crusoe.sh`）。
