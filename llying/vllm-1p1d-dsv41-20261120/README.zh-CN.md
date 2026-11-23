# DeepSeek-V4.1-Flash MXFP4 · Infera + vLLM 1P1D · AgentX 记录

部署代码与补丁在分支 `llying/example/vllm_1p1d_dsv41_flash` 的 `examples/vllm_1p1d_dsv41_flash/`
（`README.md`、`PATCHES.md`）。本目录是实验记录：

| 文件 / 目录 | 内容 |
|---|---|
| `EXPERIMENT.zh-CN.md` | 实验报告：环境、部署、门禁、两个点的配置与结果、过程中的问题、复现方法 |
| `ANALYSIS.zh-CN.md` | 分析报告：与 agg 复现和 InferenceX 的对比、每 rank 批大小、逐段延迟、瓶颈 |
| `summary.log` | 全部部署与压测的时间线（每步一行） |
| `scripts/` | 工作目录脚本：流水线、部署驱动、检查、对比与打包工具 |
| `raw/pd1p1d-c384-1120T0606/` | 1P1D c384 的原始数据 |
| `raw/pd1p1d-c768-hyper-1121T1745/` | 1P1D c768 的原始数据 |
| `reference/` | agg c384 复现的引擎指标汇总，InferenceX 公开接口返回的 MI355X vLLM AgentX 数据 |
| `vs-b300-dynamo-828k/` | 与 InferenceX B300 Dynamo-vLLM 最高点（828.2k）的配置对比与提升吞吐分析，附公开接口返回的 B300 dynamo-vllm 数据 |

## raw/ 下每个 run 的内容

部署与门禁日志原样保留（`up.log`、`smoke.log`、`accuracy*`、`kvaware*`、`golden.log`、`umbp.log`、
`*.cmd`、`mc-topo-*.json`），引擎日志压成 `.gz`。`agentx/` 下：

| 文件 | 内容 |
|---|---|
| `server_metrics_export.csv` | 正式计分阶段每个引擎指标的汇总（直方图行的列格式见分析报告第 1 节） |
| `profile_export_aiperf.json` / `.csv` / `profile_export_console.txt` | AIPerf 客户端汇总 |
| `requests.csv.gz` | 逐请求：阶段、会话、轮次、时间戳（ns）、TTFT、延迟、输入/输出长度等 |
| `engine_timeseries.csv.gz` | 每秒一行：P/D 各 rank 的在跑数、排队数、按原因分的排队数、KV 使用率（预热与正式两阶段） |
| `timeslices_key.csv.gz` | 客户端每秒时间切片：吞吐、并发、TTFT、ITL 等的 avg / p50 / p90 |
| `benchmark_command.txt`、`runtime.env`、`*.log.gz` | AIPerf 命令、运行环境与日志 |

`router_picks.csv.gz`：router 每次 kv-aware 选择的时间、角色、目标 rank、命中 block 数和请求 block 数。

以上四个压缩文件由 `scripts/pack_raw.py` 从完整数据流式裁剪得到。完整数据（含 600–770 MB 的
`server_metrics_export.json`、逐请求 `profile_export.jsonl`、完整 `router.log`）在
`/perf_apps/liyingli/bench_agentx/vllm-1p1d-dsv41-20261120/logs/<run ID>/`。

## 重新生成分析报告里的表

```bash
python3 scripts/compare_runs.py \
  c384=raw/pd1p1d-c384-1120T0606/agentx:30000,30001 \
  c768=raw/pd1p1d-c768-hyper-1121T1745/agentx:30000,30001 \
  agg=reference/agg-ix-dp2-umbp-c384-1120T0515:28000
```
