# AUS GLM-5.2 1P1D C80 benchmark — 2026-09-22

**对齐yihou设置的60分钟C80已完成，保留跨rank。** 单卡总吞吐20517 token/s（相对yihou +5.14%）、输出2610 token/s（+3.47%）；9724条正式有效完成、0正式错误、13条收尾取消。完整设置、延迟对比、warmup错误核对及剩余环境差异见[对齐轮报告](analysis/aligned-comparison.md)。

从新checkout运行请先看[复现步骤](REPRODUCE.md)。Git仅保存可复现代码、汇总和关键证据；大体积原始导出与worker日志的位置、校验值及取回方式见[产物说明](ARTIFACTS.md)。

**后续HiCache workflow已完成。** 按用户选择保留跨rank，仅开启prefill HiCache；实际备份与回读已验证，C80 profiling完成2688条有效请求、0请求错误、12条收尾取消。见[HiCache对比报告](analysis/hicache-results.md)。以下正文保留首次HiCache-off测试结果。

已完成一次 C80 AgentX 性能测试，runner 退出码 0。此轮使用 **MTP 模拟接受长度 3.61**，不是生成正确性或真实接受率测试。语义 smoke 产生重复文本并失败，原始日志已保留。

## 实际配置

- Prefill：smci355-ccs-aus-n01-33，10.235.192.136；Decode：smci355-ccs-aus-n02-21，10.235.192.128。
- 每侧 8 张 MI355X，TP8 / DP8 / EP1，合计 16 GPU；HiCache 关闭。
- GLM-5.2-MXFP4：`/perf_apps/data/models/GLM-5.2-MXFP4`；两端 282 个分片齐全，总字节数一致。
- Decode EAGLE：5 steps / 6 draft tokens，模拟接受长度 3.61；`index_share_for_mtp_iteration=false`。
- 每 rank 过滤到对应 `ionic_i`，Mooncake，GID index 1，HIP transport 与 HIP DMABUF 关闭。
- Infera `83e0f6c86cce34718f369d8669b750fa812c62d0`；SGLang `0.5.19.dev20260916+ge7f7447333`。
- 镜像 `infera-sglang:v0519-llying-aus-0922-nextnfix-hicache`，SHA256 `5c2716ea18a792b2c1c87e688e0e00afe1d20329cf45640a162a04e2c443160e`，两端一致。
- InferenceX `918524ff94045b3f091115f1051c22a8588edf2b`，AIPerf submodule `754356e9a39acc6cc6afb242d123bb57c3fb6f75`。
- 数据集 `semianalysis_cc_traces_weka_062126`，393 traces；seed 42，trajectory start ratio 0.25–0.75，idle gap cap 300 秒。

**配置与实现的关键差异：** 脚本设置了 `PD_DP_RANK_AFFINITY=1` 并传入 router 环境变量，但新构建的 router 不实现该开关，实际独立选择 P/D rank。不能将此结果标为强制同 rank。现场日志有跨 rank 配对，当前 Chat Completions → DP controller → Mooncake 接收端的代码链保留了不同 P/D rank，详见 [调研报告](analysis/cross-rank-research.md)。尚未独立验证完整 8×8 GPU/HCA 矩阵，或对每条路由样本关联 transfer completion。

## 测量结果

以下取自 [原始聚合 JSON](results/agentx-c80/agentx_conc80.json)，只统计 profiling 成功记录，排除 164 条 warmup 记录。

| 指标 | 数值 |
|---|---:|
| Profiling 发送 / 成功 / 请求错误 / 收尾取消 | 2470 / 2465 / 0 / 5 |
| 输入吞吐（含缓存命中的输入 token） | 246,332.31 token/s |
| 输出吞吐 | 1,967.04 token/s |
| 输入 + 输出吞吐 | 248,299.35 token/s |
| 输入 + 输出吞吐 / GPU（16 卡） | 15,518.71 token/s/GPU |
| 输出吞吐 / GPU（16 卡） | 122.94 token/s/GPU |
| TTFT p50 / p90 / p95 | 4.963 / 25.714 / 44.537 s |
| ITL p50 / p90 / p95 | 11.85 / 16.46 / 19.37 ms |
| E2E latency p50 / p90 | 13.344 / 52.615 s |
| Full-response interactivity，1 / p90 ITL | 60.744 token/s/user |
| 客户端 TTFT / ITL coverage | 100% / 100% |

预热 164 条全部成功，耗时 1052.86 秒。发送窗口为 **09:28:40–09:48:40 UTC**，1200 秒。客户端吞吐聚合时长为 **1229.79937 秒**（包含成功请求收尾）；phase 在 1240 秒强制完成。最后 5 条请求超过 30 秒 grace period，取消确认又超过 10 秒期限；它们占发送量约 0.202%，未计入成功记录。聚合 JSON 的 `records_total=2629` 是 2465 成功 + 164 warmup，**不是**发送量 2470，取消项需结合 runner 日志阅读。

## 验证与结果限制

- 双节点网络预检通过；22 项 Mooncake 传输全部 byte-verified，其中双向 16 项 GPU 对角线测试为 38.39–41.27 GB/s。这些预检不是非同 GPU 的独立测试。
- 语义 smoke 失败：[runner.log](results/smoke/runner.log)。模拟接受率强制接受 draft token，不能把此轮性能或零请求错误等同于正确回答。
- 65/2465 条请求（2.6%）触发输出长度偏差警告；原始输出和实际 token 计数保留，未修改 EOS/长度策略后重跑。
- 服务端指标导出出现 counter-reset 警告；P、D、router 容器 RestartCount 均为 0，启动时间保持原值。不能仅凭容器状态排除所有进程或指标异常，警告根因未验证。因此本文主要使用客户端指标；原始 JSON 中的服务端缓存命中率等统计需带此限制使用。
- 日志记录 PyTorch 缺少 `set_signal_pad_size`，multimem all-gather 被禁用；部分 BF16 GEMM shape 使用默认配置。首次 AITER JIT 编译和模型加载耗时未包含在正式测量窗口内。
- 单个 C80 点、单轮测试，不足以确定最优并发或性能方差。

## 产物与运行位置

- [汇总 CSV](results/results.csv)
- [原始聚合 JSON](results/agentx-c80/agentx_conc80.json)
- [客户端原始记录与导出](results/agentx-c80/aiperf_artifacts/)
- [完整 runner 日志](results/agentx-c80/runner.log)
- [自动生成指标图](results/agentx-c80/metrics_plots.png) 与 [workload 图](results/agentx-c80/workload_distribution_plots.png)。这些脚本的图含 warmup（2629 records），图中的全记录中位数与上表 profiling-only 数值不同。
- [网络预检](results/preflight/) 与 [启动/服务日志](results/launch/)
- [跨 rank 调研](analysis/cross-rank-research.md) 与 [现场路由样本](analysis/router-rank-picks-20260922T0936Z.log)

Docker 无法通过 NFS home 的 700 目录进行 bind mount，运行副本位于 `/perf_apps/liyingli/bench_agentx/glm52-c80-20260922`；完整 benchmark 结果已复制回本目录。未改 home 权限，未复制模型。运行结束后保留 P/D、etcd、router 服务及 AITER 缓存；无 benchmark client 遗留。

后续可用 runtime `scripts/bench-harness/stop.sh` 搭配同一 `CONFIG` 和 `MODEL` 停止本次服务。不要操作 decode 节点原有的 `xiaoming-dev` 容器。
