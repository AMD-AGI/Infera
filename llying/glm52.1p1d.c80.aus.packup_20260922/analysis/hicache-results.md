# HiCache C80 workflow 结果 — 2026-09-22

按用户选择，保留跨 rank 路由，仅开启 prefill HiCache；没有加入 router affinity patch。服务启动、长上下文请求、实际 GPU→CPU 备份、CPU→GPU 回读和完整 C80 benchmark 均已执行。Runner 退出码 0。仍使用模拟 MTP 接受长度 3.61，因此本报告不是模型回答正确性验证。

## 配置和启动

配置：[config.phase-b-hicache.sh](../scripts/config.phase-b-hicache.sh)。prefill 使用 ratio=1.5、write_through、kernel IO、page_first；decode 继续 MTP，HiCache 关闭。其余模型、镜像、节点、TP8/DP8、并发80、1200秒、seed42、393 traces 等与 phase A 一致。

只重建本任务 prefill 容器；decode、router、etcd 保留。长上下文 transport probe 返回3605 input / 32 output tokens；随后两侧 flush-cache endpoint 返回成功。新 prefill 服务端参数确认 HiCache 开启，8个rank均记录 hicache_attached=True。每rank分配约211.84 GB KV host pool和48.55 GB DSA indexer host memory；总host token capacity为37,721,600。

## 客户端对比

以下使用两轮聚合 JSON 的 profiling-only request_metrics，吞吐包含成功请求收尾。输入吞吐包含缓存命中的输入token，不能解释为实际重新计算的token速率。

| 指标 | HiCache off | Prefill HiCache on |
|---|---:|---:|
| Profiling发送 | 2470 | 2700 |
| Profiling有效完成 | 2465 | 2688 |
| Profiling请求错误 | 0 | 0 |
| 收尾取消 | 5 | 12 |
| 输入吞吐 token/s | 246332.31 | 268271.40 |
| 输出吞吐 token/s | 1967.04 | 1986.49 |
| 总吞吐 / GPU（16卡）token/s | 15518.71 | 16891.12 |
| TTFT p50 / p90（秒） | 4.963 / 25.714 | 4.311 / 19.052 |
| ITL p50 / p90（毫秒） | 11.85 / 16.46 | 11.96 / 16.19 |
| E2E p50（秒） | 13.344 | 11.712 |
| 成功记录吞吐统计时长（秒） | 1229.79937 | 1229.15378 |

此单轮对比中输出吞吐+0.99%，TTFT p50下降13.15%、p90下降25.91%，ITL变化较小。两轮回放推进数量、实际输出长度及收尾取消不同；没有重复试验，不能把这些差值全部归因于HiCache，也不能据此证明吞吐有稳定提升。

HiCache发送窗口为11:00:50–11:20:50 UTC。预热耗时895.81秒。发送停止后给30秒grace period，12条请求未在期限内完成；取消确认又超过10秒，phase在1240秒强制结束。取消率12/2700=0.444%，未计入有效完成。

## HiCache读写证据

从prefill的单独 `/metrics` 端点采集，避免用多端点合并结果替代直接计数。原始快照与JSON均保留于 [results/agentx-c80-hicache](../results/agentx-c80-hicache/)，采集脚本为 [capture_hicache_metrics.py](../scripts/capture_hicache_metrics.py)。

最终快照 `hicache-after.json`（包含本次prefill进程启动后的probe、warmup和profiling）：

| 指标 | 数值 |
|---|---:|
| GPU→CPU backup tokens | 35,890,496 |
| GPU→CPU backup bytes | 1,982,016,751,104（约1.982 TB） |
| CPU→GPU load-back tokens | 845,760 |
| CPU→GPU load-back bytes | 46,706,250,240（约46.71 GB） |
| Host used / total tokens | 35,177,344 / 37,721,600 |
| HiCache dropped tokens | 0 |

`hicache-before.json`已有probe的3584 backup tokens / 197,922,816 bytes；减去它后，warmup+profiling的备份增量为35,886,912 tokens / 1,981,818,828,288 bytes。初始无load-back计数，中途与最终均出现非零回读。不能把这些全流程累计计数误称为仅20分钟窗口内的数据。

## 原始记录核对与限制

- [accounting audit](hicache-accounting-audit.json)：2852条原始记录=162条有效warmup+2条warmup错误+2688条有效profiling。两条错误为无实际内容的 `InvalidInferenceResultError`。Runner的warmup阶段计数显示0 errors，但record processor记录了这两条；报告以原始记录说明差异。
- 聚合器的 `records_error_dropped=2` 与 `records_warmup_dropped=164` 有重叠，不能相加当作166条额外丢弃；profiling本身没有错误记录。12条收尾取消也不在2852条原始记录中。
- 159/2688条profiling请求（5.9%）触发输出长度偏差警告，上轮为65/2465（2.6%）。保持既有模拟接受率及EOS策略，未为消除警告改动负载。
- TTFT、ITL覆盖率均100%。客户端收尾有取消确认超时和未完成branch/join清理警告。
- 服务端导出仍有counter-reset警告。P/D/router的容器RestartCount均0，不足以单独解释或排除指标异常。聚合器把单rank的host capacity当作全局值等问题也使其CPU usage摘要不宜作全节点依据；本报告采用直接快照中的8-rank总量。
- HiCache write/read非零证明运行路径被覆盖，不替代KV逐字节独立正确性测试；模拟接受率下生成正确性仍不在本轮证明范围。

## Affinity问题的明确答复

上一轮镜像确实没有加入 `01-router-pd-dp-rank-affinity.patch`。Base编译当前Infera源码，overlay仅包含NextN/HiCache补丁，因而router不支持 `INFERA_PD_DP_RANK_AFFINITY`。设置环境变量不会自动新增功能。补丁对当前源码 `git apply --check` 通过，但用户明确选择保持跨rank，本轮未应用。phase B显式设 `PD_DP_RANK_AFFINITY=0`，如实记录实际行为；既存router里的旧环境变量仍不被该binary读取。详见 [跨rank调研](cross-rank-research.md)。

## 产物

- [HiCache聚合JSON](../results/agentx-c80-hicache/agentx_conc80.json)
- [两轮汇总CSV](../results/results.csv)
- [HiCache完整日志](../results/agentx-c80-hicache/runner.log)
- [启动及server info](../results/launch-hicache/)
- [长上下文探针](../results/smoke-hicache/)
- [指标图](../results/agentx-c80-hicache/metrics_plots.png)（自动脚本包含有效warmup，与上表profiling-only口径不同）

产物已回收到本workspace；运行副本仍在 `/perf_apps/liyingli/bench_agentx/glm52-c80-20260922`。当前prefill HiCache服务保留运行，benchmark客户端已退出；未操作其他用户容器。
