# 对齐设置后的 AUS C80 与 yihou C80 对比

本轮已完成。保留用户选择的跨 rank，采用3600秒profiling、每lane额外预热10次、P/D max-running和graph上限256、grouped-topk=1。Runner退出码0；正式profiling有效完成9724条，无错误记录，13条收尾取消。

本次单轮测得：相对yihou稳定C80，单卡总吞吐高5.14%，输出吞吐高3.47%；TTFT中位数低4.08%，但P90高4.29%；ITL中位数低10.42%、P90低14.43%。数值已接近yihou基线，不能再用之前未对齐的20分钟结果推断集群吞吐差距。单轮、多个设置同时改变，不证明某一项优化的独立收益，也不证明跨rank优于同rank。

## 比较口径

参考为 `yihou/glm52.p8d8.agentx-sweep.packup_20260920/results/c080/agentx_conc80.json` 的最终稳定轮（19513 token/s/GPU），不使用报告中被替代的19575旧轮。

统一读取聚合JSON中的profiling-only `request_metrics`；ITL使用`latency.itl`，不混用full-response ITL。吞吐时长包含成功请求收尾。输入token吞吐包含缓存命中的token，不等于全部重新计算的token。

机器可读数值和百分比：[aligned-comparison.json](aligned-comparison.json)。

| 指标 | yihou C80 | AUS上轮HiCache（未对齐） | AUS本轮（对齐） | 本轮相对yihou |
|---|---:|---:|---:|---:|
| 输入+输出吞吐/GPU，token/s | 19513.23 | 16891.12 | 20517.14 | +5.14% |
| 输入吞吐，token/s | 309689.01 | 268271.40 | 325664.14 | +5.16% |
| 输出吞吐，token/s | 2522.60 | 1986.49 | 2610.07 | +3.47% |
| 平均QPS | 2.61329 | 2.18730 | 2.68147 | +2.61% |
| TTFT p50，秒 | 5.69374 | 4.31090 | 5.46158 | -4.08% |
| TTFT p90，秒 | 22.12360 | 19.05189 | 23.07297 | +4.29% |
| ITL p50，毫秒 | 14.88 | 11.96 | 13.33 | -10.42% |
| ITL p90，毫秒 | 21.27 | 16.19 | 18.20 | -14.43% |
| E2E p50，秒 | 16.23218 | 11.71181 | 15.15837 | -6.62% |
| Profiling有效完成 | 9483 | 2688 | 9724 | 不比较不同时长的总数 |
| 聚合统计时长，秒 | 3629.29877 | 1229.15378 | 3627.30692 | 约相同 |

相对我们上一轮HiCache测试，本轮单卡总吞吐+21.47%、输出吞吐+31.39%；同时TTFT p50+26.69%、P90+21.11%。负载更充分时吞吐与延迟都可能上升，不能只挑一个指标判断优劣。

## 实际负载更接近

下表来自各轮AIPerf导出的`Effective * Concurrency`均值：

| 指标 | yihou | AUS上轮HiCache | AUS对齐后 |
|---|---:|---:|---:|
| 平均有效总并发 | 63.99 | 43.55 | 61.63 |
| 平均有效Decode并发 | 37.72 | 23.59 | 34.68 |
| 平均有效Prefill并发 | 26.27 | 19.96 | 26.96 |

输入长度均值：yihou 118523 token，本轮121481；输出长度均值：965.44 vs 973.62 token。虽然都叫C80，对齐前后的实际回放阶段和有效并发不同。本轮更适合与yihou长窗口作数值对照，仍非完全等环境实验。

## 设置与剩余差异

已对齐：

- C80，P8D8，16张MI355X，TP8/DP8/EP1；prefill HiCache ratio1.5、write_through、kernel、page_first。
- DURATION=3600，warmup requests/lane=10（884个wire warmup请求）。
- P/D max-running=256，P/D graph配置上限256。实际decode graph捕获列表覆盖到每rank32，符合256/DP8。
- `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`，两端容器快照均确认。
- Decode MTP EAGLE 5 steps / 6 draft / topk1，模拟接受长度3.61，`index_share_for_mtp_iteration=false`。
- 同一固定InferenceX版本和393条Weka traces、seed42、trajectory start ratio0.25–0.75、idle cap300秒。

保留差异：

1. **跨rank**：本轮`PD_DP_RANK_AFFINITY=0`，不应用affinity patch；yihou配置要求1，其历史binary未在本次重新验证。
2. **镜像/SGLang**：AUS为20260916 `ge7f7447333`，yihou为20260917 `ga9fb1c3238`；没有通过本轮重建成相同历史镜像。
3. **集群与传输设置**：AUS n01-33/n02-21 vs crsuse2-m2m-137/136；AUS保持已验证可用的`MOONCAKE_DISABLE_HIP_DMABUF=1`，yihou配置默认0（配置事实，不作为其历史运行二进制行为的证明）。
4. 模型存储路径、CPU内存和网络拓扑不同。静态配置比较见[aligned-config-differences.json](aligned-config-differences.json)，其中MODEL为便于source已统一占位，实际路径仍不同。

## 启动和请求记录核对

按用户要求，停止旧服务后等待两台节点全部GPU空闲。12:35:48 UTC的双节点检查显示16张卡均为0.096% VRAM、0% busy，harness空闲检查也全部通过，随后才launch。无GPU reset。等待证据：[wait-idle-aligned.txt](wait-idle-aligned.txt)。

预热耗时1279.16秒，runner报告884完成/0错误/0取消；原始记录审计发现其中877条有效、7条`InvalidInferenceResultError`（无实际内容），全部属于warmup。它们与884条warmup排除记录重叠，不能另加7当作正式阶段丢失。审计：[aligned-accounting-audit.json](aligned-accounting-audit.json)。

正式发送窗口：**2026-09-22 13:09:10–14:09:10 UTC**。发送9737条，有效完成9724条，正式错误0条，收尾取消13条（0.134%）。30秒grace period后取消，取消确认又超过10秒，phase在3640秒强制结束。客户端还记录了未完成branch/join清理警告；保留于runner.log。

TTFT/ITL覆盖率100%。283/9724条（2.9%）触发输出长度偏差警告。P/D/router容器RestartCount均0。服务端指标导出有counter-reset警告，缓存命中率等服务端汇总不能不带限定地用于结论。

## HiCache验证与产物

直接prefill `/metrics` 最终快照显示：累计备份84,619,392 tokens / 4,673,021,303,808 bytes；累计回读5,942,336 tokens / 328,159,563,264 bytes；dropped tokens=0。这些是本次进程全流程（含warmup）的累计值，不是仅60分钟内的增量。初始快照host-used为0，读写路径确已覆盖。

- [对齐配置](../scripts/config.phase-c-aligned.sh)
- [聚合结果](../results/agentx-c80-aligned/agentx_conc80.json)
- [AIPerf指标CSV](../results/agentx-c80-aligned/profile_export_aiperf.csv)
- [runner日志](../results/agentx-c80-aligned/runner.log)
- [实际服务快照](../results/agentx-c80-aligned/service/)
- [直接HiCache快照](../results/cache-aligned/)
- [三轮AUS汇总CSV](../results/results.csv)

完整原始AIPerf目录留在 `/perf_apps/liyingli/bench_agentx/glm52-c80-20260922/results/agentx-c80-aligned/aiperf_artifacts/`，未将GB级原始文件复制进提交范围。校验清单：[aligned-raw-artifacts-manifest.json](aligned-raw-artifacts-manifest.json)。小体积汇总、服务快照、图和日志已回收工作区。自动指标图包含有效warmup，与本报告profiling-only表不同。

本轮仍为模拟接受率性能测试，不证明生成正确性。服务保留运行，客户端已退出；未来重新launch前仍应等待两台机器所有GPU释放。
