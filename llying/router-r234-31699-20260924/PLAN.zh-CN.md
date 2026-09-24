# R2/R3 性能验证与 R1/R1+R4 跑通：job31699

2026-09-24 更新：本轮已按用户要求收敛为旁路正确性观察并完成，不能作为性能对照。自动后续任务已停止；性能目标为 P/D 均 4K，必须先确认配置。结果见 [观察报告](observation/REPORT.zh-CN.md)，拟议配置见 [PROPOSED-4K-CONFIG.json](PROPOSED-4K-CONFIG.json)。以下保留最初启动计划。

- 作业：31699，账户 emad，QOS batch；StartTime 2026-09-24 06:29:58 UTC，EndTime 次日同一时间。已经过抢占保护期。
- P：smci355-ccs-aus-n02-29 / 10.235.192.61；D：smci355-ccs-aus-n06-25 / 10.235.192.59，各 8 MI355X。
- 初查两节点无运行容器，GPU busy=0，约 0.28 GiB/卡。D 新节点已从 P 复制相同镜像并校验 ID。
- Router 从 7d28ebd1 源码在基线 Ubuntu 22.04 镜像内 release 编译，避免使用登录节点 debug/不同 glibc 的产物；SHA256 见 manifest。

## 固定配置

沿用完成过的 `chunk8k-c80-31625` 基线：C80、3600 秒正式窗口、每 lane 10 warmup（此前有效总数 884），数据 revision `23f152f6f0f9399a85901b89a6458def0ef16729`，InferenceX `918524ff94045b3f091115f1051c22a8588edf2b`。

“8K chunk”指 P 实际 8192；D 实际仍为 4096。DP=8 的启动参数分别是 65536/32768。P/D mem_fraction 均 0.85，P HiCache ratio=1.5、write_through/kernel/page_first；不使用 C1 扩容。P/D seed 434370234/30296820，与原 8K 运行一致。MTP、模拟接受长度 3.61、256 调度上限等沿用原配置。发压前检查实际 server info、镜像、容器参数、环境、GPU/host 容量和客户端设置。

每轮使用独立本地 AITER 目录，安装同一份 16 个已完成库的 seed，校验哈希；不复制编译锁。节点驱动不同，记录为本轮环境限制；本轮对照保持相同 P/D 放置。

## 顺序与干预

| 阶段 | R1 | R2 | R3 | R4 | 工作 |
|---|---|---|---|---|---|
| control | decode | shadow | shadow，host=0.5 | off | 同二进制、同观测设置的原路由对照，跑通＋正式性能 |
| r2 | decode | on | shadow，host=0.5 | off | 跑通＋正式性能，与 control 比较 |
| r3 | decode | shadow | on，host=0.5 | off | 跑通＋正式性能，与 control 比较 |
| r1-smoke | completion | off | off | off | 仅跑通 |
| r1-r4-smoke | completion | off | off | on | 仅跑通 |

R2/R3 性能阶段不暗中开启 R1。Shadow 保持原评分，观测开销在各性能阶段一致。host=0.5 是预先固定的试验值，不声称本机最优。

每阶段都从重新启动的 P/D 开始，统一 smoke、缓存重置和正式 warmup。阶段结束仅停止/归档本任务捕获的容器，等待实际 GPU 和 host 内存释放再启动下一阶段；不以“容器退出”代替资源已释放，不取消用户的 allocation。

## 门槛与证据

- 实际镜像源码确认 UnifiedTreeCore 使用 `StorageMedium.CPU = CPU_PINNED`，GPU/host store/remove 都有对应调用。原样镜像不补造事件。
- 发压前将 Router 的已选择 input length 与真实 smoke response 的 prompt_tokens 对账，并要求有效需求决策；R3 要看到真实 host store 且没有 unknown medium。
- smoke 未必产生仅在 host 的命中，正式阶段继续统计 host hits、请求级回读和重算，不能由 store 事件直接声明性能收益。
- 性能以 output tokens/s/GPU、TTFT/ITL、错误取消与完整窗口为主，结合 P/D 队列及缓存机制。历史 8K 结果仅作配置和辅助参考，当前 control 用于本轮比较。
- 作业/节点/启动身份绑定的 watchdog 已启动，抢占时仅处理本任务容器；不完整窗口标无效。

运行目录：`/perf_apps/liyingli/bench_agentx/router-r234-31699-20260924`。

`CAMPAIGN_STATUS` 和 `events/campaign.log` 为顺序任务状态；各 run 的 `STATUS`、日志及原始 trace 保存在共享目录。不能把 LAUNCHING 或 BENCHMARK_C80 状态当作测试完成。
