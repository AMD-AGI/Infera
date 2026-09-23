# C80→C112 实验恢复与瓶颈分析

恢复日期：2026-09-23。旧会话：`01a0ca3f-8d04-7f70-99ad-9343c2d3cc47`。

## 任务与恢复结果

原任务是定位 P8D8 在 C80 的吞吐限制、C112 开始退化的位置，并评估 Prefill、Decode 和联合调度的负载均衡优化空间。用户已授权实验、保存现场、阶段性提交；若原节点到期，可以跟踪 job 31526 并迁移。旧 C144 已全面过载，只作为背景证据。

旧会话记录到 18:39，仓库 PROGRESS 到 21:10。共享盘证明两档实验已于 22:15:42 完成；最终报告于 22:16–22:17 生成。恢复时分支为 `dev/pd_opt/glm_5.2_agentx`，HEAD `84d5db07`，相对 tracking branch ahead 10；保留并检查了已有未提交分析改动。未重复启动实验。

已将最终客户端结果、关联/分层/资源/span 汇总、完成标志和 SHA256 清单保存到 `../results/main-20260922/final/`。大体积逐请求及原始采样仍位于 `/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/runs/main-20260922`。清单校验的是回收文件，不代表全部原始数据已复制。

## 性能与完整性

| 指标 | C80 | C112 |
|---|---:|---:|
| profiling sent | 9662 | 9354 |
| 有效完成且 P/D 完整关联 | 9651 | 9321 |
| drain 取消（未写入客户端 JSONL） | 11 | 33 |
| runner profiling errors | 0 | 0 |
| 总 token/s/GPU | 20115.59 | 18203.50 |
| 输出 token/s/GPU | 160.52 | 156.14 |
| TTFT mean / p50 / p90，秒 | 10.65 / 5.68 / 25.09 | 30.01 / 14.30 / 78.96 |
| 请求级实际缓存命中率 | 94.91% | 92.70% |
| profiling miss tokens | 58,906,191 | 76,532,520 |
| profiling host-hit tokens | 6,933,568 | 29,092,928 |

总吞吐下降 9.51%，输出吞吐下降 2.73%。总吞吐包含缓存命中的输入 token，不等于 GPU 实际计算吞吐；C112 总输入较少，但 miss 工作增加约 29.9%，host-hit token 增至约 4.20 倍。两档 workload/cache 历史并不相同，不能把 9.51% 全部解释成纯并发效应。

完整关联率相对 sent 为 99.886% / 99.647%；延迟统计仅代表有效导出请求，取消不是零延迟。warmup 各有 3 条空内容无效记录，不能混入 profiling 错误数。profiling engine 采样：C80 P/D 为 1792/1792 次，错误 8/11；C112 为 1790/1790 次，错误 19/17。C80 warmup 开始约 8 分钟缺资源采样，见 PROGRESS。AIPerf counter-reset 警告使其累计 server_metrics 不适合作为唯一资源依据；本报告优先用请求级结果和独立采样。

## 主要退化位置

| 同进程阶段平均耗时，秒 | C80 | C112 |
|---|---:|---:|
| Prefill queue | 5.576 | 22.409 |
| Prefill forward envelope | 3.120 | 4.594 |
| Prefill transfer tail | 0.927 | 1.222 |
| Decode allocation wait | 0.034 | 0.793 |
| Decode transfer wait | 9.421 | 28.029 |
| Decode generation | 13.133 | 12.513 |

新增等待主要落在 Prefill queue（+16.83 秒）。按 input/miss/host bins 匹配后，21 个共同 strata、权重 8665 的 queue 均值仍由 5.475 升到 21.714 秒；forward envelope 由 3.045 升到 3.936 秒。粗分层无法控制 rank、trajectory 和全部上下文差异，但说明队列变化并非仅由输入长度分布造成。

C80 已有 Prefill 排队（采样总队列 mean 14.97），增加并发后为 58.43；Prefill running 仅由 9.68 增至 10.70。GPU use 的各卡阶段均值 C80 为 98.77%–98.93%，C112 为 99.86%–100.00%。这支持 Prefill 服务能力/缓存工作量是优先优化方向，但 GPU busy 不能区分有效算力、collective 等待和内存搬运，不能据此声称计算单元已饱和。

Decode generation 未恶化，KV transfer 等待包含上游 Prefill 排队和计算，不能直接作为 RDMA 时间。Decode transfer queue mean 从 25.27 增到 72.56；running 从 35.42 降到 32.62。KV budget 阻塞请求从 38/9651（0.39%）增到 268/9321（2.88%），allocation p99 从 1.66ms 增到 28.63s。当前证据指向上游延迟扩大与 Decode 预留 KV 压力并存，而非 Decode 生成速度首先全面崩坏。

不能精确断言内部子阶段的因果先后：C112 首个 10 分钟 profiling cohort 已同时出现高 Prefill queue（18.88s）和 allocation wait（1.47s），warmup 已建立压力；且 Unified Radix Cache 的 host restore 缺少有效逐请求插桩。当前结论是“主要新增等待位置”，不是已经证明的首因。

## 负载均衡与调度

新增 `analyze_balance.py` 对同次完整 8-rank scrape 比较，输出 `balance-summary.json`。只计样本，不把缺失补零，也不将采样占比当作严格时间占比。

| 同时刻证据 | C80 | C112 |
|---|---:|---:|
| P 有排队、另一 rank 零排队 / 完整采样 | 1609/1784 | 1465/1771 |
| P 有排队、另一 rank queue=running=0 | 31/1784 | 14/1771 |
| P 所有 rank 均有排队 | 0/1784 | 256/1771 |
| D 某 rank KV≥90%，另一 rank <70% | 189/1781 | 922/1773 |

Prefill 确有瞬时队列偏斜，但零 queue 且零 running 的并存很少（1.74% / 0.79%）；TP/DP collective 又可能约束所有 rank，因此不能把零 queue 等同可用 GPU。C112 整小时 miss-token CV 反而由 0.129 降至 0.070；所有 rank 的平均排队均上升。没有证据证明“简单平均分请求”能解决主瓶颈。

Decode 累计请求数非常均衡（CV 0.00155 / 0.00280），累计输入 token CV 约 0.02，却在 C112 52.00% 的完整采样中同时出现 KV≥90% 与另一 rank <70%（C80 10.61%）。这明确说明累计请求/token 均衡不足以代表瞬时 KV 占用均衡。尚未逐个阻塞请求核验替代 rank 的可分配 token、request/metadata slots 及集体通信约束，故不能量化可挽回吞吐或认定可迁移比例。

优化优先级：

1. **Decode 实际 KV/admission 感知路由。** 原源码假设显示 distinct-prefix blocks 可能低估关闭 Decode radix cache 时的实际独立 KV 分配。候选策略应计入已预留、transfer 中与 running 请求占用；先 shadow replay 判断替代 rank 能否容纳，再做单变量对照。
2. **Prefill cache affinity 与排队工作量结合。** 基于预计 miss 计算量、host 恢复需求、排队年龄评估代价；降低 cache affinity 权重可能增加 miss，不能只优化请求数 CV。
3. **联合 P/D admission。** 验证 Decode 是否过早预留 KV 等待繁忙 Prefill，并检查 Prefill ActiveGuard 到 Decode 结束才释放导致的估计滞后。延后预留也可能损失重叠，收益需实测。

这些是有实测支持的优化候选，未完成策略干预，未声称优化收益。网络各 rail 的阶段平均发送约数 GB/s，仍不足以排除瞬时拥塞或软件传输等待；不能仅凭平均带宽断言网络无问题。

## 待办与环境限制

- 补 Unified Radix Cache 的 load_back/loading_check 观测，区分 host restore 与 chunk/compute/collective。
- 完成逐阻塞请求的替代 rank 可接纳性审计、router 估计与实际生命周期对照。
- 做 workload/cache 状态匹配的 tracing on/off 对照；历史 C80 不能量化观测开销。
- 对选中的一个调度变量进行 baseline/treatment 对照；必要时补 C96 和切换初期高频采样以识别压力先后。

当前容器 SSH/Slurm socket 均返回 `Operation not permitted`，无法核实实时节点或继续远程干预；审批配置为 never，不能在此环境提权。共享盘 job journal 最后一条为 2026-09-23 01:51:18 UTC，job 31526 当时仍 PENDING/Priority；预计节点不是已分配节点，不能当作当前状态。恢复工作已完成，远程实验与因果验证仍未完成。

验证：已有 phase/缺失分母/rank/allocation 测试和 runtime 单位/阶段/host-ack 测试通过；同时刻均衡分析在完整正式采样运行成功。最终 stage/span 统计对全部有效 profiling 请求覆盖一致，原始共享盘保留不动。
