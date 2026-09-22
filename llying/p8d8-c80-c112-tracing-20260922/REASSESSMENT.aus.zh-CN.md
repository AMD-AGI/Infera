# C80→C112 tracing 复核与 AUS 适配方案

复核日期：2026-09-22。范围：C144 的 full/partial 报告、已保存结果、分析脚本；C80→C112 的配置、补丁、collector、runner 与分析代码；AUS aligned C80 报告；只读检查 AUS 正在运行的 Prefill 镜像源码。本次未启动新压测或修改服务。

## 结论

应在 AUS 上适配并实施 C80→C112 诊断，先补齐观测与统计闭环。现有包是可复用的原型，尚不足以直接回答“最先恶化的阶段”。不要直接将 C144 的队列积压解释成 C80 的瓶颈，也不要假定 AUS 的吞吐峰值一定在 C80。

C144 README 的“尚未启动”已过期：full run 实际完成，6697 个有效 profiling 请求、99 个收尾取消；28 个无效结果属于 warmup。C144 后续结论应以 analysis/full_run_rca.zh-CN.md 及对应 full-run-data.json 为准。

## C144 的证据与推断边界

| 观测 | 可以支持 | 不能据此确定 |
|---|---|---|
| P waiting p50=135，D prealloc=11、transfer=146 | 首 token 前多个阶段同时积压 | 哪个阶段最先变坏 |
| D KV usage 典型约90%–95% | KV 容量压力高 | 每个被阻塞请求是否因 KV、request slot 或 metadata slot 等待 |
| host pool 约99.96%使用，实际hit 84.60%，理论95.85% | 有限缓存下存在明显命中缺口 | pool满本身不是thrashing证据；缺口不能全部归因于容量或HiCache读取 |
| D running max=65，单rank未达32 | generation running 未达到配置边界 | preallocated/waiting 也可能占pool，不能排除request-pool admission限制 |
| Decode graph 472683次、eager=0 | 该窗口无eager fallback证据 | 不能排除graph内核本身耗时或其他点的行为 |
| Prefill miss tokens跨rank CV=1.17% | 全窗口累计计算token较均衡 | 短时热点、长序列成本差异、队头阻塞仍可能存在 |
| rail均值1.01–1.54GB/s，5秒p99=12.03GB/s，错误增量0 | 无持续链路打满/已计数硬故障证据 | 微突发、软件提交/轮询/完成通知延迟仍需测量 |

特别注意：Decode transfer queue 在destination发布后就开始等待，含等待Prefill排队、host恢复、计算及传输完成的时间。它与Prefill waiting很可能是在观察同一批请求，不能将两端队列或span时长直接相加。

AUS源码只读验证：`_active_req_count()` 包含running、transfer queue、waiting queue；ReqTimeStats已有bootstrap与alloc_wait拆分。该检查说明新增观测应覆盖真实资源等待，不能用running代替pool占用。AUS源码事实不能替代对另一集群历史镜像的核验。

## 现有 tracing 包的具体缺口

1. **phase污染**：run_point.sh的start/end覆盖整个warmup+profiling+drain，parse_req_time_stats.py按日志完成时间过滤；analyze_otlp_traces.py汇总全部span。headline却是profiling-only，因此两类统计不能直接对比。
2. **请求关联不完整**：OTLP按trace_id聚合；ReqTimeStats和AIPerf没有join。尚未证明跨P/D传播相同trace_id，也没有验证rid/bootstrap_room与client request/attempt的映射。compare_points.py只是分别放置两点结果，没有执行逐请求阶段归因。
3. **cache事件不是lookup耗时**：补丁在首次chunk的cache breakdown计算后发单个event。AUS源码确认该位置记录已materialized的cache tier；没有lookup或host restore开始/完成，无法测出HiCache阻塞多久。首次chunk、retraction/recompute的口径也需保留。
4. **allocation缺少原因**：现有bootstrap/alloc_wait可测总等待，但不能区分request pool、metadata pool、KV budget、保留decode tokens或队头大请求。
5. **传输语义未验收**：不能凭mooncake_send/recv名字认为它覆盖完整RDMA。需要核对调用点、异步submit与完成的边界，记录首个/最后chunk及done signal。允许重叠，不相加重复计时。
6. **rank字段有误用风险**：补丁的routed_dp_rank实际取disagg_prefill_dp_rank；compare_points.py把它作为两端rank。AUS跨rank下Decode必须单独记录实际执行dp_rank和所选prefill_dp_rank。
7. **完整性不是硬门禁**：日志获取和两种parser失败都被`|| true`吞掉；只要bench exit=0就标COMPLETED。trace_completeness只是按名字计数，没有客户端分母、失败请求或可接受缺失判据。
8. **collector边界会丢数/串点**：启动后、C80结束及C112之间反复停止重建collector，worker仍有异步缓冲。没有worker flush/确认或接收稳定屏障；不完整父span、晚到span可能跨文件。
9. **分析尚无时间演变**：缺少分窗口、按长度/cache/rank分层和未完成请求年龄。完成样本会低估过载尾部。缺少trace开销对照，不能用“async”当作无开销证明。
10. **适配不只是改IP**：config、assert_live_trace_config、assert_nodes_free、build、collector和run_point中都有旧节点/绝对路径/镜像绑定。run_c80_c112_trace.sh的`../../../../yihou`在当前llying布局下路径错误。EXIT清理不保证异常路径也等待显存释放；没有自动调用compare_points。

collector原始JSONL保存了events，但trace规范化结果只保留cache event。新增其他阶段event时，分析器也要同步保留；否则原始证据存在而派生timeline丢失。

## AUS 基线与适配边界

已有 aligned C80：20517.14 token/s/GPU、TTFT p50=5.46158s、p90=23.07297s，9724有效profiling请求。另一集群稳定C80为19513.23 token/s/GPU、TTFT p50=5.69374s。数值接近，适合作为新诊断的历史参照，不是tracing overhead A/B。

沿用AUS已验证配方：n01-33 Prefill / n02-21 Decode、P8D8、HiCache ratio1.5/write_through/kernel/page_first、max-running256、decode graph覆盖到每rank32、grouped-topk=1、模拟接受3.61、warmup10/lane、profiling3600秒、相同数据版本与seed。

保留AUS现有跨rank路由选择，不在诊断同时引入affinity优化；分别记录P/D实际rank。AUS镜像20260916和另一集群20260917、DMA-BUF设置等差异明确列入manifest。RDMA设备映射按现场拓扑核验，不能机械套用旧集群。若需要复现原集群同rank机制，应作为后续独立对照。

## 建议实施顺序

1. **在独立AUS工作目录适配**：保留原实验包；统一配置驱动路径、节点、端口、镜像和prefix。镜像从AUS已验证版本派生，只增加诊断；保存基底与派生digest、源码补丁、harness版本、有效配置。检查现有服务后再安排重启，停本实验服务并等两节点全部GPU真正释放，不重置GPU。
2. **补齐关键事件**：P enqueue/cache match/host load submit-ready/每chunk forward start-end/transfer submit-complete；D prealloc enter、bootstrap done、allocation block reason、allocation done/destination published、各transfer状态转换、done receive、waiting/first forward。资源快照包括实际free request slots、metadata slots、可分配KV与请求所需tokens。按状态切换记录，不每次poll刷日志。阶段含CPU调度还是GPU执行时间要写明，不为计时强制GPU同步。
3. **做观测验收的小流量smoke**：验证client→router→P/D关联、多chunk和实际host命中、两端rank、event时间顺序、span重复/缺失、collector落盘及错误路径。缓存load路径用直接counter交叉验证；不满足关联/阶段语义就不开始两轮小时级测试。
4. **检查时钟和观测成本**：单机duration使用monotonic；跨机wall-clock offset记录误差范围，不把误差内的差值认定为先后。用相同负载和缓存条件的短重复on/off对照量化observer影响；若量级接近历史约2.6%的吞吐回落，应降低采样或进一步重复校准。
5. **正式fresh deployment C80→C112**：保持同一deployment的缓存历史，完整采集warmup、profiling、drain及切点。collector连续运行并按请求身份/阶段分组，不靠重启文件划分点。引擎2秒/节点5秒采样保留，存原始AIPerf。切点前检查未完成请求，记录取消和残留。
6. **重建统计与定位**：client benchmark_phase决定请求cohort；profiling开始的请求即使在drain完成仍属于profiling。以30/60秒窗口检查到达/完成速率、队列年龄、allocation原因、miss tokens/s、host bytes/s和阶段时长；同时按input/miss/host tokens、output长度、rank、trajectory阶段分层。输出未完成/超时/取消的分母和最后阶段，避免只分析成功样本。
7. **按证据选择一次单变量验证**：两点只能定位候选瓶颈，不能单独证明因果。若C112未复现回落，先接受本集群曲线不同，再补相邻并发点；若出现多个阶段同时变化，补中间点C96或更细时间窗口。无需先回到C144。

## 归因与优化决策

| 候选首发阶段 | 应看到的证据 | 后续单变量验证方向 |
|---|---|---|
| cache locality/容量 | 同长度与trajectory层中hit先降、miss/request先升，随后P服务需求与队列升 | cache容量/保留策略/路由局部性，分别验证 |
| host promotion | 相近host-hit量下load-ready等待或单位字节耗时先升 | host IO、NUMA、搬运批次/调度 |
| Prefill compute/scheduler | 相近miss长度与batch条件下chunk服务时间或调度间隙先升 | chunk/batch策略、kernel与重叠执行 |
| Decode allocation | P服务与hit尚稳时明确的resource-block等待先升 | 针对具体资源调admission或KV预留；不盲目增max-running |
| KV handoff | P产出已就绪，但submit→complete或done通知阶段先变慢 | 对应传输/轮询/通知路径；低rail均值不能排除 |
| rank短时热点 | 分rank队列年龄/资源阻塞先分化，整体均值掩盖差异 | 独立验证路由/调度策略 |

对于C80，“瓶颈”要看限制吞吐的服务需求和资源利用，不一定有长队列；对于C112，要找条件可比后的新增等待及时间先后。输入token吞吐包含cache hit，不能作为实际计算吞吐。不同阶段p99不能相加；重叠P/D阶段需用逐请求关键路径解释TTFT，并报告未解释的残差。

本轮产出为方案复核。新tracing仍未运行，尚无AUS C112瓶颈结论。
