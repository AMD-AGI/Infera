# Infera 主分支功能提取建议与现有设计核对

核对日期：2026-10-08。本文独立保存本轮实验后的功能提取建议，并结合最新 main 和 open PR 更新实施范围。它是提取/评审计划，不表示这些候选已经通过主分支验收，也没有据此创建PR。

后续执行更新：用户已关闭#161；本轮独立创建R1 #185、会话亲和 #186、D radix＋MTP #184。原审计表保留当时状态，实际提交和验证见 [PR工作记录](../upstream-prs-20261008/README.zh-CN.md)。

主分支快照：[`ff75ec65aea83dc79498f1825f6e84f0d9c4c9ce`](https://github.com/AMD-AGI/Infera/commit/ff75ec65aea83dc79498f1825f6e84f0d9c4c9ce)。GitHub REST核对到17个open PR，已分页读取全部变更文件清单，检查相关PR描述和核心代码diff；详细PR编号、head SHA与核对时间见 [upstream-audit.json](upstream-audit.json)。结论是这个快照下的状态，未覆盖未公开分支、私下设计或未来提交；“未发现”不等于项目从未讨论过。

## 一、建议提取哪些功能

**优先提取R1（P完成释放）、可选P/D会话亲和、D radix＋MTP兼容接入。** 诊断能力和构建支持各自拆分；Triton作为已验证部署配置推荐；没有收益依据的评分策略留在实验分支。

必须分开判断“功能值得进入main”和“功能应该默认开启”。目前前者的证据比后者充分，不应把一个GLM-5.2/MI355X负载下的结果扩大为所有模型的默认行为。

| Feature | 建议与初次合入边界 | 实验依据及限制 |
|---|---|---|
| P完成后释放P路由记账（R1） | 第一优先级；拆成独立P/D记账生命周期，先显式配置 | 原容量同节点输出吞吐+4.36%、P平均排队−44.89%；340万P KV下另组同节点输出+6.31%、排队−57.58% |
| P/D各自的会话亲和 | 默认关闭的通用Router能力，提供off/prefill/both | P绑定降低漂移与miss，但对照D节点不同；D绑定在radix下使本地复用23.62%→约91.8%，不等于相同比例的容量收益 |
| D radix＋MTP opt-in | Infera参数兼容与限制检查；要求匹配的SGLang实现 | C80单开radix输出吞吐没有提升、TTFT分位数混合；其他C40同镜像对照total吞吐约+3%；不能统一承诺加速 |
| 通用诊断/指标语义 | 沿用现有metrics框架，补真正缺少的字段与关联 | D原生复用、active/resident区分、worker/rank身份帮助避免多次误判 |
| Triton DSA backend | 特定模型/硬件profile与复现证据，不改全局默认 | 同节点Output tok/s/GPU+3.38%、TTFT p50/p90−24.18%/−32.25%；ITL p90基本持平 |
| Mooncake overlay patch构建支持 | 独立小PR，无patch时行为不变 | 通用构建能力，不与某个接收网卡修复绑定 |
| R2/R3/R4、新Dynamo式P评分 | 暂缓作为性能feature提取 | R4和新P评分有负结果；R2/R3缺少支持通用启用的充分结果 |
| D HiCache默认开启 | 不建议 | 新合入的C40对照total吞吐约−4.5%、平均ITL+6.2% |

### R1：只改变P负载记账何时结束

原模式在P已经完成、请求仍在D生成时继续计入P在途负载。R1将两者生命周期分开，收益与P排队下降一致。无需等三节点容量矩阵完成才能准备这个PR。

PR中保留的不变量：完整读取P响应后释放P记账；D记账不提前释放；异常、取消路径不重复释放或泄漏；处理P传输的任务与客户端连接生命周期有明确关系。HTTP流式、非流式、NATS路径的支持范围分别测试和说明。不要带入R4、新评分、分层缓存或实验日志框架。

首次合入保留显式配置/回退方式，默认行为调整另行决定。最新main已经有中断abort逻辑，提取时必须与它兼容，见第三节。

### 会话亲和：通用能力，而不是固定rank脚本

保留会话标识、模型/角色隔离、P/D分别绑定worker/rank、首次并发一致性、活跃租约、空闲TTL、容量限制、目标失效和旧回调保护。绑定请求仍走正常缓存查询和负载记账。

默认关闭；在多轮会话负载中明确选择prefill或both。当前状态在Router进程内，重启丢失，多副本Router没有全局绑定保证；这些边界必须公开，不必第一版就做分布式会话系统。D绑定的价值与D能否保留前缀密切相关，不承诺在没有D缓存复用时自动提速。

### D radix＋MTP：Infera与SGLang分别承担自己的部分

Infera适合承载已有`3f0aa62a`一类小范围opt-in能力，保留不支持模型的拒绝检查。真正支持radix＋MTP的实现或限制放宽在SGLang；使用stock不支持版本时不能静默假装开启。版本/补丁依赖、支持组合、错误行为和有限正确性证据都需要说明。

D radix与D HiCache分别评估、分别配置，不因为前者复用成功就顺带推荐后者。模拟接受长度性能数据与真实接受率精度验证也要分开。

### 通用诊断：提炼指标，保留实验材料的边界

优先补worker/rank身份、D原生前缀复用、active/evictable/resident KV语义、阶段完成与分配等待。SGLang内部产生的指标应由对应引擎负责；Infera负责自身路由指标、联邦汇总与benchmark展示。

不把Slurm job ID、节点名、临时容器管理、整套原始trace和数千份实验产物作为核心feature的一部分。batch日志是可观察工作量，不能直接充当逐次同步GPU batch shape或padding效率。

### Triton与Mooncake采用不同路径

Triton属于后端/部署配置，提供验证过的GLM-5.2-MXFP4/MI355X配方并保留覆盖选项。保持fusion开启，不把这次收益归因于关闭fusion。

Mooncake的overlay patch接口是Infera构建能力；destination-pinned HCA等实际传输修复更适合向Mooncake上游提，再通过版本或可选镜像集成。当前特定拓扑下虽然传输失败消失，但仍有残余ACK timeout，不应把`ionic_j`映射或某个超时值变成通用默认。

## 二、最新main与open PR已经覆盖了什么

| 能力 | main快照中的状态 | 相关open PR | 对我们的提取建议 |
|---|---|---|---|
| R1：P完成独立释放 | **尚未实现同等功能**。`disagg.rs`仍为P/D共用一个ActiveGuard，随D结束释放 | [#183](https://github.com/AMD-AGI/Infera/pull/183)增加请求指标tracker，未拆P/D负载guard；[#161](https://github.com/AMD-AGI/Infera/pull/161)改派单约束，未实现R1 | 可单独提PR；需适配main中新增abort处理，并协调#183共享代码位置 |
| 跨turn会话亲和 | **未发现**会话ID→角色→worker/rank绑定、TTL/租约等实现 | **#161有不同的PD rank affinity**，见下节 | 新能力仍有提取空间；明确两种affinity的组合/互斥规则 |
| 普通D radix接入 | **已经存在**。开启KV events、D角色、Mooncake且模型兼容时自动补radix flag | 当前open变更未发现需要重做这一普通接入 | 不再提“新增D radix支持”的大PR |
| D radix＋MTP opt-in | **尚未实现**。main在speculative_algorithm非空时仍跳过自动添加radix flag | open PR未发现相同opt-in开关或等价新增实现 | 缩成MTP opt-in兼容增量，依赖SGLang支持 |
| 通用TTFT/ITL/token指标、引擎指标联邦 | main已有部分engine指标汇总，完整新方案正在PR中 | **#183直接覆盖**默认TTFT/ITL/token直方图、Python/Rust前端联邦、worker标识、运行/队列/KV相关基础序列 | 不另起平行metrics框架；在#183上补语义和缺失序列，或待合入后跟进 |
| D原生prefix、active/evictable完整语义 | 未发现我们这套完整诊断组合 | #183含`token_usage`、`num_used_tokens`、`cache_hit_rate`、running和PD队列，但当前allowlist未列`kv_evictable_tokens`/`max_total_num_tokens`等完整计算所需字段，也没有本次D allocator prefix事件 | 有增量空间；区分D本地值与P转发统计，复用worker标识及原rank labels |
| worker/rank诊断身份 | 已有worker及rank基础信息，不能统称为完全缺失 | #183注入worker_id/engine并保留原labels；[#121](https://github.com/AMD-AGI/Infera/pull/121)处理ATOM多DP rank KV事件汇聚 | 补具体关联缺口，不再创建一套身份体系；ATOM事件relay不是我们的SGLang诊断代码 |
| Triton配置 | **部分部署已经使用**。gfx942 GLM-5.2示例/recipe和e2e有`SGLANG_DSA_TRITON_PREFILL=1` | #140涉及DSA idle metadata补丁及profile，不等于本次backend性能对照 | 应提gfx950/MI355X对应配方和证据；不能声称从零增加全项目Triton支持，也不能混同两个backend开关 |
| Mooncake overlay任意patch接口 | main仍固定上游ref构建，没有`MOONCAKE_PATCHES`应用环节 | #161有其他SGLang fork/build opt-in，但不是Mooncake overlay接口；未发现同等变更 | 仍可独立提取本分支`0f2b0cf3`的通用能力 |
| benchmark/精度/profile工具 | main已有相关示例，open PR也有大量工作 | [#140](https://github.com/AMD-AGI/Infera/pull/140)精度/真实trace/按角色profile；#161 GLM PD AgentX工作流；[#164](https://github.com/AMD-AGI/Infera/pull/164)比较用draft bench | 我们的统一报告、样本完整性、D复用和KV语义应作为这些工作流的增量，不整套重建harness |
| R2/R3/R4、新P评分 | main与本次open PR未发现同等实验开关/实现 | [#108](https://github.com/AMD-AGI/Infera/pull/108)、[#110](https://github.com/AMD-AGI/Infera/pull/110)、[#111](https://github.com/AMD-AGI/Infera/pull/111)是tokenizer退化诊断/短prompt路由验证，不是新P评分 | 没有重复提案并不构成提取理由；性能证据仍不支持优先推进 |

### 最容易混淆的两种affinity

**PR #161：单次请求内的P/D rank对齐。** 当前代码先选P；若`INFERA_PD_DP_RANK_AFFINITY`开启，就限制D与P的rank编号相同。未找到该编号的D目标时返回503。它不读取会话ID、不保存跨turn绑定，也没有TTL/租约。

**我们的会话亲和：跨请求/跨turn保存各角色目标。** 例如一个会话可保持P worker-A/rank2和D worker-B/rank5，后续turn沿用各自目标。它不要求P/D的rank编号相等。

两者不是重复实现，但都修改P/D派单路径。未来若允许同时开启，需要保证首次D选择满足rank约束，后续绑定失效/故障转移仍保持定义一致。尤其P8+D4时，相同编号未必可用，不能把PR #161的严格对齐默认为我们的通用会话亲和行为。

### #183与R1也不是同一个生命周期功能

#183在streaming/部分NATS返回路径创建RequestTracker，记录请求延迟和结果。它没有把用于调度负载的P ActiveGuard独立释放；请求指标tracker完成不等于P路由预约完成。

另一方面，#183直接涉及`disagg.rs`、`proxy.rs`和`policy.rs`，与R1/会话亲和会修改相同文件。需要基于届时main和该PR状态做语义适配，不能以“功能不同”为理由忽略代码集成冲突。

### 不依赖过时PR描述判断实际内容

#161仍为Draft，其正文称`bench/glm5p2_1p1d/`尚未提交；当前API变更文件清单已经有`bench/glm5p2_pd/`工作流及rank-affinity代码。因此本文按当前head diff判断已有能力，不仅按PR正文判断。

#140是精度、trace与profile工具及DSA idle修复；即使描述提到radix正确性，也不表示实现了D radix＋MTP的参数opt-in。#164是Compare-only Draft，不能当作已接受的主分支工作流。

## 三、基于核对结果的PR拆分顺序

1. **R1独立PR**：仍为最高优先级。以最新main为基线，只提P/D记账生命周期、明确配置和测试。main现有`FireOnDrop`/`StreamEnd`/incomplete-abort处理必须保留；不能整文件覆盖旧分支的`disagg.rs`。与#183的tracker衔接也要验证。
2. **会话亲和独立PR**：默认off，范围是会话ID、目标绑定、生命周期和失效行为。评审时明确与#161 PD rank affinity的差异；不带新P评分。
3. **D radix＋MTP小PR**：只增加相对于main普通radix支持缺少的opt-in和兼容检查。遵循main目前基于`server_args`的模型限制检查，不能直接套用旧版本上下文而回退现有防护。
4. **诊断增量**：优先针对#183补D原生复用、完整KV分母/evictable语义和rank关联；正式发起前选定依赖关系，避免两个metrics框架并存。已有profile/accuracy工具优先复用#140/#161。
5. **配置与构建小PR**：MI355X对应Triton部署profile和验证说明；Mooncake overlay patch入口各自提交。实际Mooncake传输修复走对应上游及可选镜像集成，不混进Router PR。

R2/R3/R4、新首次评分与D HiCache默认开启继续暂缓。各项默认值改变应与功能引入分开讨论。本轮实验结果能支持上述提取边界，但不能代替针对最新main、支持传输路径和模型版本的回归验证。

## 四、可直接复核的来源

main使用固定SHA链接，避免后续main移动改变本文证据：

- [P/D共用guard与中断处理](https://github.com/AMD-AGI/Infera/blob/ff75ec65aea83dc79498f1825f6e84f0d9c4c9ce/rust/router/src/disagg.rs#L87)
- [普通D radix接入与MTP跳过逻辑](https://github.com/AMD-AGI/Infera/blob/ff75ec65aea83dc79498f1825f6e84f0d9c4c9ce/infera/engine/sglang/args.py#L321)
- [现有Mooncake构建脚本](https://github.com/AMD-AGI/Infera/blob/ff75ec65aea83dc79498f1825f6e84f0d9c4c9ce/deploy/docker/scripts/build_mooncake_sglang.sh)
- [gfx942已有Triton-prefill配置](https://github.com/AMD-AGI/Infera/blob/ff75ec65aea83dc79498f1825f6e84f0d9c4c9ce/examples/glm5.2_gfx942/launch/launch_prefill.sh#L48)
- [PR #183代码](https://github.com/AMD-AGI/Infera/pull/183/files)：本次head `81730d5a8751a50eae9eefbb196afd606706fbb6`
- [PR #161代码](https://github.com/AMD-AGI/Infera/pull/161/files)：精确head见 [审计快照](upstream-audit.json)
- [本轮15点性能与运行维度报告](REPORT.zh-CN.md)；[逐项机制补充](ANALYSIS.zh-CN.md)
- [其他集群D radix / D HiCache C40对照](../decode-kv-aware-mtp-radix-20260924/analysis/PERF-AB.zh-CN.md)
- [Mooncake跨rank RCA与残余ACK timeout](../cross-rank-rca-20260921/REPORT.zh-CN.md)

本次只做只读GitHub查询、fetch main和文档整理；未切换工作分支、未合并main、未发起PR、未向PR发表评论。
