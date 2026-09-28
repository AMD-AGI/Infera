# 会话分布、P 长尾与 Dynamo 策略对照

日期：2026-09-28。本次只分析已有记录和实现，未运行新的 GPU 实验、未修改路由策略。

## 1. 本轮与历史实验的会话分布

沿用历史脚本 `../../p8d8-adaptive-31625-20260923/scripts/analyze_session_routing.py`，加入 R1+R4、本轮 P 亲和。只统计正式窗口完成且 P/D 已配对请求；不同实验完成集合不同，保留匹配 turn 对分析。

| 实验 | 完成请求 | 多 turn conversation 数 | 使用多个 P rank 的 conversation 比例 | 连续 turn 换 P 比例 | 每个多 turn conversation 平均 P rank 数 | miss/input |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 9727 | 479 | 60.33% | 14.71% | 2.09 | 4.99% |
| G0 / R1 | 10033 | 482 | 58.09% | 10.49% | 1.81 | 4.55% |
| C1 | 9774 | 476 | 59.45% | 14.10% | 2.01 | 4.64% |
| C1G1 | 10171 | 481 | 56.34% | 10.68% | 1.77 | 4.43% |
| R1+R4 | 7167 | 406 | 88.42% | 37.65% | 3.33 | 10.92% |
| R1+P亲和 | 10252 | 490 | 0% | 0% | 1.00 | 3.83% |

G0 与本轮的 7,432 对匹配连续 turn（两个端点均满足同来源、同输出长度、输入差不超过8 tokens且不超过0.1%），换 P 比例为10.08%→0%。这是完成请求交集，不能解释未完成部分。

**conversation 和 root session 不同。** 实际 header 使用 `x_correlation_id`，子 agent/分支有自己的绑定。155 个根任务中，本轮仍有65个跨多个P rank，平均覆盖2.32个rank；G0为105个、2.77个rank。零迁移是单个绑定会话/连续conversation的结论，不是把整个根任务和所有分支固定到一张卡。

分布不能当作实际 KV 重复率。真实 GPU 重复副本比例需同一时刻各rank完整、兼容的前缀块集合：`(各rank块数之和 − 并集块数) / 各rank块数之和`，host应另算。本轮未保存完整驻留块hash及创建/驱逐流，因此无法追溯给出这个比例；零换rank也不代表零重复缓存。

- [分布总图](routing-overview.png) / [PDF](routing-overview.pdf)
- [相同40个conversation的turn/rank热图](conversation-turn-ranks.png) / [PDF](conversation-turn-ranks.pdf)
- [可选择实验、root session或conversation的交互图](session-routing.html)
- [全部分布统计](routing-summary.json)、[匹配turn对](common-pair-comparisons.json)、[根任务与miss统计](root-and-cache-summary.json)

## 2. P 长尾：多数等待者本身只需少量新计算

正式请求10,252条，P queue总计20,413.14秒。超过10秒的546条，占5.33%，贡献72.19%的累计排队；涉及263个绑定会话、121个根任务。

| 长尾子集（各行可能重叠） | 请求数 | 该子集累计排队 / 全部长尾累计排队 |
|---|---:|---:|
| miss≤4096 | 436（79.85%） | 81.69% |
| 总命中≥输入95% | 404（73.99%） | 77.28% |
| 有host命中 | 14（2.56%） | 2.57% |
| miss>32768 | 14（2.56%） | 2.18% |

这说明“大多数等待者自己miss太多/自己回读host”解释不了本轮长排队；不排除其他请求的计算、回读或共享资源拖慢它们。

下面绑定session为前8位，完整ID、conversation、root、request ID和相邻turn信息在CSV中。

| session / turn | P rank（与上一turn相同） | 上一turn输入 | 当前输入 | GPU命中 | host命中 | miss | P排队 |
|---|---:|---:|---:|---:|---:|---:|---:|
| c3452f30 /143 | 3 | 77416 | 78636 | 77376 | 0 | 1260 | 166.15s |
| 5e0dcc09 /17 | 6 | 97633 | 98082 | 97600 | 0 | 482 | 147.48s |
| 6e0d9099 /88 | 4 | 754580 | 755543 | 172480 | 582080 | 983 | 88.43s |

若旧输入内容完整保留，且旧KV仍驻留，那么上一turn输入可作为“应可复用前缀”的条件参考；三个例子的实际总命中分别仅比旧输入少40、33、20 tokens。**这不是精确的理论命中长度证明**：仅凭token计数不能证明输入内容前缀完全相同，存在模板变化、上下文截断、块边界和驱逐；上一turn的D输出也不自动成为P缓存。CSV将精确理论前缀留空，`append_only_prior_input_proxy`明确标为条件代理量。精确值需复原实际输入token序列并与历史缓存块核对。

**找到的长请求阻塞线索：**

| rank | 同rank长请求input | 长请求miss | 长请求forward区间 | 后续小miss请求排队 |
|---|---:|---:|---:|---:|
| P3 | 663758 | 663758 | 184.24s | 166.15s（miss1260） |
| P6 | 686276 | 468228，另有host218048 | 151.18s | 147.48s（miss482） |
| P4 | 555030 | 555030 | 156.30s | 143.51s（miss2904） |

P3长请求为session `69f2885c-6591-4a44-a6b3-ee3ccce2e61e`、turn0、rid `bff3e4ce-417f-41f4-a44b-d6e46e5b9793`。它约07:45:41进入forward，07:48:45结束；小miss请求07:45:56入队，约07:48:42才进入forward。P6/P4也出现类似模式。其他rank在窗口内仍有请求执行，不宜直接解释为全部P一起停顿。

546条长尾中，198条的至少80%排队时间与**同rank、forward区间超过30秒的某条请求**重叠；这198条贡献全部长尾排队的58.28%。这是同rank长请求阻塞的重要线索，不是完整因果证明。Forward区间包含调度间隙，并不代表独占GPU；需要chunk级调度证据确认准入、chunk续跑优先级及穿插条件。

**不能从现有记录算出每次“应该换到哪个rank”。** 本轮candidate logs关闭；选中rank在引擎执行时的实际hit不等于路由时所有候选的hit。缺少各候选当时的连续前缀、GPU/host状态和剩余工作量，不能把另一个rank的较短队列直接称为最优选择。本轮绑定有效时的策略选择就是原rank；“换rank一定更快”是另一个反事实问题。

- [546条长尾逐请求CSV](p-queue-over10s.csv)、[全部P请求CSV](all-p-requests.csv)、[长尾汇总](p-tail-summary.json)
- [三个阻塞窗口图](tail-blocking-windows.png) / [PDF](tail-blocking-windows.pdf)、[全部长尾重叠证据及长请求session](tail-blocking-windows.json)

## 3. R4与最短返回时间

R4主路径按 `在途预约有效输入tokens + 本请求输入tokens − 本rank估计命中tokens` 选P。预约直到P响应体结束才释放，不按chunk进度递减；该实验R3关闭，未完整表达分层回读代价。它替换了原P权重20的主评分，R1仅负责提前结束P记账。

用户提出的“选预计最快完成的P”方向正确；token数只是近似，当前正在处理的应是**剩余**工作，队列项应考虑实际先后与缓存复用。同样miss数量，长上下文attention、host回读、固定开销、batch/TP同步和chunk调度会产生不同耗时。

建议估计：`预计等待 + host回读关键路径 + prefill计算 + P→D传输关键路径`；若目标是TTFT还要考虑D侧就绪。各阶段可能重叠，不能机械把独立均值相加。对已绑定会话，只在预计少排的时间可靠地超过新增重算/回读时间及误差余量时考虑迁移。

现有R4的实测警示：相对G0，miss率4.55%→10.92%，平均TTFT7.58→26.93秒，吞吐−31.32%。它证明当前简化成本模型不合适，不否定预测完成时间这个优化目标。

## 4. 当前P亲和+R1与InferenceX Dynamo的差异

限定到已核对的GLM-5.2 GB200/GB300 Dynamo-SGLang配方，Dynamo固定SHA为`71eb001e17fa73c742f0afe1a6ed96836cb135fd`；不能泛指所有InferenceX后端或新版本。

| 项目 | 本轮Infera | 所核对InferenceX配方/固定Dynamo源码 |
|---|---|---|
| 会话绑定 | P-only；D仍逐请求选择 | P/D分别绑定worker/rank |
| 首次/绑定失效后选P | legacy active+recent−20×prefix hits | prefill有效工作及缓存抵扣＋活跃block投影；默认零温度选最低分 |
| 正常绑定后 | 保持原P | pin原worker/rank，不因另一个候选分更低就迁移 |
| GPU/host | 本轮R3关闭，评分未分别定价；引擎仍能使用host | 框架分层评分，固定默认GPU1、host0.75、disk0.25；是否实际获取完整层级事件仍需运行证据 |
| D负载 | R2关闭；本case现有路径主要依靠recent派单记账 | 普通分离D按在途block需求，关闭overlap抵扣和P工作项；绑定后继续记账 |
| P生命周期 | R1在P响应体完成时释放，含传输尾部 | 独立P生命周期处理；并非与D输出结束捆绑 |
| 拥塞迁移 | 未启用主动拥塞迁移 | 框架有过载失效重选，但目标配方三项过载阈值均None，不能说已启用 |
| 多Frontend | 单Router内存表 | 相关配方多Frontend，Nginx按session header保持入口亲和 |

Dynamo通用评分也不是经过校准的精确秒数预测；不能把它简化成R4，也不能忽略配方里的会话绑定。以上是源码/配方审计，不是对其运行逐请求日志的证明。

**后续优先级：** 保留P亲和+R1；先检查长prefill期间同rank短miss请求的准入与chunk穿插，尤其上述50万～70万token请求；改进首次分配以降低长会话集中；再用抽样全候选缓存/剩余工作量日志校准时间模型，评估受约束迁移。D的KV需求记账另行评估，不把P长尾直接归因到D，也不因InferenceX绑定D就默认复制会提升性能。

源码与既有审计：[Infera评分](../../../rust/router/src/policy.rs)、[R1–R4结果](../../r1r4-4k-20260924/R1-R4-SUMMARY.zh-CN.md)、[固定Dynamo公式及阈值审计](../../r1r4-4k-20260924/dynamo-alignment-plan-20260928/PLAN.zh-CN.md)、[InferenceX公开成绩对应配方](../../r1r4-4k-20260924/router-followup-20260928/REPORT.zh-CN.md)。上游：[选择器](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/kv-router/src/scheduling/selector.rs)、[GB300配方](https://github.com/SemiAnalysisAI/InferenceX/blob/b384c40/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5.2/gb300-fp4/agentic/glm5.2-agentx.yaml)。

## 复现与校验

`routing-summary.json`记录各轮原始run路径及joined文件SHA256。用历史脚本逐个传入`--case LABEL=run`、`--output 本目录`生成CSV/匹配统计；再运行`MPLCONFIGDIR=/tmp/agentx-mpl python3 analyze_followup.py`生成长尾表与图。脚本仅分析落盘数据。系统matplotlib与环境NumPy2存在ABI冲突，绘图显式使用系统兼容包路径，不修改全局环境。

验证：正式请求10,252条及546条长尾与前次报告一致；rank分组和长尾分组累计数/排队时间相符；相邻turn图已检查；候选最优rank与精确理论命中缺失值保持为空。不同运行的节点/完成集合/缓存历史不同，比较不是严格控制的因果实验。
