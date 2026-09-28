# 后续实验顺序与固定配置建议

2026-09-28。根据用户提出的顺序核对代码和历史实验后整理。本次已执行 `git pull --ff-only`，从a6c4bddc快进到b9c98f36；未启动新推理实验、未申请第三台节点。用户此前分析产物保留在本目录。

## 1. 建议顺序

保持用户的主顺序：首次P评分 → D radix → D会话亲和 → Triton组合 → P8D4 → 新增P节点的2P1D → 同三台机器的4P1D（P worker各TP4/DP4，D TP8/DP8）。增加一个同镜像参考点，按阶段冻结配置。每阶段记录完整配置、引擎/Router身份、请求明细；后续组合只采用有收益或有明确必要性的选项，不默认把R2/R3/R4都打开。

| 点 | 拓扑与负载 | 改动 | 比较对象与目的 |
|---|---|---|---|
| B0 | P8/D8，C80 | 已验证R1+P会话亲和；legacy首次P评分；TileLang；D radix/亲和关闭；新补丁镜像开关关闭 | 同镜像参考；若经核验与历史基线在实际执行路径完全等价可复用，否则跑一次以隔离引擎变化 |
| B1 | 同B0 | 只改变P首次建立绑定/过期/失效后的评分，对齐固定Dynamo版本的P成本模型 | 与B0比较P排队、miss、初次派单和吞吐；绑定期间仍保留原P |
| B2 | 同上 | 在选定P策略上打开D radix，保持MTP；D HiCache关闭、D会话亲和关闭 | 与相同P策略radix-off点比较；确认实际D命中、传输减少和净性能 |
| B3 | 同B2 | 增加D会话亲和：现有prefill模式切到both | 与B2比较命中收益、D各rank活跃KV、batch偏斜、分配尾部及吞吐 |
| B3-R2（条件点） | 同B2/B3 | 若D仍有显著KV热点，再单独考察R2或缓存+负载联合选择 | 不自动同时叠加R2；硬绑定下R2主要影响首次/重选，不会自动迁移已绑定会话 |
| B4 | P8/D8，C80 | 选定Router/cache组合，只把DSA backend由TileLang/TileLang改Triton/Triton | 与同组合的TileLang点比较；这就是最终优化组合的1P1D完整结果 |
| B5 | P8/D4，C80 | 保留B4（或已证实更好的后端）组合，D改TP4/DP4 | 比较12卡与16卡的总吞吐、每GPU吞吐、TTFT/ITL、真实KV容量；先短时确认异构TP传输与容量 |
| B6 | 2个P8组+1个D8组，C80，共24卡 | 加一个P组，D恢复/保持TP8/DP8 | 与B4比较增加P容量后的收益；不要直接拿它与P8D4解释纯P扩容收益 |
| B7 | 4个P4组+1个D8组，C80，共24卡、同三台机器 | 两台P机器各拆成两个独立TP4/DP4 worker；D保持TP8/DP8 | 与B6比较相同16P卡+8D卡下的P并行粒度、计算/通信、缓存容量及长尾；与B4比较完整方案 |

B6使用第三台作第二个P，形成2P1D，每组8卡。用户随后明确增加B7：仍使用同三台机器，将两台P各拆为两个4卡worker，形成4P1D，D继续8卡。B6与B7均24卡；P8+D4为12卡点，不和它们混为同资源比较。

## 2. 首次选择的范围

只对齐P首次/重选的评分，不把D绑定、D评分变化和新的拥塞迁移混入B1。参考固定Dynamo `71eb001e…` 的有效prefill工作、分层缓存抵扣、活跃block投影与参数语义；不是重新打开现有R4，也不是只替换一个weight常数。保留R1，P亲和TTL3600。

需要明确评分模式开关与候选观测。新实现先做离线候选/预约释放核对，再短时实机；使用已有长尾请求特征构造覆盖，测量初次选择数量与后续绑定命中率。理论上更接近Dynamo不等于必然优于legacy，应由B1结果判断后续采用哪种P策略。

## 3. 用户提交的D radix如何接入

最新提交：3f0aa62a（Infera opt-in）、b9c98f36（补丁、报告及A/B/C数据）。目录 `../../decode-kv-aware-mtp-radix-20260924/`。

- SGLang补丁：`patches/01-sglang-decode-radix-allow-eagle.patch`。
- D容器env：`SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1`。
- 实际引擎flag：`--disaggregation-decode-enable-radix-cache`。当前Infera在开启D KV events且设置opt-in后可追加它；最终以实际server-info验证。
- 仅对EAGLE/NEXTN、topk1放开，本轮EAGLE5steps/draft6/topk1符合。
- D HiCache维持off；P HiCache维持现有配置。
- 补丁01已针对从AUS实际镜像提取的hook运行 `git apply --check`，通过。使用AUS现有诊断镜像做最小overlay，保留诊断和原有修复；不把另一集群整套旧镜像/launcher直接替换过来。最新Infera源码已有补丁02对应变更，构建时避免重复应用。

另一集群A/B为P4D4、C40、每组一次：同镜像D radix使**总输入+输出吞吐**约+3.0%；D HiCache叠加后约−4.5%。其正式性能使用simulate_acc_len=3.61，正确性另用真实接受率验证。不能把总吞吐3%当成输出吞吐3%，也不能直接外推到本轮P8D8/C80。

另一集群`PD_DP_RANK_AFFINITY=1`是同请求P/D rank编号配对，不是D跨turn会话绑定。AUS继续独立选择P/D；P8D4也不能照搬同编号约束。这意味着本轮radix-only命中率可能不同，B2要实际测量。

B2特别注意：D KV事件首次接入可能改变原legacy Router看到的cache/active状态。为隔离引擎radix收益，应保持D选择规则与可见负载口径一致，可通过显式radix开关而不让新增D事件参与评分实现；如果实际同时引入了D缓存评分，应明确标为组合点，不称“仅引擎radix”。随后B3只增加D亲和。

## 4. Triton与yihou/yaocheng配置证据

参考 `../../../yaocheng/2p1d-sweep-triton-dsa-20260922/`，实际参数为：

```
--dsa-prefill-backend triton
--dsa-decode-backend triton
```

不是把top-k、MoE、通信等所有backend都改成Triton。该配方保留默认sgl-kernel top-k、grouped-topk、关闭IndexShare、fused indexer及相应all-reduce设置。

同2P1D、24卡的TileLang/Triton对照：

| conc | 输出吞吐变化 | TTFT p90变化 | full-response ITL p90变化 |
|---:|---:|---:|---:|
| 80 | −0.37% | −19.68% | +4.70% |
| 112 | +2.69% | −17.71% | +3.71% |
| 192 | +7.16% | −23.78% | −0.46% |
| 256 | +9.29% | −18.40% | +22.88% |

值得测试，但不能预设C80吞吐必升。两组只有一次sweep，C256另有边界取消/预热历史；与AUS的拓扑、镜像不同。组合的每卡**输出吞吐**作为主要比较之一，总输入+输出吞吐单列，避免缓存命中的输入token数量掩盖生成效率。

配置借鉴边界：

- 保留 `index_share_for_mtp_iteration=false`、`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`。
- `SGLANG_DSA_FUSE_TOPK`控制DSA索引top-k及索引转换融合，与TileLang/Triton attention backend及MTP IndexShare是不同开关。本轮先保留fused top-k开启、IndexShare关闭。**更正此前归因：** yihou早期注释曾把关闭fusion与乱码联系起来，但同一份poll_log在2026-09-19 12:25明确记录恢复fusion后仍乱码，推翻该假设，后续指向模拟接受率模式。不能据早期注释声称`FUSE_TOPK=0`必然造成错误；选择不混入此开关是为了保留当前配置、隔离后端变量，而非认定关闭非法。若测试fusion-off，应独立比较性能，并用真实接受率验证正确性。
- AITER all-reduce fusion、custom all-reduce、scratch reclaim、NIC/DMABUF参数先与AUS实际配置对照，不和backend变化同时改动。TP4特定的通信设置若必须调整，要在B5记录为拓扑所需变更。
- 两后端均需覆盖长前缀+短新增token的编译档位，编译缓存按镜像/配置隔离持久化。新提交中已定位TileLang首次编译约11秒的停顿，但其C40长窗口结果中影响有限；不能将全部后端差异解释成JIT。

## 5. 固定测量条件与收集项目

- 基础负载沿用C80、固定数据集/InferenceX版本、884条历史预热口径、正式3600秒、相同seed与模拟接受长度3.61；新代码组合先做短时功能及实际路径检查。真实接受率的输出正确性探测与性能统计分开。
- P8D8阶段保持P/D有效chunk4096、max_running256、现有graph设定和mem_fraction0.85。D改DP4时注意launcher32768除DP后会变为8192，若仍固定有效4096需相应改launcher值并核对server-info。
- 主指标：总输出tokens/s、输出tokens/s/全部实际GPU、TTFT/ITL分位数、成功/取消/错误；另列总输入+输出吞吐。
- 路由：P/D首次选择和pin命中、实际rank、候选评分抽样、绑定失效/迁移。
- 缓存与传输：P/D实际prefix hit、D新增分配、共享/非可驱逐/可驱逐KV、实际传输bytes或页数、D bootstrap/transfer阶段。
- 排队与负载：P chunk/queue、D每rank running和KV偏斜、分配等待。D transfer wait含等待P，不能当成纯网络耗时。
- D亲和若出现不均衡，先区分首次绑定偏斜与会话后续增长；再考虑容量约束下首次选择、软缓存亲和或有证据的重选，不默认恢复自由迁移。
- 不默认启用R4或D HiCache；R2/R3只能在其边际收益可解释时进入最终组合。

## 6. 新增4P1D的具体布置与比较口径

| 机器 | B6：2P1D | B7：4P1D |
|---|---|---|
| P机器A | P0，GPU0–7，TP8/DP8 | P0，GPU0–3，TP4/DP4；P1，GPU4–7，TP4/DP4 |
| P机器B | P1，GPU0–7，TP8/DP8 | P2，GPU0–3，TP4/DP4；P3，GPU4–7，TP4/DP4 |
| D机器C | D0，GPU0–7，TP8/DP8 | D0，GPU0–7，TP8/DP8 |

均C80、P总16卡、D总8卡。Router/cache/backend策略在B6/B7之间固定，不因worker增多就调整权重。P会话绑定的目标为worker ID+本地DP rank，不能只看rank编号。

两种形态总attention DP rank均16个（2×8与4×4）。B7不是把可调度DP rank数量翻倍；变化是TP通信/计算组从2个8卡组变成4个4卡组。可能减少组内通信、缩小长请求/同步影响范围；也可能因为TP变小让单请求计算变慢。权重副本从2份增至4份，每卡权重增加、KV余量通常减少，缓存容量变化是此拓扑的实际成本，不能预设4P一定更快。

实施时核对：

- 每个P worker有效chunk保持4096；DP4时相应launcher值为16384，启动后查真实server-info。不能沿用32768导致有效chunk变8192。
- 明确max-running/graph参数是全局还是按rank分配，保持比较口径；C80下确认不触发某组额外请求上限。若为保持每rank上限需要缩放，应写入配置差异，不能仅按相同数字宣称等价。
- 同机两个worker使用独立端口段、bootstrap/KV-event/snapshot端口、IPC标识和缓存命名空间；GPU0–3与4–7不重叠。
- NIC/传输按物理GPU映射；第二个worker的本地rank0对应物理GPU4，不能直接复用第一个worker的rank→NIC表。
- 同机双worker的CPU/主机HiCache配额合计受该节点资源约束；编译缓存按兼容配置复用，运行状态与内存池隔离。
- 参考yihou同机双TP4初始化记录，分批启动并核对RCCL/端口隔离；那是历史P4+D4证据，不冒充本次P4+P4已通过。
- 先短时验证4P注册、请求覆盖全部worker、跨turn绑定和4→8传输，再进行同口径完整C80。P缓存状态按相同流程预热，不能让B7直接继承B6的热缓存来比较。

重点同时看P长尾、实际miss、每worker/rank的chunk耗时、D供给及输出吞吐。若B7长尾改善但miss增加或D吞吐下降，要把这些变化一并报告。

## 7. 资源情况

只读核对时，allocation31999仍在RUNNING，持有 `smci355-ccs-aus-n03-33`、`smci355-ccs-aus-n10-29`；截止2026-09-29 02:52:35 UTC。需把构建、预热、每点1小时测量、服务切换和HiCache释放计入排程，不假定全部点能无缝塞进现有租期。

明确排除 `smci355-ccs-aus-n04-29`，已有P/D两种角色的HIP stream原生段错误记录；不做驱动重置来把该节点强行加入性能测试。第三台在B6临近时申请，B6/B7复用同三台；复用前两台时检查剩余租期能否覆盖两轮、预热及服务切换。只读队列查看时没有明显空闲完整节点，不能承诺立即拿到。

## 证据链接

- [D radix用法与结果](../../decode-kv-aware-mtp-radix-20260924/README.zh-CN.md)、[同镜像A/B/C报告](../../decode-kv-aware-mtp-radix-20260924/analysis/PERF-AB.zh-CN.md)。
- [Triton与TileLang对照](../../../yaocheng/2p1d-sweep-triton-dsa-20260922/results/sweep_compare.md)、[冻结配置](../../../yaocheng/2p1d-sweep-triton-dsa-20260922/config.sh)。
- [yihou关于fusion/乱码归因的后续更正](../../../yihou/glm52.p8d8.agentx-sweep.packup_20260920/spec/poll_log.md)、[同机双TP4历史记录](../../../yihou/glm52.samenode-p4d4.agentx-c40.packup_20260922/README.md)。
- [P长尾与D容量分析](CAPACITY-AND-CACHE.zh-CN.md)、[会话分布与长尾明细](REPORT.zh-CN.md)。

## 8. Fusion固定与第三节点申请时机（用户补充约束）

已核对正式C80的`c80/service/{prefill,decode}-0-container.json`：两端均未设置`SGLANG_DSA_FUSE_TOPK`及旧别名`SGLANG_NSA_FUSE_TOPK`，grouped-topk=1。实际镜像environ.py默认fusion=True；本轮IndexShare=false，dsa/utils.py相应路径保持fused top-k开启。后续保持fusion开启、IndexShare关闭，切Triton时不改变fusion。证据：[实际容器检查](fusion-runtime-check.json)、[实际镜像源码检查](fusion-source-check.json)。初始snapshot只包含Cmd、不包含Env，不能用它证明环境变量不存在。

第三节点按“准备完成+剩余两节点工作量”触发：

1. 镜像、三节点拓扑、4P双worker脚本和观测均准备可用，B1–B3已确定要带入的Router/cache组合。
2. 在B4（最终P8D8/C80后端对照）开始时申请一台，初始目标为比三节点使用时间提前约2–3小时。按届时队列和实测周转时长调整；这不是Slurm等待保证。Pending时不占GPU，不以hold/依赖阻止启动来假装已预留节点。
3. 第三节点提前到：先做该节点必需的镜像/硬件检查、编译和P初始化；当前正式点完成后优先B6/B7，把尚未做的B5移到第三台释放后。避免为了死守原顺序而让第三台闲等；不中断或污染正在测量的正式点。
4. 第三节点迟到：原两台继续B4/B5及确有必要的验证；根据队列状态更新安排，不为消耗时间重复完整实验。每次启动点前核对共同可用窗口。
5. B6/B7预留约4–5小时共同窗口：两小时正式测量，加部署、预热、P8→双P4切换、HiCache释放和余量；以实际周转修正。到最晚启动检查线仍拿不到第三台，及时报告并协调原allocation续期或重新安排三节点窗口，不静默占着两台一直等待。不擅自释放用户要求保留的原allocation。
6. B6/B7完成并清理后，第三台不再保留等待P8D4；原两台沿用用户既有保留约定。

2026-09-28约12:40UTC队列快照：31999剩余约14小时11分，到期09-29 02:52:35UTC；未见空闲完整节点，有多个1节点/4节点待调度请求。优先级、回填和batch抢占使开始时间无法保证。按4–5小时共同窗口预算，最晚启动检查线约09-28 21:52–22:52UTC；这是预算线，不是预约。第三节点排除smci355-ccs-aus-n04-29。
