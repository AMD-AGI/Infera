# Chunk 调度、P/D 配比、P 负载与 D radix/R2

2026-09-28。只分析已有数据、仓库代码及实际镜像源码；未启动GPU实验或改变配置。

## 1. Chunk 调度：代码与已有trace足以确认机制，收益仍需实验

实际镜像配置SHA256 `4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb`，与之前保存的源码manifest一致。本轮实际P为TP8/DP8、FCFS、有效chunk4096、dynamic_chunking=false。

`scheduler.py` 3879行先对已有 `chunked_req` 调用 `add_chunked_req`，随后3897行起遍历等待队列。`schedule_policy.py` 950行起的续跑用可用chunk预算截断请求并扣预算；1268行起的新请求准入在chunk剩余不足时截断，若剩余为零则返回OTHER；scheduler在非CONTINUE结果后停止该轮遍历。4K为rank每轮共享预算，不是每个请求各有4K。

因此，长请求可连续使用每轮完整预算；只需数百tokens的新请求仍可能等到长请求最后一段让出预算。这不是chunk没有启用，也不是FCFS在新旧请求间提供每轮公平轮转。

已从本轮OTLP原始span按父trace的rid/角色关联出：

| P rank / 长请求rid前缀 | miss | 实际chunk spans | chunk区间总和 | request forward区间 |
|---|---:|---:|---:|---:|
| P3 / bff3e4ce | 663758 | 163 | 184.238716s | 184.238716s |
| P6 / 04a53f29 | 468228 | 115 | 151.175438s | 151.175438s |
| P4 / 84bd4700 | 555030 | 136 | 156.303436s | 156.303436s |

数量均等于ceil(miss/4096)，区间和与引擎request记录相差不到1ms。这把上一份报告的长forward区间细化成真实多chunk执行证据。chunk span包含轮间等待/同步，不能当作纯kernel时间；没有逐次拒绝原因，不能给每条等待分摊所有gate。

如果修改chunk续跑顺序或给短请求留预算，仍需实机确认收益。短请求可能更早完成，但长请求完成变晚，D为该长请求预分配的KV等待更久；需同时看吞吐、TTFT分位数、D等待与KV压力。先用已有数据定位机制，不需要为此先重跑完整实验。

## 2. P TP8 / D TP4：值得研究效率，但不等于总batch容量增加

当前D实际为TP8、attention DP8、DCP1。保持每rank attention TP1的自然对照是D TP4+DP4，而不是保留DP8。若总running仍为39.27，则平均每rank由4.91升到9.82；这只说明请求集中到更少rank，不能保证D总吞吐或ITL改善。较大batch可能提升权重计算效率，较小TP可能减少通信，但总算力/带宽减少，单卡分摊权重更多。

当前D每rank max_total_num_tokens=3003264；8rank总池约2403万tokens，平均使用约819.5万、观测最大约1338万。平均使用率34.1%，但曾有rank接近99.8%，且72个采样点存在某rank>90%同时另一个<50%。平均未满不代表无局部容量约束。

权重索引metadata.total_size=438001945864字节，约407.92GiB；这只是checkpoint总字节，不是精确运行时权重分布。若粗略均分，TP8约50.99GiB/卡，TP4约101.98GiB/卡。本轮D的kv_cache_memory_usage_gb约154.47/卡、graph约3.04/卡。假设额外权重完全挤占KV池且KV每token布局不变，TP4的每卡池可能从约154.5降到103.5GiB，对应约201万tokens/rank，四rank约805万tokens。

**805万只是粗略容量推算，不能作为启动参数或已测容量。** 未精确建模复制权重、draft模型、量化运行布局、通信buffer、graph及不同TP工作区；启动后必须以真实KV容量为准。但它与当前平均819.5万的关系已说明：不能把34%简单外推成“减半D卡后还很宽裕”。

12卡相对16卡减少25%资源：吞吐相同则tokens/s/GPU提高33.3%；吞吐保留75%时每GPU效率持平。测试应区分成本效率与绝对吞吐，并检查TP8→TP4的传输、MTP及DP路由实际兼容性。第一步保持C80和其余策略，做短时容量/性能检查，不同时打开D radix或提高并发。

## 3. P 负载不均衡：瞬时明显，累计没有极端偏斜

本轮1782个正式采样点，P总等待队列均值5.68条：

| 指标 | 结果 |
|---|---:|
| 各rank累计miss范围 | 506万～646万tokens，max/min=1.277 |
| 各rank平均队列范围 | 0.329～0.999条 |
| 某rank队列≥4且另一个为0 | 660/1782，37.04% |
| 某rank队列≥8且另一个为0 | 246/1782，13.80% |
| miss>128K请求按P0…P7计数 | 3、2、7、4、3、2、2、1 |

全部24条miss>128K请求中，18条turn_index=0、6条在后续turn。turn0不是“必然在采集范围内首次建立Router绑定”的直接证明，但说明很多大工作量在会话早期可见；另外一些大miss在后续才出现，首次分配无法预知所有变化。

代码和时序显示单个超长请求就可能让所在rank出现长队；不能把所有问题归结为多个超长请求同时集中。空队列也不代表对应GPU完全空闲，attention DP仍共享TP部分。

首次选择可以改进：以候选实际缓存、当前剩余prefill工作、已有绑定会话的近期负载估计进行选择，尽量避免把新长会话放到已有长chunk/大量待处理工作的位置。但没有全候选历史缓存/剩余工作，不能反事实证明本次换派一定更快，也不能承诺加速百分比。优先期望改善排队长尾；总吞吐是否提高取决于共享计算效率和D余量。

## 4. D radix 与 R2：按可保留MTP开启radix的前提评估

**评估前提更新：用户已在另一个集群完成D radix实验，说明少量代码修改即可开启；本报告按能开启并保留当前MTP能力评估，不把现有检查作为排除理由。以下保留当前镜像事实，便于后续定位所需改动。**

实际镜像中： `pd_disaggregation_hook.py` 89–94行在D radix开启且speculative_algorithm非None时直接抛ValueError；本轮为EAGLE/MTP。这是现有版本的检查，不代表功能无法通过用户已验证的修改启用。DP attention下D radix也被标记EXPERIMENTAL，并提示需要prefix-aware DP rank路由。

D radix支持路径中，`decode.py` 1246行起匹配并锁定本地prefix，仅为缺失后缀申请相应新槽位；1571行发送`decode_prefix_len`；P的`prefill.py` 386–398行据此设置`start_send_idx`，避免传输D已拥有的前缀。

| | R2 | D radix |
|---|---|---|
| 改什么 | Router选择D并登记在途输入需求 | D引擎保留和复用prefix KV |
| 直接收益 | 避免长输入/KV需求集中到部分rank | 命中时减少新增分配与P→D传输；相同prefix可能共享物理页 |
| 不直接改变 | 总输入KV需求、生成计算 | 后续生成仍要使用上下文KV，不会因此免掉每步attention；P自身缺失前缀也不自动免算 |
| 前提 | 请求长度与账本生命周期正确 | 真正命中、缓存生命周期/引用计数和PD协议正确，最好有缓存感知路由 |
| 本次证据 | 历史实验改善KV偏斜/分配尾部，未显示显著吞吐收益 | 本集群当前配置关闭，无收益实测；按用户补充假定可以与MTP共存 |

理论上两者可叠加，目标是“在已有缓存收益和D负载之间权衡”。原样R2按完整输入记账，即使引擎能复用也仍保守高估新增需求；它不必然破坏引擎正确性，但可能把请求分散、降低命中。进一步优化应估计新增受保护KV占用、已有共享活跃块和输出增长，不能简单把全部cache hit都当作零资源需求。

D radix若有效，D会话亲和/缓存感知路由才有明确缓存理由；仍应平衡热点。仅开启D radix但继续按负载分散请求，命中不保证足够高。

在可保留MTP的前提下，本case更值得优先验证D radix的新增收益：R2已经证明能减少局部KV热点，但其原始分配等待仅影响少数请求；D radix则可能降低很多重复请求的P→D传输和重复分配。这个收益空间更广，不等于已经证明吞吐提升，也不能把D等待KV的全部时间当作网络传输。

单个请求仍需要完整上下文KV。顺序turn复用可减少分配/传输，但不会把该请求的物理上下文压缩成delta；同时活跃的请求共享同一前缀时，才可减少重复的活跃物理页。闲置radix缓存可驱逐，不能把resident增大直接解释为活跃内存恶化。P命中率也不能代替D命中率。

建议对照当前MTP+P亲和+R1，先固定其余配置验证D radix；单独统计D前缀命中tokens、实际传输bytes、共享/受保护KV、分配等待、TTFT/ITL和吞吐。如果原D路由导致命中低，再把D缓存感知选择或D亲和作为第二个变量。之后测试radix+R2，判断均衡带来的好处是否超过分散请求丢失的命中。最终应合并成缓存收益+新增活跃KV+生成负载的选择，而非长期机械叠加两套互不知情的评分。

D radix开启后，D会话亲和有了明确缓存理由，这与此前D不复用时的结论不同。但硬绑定可能放大热点，软缓存亲和加容量约束也应考虑。D radix若能减少同时活跃请求的重复KV，可能提高TP4的可行性；若仅减少顺序turn的传输，不能据此假定TP4的活跃KV需求会大幅降低。

InferenceX参考限定为之前核对的GLM-5.2 GB200/GB300 Dynamo-SGLang配方：显式D disable-radix-cache=true；固定Dynamo普通分离D override为overlap_score_credit=0、assume_kv_reuse=false、track_prefill_tokens=false。可参考其按block需求记账的方向，不能把这些成绩当作D radix或radix+R2的收益证据。其他后端/版本需另做源码与配方核验。

## 数据与源码

- [P负载与D内存统计](p-balance-and-d-memory.json)、[选定长短请求的chunk及阶段trace](selected-chunk-traces.json)、[分析脚本](analyze_capacity.py)。
- [实际镜像PD参数检查](source-evidence/pd_disaggregation_hook.py)、[KV cache构建](source-evidence/kv_cache_builder.py)、[提取哈希](source-evidence/manifest.json)。
- [已保存scheduler](../../p8d8-c80-c112-tracing-aus-20260922/analysis/source-evidence/managers/scheduler.py)、[schedule_policy](../../p8d8-c80-c112-tracing-aus-20260922/analysis/source-evidence/managers/schedule_policy.py)、[decode](../../p8d8-c80-c112-tracing-aus-20260922/analysis/source-evidence/disaggregation/decode.py)、[prefill](../../p8d8-c80-c112-tracing-aus-20260922/analysis/source-evidence/disaggregation/prefill.py)。
- [InferenceX配方审计](../../r1r4-4k-20260924/router-followup-20260928/REPORT.zh-CN.md)、[Dynamo固定版本评分与D override](../../r1r4-4k-20260924/dynamo-alignment-plan-20260928/PLAN.zh-CN.md)。
