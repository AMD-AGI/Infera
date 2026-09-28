# P首次选择评分：实现规格与自查

状态：实现及离线验证中，尚未运行正式性能点。

开关 `INFERA_P_DYNAMO_SCORE=off|shadow|on`，默认off。只作用于Prefill，D评分不变；已有session pin仍约束到原worker/rank，选择与预约使用同一条代码路径。要求R1=completion，与R3/R4评分互斥；R2为D侧独立控制。

固定参照为Dynamo 71eb001e… selector的默认P公式。当前实验只使用GPU/host两层、相同64-token block、无共享远端cache、无disk、无额外active-request权重或overlap衰减：

```
本请求有效工作 = max(输入tokens - (GPU命中blocks + 0.75 × host延伸blocks) × block_size, 0)
score = (在途有效P工作 + 本请求有效工作) / block_size
        + 投影活跃blocks
```

投影活跃blocks为本rank已预约full-block hash与本请求full-block hash的并集，加每请求独立的尾部partial block。已有闲置缓存不是活跃块，本次引用仍计入投影；相同完整前缀的多个在途请求共享活跃block计数，但各自有效工作仍相加。尾部按1块保守计入，不声称复现所有Dynamo registry细节。

选择、工作预约、活跃块引用预约都在同一账本锁中完成。预约由Pick携带，再随ActiveGuard移入P响应任务；未派发Pick丢弃、取消、失败或P响应体完成均释放一次。新账本与legacy active账本分开，新公式不会把二者相加导致双计。

边界：tokenizer不可用、render已确认不一致、缺block metadata时退回legacy；未知工作明确记unknown，不能当作已知零负载继续使用新评分。不同block-size候选退回legacy，不跨单位比较。缓存不存在按miss处理；缓存事件目录的新鲜度限制沿用现有体系，不声称得到了引擎物理驻留快照。

差异边界：实现限定当前两层P默认公式与预约语义，未复现Dynamo所有插件、阈值、共享cache和worker生命周期接口；并列保留Infera的legacy-load/候选顺序规则。不是给现有R4改名字，也不是精确秒级完成时间预测。

自查关注点：

- 默认off路径选择/记账不变，D始终不受P开关影响。
- shadow执行相同legacy派单，额外账本只观察，不改变释放时点（要求已有R1）。
- 原子预约覆盖pick到实际dispatch之间的并发窗口。
- 同前缀多请求、partial尾部、提前丢弃和P完成释放均无泄漏。
- session固定目标仍更新完整工作/块账本。
- host命中用0.75抵扣，GPU用1；两者为连续前缀的互斥延伸，不重复计同一块。
- 不更改fusion、MTP、chunk、客户端负载和D策略。

测试：已有单元测试通过；新增测试覆盖GPU/host候选选择、预约先于派发、共享块释放、partial尾部、未知metadata、shadow以及配置互斥。完整离线/HTTP结果完成后追加。

离线复核完成：292项单元测试通过（含16线程并发预约分布）；26项HTTP功能、4项ZMQ集成、14项render-probe通过；原有4项外部NATS服务测试保持ignored。Release构建通过。新增分层测试用固定Dynamo默认公式计算A=7、B=6.5，验证host抵扣改变选择且共享块引用正确释放；16线程未dispatch即预约测试验证4个同规格候选各分4个请求。下一步是无HiCache短时实机，验证实际引擎请求和命中事件路径。

B0参考采用已完成的同节点P亲和基线：本次overlay仅增加默认关闭的decode-radix opt-in，B1中D仍关闭radix/KV events；SGLang计算/传输内核未改，P/D原参数一致。实际runtime差异校验只有IMAGE/IMAGE_IDS及输出路径。没有为同一控制路径机械重复一小时基线；单次历史对照的时间/缓存历史局限在结果中保留。

B1回放前只读观察到P/D八rank的GPU used/evictable均为0；P host-used有一个704-token计数（可能是预检请求或计数更新延迟），没有把它冒充host完全清空的证明，也没有在回放期间再flush。正式缓存统计采用实际请求cohort。

观测口径复核：实际镜像`kv_used_tokens`已经是非可驱逐的活跃占用，`kv_evictable_tokens`是另一部分；不能再从used中减evictable。resident应为二者之和。D radix后仍按used衡量活跃压力，同时单列evictable/resident，避免把缓存填满解释成活跃容量不足。

分析代码回归：在独立临时输出目录重新解析已完成的P8D8参考数据，coverage、P cache、阶段耗时、分层、rank totals、P/D配对均与已提交结果完全一致。新分析只扩展实际DP4和多worker身份，避免把不存在的4个D rank补零或合并不同P worker的同号rank。

固定上游源码进一步核对：PromptRegistry的projection将active_blocks与incoming未共享的active blocks相加；ActiveSequences即使track_prefill_tokens=true也会acquire_prompt并增加active_blocks，prefill完成只去掉token跟踪，request free再释放块。其共享块单测验证prefill状态下仍计入该块成本。当前P按R1请求结束同时释放两部分，与本实验的独立P请求生命周期一致。
