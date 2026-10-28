# GLM-5.2 Patch 与 PR 统一进度表

状态更新：2026-10-28 05:14 UTC。公网复现入口：[PUBLIC_REPRODUCE.md](PUBLIC_REPRODUCE.md)；逐 PR／packup 对照：[PR_PACKUP_MAP.md](PR_PACKUP_MAP.md)；维护记录：[PR_STATUS.md](PR_STATUS.md)。

本轮已新建 **9 个 PR（SGLang 7、Infera 1、Mooncake 1），创建时全部设为 Draft 并写明 blocker；现 SGLang #46820／#45484／#46822／#46830 和 Infera #218 共 5 个已转 Ready for review；另提了 2 个 issue**。加上原有 #33059、#45484，目前持续维护的自有核心 PR 共 **11 个**。

新增 SGLang PR 的 lint/title 均通过，但完整 CI 多数还等维护者开放 `run-ci`。Infera #218 前一提交已执行的 CI 全绿，新增回归提交的 DCO/lint/unit/agent_sys/rust/unit-torch-cpu/docs 已通过，engine／集成 CI 排队中；Mooncake #5005 已有 29 项 CI 成功，Python 3.10 wheel 因缺少 `_structured_ranges` 模块导入失败，另有 1 项运行中。**Ready for review 表示可以正式审阅，不代表已经获批或可以合并。共 5 个已转 Ready，其余 6 个核心 PR 暂保留 Draft。历史实机证据可以复用，不再把“全部重新跑 GPU”作为默认审阅门槛。**

| Patch／一句话说明 | PR／issue 去向 | 当前进度、blocker／等谁 |
|---|---|---|
| **S10：搬 KV 前，缓存打包元数据，减少重复准备工作** | SGLang [#46819](https://github.com/sgl-project/sglang/pull/46819) | **Draft**。CPU、MI355X 字节正确性和独立缓存微测已通过；下一步收窄到已验证的缓存改动，或单独补 pinned-index／优先级／PP 时序证据，不默认重跑全模型。 |
| **S01：指标接收端堵塞时，不把调度器一起卡住** | SGLang [#46820](https://github.com/sgl-project/sglang/pull/46820) | **Ready for review**。当前提交的 4 项真实 ZMQ 回归通过，lint/title 通过；丢指标策略进入正式 review，仍等维护者开放上游 CPU CI。 |
| **D27：空闲 ROCm rank 不再误跑 DSA indexer** | SGLang [#46822](https://github.com/sgl-project/sglang/pull/46822) | **Ready for review**。生产增删行与历史实机 patch 一致，复用 R18 故障消除证据及当前 CPU 回归；后续独立故障已分开说明，仍等 CI／review。 |
| **D10＋D13/D16/D29：draft KV 和 target KV 一起按 DCP 布局搬运** | SGLang [#46824](https://github.com/sgl-project/sglang/pull/46824) | **Draft**。40 项 CPU＋53 子用例通过；EricKing626 已确认分片接口不变。按 #44322 最新实现适配，复用 pdBe-r1；DP2×DCP2 图模式乱码需定位或收窄支持范围。 |
| **D11：允许 DCP1 prefill 接 ROCm DCP decode** | SGLang [#46826](https://github.com/sgl-project/sglang/pull/46826) | **Draft**。依赖 #44322、#46824；核对最终 backend 默认选择和配置限制，并仅补历史未覆盖的组合。接口确认后可先做依赖栈适配，无需等合并才开始。 |
| **D22：可选保留 BF16 kv_b，减少 replicated-Q 路径开销** | SGLang [#46828](https://github.com/sgl-project/sglang/pull/46828) | **Draft**，默认关闭。对齐 #44322 的 replicated-Q／dtype 行为，核对 T10 与当前实现及默认关闭、数值、显存覆盖，缺什么补什么。 |
| **原 #33973：去掉分歧 rank 路径里的 host 同步，避免互相等死** | SGLang [#46830](https://github.com/sgl-project/sglang/pull/46830) | **Ready for review**。生产增删行与旧 #33973 一致；复用原 MI355X 死锁／MI300X masking 证据，并完成当前 CPU 回归及消费者检查。新整栈 GPU 运行和显存评估属于后续覆盖。 |
| **I05c：修复“产物明明存在，构建脚本却报失败”** | Infera [#218](https://github.com/AMD-AGI/Infera/pull/218) | **Ready for review**。签名提交 `5cee57e0` 新增真实 ELF／binutils 回归：修复后 11 项通过，原脚本 9 失败／2 控制项通过；本地 lint 及上游 DCO/lint/unit/agent_sys/rust/unit-torch-cpu/docs 通过，engine／集成 CI 排队中。完整镜像未重建，已明确作为集成验证限制。 |
| **M02：让 HIP 拷贝流可选使用高优先级** | Mooncake [#5005](https://github.com/kvcache-ai/Mooncake/pull/5005) | **Draft**。HIP 构建和 4/4 CTest 已通过；29 项 CI 成功，Python 3.10 wheel 缺 `_structured_ranges`，待查归因。还需仓库要求的人工逐行审核及 #4953 兼容性检查；不默认要求重跑完整 serving。 |
| **S06＝D21：OffloaderV2 搬回权重时保住 `is_shuffled` 标记** | SGLang [#45484](https://github.com/sgl-project/sglang/pull/45484) | **Ready for review**。历史三轮各 23/23 已核对原始记录，生产方法语义一致，当前提交 MI355X 回归已通过；无需为审阅重复整套模型测试，仍等上游 CI／review。 |
| **#33059：Aiter 只处理真实 query 行，再补回 padding** | SGLang [#33059](https://github.com/sgl-project/sglang/pull/33059) | **Draft**。12 项 CPU 回归通过；先对照 HaiShaw 指定的 DeepSeek-V3.2 AMD DP／stage-c 矩阵复用旧证据。新 CI 的 Qwen3-8B PD GSM8K 为 1%（门槛 >70%），需与基线比较定位，不能笼统归为环境故障。 |
| **S03/S11：把 transfer 分散到多个队列，避免挤在一条队列上** | 他人 SGLang [#42163](https://github.com/sgl-project/sglang/pull/42163) | **未重复提**。已评论贡献独立 transfer 数据和退化情况；该单也是 Draft，等 YAMY1234、维护者推进。 |
| **S05：给实际使用的 attention backend 初始化 idle 元数据** | 他人 [#45156](https://github.com/sgl-project/sglang/pull/45156)；我们原 [#45483](https://github.com/sgl-project/sglang/pull/45483) | 我们的重复单已关闭，证据已贡献且作者回应；等 WQW-Git／维护者 CI、review。 |
| **S07／之前说的 on_complete：按实际 attention-DP 宽度校验 `prefill_complete`** | 他人 SGLang [#45337](https://github.com/sgl-project/sglang/pull/45337) | **未重复提**，已贡献启动失败／成功证据；等 nvjullin 处理 review 中的性能证据问题及 CI。 |
| **D17：PP 非末尾 chunk 发 KV 前，先等计算真正完成** | 他人 [#45524](https://github.com/sgl-project/sglang/pull/45524)，相关 [#43299](https://github.com/sgl-project/sglang/pull/43299) | **未重复提**。已贡献证据；分别等 eaglewangao-us、kpjeeja 和维护者推进。 |
| **D14/15/18/24/30/31：DCP 分片搜索 index，再合并 top-k** | 他人 [#42618](https://github.com/sgl-project/sglang/pull/42618)、[#44322](https://github.com/sgl-project/sglang/pull/44322) | **未提竞争实现**。#42618 已关闭，作者确认 #44322 是替代 PR；分片策略不变。继续跟 #44322 和最终依赖适配。 |
| **D26：空闲 rank 不再打印误导性 DCP warning** | 已在 [#42618](https://github.com/sgl-project/sglang/pull/42618) 请求顺带处理 | **无独立 PR**；原请求在已关闭的 #42618 上，需核对替代 #44322 是否仍适用后随其处理。 |
| **D23：修复 sparse attention 输出轴不一致导致的 padding 崩溃** | 已合入 [#46424](https://github.com/sgl-project/sglang/pull/46424)；原 #44262 已关闭 | **无需再提**。已核实进入 main，12 项 CPU 回归＋17 子用例通过；已回复作者。 |
| **D28：padding 不再原地改坏先前规划的长度列表** | 已合入 [#45796](https://github.com/sgl-project/sglang/pull/45796) | **无需再提**，已核实主线覆盖。 |
| **D19：各 TP rank 一致释放 deferred KV／轮询 transfer** | 释放部分已合入 [#45453](https://github.com/sgl-project/sglang/pull/45453) | 释放修复不用重提；**剩余空队列轮询 guard 未提**，尚无当前主线复现，额外 collective 成本也需证明合理。 |
| **M01：HIP IPC 子分配地址正确重定位** | 他人 Mooncake [#4340](https://github.com/kvcache-ai/Mooncake/pull/4340) | **未重复提**，原作者 salexspb；我们已贡献移植验证数据，等作者／维护者。 |
| **S02：特定 absorb BMM 配置触发 LLVM 崩溃** | SGLang [issue #46829](https://github.com/sgl-project/sglang/issues/46829) | **已提 issue，未提 workaround PR**。MI355X 新复现成立；等定位编译器／配置根因。 |
| **S12：decode 保留显存持续上涨，直到空闲显存耗尽** | SGLang [issue #46831](https://github.com/sgl-project/sglang/issues/46831) | **已提 issue，未提循环 empty_cache PR**。历史证据已写明；缺最终主线栈复现和 allocator trace。 |
| **S13：没有 embedding 的 PP rank 也能取得 hidden size** | 有旧先例 [#34465](https://github.com/sgl-project/sglang/pull/34465)，已关闭未合并 | **未提**：必须先等 **llying 同意 attribution**，再厘清旧单处置；不能用 Draft 绕过归属要求。 |
| **原 #33968：HiCache 注册内存指针别名问题** | 原 [#33968](https://github.com/sgl-project/sglang/pull/33968) → 已合入 [#35233](https://github.com/sgl-project/sglang/pull/35233) | 我们旧单已关闭；主线已有修复，不再重复维护旧 workaround。 |
| **原 #33970：early KV transfer 的完成事件同步** | 原 [#33970](https://github.com/sgl-project/sglang/pull/33970) → 已合入 [#41395](https://github.com/sgl-project/sglang/pull/41395) | 我们旧单已关闭；剩余 PP／非 CUDA 问题跟 #45524／#43299。 |
| **原 #33643：MTP penalty 应统计真正提交的 token** | 原 [#33643](https://github.com/sgl-project/sglang/pull/33643)，现跟他人 [#45233](https://github.com/sgl-project/sglang/pull/45233) | 两个旧单都已关闭未合并；#45233 作者于 10-27 23:42 关闭 stale draft、保留分支。**不能算正在推进的修复**；后续先核查主线及替代 PR，再决定恢复独立修复。 |
| **S03 layer parallelism／M03 批量 HIP copy kernel** | 无 PR | 前者无明确收益、实验回到 PAR=1；后者从未编译，生产块大小也不支持原始优化前提。 |
| **A01／K01：GEMM CSV／KF kernels** | 无 PR | 正确性和集成条件未满足，按原规则暂不提交。 |
| **S04/S08/S09/S14** | 无 PR | 你明确排除的实验／退化／准确率风险项。 |
| **D12/D25、T01–T05/T07** | 无 PR | A/B 开关、探针、日志等诊断工具，不是独立生产修复。 |
| **I01/I02/I03/I06/I07：launcher、NUMA、mount、env 等** | 无核心 PR | 属于旧 bench／example；按你的要求，不包装成 Infera 核心 bugfix／feature。 |
| **I04：router rank-affinity 移植** | 无 PR | 主要是 llying 功能的移植／冲突处理，尚无独立的 yihou 当前主线修复。 |
| **I05a/b/d：旧 v0.5.21 镜像配方** | v0.5.22 已合入 Infera [#210](https://github.com/AMD-AGI/Infera/pull/210) | **不再提旧配方**。仍有价值的独立修复已拆成 #218 和 Mooncake #5005。 |
| **T06：fast bench 比较脚本** | Infera [#164](https://github.com/AMD-AGI/Infera/pull/164) | 原有 **Compare-only Draft**；不是核心修复，等你决定正式整理还是关闭。 |
| **历史 GLM-5.2 PD／DP／EAGLE 集成** | Infera [#58](https://github.com/AMD-AGI/Infera/pull/58)，替代旧 #34 | **已合并**；下游集成完成不等于所携带 SGLang patch 全部 upstream。 |
| **历史 DSA／HiCache／PD-kvd 修复** | Infera [#79](https://github.com/AMD-AGI/Infera/pull/79)，替代旧 #59 | **已合并**；不恢复旧 bundle。 |
| **历史 deployment example／结果整理** | Infera [#84](https://github.com/AMD-AGI/Infera/pull/84)、[#94](https://github.com/AMD-AGI/Infera/pull/94) | **已合并**，属于 example 维护，不计核心 feature。 |
| **历史 penalty／patch 状态文档** | Infera [#87](https://github.com/AMD-AGI/Infera/pull/87)、[#95](https://github.com/AMD-AGI/Infera/pull/95) | 已关闭未合并；功能问题和状态分别由上述 PR、`PR_STATUS.md` 接续。 |

性能说明均限定到各 patch 的独立测量，历史 simulated-MTP 数据已注明；没有拿整栈吞吐证明单项收益。GPU 验证只用了申请到的池节点，现已释放。

历史证据复用纠正：#45484、#46822、#46830 已核对生产改动与历史 patch 的等价性，结合现有回归转为 Ready；没有新增 GPU 运行。只有实际实现变化、未覆盖触发条件或评审明确要求的缺失矩阵，才应成为补测理由。详细证据链见 PR_STATUS.md 的 Historical GPU evidence reuse 段落。

最新 Draft 核对（2026-10-28 01:51 UTC）：核心仍为 **6 Draft＋5 Ready**，所有自有 PR 的 head 未变。本次只更新跟踪文档。额外的 Infera #164 是 comparison-only Draft；他人 #42163 仍为 Draft；#45233 已关闭。DCP 作者的[接口确认](https://github.com/sgl-project/sglang/pull/44322#issuecomment-6446746075)消除了“尚未答复策略问题”的等待，但依赖适配与已知图模式缺口仍需处理。
