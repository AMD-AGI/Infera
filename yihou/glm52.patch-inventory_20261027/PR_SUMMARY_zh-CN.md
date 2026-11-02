# GLM-5.2 Patch 与 PR 统一进度表

状态更新：2026-11-02 04:46 UTC。本轮明细：[PR_FOLLOWUP_20261102.md](PR_FOLLOWUP_20261102.md)；维护记录：[PR_STATUS.md](PR_STATUS.md)；逐 PR／packup 对照：[PR_PACKUP_MAP.md](PR_PACKUP_MAP.md)；公网复现入口：[PUBLIC_REPRODUCE.md](PUBLIC_REPRODUCE.md)。

纠正：按作者全量反查后，GLM-5.2 核心补丁为 **13 个：1 个已合并、1 个关闭、6 个 Ready、5 个 Draft**，即 **11 个仍开放**。之前十一项表漏了后续新增的 #47276／#47278；Infera #221 是另列的内核归档 Draft，#164 是比较工具，#208 是通用 agent_sys 工具。

[四个目标仓库的全部自有 PR](OWNED_PR_INDEX.md)：**53 个历史 PR，14 个开放（7 Ready／7 Draft）、21 个合并、18 个关闭**。不要把含他人依赖 PR 的 snapshot 总数当作我们的 PR 总数。

本轮实际完成：4 个 PR rebase／推送；D10、D11 新增 CPU 回归；关闭已失去原启用理由的 D22；4 个 Ready PR 逐一跟进 CI／审阅。**Ready 不等于获批或可以合并。** SGLang 的多数旧红灯是 draft／权限 gate；#33059 新功能 CI 仍被 Draft gate 阻挡。Mooncake 旧 wheel 缺模块已修，新 CUDA CTest 缺生成头文件已有 #5144 修复单，仍待合入。未使用 GPU，也未改动历史 packup。

| Patch／一句话说明 | PR／issue 去向 | 当前进度、blocker／等谁 |
|---|---|---|
| **S10：搬 KV 前，缓存打包元数据，减少重复准备工作** | SGLang [#46819](https://github.com/sgl-project/sglang/pull/46819) | **Draft**。已有 MI355X 字节回归和独立缓存微测；本轮修正正文里过时的“GPU 未跑”描述。仍需收窄到缓存，或验证 pinned-index／优先级干扰／PP 时序。 |
| **S01：指标接收端堵塞时，不把调度器一起卡住** | SGLang [#46820](https://github.com/sgl-project/sglang/pull/46820) | **Ready**，head 未变。已再次请求维护者开放 `run-ci` 并 review；原 4 项真实 ZMQ 测试有效，旧红灯主要是 gate，不能当作功能测试失败。 |
| **D27：空闲 ROCm rank 不再误跑 DSA indexer** | SGLang [#46822](https://github.com/sgl-project/sglang/pull/46822) | **Ready**，head 未变。已跟进 CI／review；复用 R18 等价生产改动和 CPU 控制用例，无新增 GPU 运行。 |
| **D10＋D13/D16/D29：draft KV 和 target KV 一起按 DCP 布局搬运** | SGLang [#46824](https://github.com/sgl-project/sglang/pull/46824) | **Draft**。已 rebase 主线并推送 `1bcfc5214`；43 项 CPU＋61 子用例、lint 通过。#44322 已合并，接口等待解除；还需解决或排除 DP2×DCP2 graph 乱码及未覆盖组合。 |
| **D11：允许 DCP1 prefill 接 ROCm DCP decode** | SGLang [#46826](https://github.com/sgl-project/sglang/pull/46826) | **Draft**。已 rebase 并推送 `21f8e688f`；12 项 CPU＋13 子用例、lint 通过。Triton／Q-rep 默认选择已验证；仍等 #46824 的 transport 和组合正确性。 |
| **D22：可选保留 BF16 kv_b，减少 replicated-Q 路径开销** | SGLang [#46828](https://github.com/sgl-project/sglang/pull/46828) | **本轮关闭，原启用需求被替代**。#47180 已合并，主线可直接从 FP8 权重准备 Q-rep。保留原 BF16 数值与反量化 FP8 不完全等价，但没有当前基线上的独立收益证据；不再增加此开关。 |
| **原 #33973：去掉分歧 rank 路径里的 host 同步，避免互相等死** | SGLang [#46830](https://github.com/sgl-project/sglang/pull/46830) | **Ready**，head 未变。已再次跟进 CI／review；原 CPU 控制、消费者检查和历史死锁证据保留。 |
| **I05c：修复“产物明明存在，构建脚本却报失败”** | Infera [#218](https://github.com/AMD-AGI/Infera/pull/218) | **已合并，10-30**，merge `d8babfb09`；xiaobochen-amd 批准。退出活跃队列；这不表示完整历史镜像已重新构建。 |
| **M02：让 HIP 拷贝流可选使用高优先级** | Mooncake [#5005](https://github.com/kvcache-ai/Mooncake/pull/5005) | **Draft**。rebase 后为 `8313cccb`，打包源代码回归和 lint 通过；旧 wheel 问题由已合并 #5030 修复。新 CUDA CTest 缺生成头文件，已有 #5144 在修；已贡献失败日志。还等你按仓库要求人工逐行审核。 |
| **S06＝D21：OffloaderV2 搬回权重时保住 `is_shuffled` 标记** | SGLang [#45484](https://github.com/sgl-project/sglang/pull/45484) | **Ready**，head 未变。已跟进 CI／review；原 CPU 和 MI355X 两次 onload 回归保留，无需重复全模型才可审阅。 |
| **#33059：Aiter 只处理真实 query 行，再补回 padding** | SGLang [#33059](https://github.com/sgl-project/sglang/pull/33059) | **Draft**。已 rebase 到含 #46861 的主线并推送 `ea88d74bf`；12 项 CPU、lint 通过。原 Qwen 1% 准确率已对应到共享 KV 分配回归；仍缺 HaiShaw 指定的 DeepSeek-V3.2 AMD DP／stage-c 矩阵，新功能 CI 尚未执行。 |
| **K01-family：MTP verify 行共享 KV 页读取，优化 fused logits／top-k** | SGLang [#47276](https://github.com/sgl-project/sglang/pull/47276) | **Ready，之前漏列**。head `23568fdf24`；已有独立 kernel／真实接受率准确率报告，性能 serving 使用模拟接受率；等 CI／review，需协调 #45307。本次未重跑测试。 |
| **K01-family：gfx950 FP8 64-head sparse MLA decode 专用 split-K 路径** | SGLang [#47278](https://github.com/sgl-project/sglang/pull/47278) | **Ready，之前漏列**。head `375f052eda`；PR 报告参考数值／实机准确率验证；等 CI／review。它是重新集成的 kernel，不等于原 llr37 overlay。 |
| **KernelForge 源码与历史证据归档** | Infera [#221](https://github.com/AMD-AGI/Infera/pull/221) | **Draft，之前漏列**。#220 框架已合并，但该单仍指向原堆叠分支；需调整 base、验证脱敏衍生文件及外部 fixture 可用性。属于归档／工具，不是推理 runtime 修复。 |
| **通用 agent_sys 任务技能** | Infera [#208](https://github.com/AMD-AGI/Infera/pull/208) | **Ready**，是账号 PR，但不属于 GLM-5.2 核心补丁；另行处理其 CI／review。 |
| **S03/S11：把 transfer 分散到多个队列，避免挤在一条队列上** | 他人 SGLang [#42163](https://github.com/sgl-project/sglang/pull/42163) | **未重复提**。已评论贡献独立 transfer 数据和退化情况；该单也是 Draft，等 YAMY1234、维护者推进。 |
| **S05：给实际使用的 attention backend 初始化 idle 元数据** | 他人 [#45156](https://github.com/sgl-project/sglang/pull/45156)；我们原 [#45483](https://github.com/sgl-project/sglang/pull/45483) | 我们的重复单已关闭，证据已贡献且作者回应；等 WQW-Git／维护者 CI、review。 |
| **S07／之前说的 on_complete：按实际 attention-DP 宽度校验 `prefill_complete`** | 他人 SGLang [#45337](https://github.com/sgl-project/sglang/pull/45337) | **未重复提**，已贡献启动失败／成功证据；等 nvjullin 处理 review 中的性能证据问题及 CI。 |
| **D17：PP 非末尾 chunk 发 KV 前，先等计算真正完成** | 他人 [#45524](https://github.com/sgl-project/sglang/pull/45524)，相关 [#43299](https://github.com/sgl-project/sglang/pull/43299) | **#43299 已合并；#45524 仍 Open／Ready**。后者已缩小为真正需要的 PP chunk 等待，作者补充了触发用例；两者解决不同同步路径，不能混算为已完成。 |
| **D14/15/18/24/30/31：DCP 分片搜索 index，再合并 top-k** | 他人 [#42618](https://github.com/sgl-project/sglang/pull/42618)、[#44322](https://github.com/sgl-project/sglang/pull/44322) | **#44322 已于 10-28 合并；#42618 已关闭**。后端和默认值依赖已解除，D10/D11 已适配并补测。仍有 PD transport／graph 正确性缺口。 |
| **D26：空闲 rank 不再打印误导性 DCP warning** | 已在 [#42618](https://github.com/sgl-project/sglang/pull/42618) 请求顺带处理 | **无独立 PR**；原请求在已关闭的 #42618 上，需核对替代 #44322 是否仍适用后随其处理。 |
| **D23：修复 sparse attention 输出轴不一致导致的 padding 崩溃** | 已合入 [#46424](https://github.com/sgl-project/sglang/pull/46424)；原 #44262 已关闭 | **无需再提**。已核实进入 main，12 项 CPU 回归＋17 子用例通过；已回复作者。 |
| **D28：padding 不再原地改坏先前规划的长度列表** | 已合入 [#45796](https://github.com/sgl-project/sglang/pull/45796) | **无需再提**，已核实主线覆盖。 |
| **D19：各 TP rank 一致释放 deferred KV／轮询 transfer** | 释放部分已合入 [#45453](https://github.com/sgl-project/sglang/pull/45453) | 释放修复不用重提；**剩余空队列轮询 guard 未提**，尚无当前主线复现，额外 collective 成本也需证明合理。 |
| **M01：HIP IPC 子分配地址正确重定位** | 他人 Mooncake [#4340](https://github.com/kvcache-ai/Mooncake/pull/4340) | **仍 Open／Ready，未重复提**。SGLang #46861 是绕开融合 KV 分配的 workaround；仍需 salexspb／维护者推进 Mooncake 根因修复和 release。 |
| **S02：特定 absorb BMM 配置触发 LLVM 崩溃** | SGLang [issue #46829](https://github.com/sgl-project/sglang/issues/46829) | **已提 issue，未提 workaround PR**。MI355X 新复现成立；等定位编译器／配置根因。 |
| **S12：decode 保留显存持续上涨，直到空闲显存耗尽** | SGLang [issue #46831](https://github.com/sgl-project/sglang/issues/46831) | **已提 issue，未提循环 empty_cache PR**。历史证据已写明；缺最终主线栈复现和 allocator trace。 |
| **S13：没有 embedding 的 PP rank 也能取得 hidden size** | 有旧先例 [#34465](https://github.com/sgl-project/sglang/pull/34465)，已关闭未合并 | **未提**：必须先等 **llying 同意 attribution**，再厘清旧单处置；不能用 Draft 绕过归属要求。 |
| **原 #33968：HiCache 注册内存指针别名问题** | 原 [#33968](https://github.com/sgl-project/sglang/pull/33968) → 已合入 [#35233](https://github.com/sgl-project/sglang/pull/35233) | 我们旧单已关闭；主线已有修复，不再重复维护旧 workaround。 |
| **原 #33970：early KV transfer 的完成事件同步** | 原 [#33970](https://github.com/sgl-project/sglang/pull/33970) → 已合入 [#41395](https://github.com/sgl-project/sglang/pull/41395) | 我们旧单已关闭；非 CUDA #43299 已合并；剩余 PP 问题继续跟 #45524。 |
| **原 #33643：MTP penalty 应统计真正提交的 token** | 原 [#33643](https://github.com/sgl-project/sglang/pull/33643)，现跟他人 [#45233](https://github.com/sgl-project/sglang/pull/45233) | 两个旧单都已关闭未合并；#45233 作者于 10-27 23:42 关闭 stale draft、保留分支。**不能算正在推进的修复**；后续先核查主线及替代 PR，再决定恢复独立修复。 |
| **S03 layer parallelism／M03 批量 HIP copy kernel** | 无 PR | 前者无明确收益、实验回到 PAR=1；后者从未编译，生产块大小也不支持原始优化前提。 |
| **A01／原始 K01 overlay** | 原 overlay／CSV 未整体提 PR；两个重新集成的 kernel 已有 #47276／#47278 | 不能再笼统说“K01 全部没 PR”，也不能说两个 PR 覆盖了整套 overlay。#221 是证据归档；剩余组件仍需独立验证。 |
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

性能说明仅使用各 patch 的独立测量；历史 simulated-MTP 吞吐不作为单项收益或正确性证明。历史实机证据在生产行为未变时仍可复用；只有具体未覆盖路径、实现变化或评审明确要求的矩阵才需要补测。

未提 PR 的 S13／A01／剩余 K01 组件、明确排除项及已被主线覆盖的旧 PR，处置见上表；本轮没有绕过 attribution 或重复开单。
