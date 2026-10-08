# 三个独立上游 PR：提交与复核记录

2026-10-08。按用户要求将实验功能从dev分支提取为三个独立PR；共同基线为main `ff75ec65`。未创建Mooncake PR，通用diagnostics增量继续等待#183。用户已关闭#161，本轮不以其为依赖。

| 功能 | PR | 分支 / 提交 | 范围 |
|---|---|---|---|
| R1：P完成独立释放 | [#185](https://github.com/AMD-AGI/Infera/pull/185) | `feat/pd-prefill-completion-release` / `8655e236` | 默认decode不变，显式completion模式；HTTP流式/非流式、NATS；保留main中断/abort逻辑 |
| 会话亲和 | [#186](https://github.com/AMD-AGI/Infera/pull/186) | `feat/router-session-affinity` / `5577d83e` | 默认off；模型/角色隔离、P/D独立绑定、aggregated支持、TTL/租约、失效与有界容量 |
| D radix＋MTP opt-in | [#184](https://github.com/AMD-AGI/Infera/pull/184) | `feat/sglang-decode-radix-mtp` / `98f6b4e5` | 引擎补丁＋HiCache首次参数校验前准备radix；默认off；Draft等待镜像/GPU验证 |

三个PR均直接以main为base。R1与会话亲和在guard/响应处理位置有交集，一个合入后另一个需要按两种生命周期进行rebase；不能只按文本选择一边。诊断数据、基准测试原始产物和R2/R3/R4等评分实验未混入这三个PR。

## 本次验证

- R1：289 unit、36 HTTP functional、4 ZMQ、14 render检查通过；新增NATS真实broker测试通过，确认data帧不释放P、Done/Error终止帧后释放。严格clippy、format/diff检查通过。
- 会话亲和：294 unit、39 HTTP functional、4 ZMQ、14 render检查通过；新增真实NATS 4xx测试覆盖unary/streaming，验证后续相同session重新选点。严格clippy、format/diff检查通过。
- D radix＋MTP：修订后56个CPU检查通过（25个Infera参数转发/HiCache构造顺序＋31个真实上游hook源码/补丁/构建接入检查），ruff format/check通过。真实SGLang guard测试模块因本机没有安装SGLang而跳过；没有将参数分支检查冒充GPU cache或精度验证。
- 普通cargo test明确忽略需broker的测试；随后另外启动临时本地NATS，运行上述新测试和既有broker回归。临时broker已在finally中停止。

## 已知限制与失败记录

既有 `the_bucket_bootstraps_a_cold_start_without_clobbering_a_live_view` 在R1、会话分支以及干净main上均失败：写入empty snapshot后实际blocks=0，测试期望3。其余3项既有broker测试通过。本轮没有改动KV snapshot实现或这项测试，没有宣称整个broker suite全绿；已在两个Router PR说明中保留该事实。详见 `baseline-nats.txt` 与对应分支日志。

多worktree共用cargo target时发生了一次陈旧library产物混用，表现为测试编译看不到已存在的session字段。会话亲和最终检查改用独立 `/tmp/infera-pr-session-target` 并清理本crate产物后完整重跑；本文记录的最终结果来自该次通过的运行。

历史GLM-5.2实机结果是功能动机，不是本次提取后针对新main重新跑出的性能或精度结果。本轮没有手动申请GPU；GitHub自动触发的engine/GPU CI仍需其自身完成。

## 代码复核重点

- R1只移动已启动的P记账entry，不重复触发start/finish；不能在headers到达时释放；D记账保留；错误/取消释放；默认模式不改变unary读取顺序。
- 会话绑定仍使用原始RouteTarget，保留worker＋DP rank cache/load key；不靠改写Worker的rank假装候选。首次并发、TTL、capacity、旧failure callback均有覆盖。
- upstream流错误、HTTP缺失完成标记、NATS终止状态，以及以正常Response对象返回的非成功状态都能失效绑定；客户端取消不等同于worker故障。
- D opt-in仅绕过Infera对speculative组合的默认skip，其他模型/拓扑保护仍执行；支持的SGLang版本/补丁是独立前提。

PR描述保存在 `PR-r1.md`、`PR-session.md`、`PR-radix.md`。远端head与CI状态是有时间戳的 `STATUS.json` 快照，应以GitHub当前检查为准。工作分支和原始dev分支分别保留；没有将main或这三个提取分支反向合并进实验dev分支。

## #184 引擎补丁遗漏的修正

用户指出原版只修改Infera包装层，无法让未打补丁的SGLang使能MTP＋D radix。
核对确认该问题：main镜像使用v0.5.18，原生PD hook会直接拒绝组合；实验镜像另有
`01-sglang-decode-radix-allow-eagle.patch`，并未包含在原版#184中。

提交`0cdd20f4`补齐引擎半边：默认mi35x Dockerfile应用受限补丁，只有runtime opt-in=1、
EAGLE/NEXTN、top-k=1才放行；默认运行、DCP/HiSparse/backend/cache-builder限制保留。
构建补丁可通过`APPLY_SGLANG_DECODE_RADIX_SPEC_PATCH=0`关闭；旧镜像和gfx942镜像不自动获得支持。

新增测试使用v0.5.18/v0.5.19未修改hook文件，Git blob哈希核对一致；重现旧版设置env仍拒绝，
验证补丁后radix enable、其他拒绝、源码漂移失败、幂等及pyc重新编译。45项通过、1个真实engine
测试模块跳过。没有完整重建默认镜像或重新进行GPU复用/accuracy测试，因此#184改为Draft。
SGLang #40857的hybrid-SWA ownership修复未合入、也未被此次回移；不能扩展为对这些模型的支持声明。

## #184：直接通过Decode命令行启用HiCache

按用户要求，提交`98f6b4e5`进一步支持直接增加HiCache选项。此前Infera在
`ServerArgs.from_cli_args`之后才追加radix flag，HiCache会在第一次构造期间被ChunkCache
互斥检查拦住。现在显式请求D HiCache时，会先在可变的parsed namespace和转发argv中准备radix，
再进行SGLang构造与已解析ModelConfig的兼容性检查；不重复构造模型配置、不修改冻结参数。

显式HiCache请求不依赖KV事件发布；普通radix-only自动启用仍沿用KV-events触发条件。
显式disable-radix冲突和不支持模型会拒绝，显式radix flag也不会重复追加。
PR description已包含构建步骤、共同Decode参数、radix-only和radix＋HiCache两种启动命令，
另存 [使用示例](decode-cache-usage.md)。本次56项检查通过，真实SGLang模块仍跳过，4个Bash块
语法检查通过；没有启动引擎或宣称完成GPU验证。PR保持Draft。
