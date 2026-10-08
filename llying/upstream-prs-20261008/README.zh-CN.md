# 三个独立上游 PR：提交与复核记录

2026-10-08。按用户要求将实验功能从dev分支提取为三个独立PR；共同基线为main `ff75ec65`。未创建Mooncake PR，通用diagnostics增量继续等待#183。用户已关闭#161，本轮不以其为依赖。

| 功能 | PR | 分支 / 提交 | 范围 |
|---|---|---|---|
| R1：P完成独立释放 | [#185](https://github.com/AMD-AGI/Infera/pull/185) | `feat/pd-prefill-completion-release` / `8655e236` | 默认decode不变，显式completion模式；HTTP流式/非流式、NATS；保留main中断/abort逻辑 |
| 会话亲和 | [#186](https://github.com/AMD-AGI/Infera/pull/186) | `feat/router-session-affinity` / `5577d83e` | 默认off；模型/角色隔离、P/D独立绑定、aggregated支持、TTL/租约、失效与有界容量 |
| D radix＋MTP opt-in | [#184](https://github.com/AMD-AGI/Infera/pull/184) | `feat/sglang-decode-radix-mtp` / `7a1909a9` | 显式环境变量控制argv转发，保留模型/拓扑检查；不携带SGLang补丁、不启用D HiCache |

三个PR均直接以main为base。R1与会话亲和在guard/响应处理位置有交集，一个合入后另一个需要按两种生命周期进行rebase；不能只按文本选择一边。诊断数据、基准测试原始产物和R2/R3/R4等评分实验未混入这三个PR。

## 本次验证

- R1：289 unit、36 HTTP functional、4 ZMQ、14 render检查通过；新增NATS真实broker测试通过，确认data帧不释放P、Done/Error终止帧后释放。严格clippy、format/diff检查通过。
- 会话亲和：294 unit、39 HTTP functional、4 ZMQ、14 render检查通过；新增真实NATS 4xx测试覆盖unary/streaming，验证后续相同session重新选点。严格clippy、format/diff检查通过。
- D radix＋MTP：14个CPU argv-forwarding测试通过，ruff format/check通过。真实SGLang guard测试模块因本机没有安装SGLang而跳过；测试未将stub resolver冒充真实engine验证。
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
