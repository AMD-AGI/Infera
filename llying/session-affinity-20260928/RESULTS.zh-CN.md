# 会话亲和：实现与实机验收结果

2026-09-28。**会话亲和功能已实现，修复版通过离线测试和四组实机 smoke。当前未进行正式性能测试，不能据此宣称吞吐收益。P weight=40 实验已按用户方向取消。**

最终代码修复提交：`50e4a67c`。实机二进制 SHA256：`7e221373acd5d2605fcda8d94e6bc4f69b4e774f355520a3b89035d6d8f19d6b`，文件为 `/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/artifacts/infera-router-session-v2`。

## 实现范围

- 显式读取 `X-Dynamo-Session-ID`，不改动模型输入；无 session ID 时沿用原策略。
- 支持 off、prefill、both。P/D 按模型、会话、角色分别绑定 worker/rank。
- 默认空闲 TTL3600 秒；活跃租约防止请求处理中被过期，最后一个租约释放后开始空闲计时。
- 同会话首次并发选择一致；目标消失/登记变化和请求失败可以使绑定失效；旧失败回调不能删除新的绑定。
- 绑定仍经过正常缓存查询、需求预约与 ActiveGuard；R1 的 P 完成释放保持有效。
- 提供按角色的活跃租约、绑定命中和新选择计数，日志记录选择原因。

首次选择仍采用 Infera 原评分。本版本不包含完整 Dynamo 首次评分复现、主动负载拥塞逃逸或跨 Router 状态同步。重启 Router 会丢失内存绑定。

## 验证结果

离线：285 项单元、26 项 HTTP 功能、4 项额外集成、14 项 render probe 测试通过。4 项原有外部服务测试未运行。覆盖首次并发、TTL、模型隔离、目标失效、旧回调、容量边界、P/D 租约拆分和多 rank 下的预约/释放。见 [测试输出](offline-tests.txt)。

实机使用用户 allocation31999：P=n10-29，D=n03-33；镜像 `infera-sglang:aus-0922-reqtrace`；P/D 各8卡，TP8/DP8/EP1，有效 chunk=4096，HiCache 均关闭；R1启用、P/D权重20/2、R2/R3/R4关闭。

| 模式 | 请求数 | 完整 P/D 配对 | 主动断开检查 | 验收结果 |
|---|---:|---:|---:|---|
| both，TTL3600 | 18 | 17 | 1 | P/D连续请求和并发首次请求保持绑定；独立会话覆盖8个rank |
| prefill，TTL3600 | 18 | 17 | 1 | P保持绑定；D继续独立派单，D会话计数保持0 |
| both，TTL2 | 19 | 18 | 1 | P/D都记录expired并重新选择；新选择计数增加 |
| off | 18 | 17 | 1 | 不建立会话绑定，所有会话计数为0；原有路由正常 |
| 合计 | **73** | **69** | **4** | 全部通过 |

四组结束后，P/D会话活跃租约均为0。验证使用真实请求ID匹配引擎 `request_summary`，检查两端 bootstrap room 一致及实际 DP rank；不是只看HTTP200。主动断开指客户端有意关闭流，不将其当成正式请求失败或证明引擎已在某个精确 token 停止。

修复版四组还都核对了策略日志中的 `#dpN`，并观察到重复输入的 Router GPU 缓存命中。每次重启 Router 后、无请求在途时，对 P 调用 flush_cache 建立完整的缓存事件链；无需重启模型。TTL 检查器最初未剥离日志 ANSI 颜色码，未匹配到已实际出现的 expired；修正后直接复核已保存数据，没有重复发送请求。

各组详细数据、请求响应、P/D记录、策略日志均在 [smoke-results](smoke-results/) 下，文件名包含模式、TTL和唯一标识。引擎 [PID及启动信息核对](smoke-results/engine-identity.json) 确认两端各8个scheduler保持一致，验证过程中只重启Router。

## 实机核查中修复的问题

最初实现通过临时修改 worker 的 dp_rank 来限定目标，最后发出的rank正确，但策略展开候选时把它当作独立worker，内部route_key丢失 `#dpN`，影响缓存查询和需求预约。这个问题由实机策略日志发现。

已改成 Policy 直接接受 `RouteTarget` 候选；绑定路径只缩小候选集合，不修改worker登记信息。新增/加强了8-rank的账本回归测试，确认绑定请求都记在同一rank的账本里，预约正确累加、共享块不提前释放、结束后无残留。

初版36条请求仅证明实际派单行为，未计入上表的最终验收；其记录在运行目录的 `initial-validation/` 中保留。初版二进制 `6d9efd70…` 已被修复版取代，不用于后续性能实验。

## 当前状态与下一步

正式性能配置已提出：R1＋P-only会话亲和、TTL3600、P/D权重20/2，C80/4K，P开启历史HiCache配置，复用G0基线。详见 [待确认配置](performance-proposed-config.json)。目前等待用户确认，尚未启动正式预热或压测。

两端模型保持空闲运行，HiCache关闭；最后的Router处于off模式。用户allocation31999保留，未申请新机器，也未清理无关容器。若开始正式实验，需要将P切换为HiCache配置并将Router设为prefill；D可在健康且配置一致时复用。

运行目录：`/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/runs/session-affinity-31999-smoke`。初始启动配置记录初版二进制，修复后的各阶段由保存的 `router-<mode>-<ttl>-command.json` 指明实际挂载的v2二进制；后续运行以最终配置和哈希为准，不能直接恢复初始脚本。
