# 旧会话恢复记录

日期：2026-09-23 UTC。

已读取旧会话 `01a0cda1-f3f4-7de0-a6ea-56eedfe86c30` 的本地 JSONL 记录，恢复任务上下文；未修改旧会话元数据，未修复 Azure resource mismatch。

## 初次恢复时的工作和约束（历史状态）

- 原任务：解释 P8D8、conc=80/112 性能瓶颈报告中的 router、P admission、D admission、chunk 续跑及缓存机制。
- 已有解读存于 `SCHEDULING-QA.zh-CN.md`；研究初稿为 `DYNAMO-SCHEDULING-AND-CACHE-RESEARCH.zh-CN.md`。
- 用户已决定不做给短请求保留份额的 chunk 公平性实验；优先调查长尾 miss 的可避免部分，以及调度与 cache 的联合调整。
- Dynamo 研究已有本地数据分析和 debug 方案，尚缺固定版本的源码核验。不能把初稿的待核验架构描述当成已证实实现。
- 旧会话最后两次请求均为重试下载 Dynamo 源码。

## 本会话已完成

GitHub 访问返回 HTTP 200，浅克隆成功，源码固定为：

- 仓库：https://github.com/ai-dynamo/dynamo
- commit：`ff3ac59e83c73e03b98a5d0ec192ec28847130f7`
- commit 时间：2026-09-23T12:20:43+00:00
- commit 标题：`docs: refresh community events`
- 本地目录：`/perf_apps/liyingli/bench_agentx/dynamo-source-ff3ac59e`
- 克隆后 `git status --short` 为空。

旧会话的网络下载阻塞已解除。源码下载成功不代表源码审计或部署验证已完成。

## 初次恢复时的接续位置（现已完成，见文末）

按研究初稿第 9.2 节继续：核查 router 实际评分、缓存事件与负载生命周期；分别追踪各后端 P/D 编排、D 分配和传输时序；核查 GPU/host 缓存可见性与 Planner 边界。引用应使用上述 commit 的永久链接，避免用浮动 main 解释版本行为。

保留既有本地实验结论；源码研究不得替代全候选缓存机会审计、实际 token replay 或优化 A/B 的实验证据。

## 源码研究完成更新

已在本会话完成上述固定 commit 的源码研究并更新 `DYNAMO-SCHEDULING-AND-CACHE-RESEARCH.zh-CN.md` 第 2、3、9 节。初稿的网络阻塞及待核验状态已替换；证据永久链接和 SHA256 见 `dynamo-source-audit-manifest.json`。

确认：普通 D 分离式路由主动关闭 overlap、禁止共享假设；P 有独立有效 token 记账；SGLang bootstrap 允许 P/D 并发；分层命中依赖引擎事件；Router booking 不等于物理 KV reservation。剩余工作为当前引擎事件审计、缓存机会量化、token replay/优化 A/B，以及必要时的 AMD 部署验收。
