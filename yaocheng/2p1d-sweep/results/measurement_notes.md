# 测量口径与异常记录

本轮编号 `2p1d-20260921T151735Z`，Prefill 137/138、Decode 136，每个 worker TP8/DP8，
共 24 张 MI355X。各档正式测量使用同一组服务容器和同一套引擎参数。
总吞吐包含输入和输出，输入包含缓存命中的 token；每卡指标均除以 24。
Decode 模拟 acceptance 为 3.61，沿用参考的性能设置，不作为生成正确性的证据。

## 时长与请求计数

每档正式发送窗口为 3600 秒；聚合 `duration_seconds` 还包含 grace period 内收到的响应，
因此约为 3627–3630 秒。发送停止后，未及时完成的请求由 AIPerf 按原配置取消。
所有计数保留原始导出值，不把预热、正式测量、取消请求混算成一个成功率。

| conc | 正式 records | 预热 records | records_error_dropped | error 所在阶段 | 结束 grace 取消 credits |
|---|---:|---:|---:|---|---:|
| 80 | 10551 | 884 | 1 | warmup | 7 |
| 112 | 12166 | 1243 | 4 | warmup | 22 |
| 144 | 13337 | 1598 | 7 | warmup | 31 |
| 192 | 12388 | 2128 | 7 | warmup | 40 |
| 256 | 10513 | 2845 | 28 | warmup | 210 |

已逐条读取五档的 `profile_export.jsonl`：导出的 `InvalidInferenceResultError`
全部属于 warmup，profiling 记录中没有 `error`。AIPerf 的实时 `errors=0` 是另一条
统计路径，不能单凭它推断逐请求结果没有错误。独立核对记录位于
`.tmp/validation/measured-record-accounting.json`。

## C256 预热异常与人工重试

首次 C256 客户端于 21:28 UTC 启动，21:32 开始预热。Router 在预热开始时记录了
8 次 Prefill POST 发送失败和 158 次 Decode open retry；Prefill 请求没有成功完成，
相应 Decode 请求等不到 KV，随后出现 `KVPoll.WaitingForInput` 的 1800 秒超时。
截至停止前，客户端返回 280/285 个初始预热请求，其中 3 条结果没有有效内容。
尚未进入后续 2560 次预热压力请求，也未进入正式测量。

22:06 UTC 人工停止该客户端（benchmark exit=137），归档全部证据，22:07 按原参数单独重试 C256。
没有重启 Prefill、Decode、Router 或 etcd；失败尝试不进入 `results.csv`。
证据：`.tmp/validation/c256-warmup-failure/`，以及本轮 `results/failed-attempts/`
与 `raw/failed-attempts/`（均位于套件 `.tmp/`）。
重试的初始预热再次记录了 10 次 Prefill POST 失败。参考归档也记录过慢预热，C256
最终用时约 3 小时 10 分钟；因此前一次人工中止不能证明该档无法自行完成。
参考：[notes.md，第 7–8 节](../../../yihou/glm52.p8d8.agentx-sweep.packup_20260920/notes.md)。

第二次初始预热在 22:28 UTC 前完成 275 个正常请求，两个 Prefill 的队列均为空。
余下 10 个中，2 个在 1800 秒后自然超时；22:43 UTC（距预热开始超过 32 分钟）仍有 8 个等待。
当时尚未进入压力预热或 profiling，因此进行了**人工预热恢复**：短暂停住 CPU 客户端的
`timing_manager`，再次确认阶段与部署身份，然后对本轮 Decode 调用 `/abort_request`，
释放剩余初始请求，立即恢复客户端派发。没有取消正式测量请求，没有重启任何模型服务。

22:43:38 UTC 客户端进入后续压力预热，日志明确要求每个 lane 再执行 10 个请求。
此次恢复属于 C256 的测量条件，比较结果时应保留这一差异；不能将它写成无干预完成的预热，
也不能据此宣称已修复底层 HTTP 发送问题。
详细操作记录：`.tmp/validation/c256-warmup-recovery.json`；恢复脚本位于
`.tmp/recover_warmup.py`，含阶段、等待时长、容器身份和 Prefill 空队列检查。

压力预热开始时又出现 10 次 Prefill POST 发送失败。23:00 UTC 复查时，客户端已发送
2775 个预热请求，Prefill 合计完成 2755 个，差额正好等于累计 20 次发送失败；
两个 Prefill 的运行、排队、bootstrap 和 transfer 队列均为空。客户端剩余 10 个等待
全部属于这一批失败发送，仍有 70 个后续预热请求尚未派发。

因此再次短暂停住 CPU 派发，释放这 10 个失败等待后恢复，由客户端继续补齐余下预热。
这次没有等待完整的 1800 秒，因为已核对所有有效 Prefill 工作结束且差额与失败数一致。
证据为 `.tmp/validation/c256-pressure-recovery.json`，脚本为 `.tmp/recover_pressure.py`。
两次人工取消都只发生在 warmup，不能将本轮 C256 视为无人工恢复的默认流程。

C256 最终在 23:06:39 UTC 完成全部 2845 个预热请求（285 个 snapshot primers 加
256×10 个压力请求），预热耗时 3347.42 秒。随后于 23:06:39.776 UTC 启动
独立的 3600 秒正式发送窗口。客户端 warmup `cancelled=0` 表示没有由 AIPerf 取消 credit；
服务端人工 abort 仍会作为返回记录进入错误统计，二者不是同一计数。

目前直接证据指向 Router→worker HTTP 请求失败及其后续 KV 等待；发送失败的底层原因
尚未闭合，不能将它归因于客户端 credit 丢失或 RDMA rail 故障。

## C256 正式窗口与结束边界

正式发送窗口从 2026-09-21 23:06:39.776 UTC 到 2026-09-22 00:06:39.778 UTC，
持续 3600.002 秒，期间没有人工取消或暂停。发送 10723 个请求，30 秒 grace period
结束时完成 10513 个，剩余 210 个由 AIPerf 自动取消（约占发送量 1.96%）。
随后等待取消 credit 返回的 10 秒超时，AIPerf 强制结束 phase；这发生在完整测量窗口之后，
日志保留 `grace_period_timeout=True` 和取消计数。尾部未完成请求未计入已完成请求延迟，
因此不能把导出的延迟分位数当作全部已发送请求的无截断延迟分布。

正式阶段及其结束排空日志（采集截至 00:07:23 UTC）中，Router 记录了 44 次 Decode open retry；
Prefill POST 失败为 0，KV 等待超时为 0，rail/GPU/OOM/Fatal Python 计数为 0。
Decode 的 156 条 `Aborted by AbortReq` 日志来自结束后的自动取消，与 210 个
客户端 credit 属于不同计数层级。证据：`.tmp/validation/c256-formal-transport/`。

## 服务端指标与错误审计

AIPerf 在导出 server metrics 时出现 counter-reset 警告。独立检查显示容器 ID、
启动时间保持不变，RestartCount 为零，模型没有重新初始化；因此该警告不代表服务重启。
其采样/聚合原因尚未定位。主要吞吐和延迟来自客户端请求记录，后端 counter 派生指标
应单独解读；原始导出和日志保存在 `.tmp/raw/<RUN_ID>/cNNN/`。

每档前后检查的 rail retry/WQE、GPU memory fault、Fatal Python、OOM 计数均为零。
这些审计不覆盖所有 HTTP、KV 等待或应用层错误，不能据此声称测试完全无错误。
AIPerf 清理 NFS 临时目录时还可能出现 `Directory not empty`；已完成档位的客户端和
结果验证均退出 0，原始日志保留在 `.tmp/`。

## 初始化与可复现性修复

首轮 `2p1d-20260921T140351Z` 在 AIPerf 初始化时因 Unix socket 路径超过 107 字节失败，
没有产生性能测量。客户端临时目录改为容器内短路径 `/ax-tmp`，实际文件仍写入 `.tmp/`；
Python 安装目录也移入持久缓存，避免虚拟环境指向已删除容器里的解释器。
随后清理旧服务、等待 GPU 内存释放，使用已通过的 44 项 Mooncake WRITE 检查启动本轮。

在 C112 开始前修复了冻结配置覆盖每档 `CONC` 的问题：冻结值改为默认值，允许 driver
传入该档并发。C80 已启动进程不受影响；后续各档的 `runtime.env` 均核对了实际并发。
修复未改变引擎参数，证据见本轮 `.tmp/results/` 中的 `point-control-fix.json`。

## 最终验证与清理

独立核对五档原始 JSONL、聚合 JSON、正式阶段起止时间和部署审计均通过，证据为
`.tmp/validation/final-results-check.json`。每档发送窗口实测为 3600.001–3600.002 秒，
所有 worker/Router/etcd 的容器 ID、启动时间和镜像均与部署快照一致。

2026-09-22 00:21:03 UTC，清理脚本退出 0；三台节点均已无本轮前缀的容器。
00:22 检查时，136 八卡显存占用均为 0%；137/138 仍显示 HiCache 释放期间的残留显存
（约 63–90%），尚不能立即作为空闲节点复用。停止容器与驱动释放全部显存是两个阶段，
下一次运行仍应先执行空闲检查。证据为 `.tmp/validation/final-node-state.json`。
