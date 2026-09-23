# Prefill chunk 4K/8K 验证执行记录

## 最新结果：同节点 4K 基线完成

2026-09-23，job 31625 上完成 4K C80，14:14:02 UTC 全部分析完成。Total tokens/s/GPU=20602.16，输出=2608.56 token/s；同节点 8K 分别低 4.47%/2.74%，TTFT p50 高 25.19%。9,782 条正式请求全配对；收尾取消 11，warmup 导出错误 4；资源抓取存在少量超时，已记录为缺失。完整可复用基线、时间线及复核见 [baseline4k-31625/final/REVIEW.zh-CN.md](baseline4k-31625/final/REVIEW.zh-CN.md)。下文 8K 状态作为历史记录保留。

## 当前执行：job 31625（2026-09-23）

**已完成并复核**：11:33:20 UTC 自动分析结束。输出吞吐 -1.22%，TTFT p50 +27.39%；本次未观察到 8K 收益。22 个共同 context/miss/host 分组的统一加权 forward envelope +48.48%。两条导出错误均属于 warmup，正式阶段 9584 条记录全部 P/D 关联，收尾取消 13 条。复核报告见 `chunk8k-31625/final/REVIEW.zh-CN.md`；下面的运行中状态保留为执行历史。

已完成服务配置验收、8/8 smoke 请求及 P/D trace 关联、采样新鲜度验收。实际 P chunk=8192、D chunk=4096，显存派生容量与旧 4K 一致（P 3143424、D 3003264），容器命令（除 P chunk/地址）和环境一致。动态 KV 事件端口差异已记录并由检查器按端口规则验证。

首次客户端配置因全局 HF 离线模式误解析本地 tokenizer 而失败，尚未发出 warmup 请求；证据保留于 `recovery/client-offline-failure/`。修复为仅在数据集预下载阶段启用离线读取，client overlay 的 Weka loader 显式指定旧 revision；原 correlation patch 字节不变。修复后 tokenizer 加载成功，393/393 轨迹读取与重建完成。当前有效客户端于 10:01:02 UTC 启动；10:03:53 UTC 开始 warmup，目标 884 请求（84 个初始快照请求 + 每 lane 10 个后续请求），与旧基线一致。首分钟 returned=21/884、errors=0，日志确认实际 Prefill batch #new-token=8192。Warmup 后自动进入 3600 秒正式发送窗口；当前尚无正式性能结论。

当前 driver 为 `client-recovery.pid`（PID 281045），日志 `client-recovery.log`；原 `driver.log` 与 `smoke-recovery.log` 保留历史失败和恢复记录，当前状态以 run 下 `STATUS` 和 `c80/runner.log` 为准。自动分析进程 PID 293945，日志 `finish-analysis.log`：成功完成后生成 `analysis/CHUNK8K-COMPARISON.zh-CN.md`、请求关联、资源和 trace 汇总。分析完成仅代表自动汇总完成，收益判断仍需结合分层、工作量和错误情况。

启动时从同镜像、同硬件的 Decode 节点复用了四个缺失 JIT 模块；没有替换正在编译或已存在的模块。模块哈希见 `jit-cache-reuse.json`。

用户重新分配 `smci355-ccs-aus-n02-21` 与 `smci355-ccs-aus-n03-33` 后要求继续。已确认 job 31625 为 RUNNING、QOS=batch、PreemptTime=None，结束时间为 2026-09-24 09:01:42。Prefill 使用 n03-33（10.235.192.56），Decode 保留旧基线节点 n02-21（10.235.192.128）。两节点均通过 SSH、16 GPU 空闲和服务端口检查；已有 `dev_primus_mxfp6_265` 容器仅运行 sleep，不修改或停止它们。两节点内核、ionic 驱动/固件和 libionic.so 哈希一致，证据见 `chunk8k-31625/`。

新根目录 `/perf_apps/liyingli/bench_agentx/p8d8-chunk8k-31625-20260923`，run ID `chunk8k-c80-31625`，Prefill 后台 driver PID `4181346`。两端诊断镜像 SHA256 已核验一致；09:35:32 UTC 进入 LAUNCHING，开始服务初始化。配置保持 P8D8、C80、P chunk 8192 / D chunk 4096、3600 秒、10 warmup requests/lane、acceptance 3.61。启动前、服务就绪后和发压前均检查 Slurm 分配及抢占标记。历史 job 31609 的目录和证据保留。

## 历史状态：启动期间被抢占（2026-09-23，job 31609）

08:40:02 UTC，Slurm 将 job 31609 标记为抢占，EndTime 从次日 07:51:42 缩短为当日 08:45:02。查询时 JobState 仍为 RUNNING（抢占宽限期），但两台节点的 SSH 均返回 `Access denied by pam_slurm_adopt: you have no active jobs on this node`。最后 driver 状态为 LAUNCHING；两端权重分片读取完成，尚未完成服务健康检查、smoke 或正式 C80 发压，因此没有新的性能验证结果。

共享盘启动日志、镜像 ID、GPU 空闲检查、原分配与抢占后分配已回收至 `chunk8k-31609/`。无法 SSH 后未确认远端 driver/容器清理完成；不能把共享盘的旧 LAUNCHING 状态视为当前仍正常运行。恢复有效分配后，应先核查本实验容器和 driver，再用新的 run ID 启动，避免覆盖此次启动证据。

### 抢占前执行记录

网络权限恢复。当前控制会话在登录节点 `dccs-1334-slurm`；SSH 已确认 Prefill `smci355-ccs-aus-n02-25`（10.235.192.129）和 Decode `smci355-ccs-aus-n04-29`（10.235.192.57）可访问。Slurm 确认这两台属于 liyingli 的运行中 job 31609，结束时间为 2026-09-24 07:51:42；启动前 16 张 GPU 均通过空闲检查。

独立运行根目录：`/perf_apps/liyingli/bench_agentx/p8d8-chunk8k-31609-20260923`；run ID：`chunk8k-c80-31609`。Prefill 控制节点后台 driver PID `643887`，日志 `driver.log`。两节点已导入 63 GB 旧诊断镜像并校验 SHA256；08:33:27 UTC 进入 LAUNCHING，P/D 容器已启动，正在加载模型。服务就绪后自动执行配置验收、smoke、采样验收及 C80。尚未进入正式负载。

只运行 8K C80，D chunk 保持 4K；正式时长 3600 秒，warmup 10 requests/lane，simulated acceptance 3.61。沿用旧诊断补丁和 client patch，InferenceX commit `918524ff94045b3f091115f1051c22a8588edf2b`。数据集 revision 固定为旧运行的 `23f152f6f0f9399a85901b89a6458def0ef16729`，客户端使用已有 HF 缓存并开启离线读取。完整服务配置、容器命令/环境与 runtime.env 差异均在负载前检查并保存；非预期差异会中止流程。

两节点 host RDMA 版本不同：P ionic 26.03.3.001 / firmware 1.117.5-a-77，D ionic 26.07.9.001 / firmware 1.117.5-a-147；各自保留匹配的 host libionic.so，SHA256 及内核版本见 `chunk8k-31609/*-host.txt`。这属于节点迁移的已知比较限制，不更换驱动或固件。

新节点没有该镜像对应的 AITER JIT cache，启动时会重新编译；正式比较仍须核对 warmup、工作量和服务时间。节点迁移、随机服务 seed、动态端口和显存派生容量会记录，不能将单次历史对比视为严格同节点 A/B。

## 历史执行指令与网络阻塞记录（已解除）

2026-09-23 resume 重试：`squeue -j 31609` 返回 `Unable to contact slurm controller (connect failure)`；`scontrol show job 31609` 在 15 秒内未返回，超时终止；历史节点 `smci355-ccs-aus-n01-33` 的 SSH 返回 `Temporary failure in name resolution`。当前仍无法确认分配或连接节点，未启动新实验。以下 socket 权限错误为此前会话的观测。

用户确认 job `31609` 已分配节点，要求先执行 C80 8K，只有性能存在疑点时才在同两节点回测 4K。镜像与其他配置保持旧 4K C80 一致。已从旧运行的 `runtime.env` 和 image-id 核实镜像、InferenceX commit、并发、时长、warmup 和 acceptance，保存为 `chunk8k-c80-31609.expected.json`。该文件是预期配置，不是新实验结果或已验证的远端配置。

当前会话再次查询 Slurm 仍报 socket 权限错误，SSH 也仍报 `Operation not permitted`；尚未获知当前节点身份、未部署或启动新负载。必须解决当前会话的网络访问限制后才能执行。下文 A/B/A 仅保留为历史方案，不是当前启动要求。

2026-09-23：用户要求先 push 现有工作，再用当前两台机器验证 chunk_size。

## 已完成与阻塞

- Git push 成功：`80091f1a` 已推送到 `origin/llying/p8d8-tracing-aus-20260922`（远端由 `0bc919ae` 更新）。默认开发分支 push 因远端有新提交被拒绝，未强推，沿用既有实验分支。
- 尚未启动新实验、修改远端服务或产生优化结果。
- 本会话 `squeue` 返回 `Error creating slurm stream socket: Operation not permitted`。
- 两台历史节点 `smci355-ccs-aus-n01-33`、`smci355-ccs-aus-n02-21` 的 SSH 均返回 `socket: Operation not permitted`。SSH 使用 `-F /dev/null` 排除系统 SSH 配置权限错误后，仍然如此。
- 共享盘 `ALLOCATION_READY.json` 是 04:51:34 UTC 的旧记录，包含抢占/结束时间；不能据此否定用户所述当前有两台机器，也不能据此确定当前分配。恢复连接后先查询当前作业和节点。

## 历史 A/B/A 方案（当前不执行）

先 C80 A1(4K) → B(8K) → A2(4K)。每档独立运行目录、重启本实验的 P/D 服务以统一缓存起点，再执行相同 smoke、warmup 和正式采样流程。禁止沿用旧 C80 的缓存状态直接比较新 B。保留原始记录和服务日志，停止服务前完成回收。只操作已确认属于本实验的容器，不执行 GPU reset。

| 配置 | A1/A2 | B |
|---|---:|---:|
| `PREFILL_CHUNK_SIZE`（命令行） | 32768 | 65536 |
| Prefill 实际 `chunked_prefill_size` | 4096 | 8192 |
| Decode chunk 命令行值 | 32768 | 32768 |
| 并发 | 80 | 80 |
| warmup | 10 requests/lane | 10 requests/lane |
| 正式发送窗口 | 3600 秒 | 3600 秒 |

保留同一诊断镜像、模型、trace 数据及顺序、router affinity/overlap、memory fraction、host cache、simulated acceptance 和 tracing 设置。三个阶段加 warmup、drain、启动可能超过五小时；启动前确认当前分配剩余时间。时间不足时可做等长短试筛选，但必须标记为筛选，不能视为正式收益验证。

配置加载顺序：`config/config.sh` source 历史 phase-c 配置；chunk override 必须在 source 之后赋值，且 launch/engine 重复 source 时一致生效。不能只在调用 shell 临时设置变量就认定参数改变。原 `scripts/run.sh` 固定执行 C80→C112，不能直接作为 A/B/A runner 使用。

## 每档负载开始前的门禁

1. 记录当前 Slurm 分配、两节点 GPU/容器状态；检查模型、镜像 SHA256、内核 cache 和共享目录。
2. 对比有效配置，除 P chunk 和运行标识外应一致；在 `server-info` 与启动日志核实 P 的 4096/8192，D 保持原值。
3. 完成现有 smoke RID/P-D 关联验收，以及采样新鲜度和采样器存活检查。
4. 保存 workload 文件摘要、warmup 配置、有效配置与镜像 ID；禁止覆盖 `main-20260922`。

## 判定与后续

- 主要比较：完成吞吐、P queue、P forward envelope、TTFT p50/p90/p99，并按 miss/context 分层比较 chunk 数与服务耗时。
- 记录输入/miss/host/output 实际工作量和缓存命中率；闭环轨迹进度可能不同，不能只比较 headline token/s。
- 检查短 miss 请求尾延迟是否恶化、错误/取消/retraction 是否增加，以及 D 等待上游期间的 KV 驻留是否下降。
- B 的收益必须超过 A1/A2 的自然波动，并有服务耗时下降的机制证据。chunk 数减少但 forward 不降，或短请求尾延迟恶化，均限制或否定该优化。
- A/B/A 是初始验证，不能以三个点冒充统计置信区间；必要时重复。C80 有效后再做 C112 配对验证；尚无依据承诺吞吐提升比例。

历史下一项动作已完成：当前两节点分配和网络访问已确认，执行状态见文首。
