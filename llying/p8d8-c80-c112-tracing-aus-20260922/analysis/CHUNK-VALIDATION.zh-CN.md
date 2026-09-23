# Prefill chunk 4K/8K 验证执行记录

2026-09-23：用户要求先 push 现有工作，再用当前两台机器验证 chunk_size。

## 已完成与阻塞

- Git push 成功：`80091f1a` 已推送到 `origin/llying/p8d8-tracing-aus-20260922`（远端由 `0bc919ae` 更新）。默认开发分支 push 因远端有新提交被拒绝，未强推，沿用既有实验分支。
- 尚未启动新实验、修改远端服务或产生优化结果。
- 本会话 `squeue` 返回 `Error creating slurm stream socket: Operation not permitted`。
- 两台历史节点 `smci355-ccs-aus-n01-33`、`smci355-ccs-aus-n02-21` 的 SSH 均返回 `socket: Operation not permitted`。SSH 使用 `-F /dev/null` 排除系统 SSH 配置权限错误后，仍然如此。
- 共享盘 `ALLOCATION_READY.json` 是 04:51:34 UTC 的旧记录，包含抢占/结束时间；不能据此否定用户所述当前有两台机器，也不能据此确定当前分配。恢复连接后先查询当前作业和节点。

## 实验顺序

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

当前下一项动作：恢复本会话对计算节点和 Slurm 的正常访问，确认当前两节点分配，然后部署上述独立对照。
