# 同节点 4K C80 可复用基线

**已完成并复核**：9,782 条正式请求全部 P/D 配对；基线清单检查通过。Total tokens/s/GPU=20,602.16，输出吞吐=2,608.56 token/s。同节点 8K 分别低 4.47% 和 2.74%。结果、采样缺口和取消/错误说明见 [复核报告](final/REVIEW.zh-CN.md)，以下保留执行过程。

用户于 2026-09-23 授权：先提交现有工作到 GitHub，再在当前两节点回测 4K C80，形成后续第 2～6 项使用的基线，同时进行其他准备。

当前工作提交：`b2a0c41b` 已推送至 `origin/llying/p8d8-tracing-aus-20260922`。

- 分配：job 31625，Prefill n03-33（10.235.192.56），Decode n02-21（10.235.192.128）。
- 根目录：`/perf_apps/liyingli/bench_agentx/p8d8-baseline4k-31625-20260923`。
- Run ID：`baseline4k-c80-31625`。
- 启动 driver PID：1320253；自动分析 PID：1367481。
- 正式配置：P8D8，P/D actual chunk 4096/4096，P/D memory fraction 0.85/0.85，HiCache ratio 1.5，kernel/page_first，acceptance 3.61，C80，10 warmup requests/lane，3600 秒正式发送窗口。
- 与同节点 8K 比较时只改变 Prefill chunk；模型、镜像、诊断与客户端 correlation patch、数据集 revision 及 dataset pinning 保持一致。

## 退出与释放观测

**资源门槛已于 12:34:18 UTC 通过**：16 张 GPU 均约 0.10% 显存占用，host/进程检查全部通过。距 12:04:34 停止请求共 1784 秒（29 分 44 秒）；其中资源 gate 记录窗口为 1674 秒。4K 于 12:34:20 进入 LAUNCHING。

8K 完整结果和诊断已回收。12:04:34 UTC 对本实验 P/D 容器发出停止请求；D 于 12:04:48 返回，P 于 12:05:01 返回。容器退出不代表 GPU 资源已释放。随后 P 的共享内存已大幅下降，但 GPU 仍有约 60% 残余占用，ROCm 未显示活跃 KFD PID。

启动 gate 每 15 秒记录前一容器退出状态、逐 GPU 显存/busy、host Shmem 和 MemAvailable。要求 8 GPU 各自显存低于 2%、busy≤5%，大块共享内存消退且 host 可用容量足够，再启动新服务。数据保存在 run 的 `snapshot/resource-release.jsonl`，通过后生成 `resource-release.passed.json`。保持原开发容器，不执行 GPU reset。

## 基线采集及完成标志

12:38:41 配置验证通过，P/D chunk=4096/4096、GPU token 容量=3143424/3003264、host tokens=4715200/rank。8/8 smoke 和采样门槛通过；12:39:47 启动客户端，12:42:15 开始目标 884 请求的 warmup。

1. 验证分配、资源释放、镜像和模型路径。
2. 新服务启动后核对完整 server-info 与容器命令/环境；仅接受计划中的 P chunk 和运行标识/动态端口差异。
3. 执行相同 8 请求 smoke，验证 P/D RID/room/trace；flush 后开始 fresh samplers。
4. 核对 runtime.env、固定 dataset revision、工作量摘要，然后执行相同 warmup 与完整 C80。
5. 保存原始 export、diagnostics、OTLP、engine/node 采样，区分 warmup/profiling、错误/取消与未导出请求。
6. 自动生成 `analysis/baseline-manifest.json`、阶段/资源/span/chunk 汇总、`analysis/8k-vs-4k/COMPARISON.zh-CN.md` 和 `analysis/4k-baseline-drift/COMPARISON.zh-CN.md`。

基线完成只代表本次采集满足既定检查，不代表已获得运行间方差。后续候选必须使用同样缓存起点；显著收益仍需重复或相邻对照。

## 后续准备

`preparation/PLAN.zh-CN.md` 和 `cases.json` 保存第 2～6 项的参数、预言、否定条件与前置检查。其中 2 的 host 容量精确锁定、3b 的 backend/layout 兼容性、5/6 的新插桩和完整 admission 信息仍需专项验证；不因有了基础 4K 结果就假定这些条件已满足。

## 显存释放慢的内核证据

约 12:30 UTC，对活跃回收线程 kworker/26:2+events 做了 3 秒、99 Hz CPU perf 采样，298 个样本、无丢失。98.64% 样本在 `list_insert_sorted.isra.0`；调用链是 `__drm_buddy_free → amdgpu_vram_mgr_del → ttm_bo_release → amdgpu_vm_fini → drm_release → delayed_fput`。采样发生在旧服务退出后、4K 启动前，不计入性能窗口。

已安装驱动 `/usr/src/amdgpu-6.14.14-2212064.22.04/drm_buddy.c` 的该函数遍历同阶 free list，按 block offset 顺序插入。大量块释放时反复线性遍历可能带来高开销；本次采样证明回收线程的主要 CPU 时间在该路径，但没有测量 free-list 长度，不能据此给出剩余时间或断言全部退出延迟都来自同一原因。

文本证据见 release-perf-report.txt、drm-buddy-insert-source.txt、driver-release-evidence.txt；原始 perf.data 保存在 8K run 的 shutdown/release-perf.data。当前不修改驱动、不重置 GPU，继续以实际显存释放状态作为启动门槛。

13:04:52 UTC warmup 完整结束，Runner completed=884/cancelled=0/errors=0，耗时 1357.27 秒；同秒进入 profiling，正式发送窗口计划至 14:04:52 UTC。导出错误归属在最终分析另行核对。
