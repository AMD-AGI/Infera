# 交接：B 线（GLM-5.2 P4D4 prefill 侧：GEM + offload 故障、vLLM 多级 offload 在 PP 下起不来）

2026-11-18 04:20 起由新 session 在 v3-258（作业 294784）上做。主 session 继续在 v3-135（294199）上做 P4D4 计分（API server 瓶颈等），
A 线（decode kernel / 通信）在 v3-240（295840）上由另一个 session 做。三边不共用工作目录。

## 1. 节点与执行方式

- v3-258 由作业 294784（`hold_n2.sbatch`）持有，**作业的批处理脚本里常驻一个 runner**，执行 `/home/liyingli/bench_agentx/vllm-pd-20261117/queue-n2/`
  下的 `*.sh`。从登录节点投递：`/home/liyingli/bench_agentx/vllm-pd-20261117/q2.sh [-w] <名字> '<命令>'`（名字以 `L` 开头的是长任务）。
  - 该队列从现在起归 B 线专用，主 session 不再往里投递。
  - 命令默认在主 session 的工作目录里执行：**每条命令先 `cd` 到自己的工作目录**（例如 `/home/liyingli/bench_agentx/vllm-trackB-20261118/`），
    输出只写到自己的目录。
  - **不要 `touch queue-n2/STOP`、不要结束 runner**：runner 一退出，批处理脚本结束，作业和节点就没了。作业还剩约 10.5 h（到 11-18 15:00 前后）。
  - spur 约束：每个节点只允许一个进入作业的会话；任何作业步骤非零退出都会结束整个作业（runner 已处理，命令里不要自己 `srun`）。
- 不要碰：主 session 工作目录里除 `queue-n2/` 以外的东西（尤其 `queue/`，v3-135 的 runner 会执行那里的任何 `*.sh`）；v3-135、v3-240 上的容器。
- 节点准备（主 session 已投递，队列项 `queue-n2/1118-040712-Lprep-n2`）：在本地 NVMe 上从提交 `0c5e5e89` 构建镜像 `infera-vllm:v0340-0c5e5e89`
  （日志 `logs/build-n2-0c5e5e89.log`），并拷贝模型到 `/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4`（`logs/stage-n2.log`），
  预计 04:45 前后完成；`Lprep-n2.rc` 出现后用 `docker images` 确认。
- 工具：`../vllm-pd-tools-20261117/`（README 说明每个脚本），主 session 工作目录 `scripts/` 是最新版本；复制到自己的目录后改路径（脚本里 `R=` 指主 session 的工作目录）。
  不要注入宿主机的 libionic。

## 2. 当前基线（P4D4，单节点：prefill TP1×PP4 GPU 0–3，decode DP4 GPU 4–7）

- 最好成绩 s7b = 30,091 tok/s/GPU（C104，1800 s；sglang 53,351）：Python 串行路由 + 缓存感知准入 128 / 0 + 1.25 TB CPU 层（LRU）+ offload 描述符上限 512
  + decode 单 API server + 路由 `--ulimit nofile`。配置见 `queue/s8.sh`（s8 = s7b + 4 个 API server，正在 v3-135 上跑）与 README 1h。
- prefill 目前**不用 GEM**、显存目标 0.6（GPU KV 约 460 万 token），因为 GEM + OffloadingConnector 会故障；sglang 的 prefill 用 GEM、0.88（约 1,200 万 token）。
- 1800 s 测量里命中率已接近理论上限（现算约 5%），但 sglang 的成绩是 3600 s，工作集随时间增长，1.25 TB 可能不够；CPU 层再大会主机 OOM（README 0、U10）。

## 3. 任务

### B1 GEM KV + OffloadingConnector：第一次 offload 写出 GPU 访存故障（U2）

- 已知事实见 README 1b。复现（聚合 PP4，GPU 4–7，无 PD）：`scripts/repro_gem_offload.sh`（原队列项，MultiConnector[GEM Mooncake + 100 GB offload]，
  smoke 时 `HSA_STATUS_ERROR_MEMORY_FAULT`）；独立拷贝测试 `../vllm-pd-tools-20261117/scripts/swap_fault_test.py` 在所有组合下都正常。
- **更正**：ROCm 上 `CPUOffloadingSpec._uses_shared_region()` 为 False，CPU 层是每个 rank 自己的锁页主机张量，不是 `/dev/shm` 共享区；1b 里
  "`/dev/shm` 背后的共享 offload 区"这个嫌疑可以排除，换成"vLLM 在 ROCm 上怎么分配 / 注册这块锁页内存"。
- 仍待排查：GEM 池里的 KV 张量与 offload 拷贝的组合（vLLM 怎样把 KV 张量交给 offload worker，地址 / 大小是否越界）、150 GB 级单个 GEM BO、
  `hipMemcpyBatchAsync` 的 `srcAccessOrder=Any`、拷贝发生的流与 GEM 映射的上下文。
- 目标：prefill 用 GEM、显存目标约 0.88 能稳定跑 P4D4；修法可能在 vLLM（offload）、Infera（`infera/engine/vllm/gem_mooncake.py`）或 GEM 分配器
  （`../vllm-image-v0340-20261117/gem/hip_gem_alloc.c`）。

### B2 vLLM `TieringOffloadingSpec` 在 PP > 1 时起不来（U11），修好后试 NVMe 二级层

- 现象与根因见 README 1e、CANDIDATES U11：共享区域大小来自 `worker_kv_bytes_per_block`（`vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py`，
  本 worker 的 KV 张量大小 / 块数），PP 各 stage 层数与 indexer 层分布不同，各 worker 算出的大小不一致，加入方等创建方文件长到自己的期望值 30 s 超时；
  行内各 worker 的槽位偏移也会不一致。需要所有 worker 与调度器用同一个值（例如各 stage 的最大值）。
- 修好后在 P4D4 上试 FS 二级层（本地 NVMe xfs 卷 `/mnt/m2m_nobackup`，约 22 TB 空闲）：配置见 `queue/s6.sh`（s6），每轮用新目录、跑完删除
  （AgentX 每轮回放同样的 prompt，残留数据会虚高命中）。和 s7b 比，最好做 3600 s（与 sglang 同时长）。
- 覆盖 vLLM 文件的方式：bind mount 单个文件进容器（参考 `../vllm-kf-port-20261117/scripts/run_overlay_test.sh`），不用重建镜像。

## 4. 结果与提交

- 仓库 `/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52`，分支 `dev/pd_opt/vllm_glm52_agentx`；用自己的 `git worktree`。
- 结果放新目录 `llying/vllm-prefill-trackB-20261118/`（README.zh-CN.md + scripts + results）；完成一项就 `git commit -s`、`git pull --rebase` 后 push。
- **不改主 session 维护的共享文件**：`../vllm-pd-progress-20261117/`、`../upstream-vllm-20261117/CANDIDATES.zh-CN.md`、本目录的 `README.zh-CN.md`。
  结论（包括对 U2 / U11 的更新）写在自己的 README 里，由主 session 合并。
