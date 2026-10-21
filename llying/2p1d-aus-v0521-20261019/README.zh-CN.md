# GLM-5.2-MXFP4 2P(TP4)1D(TP8) on AUS，sglang v0.5.21 + Mooncake 跨 rank 修复，AgentX 吞吐

目标：在 AUS 两台 MI355X 上重建 3P1D 实验里的 2P1D 形态（t5b-2p1d：C176 = 28,900 tok/s/GPU，见
`../3p1d-tp4-tp8-v0521-20261012/RESULT.zh-CN.md` §3、§4.11），并在此基础上推高吞吐。decode 固定 TP8 + DP8 attention，每卡吞吐按 16 张 GPU 计。

- 结果与分析：[`RESULT.zh-CN.md`](RESULT.zh-CN.md)（§0 摘要与进度图，§1 实验列表，§2 每轮分析，§3 数据表，§4 决策记录）
- 过程时间线：[`operations/STATUS.zh-CN.md`](operations/STATUS.zh-CN.md)
- 目前最高：**a1-spf C176 = 35,377 tok/s/GPU**（write_back + prefill shortest-prefill-first），为 t5b 的 122%、3P1D 20 卡最高（31,829）的 111%。

## 部署

| 实例 | 节点 | GPU |
|---|---|---|
| prefill-0（TP4/DP4，PID 偏移 30000） | 分配到的第一台（P） | 0–3 |
| prefill-1（TP4/DP4，PID 偏移 60000，同节点第二个 prefill） | P | 4–7 |
| decode-0（TP8/DP8） | 分配到的第二台（D） | 0–7 |
| etcd / router / AgentX 客户端 | P | — |

Slurm 作业为 QOS batch、可被抢占且自动重新排队（`Requeue=1`），每次重新开始节点可能不同：34389 = n02-25 + n02-29；34439 第一次 =
n02-29 + n03-33，restart 1（ROUND=a1）= n01-29 + n02-33。节点名、IP 由 `scripts/set_nodes.sh` 写进 `config/topology.2p1d.aus.tsv` 与配置。

## 运行方式

- 运行根目录 `/perf_apps/liyingli/bench_agentx/2p1d-aus-v0521-20261019`（docker 不能 bind mount NFS home），节点本地写目录 `/scratch/liyingli_2p1d`。
  脚本与 harness 来自 3P1D（本身是 AUS 1P1D harness 加同节点多实例支持），配置文件名沿用 `config/config.3p1d.sh`（AUS 版）。
- **ssh 替身** `tools/sshshim/ssh`：每条 `ssh host cmd` 变成 `srun --jobid=$SHIM_JOB --overlap --gres=gpu:8 -w host`（作业的 extern step 结束后
  pam_slurm_adopt 会拒绝 ssh）。step 结束会杀掉其进程，所以长期进程都在登录节点上：
  - `scripts/campaign.sh`：作业每次（重新）开始 → ROUND = a<重启次数> → `scripts/prepare_alloc.sh`（节点名、镜像复用 / 拷贝 / 重建、清理旧容器、
    chown、数据集缓存、16 卡 stream 检查）→ `config/round.<ROUND>.sh` ← `operations/current_stage` → 队列 ← `operations/queue.next.tsv` 模板 → 启动；
    作业离开 RUNNING 时停掉本轮进程并告警。作业运行期间重启 campaign 用 `RESUME_ROUND=<当前轮>`。
  - `scripts/guard_containers.sh`（campaign 在每台节点上作为 srun step 启动）：抢占 / 作业结束时 Slurm 给 step 发 SIGTERM，守护删除本节点
    所有 `glm52-llying-2p1d-*` 容器（否则容器会留在已分给别人的节点上占 GPU）。**不要手动杀守护的 srun 进程**——那同样会删掉实验容器。
  - `scripts/tune_queue.sh`（队列驱动，`start_queue.sh`）、`sampler.py`、`hang_watch.py`；日志跟随 `scripts/follow_supervisor.sh` +
    `bench-harness/follow_remote.sh`（`engine.sh` 写 `<log>.follow` 标记，`FOLLOW_ON_NODE=0`）。
  - 巡检 `scripts/patrol.sh`（每 2 分钟：作业状态、队列失败 / 空转、引擎健康、日志致命模式、传输失败、客户端停滞 → `runtime/patrol/ALERT.log`），
    `scripts/wait_event.sh` 等待告警或队列推进。
- 起栈 `scripts/bringup_fast.sh`（decode 直接模拟接受 3.61；已知答案门禁在第一次分配上通过，最终配置的门禁按用户决定在所有优化后统一跑）。
- 镜像 `build/build_images.sh`：`lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008`（digest `sha256:6c2c0b0e…`）→ yihou v0.5.21 mc4340 recipe
  （Infera `68f06e29`）→ Mooncake #4340 移植 + dest-local-rail + IB timeout + dest pinned HCA；标签 `infera-sglang:v0521-llying-destpin-aus1019`，
  每次在新节点上重建 ID 会变（`EXPECTED_IMAGE_ID` 由 prepare 更新）。Docker Hub 凭据已过期，用空 `DOCKER_CONFIG` 匿名拉取。

## AUS 上必须的设置（与 crsuse2 的 3P1D 不同）

| 设置 | 原因 |
|---|---|
| `MOONCAKE_DISABLE_HIP_DMABUF=1` | 为 0 时起栈即段错误（RESULT §2.1）；AUS 历来用 1 |
| prefill `ROCM_QUICK_REDUCE_QUANTIZATION=NONE` | prefill 挂起的直接证据：同组 rank 分别卡在 QuickReduce 与 RCCL（RESULT §2.4） |
| prefill `HSA_NO_SCRATCH_RECLAIM=1` | 两个 prefill 都设（对挂起无效，保留） |
| `INFERA_NODEPORT_RANGE=24500-32767` | 覆盖固定端口与 AUS NodePort 段 |
| HiCache 每 rank 250 GB | 两个 prefill 共用一台 3 TB 机器 |

## 配置阶段（`config/`）

| 文件 | 内容 |
|---|---|
| `stage-b1-wt.sh` / `stage-b1-wt-qr0.sh` | write_through 基线（后者关 QuickReduce） |
| `stage-b2-wb.sh` | + HiCache write_back |
| `stage-b4-spf.sh` | + prefill `--schedule-policy shortest-prefill-first`（目前最优；作业重新开始后从这里续跑） |
| `stage-b3-pcwb.sh` / `stage-b5-pcwb-spf.sh` | + `--disaggregation-decode-allocation-policy prefill_complete`（挂 `mounts.pcwb.tsv` 的 v0.5.21 校验修复） |

## 分析脚本（`scripts/`）

`tune_report.py`（生成 RESULT §3 的表）、`sample_window.py`（TTFT 分段）、`prefill_rank_analysis.py`（每 rank 负载、长请求独占、逐秒不均衡）、
`decode_pace2.py`（decode 每步耗时 vs 上下文，冷启动分析）、`start_profile_round.sh` + `trace_breakdown.py`（轮次中抓 decode profile 并分解）、
`gpu_stream_check.sh`、`hicache_check.py`（HiCache 正确性）。进度图 `figures/make_progress_svg.py`。
