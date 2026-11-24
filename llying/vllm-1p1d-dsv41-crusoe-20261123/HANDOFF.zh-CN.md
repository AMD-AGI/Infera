# 交接说明（2026-11-24 14:00 UTC）

接手的 agent 先读本文件，再读 `README.zh-CN.md`（结论）与 `EXPERIMENT.zh-CN.md`（第 4 节问题时间线与现场分析、第 5 节各轮
实验与瓶颈分析）。用户的要求、约束与当前状态都在这里。

## 1. 目标与用户的要求

- 目标：DeepSeek-V4.1-Flash 1P1D，P、D 各 2 卡（共 4 卡），Infera + vLLM，在 crusoe 集群上推高 AgentX 吞吐。
- 方法（用户认可）：在饱和并发上找瓶颈 → 用现成配置或自己实现去解 → A/B 后比峰值对峰值。
- **当前优先级（用户 09:14 指示）：先解决稳定性**（DP2 死锁 / 整机 GPU 停顿、预热切换时连接被重置），否则各种测试不确定地失败、
  浪费时间。测 c576 / c704 时若再遇到，就按现场信息优先解决这两个问题；没遇到就专门做稳定性。
- 用户希望接手的 agent **自主决策连续运行约 12 小时**。注意：agent 只在一轮回复里持续调用工具时才在工作，回复一结束就停；
  要用 `AwaitShell`（等后台监控的输出）一直等下去，不要提前结束回复。
- 约束：
  - 只用 `amd-hyperloom` / `amd-hyperloom-qos`，每个作业只申请 4 卡（`--gres gpu:4 -c 118 --mem 1200G`，半节点）。
  - 登录节点不跑大内存任务；只做读日志、小脚本分析。需要计算就提交作业。
  - 后台必须有监控，及时发现问题，避免作业空转。
  - 实验报告、数据、可复现脚本要及时提交到 `dev/pd_opt/vllm_dsv41_agentx` 的本目录；提交用 `git commit -s`，消息末尾加
    `Co-authored-by: Cursor <cursoragent@cursor.com>`，不要绕过 hooks。
  - 不要 `srun --overlap` 进正在跑的实验作业（spur 会因作业步异常结束整个作业；R1b 被取消疑似因此）。
  - 不要修改正在被作业执行的脚本（bash 是边读边执行的）；要改就复制成新文件（因此有 pipeline3…7、pipeline_hang）。
  - 同一账户关联同时运行的作业数有上限，用户的另一个任务（`lyl-vpd-*`）会占用作业槽。

## 2. 路径

| 用途 | 路径 |
|---|---|
| 工作目录（脚本、arm、日志、打包） | `/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/`（下称 `$W`） |
| 本仓库目录（报告与数据） | `/home/liyingli/bench_agentx/baseline/Infera-vllm-dsv41/llying/vllm-1p1d-dsv41-crusoe-20261123/`，分支 `dev/pd_opt/vllm_dsv41_agentx` |
| 部署 kit | `/home/liyingli/bench_agentx/baseline/Infera-dsv41-kit/examples/vllm_1p1d_dsv41_flash/`，分支 `llying/example/vllm_1p1d_dsv41_flash`（`240c72a4f` 已推送未合入 dev） |
| AUS 基线数据 | 仓库 `llying/vllm-1p1d-dsv41-20261120/raw/` |
| vLLM 源码（镜像对应 9c1d45c2） | `/home/liyingli/bench_agentx/baseline/src-snapshots/vllm-9c1d45c2/` |
| AIPerf（agentx-harness 1.0.14）源码 | `/tmp/ah/src/aiperf/`（PyPI wheel 解开；/tmp 可能被清，需要时重新下载） |
| 时间线 | `$W/logs/summary.log`（每个 run 每一步一行），各 run 原始日志 `$W/logs/<run>/`，打包 `$W/packed/<run>/` |

## 3. 脚本（都在 `$W`，仓库 `scripts/` 是副本）

- 正式跑分：`run_opt5.sbatch` → `scripts/pipeline7.sh`（节点体检 → 拷模型 / 取镜像 → 起栈 → smoke →（GATE=1 门禁）→ JIT 预热 →
  AgentX → 节点上分析与打包；任一 leg 死锁时保留现场并重新部署一次；AgentX 期间每 10 s 记录 TCP 计数器）。
  `sbatch -J lyl-dsv41-<name> -x "$(paste -sd, exclude_nodes.txt)" --export=ALL,ARM=arms/r4-a4w4p.sh,CONC=640,GATE=0,PROFILE_AT=0,STEPLOG=0,BASELINE=<dir:P,D> run_opt5.sbatch`
- 死锁复现：`run_hang.sbatch` → `scripts/pipeline_hang.sh` + `scripts/hang_repro.py`（空闲 IDLE_S → 1 个长请求 → BURST 个突发，
  循环至 BUDGET_S；死锁时现场移到 `hangN/` 并重新部署；每 10 s 记录 GPU 忙碌度与 compaction / migration 计数器到
  `host-telemetry.txt`；最后一行 `RESULT arm=… cycles=… hangs=…`）。
- 监控：`bash $W/scripts/monitor.sh`（登录节点；打印 summary.log 新行、作业 .out 新行、`JOB ENDED`、`NO PROGRESS`）。
  当前有一个旧进程（pid 2263056）在跑，接手后先 `pkill -f scripts/monitor.sh` 再自己起一个，用 `notify_on_output` 关注
  `JOB ENDED|NO PROGRESS|FAILED|engine died|stall watch|AgentX c[0-9]+ rc|RESULT|hang [0-9]+ in`。
  注意：`AwaitShell` 的 pattern 会匹配到已有的旧输出，要用带时间戳或新 run 名的 pattern。
- 现场采集：`scripts/hang_watch2.sh`（两条 leg 各一个，出现 60 s 停顿行时 py-spy dump 两次 + GPU 状态）。
- 分析：`scripts/steplog_summary.py`、`trace_cat.py`、`compare_runs.py`、`prefill_bigsteps.py`、`rccl_bench.py`（+ `bench_rccl.sbatch`）、
  `jit_warm.py`、`pack_raw.py`。
- 节点排除名单：`$W/exclude_nodes.txt`（体检失败的节点自动加入）。

## 4. 结论摘要

- 当前最佳：`arms/r4-a4w4p.sh`（R1 容量设置 + P 的 MoE 用 FP4 激活 `VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4=1`）：
  **c640 486,737 tok/s/GPU**（crusoe 基线 463,531，+5.0%；AUS 471,698），c768 477,212。c576、c704 尚无有效数据。
- 瓶颈：P 每 token 计算；P GPU 时间 32–38% 是 MoE AllGather + ReduceScatter（单条 xGMI，约 54 GB/s/方向，RCCL 调优无效）；
  前缀命中已贴近理论上限；D 有余量。现成配置基本到头，再往上要改代码（EXPERIMENT 5.4 节）。

## 5. 稳定性问题现状（详见 EXPERIMENT 4.1–4.3）

1. **DP2"死锁" / 整机 GPU 停顿**：今天 crusoe 7 次引擎死亡（decode 4、prefill 3）+ R7b 一次 3 分钟全机停顿（未死）。
   两种现场：GPU 100% 自旋（R5 decode）与 GPU 空闲、等 GPU 的线程都在 `hipEventSynchronize` 忙等（R7 prefill、R8 decode、R7b 全机）。
   R7b 两个独立引擎、四张卡同时停 → 判断在驱动 / KFD 层而非 vLLM。工作假设：几百 GB 注册给 GPU 的主机内存（UMBP 449 GB、Engram，
   userptr 未钉住）遇到内存规整 / 页迁移时 KFD 撤队列并重新校验，耗时可达分钟级。另一个混杂因素：死锁的轮次都带
   `--profiler-config` 与逐步记录器（R7b 关掉后没死，但仍有全机停顿）。
2. **预热切换时连接被重置**：AIPerf 预热进入第二阶段（所有 lane 同时发）后约 1–2 s 内一批请求被对端重置（9 / 1 / 111 / 75 个），
   任一根请求失败 AIPerf 即中止整轮（`aiperf/credit/callback_handler.py`，设计如此，无错峰开关）。被重置的都是刚发出、还在 router
   里排队（kv-aware 选路串行，约 6 ms/个）的请求，且同一毫秒一起被重置。已排除：router 空闲超时（axum 0.7.9 未设 timer）、
   请求体上限（8 MB）、nofile、上游连接池、内核 listen 参数（登录节点 somaxconn 4096）。来源未定位；pipeline7 的 `tcp-counters.txt`
   用于下次定位（看切换前后 ListenOverflows / ListenDrops / TCPAbortOnData / TCPAbortOnClose 等的增量）。

## 6. 正在运行（11-24 19:30 更新）

- 稳定性根因已定位（EXPERIMENT 4.4–4.6）：页迁移导致 KFD 撤队列；`VLLM_ROCM_ENGRAM_HIP_VMM=1`（`arms/r4-hipvmm.sh`）修好了
  Engram 这部分。UMBP 主机池仍受影响，需要管理员预留 hugetlb 大页（或设 `vm.compact_unevictable_allowed=0` 并由我们 mlock）。
- **R9**（作业 309628，v3-178）：`run_opt6.sbatch` → `pipeline8.sh`，`r4-hipvmm` c640，`GATE=1 PROFILE_AT=3300 STEPLOG=1`，
  `BASELINE` = R5。run 名 `pd1p1d-c640-r4-hipvmm-<时间>`。
- 分析：`python3 $W/scripts/slow_steps.py $W/logs/<run>`（停顿与页迁移、KFD 撤队列对齐）；`pipeline8.sh` 结束时自动生成
  `analysis/slow-steps.txt`，summary 里有停顿次数和 UMBP 失效次数。
- 监控：`scripts/monitor2.sh`（`setsid nohup` 启动，输出 `$W/logs/monitor2-<时间>.txt`，`monitor2-current.txt` 指向最新；
  重要事件带 `ALERT#N` 编号）。ssh 断开时前台进程会随会话退出，所以要脱离会话启动。
- 用户的 GPU 利用率 watchdog 会取消连续 45 分钟利用率低于 25% 的作业；旧的 `pipeline_hang.sh` 循环会被它取消，
  要用带负载的 `pipeline_hang2.sh`。

## 7. 建议的下一步

1. 跟 hangB0：一旦出现 `hang N in …`，读 `hangN/` 的 py-spy 与 GPU 状态，并把停顿时刻与 `host-telemetry.txt` 的 compaction /
   pgmigrate 计数器、`gpu_busy_percent` 对齐；同时看 `sysctl` 首行（计算节点的 numa_balancing / compaction_proactiveness）。
   - 对得上 → 整理证据给用户（需要管理员关闭主动规整或用 hugetlb 承载 UMBP；或试 Engram `use_thp`、减小 UMBP 注册量做 A/B）。
   - 对不上 → 用 `pipeline_hang.sh` 做 A/B：`STEPLOG=0 PROFILE_AT=0`、decode `--no-async-scheduling`、`GPU_MAX_HW_QUEUES=16`、
     关 DSpark，按 hangs/cycles 比较。
2. 连接重置：下次出现时读 `tcp-counters.txt`；必要时在 `hang_repro.py` 里加一个"数百个并发长请求同时到 router"的模式离线复现。
3. 稳定后补测峰值（c576 / c704，基线用 `logs/pd1p1d-c640-crusoe-1123T1307/agentx-c640/result/conc_640/aiperf_artifacts:31300,31301`）。
4. 每个结果都写进 `EXPERIMENT.zh-CN.md` / `README.zh-CN.md`、拷数据到 `raw/`、提交推送。
