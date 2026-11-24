# 交接说明（2026-11-24 14:00 UTC；11-25 00:00 更新第 4–7 节）

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

## 4. 结论摘要（11-25 00:00 更新）

- **当前最好**：`arms/r4-hipvmm.sh`（R4 + `VLLM_ROCM_ENGRAM_HIP_VMM=1`）+ router backlog 修复（`pipeline9.sh`）：
  **c704 494,398 tok/s/GPU**（R10b；R5 c640 486,737，crusoe 基线 463,531，AUS c640 471,698）。同配置 c640 465,373（R9b，
  那一轮 UMBP 失效较多）。
- 瓶颈仍是 P 的每 token 计算（MoE AllGather + ReduceScatter 占 P GPU 时间 32–38%）；HIP VMM 让 D 的 ITL 低了 9–12%。
- 之前的 R4 / R5 结论（a4w4 有效、EP2 / RCCL / DBO 无效等）不变，见 EXPERIMENT 第 5 节。

## 5. 稳定性问题现状（11-25 00:00：两个都已定位并修复，详见 EXPERIMENT 4.2 节末尾、4.4–4.8）

1. **GPU 停顿 / "DP2 死锁"**：页迁移（直接规整、khugepaged）碰到注册给 GPU 的主机内存（userptr），KFD 撤下所有相关进程的
   GPU 队列，并重新校验整段内存（Engram 每条 leg 189 GiB，UMBP 418 GiB），要 45–200 s，超过 300 s 就被 vLLM 判死。
   证据：KFD 的 `/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms` 与每次停顿等长（hangC0）。修复：
   `VLLM_ROCM_ENGRAM_HIP_VMM=1`（hangC2 0 次停顿，起栈 25–36 分钟降到 6–10 分钟）。**UMBP 主机池仍会被撤队列**，期间 UMBP
   缓存失效（R9b 损失约 4%），需要管理员：`vm.compact_unevictable_allowed=0` 加我们的 `instr/mlockall.c`，或者预留 hugetlb
   大页加 kit 的 `UMBP_HUGEPAGES=1`；根本修法是 MORI 把 DRAM 层放进 HIP VMM 主机内存。**已经问用户，尚未答复。**
2. **预热切换时的连接重置**：router listen backlog 128 → SYN cookie 失败 → RST（R9：`SyncookiesFailed` +58 = 58 个重置）。
   修复：`instr/listen_backlog.c`（LD_PRELOAD，backlog 4096）+ kit `ROUTER_PRELOAD`（`de90d8127`）+ `pipeline9.sh`。
   R9b（c640）、R10b（c704）预热切换都是 0 错误。根本修法是 Infera router 自己用更大的 backlog。

## 6. 正在运行（11-25 00:00 更新）

- **R12**（作业 309933，v3-006）：`arms/r12-p32k.sh`（r4-hipvmm + P `--max-num-batched-tokens 32768`）c704，`GATE=1`，
  与 R10b 同并发 A/B。
- **R11**（作业 309932，`scontrol hold` 中）：`r4-hipvmm` c768，用 `scontrol release 309932` 放行，用来确定峰值。
- 正式跑分：`sbatch -J lyl-dsv41-<name> -x "$(paste -sd, exclude_nodes.txt)" --export=ALL,ARM=arms/r4-hipvmm.sh,CONC=704,GATE=0,PROFILE_AT=3300,STEPLOG=1,BASELINE=<artifacts>:31300,31301 run_opt7.sbatch`
  （`run_opt7.sbatch` → `pipeline9.sh` = pipeline7 + 主机遥测 + 停顿报告 + backlog 修复；summary 里有停顿次数、UMBP 失效、
  router backlog）。注意：`PROFILE_AT=3300` 抓 profile 本身会造成 P 约 20 s、D 约 6 s 的"停顿"，停顿报告里那两条可忽略。
- 稳定性复现：`run_hang2.sbatch` → `pipeline_hang2.sh` + `hang_repro2.py`（带负载，不会被 GPU 利用率 watchdog 取消）。
- 分析：`python3 $W/scripts/slow_steps.py $W/logs/<run>`；连接测试 `scripts/keepalive_reuse.py`。
- 监控：`scripts/monitor2.sh`（`setsid nohup` 启动，输出 `$W/logs/monitor2-<时间>.txt`，`monitor2-current.txt` 指向最新；
  重要事件带 `ALERT#N` 编号）。ssh 断开时前台进程会随会话退出，所以要脱离会话启动。
- 用户的 GPU 利用率 watchdog 会取消连续 45 分钟利用率低于 25% 的作业。
- 同时运行的作业数受整个 `amd-hyperloom` 账户的上限约束（`AssocMaxJobsLimit`），排队时间不确定。

## 7. 建议的下一步（11-25 00:00）

1. 看 R12（32k prefill 步）对 R10b 的 A/B：P 每步耗时、每 rank 现算 tok/s、λ。有效就在 c704 / c768 上采用。
2. 放行 R11（c768），确定这套配置的峰值；必要时补 c640 重跑（R9b 那轮 UMBP 失效偏多）。
3. 用户同意后找管理员处理 UMBP（`vm.compact_unevictable_allowed=0` 优先）；kit 里还需要加一个给 UMBP 服务端追加 preload 的开关
   （现在 `UMBP_PRELOAD` 在 GEM 模式下固定为 `libhip_gem_alloc.so`）。
4. 再往上要改代码（EXPERIMENT 5.4 节：MoE 通信与计算重叠、fp8 dispatch 等）。
5. 每个结果都写进 `EXPERIMENT.zh-CN.md` / `README.zh-CN.md`、拷数据到 `raw/`、提交推送。
