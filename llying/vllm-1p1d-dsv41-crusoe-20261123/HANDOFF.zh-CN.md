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

## 5. 稳定性问题现状（11-25 03:05：GPU 空闲型停顿与连接重置已修复；GPU 自旋型 decode 死锁仍在，见 EXPERIMENT 4.10）

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

- R12（32k prefill 步）已完成：367,981，但计分窗口内 UMBP 28.8% 时间不可用，结论待定，需在 UMBP 修好后重测（EXPERIMENT 4.9、5.2）。
- R11（作业 309932，c768）：第一次 AgentX decode 自旋型死锁（4.10），重新部署后被取消（用户选择修复前暂停正式跑分）。
- 我的监控 `scripts/monitor2.sh` 仍在后台跑（输出 `logs/monitor2-1124T1835.txt`），接手后先 `pkill -f scripts/monitor2.sh` 再起自己的。
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

1. 记录 R11（c768）结果，确定这套配置的峰值；必要时补 c640 重跑（R9b 那轮 UMBP 失效偏多）。
2. 每轮都要看计分窗口内 UMBP 不可用时长（EXPERIMENT 4.9 的统计方法），否则轮间比较不成立；R12 需在 UMBP 修好后重测。
3. 用户同意后找管理员处理 UMBP（`vm.compact_unevictable_allowed=0` 优先）；kit 里还需要加一个给 UMBP 服务端追加 preload 的开关
   （现在 `UMBP_PRELOAD` 在 GEM 模式下固定为 `libhip_gem_alloc.so`）。
4. 再往上要改代码（EXPERIMENT 5.4 节：MoE 通信与计算重叠、fp8 dispatch 等）。
5. 每个结果都写进 `EXPERIMENT.zh-CN.md` / `README.zh-CN.md`、拷数据到 `raw/`、提交推送。

## 8. 11-25 17:00 更新（Claude Code 接手 Cursor 会话后）

**提交署名**：`git commit -s`，尾注改为 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。

**稳定性结论（EXPERIMENT 4.11）**：所有"撤队列"都是同一机制：进程里注册给 GPU 的普通内存（KFD userptr：`hipHostRegister`、
`hipHostMalloc`、torch pinned、vLLM UVA 缓冲都是）被内核迁移或写保护时，KFD 撤下该进程所有 GPU 队列，并重新校验整段内存。
规整来源既有我们自己（引擎 THP 缺页），也有同机作业（R13 的 16:09 风暴），所以独占节点也不够，必须把大块内存移出 userptr。
- Engram 表：`VLLM_ROCM_ENGRAM_HIP_VMM=1`（已有）。
- UMBP 418 GiB 池：`instr/umbp_vmm_pool.so`（HIP VMM 主机内存，`UMBP_VMM_NUMA=interleave`）；`hipHostMalloc` 方案无效（探针证实仍是 userptr）。
- 引擎关 THP：`THP_OFF=1`（`instr/thp_off.pth`），去掉我们自己触发的规整（hangF：84 分钟 0 规整、0 撤队列）。
- vLLM UVA 缓冲（`all_token_ids` 4 GiB 等）：`instr/uva_vmm.py`（`VLLM_UVA_VMM=1`），引擎 userptr 4.4 GB → 12–146 MB。
- 剩余：引擎约 3,360 次小 `hipHostMalloc`（torch 锁页缓存、RCCL）；UMBP 服务端在风暴下仍每 2 s 被撤约 0.5 s（来源未查清）。
- GPU 自旋型 decode 死锁（R11、hangE 各一次）原因未查清；`pipeline11.sh` 的 decode 逐步记录每步落盘并记开始，下次可抓现场。

**脚本**：正式跑分 `run_opt9.sbatch` → `scripts/pipeline11.sh`（默认 VMM 池 + NUMA 交替 + THP_OFF + UVA_VMM + hostalloc 日志 +
`kfd_evict_summary.py`）。`.so` 要先由 `probe_kfd.sbatch`（`SHIM_TESTS_ONLY=1`）在镜像里编译测试，用 `afterok` 串起来，
`UMBP_VMM_SO` / `HOSTALLOC_SO` 指向 `logs/kfdprobe-<作业号>/`；`uva_vmm` 的 GPU 测试是 `test_uva_vmm.sbatch`。
复现：`run_hang5.sbatch` → `pipeline_hang5.sh`。KFD 单元探针：`probe_kfd.sbatch` + `scripts/kfd_migrate_probe.py`。
提交示例见 EXPERIMENT 4.11 末尾（R14 的 sbatch 参数）。

**验收标准（用户认可）**：计分窗口内 UMBP 服务端撤队列 0 s；引擎撤队列 < 1%、单次 < 5 s；连续两轮无 decode 死锁；每轮报告
`analysis/kfd-evictions.txt`。达不到的轮次不参与峰值比较。

**监控**：`setsid nohup bash scripts/monitor2_keepalive.sh`（monitor2 挂了自动重启）。ssh 会话结束时登录节点会杀掉该会话
留下的进程（`setsid` 也一样），所以 Claude 要在 tmux 里运行才能在断线后继续。

**正在运行（17:00）**：R14（作业 311936，v3-006，`pd1p1d-c704-r4-hipvmm-1125T1617`）= R13 配置 + UVA_VMM，对比 R10b 494,398。

## 9. 11-26 07:30 更新：结论盘点、正在运行、下一步方法（接手者先读本节）

**正在运行**：R17 同节点 A/B（作业 313299，v3-233，`run_pair.sbatch` → `pipeline11.sh`，两组都**没有** dp_never_pause）：
A = r4-hipvmm（R14 配置，06:04 decode 死锁后已重新部署重跑，约 08:00 结束），B = r15-shexp + `AUX_PRIO=1`（约 10:00 结束）。
结果看 `logs/job-313299.out` 的 "run A/B pipeline rc" 行与两组的 `analysis/`；P 每 token 现算速度看逐步记录。
排队：313424（dp_never_pause 起栈快速检查，`run_opt10.sbatch` → `pipeline12.sh`），可取消，由后续长作业第 1 轮代替。

**能确定的结论**
- 稳定性（原理 + 对照证据，EXPERIMENT 4.11）：所有撤队列都是 KFD userptr 被迁移；Engram HIP VMM、router backlog、UMBP VMM 池
  （NUMA 交替）、THP_OFF、uva_vmm 都有效。它们不提升性能（R14 比 R10b +1.8%，在节点误差 ±3% 内），作用是让测试有效。
- 无效（原理 + 数据）：加大 KV（命中已到理论上限 95.8% / 96.0%，EXPERIMENT 5.3）、EP2 + MegaMoE（P 满步慢 35%）、
  RCCL 通道 / 协议（通信已顶单条 xGMI 带宽 ~53 GB/s）、DBO（vLLM 对 Engram 模型拒绝）。
- shared expert 与 MoE 通信重叠：机制在引擎里证实（高优先级辅助流后 74% 与 RCCL 重叠；微基准每 16k 步省 ~26 ms），吞吐待 R17。
- decode DP2 死锁：首次逐步现场（一 rank 已暂停不跑 dummy，另一 rank 开始真实步未唤醒它），补丁 `instr/dp_never_pause.py`
  已写，**未验证**（需连续多轮无死锁）。

**证据不足 / 无效的测试**：a4w4（R5 c640 +5% vs crusoe 基线，但换了节点、混有 R1 改动、单次）；R12 32k（UMBP 29% 不可用）；
R15（优化未生效，只测出节点差 ~3%）；并发峰值（c640 / c704 / c768 配置与节点都不同）；基线 c704、R4、R7、R8、R11（被死锁 /
停顿 / 重置打断）。**当前没有一项吞吐优化在稳定环境下用对照实验证实过**；503,150（R14 c704）主要是"不再被干扰打断"的结果。

**下一步方法（用户 11-26 要求）**
1. 固定测量环境：同节点配对对比；用户在考虑整机独占、两半同时跑 A / B（需改 kit 端口 / 容器名，查 TTM 上限：两套栈各 ~800 GiB
   驱动内存，默认上限约整机一半 ~1.37 TiB，可能要把 UMBP 池调小），先做 A/A 量出两半差异。用户不接受"另 4 卡跑占位负载"
   以外的方案前，先问清楚；**不要**用占位负载占资源。
2. 用分量指标：P 每 token 现算速度 / 步时长（逐步记录、profile）为主，端到端吞吐为确认；每个候选事先写预期影响的分量与幅度。
3. 按此重测：a4w4 单独隔离、shared expert 重叠（R17）、32k 步 + P 关 cudagraph（`arms/r16-p32k-nograph.sh`）、并发扫描定峰值。
4. 之后都用 `pipeline12.sh`（dp_never_pause 默认开），记录连续无死锁的轮数。

**工具**：`run_pair.sbatch`（同节点 A/B，pipeline11）/ `run_pair2.sbatch`（pipeline12）、`run_gpu_script.sbatch`（单 / 双卡微基准，
`SCRIPT_ENV`、`NPROC`）、`scripts/shexp_overlap.py`、`trace_overlap.py`、`kfd_evict_summary.py`；`pipeline11/12.sh` 的
`STOP_AFTER_SMOKE`、`PROFILE_SMOKE`（全新 16k 请求抓 profile）、`AUX_PRIO`、`SHEXP_PROBE`。

**运行注意**：会话要在 tmux 里（ssh 断开会杀掉会话及其进程）；`monitor2_keepalive.sh` 要重启；hyperloom 每人同时 1 个作业；
不要修改正在被作业执行的流水线脚本（复制成新文件）；agentx 账户只被批准用于编译作业。
