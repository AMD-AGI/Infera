# 交接：GLM-5.2 P4D4 on Infera + vLLM v0.34.0（主线，2026-11-18）

目标：在 crusoe 上用 Infera + vLLM（ROCm）复现并超过 infera + sglang 的 GLM-5.2 PD recipe（P4D4 C104 53,351 tok/s/GPU、P8D8 C208 54,468），
vLLM 缺的优化走 vLLM / Infera / Mooncake / ROCm 上游。本文件是主线（P4D4 计分、总体协调）的交接；A 线（decode kernel / 通信，v3-240）和
B 线（prefill GEM / 多级 offload，v3-258）各有自己的 session 与目录。

## 0. 11-18 晚间更新（新主线 session；本节取代下面第 1、4、6 节里过时的说法）

### 0.00 新主线 agent 接手（11-19 08:45，先读这里，再读 0.0）

- **分工**：本 agent 是主线（计分、瓶颈分析、优化，目标是超过 infera + sglang：P4D4 C104 53,351、P8D8 C208 54,468）。另一个 upstream agent 并行把
  feature / 优化 / fix 做成上游 PR，交接在 `../upstream-vllm-20261117/HANDOFF-upstream.zh-CN.md`。两边共用分支和节点配额，规则见下面"协调"。
- **成绩（都是单次）**：3600 s 有效成绩最高仍是 **m10 = 48,572**（sglang 的 91%）；1800 s 最高 **m14 = 49,183**（decode 内部 DP 均衡）；f1 48,173 / f2 44,691
  （m14 + decode 显存目标 0.93，两台同配置差 7%）。故障：m9b / m13 decode 崩溃、m12 decode 卡死，更像 decode 显存余量不够（P4D4 README 0、1r、1s、1t）。
- **当前作业**：只有 P8D8 重试 298394（`p8d8_once.sbatch`，配置 `p8d8-conf/c208-v3comm-dplb-dutil093-nonotify-3600.env`，两边关 Mooncake 通知 QP），
  在排队（`QOSGrpNodeLimit`）。没有任何 hold 作业。
- **规则（用户原话 + 11-19 新规）**：
  - 仓库 `/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52`，分支 `dev/pd_opt/vllm_glm52_agentx`，多个 session 共用，push 前 `git pull --rebase`；`git commit -s`。
  - **不允许用 hold 作业占机器，不跑 benchmark 必须释放**：只用跑完即退的作业（单节点 `run_once.sbatch`，两节点 `p8d8_once.sbatch`），分析在同一个作业里做完。
  - 不要碰 `queue/`、`queue-n2/`。登录节点只跑 `scripts/monitor.sh`（加编辑文件、git）；分析、读大文件、构建、测试都在计算节点的作业里做。
  - spur：每个节点只能有一个进入作业的会话；作业步骤非零退出会结束作业；不注入宿主机 libionic；不改正在执行的 bash 脚本（队列项一律经 `scripts/snap.sh` 跑快照）。
  - 自主决策；后台任务设监控；完成一项就整理、提交、推送；结论要有数据，单次要注明；发现之前结论有误直接在文档里更正；文档用中文。
- **协调（与 upstream agent）**：
  - 配额：每人同时约 2 个作业 / 2 台节点（`QOSMaxJobsPerUserLimit`、`QOSMaxNodePerUserLimit`），组还有总配额（`QOSGrpNodeLimit`）。**主线优先**；
    upstream agent 最多同时 1 个单节点、≤ 2 h 的作业，作业名以 `lyl-up-` 开头。提交前都先看 `squeue -u liyingli`。
  - 目录：主线拥有工作目录 `/home/liyingli/bench_agentx/vllm-pd-20261117/`（scripts、overlay*、runs、once、p8d8-conf）和仓库里的
    `llying/vllm-p4d4-crusoe-20261117/`、`llying/vllm-pd-progress-20261117/`、`llying/vllm-pd-tools-20261117/`；upstream agent 只读这些。
    `infera/`、`rust/` 源码两边都可能改：改之前在 PROGRESS 里写一行，改完带测试提交，避免同时改同一个文件。
- **怎么跑**：
  - P4D4：写 `once/<名字>.sh`（一轮计分 + 分析，示例 `once/p4d4-m14b-3600.sh`，**已写好、未提交**），
    `sbatch -x "$(paste -sd, /home/liyingli/bench_agentx/pd-repro-20261116/exclude_nodes.txt)" -J lyl-vpd-<名字> --export=ALL,ONCE=<名字> run_once.sbatch`。
  - P8D8：写 `p8d8-conf/<tag>.env`，`sbatch -x ... --export=ALL,P8D8_TAG=<tag> p8d8_once.sbatch`。
  - 监控：`JOBNAME=<作业名> QDIR=<作业专用队列目录，单节点作业随便给一个> bash scripts/monitor.sh` 放后台，等 `"event":"running"` / `"event":"rc"`；
    AwaitShell 的 pattern 匹配整个输出，要用只会新出现的字符串（例如带作业号或项名）。
- **建议的下一步**：
  1. 提交 `once/p4d4-m14b-3600.sh`：m14 原配置（DP 均衡 + decode 显存目标 0.95）跑 3600 s，看能否超过 m10。m14 的 1800 s 49,183 是最好单次，f2 提示 0.93 损失约 8%。
  2. decode 是瓶颈（交互 60 对 69.9、ITL 16.5 对 14.3 ms）：先量清 decode 在 torch 池外用了多少显存（aiter AG/RS 缓冲区、scratch、hipBLASLt），
     在"KV 容量"和"不崩"之间找到可证明安全的显存目标；再看 decode 每步时间（A 线 kernel 方向）。
  3. P8D8：298394 若仍失败，减少端点（`MC_NUM_QP_PER_EP=1`、每 worker 少用网卡）或查 ionic QP / MR 上限；8 张网卡同时用时主机内存跨节点只有约 4 GB/s（单卡 30 GB/s），也要查。
- **坑**：登录节点上的 monitor 在 11-19 02:45 被一起清掉过（要定期 `ps` 检查）；`scripts/p4d4_run.sh` 11-18 22:38 被外部清空过（提交前确认脚本非空，以仓库
  `../vllm-pd-tools-20261117/scripts/` 为准）；复用 run 目录会留下旧的 `samples.jsonl.stop`；prefill 每轮有约 1,400 条 hipBLASLt `INTERNAL_ERROR` 警告（不致命，原因未查）。

### 0.0 11-19 06:15 更新

- **机器使用规则（11-19 06:10 起，严格执行）：不允许用 hold 作业占机器，不跑 benchmark 时必须释放。** `hold_n*.sbatch` + 队列 runner 的方式停用。
  当时 n3（296780，v3-006）、n4（296781，v3-124）上正在跑 f1 / f2 计分；已在两个队列放 `STOP`（runner 不再接新任务），并排了
  `scripts/finish_round.sh`（等计分结束、在节点上跑完全部分析、结果写进 `../vllm-p4d4-crusoe-20261117/results/f1|f2/`），全部结束后作业自动退出、节点释放。
  以后每次计分都要提交"跑完就退出"的作业，分析也在同一个作业里做完。
- **跑完即退的 P8D8 作业**：`p8d8_once.sbatch`（两节点）。NODE0 跑批处理脚本本身，NODE1 只起一个步骤（`scripts/job_runner.sh`，作业专用目录 `queue-j<JOB>/`）；
  顺序是两台体检 → 两台 `node_prep.sh`（新节点会拷模型、建镜像）→ `p8d8_run.sh` → `scripts/p8d8_finish.sh`（结果写进 `../vllm-p4d4-crusoe-20261117/results/p8d8-<tag>/`）→ 退出。
  配置放 `p8d8-conf/<tag>.env`，提交：`sbatch -x "$(paste -sd, exclude_nodes.txt)" --export=ALL,P8D8_TAG=<tag> p8d8_once.sbatch`。
  06:17 已提交 298231（`c208-v3comm-dplb-dutil093-nonotify-3600`：m14 / f 配置 + 两边 `MC_RDMA_NOTIFY_ENABLED=0`，3600 s），等 f1 / f2 的作业结束后开始。
- **最好的有效成绩**：3600 s 同口径仍只有 m10 = 48,572；1800 s 的 **m14 = 49,183**（m10 配置 + decode 用 vLLM 内部 DP 均衡 `INFERA_VLLM_DP_ENGINE_LB=1`，
  v3-124，单次；TTFT p90 8.61 s、交互 p90 55.7、ITL p90 17.9 ms，均明显好于 m10）。f1 / f2 = m14 配置 + decode 显存目标 0.93，3600 s，两台各一次（06:04 起）。
- **m9b / m12 / m13 的崩溃和卡死更可能是 decode 显存余量不够**（推断）：m13 与 m9b 同一崩溃签名（hipBLASLt `INTERNAL_ERROR` → `HSA_STATUS_ERROR_OUT_OF_RESOURCES`），
  崩溃前也有 m12 那样的"生成速度掉到几十 tok/s"。decode 显存目标 0.95 只给 torch 池外留约 14 GB。f1 / f2 用 0.93 验证。之前把 m12 归因于精确锁页、把 m9b 归因于
  `--max-num-batched-tokens 1024` 都不成立（P4D4 README 1r、1s 已更正）。
- **P8D8 第一次尝试（11-18 22:46）在 smoke 阶段失败**：跨节点 Mooncake 建连失败（prefill `Notification QP initialization failed: -302` 1,272 次，decode 反复重建连接），
  同节点回环没有这类错误。单进程主机内存跨节点测试（`scripts/mc_xnode.sh`）正常，所以怀疑是 8×8 worker 全互连时通知 QP 的资源不够；下次试 `MC_RDMA_NOTIFY_ENABLED=0`（P4D4 README 1s）。
- `scripts/p4d4_run.sh` 在 11-18 22:38 被外部清空过（原因不明，不是本 session 写的），已从仓库副本恢复；共用目录里的脚本以仓库 `../vllm-pd-tools-20261117/scripts/` 为准。

- **最好的有效成绩（与 sglang 同口径 3600 s）：m10 = 48,572 tok/s/GPU，sglang 53,351 的 91%**（v3-006，单次；重复轮 m12 在测量第 12–14 min
  decode 卡死、无效，见 `../vllm-p4d4-crusoe-20261117/README.zh-CN.md` 1r）。TTFT p50 / 均值 2.88 / 5.20 s 已好于 sglang（4.20 / 5.65），差距在 decode（交互 59.4 对 69.9）。
- **精确锁页覆盖层已换成直接 `hipHostAlloc`**（22:00）：m12 卡死时主机反复出现 KFD 进程驱逐告警，第一版用的"mmap + `hipHostRegister`"是唯一新的驱动层因素；
  新版与 torch 锁页同一机制、只是不取整。用法不变（同一个覆盖层文件）。
- **怎么开最好的配置**（`/home/liyingli/bench_agentx/vllm-pd-20261117/`，队列项经 `scripts/snap.sh` 跑 `scripts/p4d4_run.sh` 的快照）：
  ```bash
  OV=$R/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/gpu_worker.py:ro
  D_MODE=v3comm P_GEM=1 P_UTIL=0.88 RENDER_TIMEOUT=600 P_RENDER=4 D_RENDER=4 DURATION=3600 USAGE_LOG=1 \
    CPU_BYTES=1900000000000 P_MOUNT_X="$OV" bash scripts/snap.sh p4d4_run.sh <TAG>
  # 另起一个队列项做内存守护：bash scripts/memguard.sh p4d4-<TAG> 120
  ```
  即：Rust 路由串行 + 两边 keep-alive 120 s（s13）+ decode A 线 v3comm（#47897 v3 + aiter AG/RS 修正）+ 两边 `--renderer-num-workers 4`
  + prefill GEM 0.88（`bin/libhip_gem_alloc-vmm.so`）+ CPU 层 1.9 TB（精确锁页覆盖层 `overlay-cpupin/`，否则 1.25 TB 以上会因 2 的幂取整 OOM）。
- **这次 session 的关键发现**（细节在 P4D4 README 1l–1q）：
  1. 每轮测量期开头 5 min 都只有约 22K（测量方式的瞬态），1800 s 成绩低估了 3600 s 同口径的数；同配置两台节点差约 10%（差在 decode）。
  2. prefill 渲染在单线程里排队（每请求 1.35 s，成本是 Rust 分词）→ renderer 4 线程降到 0.2 s。
  3. prefill 多算的主要来源是长上下文会话停顿数分钟后被 CPU 层整段淘汰（容量）；ROCm 上 CPU 层经 torch 锁页分配器按 2 的幂取整（U26），
     改成精确锁页后 1.9 TB 只占 1.90 TB，prefill 现算降到 ISL 的 3.92%（低于 sglang）。
  4. decode 是现在的瓶颈：rank 间 KV 不均（一个 rank 满、另一个 78%），最忙 rank 请求数是平均的 1.28 倍；m14 试 vLLM 内部 DP 均衡（`INFERA_VLLM_DP_ENGINE_LB=1`），
     m13 试路由 `INFERA_DECODE_REQUEST_WEIGHT=100000`。
  5. 踩过的坑：v3-106 docker 坏（NVMe 卷晚挂，遮住 containerd 数据目录）；burst 节点 23–70 min 就被抢占，不适合计分；decode
     `--max-num-batched-tokens 1024` 会在 885 token 的混合批次上崩；s12 系列 prefill 崩溃不是 tokenizers（worker 侧，可能是 OOM 连锁），
     开 faulthandler 后 m6 / m10 的 prefill GEM 0.88 都没再崩。
- 上游：`../upstream-vllm-20261117/CANDIDATES.zh-CN.md`（U1–U26）、`PR-PLAN.zh-CN.md`；本分支已提交 Infera 侧 U14 / U15 / U16 测试 / U24 / DP 引擎均衡开关的代码与测试，
  vLLM 侧 U12 / U13 / U26 有 PR 草稿。

## 1. 现在在哪（11-18 中午的状态，已被第 0 节取代）

- **最好的有效成绩：s13 = 42,724 tok/s/GPU（sglang 的 80%）**（11-18 11:07–11:37，s10 + decode #47897 v2 + 两边 keep-alive 120 s，prefill 不用 GEM；
  6,202 完成 / 1 丢弃，TTFT p50 / p90 / 均值 5.62 / 18.95 / 9.16 s，交互 57.9 / 52.3；单次）。此前 s9 = 36,923、s10 = 36,265。详细表见 `../vllm-p4d4-crusoe-20261117/README.zh-CN.md` 第 0、1h–1k 节。
- **当前瓶颈在 decode**：decode KV 满（每 rank p50 0.85–0.90）、每步慢（ITL 约 19 ms 对 sglang 约 14 ms）。prefill 命中已接近理论上限（现算约 5%），
  路由、KV 传输（0.19 s / 次）都不是瓶颈（README 1k）。
- 组合配置 s12b / s12c 的结果见第 4 节（本文件最后更新时补）。

## 2. 环境

- 仓库 `/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52`，分支 `dev/pd_opt/vllm_glm52_agentx`（三个 session 共用，push 前 `git pull --rebase`）。
  本分支上的代码改动：`infera/engine/vllm/gem_mooncake.py`（GEM 内存池）、`infera/engine/vllm/args.py`（v0.34 适配）、
  `infera/router/disagg_protocols/vllm_mooncake.py`（Python 路由串行开关）、`rust/router/src/disagg.rs`（Rust 路由 vLLM 串行，U16）。
- 工作目录 `/home/liyingli/bench_agentx/vllm-pd-20261117/`：`scripts/`（最新脚本，分支里的副本在 `../vllm-pd-tools-20261117/scripts/`）、
  `queue/`（v3-135 的 runner）、`queue-n2/`（v3-258，**B 线专用**）、`queue-n3/`（新节点，`hold_n3.sbatch` / `q3.sh`）、`runs/<run>/`（每轮原始数据，
  含实时落盘的 `vpd-prefill/decode/router.log`）、`logs/summary.out`（每轮一行成绩）、`bin/`（`infera-router-vllmserial`、`libhip_gem_alloc-vmm.so`）、
  `overlay/`（#47897 覆盖层 v2）。
- 节点 / 作业：294199（v3-135）已到期；**296460（`hold_n3.sbatch`）11-18 约 13:40 起在 crsuse2-m2m-v3-106，24 h**，批处理脚本里的 runner
  执行 `queue-n3/`，用 `q3.sh` 投递（新节点要先建镜像、拷模型）。294784（v3-258，B 线）：B 线报告该节点约 1.27 TB 锁页内存泄漏、需重启后才能再跑大 CPU 层。
- 新节点准备（参考 `queue-n2/1118-040712-Lprep-n2.run`）：从提交 `0c5e5e89` 用 `git archive` 构建 `infera-vllm:v0340-0c5e5e89`
  （Dockerfile `../vllm-image-v0340-20261117/`），`scripts/stage_model.sh /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 /mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4`。
  AgentX 第一次运行会在节点本地 clone InferenceX（`918524ff`）。`pd1_up.sh` 自动取节点 IP，但**队列项模板（`s9.sh`、`s10.sh` 等）里写死了 v3-135 的 IP `10.245.144.166`**（采样、监控的 URL），换节点要改。
- **登录节点只跑监控**（`scripts/monitor.sh`，打印 `AGENT_LOOP_WAKE_vpd` 事件）；分析、打包、读大文件一律经 `q.sh` / `q3.sh` 在计算节点执行
  （曾因在登录节点读几百 MB 的 JSON 卡死会话）。等待时匹配 monitor 的 `"event":"rc"` 事件，别只匹配某一个队列项。
- spur：每节点一个 exec-into-job 会话；作业步骤非零退出即结束作业；不注入宿主机 libionic；不要改正在执行的 bash 脚本（`pd1_up.sh` 每轮执行一次）。

## 3. 跑一轮 P4D4

- 以 `../vllm-p4d4-crusoe-20261117/queue/s10.sh`（Rust 路由串行，单 API server）或 `s9.sh`（Python 路由 round-robin）为模板复制成新队列项，
  只改要比较的变量；名字以 `L` 开头。部署（P TP1×PP4 GPU 0–3、D DP4 GPU 4–7、etcd、路由）由 `scripts/pd1_up.sh` 完成，开关见文件头：
  `P_GEM / D_GEM`、`P_ARGS / D_ARGS`、`P_ENV / D_ENV`、`P_MOUNT_EXTRA / D_MOUNT_EXTRA`、`P_KV_EXTRA`、`ROUTER_BACKEND`、`ROUTER_ENV`、
  `ROUTER_POLICY_ARGS`、`ROUTER_MOUNT_EXTRA`。计分 `scripts/score_p4d4.sh RUN`（`SMOKE=1 DURATION=1800`），一行成绩写 `logs/summary.out`。
- 一轮约 70 min：部署 12–14 min，AgentX 预热 24–35 min，测量 30 min。**AgentX 预热阶段任何根请求失败都会中止整轮**；叠加没验证过稳定性的
  改动前先做 10 min 左右的稳定性检查。
- 分析工具（计算节点上跑）：`ttft_breakdown.py RUN`（引擎内 vs 路由的 TTFT / 排队）、`hit_timeline.py`（prefill 命中随时间）、`mc_xfer.py`
  （Mooncake 传输）、`samples_summary.py`、`collapse_top.py`（py-spy raw）、`trace_decode.sh` + `trace_cat.py`（decode torch trace 分类）、
  `ckpt_sizes.py`（checkpoint 各类权重大小）。多 API server 时 vLLM `/metrics` 只来自一个进程，`samples.jsonl` 的引擎数字不可信，看路由与 AIPerf 导出。

## 4. 成绩（C104，1800 s；sglang 53,351）

| 轮 | 相对上一轮的变化 | tok/s/GPU | TTFT p50 / 均值 s | 备注 |
|---|---|---|---|---|
| s4 | Python 串行路由 + 缓存感知准入 + 1 TB CPU 层 | 26,541 | 22.5 / 33.8 | |
| s6c | ARC 淘汰 | 23,864 | 15.3 / 40.8 | 不用 ARC |
| s7b | 1.25 TB CPU 层 + 路由 nofile | 30,091 | 28.5 / 26.3 | |
| s8 | P / D 4 API server | 31,384 | 23.9 / 24.3 | API server 不是瓶颈 |
| **s9** | 路由 round-robin（不 tokenize） | **36,923** | 6.8 / 12.4 | 最好 |
| s10 | Rust 路由 + vLLM 串行（kv-aware） | 36,265 | 6.0 / 11.3 | 同 s9 |
| s11 | s10 + 4 API server + renderer 8 | 中止 | | Rust 路由 → decode 502（keep-alive 竞争） |
| s12 | s11 + prefill GEM 0.88 + decode #47897 | 中止 | | prefill EngineCore 段错误 |
| s12b | s12 + keep-alive 120 s | 无效（17,853，641 完成 / 79 丢弃） | 2.6 / 4.7 | 测量第 6 min prefill PP0 worker 被信号杀掉（09:11:41，exit code None） |
| s12c | s12b，prefill 单 API server | 中止 | | 无崩溃；prefill 500 `Chat template rendering timed out after 30.0s`（单 API server、单 renderer 线程，`VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT` 默认 30 s） |
| s12d | s12c + 两边 `VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT=600` + prefill renderer 4 线程 | 无效（34,899，4,162 完成 / 473 丢弃） | 5.4 / 11.8 | 测量中 prefill worker 死亡，之后 377 个"no prefill worker"、50 个 unreachable、36 个 500 |
| **s13** | **s10 + decode #47897（v2，组 4）+ 两边 keep-alive 120 s，prefill 不用 GEM** | **42,724** | 5.62 / 9.16 | 6,202 / 1；最好（单次） |

## 5. 已知问题（按影响排序）

1. Infera Python 路由 kv-aware 在事件循环里同步 tokenize（U15）：用 Rust 路由（`ROUTER_BACKEND=rust ROUTER_ENV=INFERA_VLLM_MOONCAKE_SERIAL=1
   ROUTER_MOUNT_EXTRA="$R/bin/infera-router-vllmserial:/usr/local/bin/infera-router:ro"`），或 Python 路由 `ROUTER_POLICY_ARGS="--router-policy round-robin"`。
2. Rust 路由（reqwest，空闲连接 90 s）与 vLLM uvicorn（keep-alive 5 s）的连接复用竞争 → 502：两边引擎加 `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120`（s12b 起）。
3. 路由描述符：连接峰值 1,150–1,470，`pd1_up.sh` 已给路由容器 `--ulimit nofile=1048576`（U14）。
4. **prefill GEM 0.88 在 v3-135 上 3 次都挂了**（s12 EngineCore 段错误 `PyEval_RestoreThread`；s12b PP0 worker 被信号杀掉；s12d 单 API server 也挂），
   与 API server 个数无关。B 线 b1-r1（同 GEM 0.88、Python 路由、无 #47897、v3-258）跑完 1800 s 无故障（1 次）。差别：Rust 路由、decode #47897、
   keep-alive、节点。需要和 B 线一起查（拿 core dump / dmesg；先在 v3-258 上用本线配置复跑一次区分节点因素）。在此之前组合配置里先不用 prefill GEM（s13）。
5. GEM + offload 故障已由 B 线定位并绕过（U2：clr 的外部内存没有 `deviceId`；`bin/libhip_gem_alloc-vmm.so` 改用 VMM 导入，挂到
   `/opt/infera-gem/libhip_gem_alloc.so`，`P_GEM=1`，prefill 显存目标 0.88）。
6. 每轮 3–9 条"空内容"失败（固定在少数会话，与长度无关），原因未查；落在根请求上就会中止 AgentX 预热。
7. `TieringOffloadingSpec` 在 PP>1 起不来（U11，B 线在修），修好后可试 NVMe 二级层（3600 s 计分时工作集更大）。

- **prefill 前端是 prefill 变快之后的新瓶颈**：GEM 0.88 让 prefill 更快以后，单 API server 的 chat 模板渲染（Python Jinja，约 11 万 token 的对话）排队
  超过 30 s；而 prefill 开 4 个 API server + GEM 时两次挂掉（第 5 节第 4 条）。要么查清崩溃后用多 API server，要么用 renderer 线程 + 放宽超时（s12d），
  根本解法是不让引擎重复渲染（路由或上游把渲染结果 / token id 传给引擎）。

## 6. 下一步（建议顺序）

1. 看 s12b / s12c：选一个稳定的组合配置（GEM 0.88 + #47897 + keep-alive，前端扩展是否保留）。
2. **decode 每步**（主攻）：A 线的 aiter 自定义 AG/RS 修正（`../vllm-decode-trackA-20261118/overlay-comm/`，U13；每步 MoE 通信 15–26%）接进 P4D4；
   decode EngineCore 主线程约 99% CPU（py-spy：`queue/1118-082247-Lpyspy-s12b` 若已采到，结果在 `runs/<s12b>/pyspy/`）；草稿注意力元数据每步重建
   （`ROCM_AITER_MLA_SPARSE cannot update draft decode metadata in place`）。
3. decode 容量：`--gpu-memory-utilization 0.97`、减小 decode 的 `--max-num-batched-tokens`（激活峰值 6.7 GiB）。
4. 用最好配置跑 3600 s（与 sglang 同口径），并重复一次量化波动（目前都是单次）。
5. P8D8（两节点：`scripts/pd_role_up.sh`、`scripts/pd2_drive.sh`，未测）。
6. 上游：`../upstream-vllm-20261117/CANDIDATES.zh-CN.md`（U1–U16）；优先 U12（#47897，草稿 `PR-U12-*.md` 与 A 线的 v3 改法）、U13、U15 / U16 / U14（Infera）、
   U2（Infera GEM VMM 导入 + ROCm clr）、U11。

## 6.1 A / B 线截至 09:40 的新结论（未合并进共享文件）

- A 线：#47897 v3（组大小来自 indexer metadata）+ aiter 自定义 AG/RS 修正（FULL graph 下放行）组合，decode 每步 49.66 ms（−6.1%），门禁 199/200
  （`../vllm-decode-trackA-20261118/README.zh-CN.md`）。下一步可直接把 v3 + comm 覆盖层接进 P4D4 decode。
- B 线：写回式 offload 调研——vLLM 0.34 的 `SimpleCPUOffloadConnector` 有 `lazy_offload`（U9 的候选解），在 ROCm / PP 下的可用性见
  `../vllm-prefill-trackB-20261118/README.zh-CN.md`。

## 6.2 A / B 线交付（11-18 下午，**新 session 第一件事是读并合并**）

- **已合并（11-18 15:40，新主线 session）**：CANDIDATES（U1–U25）、P4D4 README 第 0 节、KF README、PROGRESS（D30–D33）。v3-106 docker 损坏、已换节点，见 PROGRESS D30 / D31。

- A 线：`../vllm-decode-trackA-20261118/HANDOFF-to-main.zh-CN.md`（要合并的 CANDIDATES 条目：U6 / U12 / U13 替换、新 U17 EP shared-expert bug、U18 MoRI；
  推荐的 decode 开法；PR 草稿 `PR-U12-*.md`、`PR-U13-*.md`）。A 线最终 A/B：v3 + aiter AG/RS 相对原版 decode 每步 −6.3%（104 请求）/ −5.3%（64 / 32）。
- B 线：`../vllm-prefill-trackB-20261118/README.zh-CN.md` 0.1 节（给主线的更新：U2 已解决、U11 已修、U9 / U10 说明、新候选）。另：
  Simple lazy（写回式）P4D4 3600 s = 28,696，不如写穿；**host 侧多次崩溃（tokenizers 段错误、libtorch_cpu SIGILL）指向 HF tokenizers 0.23.3 堆损坏**，
  建议加 faulthandler / core dump、试 `TOKENIZERS_PARALLELISM=false` 或换 tokenizers 版本——主线 s12 / s12b / s12d 的 prefill 崩溃很可能同因，
  先按这条查，再决定 prefill GEM 是否可用。

## 7. 与 A / B 线的协作

- A / B 线只写自己的目录（`../vllm-decode-trackA-20261118/`、`../vllm-prefill-trackB-20261118/`），不改共享文件；主线负责把结论合并进
  `PROGRESS.zh-CN.md`、`CANDIDATES.zh-CN.md`、P4D4 README、KF README。合并前看它们最新的 README 与提交（`git log -- llying/vllm-*-track*`）。
- B 线交接 `../vllm-p4d4-crusoe-20261117/HANDOFF-trackB.zh-CN.md`；A 线交接 `../vllm-kf-port-20261117/HANDOFF-trackA.zh-CN.md`。
