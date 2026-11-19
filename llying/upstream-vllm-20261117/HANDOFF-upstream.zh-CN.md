# 交接：上游 PR / issue（upstream agent，11-19 起；11-19 14:30 更新，交给新会话）

## 00. 新会话从这里开始（11-19 14:30）

- **11-19 15:20 进度（接手的会话）**：第一批写完，等用户 review：`PR-U26-…md`（补了主机内存与 J3）、`PR-U12-…md`（按 main 重写）、
  `PR-U13-aiter-agrs-full-capture.md`、`PR-U19-rocm-pin-worker-slots.md`、`PR-U2-rocm-clr-imported-memory-batch-copy.md`（issue）。
  监控换成新会话的后台 shell（pid 3175390），旧的 2238784 已不在。clr issue 要的宿主 OS / CPU 由 CPU 作业 299146（v3-092）采集，在
  `upstream-20261119/results/platform-info/`。上游 vLLM main 已到 `79e937641`（比 `f0c592bfd` 新 13 个提交，只在 `vllm/envs.py` 上与 U26 / U12 重叠）。
  `gh` 对 ROCm 组织没有 SAML 授权，开 ROCm issue 前要用户授权或走网页。

- **作业**：只剩 J3b = 299112（P4D4 on main，8 卡独占，`-q amd-agentx-qos` 排队中，`QOSGrpNodeLimit`）。J1 / J1b / J2a / J3 都已结束、节点已释放。
  J3b 开跑后约 2–2.5 h（准备里要用 nightly 镜像重编 Infera 实验镜像，约 5 min；第一次在 main 上部署 PD，可能起不来——起不来就看
  `results/p4d4-main/log`、`p4d4/runs/p4d4-main-…/`、`logs/build-main-<作业号>.log`，记录原因即可，不必为它改 PR）。结果会写到本目录
  `results/main-<作业号>/`。
- **监控**：旧会话在登录节点留着一个 `bash jobs/monitor.sh`（pid 2238784）。新会话先 `ps -eo pid,args | grep jobs/monitor.sh`，旧的可以 kill，
  再用自己的后台 shell 启动 `cd /home/liyingli/bench_agentx/upstream-20261119 && bash jobs/monitor.sh`（输出 `AGENT_WAKE_up` 行）。
- **下一步：写 PR 说明和 issue**（数据都已齐，见第 0 节和 `PR-PLAN.zh-CN.md` 第 0 节），每项一个 `PR-<编号>-<短名>.md`（正文英文，其余中文；
  格式参照新版 `PR-U26-rocm-exact-pinned-cpu-offload.md`：分支、只影响 ROCm 的说明、验证数据表（单次注明）、PR description（Problem / Change /
  Tests / Validation / Notes））：
  1. 第一批（约 1–1.5 h）：U26（初稿已有，补 J3：每 worker 474.5–474.9 GB、无崩溃）、U12（更新本目录旧稿；数据见 A 线 README 第 2 节与 `PR-U12-…md`、
     main 单测 126/108、对照必须是 aiter）、U13（A 线 `PR-U13-…md` 起稿，main 单测 16/10，P4D4 确认生效）、U19（测量数据见第 0 节）、
     ROCm clr issue（`kit/checks/rocm-clr/batch_ext_repro.c`、`results/`… 复现输出在 `upstream-20261119/results/rocm-clr-repro/`，B 线 README 1.3 的源码分析）。
  2. 第二批（约 1 h）：U11（含 CUDA 测试需求）、U20、Infera 6 项（U24 用 J3 的 0 错误 + 繁忙客户端 2 / 7,545 对 0 / 7,548；DP 均衡用 m14 / f1 / f2 /
     J3 的数据，注明都是单次、节点不同）。
  3. 第三批（约 30 min）：vLLM nightly 上 offload 测试清理阶段随机中止的 issue（`results/main-u20/*.out`）；Mooncake 通知 QP、decode 显存余量
     两份"观察记录"（只观察到一两次，先不提 issue）。
  4. 写完一起给用户 review；**用户确认前不在 GitHub 上开 PR / issue**。
- **待用户决定**：U12 是否加 `Co-authored-by: yihou <yihou@amd.com>`（kernel 来自其 sglang #47897）；提交上游时是否去掉环境自动加的
  `Co-authored-by: Cursor` 行。
- **可选（不阻塞写作）**：J2——U12 + U13 在 main 上的聚合 A/B（8 卡独占，A 线 `agg_up.sh` / `round.sh` 改成 nightly 镜像 + 同步代码）、
  U11 聚合 PP4 读回、U20 PP4 起栈复现（4 卡非独占）。按第 2 节的资源规则提交。
- PR 分支提交前先把 `-main` 分支挪到当时最新的 main 并在新 nightly 上重测（`steps/main-*.sh` 用到的 `NIGHTLY_COMMIT` / `IMG_NIGHTLY` 在
  `jobs/lib.sh`，换成最新 nightly 镜像名里的提交）。

目标：把 GLM-5.2 PD（Infera + vLLM ROCm，crusoe MI355X）**最佳 recipe 里用到的、以及试过的策略里涉及的** feature、优化和 fix，整理成可以提交的上游
PR / issue（vLLM、Infera、Mooncake、ROCm）。每一项要有：最小改动、测试、验证数据（单次要注明）、PR 说明。**在 GitHub 上开 PR / issue 之前先给用户 review。**
主线 agent 已于 11-19 约 11:00 停止，P4D4 端到端验证也由本 agent 做。

## 0. 当前状态（11-19 14:30）

还没有任何 PR / issue 提交到 GitHub。代码都在仓库外 `/home/liyingli/bench_agentx/upstream-20261119/`，进度表见 `PR-PLAN.zh-CN.md` 第 0 节。

| 编号 | 分支（main 版 = 上游提交用；v0.34.0 版只给 v0.34.0 上的 P4D4 用） | 状态 |
|---|---|---|
| U26 精确锁页（只限 ROCm） | vLLM `u26-rocm-exact-pin-main` / `u26-rocm-exact-pin` | 单测 main 254/251、v0.34.0 226/223；主机内存 37.25 GiB 对 torch 64 GiB（分配 7.8 s 对 13.5 s）；J3（P4D4，PR 代码）每 worker 474.5–474.9 GB、无崩溃；PR 说明初稿 `PR-U26-…md`（待补 J3） |
| U12 gfx950 gluon logits | `u12-gfx950-gluon-mqa-logits-main` / `…` | 单测 main 126/108、v0.34.0 72/54；待聚合 A/B（main）与 PR 说明；yihou@amd.com 署名待用户定 |
| U13 FULL 捕获放行 AITER AG/RS | `u13-aiter-agrs-full-capture-main` / `…` | 单测 main 16/10、v0.34.0 16/10；待聚合 A/B 与 PR 说明 |
| U11 PP 下按最大 stage 分槽 | `u11-offload-pp-worker-bytes-main` / `…` | 单测 main 1,710/1,706；待聚合 PP4 读回；**需 CUDA 测试**（见 PR-PLAN 跨平台表） |
| U20 Simple offload 切不开时按页对齐 | `u20-simple-offload-page-align-main` / `…` | 单测 main 282/275（逐文件进程，因 main 原有的清理阶段崩溃）；待 PP4 起栈复现 |
| U19 ROCm 只注册本 worker 的槽 | `u19-rocm-pin-worker-slots-main` / `…` | 测量：1 TB 区域 4 worker，现规则 2/4 在 687 GB 失败、608–616 s；新规则 4/4、38–41 s；单测 main 814/812 |
| Infera 6 项（U16 串行、U24 池空闲超时、U14 nofile、U15 分词移出事件循环、DP 引擎均衡、vLLM ≥ 0.34 参数解析） | `infera/` 克隆里 `pr/*`，工作树 `iwt/` | Python CI（3.10）/ Rust CI 全过；U16 首轮 clippy / fmt 问题已修；参数解析在 vLLM 0.34 容器内 main 3/3 失败、本分支 5/5 通过；U24 繁忙客户端竞争检查旧设置 2 / 7,545 错误、新设置 0 |
| ROCm clr `deviceId`（U2 根因） | — | ROCm 10.0 / HIP 7.15.26333 上复现（`kit/checks/rocm-clr/batch_ext_repro.c`）；issue 正文待写 |
| 新：vLLM nightly 上 offload 测试清理阶段随机中止（`empty_host_cache`） | — | 待整理成 issue |
| 未开始 | U17（EP + 融合共享专家）、U18（MoRI）、Infera U2（GEM VMM 导入进 `deploy/`）、Mooncake 通知 QP、decode 显存余量崩溃 | — |

作业（结果在 `results/<步骤>/`，计划日志 `logs/plan-<plan>-<作业号>.log`）：
- J3 = 298999（v3-191）：P4D4 用 v0.34.0 PR 代码（U26 / U12 / U13）+ DP 均衡 + U24 路由 + **无 keep-alive 绕法**，3600 s，13:18 开始计分；结果写到
  `results/pr-298999/`（本目录）。**结果：43,269 tok/s/GPU（单次，新节点），路由 5xx / "error sending request" 全程 0 次，PR 代码三项都生效，
  无崩溃**；比 m10 低 11%，原因未定（节点差异 7–10% 以前就有），见 `results/pr-298999/pr-checks.txt`。
- J3b = 299112（排队中）：同样配置但 vLLM = main `f0c592bfd` + U26 + U12 + U13（工作树 `wt/p4d4-main-stack`），Infera 实验镜像用 nightly 镜像重编
  （准备项 `p4d4main`），引擎启动时把 vLLM Python 代码同步成该树（`p4d4/scripts/pd1_up.sh` 的 `VLLM_SRC`）。第一次在 main 上部署 PD，可能起不来，起不来就记录原因。
- J2a = 299106（v3-015，2 卡非独占，已结束）：U19 单测通过（main 补了基线；v0.34.0 那份 772 个全过，不作依据）。
- 待提交 J2：U12 + U13 在 main 上的聚合 A/B（两个 4 卡引擎，8 卡），U11 聚合 PP4 读回、U20 PP4 起栈复现（4 卡非独占）。

## 1. 先读

- 本目录：`CANDIDATES.zh-CN.md`（U1–U26）、`PR-PLAN.zh-CN.md`（第 0 节是 11-19 的状态、跨平台影响表、测试坑）、`PR-U26-…md`（新版）、`PR-U12-…md`（旧稿，待按 main 更新）。
- A 线 `../vllm-decode-trackA-20261118/`（U12 / U13 / U17 / U18），B 线 `../vllm-prefill-trackB-20261118/`（U2、U11、U19、U20），主线 P4D4 README 第 0 节与 1o–1t。

## 2. 规则（11-19 用户要求，取代旧版）

- **PR 基于最新 main 编辑和测试**，能否提交只看"main 对比 main + PR"。main 前进了就把分支挪到新 main 重测。v0.34.0 版本只是从 main 版回移的副本，
  只在该项要进 v0.34.0 上的 P4D4 时才做、才测（旧版"基于 v0.34.0 写、再 rebase 到 main"作废）。
- main 上的测法：nightly 镜像（`vllm/vllm-openai-rocm:nightly-<commit>`，镜像名里的提交就是它编译时的 main）+ 容器里把 vLLM Python 代码同步成被测树
  （`kit/checks/sync_vllm.py`，保留镜像的 `.so`）；步骤开跑前自动检查两提交之间没有改 ROCm 相关的编译代码（`csrc/cpu`、cmake 的 CPU / flash-attn 例外）。
- **只影响 ROCm**：PR 尽量用平台判断限定在 ROCm；无法避免影响 CUDA 的记进 PR-PLAN 跨平台表（目前 U11 需要 CUDA 测试，U20 只在原布局切不开时生效）。
- **不允许 hold 节点**：只用跑完即退的作业；收尾等待窗口默认 0（用户 11-19 允许最多 10 min，仅在明确要追加步骤时开，`DRAIN_S=600`）。
- **GPU 利用率**：按需申请。不用 GPU 的只申请 CPU（`-c 16 --mem 64G`）；不用满 8 卡就不独占（`--gres gpu:N -c … --mem …`，节点会与别人共用）；
  只有 P4D4、两个 4 卡引擎的 A/B 这类用满 8 卡的才 `--gres gpu:8 --exclusive`。能放在一起的步骤放进同一个作业并行跑。
- QoS：`-A amd-agentx -q amd-burst-qos` 马上能拿到（可被抢占，适合可续跑的单测）；不能被打断的计分用 `-q amd-agentx-qos`（组上限 10 台常满，要排队）
  或 `-A amd-primus-cicd -q amd-primus-cicd-qos`（组上限 11 台）。坏节点列表 `kit/jobs/exclude_nodes.txt`（v3-198 docker run 失败、v3-226 无 docker 权限）。
- 仓库里只写本目录；主线目录（`/home/liyingli/bench_agentx/vllm-pd-20261117/`、`llying/vllm-p4d4-crusoe-20261117/`、`…-progress-…`、`…-tools-…`）
  只读，P4D4 用的是拷到 `upstream-20261119/p4d4/` 的副本（`finish_round.sh` / `post_run.sh` 结果写本目录 `results/`，`p4d4_run.sh` 加了 `KEEPALIVE_ENV`、
  `pd1_up.sh` 加了 `VLLM_SRC`）。改 `infera/`、`rust/` 源码前在主线 PROGRESS 写一行。
- 登录节点只跑监控、编辑文件和 git；构建、测试、读大文件都在作业里。**不要修改正在执行的步骤脚本**（11-19 犯过两次；新版 runner 已改为执行快照）。
- 共用分支 push 前 `git pull --rebase`，`git commit -s`；文档中文，PR 说明正文英文；结论要有数据，单次注明；发现之前结论有误直接更正。

## 3. 作业设施（`upstream-20261119/`，副本在本目录 `kit/`）

- `jobs/run_plan.sbatch`：一个作业执行一份计划 `plans/<名>.plan`；资源在提交时按需给（见文件头的三个例子）；`PREP` 用 `+` 连接
  （`v034`、`nightly`、`py310`、`py312`、`rust`、`model`、`p4d4`、`p4d4main`），`jobs/prep_node.sh` 后台准备、步骤用 `need_ready` 等待。
- `jobs/runner.sh`（11-19 14:00 版）：计划每行 `ID NEEDS TIMEOUT ON_FAIL [GPUS]`；按作业实际看到的 GPU（非独占作业只看到自己的）并行调度，
  步骤拿到 `GPU_LIST`（作业 GPU 的编号）和 `STEP_CPUS`；`GPUS=0` 走 CPU 通道，`all` 独占整个作业；步骤脚本先拷快照再执行；
  key = 步骤脚本 + 计划行 + `# key:` 命令输出，已通过且 key 未变的跳过，改了代码 / 步骤的自动重跑（同一作业内也会）。
- `jobs/lib.sh`：`gpu_args` 只映射步骤分到的 render 节点、按作业 CPU / 内存加 `--cpus` / `--memory`、打 `up.job=<作业号>` 标签（作业退出只删本作业容器）；
  `run_vllm` / `pytest_in`（v0.34.0 挂改动文件）；`base_cache`（基线结果按"基线提交 + 镜像 + 测试列表"缓存复用）。
- `steps/_vllm_main.sh`：main 对比的公共逻辑（同步代码、编译代码检查、`PER_FILE=1` 逐文件进程、基线缓存）；`steps/_vllm_unit.sh`：v0.34.0 版。
- `checks/junit_diff.py`：只算新增失败；收集失败或零通过直接判失败。`checks/junit_merge.py`、`checks/u26_host_mem.py`、`checks/u19_region_pin.py`、
  `checks/u24_race/`、`checks/u12_diff_probe.py`、`checks/rocm-clr/`。
- `jobs/monitor.sh`：登录节点后台监控（作业状态、步骤结束、心跳），输出 `AGENT_WAKE_up` 行；`jobs/probe_alloc.sbatch`：看一次非独占分配长什么样；
  `jobs/shm_cleanup.sbatch`：删某节点 `/dev/shm` 里 U19 测量的残留（`-w <节点>`）。
- 提交示例：`sbatch -x "$(paste -sd, jobs/exclude_nodes.txt)" -J lyl-up-<plan> -A amd-agentx -q amd-burst-qos --gres gpu:2 -c 32 --mem 128G -t 01:00:00 --export=ALL,PLAN=<plan>,PREP=v034+nightly,JOB_MEM_G=128 jobs/run_plan.sbatch`。

## 4. 11-19 踩过的坑（都已修，见 PR-PLAN 第 0 节）

- U12 测试容器没设 `VLLM_ROCM_USE_AITER=1`，对照其实是 vLLM 的 PyTorch 参考回退 `fp8_paged_mqa_logits_torch`（差 2⁻¹⁰；11-19 更正：之前写成"Triton 回退"，
  日志原文是 "using the PyTorch fallback"）；对 aiter 只差 7.63e-6（logits 最大 56–63，约 2 个 fp32 ulp）。测试已强制 aiter 并断言。
- U13 首轮两边缺 `ray`，一个测试没跑却判了通过；U11 新测试按 v0.34.0 接口写，main 上要 `full_layer()`（RFC #42449）；`infera-v034` 判定取了 `tail` 的退出码。
- 日志只留最后 60 行导致崩溃看不出原因（现在完整输出存 `/out/<名>.out`）；Infera Python CI 的 key 覆盖了 Rust 改动导致白跑 14 min（现在按目录取 key）。
- 用 8 卡独占节点跑单卡 / 纯 CPU 的工作（J1、J1b GPU 大多闲置）：已改为按需申请 + 并行。浪费核算：J1 约 37 min、J1b 约 15 min 重跑，空等约 13.5 min。

## 5. 接下来

1. J3 结果（约 14:30）：tok/s/GPU、TTFT / ITL、路由 5xx 与 "error sending request"、`pr-checks.txt`，写进 PR-U26 / U24 / DP 均衡的说明。J3b 能否在 main 上起来。
2. 写 J2（main 上：U12 + U13 聚合 A/B 用 A 线的 `agg_up.sh` / `round.sh` / `decode_bench.sh` / `gate.sh`，挂 nightly 镜像 + 同步代码；U11 用 B 线
   `b2_agg_tier.sh` + `tier_probe.py`；U20 PP4 + Simple + 自动 `num_cpu_blocks` 起栈），按 8 卡独占 / 4 卡非独占分成两个作业。
3. 各项 PR 说明（英文正文）与 issue 正文：U12、U13、U11、U19、U20、Infera 6 项、ROCm clr、nightly 清理崩溃；然后一起给用户 review。
4. 之后：U17 / U18、Infera U2、Mooncake 通知 QP、decode 显存余量。
