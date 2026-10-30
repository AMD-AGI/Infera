# infera-sglang GLM-5.2 在 AUS 上按 InferenceX 流程实测（10-29 起，持续更新）

目标：在本集群（AUS / smci355，Compute-DCPT 分区）上，用 InferenceX CI 的同一套脚本和流程跑
[InferenceX#4629](https://github.com/SemiAnalysisAI/InferenceX/pull/4629) 的 `glm5.2-fp4-mi355x-infera-sglang-agentic-disagg`：
原样 lmsysorg 镜像 + 作业启动时装 Infera wheel + 打补丁，经 srt-slurm（带 #630 infera frontend）拉起，跑 AgentX 与 GSM8K eval。
先做功能对齐（三种 PD 形态各一点、AgentX fast 模式），再选两个点做性能对齐（3600 s，与 #210 对比）。
过程中发现的 PR 问题直接修并推送。

## 0. 当前状态（随时更新）

- 10-30 08:00 UTC：在占位作业 34971（n02-29 + n05-33）里跑，4+4 C80 功能已通过（AgentX 加 GSM8K），1P1D C88 运行中，2P1D C176 排队；之后在 srt-slurm `59920ea1` 加原版 630 补丁上跑两个性能点。
- #4629：merge 上游 main（只有 changelog 冲突；上游 #4601 把 srt-slurm 升到 `59920ea1`，回移随之撤回）已推送 `4a400018`，GitHub 显示 MERGEABLE；与上游 main 对比全量 infx 测试，没有新增失败。
- #222：已推送 `f668fbf7`，并在 GPU 上验证通过（见 0.0），建议合入。

## 0.0 结果（随时更新）

| 点 | 方式 | AgentX | GSM8K（1319 题） | 结论 |
|---|---|---|---|---|
| 4+4 C80 | fast 1200 s，srt-slurm `ae3d6877` 加回移后的 630 补丁 | 33,121 tok/s/GPU，e2e-norm p90 20.7，TTFT p90 5.2 s，0 失败 | strict 0.9712 / flexible 0.9727 | **功能通过** |
| 1P1D C88 | fast | 运行中 | — | — |
| 2P1D C176 | fast | 排队 | — | — |

4+4 C80 的验证细节：
- 两台节点日志都有 `[infera-sglang-patches] all five applied`，Mooncake 识别到 ionic_0–7；
- 起栈约 15 分钟（router 06:10，两个 worker 登记完成 06:25）；
- 2605 个请求发出、2598 个完成，结束时 7 个在途被取消，0 失败。

**Infera #222 在 GPU 上验证通过**：
- srt-slurm 给 prefill 分配的 KV 事件端口 5200 经 `--kv-events-config` 传给包装层，SGLang 在 `tcp://*:5200` 上发布，包装层在 etcd 里登记的是同一端口，router 订阅 `tcp://<prefill>:5200`；decode 侧为 2606，4 个 DP rank 同理。
- 2769 次 prefill 选路中 1704 次命中缓存前缀（62%），与服务端 prefix cache 命中率 63.8% 一致。
- decode 关闭了 radix cache（`disable_radix_cache: True`），0 次命中属正常；router 那一条 "kv event feed is not being applied" 警告（decode dp2）也是这个原因。
- 未覆盖：#222 对 vLLM 包装层的改动。

## 0.1 实测中的决定（汇总）

| # | 决定 | 依据 |
|---|---|---|
| D1 | 用 AMD 内部 InferenceY（本集群上的 InferenceX CI 仿真）的 smci355 集群记录和 `manual_launch.py` 跑点，不自己写拉起脚本 | runner 以下与 CI 完全相同；只改分区、目录、网卡等站点项（TEST-ONLY 提交） |
| D2 | recipe 设 `MOONCAKE_DISABLE_HIP_DMABUF=1`，不实测 0 | CI 机器上 ATOM PD 用同一个 Mooncake（`78566d45`）、同为 1 GiB MR，关 dma-buf 跑通；TileRT 设 0 是因为单次注册 12.54 GB，我们按 1 GiB 分片注册，不受影响；本集群设 0 起栈即段错误。风险：CI 上若 1 GiB 的 peer 注册也被拒，要改 0 |
| D3 | 只读挂载宿主机 ionic provider | 照 Kimi-K3 vLLM 的 recipe（上游和 InferenceY 都这么挂） |
| D4 | 补丁放在 `configs/infera-sglang-patches/`，由安装脚本调用；任何一个没打上就让作业失败，不静默跳过 | waiver 先例 #2508 的做法；安装脚本单测里用 `INFERA_SGLANG_PATCHES=0` 跳过补丁步骤，另加一个测试保证"补丁打不上则安装失败" |
| D5 | router 检查改为 `launch_rust._find_binary()` | #223 之后 router 在 wheel 的 `infera/_bin/` 下，不在 PATH 上，`shutil.which` 必然失败 |
| D6 | Slurm 账户用 `xiewen12`（用户 10-30 同意） | 默认账户 `emad` 在 Compute-DCPT 上不能 srun/salloc，只能 sbatch；launcher 导入镜像要用 srun。`xiewen12` 的 fairshare 也更高（0.257 对 0.107） |
| D7 | 镜像导入用上游 `mi355x-amds` 的方式（enroot 直接从 registry 匿名拉取），不用 InferenceY 的 dockerd | 节点 docker 数据盘满了（n02-25），共享 home 里的 Docker Hub token 也过期了；上游 CI 本来就不经过 dockerd。节点上的 docker 镜像可能有别人的，不去清理 |
| D8 | 拉起前清掉当前 shell 里残留的 `SLURM_*` 环境变量 | 会话里残留旧 salloc（31625）的变量，srun 会去找已失效的作业 |
| D9 | Compute-DCPT 节点的 host 网卡用 `fenic` | 这些节点上没有 InferenceY 节点的 `enp81s0f1`（节点探测结果） |
| D10 | 测试时作业墙钟时间改成 3 h（recipe 写的是 8 h） | 便于 backfill 排上；一个 fast 点约 1 h，3600 s 的点约 2 h |
| D11 | 性能对齐选 4+4 C80 和 1P1D C64 | 8 卡吞吐端与 12 卡交互端各一个，覆盖两种形态、Pareto 两端 |
| D12 | 三个点先照 CI 方式各自排队；同时排一个 2 节点占位作业（用户 10-30 定），排队作业失败后的重跑都放在占位作业里 | 单独排队的作业失败后重新排要几个小时。占位作业里执行的是 srtctl 生成的同一份 sbatch 脚本、配置和提交时的完整环境（`render_point.sh` 用假 sbatch 记下来），`do_sweep` 及以下与 CI 相同；不同的只有提交、跟踪、收集这一层 |
| D13 | 镜像导入回到 InferenceY 的 dockerd 方式，由我在占位作业里预先导入成 squash | rootless pyxis/enroot 从 registry 解包 lmsysorg 镜像会失败（`/root/go` 下的只读目录，`Cannot mkdir: Permission denied`），三个排队作业都因此在 etcd 这一步失败；上游 CI 集群上是否同样失败未知 |
| D14 | 测试配置去掉 GPU exporter 边车 | 这几个点都是 POWER=0；它的镜像现在拉不下来（Docker Hub 匿名拉取限流） |
| D15 | 给 InferenceY 的 `manual_launch.py` 加 `--eval-only` | CI 对 run-eval 的行另起一个 "multi-node agentic eval" 作业（run-eval、eval-only 为 true，conc 取 eval-conc）；manual_launch 原本把这两项写死成 false |

## 0.2 实测发现并已修复的 PR 问题

| PR | 问题 | 修复 |
|---|---|---|
| #4629 安装脚本 | #223 之后 router 打包在 wheel 里、不在 PATH 上，原来用 `shutil.which` 检查必然失败 | 改为 `launch_rust._find_binary()`（`996ea049`，已推送） |
| #4629 带的 `630-infera-frontend.patch` | 补丁是按 srt-slurm 新 main 写的，用了 `runtime.reuse_container`、`container_prepare_timeout_seconds`；InferenceX 固定的 `ae3d6877` 没有这两个参数，router 起不来（`AttributeError`）。单元测试 mock 了进程启动，所以没测出来 | 按 `ae3d6877` 重新生成补丁，router 调用与同版本 static router 一致（`a8986514`，待实测通过后推送）；#630 本身基于 main，不受影响 |

## 0.3 InferenceX 维护者对 #4629 的评审（10-29，实测开始后才看到）

functionstackx（核心维护者）CHANGES_REQUESTED："No patches to sglang engine, it doesn't make sense for AMD to have patches to sglang engine considering that AMD has merge perms on sglang engine"；
另在安装脚本上建议 "put this wheel on pypi plz"。补丁加 waiver 的路线因此不可行：B1/B2/B6/B7/B8 需要先合进 SGLang、换镜像。
实测仍然带补丁跑，用来验证其余部分，并证明这 5 个修复够用。如何回复维护者、是否撤掉已推送的补丁提交，待用户决定。

## 1. 实测方式

### 1.1 为什么能在本集群上"按 CI 方式"跑

AMD 内部 fork `AMD-ROCm-Internal/InferenceY`（`AGENTS.md`："a local fork used to reproduce and debug InferenceX CI on the smci355 cluster"）
在本集群 n07–n09（`Compute-Group01`，另一个 Slurm 控制器，我们不可见）上挂了 6 个 GitHub runner（`/perf_apps/duwang/actions-runner-emu-0{4..9}`），
`clusters.mi355x-amds` 被改写成 smci355 的记录：ionic_0–7、`recipe-replace` 把上游 recipe 的 `rdma*` / `eno0` 换成本地名、pyxis 容器、
`hooks/mi355x-amds/setup.sh`（RDMA 预检、GPU 显存清空等待、hugepage）、`/apps/data/models` 共享权重。它还带 `runners/manual_launch.py`：
不经 runner 手动跑一个 matrix 点，runner 以下（squash 导入、recipe 绑定、合成接受率注入、srtctl 提交、`srt_agentic.sh` 客户端、收集与归档）与 CI 完全相同。

我们的做法（测试树 `/home/liyingli/code/bench_agentx/infx-aus`，本地 git，不推送）：

| 提交 | 内容 |
|---|---|
| 1 | InferenceY `cde1f5adb` 工作区快照（runner emu-04） |
| 2 | 叠加 #4629（`c9560348`）的 diff（perf-changelog / patches README 两处冲突与测试无关，未带） |
| 3+ | 叠加 #4629 分支上后续的修复提交（与推送到 PR 的内容逐字相同） |
| TEST-ONLY | 只为在本集群上跑：分区 `Compute-Group01` → `Compute-DCPT`；squash / HF 缓存 / aiperf 缓存 / 归档目录改到 `/apps/liyingli`；host 网卡 `enp81s0f1` → `fenic`（Compute-DCPT 节点上没有 enp81s0f1）；recipe 写死的上游节点本地 HF 缓存 `/var/lib/hf-hub-cache` 经 `recipe-replace` 换成 `/apps/liyingli/cache/hf-hub-cache`；Infera 用本地 wheel（`--find-links /configs/infera-wheels`）且 master 的 `router.version` 改成 `0.3.3.dev5`（正式版发布后换回 GitHub release） |

srt-slurm 固定 `ae3d6877`，启动时依次打 `573-participating-gpus.patch`、`630-infera-frontend.patch`、`node-exporter-host-native.patch`（与 CI 相同的 `checkout.py` 流程）。
提交作业时 `SBATCH_QOS=batch`（我们能用的最高 QoS）。

### 1.2 Infera wheel

- #222 分支 merge 最新 main（含 #223 的打包方式）→ `f668fbf7`，已推送。计算节点上单元测试 `tests/unit/engine`、`kv`、`server` 566 passed / 1 skipped。
- 按 #223 的 release 流程本地打包（`scripts/build_infera.sh`，计算节点）：`python -m build` → `build_router.sh`（manylinux_2_28 容器，Rust 1.99.1）→
  `bundle_router_wheel.sh`（auditwheel 打 `manylinux_2_28_x86_64` 标签）→ 每个 wheel 单独 venv 安装检查，`_find_binary()` 解析到 wheel 内置的 router。
- 产物：`amd_infera-0.3.3.dev5-py3-none-manylinux_2_28_x86_64.whl`（sha256 `1b64f07f…`），版本号由 setuptools_scm 从 git 生成。

## 2. 对 #4629 的修改（已在本地提交，实测通过后推送）

| 文件 | 改动 | 原因 |
|---|---|---|
| `configs/infera-sglang.sh` | 装完 Infera 后执行 `infera-sglang-patches/apply.sh`；router 检查改为 `launch_rust._find_binary()` | 补丁原先没有接入；#223 之后 router 在 wheel 的 `infera/_bin/` 下而不在 PATH 上，原来的 `shutil.which` 检查会失败 |
| `configs/infera-sglang-patches/` | B1/B2/B6/B7/B8 五个补丁 + `apply.sh`（锚点匹配、可重复执行、逐个核对标记，任一失败作业失败） | 原样镜像缺这 5 个修复（源码逐行核对见 waiver） |
| `docs/waiver/4629.md` | 按 #2508 的格式：涵盖配置、改了什么、上游 PR / issue、原样镜像为什么跑不了、移除计划 | checklist 要求 |
| recipe | 加 `MOONCAKE_DISABLE_HIP_DMABUF: "1"`；`extra_mount` 只读挂宿主机 ionic provider | dma-buf 按同集群 ATOM PD 先例（同一 Mooncake `78566d45`、同为 1 GiB MR）定为 1，不实测；libionic 照 Kimi-K3 vLLM 的做法 |
| `infx/tests/srt_slurm/test_setup_scripts.py` | 安装脚本测试跳过补丁步骤（`INFERA_SGLANG_PATCHES=0`），新增"补丁打不上则安装脚本失败"的测试 | |

## 3. 结果

（待补）

## 4. 时间线

- 10-29 12:54 #222 merge main、单测、打 wheel（计算节点 n05-33，作业 34908）；12:58 推送 #222。
- 10-29 13:00–14:20 搭测试树（InferenceY 快照 + #4629 + TEST-ONLY 适配），补丁脚本与 waiver。
- 10-29 14:26 节点探测：Compute-DCPT 节点 host 网卡是 `fenic`（不是 InferenceY 节点的 `enp81s0f1`），无 `/var/lib/hf-hub-cache`；
  ionic_0–7、宿主机 provider `libionic-rdmav34.so → libionic.so.1.1.39`、`/etc/slurm/task_prolog.sh`、pyxis 都在。
- 10-29 14:38 拉起功能点 1（4+4 C80，`--fast`）。
