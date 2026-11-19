# 交接：上游 PR / issue（upstream agent，11-19 起）

目标：把 GLM-5.2 PD（Infera + vLLM v0.34.0 ROCm）**最佳 recipe 里用到的、以及试过的策略里涉及的** feature、优化和 fix，整理成可以提交的上游 PR / issue
（vLLM、Infera、Mooncake、ROCm）。每一项要有：最小改动、测试、验证数据（单次要注明）、PR 说明。主线 agent 同时在跑计分优化，交接见
`../vllm-pd-progress-20261117/HANDOFF-main.zh-CN.md` 第 0.00 节。

## 1. 先读

- 本目录：`CANDIDATES.zh-CN.md`（U1–U26：问题、证据、修法、目标、状态）、`PR-PLAN.zh-CN.md`（每项代码在哪、测试、验证、还差什么）、
  `PR-U12-gfx950-gluon-paged-mqa-logits.md`、`PR-U26-rocm-exact-pinned-cpu-offload.md`。
- A 线（decode kernel / 通信）：`../vllm-decode-trackA-20261118/`（U12 / U13 / U17 / U18 的 PR 草稿与覆盖层）；B 线（prefill GEM / offload）：
  `../vllm-prefill-trackB-20261118/`（U2、U9、U10、U11、U19、U20、U22）。
- 主线结果：`../vllm-p4d4-crusoe-20261117/README.zh-CN.md` 第 0 节和 1o、1r、1s、1t（精确锁页、decode 故障、P8D8 跨节点失败、f1 / f2）。

## 2. 最佳 recipe 用到了什么（P4D4，m10 / m14 / f1）

| 部分 | 用到的 | 上游归属 |
|---|---|---|
| decode kernel | gfx950 gluon paged-MQA logits（移植 sglang #47897，组大小来自 indexer metadata） | vLLM U12 |
| decode 通信 | aiter 自定义 AG/RS 在 FULL graph 捕获时放行 | vLLM U13 |
| prefill CPU 层 | ROCm 上 CPU offload 张量精确大小锁页（现为直接 `hipHostAlloc`；第一版 mmap + `hipHostRegister` 已弃用，原因见 README 1r / 1s） | vLLM U26 |
| prefill GEM KV | GEM 分配器改用 HIP VMM 导入（根因是 ROCm clr 导入外部内存不设 `deviceId`） | Infera U2（部署）+ ROCm clr issue |
| 路由 | Rust 路由 vLLM Mooncake 串行下发、上游连接池空闲超时（防 502）、`RLIMIT_NOFILE` 提升 | Infera U16、U24、U14 |
| decode 选 rank | `INFERA_VLLM_DP_ENGINE_LB=1`：Infera 不再指定 DP rank，交给 vLLM 内部负载均衡（提交 `209a4bd88`，`infera/engine/vllm/args.py` 的 `_dp_identity`） | Infera（新条目，尚未编号） |
| 其余 | `--renderer-num-workers 4`、keep-alive 120 s、缓存感知准入、Mooncake 环境变量：都是现有选项，不需要 PR | — |

试过但没进最佳配置、仍值得上游的：U11（`TieringOffloadingSpec` PP > 1）、U19 / U20 / U22（offload 相关）、U15（Python 路由同步分词）、U25（重复渲染 chat 模板）、
U17（EP + AITER 融合共享专家结果错误）、U18（MoRI 接入慢）、U9 / U10（写穿 / CPU 层容量，设计讨论类）。

**新发现、需要先做最小复现再开 issue 的**（都只观察到一两次）：
- Mooncake（U27）：P8D8 跨节点时 `Notification QP initialization failed: -302`（`ERR_MEMORY`）1,272 次，同节点回环为 0，单进程跨节点测试不复现，怀疑是 8×8 worker 全互连时
  通知 QP 资源不够；**两边 `MC_RDMA_NOTIFY_ENABLED=0` 的绕法已验证**（11-19 重试 smoke 通过、预热 2,302 个请求 0 错误，单次）；镜像里的 Mooncake 比 `faae8dd4` 新（有 `MC_RDMA_NOTIFY_*`），要先确认版本和源码。另外 vLLM 镜像的 Mooncake 没打 dest-local-rail / ACK 超时
  两个补丁（Mooncake#4540 / #4541，目前只进了 SGLang 镜像）；8 张网卡同时用时主机内存跨节点只有约 4 GB/s（单卡 30 GB/s）。工具：主线的 `scripts/mc_xnode.sh`。
- decode 在显存目标 0.95 时 hipBLASLt `INTERNAL_ERROR` → `HSA_STATUS_ERROR_OUT_OF_RESOURCES` 崩溃（m9b、m13）、变慢后卡死（m12，伴随 amdgpu `evict_process_worker`）：
  怀疑 ROCm 上 vLLM 的显存预算没有给 torch 池外的分配（scratch、aiter AG/RS 缓冲区等）留够余量。
- ROCm：锁页 CPU 层进程异常结束后 `Shmem` 归还慢（U23，已部分更正：一次 2.5 min 内归还）。

## 3. 交付顺序（建议）

1. **U26**：按 `PR-U26-...md` 的新版（`hipHostAlloc`）。上游做法是给 `CudaRTLibrary`（`cuda_wrapper.py`）加 `cudaHostAlloc` / `cudaFreeHost`（HIP 映射
   `hipHostMalloc` / `hipHostFree`），加单测（总大小不取整、`is_pinned()`、`swap_blocks` 往返），开关登记进 `vllm/envs.py`。
2. **U12 + U13**：从 A 线草稿出发；U12 的 kernel 来自 sglang #47897，要确认许可和来源注明。
3. **Infera**：U16、U24、U14、U15、DP 引擎均衡开关，从本分支按条目拆成独立 PR 分支（每条只含自己的提交和测试；提交号见 `PR-PLAN.zh-CN.md` 第 2 节）。U24 还差一轮验证。
4. **U11（+ U19 / U20）**：offload 修复和单测。
5. **issue**：ROCm clr `deviceId`（B 线已有不依赖 GEM 的最小复现）、Mooncake 通知 QP、decode 显存余量崩溃。

每一项在本目录写 `PR-<编号>-<短名>.md`（PR 说明正文用英文，其余用中文），并更新 `PR-PLAN.zh-CN.md` 和 `CANDIDATES.zh-CN.md` 的状态。
**在 GitHub 上真正开 PR / issue 之前先给用户 review**（对外操作）。

## 4. 规则

- 仓库 `/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52`，分支 `dev/pd_opt/vllm_glm52_agentx` 多个 session 共用，push 前 `git pull --rebase`，`git commit -s`。
  本 agent 在仓库里只写 `llying/upstream-vllm-20261117/`；`infera/`、`rust/` 源码要改时先在主线 PROGRESS 里写一行，改完带测试提交。
- 上游代码放在仓库外的独立工作目录（建议 `/home/liyingli/bench_agentx/upstream-20261119/`：vLLM clone 基于 v0.34.0 写、再 rebase 到 main；Infera / Mooncake 同理），
  不要在共用分支上做 PR 分支。
- 主线的工作目录 `/home/liyingli/bench_agentx/vllm-pd-20261117/`（scripts、overlay*、runs、queue*、once、p8d8-conf）和仓库里的
  `llying/vllm-p4d4-crusoe-20261117/`、`llying/vllm-pd-progress-20261117/`、`llying/vllm-pd-tools-20261117/` **只读**。不要碰 `queue/`、`queue-n2/`。
- **机器**：不允许 hold 作业占机器，只用跑完即退的作业。需要 GPU 的测试写成 `once/up-<名字>.sh`，用主线的
  `sbatch -x "$(paste -sd, /home/liyingli/bench_agentx/pd-repro-20261116/exclude_nodes.txt)" -J lyl-up-<名字> -t 02:00:00 --export=ALL,ONCE=up-<名字> run_once.sbatch`
  （容器名以 `up-` 开头）。和主线共用每人约 2 个作业 / 2 台节点的配额，**主线优先**：最多同时 1 个单节点、≤ 2 h 的作业，提交前看 `squeue -u liyingli`。
- 登录节点只跑 `scripts/monitor.sh`（加编辑文件、git）；构建、测试、读大文件都在作业里做。spur：每个节点只能有一个进入作业的会话、作业步骤非零退出会结束作业、
  不注入宿主机 libionic、不改正在执行的 bash 脚本。
- 结论要有数据支撑，单次要注明；发现之前的结论有误，直接在文档里更正；文档用中文。
