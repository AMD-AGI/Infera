# 上游 PR / issue 计划（11-18，主线整理；11-19 起 upstream agent 推进，见第 0 节）

## 0. 11-19 进度（upstream agent）

代码在仓库外的 `/home/liyingli/bench_agentx/upstream-20261119/`：`vllm/`（blobless clone）、`infera/`（从本仓库克隆，`origin` 指向
AMD-AGI/Infera），各分支的工作树在 `wt/`、`iwt/`。vLLM 每项两个分支：基于 v0.34.0（与实验镜像一致），以及基于 main `9c1d45c2`
（与 `vllm/vllm-openai-rocm:nightly-9c1d45c2…` 镜像同一提交，测试时编译好的算子对得上）。Infera 每项一个分支、一个带签名的提交，
用只含 `infera/`、`rust/`、`tests/` 的补丁从 `origin/main` 拆出（不带 `llying/`）。**还没有任何 PR / issue 提交到 GitHub**（先给用户 review）。

| 编号 | 类型 | 内容 | 分支 | 状态 |
|---|---|---|---|---|
| U26 | fix | ROCm CPU offload 张量精确锁页（`CudaRTLibrary.cudaHostAlloc` / `cudaFreeHost`，随最后一个引用释放） | vLLM `u26-rocm-exact-pin(-main)` | 单测通过：v0.34.0 226 / 基线 223，main 254 / 251；P4D4（J3，PR 代码）每 worker 474.5–474.9 GB 精确锁页，计分中 |
| U12 | opt | gfx950 gluon paged-MQA logits（sglang #47897，作者 yihou@amd.com，AMD MIT） | vLLM `u12-gfx950-gluon-mqa-logits(-main)` | 单测 main 126 / 108、v0.34.0 72 / 54 通过（测试曾误把 PyTorch 参考回退当对照，已改为强制 aiter）；P4D4 decode 已确认走 gluon；PR 说明 `PR-U12-…md`（11-19 第一批，待用户 review） |
| U13 | opt | FULL graph 捕获时放行 AITER AG/RS | vLLM `u13-aiter-agrs-full-capture(-main)` | 单测通过：v0.34.0 16 / 10，main 16 / 10（首轮因缺 `ray` 未收集，已修判定）；P4D4 已确认走 AITER AG/RS |
| U11 | fix | PP 下按最大 stage 分槽（新字段 `compare=False`，否则十几处 `KVCacheConfig` 相等断言会失败） | vLLM `u11-offload-pp-worker-bytes(-main)` | v0.34.0 单测 1,633 / 1,629 通过（1 个失败原版也有）；main 上新测试需用 `full_layer()`（RFC #42449 接口变化），已改，重跑中 |
| U19 / U20 | fix | Tiering 整块锁页 / Simple 页对齐 | vLLM `u20-simple-offload-page-align(-main)`；U19 未建 | U20：只在紧凑布局切不开时才按页对齐（不影响能工作的配置），main 上 PR 轮测试进程在清理阶段崩溃，原因待查（已改为保存完整日志重跑）；U19：J1b 测量中 |
| U16 | feature | vLLM Mooncake 串行下发（Python + Rust，补了 Python 测试） | Infera `pr/vllm-mooncake-serial` | CI 通过（Python 3,055 / 3,054；Rust 370 / 369，fmt / clippy 干净：首轮抓到多余的 `mut` 和一处格式，已修） |
| U24 | fix | Rust 路由连接池空闲超时 | Infera `pr/router-pool-idle` | CI 通过（Rust 370 / 369）；本机回环竞争检查新旧设置都是 0 错误（未复现，见下），繁忙客户端变体与 J3（无 keep-alive 绕法）进行中 |
| U14 / U15 | fix / opt | nofile / 分词移出事件循环 | Infera `pr/server-nofile`、`pr/kv-aware-prepare` | CI 通过（Python 3,058 / 3,057 对 main 3,054） |
| 新 | feature | `INFERA_VLLM_DP_ENGINE_LB=1` | Infera `pr/vllm-dp-engine-lb` | CI 通过（3,058）；vLLM 0.34 容器内 4 / 4；J3 P4D4 3600 s 计分中 |
| 新 | fix | vLLM ≥ 0.34 的 `make_arg_parser` 挪位（Infera main 仍 import 旧位置，起不来） | Infera `pr/vllm-034-arg-parser` | CI 通过（3,056）；vLLM 0.34 容器内：main 3 / 3 失败（`No module named vllm.entrypoints.openai.cli_args`），本分支 5 / 5 通过 |
| U2 | issue | ROCm clr `deviceId`（B 线最小复现） | — | J1 复跑后写 issue |

**跨平台影响（11-19 用户要求：尽量只影响 ROCm；无法避免的记下来，需要在 CUDA 上测）**

| 编号 | 对 CUDA / 其他硬件的影响 | 处理 | 需要 CUDA 测试 |
|---|---|---|---|
| U12 | 无：只在 gfx950 且开关打开时生效，代码在 ROCm 专用模块 `rocm_aiter_mla_sparse.py` 和新文件里 | — | 否 |
| U13 | 无：改在共用的 `CudaCommunicator._can_use_aiter_ag_rs`，但 `use_aiter_ag_rs` 只在 ROCm + AITER 时为真，CUDA 第一行就返回 False | PR 说明里写明 | 否（CPU 上的 mock 单测覆盖判断逻辑） |
| U26 | 已改为无：精确锁页只在 `is_rocm()` 时启用；`CudaRTLibrary.cudaHostAlloc` / `cudaFreeHost` 改成用到时才解析（不进 `exported_functions`），CUDA 进程不会查找这两个符号 | 11-19 12:00 修改，`v034-u26b` / `main-u26` 复测 | 否 |
| U20 | 已改为对能工作的配置无影响：先按原来的紧凑布局算锁页分块，只有切不开（现在会起栈失败）时才把每层起点按页对齐；这条逻辑与平台无关（CUDA 的 Simple offload 同样会触发） | 11-19 12:00 修改 | 建议：CUDA 上 Simple offload 的现有测试（`tests/v1/simple_kv_offload/`、`test_simple_cpu_offload_connector.py`）跑一遍；只有触发回退的配置行为才变 |
| U11 | **无法避免**：这是通用的正确性问题（CUDA 上 PP > 1 + `TieringOffloadingSpec` 同样起不来，`CPUOffloadingSpec` 的槽宽也按本 worker 算），修复会让 PP > 1 时较小 stage 的槽宽变成最大 stage 的值（多用 2–7% 主机内存） | 新字段 `compare=False`；单测是纯 CPU 的 | **是**：CUDA 上 PP2 + `TieringOffloadingSpec` 起栈与读回、PP2 + `CPUOffloadingSpec` 读回；`tests/v1/kv_connector/unit/offloading_connector/`、`tests/v1/core/` 在 CUDA CI 上跑 |
| Infera 各项 | 与 GPU 硬件无关（路由、启动、参数解析） | — | 否 |

**执行方式（11-19，领导要求不 hold 节点）**：按预热类型拆成跑完即退的作业，每个作业执行一份计划（`plans/<名>.plan`，每行一步：
依赖、超时、失败策略），步骤脚本在 `steps/`，结果在 `results/<步骤>/`（`log`、`rc`、`verdict.txt`、失败时 `diag/`）。
已通过且 key（步骤脚本 + 所测分支的提交号）未变的步骤，重新提交时自动跳过；失败只跳过依赖它的步骤，独立步骤照常跑；计划在每一步之间重读，
作业忙时可以追加步骤，计划跑完最多再等 10 min（用户 11-19 选定，待领导确认）就退出。准备工作只按需做：官方镜像 + 挂载改动文件（不编译），
模型拷到 NVMe 实测 21–109 s，Infera 实验镜像在新节点上约 5 min。J1（单测 / 最小复现，可续跑）用 burst QoS（11-19 11:30 agentx 10/10、
primus-cicd 11/11 被 12–24 h 作业占满，分区另有 39 台空闲）；J3（P4D4 计分，不能被抢占）排 agentx。脚本副本在本目录 `kit/`。

- J1 = 作业 298936（11:34 起，v3-057，burst）：vLLM 各项 v0.34.0 单测 + main 对比（11:45 起按用户意见改：nightly `9c1d45c2` 镜像里把
  vLLM Python 代码同步成 main `f0c592bfd` 作基线、main + PR 作对比，只保留镜像的 `.so`；两提交之间只改了 `csrc/cpu` 和 flash-attn
  cmake，步骤开跑前自动检查），U26 主机内存、ROCm clr 复现。J1b = 作业 298969（12:17 起，v3-006，burst）：Infera CI、U24 竞争检查、
  U19 测量。J3 = 作业 298999（12:43 起，v3-191，agentx）：P4D4 用 PR 代码 3600 s（两次分到坏节点：v3-198 docker run 失败、
  v3-226 无 docker 权限，已加入 `kit/jobs/exclude_nodes.txt`）。
- **J3 结果（P4D4，v0.34.0 + PR 代码，无 keep-alive 绕法，3600 s，v3-191，单次）**：43,269 tok/s/GPU，2.985 req/s，完成 / 丢弃 10,831 / 12，
  交互 p50 / p90 60.3 / 41.7，ITL 16.6 / 24.0 ms，TTFT p50 / p90 / 均值 3.57 / 17.84 / 8.48 s；**路由 "error sending request" / "unreachable" / 5xx
  全程 0 次**（U24，单次）；日志确认 gluon、AITER AG/RS、每 worker 474.5–474.9 GB 精确锁页都生效；无 OOM / 驱逐 / 崩溃。吞吐比 m10（48,572，
  v3-006）低 11%，比 f2（44,691，v3-124）低 3%：节点首次使用、单次，节点间差异以前就有 7–10%，不能归因于 PR；decode 仍受 KV 限制（71% 的采样
  有 rank KV > 95%）。详见 `results/pr-298999/pr-checks.txt`。
- 测试本身踩过的坑（已修，结论以修后为准）：U12 测试容器没设 `VLLM_ROCM_USE_AITER=1`，对照其实是 vLLM 的 PyTorch 参考回退
  `fp8_paged_mqa_logits_torch`（差 2⁻¹⁰ 级；11-19 更正：之前写成"Triton 回退"）；对 aiter 只差 7.63e-6（logits 最大 56–63，约 2 个 fp32 ulp），与 A 线一致；U13 首轮两边都缺 `ray`、一个测试也没收集到却判了通过（对比脚本已改为收集失败
  或零通过即失败）；`infera-v034` 首轮判定取了 `tail` 的退出码、容器没挂 GPU。
- U24 本机回环竞争检查（`kit/steps/u24-race.sh`，uvicorn + 按路由设置构建的 reqwest，64 并发，间隔落在 keep-alive 附近，单次）：
  keep-alive 1 s 时池空闲 90 s / 0.8 s 各约 7,700 个请求，keep-alive 5 s 时 90 s / 4 s 各约 3,870 个请求，全部 0 错误，**空闲的本机
  环境下复现不出旧设置的竞争**；正在跑"客户端繁忙"变体（1 个工作线程、每个响应后占用 10 ms CPU）。U24 的主要证据仍是 s10 / s11 的
  路由日志与 J3。
"本分支"指 `dev/pd_opt/vllm_glm52_agentx`；Infera 的 PR 目标是 Infera 主线，从本分支按条目拆出（每条只含自己的提交）。

## 1. vLLM

| # | 内容 | 材料 | 验证 | 还差 |
|---|---|---|---|---|
| U12 | gfx950 gluon paged-MQA logits（移植 sglang #47897），组大小由 indexer decode metadata 传入 | `../vllm-decode-trackA-20261118/PR-U12-gfx950-gluon-paged-mqa-logits.md`、`overlay-v3/rocm_aiter_mla_sparse.py.diff`、kernel 文件 `../vllm-kf-port-20261117/overlay/vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py` | 函数级测试（含 CUDA graph 重放）；聚合 DP4 MTP3 纯 decode 每步 −2.7% 至 −3.9%（3 次）；P4D4 s13（覆盖层 v2）42,724 | 开关登记进 `vllm/envs.py`；kernel 文件保留 AMD MIT 头并注明来源 sglang #47897（许可需确认）；P4D4 上 v3 的端到端（m2） |
| U13 | aiter 自定义 AG/RS 在 FULL graph 捕获时放行（`mode == NONE and is_current_stream_capturing()`） | `../vllm-decode-trackA-20261118/PR-U13-aiter-ag-rs-full-capture.md`、`overlay-comm/cuda_communicator.py.diff` | 聚合模式每步 −2.7%，与 U12 合计 −6.3%，门禁 196/200 | P4D4 端到端（m2）；更彻底的改法（runner 捕获时把 forward context 设成 FULL）可在 PR 讨论里提出 |
| U26 | ROCm 上 CPU 层张量精确锁页（匿名 mmap + `host_register`），不经按 2 的幂取整的 torch 锁页分配器 | `PR-U26-rocm-exact-pinned-cpu-offload.md`、`../vllm-p4d4-crusoe-20261117/overlay-cpupin/gpu_worker.py.diff` | 最小测试（2 × 20 GB：+64 GiB 对 +37 GiB）；P4D4 m10 1.9 TB 实际 1.90 TB，3600 s 48,572 | 单测；开关登记进 `envs.py` |
| U11 | `KVCacheConfig.max_worker_bytes_per_block` 跨 worker 取最大，修 PP > 1 时 `TieringOffloadingSpec` 区域大小 / 槽位不一致 | `../vllm-prefill-trackB-20261118/results/b2/vllm-offload-pp.diff`（3 文件） | 聚合 PP4 读回 9/9（修前复现超时）；P4D4 4 个 PP worker 加入同一共享区 | 单测（vLLM `tests/v1/kv_connector/` 下按 PP 构造不同层数的 KVCacheConfig） |
| U19 | ROCm 上 `TieringOffloadingSpec` 每个 worker 锁页整块共享区 | B 线 README 2 节（日志） | b2-r1 日志 | 写最小修法 |
| U20 | `SimpleCPUOffloadConnector` 的 `num_cpu_blocks` 页对齐 | B 线 README 3.5 节 | 绕法 `num_cpu_blocks=389120` | 写修法 + 单测 |
| U17 | EP + AITER 融合共享专家 + AG/RS 分发结果错误 | `../vllm-decode-trackA-20261118/overlay-ep/` | 修后 196/200（修前 1/200） | 作为 issue + PR |
| U18 | MoRI 接入：接收缓冲区切片、IntraNode 零拷贝 combine | `../vllm-decode-trackA-20261118/overlay-mori/` | 63.56 → 54.85 ms/步 | 多节点 EP 场景再验证 |

## 2. Infera（本仓库）

| # | 提交（本分支） | 测试 | 验证 | 还差 |
|---|---|---|---|---|
| U16 Rust 路由 vLLM Mooncake 串行下发（`INFERA_VLLM_MOONCAKE_SERIAL=1`） | `4834e0183`（实现）、`e25f54c1e`（测试） | `disagg::tests::vllm_serial_pull_posts_the_decode_after_the_prefill_reply`：prefill 未回复前 decode 不会收到请求；prefill 失败时 decode 永远收不到 | P4D4 s10 = 36,265、s13 = 42,724 都用此路径 | 开关目前是环境变量（与 Python 后端一致），上游可能希望做成 CLI 参数 |
| U24 Rust 路由上游连接池空闲超时（默认 4 s，`INFERA_UPSTREAM_POOL_IDLE_S`） | `7167dffea` | `proxy::tests::pool_idle_defaults_below_uvicorn_keep_alive`（解析）；连接复用竞争本身是时序问题，没写确定性测试 | 待测：用 `bin/infera-router-vllmserial-u24`、去掉两边 `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120` 跑一轮，看是否还有 502 | 一轮验证 |
| U14 路由启动时把 `RLIMIT_NOFILE` 软上限提到硬上限（在选择后端之前，Rust 路由经 exec 继承） | `a7481d70f` | `tests/unit/server/test_nofile_limit.py`（4 例） | 部署脚本此前用 `--ulimit nofile=1048576` 绕过（s7b 起） | — |
| U15 Python 路由 kv-aware 策略在事件循环里同步分词 | 见本条提交（`Policy.prepare()` + `prepare_pick`；kv-aware 策略在单线程池里分词 / 哈希，`pick` 复用；结果经 `ContextVar` 按请求对象身份传递，不写进转发的请求体） | `tests/unit/router/test_kv_event_aware_policy.py` 新增 3 例（prepare 在非事件循环线程哈希、pick 不再哈希；结果只属于同一个请求对象；不调 prepare 时照旧）；`tests/unit` 2,707 通过（gaie 目录在本镜像里缺 grpc 依赖，未跑） | s8 py-spy：87% 在 `tokenizer.encode` | 线程池默认 1 个线程（共享 tokenizer 不能多线程同时用，U21）；多线程并行需要每线程一个 tokenizer 副本，后续可做 |
| U2 GEM 分配器改用 HIP VMM 导入 | `ea8aa2b9f`（B 线，`llying/vllm-image-v0340-20261117/gem/hip_gem_alloc.c`） | B 线隔离测试 `scripts/swap_dev_test.py`、聚合复现 | P4D4 门禁 + b1-r1 = 33,203 | 分配器目前在 llying 目录、随实验镜像构建；进 Infera 主线要放到 `deploy/` 下并接进正式 Dockerfile |
| — 路由逐请求 prefill 命中日志（`INFERA_LOG_PREFILL_USAGE=1`） | `c12af42e9` | — | 诊断用 | 是否进上游待定（诊断开关） |

- 测试命令（在镜像里，`rust/` 只读挂载）：`cargo test --release --lib -p infera-router`；`python3 -m pytest -q tests/unit/server/`。
- 已知与本改动无关的间歇失败：`metrics::tests::ttft_series_follow_the_prefill_worker`（全局 Prometheus 注册表在并行测试之间串扰；完整跑 3 次失败 2 次，单独跑通过）。

## 3. ROCm / Mooncake

| 目标 | 内容 | 材料 |
|---|---|---|
| ROCm clr | `hipExternalMemoryGetMappedBuffer` 不设 `deviceId`，`hipMemcpyBatchAsync` 把非 0 设备上导入内存的拷贝放到 device 0（U2 的根因） | B 线 `scripts/batch_ext_repro.c`（不依赖 GEM 的最小复现）、`results/b1/batch_ext_repro.txt` |
| ROCm 驱动 / 运维 | 锁页 CPU 层的进程异常结束后约 1.27 TB `Shmem` 不归还（U23） | B 线 `results/b1/control-failures/` |
| 节点运维 | v3-106：NVMe 卷在 dockerd 启动后才挂到 `/var/lib/containerd` 上，docker 拉取 / 构建 / 起容器全部失败 | 主线 PROGRESS D30（需要报给集群管理员） |
