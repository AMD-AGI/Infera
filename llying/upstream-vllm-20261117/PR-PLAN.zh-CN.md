# 上游 PR / issue 计划（11-18，主线整理）

按 `CANDIDATES.zh-CN.md` 的优先顺序列出每一项的现状：代码在哪、测试、验证数据、还差什么。所有 P4D4 数据都是单次运行。
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
