# 把 infera + sglang 的 GLM-5.2 AgentX Pareto 发布到 InferenceX：还差什么、怎么做、风险与工期（10-27）

对象：[Infera PR #210](https://github.com/AMD-AGI/Infera/pull/210) 的第一版 Pareto（MI355X，GLM-5.2-MXFP4，PD，PP4 prefill + DPA decode，7 个点）。
本报告只做分析，没有上机。InferenceX 的规则、流程和先例取自 `SemiAnalysisAI/InferenceX` main `7ca692b9`（10-27）、其固定的
`NVIDIA/srt-slurm` v2.59.1（`ae3d6877`），以及 10-27 拉取的 InferenceX API（`evidence/inferencex-api-glm52-20261027.json`）。

## 0. 结论

1. **InferenceX 不收外部数字，只发布它自己 CI 跑出来、经 PR 评审合入的结果。**
   "发布"的实际含义是：向 `SemiAnalysisAI/InferenceX` 提交一个 recipe PR，由 SemiAnalysis 的 `mi355x-amds` 集群（10 节点，Slurm + enroot，经 srt-slurm 拉起）
   用上游镜像重跑，sweep 全绿、eval 通过、AMD CODEOWNER 签 checklist、核心维护者批准后合入，合入时摄取这次 sweep 的结果。
   PR #210 的数字是在 crsuse2 上用 docker、自建镜像、手工脚本和 `918524ff` 客户端测的，**不能直接上榜，只能作为预期**。
2. **五道硬门槛**，按难度排序：
   - **镜像必须是上游 `lmsysorg/sglang-rocm`，不能打补丁。** 我们的镜像烘焙了 9 个 SGLang 补丁、重编了 Mooncake、换了 libionic、装了 infera。
     每个补丁要么在上游合入并等到含它的 nightly，要么在 PR 里逐项写 waiver（`docs/waiver/<PR>.md`）。
   - **必须由 srt-slurm 拉起。** srt-slurm 没有 infera 这种 frontend，需要新增一个（上游 PR，或先在 InferenceX 里带补丁）。
   - **生产配置要先进 SGLang cookbook，且上游 PR 要在 InferenceX PR 之前 MERGED**，包括 prefill、decode 和 router 的启动命令。
   - **在 SemiAnalysis 集群上跑稳。** 我们有两个未解决的问题会直接伤到共享 CI：prefill GPU 内存访问错误（之后要管理员重置 GPU），以及开 HiCache 撤栈后显存 10–65 分钟才释放。
   - **评审流程**：AMD CODEOWNER（`@seungrokj`、`@chunfangamd`、`@billishyahao` 等）签最新版 checklist，核心维护者批准，维护者发 `/use <run_id>`。
3. **仪表板的 Pareto 口径已经变了。** 现在的标准前沿以 **E2E normalized interactivity P90** 为横轴（逐请求 E2EL / OSL 的 p90 取倒数，含 TTFT），
   不再是 PR 里用的 decode interactivity p90。按新口径重算（第 2 节），1P1D C88 被 4+4 C80 支配，其余 6 点仍在前沿（满足"每条曲线 ≥ 5 点"的建议）；
   横坐标从 43–112 变成 14–51，**PR 里"比 MI355X 线高 1.4–2.2 倍"的对比需要按新口径重做**。PP4 prefill 饱和时的 TTFT 长尾在新口径下被直接计入。
4. **工期**：按 1–2 人投入，关键路径约 **7–9 周**；乐观 5–6 周（waiver 被接受、集群上一次跑通），悲观 3 个月以上（上游合入或 cookbook 卡住、集群问题反复）。
   最长的两段都依赖外部：引擎去补丁（上游 SGLang 评审），以及在 SemiAnalysis 集群上调通（只能经 CI 迭代，每轮以天计）。
5. **建议分两期**：
   - 第一期只发单节点 8 卡的 4+4（1 个 PP4 prefill + 4 卡 decode，同机），并发从约 C24 到 C80，凑够 5 个以上点。
     单节点免掉跨节点 RDMA 和两 prefill 同机的问题；按新口径 4+4 C64（24.6）和 1P1D C64（25.4）几乎重合，4+4 在较低并发下可能就能覆盖高交互端（需实测）。
   - 第二期用 `append-only` 追加 2P1D 和 1P1D（12 卡）。`append-only` 要求镜像不变，所以第二期最好赶在换镜像之前。
6. **需要你决定的事**见第 6 节：dashboard 上的 framework 名字、router 的接入方式、首发范围、KV 传输用 Mooncake 还是 MoRI。

## 1. InferenceX 的发布流程（与我们相关的部分）

```text
我们的 PR（SemiAnalysisAI/InferenceX）
  ├─ configs/amd-master.yaml               新条目：镜像、模型、framework、每个点的拓扑与并发
  ├─ benchmarks/multi_node/srt-slurm-recipes/glm5.2/sglang/mi355x-fp4/agentx/disagg-variants.yaml
  │                                         srt-slurm recipe 片段：base + 每点一个 override
  ├─ perf-changelog.yaml                   末尾追加一条（触发 sweep 与结果收集）
  ├─ （可能）infx/ 代码、srt-slurm 补丁、docs/waiver/<PR>.md
  └─ 标签 full-sweep-fail-fast → mi355x-amds 上每点一个 Slurm 作业：起栈 → AgentX 3600 s → GSM8K eval
        → 全绿 → AMD CODEOWNER checklist → 核心维护者批准 → /use <run_id> → merge_with_reuse 合入并摄取结果
```

- **每点全新部署**，与我们的做法一致；合成接受率由 driver 按 `infx/golden_al_distribution/` 注入（GLM-5.2 thinking_on K=5 → 3.61），recipe 里不能写死。
- **Eval**：每个拓扑在最高并发跑一次 agentic GSM8K（真实接受率），门槛默认 0.90。我们 PD example 实测 0.9666 / 0.9674。
- **AgentX 客户端**：用 InferenceX 当时 main 上的版本，不是我们固定的 `918524ff`。ATOM 的经验是客户端版本不同，数字会有差别。
- **会话亲和 header**：InferenceX 的 recipe 统一让 AIPerf 发 `X-Dynamo-Session-ID`（`AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID`），
  infera router 目前只认 `X-Session-ID`（`rust/router/src/session_affinity.rs`）。
- **最相近的先例**：
  - `dsv4-fp4-mi355x-sglang-disagg-agentic-umbp-dspark`（已合入）：同一集群上的 SGLang v0.5.22 PD agentic，上游镜像，`sglang-router` 作 frontend，MoRI 传 KV。
    可以照抄它的集群写法：`@fabric.*` 引用、`rdma0-7`、`host_setup`、`SGLANG_SIMULATE_ACC_*`。
  - [InferenceX #4045](https://github.com/SemiAnalysisAI/InferenceX/pull/4045)（ATOM PD，10-14 开，至今 OPEN）：
    带了一个 srt-slurm 补丁（PP 的 TP 计算）、cherry-pick 了缓存统计的 infx 改动。评审要求补 Mooncake 的缓存统计；bot 提示 infx 包改动（+479 行）超出常规。
    PR 自述 ATOM 自己的 CI 在另一集群 C112 跑到 46.1k，SemiAnalysis 集群只有 39.2k，**集群间差约 15%**。

## 2. 按仪表板口径重算 PR #210 的 7 个点

`scripts/e2e_norm_table.py` 从 `../2p1d-v0522-20261020/results/` 读 AgentX JSON（其中已有 `e2e_norm_intvty`）：

| 点 | GPU | tok/s/GPU | interactivity p90（PR 用的） | E2E normalized p90（仪表板） | TTFT p90 s | 新口径下在前沿 |
|---|---:|---:|---:|---:|---:|---|
| 2P1D C176 | 16 | 38,392 | 43.2 | 14.4 | 9.5 | 是 |
| 1P + 4 卡 decode C80 | 8 | 37,310 | 44.3 | 14.9 | 10.8 | 是 |
| 1P + 4 卡 decode C64 | 8 | 32,532 | 48.6 | 24.6 | 4.2 | 是 |
| 1P1D 均衡 C88 | 12 | 31,088 | 71.7 | 14.1 | 11.5 | **否**（被 4+4 C80 支配） |
| 1P1D 均衡 C64 | 12 | 28,981 | 86.0 | 25.4 | 5.9 | 是 |
| 1P1D 均衡 C48 | 12 | 22,331 | 103.9 | 39.2 | 3.0 | 是 |
| 1P1D 均衡 C32 | 12 | 16,309 | 111.7 | 51.2 | 2.3 | 是 |

- decode 按请求数均衡在旧口径下把 C88 的 p90 从 50.7 拉到 71.7，但新口径把 TTFT 摊进来，C88 的 TTFT p90 11.5 s 让它掉到 14.1。
  **新口径下真正的杠杆是 TTFT 尾部**：prefill 饱和时的排队，以及 shortest-prefill-first 把长的冷首轮一直往后推（主报告记录过 C88 下 ≥32k 新 token 的首轮 TTFT p90 224 s）。
- InferenceX API 不返回 E2E normalized 字段（仪表板从逐请求 trace 计算），所以现有 MI355X / B300 线在新口径下的位置要到仪表板上看，PR 里的倍数对比需要重做。
- 现有 MI355X SGLang 单机线（10-26 更新）最高为 C16 18.3k @ decode p90 51.5，我们各点在吞吐上仍明显高于它。

## 3. 差距清单

| 项 | 我们现在（PR #210） | InferenceX 要求 | 差距 |
|---|---|---|---|
| 镜像 | `infera-sglang:v0522-*`：上游 v0.5.22 nightly + 9 个 SGLang 补丁 + 重编 Mooncake + libionic + infera | 上游 `lmsysorg/sglang-rocm` 原样；补丁只能走 waiver | 大 |
| 拉起方式 | docker、`up.sh` / `worker.sh`、ssh 到 decode 节点 | srt-slurm recipe（Slurm + enroot，`--container-remap-root`） | 大 |
| router | `python -m infera.server --router-backend rust`，Rust 二进制只在自建镜像里 | srt-slurm 注册的 frontend 类型；按 master 配置的版本安装 | 中 |
| 引擎进程 | `python -m infera.engine.sglang`（以子进程拉起 SGLang，负责 etcd 注册、PD 启动探测、KV 事件转发） | srt-slurm 的 SGLang 后端直接起 `sglang.launch_server`；Dynamo 有自己的 worker 启动模式 | 中 |
| 合成接受率 | `SIM_ACCEPT_LEN=3.61` 手动设 | driver 注入，recipe 不能写死 | 小 |
| 会话亲和 | `X-Session-ID` | `X-Dynamo-Session-ID` 或 `X-Correlation-ID` | 小（router 改动） |
| KV 传输 | Mooncake，8 条 ionic，`MC_MAX_MR_SIZE=1 GiB` | 集群 rail 名 `rdma0-7`；AMD 的 SGLang PD 先例用 MoRI | 中 |
| 集群 | crsuse2 138 / 140，docker，有 shell | mi355x-amds，Slurm，**我们没有登录权限**，只能经 CI 调试 | 大 |
| 文档 | Infera README | SGLang cookbook 上游 PR 已 MERGED | 中 |
| 客户端 / 指标 | `918524ff`，decode interactivity p90 | main 上的客户端，E2E normalized p90 | 中 |
| 稳定性 | prefill 内存访问错误（未解决），HiCache 撤栈后显存释放慢 | 共享 runner，不能弄坏节点 | 大 |

## 4. 工作项

### W1. 对齐与分工（第 1 周开始，贯穿全程）

- **内容**：
  - 与 AMD 的 InferenceX 团队（`amd-master.yaml` 和 `mi355x-*` recipe 的 CODEOWNER）确认：谁提 PR、谁签 checklist、谁帮打 sweep 标签。
    fork PR 每次 push 都要维护者重新打标签，所以最好能在同一仓库开分支。
  - 与 SemiAnalysis 维护者确认三件事：infera 作为 SGLang 之上的部署层（类比 `dynamo-sglang`）是否接受；dashboard 上用什么 framework 名；waiver 能接受到什么程度。
  - 争取 mi355x-amds 上一到两个节点的交互访问用于首次调通，至少拿到失败作业的服务端日志。
- **风险**：维护者可能要求先合上游再提，或者不愿在 dashboard 上新增 framework 名。GLM-5.2 的主引擎是 SGLang（`docs/MODELS.md`），这一点对我们有利。
- **工期**：沟通 1 周，与 W2–W4 并行。

### W2. 引擎去补丁：在上游镜像原样运行（开发 1–2 周；等上游合入 3–6 周以上，不可控）

逐个判断补丁对本配置是否必需（v0.5.22 镜像，补丁清单见 `deploy/docker/patch.upstream.status.md`）：

| 补丁 | 作用 | 本配置需要吗 | 上游状态 | 路径 |
|---|---|---|---|---|
| `sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py` | HIP 上 DP padding 行数不一致 | 需要（decode DPA） | [sglang#33059](https://github.com/sgl-project/sglang/pull/33059) OPEN（7 月起） | 推合入；否则 waiver |
| `sglang_dsa/dsa_dp_sync.diff` | DP 分支上的 host 同步导致死锁 | 需要（decode DPA） | [sglang#33973](https://github.com/sgl-project/sglang/pull/33973) **CLOSED** | 重新提 PR；否则 waiver |
| `sglang_dsa/v0522/dsa_page_table_rows.diff` | MTP 下 page table 与 top-k 行数不一致 | 需要实测：可能只靠 `index_share_for_mtp_iteration:false` 就够（gfx942 镜像就是这么替代的） | [sglang#32209](https://github.com/sgl-project/sglang/pull/32209) OPEN（第三方） | 先测只用 override；不行再 waiver |
| `sglang_disagg/v0522/patch_prefill_nonfinal_chunk_wait_event.py` | PD 分块 prefill 的非末块 KV 等待事件 | 需要（正确性） | 未找到上游 PR | 提 PR + waiver |
| `sglang_rocm/patch_hicache_rocm_staged_write_back.py` | ROCm 上 DSA pool 的 HiCache write_back 门控不一致，首次写回崩溃 | 需要；或改用 `write_through`（性能待测） | 未找到上游 PR | 提 PR；同时测 `write_through` |
| `sglang_kv_events/patch_kv_metrics_send_noblock.py` | KV 指标阻塞发送让调度器停住 | 需要（kv-aware 路由依赖 KV 事件） | 未找到上游 PR | 提 PR + waiver |
| `sglang_pp/v0522/patch_deepseek_v2_pp_embed_dim.py` | `pp-size > 2` 时中间级无 embedding，加载失败 | 需要（PP4 prefill） | 未找到上游 PR | 提 PR + waiver（小而清楚，合入概率高） |
| `sglang_rocm/v0522/patch_mla_absorb_bmm_default_config.py` | 保留 aiter 自己的 MLA absorb BMM 配置 | 只影响性能 | — | 去掉后测性能损失；需要的话提 PR |
| `sglang_responses/v0522/patch_responses_unstreamed_tool_args.py` | Responses API 非流式 tool 参数 | 不需要（AgentX 走 chat completions，需确认） | — | 去掉 |
| `sglang_decode_radix_spec` | — | 在 v0.5.22 上是 no-op | — | 去掉 |
| Mooncake 重编（`faae8dd` + 两个 RDMA diff）+ libionic ABI 4 | 跨节点 HIP 路由、目的端本地 rail、ionic ABI 匹配 | 需要，如果用 Mooncake | 上游镜像自带较旧的 Mooncake | 换 MoRI（集群上已验证），或等镜像升级 Mooncake |

- **实现路径**：
  1. 在自己的节点上用**原样上游镜像**加运行时 `pip install amd-infera`，逐个关掉补丁，跑 smoke + GSM8K + 一个性能点，确认最小必需集合。
  2. 对必需项提上游 SGLang PR（#33973 需要重开）。
  3. 对未合入的写 waiver：改了什么、为什么原样镜像跑不了、上游链接、移除计划（参考 `docs/waiver/2508.md`）。
     注意 waiver 只覆盖"在 InferenceX 脚本里给固定镜像打补丁"，**不能用自建镜像**。
  4. 新的 `lmsysorg/sglang-rocm` nightly 包含补丁后，换镜像、删 waiver。
- **限制**：
  - 草稿模型精度规则：不能改 MTP 层精度，我们没改。
  - `--json-model-override-args` 会被按"改模型结构"的条款审查。`index_share_for_mtp_iteration:false` 关掉的是跨 MTP 迭代复用 index，计算只多不少，要在 PR 里写清楚。
- **风险**：
  - 上游合入不可控：#33059 自 7-31 开着，近 3 个月；#33973 10-03 被关闭。
  - 补丁多时 waiver 可能被拒；SemiAnalysis 对"AMD 自家引擎打补丁"敏感。
  - 换 MoRI 要重新调传输参数，性能可能变化。

### W3. infera 侧：打包、router 适配、发版（1–1.5 周）

- **打包**：现在的 wheel（`amd_infera-0.3.1-py3-none-any.whl`）不含 `infera-router` Rust 二进制，也没有发 PyPI。InferenceX 的先例是按 master 配置的版本 `pip install <包>==<版本>`
  （`vllm-router.sh`、`glm5.3-tilert-rocm.sh`）。两种做法：
  - 发一个带二进制的 manylinux x86_64 wheel 到 PyPI（maturin 或 setuptools-rust），版本号就是 master 里的 `router.version`；
  - 或发一个独立的 router 容器镜像（`ghcr.io/amd-agi/infera-router:<版本>`），用 srt-slurm 的 `frontend.container_image`。
  引擎侧的 `infera.engine.sglang` 是纯 Python，运行时 `pip install` 不会动镜像里的 torch / sglang（`pyproject.toml` 里 sglang extra 不拉引擎）。
- **router 改动**：
  - 会话 header 可配置，至少接受 `X-Dynamo-Session-ID` 和 `X-Correlation-ID`。
  - 如果 W4 选静态方案：加静态 worker 列表的 discovery，包括 prefill bootstrap 端口、KV 事件端点、decode DP size。
- **发版**：发一个包含 #210 router 改动（`--decode-request-weight`）的 infera 版本（如 v0.3.2），InferenceX 里固定这个版本。
- **审计**：确认本配置下 infera 包装层没有对 SGLang 做 monkey-patch。
  `kv_probe.py` 会 hook `RadixCache`，`kvd_wiring.py` 会改 `ServerArgs`，但只在 probe 传输和 kvd 后端时启用，本配置用的是 zmq KV 事件，应该不触发。
  需要在 PR 里能说清楚，否则会被当成"修改服务栈"。
- **风险**：小。PyPI 包名和发布权限需要 AMD 内部流程。

### W4. srt-slurm：新增 infera frontend（开发 1–1.5 周；上游评审 1–3 周，期间带补丁）

srt-slurm 的 frontend 是注册制的（`src/srtctl/frontends/`，`@register_frontend`），有两种基类。两条路：

| | A. 静态 router（`StaticRouterFrontend`） | B. 动态发现（`DynamicFrontend`，同 Dynamo） |
|---|---|---|
| 引擎怎么起 | srt-slurm 原样起 `sglang.launch_server` | 起 `python -m infera.engine.sglang`（以子进程拉起 SGLang，再注册到 etcd） |
| srt-slurm 改动 | 小：约 50–100 行子类，同 `atomesh.py`（45 行）、`smg.py` | 中：新 frontend + SGLang 后端的 worker 启动模式 + etcd 服务 |
| infera 改动 | 大：router 要支持静态 discovery；PD 启动探测、KV snapshot 这些包装层功能没有了 | 小：沿用现在的行为 |
| 与我们实测的一致性 | 有差异，要重新验证 | 与 PR #210 完全一致 |
| 合规说明 | 最干净：镜像原样，router 是独立进程 | 要说明包装层只是子进程管理和注册，类比 `dynamo.sglang` |

- **建议 B**：能复现我们测过的配置，infera 侧改动最少。如果 srt-slurm 维护者不接受，再退到 A。
- **实现路径**：按 srt-slurm 的 `frontends/AGENTS.md`，提交 frontend 模块、`tests/test_infera_frontend.py`、`docs/infera.md` 和一个 `examples/` recipe，向 NVIDIA/srt-slurm 提 PR。
  合入前在 InferenceX 的 `runners/srt-slurm/patches/` 带补丁（先例：`573-participating-gpus.patch`，以及 #4045 的 `604-atom-tp-from-pp.patch`）。
- **风险**：srt-slurm 是 NVIDIA 的仓库，评审节奏不可控。好在已经收过 `atomesh`、`tilert-router`、`smg`、`llm-d` 这些非 NVIDIA 的 frontend。

### W5. InferenceX 集成（1–2 周，含 dry-run）

- **master 条目**，例如 `glm5.2-fp4-mi355x-<fw>-disagg-agentic-mtp`：
  - 镜像：上游 `lmsysorg/sglang-rocm:v0.5.22-…`，按 digest 固定；模型 `amd/GLM-5.2-MXFP4`；`runner: cluster:mi355x-amds`。
  - 拓扑：每个点写 prefill `tp: 1, pp: 4` 和 decode 的 `tp`、`dp-attn`；`kv-offloading: dram`，`kv-offload-backend: { name: hicache }`；`router: { name: infera-router, version: … }`。
  - 注意 `dram-utilization` 与 HiCache 大小要匹配：每台 4 卡服务器约 1,499 GB，我们每个 PP4 prefill 用 4 × 250 GB。
- **recipe**：一个 `disagg-variants.yaml`，`base` 放共用参数，每个点一个 `override_<topology>_c<conc>`。
  - 主要是翻译 `worker.sh`：rail 写 `@fabric.rdma-devices`，GID / TC 用 fabric 字段；GPU 可见性由 srt-slurm 用 `ROCR_VISIBLE_DEVICES` 处理。
  - 去掉 `SIM_ACCEPT_LEN`；`benchmark.env` 设 `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID`。
  - 用 `uv run --extra recipes infx generate --config-key …` 和 `srtctl dry-run` 验证。
- **framework 名**（见第 6 节）：
  - 新增 `infera-sglang`（类比 `dynamo-sglang`）：要改 `infx/srt_slurm/synthetic_acceptance.py` 的 `ENGINES`、`infx/results/identity.py`、launch policy，
    还要给 InferenceX-app 提 PR（`packages/constants/src/framework-aliases.ts`、framework family 等）。这样 dashboard 上能看到 infera。
  - 沿用 `sglang-disagg`：infx 基本不用改，但 dashboard 上会显示成 `mori-sglang`（`identity.py` 的别名），infera 不可见，传输也标错。
- **缓存统计**：`infx/results/agentic/backends/sglang.py` 按 `sglang:` 指标汇总。要确认 PD 下只统计 prefill 端点、HiCache 命中算对（ATOM PD 为此改过 infx，见 #3964）。
- **其他**：`perf-changelog.yaml` 末尾追加一条；PR 标题中英双语；描述里写 AI 模型声明。
- **风险**：infx 包的代码质量门槛高，改动要尽量小；InferenceX-app 是跨仓库改动，多一轮评审。

### W6. 在 SemiAnalysis 集群上调通并跑稳（2–4 周，不确定性最大）

需要解决或确认的问题：

| 问题 | 现状 | 在 CI 上的后果 | 处理 |
|---|---|---|---|
| prefill GPU 内存访问错误（0xfffffffff000） | 约 19 个实例小时出现 1 次（TP4 prefill），PP4 下还没见到；出现后 GPU 要 `rocm-smi --gpureset` | 把共享节点弄坏，后面所有 AMD 作业受影响 | **提交前必须定位**，或在 PP4 下连续跑足够时长证明不再出现 |
| HiCache 撤栈后显存 10–65 分钟才释放 | 不开 HiCache 时 20 秒 | 同节点下一个作业起不来或被 host_setup 拒绝 | 定位 pinned host 内存注销的耗时（可试 hugepage、分块注册）；必要时在 teardown 里等待 |
| enroot 下 pinned 内存 | 容器 root 没有 `CAP_IPC_LOCK`，注册计入 `RLIMIT_MEMLOCK`（#4045 遇到过） | 每 rank 250 GB 的 HiCache 池注册失败 | preamble 里 `ulimit -l unlimited`，按 #4045 的方式验证 |
| ionic MR 注册（Mooncake） | `MC_MAX_MR_SIZE=1 GiB` 绕过 | 同 | 沿用；换 MoRI 就不存在 |
| 两个 prefill 同机 | docker `--ipc host` 下 RCCL 共享内存名冲突，我们烧 3 万个 pid 绕开 | enroot 共享宿主 PID 命名空间，可能不再冲突 | 实测确认；绕法要换掉（InferenceX 不允许这种 hack） |
| 4+4 同机（第一期） | 我们测的是跨两台机器各用 4 卡 | P → D 走本机网卡，HiCache 与 decode 共享主机内存 | 先在自己的节点上实测同机 4+4 |
| 模型权重 | mi355x-amds 上没有预置 GLM-5.2-MXFP4 | 首次要下载到节点本地缓存 | 照 #4045 的做法用节点本地 hub cache |
| 冷启动 10–15 分钟 + JIT | 我们靠具名卷缓存 | 每点重新 JIT | 调大 `health_check` 次数；接受启动开销 |

- **限制**：我们没有该集群的登录权限，每轮只能"提交 → 排队 → 看日志"，每轮以天计。共享 runner 上不能留下 root 文件，cancel 时也要清理（CONTRIBUTING 有专节）。
- **缓解**：
  - 先在自己的节点上用 Slurm + enroot（或至少非 docker、非特权、`ulimit -l` 受限）模拟一遍，把能复现的问题在本地解决。
  - 首次 sweep 用 `--trim-conc` 或单点，尽早暴露问题。
  - 请 AMD InferenceX 团队提供失败作业的日志。

### W7. 按 CI 口径重测，并针对新指标调整配置（1 周机时，可以马上开始）

- 换成 InferenceX main 上的 AgentX 客户端（当前 `7ca692b9`）重测前沿点，看客户端版本差多少。
- 以 E2E normalized p90 为目标重新选形态：
  - 测同机 4+4 在 C24 / C32 / C48 的表现（判断第一期单节点能否覆盖高交互端）；
  - 测 prefill 调度策略（shortest-prefill-first 和 FCFS）对 TTFT 长尾的影响；
  - 测 `--pp-async-batch-depth 1`（4+8 C112 +7.2%，未进 PR）在新口径下是否仍有收益。
- 用原样上游镜像加运行时 infera（W2 的最小补丁集），确认性能损失有多大。
- **风险**：去补丁、换客户端、新口径三者叠加，可能让曲线明显低于 PR #210。这是对外承诺数字前必须先知道的。

### W8. SGLang cookbook（1–2 周，与 W5 / W6 并行）

- 向 `sgl-project/sglang` 的 `docs/cookbook/` 提 GLM-5.2 MI355X PD 页面（或在现有 GLM 页面加一节）：
  prefill 与 decode 的 `sglang.launch_server` 参数、环境变量，以及 infera router 的安装和启动命令。
  只给 InferenceX 用的（合成接受率、AIPerf 设置）不用写，但要在 checklist 里说明理由。
- 必须在 InferenceX PR 合入前 **MERGED**，开着的或草稿 PR 不算。
- **风险**：SGLang 维护者是否愿意在官方 cookbook 里写第三方 router。InferenceX 的 checklist 明确允许 cookbook 包含 Dynamo 这类部署层的安装和 router 命令，
  但没在 cookbook 源码里搜到现成的 Dynamo 页面可以照搬，infera 知名度也低。退路是 PR 里只写 SGLang 原生命令，router 部分链到 infera 文档，这需要 CODEOWNER 认可。

### W9. sweep、评审、合入（2–3 周）

- `full-sweep-fail-fast`。第一期 5–7 个点，每点约 15–20 分钟起栈 + 60 分钟计分 + 撤栈（HiCache 释放慢时更久），单节点点可以并行。
  集群空闲时半天到一天出结果；与其他 AMD sweep 抢 runner 时要几天。按 2–3 轮估计。
- PR sweep 跑完后 bot 会贴 "unofficial run" 的 dashboard 链接，**这是第一个可以对内展示的里程碑**（还不是正式发布）。
- AMD CODEOWNER 按最新模板签 checklist，CI 的 Claude 校验器逐条复核；核心维护者在 Slack 上批准；`/use <run_id>`；`merge_with_reuse` 合入，摄取结果。
- **风险**：
  - checklist 里的"镜像原样 / 补丁 waiver / cookbook 已合入"三条，任何一条不过都卡住。
  - 每条曲线少于 5 个点要管理员豁免。
  - 合入后 GitHub Actions 失败由作者负责处理。

## 5. 时间线（1–2 人，按周）

```text
周        1    2    3    4    5    6    7    8    9
W1 对齐   ██████████(持续沟通)
W7 重测   ██████████
W2 去补丁 ██████████ ─ ─ ─ ─ ─ ─ 等上游合入（不可控，waiver 兜底）─ ─ ─
W3 infera      ███████
W4 srt         ██████████ ─ ─ 上游评审，期间带补丁 ─ ─
W8 cookbook         ██████████
W5 集成                  ██████████
W6 集群调通                    ███████████████
W9 sweep/合入                             ███████████████
里程碑：  第 3 周 本地原样镜像跑通    第 6 周 PR + 首次 sweep（unofficial run）    第 8–9 周 合入
```

- 关键路径：W2 / W3 / W4 开发（约 2 周）→ W5（1.5 周）→ W6 经 CI 调通（2–3 周）→ W9（2 周），合计约 7–9 周。
- 乐观 5–6 周：上游补丁 waiver 被接受，集群上一两轮就跑通，cookbook 很快合入。
- 悲观 3 个月以上：维护者要求先合上游、内存访问错误或显存释放问题在 CI 上反复、cookbook 不接受第三方 router。
- 第二期（2P1D、1P1D 12 卡，2 节点）用 `append-only` 追加：2 节点调通约 2 周，sweep 加评审约 2 周。前提是镜像未变。

## 6. 需要决定的事

1. **dashboard 上的名字**：
   - 新增 `infera-sglang`：infera 可见，要改 infx 和 InferenceX-app，多 1–2 周和一轮跨仓库评审；
   - 沿用 `sglang-disagg`：快，但 infera 不可见，而且会显示成 `mori-sglang`。
   如果目标是展示 infera，只能选前者。
2. **router 接入**：B（动态发现，包装层起引擎，与实测一致，建议）还是 A（静态 router，合规最干净，infera 要加静态 discovery）。
3. **首发范围**：第一期只做单节点 4+4（建议），还是一次提交全部形态。
4. **KV 传输**：
   - 保留 Mooncake：要么等上游镜像升级 Mooncake，要么在 waiver 里重编，后者很难被接受；
   - 换 MoRI：集群上已有 SGLang PD 先例，免掉重编和 libionic，但要重新调传输参数并重测。

## 7. 风险汇总

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 上游 SGLang 补丁迟迟不合入 | 高 | 只能靠 waiver，可能被拒 | 尽早提 PR；用实测压缩必需补丁集；waiver 写清移除计划 |
| prefill 内存访问错误在 CI 上出现，弄坏共享节点 | 中 | 很高（影响所有 AMD 作业，信誉受损） | 提交前定位根因，或 PP4 下长时间无复现的证据 |
| HiCache 撤栈后显存释放慢，导致 sweep 不稳 | 高 | 高 | 定位 pinned 内存注销；teardown 里等待；必要时缩小 HiCache |
| 在 SemiAnalysis 集群上数字比 crsuse2 低（集群差约 15%，客户端、去补丁、新口径叠加） | 高 | 中（曲线比预期低） | W7 先在本地按 CI 口径重测，对外只引用 CI 数字 |
| SGLang cookbook 不接受 infera router 命令 | 中 | 高（checklist 过不去） | 早沟通；退路是 cookbook 只写 SGLang 原生命令，router 链到 infera 文档 |
| srt-slurm 不接受 infera frontend | 低–中 | 中 | 在 InferenceX 里带补丁（有先例）；退到静态 router 方案 |
| 维护者不接受新增 framework 名 | 中 | 中（infera 不可见） | W1 先确认；退路是沿用 `sglang-disagg` 并在 metadata 标注 router |
| 没有集群登录权限，调试慢 | 高 | 中（拉长工期） | 本地模拟 Slurm + enroot；请 AMD 团队协助拿日志或节点 |
| ATOM 的 PD（#4045，或 Infera #209 的 48.2k）先上榜，在高吞吐端支配我们 | 中 | 中（我们的点主要在中、高交互段有价值） | 突出"SGLang 是 GLM-5.2 的主引擎"；继续 decode 优化 |

## 8. 依据与来源

| 内容 | 位置 |
|---|---|
| 提交与评审规则 | InferenceX `CONTRIBUTING.md`、`AGENTS.md`、`inferencex-e2e/docs/PR_REVIEW_CHECKLIST.md`（main `7ca692b9`） |
| 引擎优先级、AgentX Pareto 口径 | `inferencex-e2e/docs/MODELS.md`（Engine submission policy；E2E normalized interactivity；North-star Pareto policy） |
| 补丁 waiver 示例 | `inferencex-e2e/docs/waiver/2508.md` |
| recipe 规范与 router 安装机制 | `benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`；`infx/srt_slurm/workload.py` 的 `INSTALLERS` |
| 集群 | `configs/runners.yaml` 的 `mi355x-amds`（10 节点，8 卡，约 3 TB 主机内存，`rdma0-7`，Slurm + enroot） |
| 同集群 SGLang PD 先例 | `amd-master.yaml` 的 `dsv4-fp4-mi355x-sglang-disagg-agentic-umbp-dspark`；`dsv4/sglang/mi355x-fp4/agentx/disagg-umbp-dspark.yaml` |
| ATOM PD 先例 | [InferenceX #4045](https://github.com/SemiAnalysisAI/InferenceX/pull/4045)（OPEN） |
| srt-slurm frontend 机制 | NVIDIA/srt-slurm `ae3d6877`：`src/srtctl/frontends/AGENTS.md`、`atomesh.py`、`core/schema.py` 的 `FrontendConfig` |
| 我们的镜像补丁 | Infera `deploy/docker/Dockerfile.sglang`、`deploy/docker/patch.upstream.status.md`（PR #210 head `8859b247`） |
| 我们的数据 | `../2p1d-v0522-20261020/results/`；PR #210 描述 |
| InferenceX 数据快照 | `evidence/inferencex-api-glm52-20261027.json`（`https://inferencex.semianalysis.com/api/v1/benchmarks?model=GLM-5.2`，10-27 07:45 UTC） |

## 9. 复现

```bash
cd /home/liyingli/bench_agentx/baseline/Infera/llying
python3 inferencex-publish-plan-20261027/scripts/e2e_norm_table.py   # 第 2 节的表
```
