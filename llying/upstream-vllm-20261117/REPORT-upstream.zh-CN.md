# GLM-5.2 PD（Infera + vLLM ROCm，MI355X）上游 PR / issue 工作报告

截至 2026-11-20 07:00 UTC。所有成绩若无说明都是**单次运行**；"main"指 vLLM / Infera 的上游主分支。每项的细节、PR 正文和原始数据在同目录的
`PR-<编号>-<短名>.md`，候选总表在 `CANDIDATES.zh-CN.md`，过程记录在 `HANDOFF-upstream.zh-CN.md`。

## 0. 摘要

**目标**：把 GLM-5.2 PD 最佳 recipe 用到的、以及试过的策略里涉及的 feature / 优化 / fix，整理成可提交的上游 PR 或 issue（vLLM、Infera、Mooncake、ROCm），
每项要有最小改动、测试、实测数据和说明。

**结果**（共 29 个候选，见第 3 节总表）：

| 去向 | 项 |
|---|---|
| 已开 PR，等你 review | vLLM（在你的 fork 里）：U13、U19、U11；Infera（AMD-AGI/Infera draft）：U16、U24、U14、U15、U28、U7 |
| 上游已有人做，不再开 PR | U12（↔ vLLM #68596）、U26（↔ #60909）、U20（↔ #66594）、U18 切片（↔ #63412）；nightly 清理崩溃已由 #68710 修好 |
| 写好未提交 | U17（代码 + 单测，pre-commit 过，**GPU 测试未跑**）；U2 的 ROCm clr issue（`gh` 缺 ROCm 组织授权）；给 4 个重复 PR 的留言稿 |
| 观察记录（证据不足，不提 issue） | U27（Mooncake 通知 QP）、U29（decode 高显存目标下偶发崩溃） |
| 未做 / 不做 | 调研类、已有绕法或价值低的其余各项（第 3 节） |

**最重要的几条证据**：

- U13（AITER AG/RS 在 FULL graph 捕获时生效）：decode 每步 −2.7%（51.49 对 52.86–53.12 ms），与 gfx950 gluon logits 合用 −6.3%；main 上单测 16 / 10、
  新测试在 main 的代码上失败。端到端是在 v0.34.0 上用同一补丁测的。
- U19（ROCm 上共享区只注册本 worker 的槽）：1 TB 区域 4 个 worker，现规则 2 / 4 注册失败、成功的也要 608–616 s；新规则 4 / 4、38.5–41.2 s。
- U11（PP 下统一 worker 槽宽）：不修时 `TieringOffloadingSpec` + PP4 起不来（加入方 30 s 超时）；修后读回 9 / 9；main 上单测 1,711 / 1,707。
- U17（DP + EP + 融合共享专家结果错误）：v0.34.0 上 GSM8K 1 / 200 → 修后 196 / 200；main 上代码仍有同样问题（读代码确认）。
- P4D4 端到端（J3，v0.34.0 + U26 / U12 / U13 + DP 均衡 + U24，不加 keep-alive 绕法）：43,269 tok/s/GPU，3600 s 内路由 0 次错误、无崩溃。

**需要你决定的**：第 10 节（review、何时用 GPU、#5584、留言稿、U11 的上游提法）。

## 1. 背景

- recipe（主线 P4D4，crusoe MI355X，8 卡一台）：prefill TP1 × PP4，GEM KV、显存目标 0.88，CPU 层 1.9 TB（OffloadingConnector）；decode DP4 TP1、MTP3、fp8 KV、
  显存目标 0.95；Rust 路由，vLLM Mooncake 串行下发；AgentX C104。最好的 3600 s 成绩 m10 = 48,572 tok/s/GPU（v3-006），是 sglang 同口径 53,351 的 91%。
- 候选来源：主线 P4D4 实验与 A 线（decode kernel / 通信）、B 线（prefill GEM / offload）的交付，汇总为 `CANDIDATES.zh-CN.md` 的 U1–U29。
- 本工作分两段：11-19 上午由上一个 upstream 会话建分支、写测试、跑 J1 / J1b / J2a / J3；11-19 14:30 起由本会话接手写 PR 说明、查重、提交，11-20 上午做不占 GPU 的收尾。

## 2. 方法

### 2.1 提交依据与测法

- **PR 一律基于最新 main 编写和测试**；v0.34.0 上的版本只是给 P4D4 用的回移副本，不作为上游依据。
- **vLLM 的 main 测法**：用 ROCm nightly 镜像（`vllm/vllm-openai-rocm:nightly-9c1d45c2`），在容器里把 vLLM 的 Python 代码同步成被测树、保留镜像编译好的扩展；
  同一组测试先跑 main（基线）再跑 main + PR，用 junit 结果比较"只看新增失败"。开跑前自动检查两提交之间没有改 ROCm 编译代码（例外都写明了，见 2.4）。
- **"修复前失败"**（vLLM `pr-checklist` 要求）：把 PR 的新测试放到 main 的代码上跑，确认按预期失败。
- **pre-commit**：vLLM / Infera 各自的 pre-commit 配置，在容器里对改动文件跑（宿主机的 `.git` 不被 root 改写）。
- **Infera**：Python CI（3.10，`pytest -m "not slow and not integration"`）、Rust CI（`cargo fmt --check`、`clippy -D warnings`、`cargo test --locked`），
  涉及 vLLM 0.34 接口的另在 vLLM 0.34 容器里跑。
- **端到端**：大多来自 v0.34.0（A / B 线和主线 P4D4），PR 正文都明说"在 v0.34.0 上用同一补丁测"；main 上的端到端（J3b、U13 的聚合 A/B）还没有。

### 2.2 证据分级（用于读第 4–7 节）

| 级别 | 含义 |
|---|---|
| A | main 上单测通过、无新增失败，且新测试在 main 的代码上按预期失败 |
| B | 端到端在 v0.34.0 上用同一补丁测得（main 上同一代码路径已核对未变） |
| C | 单次运行、或节点不同，只能作为方向性证据 |

### 2.3 上游规则（决定了"怎么提"）

- vLLM `AGENTS.md`（违反"可能导致自动封号"）：开 PR 前必须查重，已有 open PR 做同一件事就不开；不允许纯 agent PR，提交人须审过每一行；
  开 PR（含 draft）前要过 `pr-checklist`，正文要写明 AI 协助、测试命令和 pytest 汇总行（含耗时）。**因此 vLLM 的三项先开在你的 fork 里**，你审完再开到上游。
- Infera 是公司内部仓库：按你 11-19 的决定直接在 AMD-AGI/Infera 开 draft（draft 不自动请求 CODEOWNERS）。
- 尽量只影响 ROCm；影响 CUDA 的写进跨平台说明（目前只有 U11）。

### 2.4 资源使用（本工作的全部作业，都是跑完即退）

| 作业 | 时间（UTC） | 资源 | 内容 |
|---|---|---|---|
| J1 298936 | 11-19 11:34–13:59 | 8 卡（v3-057） | vLLM 各项 v0.34.0 / main 单测、U26 主机内存、clr 复现、U12 diff 探查 |
| J1b 298969 | 11-19 12:17–13:59 | 8 卡（v3-006） | Infera Python / Rust CI、vLLM 0.34 容器检查、U24 竞争检查、U19 测量 |
| J3 298999 | 11-19 12:43–14:24 | 8 卡独占（v3-191） | P4D4 3600 s（PR 代码，v0.34.0） |
| J2a 299106 | 11-19 14:07–14:23 | 2 卡非独占（v3-015） | U19 单测 |
| 299146 | 11-19 14:47 | 1 CPU | 宿主 OS / CPU 信息（clr issue 用） |
| J4 299174 | 11-19 15:11–15:26 | 2 卡非独占（v3-221） | U13 / U19 在 main `8828834fb` 上重测、pre-commit、修复前失败 |
| J3b 299112 | 11-19 15:47–15:53 | 8 卡独占（v3-193） | P4D4 on main；排队 1.5 h，刚开跑即按你的要求取消 |
| J5 299267 | 11-19 15:59–16:25 | 2 卡非独占（v3-221） | U11 重测、pre-commit、修复前失败；Infera pre-commit |
| J6 300391 / 300403 | 11-20 05:45–05:50 | 16 CPU，无 GPU（v3-127） | U17 pre-commit |

J1 / J1b 是上一个会话在"按需申请"规则之前提交的整节点作业，GPU 大多闲置（HANDOFF 第 4 节有核算）；之后的作业都按需申请。

## 3. 全部候选项的去向

| # | 内容 | 仓库 | 去向 |
|---|---|---|---|
| U1 | MooncakeConnector 支持自定义 KV 内存池（AMD GEM） | vLLM（依赖 Mooncake） | 未做：依赖你在 Mooncake 的 #5584 / #5587 先合入 |
| U2 | GEM KV + offload 在非 0 号卡上访存故障，根因在 ROCm clr | ROCm / Mooncake | clr issue 写好待提（缺授权）；修复应进 Mooncake #5584（6.2） |
| U3 | 缺按未命中 token 排序的准入（SPF） | vLLM | 未做（待 A/B） |
| U4 | Mooncake 并发下发 + prefill 重排会循环等待 | Infera / vLLM | Infera 侧即 U16；vLLM 侧未做 |
| U5 | 多 API server 时 `/metrics` 是局部的 | vLLM | 未做（待核实） |
| U6 | DP 下每个 rank 读全部权重分片，起栈 11 min | vLLM | 未做（待量化） |
| U7 | Infera 适配 vLLM 0.34 | Infera | 参数解析已开 draft #251；其余子项（grpcio 冲突、KV 事件端口、块大小）未做 |
| U8 | KF kernel 在 vLLM ROCm DSA 路径上的对应 | vLLM / aiter | 调研完成，只有 U12 值得提 |
| U9 | OffloadingConnector 只有写穿 | vLLM | 调研完成；上游 RFC #61396 / #65681 在做写回 |
| U10 | CPU 层锁页超过物理内存时被 OOM 杀而不是报错 | vLLM | 未做 |
| U11 | PP 下 KV offload 各 worker 槽宽不一致 | vLLM | **fork PR #4，待 review**（与上游 #50653 部分重叠，4.3） |
| U12 | gfx950 gluon paged-MQA logits | vLLM | **与上游 #68596 重复，不开**；留言稿待发（5.1） |
| U13 | FULL 捕获时放行 AITER AG/RS | vLLM | **fork PR #1，待 review**（4.1） |
| U14 | Infera 启动时提高 nofile | Infera | **draft #248**（4.4） |
| U15 | Infera Python 路由分词移出事件循环 | Infera | **draft #249**（4.4） |
| U16 | Infera vLLM Mooncake 串行下发 | Infera | **draft #246**（4.4） |
| U17 | DP + EP + 融合共享专家 + AG/RS 结果错误 | vLLM | 修复已移植到 main、pre-commit 过；**GPU 测试未跑**（6.1） |
| U18 | MoRI 接收缓冲区切片 / 零拷贝 combine | vLLM | 切片与上游 #63412 重复，不开；零拷贝价值低，不做（5.4） |
| U19 | ROCm 共享区只注册本 worker 的槽 | vLLM | **fork PR #2，待 review**（4.2） |
| U20 | Simple CPU offload 切不开时按页对齐 | vLLM | **与上游 #66594 重复，不开**（5.3） |
| U21 | P4D4 主机侧偶发崩溃（tokenizers 嫌疑） | Infera / vLLM / HF | 未做（定位中） |
| U22 | Simple lazy offload 覆盖窗口不够 | vLLM | 未做（等上游 #66708 后复测） |
| U23 | 锁页 CPU 层进程异常结束后 Shmem 不归还 | ROCm / 运维 | 未做；主线后来见过 1.3 TB 在 2.5 min 内逐步释放，可能只是慢 |
| U24 | Rust 路由连接池空闲超时长于引擎 keep-alive → 502 | Infera | **draft #247**（4.4） |
| U25 | 引擎重复渲染 chat 模板 | Infera / vLLM | 未做（短期用 renderer 4 线程绕开） |
| U26 | ROCm CPU 层按精确大小锁页 | vLLM | **与上游 #60909 重复，不开**；留言稿待发（5.2） |
| U27 | Mooncake 跨节点建不起通知 QP | Mooncake | 观察记录（7.1） |
| U28 | Infera DP 引擎均衡（让 vLLM 选 rank） | Infera | **draft #250**（4.4） |
| U29 | decode 高显存目标下偶发崩溃 / 卡死 | vLLM / ROCm | 观察记录（7.2） |
| — | vLLM nightly 上 offload 测试清理阶段随机中止 | vLLM | 上游 #68710 已修（5.5） |

## 4. 已开 PR、等你 review

### 4.1 U13：FULL CUDA graph 捕获时放行 AITER 自定义 AG/RS（[llying-001/vllm#1](https://github.com/llying-001/vllm/pull/1)）

- **问题**：ROCm 上 DP MoE 的 dispatch / combine 本该在 FULL graph 里走 AITER 的自定义 all-gather / reduce-scatter（#48247 引入），但判断函数
  `_can_use_aiter_ag_rs()` 要求 forward context 是 FULL 且 batch descriptor `uniform`。Model Runner V2 捕获 FULL graph 时在 `torch.cuda.graph(...)` 里以
  `cudagraph_runtime_mode=NONE`、不带 descriptor 跑 forward（`cudagraph_utils.py` 第 536–540、822–826 行，main 上已核对），重放时不跑 Python，于是这条路径
  从未生效，每步 156 次集合通信都走 RCCL 的通用 kernel。A 线的诊断版在每次捕获期调用都打出 `mode NONE capturing=True`。
- **改法**：把"NONE 且当前流正在捕获"视为 FULL 捕获，并去掉 `uniform` 要求（`sizes is None` 已保证各 rank token 数相同）。PIECEWISE、eager 仍走 RCCL。
  `use_aiter_ag_rs` 只在 ROCm + AITER 自定义 allreduce 时为真，CUDA 不受影响（上游有 #48247 在 CUDA 上回退的 issue #59765，正文特别写明）。
- **验证**：A 级——main `8828834fb` + PR `f07e10429`：`test_comm_ops.py -k aiter` 16 通过（main 10），新增 6 个用例里 2 个在 main 的代码上失败（正是
  FULL 捕获和"无 uniform descriptor"两种）；pre-commit 17 个适用 hook 全过。B 级——A 线 v0.34.0，GLM-5.2 DP4 TP1 MTP3，104 请求 × 64K / 4K：每步
  51.49 ms 对原版 52.86–53.12 ms（6 次），通信 6.10 对 7.8–10.2 ms / 步，GSM8K 196 / 200（原版 195–199）；与 gfx950 gluon logits 合用 49.77 对 53.12 ms
  （−6.3%）。P4D4（J3）日志确认走了 AITER AG/RS。
- **没做的**：main 上的端到端 A/B。
- **讨论点**：更彻底的改法是 runner 在 FULL 捕获时把 context 设成 FULL（其他"捕获时按 FULL 判断"的代码也有同样盲区），PR 只改通信器。

### 4.2 U19：ROCm 上共享 offload 区域只注册本 worker 的槽（[llying-001/vllm#2](https://github.com/llying-001/vllm/pull/2)）

- **问题**：多个 worker 共用一块 CPU offload 共享区（每行由各 worker 的槽拼成）时，`pin_mmap_region` 只在"每行能省 ≥ 3 MiB"时才只注册本 worker 的槽。
  GLM-5.2 PP4 每行 3.19 MB、槽 0.80 MB，只省 2.4 MB，于是 4 个 worker 各注册整块区域。B 线的 P4D4（1.25 TB 区域）里 rank 0 / 2 注册到一半被驱动拒绝，
  这两个 stage 的 CPU 层退成可分页内存。
- **改法**：只在 ROCm 上把门槛降到"每行省一页"；其他平台不变。
- **验证**：A 级——main `8828834fb` + PR `8bf00c65f`：814 通过（main 812），新测试在 main 的代码上失败（main 整块注册）；pre-commit 过。机制测量（J1b，
  v0.34.0 镜像，选择规则与 main 相同）：1 TB 区域、4 个 worker 同时注册，现规则 2 / 4 在 687.18 GB 处失败、成功的两个用了 608 / 616 s；只注册槽 4 / 4、
  每个 313,805 次调用、38.5–41.2 s。
- **代价（请你看）**：每行一次注册调用，约 8,000 次 / s / worker；行越小调用越多（0.5 MB 行的 1 TB 区域约 4 min）。正文提了按"worker 数 × 区域大小"选规则的备选。

### 4.3 U11：PP 下 KV offload 按最大 stage 统一每个 worker 的槽宽（[llying-001/vllm#4](https://github.com/llying-001/vllm/pull/4)）

- **问题**：`build_offloading_config` 按本 worker 的 KV 字节数定槽宽，PP 各 stage 层数不同（GLM-5.2 PP4 每块 796,416 / 779,520 / 779,520 / 742,656 字节），
  于是共享区（`TieringOffloadingSpec`）的总大小、槽偏移各 worker 不一致：主线 s6 里 PP1 / PP2 等区域长到自己期望的大小，30 s 超时。
- **改法**：`get_kv_cache_configs` 里与 `num_blocks` 取最小同处，记下各 worker 每块字节数的最大值（新字段 `max_worker_bytes_per_block`，`compare=False`），
  `build_offloading_config` 按它分槽。较小的 stage 多用 2–7% 主机内存。
- **验证**：A 级——main `8828834fb` + PR `de267df1a`：1,711 通过 / main 1,707（各 1 个 main 原有失败），新测试 2 个在 main 的代码上失败；pre-commit 过。
  B 级（B 线，v0.34.0）：聚合 PP4 不修时第 4 个 stage 30 s 超时、修后 9 / 9 正确并有 FS 层命中；P4D4 上 4 个 PP worker 加入同一块 1,250 GB 区域。
- **与上游的关系（要你定）**：[#50653](https://github.com/vllm-project/vllm/pull/50653)（08-01 起 open，同样是 GLM-5.2 PP4）把各 stage 的 CPU chunk 数统一成最小值，
  修私有张量路径的越界块号；它不统一槽宽，所以共享区那部分只有 U11 修。两者改同一个函数。可以单独提，也可以并进 #50653。
- **跨平台**：不是 ROCm 独有，CUDA 上同样会触发；我们只能在 MI355X 上验证，正文请上游在 CUDA 上补测。

### 4.4 Infera 六项（AMD-AGI/Infera draft #246–#251，分支 `llying/<主题>`）

六个分支都基于 Infera main `962845a65`（至今未变）；Python / Rust CI 在 J1b、pre-commit 在 J5 都过了；查重都没有重复。

| PR | 项 | 问题（证据） | 改法 | 验证 |
|---|---|---|---|---|
| [#246](https://github.com/AMD-AGI/Infera/pull/246) | U16 串行下发 | vLLM Mooncake 两段并发下发 + prefill 重排时循环等待：s2b prefill 运行 0 / 排队 76 / KV 99.7%，返回停住超过 1 min（decode 侧指标不全，机制是推断） | `INFERA_VLLM_MOONCAKE_SERIAL=1` 时 Python / Rust 路由都等 prefill 回复后再发 decode | Python 3,055 / 3,054，Rust 370 / 369；s4 起所有 P4D4 轮（含 m10 48,572、J3）都没再停住 |
| [#247](https://github.com/AMD-AGI/Infera/pull/247) | U24 池空闲超时 | reqwest 空闲连接留 90 s、uvicorn 5 s 就关，写到正在关闭的连接上 → 502；s11 因根请求 502 中止，s10 有 2 次 5xx | 上游连接池空闲超时默认 4 s（`INFERA_UPSTREAM_POOL_IDLE_S`） | Rust 370 / 369；J3 去掉 keep-alive 绕法后 15 min 预热 + 3600 s 路由 0 次错误（C 级）；本机机制检查只在 keep-alive 1 s 时复现（2 / 7,545 对 0 / 7,548），5 s 时都是 0——**证据偏弱，正文如实写** |
| [#248](https://github.com/AMD-AGI/Infera/pull/248) | U14 nofile | 容器默认软上限 1024，路由 `Too many open files` 2,048 次后 502，s5 / s7 中止；峰值约 1,150–1,470 个描述符 | 启动时把软上限提到硬上限（无限时 1M），Rust 路由经 exec 继承 | Python 3,058 / 3,054；部署里一直用 docker `--ulimit` 绕开，**本 PR 未单独上部署** |
| [#249](https://github.com/AMD-AGI/Infera/pull/249) | U15 分词移出事件循环 | kv-aware 策略在事件循环里分词 10 万 token 级 prompt：路由单核 100%、`/metrics` 60 s 不响应、py-spy 87% 在 `tokenizer.encode` | `Policy.prepare()` 在单线程池里分词 / 哈希，`pick()` 复用 | Python 3,057 / 3,054；之后部署换了 Rust 路由，**本 PR 没有部署前后对比**；分词仍是串行的 |
| [#250](https://github.com/AMD-AGI/Infera/pull/250) | U28 DP 引擎均衡 | 路由替 decode 选 DP rank 不跟踪输出增长：m10 最忙 rank 在跑数是平均的 1.28 倍（p90 1.52），60% 的时间有 rank KV > 95% | `INFERA_VLLM_DP_ENGINE_LB=1`（默认关）让 vLLM 自己选 rank | Python 3,058 / 3,054，vLLM 0.34 容器 4 / 4；同节点 f1 在跑数均衡 1.19 / 1.40，但 KV 占满比例没改善，吞吐 −0.8%（KV 少 4%）；另一节点 m14 前 1800 s 49,183 对 46,083——**结论不明，所以默认关**（C 级） |
| [#251](https://github.com/AMD-AGI/Infera/pull/251) | U7 参数解析 | vLLM 0.34 挪了 `make_arg_parser`，Infera 的 vLLM worker 起不来（`No module named vllm.entrypoints.openai.cli_args`） | 先导入新位置，旧版本回退 | vLLM 0.34 容器内 main 3 / 3 失败、本分支 3 / 3 通过 + 新增 2 个；所有 vLLM 0.34 上的 P4D4 都带着它 |

#250 与 #251 都改 `infera/engine/vllm/args.py`，后合的要 rebase（两边正文已互相标注）。正文去掉了"提交人已 review"，你审完可以加回。

## 5. 与上游重复、未开 PR 的项

按 vLLM 的查重规定，以下都不再开 PR。我们的数据仍有价值，准备了留言稿（未发）。

### 5.1 U12 ↔ [#68596](https://github.com/vllm-project/vllm/pull/68596)（@RuibinCheung，AMD，11-19 06:11 开）

- 两者是同一个 gfx950 gluon paged-MQA logits kernel（从 aiter `e7d2453` 裁剪重调，文件名也相同），都从 decode metadata 取每请求的行数；#68596 覆盖更广
  （page 16 / 32 / 64、奇数组、最多 1,536 行、无开关）。
- 我们的数据（我们的移植，v0.34.0）：kernel 2.24–2.54 倍；DP4 MTP3 每步 −2.7% 至 −3.9%（3 次对 5 次）；GSM8K 196 / 200。留言稿在 `PR-U12-…md` 末尾。
- 附带：之前已加的 `Co-authored-by: yihou` 只在我们本地分支上，不再用到。

### 5.2 U26 ↔ [#60909](https://github.com/vllm-project/vllm/pull/60909)（@lobo235，外部贡献者，10-09 起）

- 同样用 `hipHostMalloc` 给 ROCm 私有 CPU KV 层按精确大小分配，另外还覆盖 CPU 上的权重；它只在消费级卡上测过。
- 我们的数据：torch 分配器 1.25 TB 实锁 1,360 GiB（+17%）、1.6 TB 时主机 OOM；精确分配 2 × 20 GB +37.25 GiB 对 +64.0 GiB；1.9 TB 层每 worker
  474.5–474.9 GB，三次 3600 s 运行无 OOM / 崩溃。留言稿在 `PR-U26-…md` 末尾。

### 5.3 U20 ↔ [#66594](https://github.com/vllm-project/vllm/pull/66594)（@cjackal，11-09 起）

- 同一个起栈失败（`No block and page boundary fits the registration chunk limit`），同样给 `host_registration_ranges` 加层偏移，只在需要处补齐。
- 我们的实例：GLM-5.2 PP4、自动 `num_cpu_blocks=392,382` 失败，`389,120` 可以。留言稿在 `PR-U20-…md` 末尾。

### 5.4 U18 ↔ [#63412](https://github.com/vllm-project/vllm/pull/63412)（@basemam，10-23 起）

- MoRI `dispatch()` 返回整块接收缓冲区；#63412 用主机侧已有的 `num_tokens_across_dp_cpu` 切片，与我们的修法相同。我们的数据：63.56 → 55.26 ms / 步（EPv1，单节点）。
- 另一项（IntraNode 零拷贝 combine）只再省约 0.4 ms，且 MoRI 不支持融合共享专家、main 已有 EPv2，不做。留言稿在 `PR-U18-…md`。

### 5.5 nightly 清理崩溃 ↔ [#68710](https://github.com/vllm-project/vllm/pull/68710)（11-19 22:52 已合并）

- 我们 J1 的 Simple offload 测试在整组运行的清理阶段 `Fatal Python error: Aborted`（PR 与基线两边都有，位置每次不同），当时前面测试留下的拷贝线程还活着。
  #68710 报的是同一签名：`SimpleCPUOffloadConnector` 没有 `shutdown()`，拷贝线程不停、锁页 CPU 缓存不取消注册，后来同一进程里的注册落到同一地址时 ROCm 运行时中止。
  不再提 issue；它进 nightly 后我们的测试就不必逐文件分进程跑了。

## 6. 写好、尚未提交

### 6.1 U17：DP + EP + AITER 融合共享专家 + AG/RS 分发时，共享专家被重复算或漏算

- **问题**：EP 下 `init_aiter_topK_meta_data` 按行号决定每个 rank 给哪些 token 算共享专家（`row % ep_size == ep_rank`），路由返回含这些列的 `topk_ids`；
  DP 的 naive dispatch 把它原样 all-gather，标记还是源 rank 本地行号的，于是有的 token 的共享专家在所有 rank 上都算（被 combine 加 ep_size 次）、有的一次都没算。
  MoRI 后端对这个组合直接断言拒绝，AG/RS 后端没有检查。A 线（v0.34.0）：GLM-5.2 DP4 + EP + 融合，GSM8K 1 / 200、输出全部失控。
- **main 上仍有**（11-20 读代码确认）；上游没有重复的 PR / issue。
- **改法**：dispatch 之后用本 rank 的 meta 缓冲区、按 gather 后的行号重写共享专家列（`localize_fused_shared_expert_ids`），只在 ROCm；modular `prepare` 和
  monolithic `prepare_routed` 都加。分支 `u17-aiter-fused-shared-ep-main` `d9dd5161c`（3 个文件，+90 行），新单测模拟 2 / 4 个 rank 的 gather 结果。
- **验证**：pre-commit 过（J6，CPU）；B 级（A 线 v0.34.0）修后 196 / 200、长上下文 32 / 32 + 31 / 32。**单测和 main 上的 DP4 + EP 门禁要 GPU，没跑**，所以还没开 PR。
- 备选：像 MoRI 一样拒绝这个组合（正确但失去融合，约 2.4 ms / 步）。

### 6.2 U2：ROCm clr issue，以及修复该落在哪里

- **根因**（B 线定位，11-19 在 rocm-systems develop `fca23cc6` 上核对仍在）：`hipImportExternalMemory` / `hipExternalMemoryGetMappedBuffer` 导入的内存不设
  `deviceId`（默认 0），`hipMemcpyBatchAsync` 按它分组，把非 0 号卡上导入内存的拷贝放到 device 0 的 null stream 上执行 → 访存故障；`hipIpcOpenMemHandle`、
  `hipMemImportFromShareableHandle` 都会设。最小复现（74 行 C）在 J1 上：device 1 / 3 故障，device 0 与普通 `hipMalloc` 正常。网页查重没有同样的 issue。
- **issue 正文已写好**（`PR-U2-…md`），没开：`gh` 的 token 没有 ROCm 组织的 SAML 授权。
- **修复的落点（11-20 新发现）**：Infera main 里没有 GEM 代码；而你在 Mooncake 开的 [#5584](https://github.com/kvcache-ai/Mooncake/pull/5584)（`HipGemAllocator`，
  有人在 review）导入 GEM BO 用的正是 `hipImportExternalMemory`。它在 Mooncake 自己的传输里查表绕开了"报成 device 0"，但 vLLM 的 CPU offload 直接调
  `hipMemcpyBatchAsync`，GEM KV + offload + PP rank 在非 0 号卡上时仍会故障。建议把 B 线的 VMM 导入（实测批量拷贝 51 GB/s、KFD 记账不变、P4D4 门禁 +
  1800 s 通过）移进 #5584。**#5584 没动**，等你定。

### 6.3 留言稿（都未发）

#68596（U12）、#60909（U26）、#66594（U20）、#63412（U18），各在对应文档末尾；都写明数据来自我们自己的移植 / 补丁，不是对方 PR 的代码，并提出可以帮测。

## 7. 观察记录（证据不足，不提 issue）

### 7.1 U27：Mooncake 跨节点在 P8D8 规模下建不起通知 QP

- 只出现过一次（11-18，第一次 P8D8）：prefill 侧 `Notification QP initialization failed: -302`（`ERR_MEMORY`）1,272 次，decode 侧 314 次重连后放弃，HTTP 500；
  同节点回环 0 次，单进程跨节点测试复现不了；两边 `MC_RDMA_NOTIFY_ENABLED=0` 后打通（预热 2,302 个请求 0 错误）。vLLM 的连接器不依赖通知 QP，所以绕法安全。
- Mooncake 仓库无同样的 issue（相关 RFC #4061）。提之前要：确认镜像里 Mooncake 的版本（构建脚本默认 v0.3.14，是否覆盖待核实）、做多进程复现。

### 7.2 U29：decode 在高显存目标下偶发崩溃 / 卡死

- 三次故障（m9b、m13 崩在 hipBLASLt `INTERNAL_ERROR` → `OUT_OF_RESOURCES`；m12 伴随 KFD 进程驱逐越来越慢直到卡死），都在显存目标 0.95 + AITER AG/RS 缓冲区、
  decode 走抢占重算时。主线推断是"显存余量不够"。
- **新数据（与该推断相反）**：J3 同样 0.95 + AITER AG/RS，跑满 3600 s 无故障（它的 CPU 层用 `hipHostAlloc` 锁页，而 m12 / m13 用的是第一版 mmap + 注册）。
  两种假设现在分不开，需要同配置重复运行。

## 8. 端到端运行（P4D4）

| 运行 | 配置 | 结果 |
|---|---|---|
| m10（主线，v3-006，3600 s） | 覆盖层版本，DP 均衡关，keep-alive 120 绕法 | 48,572 tok/s/GPU（最好） |
| f1 / f2（主线，3600 s） | m14 配置 + decode 0.93 + `hipHostAlloc` 锁页 | 48,173（v3-006）/ 44,691（v3-124），无故障 |
| **J3**（298999，v3-191，3600 s） | v0.34.0 + U26 / U12 / U13 的 PR 代码 + DP 均衡 + U24，**不加 keep-alive 绕法** | 43,269 tok/s/GPU，2.985 req/s，完成 / 丢弃 10,831 / 12，TTFT p50 / p90 3.57 / 17.84 s，ITL 16.6 / 24.0 ms；路由 "error sending request" / 5xx 全程 0；三项 PR 代码都确认生效；每 worker 精确锁页 474.5–474.9 GB；无 OOM / 崩溃 |
| J3b（299112） | main + U26 / U12 / U13 | 排队 1.5 h，开跑 6 min 时按你的要求取消（准备已完成，引擎刚启动，容器已清理） |

- J3 比 m10 低 11%：J3 在第一次使用的节点上、单次，节点间差异以前就有 7–10%，不能归因于 PR；decode 仍受 KV 限制（71% 的采样有 rank KV > 95%）。
- J3 主要用来确认"PR 代码能在完整部署里跑通且生效"（U26、U12、U13、U24、U28），不是用来证明吞吐收益。

## 9. 更正与踩过的坑

**对之前结论的更正（都已改在对应文档里）**：

1. U12 第一次对比"失败"时，对照其实是 vLLM 的 **PyTorch** 参考回退（测试容器没设 `VLLM_ROCM_USE_AITER=1`），之前写成"Triton 回退"；对 aiter 的差是 7.63e-6，
   在 logits 56–63 的量级上约 2 个 fp32 ulp（之前写"最低一位"）。
2. 主机内存是 MemTotal **2,752 GiB**（之前写"2.75 TB"）。
3. U29 的"显存余量不够"推断被 J3 削弱（7.2）。
4. U2 的修复落点从"Infera `deploy/`"改为"Mooncake #5584"（6.2）。

**过程中的问题（已修，不影响结论）**：

- J4 的 pre-commit 首轮因容器里 git 报"dubious ownership"一个 hook 都没跑；"修复前失败"首轮因我漏写一个预期用例、`-k` 表达式引号丢失而误判。都在同一作业里修正重跑。
- U17 的 pre-commit 首轮 mypy 抓到测试对 Optional 取下标，已修。
- 在 blobless 克隆上跑 `git log -S` 会逐个拉取历史数据，我在登录节点上误跑后立即停掉，克隆完好。
- J3 拆除容器 30 s 后 Shmem 仍有 1,547 GiB，之后没有读数；主线见过同类内存 2.5 min 内逐步释放，倾向于"释放慢"，未验证。

## 10. 待你决定 / 下一步

1. **review**：vLLM fork PR [#1](https://github.com/llying-001/vllm/pull/1)（U13）、[#2](https://github.com/llying-001/vllm/pull/2)（U19）、[#4](https://github.com/llying-001/vllm/pull/4)（U11）；
   Infera draft [#246](https://github.com/AMD-AGI/Infera/pull/246)–[#251](https://github.com/AMD-AGI/Infera/pull/251)。审完后：vLLM 的开到上游、Infera 的转 ready，正文里补回"提交人已 review"。
2. **U11 的上游提法**：单独提，还是并进 #50653。
3. **何时用 GPU**：
   - 开到上游前，把 U13 / U19 / U11 rebase 到当时最新 nightly 对应的 main 再重测（2 卡非独占，约 30 min；在最新 main `23b5918d9` 上已确认无冲突）；
   - U17 的单测和 DP4 + EP 门禁（4 卡非独占），通过后开 fork PR；
   - J3b：按原配置（main `f0c592bfd` + U26 / U12 / U13），还是换新 nightly、只带 U13（能直接给 U13 提供 main 上的端到端数据，但第一次在新 nightly 上部署，风险更大）。
4. **Mooncake #5584** 是否改成 VMM 导入。
5. **四份留言稿**是否发。
6. **ROCm clr issue**：给 `gh` 做 ROCm 组织的 SAML 授权，或你在网页上开。

## 附录 A：分支与提交（仓库外 `/home/liyingli/bench_agentx/upstream-20261119/`）

| 项 | 分支 | 提交 | 基于 | 状态 |
|---|---|---|---|---|
| U13 | `u13-aiter-agrs-full-capture-main` | `f07e10429` | vLLM main `8828834fb` | fork PR #1（旧版留在 `-f0c592bfd`） |
| U19 | `u19-rocm-pin-worker-slots-main` | `8bf00c65f` | `8828834fb` | fork PR #2 |
| U11 | `u11-offload-pp-worker-bytes-main` | `de267df1a` | `8828834fb` | fork PR #4 |
| U17 | `u17-aiter-fused-shared-ep-main` | `d9dd5161c` | `8828834fb` | 本地，未推 |
| U12 / U26 / U20 | `u12-…-main` `c63368efb` / `u26-…-main` `31d3966af` / `u20-…-main` `83d93e2a5` | — | `f0c592bfd` | 本地保留，不开 |
| Infera ×6 | `pr/*`（AMD-AGI 上为 `llying/<主题>`） | `561b7feb7` / `905c649e8` / `34b4b1bc0` / `20c4ba4c0` / `4885756d5` / `907933373` | Infera main `962845a65` | draft #246–#251 |

## 附录 B：文件索引（本目录）

- 每项：`PR-U2-…`（clr issue + #5584 建议）、`PR-U7-…`、`PR-U11-…`、`PR-U12-…`、`PR-U13-…`、`PR-U14-…`、`PR-U15-…`、`PR-U16-…`、`PR-U17-…`、`PR-U18-…`、
  `PR-U19-…`、`PR-U20-…`、`PR-U24-…`、`PR-U26-…`、`PR-U27-…`、`PR-U28-…`、`PR-U29-…`。
- 总表与过程：`CANDIDATES.zh-CN.md`、`PR-PLAN.zh-CN.md`、`HANDOFF-upstream.zh-CN.md`。
- P4D4 结果：`results/pr-298999/`；作业脚本副本：`kit/`（原件和全部原始结果在 `upstream-20261119/`）。
