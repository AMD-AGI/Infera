# GLM-5.2 PD 上游 PR 复核报告：与上游 main / PR 的对比、review 结论、需重新验证项

截至 2026-11-23 14:22 UTC。本报告接在 `REPORT-upstream.zh-CN.md`（11-20，分支 `dev/pd_opt/vllm_glm52_agentx`，目录
`llying/upstream-vllm-20261117/`）之后，只做三件事：把其中 10 个 PR 与上游对比、对上游没有的做 review 并处置、标出哪些已有验证结论因 main 更新而需要重做。

本次**没有跑任何测试**（这台机器没有 torch / GPU），所有结论来自读代码和 `git` / `gh` 对比。

## 0. 摘要

对比基线：

| 仓库 | PR 基线 | 现在的 main | 差距 |
|---|---|---|---|
| vLLM（`vllm-project/vllm`） | `8828834fb`（11-19） | `afb35f85e`（11-23 13:51 UTC） | 239 个提交（距 J3b 用的 `f0c592bfd` 256 个） |
| Infera（`AMD-AGI/Infera`） | `962845a65` | `962845a65` | **没有变化** |

逐项结论：

| 项 | PR | 上游是否已有 | 本次处置 | main 更新对实现的功能性影响 |
|---|---|---|---|---|
| U13 | fork [#1](https://github.com/llying-001/vllm/pull/1) | **部分已有**：main 的 #68006 覆盖了一半 | 不提交，保持现状；已备好改写补丁 | **有，需改实现并重新验证**（第 2.1 节） |
| U11 | fork [#4](https://github.com/llying-001/vllm/pull/4) | **已有**：上游 open draft #69348 做同一件事 | 不提交，保持现状 | 与 main 冲突（#67998）；不再适用 |
| U19 | fork [#2](https://github.com/llying-001/vllm/pull/2) | 没有 | review 通过；**未开到上游**（第 4 节） | 无；但与上游 open PR #69254 有耦合（第 2.3 节） |
| U17 | fork [#5](https://github.com/llying-001/vllm/pull/5) | 没有 | review 通过；**未开到上游**（第 4 节） | 无 |
| U16 | Infera [#246](https://github.com/AMD-AGI/Infera/pull/246) | 没有 | review 通过，**已转 ready** | 无 |
| U24 | Infera [#247](https://github.com/AMD-AGI/Infera/pull/247) | 没有 | review 通过，**已转 ready** | 无 |
| U14 | Infera [#248](https://github.com/AMD-AGI/Infera/pull/248) | 没有 | review 通过，**已转 ready** | 无 |
| U15 | Infera [#249](https://github.com/AMD-AGI/Infera/pull/249) | 没有 | review 通过（有非阻塞意见），**已转 ready** | 无 |
| U28 | Infera [#250](https://github.com/AMD-AGI/Infera/pull/250) | 没有 | review 通过（有非阻塞意见），**已转 ready** | 无 |
| U7 | Infera [#251](https://github.com/AMD-AGI/Infera/pull/251) | 没有 | review 通过，**已转 ready** | 无 |

**需要重新验证的只有 U13**。U17、U19 和 Infera 六项的已有验证结论不受 main 更新影响，开到上游前按原计划 rebase 后例行重跑单测和 pre-commit 即可。

一处与原报告不同的事实：vLLM 的四个 fork PR 在 GitHub 上本来就不是 draft（open、非 draft，base 是 fork 自己的 main）。所以"保持 draft"在它们身上的含义是"不开到
`vllm-project/vllm`"，本次四个都没有动。

## 1. 方法

- 上游 main 是否已包含：对每个 PR 涉及的文件和它依赖的代码路径，看 `8828834fb..afb35f85e` 之间的提交，并把 PR 分支试合到最新 main。
- 上游 PR 是否已包含：`gh search prs --repo vllm-project/vllm` 按函数名、关键字、作者检索（open 与 merged），对命中的读正文和 diff。Infera 侧列出全部 22 个 open PR
  和 11-10 以来关闭的 PR，并比对改动文件的交集。检索不可能穷尽，只能说"按这些关键字没有找到"。
- review：逐行读 diff 及其调用方 / 被调用方。Infera 六项另确认了 CI（lint、unit、unit-torch-cpu、rust、agent_sys、dco 全过，e2e 三项按路径规则跳过）。

## 2. vLLM 四项

### 2.1 U13：上游覆盖了一半，实现要改，**需要重新验证**

- **上游的变化**：[#68006](https://github.com/vllm-project/vllm/pull/68006)（`1934f186e`，11-20 合入）让 Model Runner V2 捕获 FULL graph 时以
  `CUDAGraphMode.FULL` 跑 forward（`cudagraph_utils.py` 里 `forward_fn(CUDAGraphMode.NONE)` 改成 `FULL`）。这正是原报告 4.1"讨论点"里说的更彻底的改法。
- **还没被覆盖的一半**：main 上 `_can_use_aiter_ag_rs()` 仍以 `bd is not None and bd.uniform` 结尾。捕获时 runner 不传 descriptor，`set_forward_context`
  补一个默认的 `BatchDescriptor(num_tokens=...)`，`uniform` 是 `False`；在 `vllm/` 下 `uniform` 只有定义处和这一处读取，没有任何地方把它设成 `True`。
  所以 **main 上 AITER AG/RS 的快速路径仍然从不生效**，U13 去掉 `uniform` 要求的那部分仍然需要。按关键字没有找到上游做这件事的 PR。
- **对 U13 实现的影响**：
  - PR 里"`NONE` 且当前流正在捕获就视为 FULL 捕获"的分支现在是多余的（捕获时模式已经是 FULL）。
  - 新测试里 `("NONE", True, None, True)` 这一例断言的是 main 已经不需要的行为。
  - 试合到最新 main 文本上无冲突，但不应原样提交。
- **改写后的补丁**（本地，未推送）：`U13-rework-on-afb35f85e.patch`（2 个文件，+42 / −4，基于 `afb35f85e`）。只保留"模式是 FULL 即可，不再要求 `uniform`"，
  测试改为用真实的默认 `BatchDescriptor`。只做了语法检查，**没有运行**。
- **需要重做的验证**（原结论都在 #68006 之前取得：J8 在 `8828834fb`，J3b 在 `f0c592bfd`）：
  1. 单测 `tests/distributed/test_comm_ops.py -k aiter`，以及新测试在 main 的代码上失败。
  2. pre-commit。
  3. decode 的 A/B（原 −2.0% 至 −3.1%）。#68006 之后，其他按 FULL 判断的代码在捕获期也会生效，main 一侧的基线已经变了，原数字不能直接沿用。

### 2.2 U11：上游已有同样的 PR，不再提交

- [#69348](https://github.com/vllm-project/vllm/pull/69348)（@xsxszab，11-23 06:27 UTC 开，open draft）与 U11 是同一个修法：在 `build_offloading_config`
  里把 `worker_kv_bytes_per_block` 取为各 PP stage 中最大的每块字节数。它引用的症状也相同（`TieringOffloadingSpec` 报
  `Timed out waiting for mmap file to reach ... bytes`），并在正文里说明了与 #50653 的区别。
- 它建立在已合入 main 的 [#67998](https://github.com/vllm-project/vllm/pull/67998)（`8399da378`，11-23）之上：#67998 给 `KVCacheConfig` 加了
  `pipeline_stages`（每个 stage 的 `bytes_per_block`）。U11 自己加的字段 `max_worker_bytes_per_block` 因此被取代，并且与 main 在
  `vllm/v1/kv_cache_interface.py` 冲突（两者在同一位置加字段）。
- 按 vLLM `AGENTS.md`"已有 open PR 做同一修复就不再开"，U11 不提交。原报告第 10 节第 2 条（单独提还是并进 #50653）不再需要决定。
- 我们的数据对 #69348 仍有用：它只在 2 × B300 上用 Qwen3-0.6B 测过，没有 ROCm 和大模型的结果。我们有 GLM-5.2 PP4 的实例（每块
  796,416 / 779,520 / 779,520 / 742,656 字节，修后读回 9 / 9，P4D4 上 4 个 PP worker 加入同一块 1,250 GB 区域）。可以留言并提出帮测，**留言没有发**。
- 如果 #69348 合入，P4D4 的 `TieringOffloadingSpec` + PP4 应该在它上面复测，而不是继续带 U11。

### 2.3 U19：上游没有，review 通过

- **main**：`vllm/v1/kv_offload/cpu/gpu_worker.py` 和对应测试文件自基线以来没有提交，`min_savings = max(page, MIN_HOST_REGISTER_SAVED_BYTES)` 原样；
  `shared_offload_region.py`、`cpu/spec.py`、`tiering/spec.py` 也没变。试合无冲突。**对实现没有功能性影响。**
- **上游 PR**：没有重复，但有一个要协调的：
  - [#69254](https://github.com/vllm-project/vllm/pull/69254)（@yuzhouo7，11-22 开，open）：在 ROCm 上启动时按 amdkfd 的全节点系统内存上限检查 CPU offload 区域，
    `TieringOffloadingSpec` 超限就 `ValueError`。它不改注册规则，而是在 `cpu/spec.py` 里**照抄了 `pin_mmap_region` 的选择规则**（同样引用
    `MIN_HOST_REGISTER_SAVED_BYTES`）来估算注册量。
  - 耦合点：如果两者都合入而 #69254 的估算不跟着 U19 改，它会把"每行省不到 3 MiB"的配置按整块区域 × worker 数计。按它的公式估算，我们的主机
    （MemTotal 2,752 GiB）上限约 2,707 GiB、扣掉每 worker 10 GiB 后约 2,667 GiB；GLM-5.2 PP4 的 1.25 TB 区域按整块计是 4 × 1,164 ≈ 4,657 GiB，会在启动时报错，
    而 U19 实际只注册约 1,164 GiB。**这是按公式的估算，没有实测。**
  - #69254 还给出了 U19 正文里"驱动的上限"的具体来源（amdkfd 的系统内存上限），U19 的说明可以引用它。
  - [#52296](https://github.com/vllm-project/vllm/pull/52296)（open draft，08-14 起，跨 worker 协调注册并统一回滚）、
    [#61054](https://github.com/vllm-project/vllm/pull/61054)（vGPU / WSL2 上的设备指针）改同一个文件，但做的是别的事。
- **review**：改动只有一处条件，ROCm 之外行为不变；原有测试补了 `is_rocm → False` 使其与平台无关。没有发现问题。代价（每行一次注册调用，行越小调用越多，
  没有上界）正文已写明，上游多半会要求给一个界。
- **已有验证**：单测（main `8828834fb`）+ 机制测量（v0.34.0 镜像）。U19 没有进过 P4D4 端到端（J3 / J3b 走的是 U26 的私有锁页路径）。不需要因 main 更新重做。

### 2.4 U17：上游没有，review 通过

- **main**：`experts/rocm_aiter_moe.py`、`prepare_finalize/naive_dp_ep.py`、`tests/kernels/moe/test_rocm_aiter_topk.py` 自基线以来没有提交；调用方
  `expert_map_manager.py`、`router/aiter_shared_routed_fused_moe_router.py` 也没变。试合无冲突。
- 同目录里变了、但不触及这条路径的：`modular_kernel.py` 和 `router/fused_topk_router.py`（都来自 #58672：前者加 `RouterWeightApplication`，只改
  `apply_router_weight_on_input` 的传递；后者在本 rank 没有 token 时跳过路由）、`_aiter_ops.py`（+300 行，没有涉及共享专家 / EP 的改动）。**对实现没有功能性影响。**
- **上游 PR**：按 `init_aiter_topK_meta_data`、shared expert + EP 等关键字没有找到重复。命中的 #67570（Qwen4Exp replicated MoE）、#64737（无 gate 的 MoE 路由，
  11-18 已合入，在基线之前）是别的事。
- **review**：没有发现问题。专门核对了一点：`localize_fused_shared_expert_ids` 断言 gather 后的行数不超过 meta 缓冲区，而缓冲区按
  `max_num_batched_tokens * dp_size` 分配，所以 DP 下 gather 的行数不会越界。
- **已有验证**：单测 + J8 的 DP4 + EP 端到端（main `8828834fb`，0 / 200 → 196 / 200）。结论仍然适用，不需要因 main 更新重做端到端。

## 3. Infera 六项（#246–#251）

Infera main 自 `962845a65` 以来没有提交，六个分支都仍可直接合并；其他 open PR 里没有做同样事情的。文件交集只有两处：#250 与 #251 都改
`infera/engine/vllm/args.py`（已知），#248 与 #182（10-08 的 usage telemetry draft）都改 `infera/server/__main__.py`。

**六个都已转为 ready**，GitHub 已自动向 CODEOWNERS 请求 review（xiaobochen-amd、jiejingzhangamd、limou102）。合并状态是 BLOCKED，即等待 review 批准。

| PR | review 结论 | 非阻塞意见 |
|---|---|---|
| #246 串行下发 | 通过 | 串行路径固定走 HTTP，即使两个 worker 都注册了 NATS（与 ATOM 的串行路径一致），正文可以补一句。Python 侧在 import 时读环境变量；测试用 `importlib.reload` 会生成新的类对象，CI 里无害 |
| #247 池空闲超时 | 通过 | Rust 路由只有这一个上游 client，所以覆盖全部请求。默认 4 s 对所有部署生效。Python 路由没有改，我没有核对它的连接池设置。证据偏弱，正文已如实写 |
| #248 nofile | 通过 | 提高后的软上限会被子进程继承（Rust 路由经 exec 继承是本意） |
| #249 分词移出事件循环 | 通过 | 见下 |
| #250 DP 引擎均衡 | 通过 | 见下 |
| #251 参数解析 | 通过 | 无 |

**#249 的意见**：

1. 注释说用单线程是因为 tokenizer 不能多线程共用，但这个保证并不完整：`render_probe.py:246` 本来就在默认线程池里调用 `hasher.token_ids_for`；
   `pick()` 在没有预计算结果时也会回到事件循环线程上分词。两种情况都是本 PR 之前就有的，不是它引入的。后续可以让 probe 也走同一个线程池，或在 `BlockHasher` 里加锁。
   这一点也许和 U21（怀疑 HF tokenizers 并发导致的崩溃）有关，只是线索，没有验证。
2. `MixedRouter` 的重试循环每次重试都会重新分词（在线程里，不阻塞事件循环）。
3. 分词仍是串行的，吞吐上界还在（正文已写）。

**#250 的意见**：

1. 环境变量是进程级的，没有限定角色。如果设在 DP > 1 的 **prefill** 引擎上，worker 注册时不带 `dp_size`，路由给 decode 写的 `remote_engine_id` 就不会带
   `_dp{K}` 后缀，KV 拉取会找不到对端。这是按 `vllm_mooncake.py` 的寻址规则推断的，没有实测。默认关闭，建议加一句文档或只对 decode 生效。
2. 与 #251 在 `args.py` 的同一位置各插入一个函数，后合的那个一定有文本冲突，需要 rebase。

**vLLM 新 main 对这六项的影响**（它们的端到端都带着 vLLM）：

| PR | 依赖的 vLLM 行为 | 在 vLLM main `afb35f85e` 上核对的结果 |
|---|---|---|
| #251 | `make_arg_parser` 和 `FlexibleArgumentParser` 的位置 | `vllm/entrypoints/common/cli_args.py`、`vllm/utils/argparse_utils.py` 都在（v0.34.0 同）；旧的 `entrypoints/openai/cli_args.py` 已不存在。兼容 |
| #250 | 不带 rank 头时 vLLM 自己选 DP rank | `f0c592bfd` 以来 `core_client.py` 只有 #68554（关停时的清理宽限），不涉及选 rank |
| #246 | PD 用的 `MooncakeConnector` 的 `kv_transfer_params` | `f0c592bfd` 以来该目录只有 #63830（删 `mamba_cache_mode`） |
| #247 / #248 / #249 | 无 | — |

结论：六项都不需要因 main 更新重新验证。

没有改 PR 正文。原报告说"你审完可以把'提交人已 review'加回正文"——那句话是你本人的声明，我没有代加。

## 4. vLLM 的 U17 / U19 为什么没有开到上游

按你的指示，上游没有的应当 review 后提交。Infera 六项已照做。vLLM 的两项我停在了"review 通过"，原因是 vLLM `AGENTS.md` 的硬性规定（违反可能导致自动封号）：

- "Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end."
- "The submitting human must review every changed line and run relevant tests."
- 开 PR（含 draft）前要跑 `pr-checklist`，正文要有可复现的测试命令和 pytest 汇总行。

我的 review 不能替代提交人逐行审；而且两项的测试结果都是在 `8828834fb` 上取得的，之后 main 多了 239 个提交，正文里的测试结果需要在 rebase 后重跑。
所以这一步留给你确认（第 6 节）。

## 5. 其他与上游重复的项、以及下次端到端的基线

原报告第 5 节的四个上游 PR 现在都还是 open、未合并：#68596（U12）、#60909（U26）、#66594（U20）、#63412（U18）；#50653 也仍 open，10-29 之后没有更新。

**下次在 main 上跑 P4D4 时基线已经不同**。J3b（48,417 tok/s/GPU）用的是 `f0c592bfd`。之后 main 上与 recipe 相关的变化包括：

- #66234：去掉 EAGLE / MTP 的"丢最后一块"逻辑，lookahead 哈希成为唯一行为；同时重构了 OffloadingConnector 的 scheduler / state / store trigger。
- KV offload：#66719、#67961（`access_mode`）、#69110、#69048、#68451。
- #68006（FULL 捕获模式），以及 `vllm/v1/attention/` 和 `_aiter_ops.py` 上的 19 个提交。

这些不是某个 PR 的实现冲突，我也**没有逐项分析它们对 recipe 的影响**；只是提醒：新跑出来的数字不能和 J3b 一对一比较，A/B 要在同一个 main 上做两侧。

## 6. 待你决定

1. **U17、U19 开到上游**：你逐行审完后告诉我，我可以 rebase 到新的 ROCm nightly 对应的 main、更新正文（U19 加上与 #69254 的关系），测试需要在集群上重跑后再开。
2. **U13**：是否采用改写补丁（推到 fork 分支 `u13-aiter-agrs-full-capture-main` 或新分支），然后重跑第 2.1 节列的三项验证。
3. **U11**：是否关闭 fork PR #4；是否在 #69348 下留言提供 GLM-5.2 PP4 的数据并提出帮测。
4. ~~本报告是否提交~~：已按你的要求提交到 `dev/pd_opt/vllm_glm52_agentx` 的 `llying/upstream-vllm-20261117/`。
5. 原报告第 10 节的第 4–6 条（Mooncake #5584、四份留言稿、ROCm clr issue）本次没有涉及，仍待定。

## 附录 A：本次的操作记录

| 操作 | 状态 |
|---|---|
| Infera #246–#251 `gh pr ready` | 已做，六个都是 open / 非 draft，review 已请求 |
| vLLM fork PR #1 / #2 / #4 / #5 | 没有改动（状态、分支、正文都没动） |
| 在 `vllm-project/vllm` 开 PR、留言 | 没有做 |
| 改任何 PR 的正文、发 review 评论 | 没有做 |
| 推送任何分支 | 没有做 |

## 附录 B：文件

- 本报告：本目录 `REPORT-upstream-recheck.zh-CN.md`
- U13 改写补丁：本目录 `U13-rework-on-afb35f85e.patch`（基于 vLLM main `afb35f85e`，未运行）
- 对比用的临时克隆：`/tmp/upstream-review/vllm`（含 fork 的四个分支）、`/tmp/upstream-review/Infera`，可以删除

## 附录 C：检索过的上游 PR（vLLM）

| 项 | 检索词 | 命中并读过的 |
|---|---|---|
| U13 | `_can_use_aiter_ag_rs`、`AITER AG/RS`、`use_aiter_ag_rs`、`BatchDescriptor uniform` | #48247、#63890（已合入）、#64428（open，QuickReduce，另一件事）、#68006、#60839 |
| U11 | `worker_kv_bytes_per_block`、`KV Offload PP`、`pipeline_stages offloading`、作者 xsxszab / StellaH-ter 的 open PR | **#69348**、#67998、#50653、#68020（canonical layout 的 PP 支持，draft） |
| U19 | `pin_mmap_region`、`shared offload region register ROCm`、`hipHostRegister offload` | **#69254**、#52296、#61054、#66391 / #58497 / #51081（已合入，在基线之前） |
| U17 | `init_aiter_topK_meta_data`、`fused shared expert expert parallel AITER`、`naive_dp_ep shared expert` | #67570、#64737、#24097 |
