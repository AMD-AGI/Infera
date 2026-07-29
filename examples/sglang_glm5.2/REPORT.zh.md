# GLM-5.2-FP8 1P1D on MI325X — 实验记录

两台 8×MI325X（gfx942）节点，一条 prefill 腿、一条 decode 腿，KV 走 Mooncake RDMA
（Broadcom bnxt_re RoCEv2 fabric），前面挂 Infera router。全部跑在已有容器里；镜像
不能重建，所以本该在镜像里做的修复都用脚本在运行时复现。

| | |
|---|---|
| prefill | `10.32.17.210:30001`（`tus1-p15-g46`） |
| decode | `10.32.17.209:31001`（`tus1-p15-g45`） |
| router / etcd | `10.32.17.210:8000` / `:2379` |
| 模型 | `/wekafs/models/GLM-5.2-FP8`，TP8 + DP-attention，fp8_e4m3 KV，tilelang DSA |
| SGLang | v0.5.16 |
| Mooncake | upstream #2682（`01d1eb2a`）+ HIP transport gate |

英文版：[`REPORT.md`](REPORT.md)。**它目前落后于本文**：§3.1 在那边还写作 "Open"
（实际已修复并验证），§5.1 整节缺失，§1.2 的死锁根因与 PD + MTP 跑通结论、§1.3 的直接
复现与上游对照、`max_running_requests` 的解释也都还没同步。以中文版为准。

---

## 1. 阻塞问题与已知雷区

### 1.1 Mooncake 对跨机对端强行走 HIP IPC

每个请求都死在 KV 交接这一步：

```
E0728 12:41:38 hip_transport.cpp:70] HipTransport: hipIpcOpenMemHandle failed
                                     (Error code: 17 - invalid device pointer)
[DP0 TP0] Session 10.32.17.209:16680 failed.
[DP0 TP0] Prefill transfer failed ... Failed to send kv chunk
```

`hipIpcOpenMemHandle` 是机内 GPU P2P，永远打不开另一台主机的 handle。上游 Mooncake
#2682 会无条件安装 HIP transport，而 `selectTransport` 又优先选它而不是 RDMA。

实测：运行时开关 `MC_USE_HIP_IPC=0` **没用**——设了之后两条腿依然各打印 8 次
`HIP transport installed for intra-node GPU P2P`。它只是调节 transport 的行为，
并不能阻止它被安装和被优先选中。只有源码级 gate 有效。

修复：`patch_mooncake_hip.sh` 打上
`deploy/docker/patches/mooncake_cpp/transfer_engine_impl.diff`（把安装动作 gate 在
`MC_ENABLE_HIP_TRANSPORT` 之后），增量重建 `engine.so`（约 30 秒，只有一个编译单元），
覆盖装到 pip 模块上，并把产物按原始 .so 的哈希为 key 缓存到共享盘，第二个节点直接
安装缓存而不用重编。两个节点报告的原始哈希都是 `10f75b32e43501e9`，顺带确认了两边
跑的是同一个镜像。

打完之后：两条腿上 `HIP transport installed` 0 次，`installTransport, type=rdma` 8 次。
两个 leg 脚本现在都会在 gate 缺失时直接拒绝启动，而不是加载 20 分钟后在第一个请求上
才炸。

同目录下的 auto-chunk-MR patch **故意没有**应用：它修的是 ionic 网卡上约 2 GiB 的
`max_mr_size` 截断问题，而这套 fabric 的 bnxt_re 报告 `max_mr_size = 512 GiB`，
远高于约 118 GiB 的 KV pool。

### 1.2 MTP 让 decode 腿在 PD warmup 阶段死锁

decode 腿加载完成、`/model_info` 也能响应，然后卡在
`Start of pd disaggregation warmup` 整整 25 分钟。整个过程 `/health` 返回 503，
因为 `server_status` 一直没离开 `Starting`。

对全部 8 个 scheduler 做 py-spy，每个 rank 都停在同一帧；12 秒内采样 6 次，栈完全
一致——是真死锁，不是热循环：

```
torch.distributed.all_reduce
  _all_reduce_in_place (parallel_state.py:884)
  _dp_gather_via_all_reduce (dp_attention.py:463)
  dp_gather_replicate (dp_attention.py:629)
  forward (deepseek_nextn.py:273)          <- MTP draft model
  _execute_idle (eager_runner.py:406)
  _draft_extend_for_decode (eagle_worker_v2.py:945)
  event_loop_overlap_disagg_decode (decode.py:2057)
```

为什么会走 eager：在 ROCm 上，只有当 draft attention backend 是
`AiterMultiStepDraftBackend` 时才会 capture draft-extend 的 CUDA graph
（`eagle_worker_v2.py:421-430`）；`supports_cuda_draft_extend_graph` 还额外要求
`_is_cuda or _is_musa`（`eagle_worker_v2.py:463-465`）。注意
`DeepseekSparseAttnBackend` **确实**在白名单 `graph_supported_backend_types` 里，
但它是在 `if _is_cuda or _is_musa` 分支内部才被 append 进去的。HIP 下两个条件都不
满足，于是 draft extend 永远走 eager。日志也印证了：只 capture 了 `target verify`
和 `draft decode` 两种图。

#### 缺这张图不是死锁的原因

本报告早先的版本把责任归给了 eager 的 draft-extend 路径。那是错的，单机那次运行直接
证伪了它。`inference_glm5p2_sglang/tmp/mtp/server_safe_dp8.log` 是单节点上的
DP8 + MTP 运行记录，它：

- capture 了完全相同的两种图：`target verify`（`num_tokens_per_req=4`、
  `bs=[1..6]`）和 `draft decode`（`num_tokens_per_req=1`、`bs=[1..6]`）；
- 全文 "draft extend" 出现 **0 次**，同样没有这张图；
- `max_running_requests` 同样是每 rank 6；
- 而且服务了 **197 个请求，全部 HTTP 200**，accept len **3.80–3.85 / 4** 个 draft
  token。

所以"DP attention + MTP + eager draft extend"本身是一个完全能正常工作的组合。要让它
死锁，必须再加上某个 PD 独有的因素。

#### 那个"PD 独有的因素"找到了：一个 rank 落后整整一个 spec stage

2026-07-29，打上 §1.3 的 DSA 补丁后重新以 MTP=1 起 PD（两条腿都打了补丁）。结果：

- **prefill 腿的 PD warmup 17 秒就过了**（`Start of pd disaggregation warmup` 13:59:54
  -> `End of disaggregation warmup` 14:00:11 -> 服务就绪），
- **decode 腿卡死在同一个地方**，`/health` 恒 503，warmup 开始后除 mooncake 的 RDMA 日志
  外没有任何 scheduler 前向活动，**全程 0 次 assert**。

也就是说 §1.3 的补丁没有修掉这个死锁，两者是独立的两个 bug。

这次对全部 8 个 rank 做 py-spy 并**逐 rank 标注**，拿到了比上面那份更关键的信息——上面写
"每个 rank 都停在同一帧"是采样不完整造成的错觉，真实情况是 **7 + 1 分裂**（连续两次采样
一致，分叉的都是 DP5）：

| rank | 栈顶 | 所处 spec stage |
|---|---|---|
| DP0-4、DP6-7（7 个） | `_execute_idle` (`eager_runner.py:406`) | `forward_batch_generation:1205` = **draft_extend** |
| **DP5（1 个）** | `_execute_decode` (`eager_runner.py:245`) | `forward_batch_generation:1192` = **draft** |

`forward_batch_generation` 里一次迭代的顺序是 `draft`(:1192) -> `verify`(:1195) ->
`draft_extend`(:1213)，每一段都带 DP 集合通信。所以这不是"形状不匹配"，而是
**collective 序列错位**：7 个空转的 rank 已经跑到第三段，DP5 还在第一段——而且它是唯一
一个有真实工作的 rank（跑的是 `_execute_decode` 而不是 `_execute_idle`）。DP attention 要求
所有 rank 执行同一串 collective，谁跨 stage 分叉，谁就把整组挂死。

**为什么这是 PD 独有的。** 这个"只有一个 rank 有活、其余 7 个空转"的瞬态正是 PD 造出来的：
8 个 warmup 请求的 KV 是通过 RDMA 逐个到达的，某一刻只有 KV 已经到位的那个 rank 拿到
PREBUILT 批次可以解码，其余 7 个还在等传输，只能走 idle。单机上不存在这个瞬态，prefill 和
decode 在同一个批次流里，8 个 rank 是一起换挡的。

**这也说明它不是 warmup 专属问题，不要指望 `--skip-server-warmup` 能绕过。** 串行流量下
永远是"1 个 rank 有活、7 个空转"，和触发条件完全一致。`--skip-server-warmup` 只是把撞墙的
时间点从 warmup 推到第一个真实请求。

#### 根因：GLM-5.2 的 IndexShare 让 `can_cuda_graph` 变成逐 rank 的决定

分叉点是 `eagle_worker_v2.py:511-517`：

```511:517:/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py
        if (
            can_cuda_graph
            and not forward_batch.forward_mode.is_idle()
            and self.seed_dsa_topk_from_draft_extend
            and draft_input.dsa_topk_indices is None
        ):
            can_cuda_graph = False
```

`can_cuda_graph` 每个 rank 自己算，而这个条件里有 `is_idle()`：**空转的 rank 因为
`not is_idle()` 为假而保住 graph 路径，有真实工作的那个 rank 却掉进 eager。** 于是同一个
DP step 里 7 个 rank 在 graph replay、1 个在 eager 跑 `draft()`。graph 是用
`get_default_mode_in_cuda_graph()` 捕获的（在 `SGLANG_USE_ROCM700A=1` 下是 SUM_LEN），
eager 走 `get_dp_padding_mode()`（MAX_LEN），collective 的条数和 gather 缓冲区大小都不一致
——正好对上实测的 7+1 分裂。

**为什么单机不中、PD 中。** 条件里的 `draft_input.dsa_topk_indices is None` 是分水岭：

- 单机：请求在本地做过 prefill，`_draft_extend_for_prefill`（`eagle_worker_v2.py:1151`）
  给它播了种，`dsa_topk_indices` 不是 None → busy rank 也保住 graph → 全员一致。
- PD decode 腿：收到的是 PREBUILT 批次，`output_dsa_topk_indices` 在多条路径上就是 None
  （`disaggregation/prefill.py:671`、`:674`、`:1199`，以及 `decode.py:1783` 在全为负值时
  也置 None）→ busy rank 掉 eager → 分裂。

触发开关是 GLM-5.2 checkpoint 自带的 IndexShare：`config.json` 里
`index_share_for_mtp_iteration = true`，配上 `topk == 1`，就让
`seed_dsa_topk_from_draft_extend` 为真（`eagle_worker_v2.py:243-251`）。

顺带排除两个先前的嫌疑：`speculative_skip_dp_mlp_sync` 方向相反（默认 False = sync 是开着
的），自适应步数也无关（`speculative_adaptive=False`，`activate_step_by_batch` 是 no-op，
`speculative_num_steps` 恒为 3，`:1179` 那条 `== 0` 分支走不到）。

#### 这个 guard 是 #30839 引进来的，上游正在修，全部还没合

`can_cuda_graph = False` 那三行不是老代码，是 **#30839「Stabilize GLM-5.2 MTP IndexShare
across PD and CUDA graph replay」**（2026-07-14，本地 `78dc581518`）加的，目的是避免用陈旧/缺失
的 seed 去 replay graph——修对了正确性，却把 `is_idle()` 写进了一个每 rank 各自计算的判断里。
我们跑的 v0.5.16 已经包含它，`v0.5.15.post1` 有对应的 cherry-pick #31083。

问过之后，上游的在办事项如下，**截至 2026-07-29 全部 open**：

| 上游条目 | 是什么 | 状态 |
|---|---|---|
| issue #32527（07-27） | 和本节完全同一个 bug：同样的 7+1 分叉、同样定位到 `is_idle()`、同样用 `index_share_for_mtp_iteration=false` 绕过 | open |
| PR #32209（07-23） | 正牌修复，+616/-25、12 个文件，一次收拾 5 个故障模式 | open，**CI 红** |
| PR #32196（07-23） | `[PD] Keep EAGLE DP graph and token metadata consistent`，同一片区域 | open |
| PR #31477（07-16） | `dsa/utils.py:66` 那条 TODO 的修法，把 seed 在 decode 侧重映射到本地 KV slot 后重新启用 fused TopK。**纯性能，不修死锁** | open，**CI 绿 + 已获 1 个 approve**，最接近合入 |
| PR #32722（07-29） | 这条轴上的回归测试，故意在 main 上是红的 | draft |
| issue #30854（07-11） | `dsa_seed_topk` copy_ 在 PD decode 崩，同族 | open |

#32209 的修法比关配置讲究：给 MLP-sync 那次 all-gather **追加一路 draft-graph 资格投票**（不新增
collective），让全 8 个 rank 一起决定要不要退 eager，并且只把 eager 限制在 draft 阶段——它测出
原来的写法会让 9.7% 的 decode 批次连 target verify 和 draft-extend 的 graph 一起丢掉。issue
#32527 里还有个三行版本：不退 eager，而是塞一个全零的 dummy seed 张量顶上，让所有 rank 都留在
graph 路径。两种都能解死锁，我们的配置覆盖等价于第三种。

**别把 #31477 当成这个死锁的候选修法。** 它和 #32209 修的是两件不同的事：#32209 管的是"seed 缺
失时全体 rank 要不要一起退 eager"（活性），#31477 管的是"seed 存在时能不能被 fused TopK 吃掉"
（性能，实测 TPOT −3.28% / ITL −3.08%，acceptance 66% 不变）。我们死锁的场景恰好是 seed **不
存在**，所以 #31477 合了也解不了，它也没有动那个逐 rank 判断的结构问题。

**一条我们能给上游的反例。** #32209 作者说"在 prefill 侧也开 MTP 就能避开这个 hang，因为那样
decode 起步时 seed 一定在"。我们这轮**两条腿都开着 MTP=1**（脚本本来就强制两边一致，prefill 腿
warmup 17 秒正常通过），decode 腿照样死锁。所以在 HIP/aiter + Mooncake + prefill 也是 DP8 的
形态下，prefill 侧开 MTP **不是**充分条件。要拿这个去提 issue 得重跑一轮留证据——当时那两份腿日
志被后来成功的那轮同名覆盖了。

#### 绕过办法：关掉 IndexShare，PD + MTP 当场跑通

`index_share_for_mtp_iteration` 来自模型 config 而不是命令行，但可以用
`--json-model-override-args` 覆盖。两条腿都加：

```
EXTRA_ARGS='--json-model-override-args {"index_share_for_mtp_iteration":false}'
```

（覆盖会传播到 draft 模型：`ModelConfig.from_server_args` 对 draft 也一律透传
`model_override_args`，见 `configs/model_config.py:524-545`。）

实测结果，同样是 MTP=1 的 1P1D：

| | IndexShare 开（默认） | IndexShare 关 |
|---|---|---|
| decode 腿 PD warmup | **无限死锁**，`/health` 恒 503 | **11 秒完成**（15:14:01 -> 15:14:12） |
| prefill 腿 PD warmup | 17 秒通过 | 16 秒通过 |
| 端到端（经 router） | 不可用 | 正常，跨节点返回 `Paris` |
| `verify_correctness` | 跑不了 | **全部通过**（9 项全 100%，needle 9/9、humaneval-long 20/20、长上下文差值 +0%） |
| 两条腿 assert / crash | — | 0 / 0 |
| accept len（真实 prompt） | — | **3.78 / 4**（n=41），与单机基线一致 |

**在 PD 下这个覆盖不花钱。** 直觉上关掉 IndexShare 应该损失一点性能，但 PD 形态下并没有：上游
自己的 `should_use_dsa_fused_topk()`（`dsa/utils.py:68-77`）只要
`disaggregation_mode != "null"` 且 seeding 开着就返回 False，DSA backend 于是
`use_fused_topk = False` 并打出一行 `Disabling fused DSA top-k for IndexShare under PD
disaggregation.`——**两条 PD 腿上 IndexShare 的消费端本来就是关着的**，`get_indexer_metadata`
里 `force_unfused` 恒为真。留着 `seed_dsa_topk_from_draft_extend=True` 只剩坏处：产出没人用，
却驱动了上面那个逐 rank 的 graph 判断。实测也对得上，accept len 3.78/4 与单机 IndexShare 开启
时完全一致；#32209 下面另一个复现者报的是关前关后 3.239 对 3.24。

**但这个"不花钱"有保质期。** 让它免费的前提正是 PD 下 fused TopK 被关着，而 #31477 就是专门来
解除这个限制的（把 seed 在 decode 侧重映射到本地 KV slot 后重新启用 fused TopK，CI 已绿、拿到
一个 approve，很可能比 #32209 先合）。**它一合，PD 下的 IndexShare 就真的有用了，我们这个覆盖
也就从零成本变成要付 ~3% TPOT。** 到那时应该换成 #32209 或者 #32527 里的 dummy-seed 三行版，而
不是继续关 IndexShare。这是升级 SGLang 时要重新评估的一条。

顺便一个排错提示：`SGLANG_DSA_FUSE_TOPK=false` **解不了**这个死锁（上游复现者验过）。它只动消费
端，不改 `seed_dsa_topk_from_draft_extend`，那个 graph 判断照样触发。必须从模型 config 这一层关。

在 #32209 合进来之前，这个覆盖是 PD + MTP 的**必需配置**，和 §1.3 的补丁一样属于硬前置。

#### PD + MTP 的性能：TPOT 降到 1/3.4，但吞吐这一栏不要直接对比

conc=64 / ISL 8192 / OSL 256 / 128 条 prompt，与 §4 之外单独跑的 MTP=0 PD 基线**同一负载**：

| | MTP=0 PD | MTP=1 PD |
|---|---|---|
| 成功率 | 128/128 | 128/128 |
| req/s | 1.40 | 0.39 |
| output tok/s | 357.5 | 100.1 |
| Mean TTFT | 18.8 s | 118.5 s |
| Mean ITL（每个 decode step） | 81.3 ms | 94.9 ms |
| **Mean TPOT（每个 output token）** | **81.3 ms** | **23.9 ms** |
| assert / crash / KV 错误 | 0 | 0 |

**吞吐和 TTFT 这三行不能当成 MTP 的代价来读**，因为两轮的 `max_running_requests` 差了 42 倍：
MTP=0 那轮是 2048，MTP 一开就被 SGLang 塌缩到 48（每 rank 6，见 §1.2 的
`speculative_hook.py` 那段）。整条流水线被这个 48 卡住，118 秒的 TTFT 基本都是排队。想做有
意义的吞吐对比，两边都必须显式给 `--max-running-requests`。

有意义的是后两行：每个 decode step 从 81.3 ms 变成 94.9 ms（MTP 的额外前向让单步慢 17%），
但一步吐出 ~3.78 个 token，于是 **TPOT 从 81.3 ms 降到 23.9 ms——单请求解码快约 3.4 倍**。
这就是 MTP 在这套栈上该有的收益，而且 accept len 3.78/4 说明 draft 质量没有因为关掉
IndexShare 而下降（与单机 IndexShare 开启时的 3.78 完全一致）。

#### 真正 PD 独有的东西

PD 中 decode 腿会调度一种 `PREBUILT` batch——代表"prefill 已在另一台机器上完成"的
假批次（`disaggregation/decode.py:get_new_prebuilt_batch`）。`PREBUILT` 这个 forward
mode 在别处根本不存在，而 DP MLP sync 对它有一段特判：

```338:341:/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/dp_attn.py
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()
```

这个内嵌的 idle batch 只在 PD 的 decode 循环里被拆开
（`disaggregation/decode.py:2080-2084`，是 `inner_idle_batch` 全代码库唯一的消费点），
而死锁恰好就在这条路径上：py-spy 里的 `decode.py:2057` 就是
`event_loop_overlap_disagg_decode` 中的 `self.run_batch(batch)`，
`eager_runner.py:406` 就是 `_execute_idle` 里的 `model.forward(...)`。每个 rank 当时
都在跑一个**嵌在 prebuilt batch 里的 idle draft extend**，DP gather 的 `all_reduce`
再也没返回。

单机 DP8 永远不会构造 `PREBUILT` batch，所以再多的单机 DP attention + MTP 测试也覆盖
不到这个交互。下一步要查的是：8 个 rank 看起来都进了这个 gather，为什么它还是完不成。
最可能的两个方向是各 rank 对 `global_num_tokens` 不一致（all_reduce 尺寸不匹配会挂），
或者尽管 Python 栈帧相同，实际有 rank 进的是不同的 collective。

**但打点要打对地方。** 打 `mlp_sync_info.global_num_tokens` 大概看不出问题——它是
all_gather 出来的，天生一致。真正可疑的是 draft extend 自己的 forward batch：它由
`prepare_for_draft_extend`（`speculative/eagle_worker_v2.py:900` 附近）从 target batch
派生，**不再走第二次 MLP sync**，gather 尺寸是本地重算的。要按 rank 打印的是那个派生
batch 的 gather 尺寸，再配 `NCCL_DEBUG=WARN` 区分上面那两种可能。

#### warmup 本身就是一次 8 路并发

补一条容易漏掉的事实：卡住的 warmup 不是一个请求，是 **8 个并发请求**。
`_send_disaggregation_warmup_requests`（`entrypoints/http_server.py:2077-2113`）用
`asyncio.gather` 对 `range(dp_size)` 同时发起，每个带 `routed_dp_rank=i`、
`bootstrap_host=FAKE_BOOTSTRAP_HOST`、`input_ids=[10,11,12,13]`。而单机那条路走的是
同一函数外面的 `disaggregation_mode == "null"` 分支（`http_server.py:2224-2233`），
只发**一个** `/generate`。这是"单机 DP8 + MTP 好端端的"与"PD warmup 挂死"之间除
`PREBUILT` 之外的第二个差异，而且两者指向同一件事：要让 8 个 rank 同时进到那个 idle
draft extend，先得有 8 个 rank 同时有活。

顺着这条能推出一个比打点更便宜的二分实验。`prepare_mlp_sync_batch_raw`
（`scheduler_components/dp_attn.py:230-236`）里 `PREBUILT` 和 `IDLE` 一样上报
`num_tokens = 0`，所以只有 1 个请求在飞时 8 个 rank 全报 0 →
`max(mlp_sync_info.global_num_tokens) == 0` → `need_idle_batch = False` → 不构造
`inner_idle_batch` → `_run_batch_prebuilt`（`disaggregation/decode.py:2076-2084`）直接
返回空的 `GenerationBatchResult`，连一次前向都不做。换句话说，py-spy 抓到的那个"嵌在
prebuilt batch 里的 idle draft extend"，在串行流量下根本构造不出来。

于是给 decode 腿加 `EXTRA_ARGS="--skip-server-warmup"`（它跳过整段 warmup 并把
`server_status` 直接置 `Up`，`http_server.py:2286-2291`）：

- 腿起来了、串行的 `verify_correctness` 跑过 → 死锁被限定在并发场景，"MTP on PD 的
  正确性"当天就能拿到，根因可以推到压测之前再修；
- 第一个真请求就挂 → 死锁与并发无关，是 prebuilt + draft-extend 这条路的通病，
  那就得老老实实上打点。

两种结果都比"再等 25 分钟看 warmup 挂在哪"信息量大，成本是多一个环境变量。

**但先把期望调低一点：同事那边也撞到了 PD + MTP 的 decode 腿挂死，而且是挂在第一个真
请求上。** `deploy/docker/patches/sglang_dsa/README.md` 的 "Known limitation" 一节写着：
打完 §1.3 的补丁后 crash 消失、server 能干净起来，但 **the first routed request
deadlocks in the EAGLE draft-extend metadata path**，栈是
`dsa_backend.py::init_forward_metadata` 里一次 `.max().item()` 的 GPU→CPU 同步与跨 DP
rank 的 collective 抢跑。那是 gfx950 / 0.5.15.post1 / MXFP4，站点和我们的
`deepseek_nextn` → `dp_gather_replicate` 也不是同一处，但"PD + MTP + DP-attention、
decode 腿、draft-extend 路径、挂死"这四项全中。

两种可能：是同一个 bug 的两个观测点（比如同步点位置随版本变了），或者这条路上有两个
独立的雷。前者意味着 `--skip-server-warmup` 只是把挂死推迟到第一个真请求，实验依然值得
做（它把答案从"不知道"变成"知道是哪一种"），但别指望它直接放行。后者意味着修完一个还有
一个。无论哪种，同事那边有一个可复现的环境，值得先合并信息再动手，而不是各自查一遍。

#### MTP 必须两条腿都开

值得记下来，因为不直观：`MTP` 是这一对之间**必须一致**的参数，和 `page_size`、
`kv-cache-dtype` 同一类。两个 leg 脚本传的 `MTP_ARGS` 完全相同，这是必需的——
尽管 prefill 阶段根本不发生投机：prefill 只吐一个 token，没有 draft/verify 循环。
它做的是 decode 腿**第一个** draft 步骤所依赖的全部准备工作：

1. 调用 `_draft_extend_for_prefill`（`eagle_worker_v2.py:1151`），把整个 prompt 过一遍
   draft 模型，填充 draft 模型自己的 `draft_token_to_kv_pool`。
2. 把这个 draft pool 追加进注册给 RDMA 的 buffer 列表
   （`disaggregation/prefill.py:186-194`，注释 "We should also transfer draft model
   kv cache"），于是 Mooncake 搬的是 target KV **加** draft KV。
3. 逐请求保存 EAGLE 的种子状态——`output_topk_p`、`output_topk_index` 和
   `hidden_states_tensor`（`disaggregation/prefill.py:661-667`）——放进专为此存在的
   aux buffer（`disaggregation/utils.py:298-306`，注释 `# For PD + spec decode`）。

第 3 点是硬性要求：EAGLE 的 draft 输入是 **(上一个 token 的 embedding, 该 token 在
target 模型里的 hidden state)**，而这个 hidden state 只有 prefill 会产出。只在 decode
开 MTP，两条腿注册的 KV buffer 数量就对不上，种子也永远送不到。

对调试的影响：**不能靠关掉 prefill 侧 MTP 来缩小 decode 侧死锁的范围**——那只会让这
一对直接失配，而不是隔离出 bug。但它的代价是实打实的：加载 draft 模型要把 704 GiB
checkpoint 再读一遍（启动时间约 2 倍），外加 draft pool 的显存和每次 prefill 多一次
draft-extend 前向，而收益全部落在 decode 腿上。

状态：**已绕过**，关掉 MTP 运行（`MTP=0`，现在是两个 leg 脚本的默认值）。根因仍未
解决，见第 6 节。

#### 顺手解决掉的一个谜：`max_running_requests` 为什么塌到每 rank 6

开 MTP 时 `max_running_requests` 是全局 48 / 每 DP rank 6，decode CUDA graph 只 capture
`bs=[1..6]`；不开 MTP 时是 2048、`bs=[1,2,4,...,512]`。本报告早先把它记成"待解释"，
其实一行代码就能解释完，而且**不是显存推导出来的**：

```507:510:/sgl-workspace/sglang/python/sglang/srt/arg_groups/speculative_hook.py
    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )
```

`_handle_eagle_family` 在这个值为 `None` 时硬编码成 48，之后
`resolve_max_num_reqs`（`mem_cache/kv_cache_configurator.py:1629-1632`）再除以
`attn_dp_size`：48 // 8 = **6**。日志里那行 warning 原样存在
（`tmp/mtp/server_safe_dp8.log`，`[2026-07-27 12:32:21]`），而且同一次运行的
`max_total_num_tokens=2100544` 与不开 MTP 的 2194496 几乎一样——KV 容量根本没缩，
`resolve_max_num_reqs` 里那条 `token_capacity // 2` 的上限差着五个数量级，压根没参与。

三条推论：

1. **与 PD 无关，pd-mixed 一模一样中招。** 那条日志来自 `disaggregation_mode='null'`
   的单机运行。这个 hook 在 `ServerArgs` 后处理阶段执行，只看
   `speculative_algorithm`，不看部署形态，所以 §5.1 那条"单机 pd-mixed 上验 MTP +
   kv-aware"的路径同样会被压到每 rank 6——要在那边测吞吐，必须显式给
   `--max-running-requests`，否则量到的是这个默认值的天花板，不是硬件的。
2. **它是一个 flag，不是一堵墙。** DP8 下想要每 rank N，就传 `8 × N`。这也消解了
   "MTP 把并发砍 40 倍所以不值得开"这个顾虑。
3. **但抬高它不免费。** `server_args.py:4064-4070` 里 decode 角色的
   `activation_tokens = max(running_requests × draft_tokens, 2048)`，
   而 decode CUDA graph 的 bs 列表也跟着 `max_running_requests` 长。抬之前要留出
   `MEM_FRAC` 余量，并且重新确认 graph capture 的时间和显存。

### 1.3 还没撞到但一定会撞到：DP attention + MTP 并发下 DSA indexer 行数不匹配

来源不是我们这边的运行，而是同事在 8×MI355X（gfx950）、ROCm 7.2.0、sglang
0.5.15.post1、GLM-5.2-**MXFP4**、单机 `--dp-size 8 --enable-dp-attention --ep-size 8`
+ EAGLE `steps=3 topk=1 draft-tokens=4` 上，**第一个 batch 就崩**：

```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
dsa_indexer.py::_get_topk_paged -> metadata.topk_transform
-> torch.ops.sgl_kernel.fast_topk_transform_fused
```

补丁已经有了：Infera PR #34，
`deploy/docker/patches/sglang_dsa/dsa_indexer_hip_dp_padded_rows.diff`。

**记在这里是因为这跟 MXFP4 无关，我们这套配置同样在雷区里。** 那个 assert 比的是
`score.size(0)` 和 `lengths.size(0)`（`sgl-kernel/csrc/elementwise/topk.hip:397`），
是纯行数问题，跟权重用什么格式存没有关系。逐条核对过两边的运行时，除了并发以外走的是
同一条路：

| | 我们（单机 DP8 + MTP） | 同事（MI355X） |
|---|---|---|
| arch / ROCm | gfx942 / 7.2.0 | gfx950 / 7.2.0 |
| SGLang | v0.5.16 | v0.5.15.post1 |
| 权重量化 | FP8 | MXFP4 |
| paged-MQA backend | aiter | aiter |
| `_use_aiter_preshuffle` | True（triton 3.6.0） | True |
| `page_size` | 64 | 64 |
| `dsa_topk_backend` | `sgl-kernel` | `sgl-kernel` |
| 峰值并发 | **≤ 1** | **64** |

`--dsa-prefill-backend tilelang --dsa-decode-backend tilelang` 在这里救不了我们：它管的
是稀疏 MLA attention，indexer 的 paged-MQA backend 在 ROCm 上是**强制** aiter 的
（`dsa/paged_mqa_logits_backend.py:24-30`，`value not in ("auto","aiter")` 直接抛错）。
两行有问题的代码在 v0.5.16 里也原样存在：`dsa_indexer.py:977-987` 把**没切片**的
`q_fp8` 交给 `aiter_paged_mqa_logits`，而 padding 恢复被 `not _is_hip` 挡掉：

```1037:1039:/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
        # Restore possible padding exist in the hidden states.
        if not _is_hip and q_offset < q_fp8.shape[0]:
            pad_len = q_fp8.shape[0] - q_offset
```

CUDA 路径（`deepgemm_paged_mqa_logits_split`）会先把 q/weights 切到 `q_offset` 再算、
算完补回 padding，所以它一直是对的；aiter 两件事都没做。

**为什么我们那次没撞到：负载形状，不是配置更安全。** padding 只在各 DP rank 的 token
数**接近但不相等**时才产生，判定在 `dp_attention.py:99-106`——decode 类 batch 走
`sum*2 >= max*dp_size` 才选 MAX_LEN（把每个 rank 补到全组最大），否则 SUM_LEN（各按自己
的长度，不补）。而 `tmp/mtp/server_safe_dp8.log` 里 **70 条 `Decode batch` 全部是
`#running-req: 1`**，全程只有一个请求：`global_num_tokens = [4,0,0,0,0,0,0,0]`，
`sum*2 = 8 < max*dp = 32` → SUM_LEN → 没有一个 rank 拿到假行。同事那边 conc=64 / dp8，
每 rank 7–9 个请求（28–36 token），`sum*2 ≈ 456 ≥ max*dp ≈ 288` → MAX_LEN → 低于最大值
的 rank 全部被补齐。**我们那 197 个请求全绿，只说明这套压测太温和。**

**为什么恰好炸在 DRAFT_EXTEND_V2。** 对方实测一轮 conc=64 出现 27 种 padded 形状，
26 种在 `DRAFT_EXTEND_V2`（典型 `q_fp8=(36,32,128)` vs `q_offset=32`，即 9 req×4 行 vs
8 req×4），`TARGET_VERIFY` 一次都没有。代码上对得上：eager draft-extend 下
`_pad_inputs_to_size`（`forward_batch_info.py:1364-1416`）把 `input_ids` 补到 DP 组最大
token 数、把 `extend_seq_lens` 这个 GPU tensor 补到 `bs`（填 0），但**不动
`extend_seq_lens_cpu`**；而 `q_offset = sum(metadata.get_dsa_extend_len_cpu())`
（`dsa_indexer.py:922`）读的正是那个 CPU list，于是行数涨到 36、`q_offset` 还停在 32。
`TARGET_VERIFY` 安全是因为那条路上 `batch_size` 会按
`num_tokens // num_tokens_per_req` 重算（`forward_batch_info.py:1336-1342`），B 和行数
一起被补，自然一致。注意 §1.2 已经证明 HIP 下 draft extend **永远走 eager**，所以这条
路径我们一直在跑，只是负载形状没触发它。

**对我们的影响是直接的。** 开 MTP 后每 rank 上限 6 个请求，只要 8 个 rank 出现
`[6,6,6,5,6,6,6,6]` 这种分布（tokens `[24,…,20,…]`，`sum*2=376 ≥ max*dp=192` → MAX_LEN），
那个 20 的 rank 就会被补到 24 行而 `q_offset` 还是 20 → 同一个 assert。也就是说
**§1.2 的死锁修好之后，第一次给 dp8 + MTP 加并发就会崩在这里**，先打 §1.3 的补丁再压测。

#### 正确性套件碰不到它，但 PD warmup 碰得到

这条决定了"要不要先验 §1.3 才能跑 MTP"的答案，所以单独记一下。

`verify_correctness.py` 是**严格串行**的：每个用例都是一次阻塞的 `session.post`
（`chat()`，脚本里没有线程池也没有 asyncio），连 20 道 HumanEval 也是 `for` 循环一道
一道打。所以整套套件的在飞请求恒为 1，`global_num_tokens = [4,0,…,0]` →
`sum*2 = 8 < max*dp = 32` → SUM_LEN → 一个 padded 行都不会产生。上面那 197 个请求就是
这套套件（`tmp/verify_mtp_dp8.txt`，结论行"全部通过"、needle 9/9 一直到 58,695 token），
配套日志里 `#running-req` **全程峰值为 1**。也就是说 **§1.3 不需要先验证，就能拿到
"MTP + DP-attention + DSA + 每 rank 16384 分块"这一整套组合的正确性结论**——单机上已经拿到了。

**但补丁还是要提前带上，因为 PD warmup 是并发的。** 见 §1.2：那 8 个 warmup 请求由
`asyncio.gather` 同时发出，一个 rank 一个。这正是"8 个 rank 各有请求、到达有轻微时间差
→ 数量接近但不相等"的形状，也就是 MAX_LEN padding 的触发条件。所以哪怕后续压测全部串行，
**MTP 一开、warmup 第一步就已经站在这个雷上了**。补丁的成本接近零，当保险带上，
不要等"验证过了"再打。

**还有一个没解释清楚的点，别据此认为低并发安全。** 那 27 种形状里有 1 种在 `IDLE`
（`q_fp8=(4,32,128)` vs `q_offset=1`），行数 4 对应"全机只有 1 个请求"——可按上面那条
启发式，`[4,0,…,0]` 应该判到 SUM_LEN、空 rank 拿到 0 行才对（我们那次也确实如此）。
这个矛盾还没查清，静态读代码没读出来。在打补丁之前不要因为"我们低并发跑得过"就断定
低并发不会中。

warmup 那件事给它提供了一个候选解释：行数 4 看着像"全机只有 1 个请求"的稳态，但如果这
一条其实来自"8 个 rank 几乎同时各来一个"的瞬态，那 SUM_LEN/MAX_LEN 的判定就不该按稳态
并发数去算。没有对方的日志证实，先记为线索——不改变"补丁要提前打"这个结论。

顺带记一个可核对的环境差异：我们镜像里带了 `SGLANG_USE_ROCM700A=1`（仓库里没有任何脚本
设它），它会让所有 cuda-graph 路径用 SUM_LEN 而不是 MAX_LEN
（`dp_attention.py:108-115`，ROCm 7.0.0-alpha 的 RCCL 绕过）。它不参与
`get_dp_padding_mode` 的判定，所以**不是**我们没撞到的原因；但如果对方那边有一部分
padding 来自 graph 路径，这个变量值得对一下。

#### 已在我们这套栈上直接复现（2026-07-29）

上面全部是从同事的数据推出来的。现在有本地的直接证据，两条结论要改：**MTP=0 下这个 bug
根本不可达**，而 **MTP=1 下它不需要压测，服务自带的 warmup 就会把它打出来**。

**MTP=0 复现不出来，而且不是负载不够。** 在跑着的 PD 对（MTP=0）上打 conc=64 / ISL 8192 /
OSL 256 / 128 条 prompt，8 个 DP rank 的 `#running-req` 同时分布在 1..7（典型
`[5,5,4,4,6,4,5,6]`），正是上面说的 MAX_LEN 触发形状；结果 128/128 全成功，两条腿
`lengths.size` 命中 0 次。原因是 MTP=0 时有四道闸门同时关着：

1. `_get_topk_paged` 的入口是 `decode_or_idle || target_verify || draft_extend_v2`
   （`dsa_indexer.py:1992-1996`），后两个都需要投机解码；
2. 带 extend 的 batch 一律走 SUM_LEN（`dp_attention.py:92-97`）。那条能覆盖它的
   `max_len_with_idle` 只对 hybrid-SSM 模型开（`:316`，看 `hybrid_override_pattern`），
   GLM-5.2 不是；
3. 纯 decode batch 确实判 MAX_LEN，但它跑在 CUDA graph 里，而 graph 是用
   `get_default_mode_in_cuda_graph()` 捕获的，在 `SGLANG_USE_ROCM700A=1` 下它返回 SUM_LEN
   （`decode_cuda_graph_runner.py:800`）；
4. 最关键：**即使 decode batch 被补了，`q_offset` 也会跟着涨。** decode/idle 两条路
   （graph `dsa_backend.py:1254`、eager `:815`）都是 `[1] * batch_size`，与行数同源。

第 4 条顺便否掉了"用 `--disable-cuda-graph` 逼 decode 走 eager、绕开第 3 条"这个后路。
**只有 `DRAFT_EXTEND_V2` 直接取 `extend_seq_lens_cpu`（`dsa_backend.py:855`）**，也就是
`_pad_inputs_to_size` 唯一不更新的那个 list。§1.3 和 MTP 是绑死的。

**MTP=1 单机 warmup 直接崩，压测都用不上。** 不打补丁，用
`sglang_naive_engine.sh`（TP8/DP8 + EAGLE `steps=3 draft=4`）起单机服务：13:06:11 warmup
的 8 个请求（每 rank 一个 `#new-seq: 1, #new-token: 64`）同时进来，DP0 和 DP3 立刻抛出同
一个 assert，`scheduler_0 crashed with exit code -3`，服务根本起不来。栈存在
`repro_dsa/EVIDENCE_unpatched_crash.txt`：

```
eagle_worker_v2.py:945  _draft_extend_for_decode
  -> eager_runner.py:406  _execute_idle
    -> deepseek_nextn.py:412 -> dsa_indexer.py:1996 forward_cuda -> _get_topk_paged:1036
      -> fast_topk_transform_fused -> RuntimeError: Expected lengths.size(0) == B
```

**但它是竞态，不是必然——这一点很重要。** §1.2 里引的那次单机 DP8 + MTP 运行
（`tmp/mtp/server_safe_dp8.log`）配置几乎相同，warmup **过了**，还服务了 197 个请求。
两次的差别只能是 warmup 那 8 个并发请求的到达时序：`asyncio.gather` 发出的 8 个请求若
恰好步调一致，每个 rank 的 draft-extend token 数相同，补到全组最大值是个空操作，不产生
假行；一旦有 rank 还没进来（0 token）而别的 rank 已经在 draft-extend，那个空 rank 就被
补出假行、当场 assert。所以**同一套配置有时过有时崩**，复现不出来不等于没有问题，
过了一次也不等于安全。

这条栈把上面两个悬着的点都落实了。`eager_runner` 证实 HIP 下 draft-extend 不进 graph，
gate 在 `eagle_worker_v2.py:421-476`：`supports_cuda_draft_extend_graph` 要求
`_is_cuda or _is_musa`，而 `DeepseekSparseAttnBackend` 也只在 `_is_cuda or _is_musa` 时才
被加进 `graph_supported_backend_types`，HIP 两条都不满足。更重要的是**炸点是
`_execute_idle`**：出事的 rank 自己没有请求，是被 MAX_LEN 补到全组最大值的——这正是上面
"没解释清楚的那 1 种 IDLE 形状"，而当时猜的"瞬态而非稳态"就是答案：warmup 那 8 个
`asyncio.gather` 请求同时到、每 rank 1 个，判定按瞬态走 MAX_LEN，没拿到请求的 rank 被补出
假行。低并发不安全，是因为"低并发"在 warmup 那一瞬间根本不成立。

**因此措辞要改一处：补丁不是"保险"，是硬前置。** 上面写的"成本接近零，当保险带上"说轻了——
不打补丁，MTP=1 的服务连 warmup 都过不去，不存在"先跑起来再说"这个选项。

**打上补丁后的 A/B（同一节点、同一配置、同一负载，唯一变量是补丁）：**

| | 未打补丁 | 打了补丁 |
|---|---|---|
| warmup | DP0/DP3 抛 assert，`scheduler_0 crashed with exit code -3` | 通过 |
| 服务状态 | 起不来 | `The server is fired up` @13:29:38，`/health` 200 |
| `verify_correctness` | 跑不了 | **全部通过**（needle 9/9、humaneval 20/20、humaneval-long 20/20、code-retrieval 2/2、deep-api 3/3） |
| conc=64 压测 | 跑不了 | 128/128 成功，assert 0 次 |

压测那轮值得单独看一眼：各 rank 的 `#running-req` 同时出现 4 / 5 / 6（`DP4=5, DP5=5` 而
其余 4 或 6），也就是**不相等**——MAX_LEN padding 确实在活跃状态下发生了，补丁在真实负载下
同样成立，不只是在 warmup 那一瞬间。

accept len 记一个坑：随机 prompt（`bench_serving --dataset-name random`）下均值 3.97/4，
真实 prompt（正确性套件那段）均值 **3.78/4**，和 §1.2 引的单机基线 3.80–3.85 一致。随机
token 会让模型进入重复输出、draft 全中，**不要用随机负载报 accept len**。

**曾经怀疑 §1.2 和 §1.3 是同一个 bug——已验证，不是。** 怀疑的理由是这条栈的前两帧
（`_draft_extend_for_decode` -> `_execute_idle`）和 §1.2 死锁栈完全相同。补丁打上后重新在
PD 上开 MTP=1 实测：**decode 腿依然卡在同一个位置**，`/health` 恒 503，全程 0 次 assert。
所以两者是独立的两个 bug，DSA 补丁修不了 §1.2。§1.2 的新证据见那一节。

#### 上游在别的后端上做同一个修复，HIP/aiter 这份还没人做

这个 assert 和 §1.2 是同一个源头：#30839 让某些 rank 退到 eager，而 eager 路径上 q 被按 attention
TP 物理 padding、KV 长度和 block table 却还是逻辑 batch，于是 indexer 的行数对不上。别的后端已经
在修，**做法和我们这份补丁一模一样**：

| 上游条目 | 后端 | 做法 | 状态 |
|---|---|---|---|
| PR #32762（07-29） | NPU | 调 `npu_lightning_indexer` 前把 q 和 indexer weights 裁到 `num_token_non_padded_cpu`，算完用零行补回物理长度 | open，CI 红 |
| PR #32209 第 4 项 | TRT-LLM | "Trim padded query and top-k rows to the real batch... then restore the physical DP shape" | open，CI 红 |
| 本补丁 | **aiter / HIP** | 调 `aiter_paged_mqa_logits` 前把 `q_fp8`、`weights` 裁到 `q_offset`，算完恢复 padding | 上游无对应项 |

三者是同一个 bug 在三个 kernel 上的三份表达。**aiter/HIP 这条路径上游还没有人碰**，所以这份补丁
不只是我们的本地权宜，是个可以直接提上去的东西——形状上照着 #32762 写就行。

## 2. 可用配置

`IB_DEVICE=rdma0`，两条腿其余部分一致：

```
TP8 / DP8 + --enable-dp-attention
--kv-cache-dtype fp8_e4m3, page_size 64
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
--chunked-prefill-size 131072   (-> 每 DP rank 16384)
--mem-fraction-static 0.85
--disaggregation-transfer-backend mooncake --disaggregation-ib-device rdma0
```

两条腿在 `max_total_num_tokens=2194496`、`max_running_requests=2048`、
`page_size=64`、`context_len=1048576` 上完全一致。

`MTP=0` 和 `MTP=1` 都可用。开 `MTP=1` 需要额外满足两个硬前置，两者同源于上游 #30839：

| 前置 | 怎么满足 | 不满足的后果 |
|---|---|---|
| `sglang_dsa` 补丁（§1.3） | 两个节点各跑一次 `bash patch_sglang.sh` | warmup 第一批就抛 `lengths.size(0) == B`，scheduler 退出 -3 |
| IndexShare 关闭（§1.2） | 脚本在 `MTP != 0` 时自动加 `--json-model-override-args '{"index_share_for_mtp_iteration":false}'` | decode 腿在 PD warmup 无限死锁，`/health` 恒 503 |

`MTP=1` 下 `max_running_requests` 会被 SGLang 塌缩到 48（每 rank 6），要压吞吐必须显式传
`--max-running-requests`（§1.2）。

> 这套 chunk 设置需要配合 §3.1 的 SGLang 补丁才是正确的。每 rank 16384 意味着超过
> 约 16k 的 prompt 都会分块 prefill，而未打补丁的 mooncake 会在 PD 传输里静默损坏
> 非末尾分块的 KV。补丁验证跑在 `MEM_FRAC=0.80` 上。

> `IB_DEVICE=rdma0` 是**正确性要求，不是性能调优**：八条轨道各自是独立的 /31，跨轨道
> 配对的 RoCE QP 到不了 RTR（§4）。而 Infera 那个本该兜住这件事的保护在本集群上不会
> 触发，所以去掉这个变量不会有任何报错，只会在压力下超时。

CUDA graph 状态（专门查过，因为 PD 场景下真正重要的是 decode 侧）：decode 图是**开**
的，`backend=full`，`bs=[1, 2, 4, 8, ..., 496, 512]`，且每个 decode batch 都打印了
`cuda graph: True`。prefill 阶段的图是关的，这是 SGLang 自身的默认行为
（`cuda_graph_config` 解析出 `prefill.backend='disabled'`），不是这些脚本设的。

---

## 3. 正确性

用 `inference_glm5p2_sglang/verify_correctness.py` 打 router，跑全量套件。
原始输出：`verify_pd_1p1d.log` / `.json`。

下面这张表是**打 §3.1 补丁之前**的结果，保留它是为了记录那两个 needle 失败长什么样；
补丁之后的结果见 §3.1 末尾。

| 检查项 | 结果 | 说明 |
|---|---|---|
| weights | 2/2 | indexer FP8 反量化自洽性，离线 |
| basic | 7/7 | 短问答 |
| determinism | 1/1 | greedy 解码 3 次复现一致 |
| idle | 3/3 | vLLM 上见过的"空闲后首请求损坏"在这里没复现 |
| **needle** | **7/9** | 两个失败都在 58k 的深度 10% 和 50%，见下 |
| humaneval | 20/20 | 短上下文 |
| humaneval-long | 18/20 | 8k 真实源码填充 |
| code-retrieval | 2/2 | 14k 跨文件符号检索 |
| deep-api | 3/3 | 自造 API 埋在 50% 深度 |

HumanEval A/B 差值 −10%（IndentationError 和 KeyError，属于模型噪声，套件自身判定
"未见长上下文退化"）。整轮零 KV 传输错误。

### 3.1 已修复：非末尾 prefill 分块的 KV 在 PD 传输中被竞争损坏

```
FAIL ~64k 深度 10%  ptok=58695  want=2183762  got='2183</think>2183</think>218</think>218</think> the'
FAIL ~64k 深度 50%  ptok=58594  want=7544440  got='7549.8.8.</think>7549. The</think>7549.8.8759</thi'
ok   ~64k 深度 90%  ptok=58611
```

这不是单纯检索不到：开头几位是对的（`2183` 完全正确；`7544440` 对了 `754`），
然后输出退化成重复的 `</think>`——而请求明明设了 `enable_thinking: false`。
4k 和 16k 的九个用例全过，只有约 58k 失败，且只在浅/中深度失败。

#### 定位：触发条件是分块 prefill，不是上下文长度

以每 rank 的 `chunked_prefill_size` 边界（16384 token）为中心扫描 prompt 长度，
断点精确落在分块边界上：

| prompt token 数 | prefill 块数 | 结果 |
|---|---|---|
| 14,642 | 1 | 通过 |
| 18,364 | 2 | 失败 |
| 29,293 | 2 | 失败 |
| 36,537 | 3 | 失败 |
| 43,932 | 3 | 失败 |
| 58,695 | 4 | 失败 |

只要 prompt 装得进一块就不失败。64k 从来就不是关键数字——18k 就已经坏了。
（"装不进就必然失败"这半句后来被推翻了，是时序问题，见本节末尾。）

还有两个事实值得记下来。

第一，失败形态不是"没找到"而是"截断"：模型返回 7 位 needle 的前 3–4 位，然后开始
复读——`4196795 → '4196'`、`1186494 → '1186'`、`2183762 → '2183'`、
`9706003 → '97075...'`。needle 的内容显然是够得着的，只是不完整。（当时据此以为
MLA KV 完好、坏的是"哪些 token 被选中"，后来证明反了：坏的正是搬过去的 KV 内容本身。）

第二，needle 埋在哪一块决定成败，而且是一条硬边界。needle 测试把一个 7 位秘密数字埋进
一大段无关文本再问模型它是多少，"深度 10%" 指埋在整篇 prompt 约 10% 的位置。58,695
token 在每 rank 16384 的 chunk 下被切成 4 块，对应关系是：

| prefill 块 | token 范围 | 落在这里的深度 | 结果 |
|---|---|---|---|
| 第 1 块 | `[0, 16384)` | 10%（≈ 5,870） | 取不回 |
| 第 2 块 | `[16384, 32768)` | 50%（≈ 29,348） | 取不回 |
| 第 3 块 | `[32768, 49152)` | — | — |
| 第 4 块（最后） | `[49152, 58695)` | 90%（≈ 52,826） | **取回** |

所以规律不是"越靠后越准"这种强弱关系，而是：**needle 落在最后一块里就答对，落在前面
任何一块就答错。** 换成 decode 腿的视角——只有最后一块搬过去的 KV 是完好的，前面几块
是坏的，那段上下文对它等于不可用；模型能察觉附近有个数字（开头几位对），但读不出完整值。

后来用 29,293 token 的深度细扫把这条边界钉到了小数点：深度 50% 失败、56% 通过，而
16384/29293 = 55.9%，正好是第 1 块和第 2 块的分界。

最初据此推断问题在 DSA indexer 的 index-K cache 上（"第 1..n-1 块写入的 index-K
不在 top-k gather 预期的位置上"）。**这个归因是错的**，下面的对照实验推翻了它。

#### 关键对照：同样的分块，去掉 PD 就全过

用完全相同的 DSA 设置（`tilelang` 前后端、`fp8_e4m3` KV、DP attention、每 rank
16384 chunk、`MEM_FRAC=0.80`、同样 1,949,568 的 KV pool）起一个**聚合式非 PD**
单机 server，对 58,695 token 的 prompt 确认调度器真的切了 4 块
（16384×3 + 9472）、indexer 走的也是同一条 `need_chunk=False` 路径：

| 部署 | 29,293 深度 10/50 | 58,695 深度 10/50 | 合计 |
|---|---|---|---|
| PD 分离 | 失败 / 失败 | 失败 / 失败 | 5/9 |
| 聚合式非 PD | 通过 / 通过 | 通过 / 通过 | **9/9** |

**分块 prefill 和 DSA indexer 本身是正确的**，内容是在传给 decode 腿的路上坏掉的。

配套的排除工作也都是实测而非走读：PAGED top-k transform 在五种分块形状上与 torch
参考实现逐位一致（包括上游自己测试里跳过的 `row_starts` 情形）；aiter 的
`fp8_mqa_logits` 与行数无关（一次性 launch 和按行切片 launch 在三种真实形状上只差
fp8 舍入，0 行不一致）；在线打点确认第 2 块的 metadata 完全正确（`ks=0`、`ke` 一直
到 29,293、458 页的 page table 覆盖整个上下文）；两条腿都没有 DSA state 页表长度
不匹配的告警。

#### 根因：mooncake 忽略了 early-send 的 CUDA 屏障

`prefill.py` 在把"可能还在被 forward 写"的页交出去之前，会记录一个 CUDA event 当
屏障（early-send 路径，`prefill.py:1047-1055`，注释写得很清楚：*"Record a completion
event now so the transfer worker can wait on those writes before the RDMA read,
instead of racing them."*）。但**这个 event 只有 `mori` 后端会读取**——
`mooncake/conn.py` 整个文件里没有任何 `wait_event` / `synchronize()`。我们用的是
mooncake，所以这道屏障从来没生效过。

而且 overlap 调度下真正搬运非最后一个分块的那次传输（`process_batch_result_disagg_prefill`
里的 `send_kv_chunk(..., last_chunk=False)`）连 event 都没有记录。于是这些页的 RDMA
读取与写它们的 forward 之间**完全没有同步**，读到了半写状态。最后一块永远是对的，
因为它走采样路径，前面有 `copy_done.synchronize()`（`prefill.py:619`）真正卡住了主机。

这条解释对上了全部现象：只在 PD 下出现；只有非末尾分块损坏；损坏边界精确落在分块
分界；块大小和 indexer 快慢路径会改变结果（它们只改变 forward 耗时，不改变算术）；
以及部分损坏——`2013705` 返回 `2013701`，只错最后一位。

最有说服力的一条证据是 send 追踪：失败请求的 prefill 侧其实完全正确，第 1 块送
page 2–257 覆盖 `[0,16384)`、第 2 块送 258–459 覆盖 `[16384,29293)`，严丝合缝无空洞
无重叠，index-K 覆盖全部 458 页，而且第 1 块的数据从发送到最后一块期间校验和保持
**不变**。送什么、送多少都没错，错的是**什么时候读**。

#### 修复

三处改动（都在 SGLang 侧），本质上是把 `mori` 早就有的处理照搬到 `mooncake`：

| 文件 | 改动 |
|---|---|
| `disaggregation/common/utils.py` | `TransferKVChunk` 增加 `wait_event` 字段，让屏障能随工作单元传递 |
| `disaggregation/mooncake/conn.py` | `send()` 取出并透传 event；`add_transfer_request()` 接收；`transfer_worker` 在读设备内存**之前** `synchronize()` |
| `disaggregation/prefill.py` | overlap 下那个原本完全没有屏障的非末尾分块传输，也记录 `forward_stream` 的 event |

`_early_send_wait_event` 这个属性的读写方现在是对称的：写在
`prefill.py:1055`（上游原有的 early-send 路径）和 `prefill.py:761`（我们补的 overlap
非末尾分块），读在 `mori/conn.py:1433`（上游原有）和 `mooncake/conn.py:1816`（我们补的）。
在补之前，mooncake 侧是四个点里唯一缺的那个。

**已落地成补丁**（此前只是容器里的未提交改动，丢了不会报错、只会让 18k 以上的 prompt
重新静默返回半个 needle）：
`deploy/docker/patches/sglang_disagg/mooncake_early_send_wait_event.diff`，
用 `patch_sglang.sh` 幂等应用，两条 leg 脚本在它缺失时**直接拒绝启动**——和 §1.1 的
mooncake gate 同一套路。补丁是从容器现场导出的，反向应用能干净通过，也就是说它与当前
跑着的代码字节级一致。

#### 验证

在**原先失败的那套配置**上（overlap 开启、每 rank 16384、`MEM_FRAC=0.80`），日志确认
仍是真正的 4 段分块：

| 测试 | 修复前 | 修复后 |
|---|---|---|
| needle 三长度 × 深度 10/50/90 | 5/9 | **9/9** |
| 29k 深度细扫（5,20,35,44,50,56,62,75,95） | 4/9 | **9/9** |

原先失败的深度 5/20/35/44/50 全部通过，分块边界消失。全量套件：needle 9/9、
长上下文 HumanEval 20/20、其余全 100%，结论行 `结论: 全部通过`
（短上下文 HumanEval 掉 1 个，是生成代码的采样抖动——上一轮它反而是长上下文掉 1 个）。

**因此 `CHUNK=524288` + `MEM_FRAC=0.80` 那套绕过可以撤掉**：chunk 大小和显存比例都恢复
自由，也不需要牺牲 prefill 的 overlap 调度。本轮验证跑在 `MEM_FRAC=0.80`
（KV pool 1,949,568）上；0.85 的 OOM 当初是大单块造成的，回到 16384 的 chunk 后应该
可以恢复 0.85 拿回完整 KV pool，但还没重测。

另外补一条：这个 bug 与 DSA 无关，任何在 mooncake + overlap 调度下跑分块 prefill 的
PD 部署都会中，只是 DSA 的稀疏检索让它以"取回部分数字"这种显眼方式暴露出来。值得提
到上游——最接近的既有报告 [#25583](https://github.com/sgl-project/sglang/issues/25583)
（GLM-5-FP8 + NSA + 70k prompt，症状一致）因无人跟进被自动关闭，我们这轮的非 PD
对照实验能让报告站得住。

早先的一个假设也被否掉：`--disable-chunked-prefix-cache` 在这里是空操作。那个 flag
控制的是另一套只用于 MHA 的机制，而 `dsa` 不在
`CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS` 里（`server_args.py:133-142`），
所以 `maybe_disable_chunked_prefix_cache`（`misc_utils.py:16-37`）早就为这个模型
强制关掉它了。实测也确认：带上这个 flag 跑一轮，失败原样复现。

#### 为什么单机配方从没暴露这个问题

`run_sglang_mtp.sh` 默认 `ENABLE_DP_ATTENTION=0`，而 SGLang 只在开启 DP attention
时才把 `chunked-prefill-size` 除以 `dp_size`。所以那套配方跑的是 131072 的 chunk，
任何低于这个长度的 prompt 都一次 prefill 完——它**根本没走过分块 prefill**。
我们的 PD 配置开了 DP attention，chunk 变成 131072/8 = 16384，于是每个超过约 16k
的 prompt 都开始分块。再叠加上"单机没有 PD 传输"这一层，这个竞争在单机配方下**不可
能出现**：它需要分块 prefill 和 PD 传输同时存在。

顺带修正上面那张长度扫描表的读法：并不是"装不进一块就必然失败"。竞争是否咬中取决于
时序——`CHUNK=524288` 下 117,159 token 的 prompt 确实切了两块，深度 10% 的 needle
落在第 1 块里却能正确取回，因为 65536 的大块把 forward 拉长、避开了竞争窗口。当初
以为大 chunk"修好了"问题，其实只是把它藏了起来。

---

## 4. 吞吐基线，以及单条 RDMA 轨道是否成为瓶颈

`bench_rails.sh` + `sglang.bench_serving`，输入 8192 token / 输出 128 token，
`--random-range-ratio 1.0`，每个并发档 5 个 prompt，经由 router。
单轨道（`IB_DEVICE=rdma0`），MTP 关闭。原始数据：`bench_rail_rdma0/`。

> **适用范围。** 这一节的数字全部取自 8k 输入 / 128 输出、每并发档 5 个 prompt 这一种
> 负载形状，且都在 §3.1 的补丁之前。它足以回答"单条轨道会不会成为带宽瓶颈"，但**不足以
> 当作 code agent 那种长上下文、多轮共享前缀、长输出场景下的结论**。待测项见 §6。

| 并发 | req/s | 输出 tok/s | 平均 TTFT | p99 TTFT | 平均 TPOT |
|---|---|---|---|---|---|
| 16 | 1.00 | 128 | 6.6 s | 12.7 s | 65.4 ms |
| 64 | 1.84 | 235 | 22.8 s | 39.6 s | 75.1 ms |
| 256 | 2.09 | 268 | 100.4 s | 124.0 s | 77.0 ms |

三轮下来两条腿都零 RDMA 传输失败。

同样负载打到自动发现的轨道上（不设 `IB_DEVICE`，每个 rank 都发现全部 8 张 HCA，
两条腿都在 rdma0–rdma7 上建了 context）：

| 并发 | req/s | 输出 tok/s | 平均 TTFT | p99 TTFT |
|---|---|---|---|---|
| 16（冷） | 0.57 | 73 | 17.7 s | 37.3 s |
| 16（热，复测） | 0.53 | 68 | 19.1 s | 29.0 s |
| 64 | 1.64 | 210 | 25.1 s | 38.0 s |

同样零 RDMA 错误，说明多轨道在这套 fabric 上确实能正常建 QP——但在 c=16 下吞吐只有
钉死单轨道的**一半**（0.53 vs 1.00 req/s，TTFT 19.1 s vs 6.6 s）。热态复测排除了
JIT 预热这个解释。合理的解释是：KV 只需要单条轨道约 3% 的带宽（见下），摊到 8 条轨道上
买不到任何带宽，反而要付 8 倍的 QP 建连和分片开销。

但**这还不足以当成"钉单轨道更快"的结论**。差距主要来自 c=16 那一档，而每个并发档只有
5 个 prompt——这个样本量报 p99 TTFT 没有统计意义；而且到 c=64 差距已经收窄到
1.64 vs 1.84，趋势是并发越高越接近。所以这里只记录为"在 8k/128 这一种形状下的观测"，
是否普遍成立要等真实负载实测。

请求吞吐在 2.1 req/s 附近饱和，而 TTFT 随并发近似线性增长——瓶颈是 prefill 腿，
其余都在它后面排队。TPOT 稳定在约 77 ms，说明 decode 腿远未吃满。

这也回答了最初关于是否该钉 `IB_DEVICE=rdma0` 的问题。饱和时 prefill 腿产出
`2.09 req/s × 8192 tok = 17,120 tok/s`，而 GLM-5.2 的 KV 是 53.6 KB/token
（MLA latent `(512+64)×78 = 43.9 KB` + DSA indexer K `128×78 = 9.8 KB`，fp8）。
所以 KV 出口峰值是 **0.92 GB/s = 7.3 Gb/s**，而这里用 `ib_write_bw` 实测单条轨道是
**229.5 Gb/s**：约占 **3.2%**。要打满一条轨道需要约 535,000 prefill tok/s，大概是
这套硬件跑约 700B MoE 所能产出的 30 倍。

这个 3.2% 对"上下文变长"本身是稳的：KV 出口 = prefill token 吞吐 × 53.6 KB/token，而
prefill token 吞吐是算力受限的，跟这些 token 是摊在 5 个 100k 请求上还是 60 个 8k 请求上
基本无关。单纯把 8k 换成 128k 不会动这个数字。

真正可能推翻它的不是长上下文，而是 **agentic 的前缀复用**。多轮之间共享长前缀时，prefill
侧命中 radix cache 就不必重算，但 decode 腿需要的是**整个上下文**的 KV。一旦 decode 侧没有
复用（`pop_decode_prefix_len` / `req_to_decode_prefix_len` 这套机制正是为了让 decode 上报
"这段前缀我已经有了"、从而让 prefill 少发），传输量就和 prefill 算力**解耦**了——prefill
只算 10k 新 token，却要搬 100k 的 KV。命中率 90% 时 KV 出口约翻 10 倍，3.2% 变成约 32%：
仍然没打满一条轨道，但已经不能当作可忽略，而且 p99 的恶化会比饱和早得多出现。这是 code
agent 形态负载必须实测的一项，和 §5 的 kv-aware 是同一件事的两面。

所以在 1P1D 下单条轨道不是吞吐瓶颈。**钉死一条的可靠理由是正确性，不是性能**：8 条轨道
每条都是通往 leaf 交换机的独立 /31（prefill 节点上是 `10.115.46.101/31`、`.111/31`
……；decode 节点上是 `10.115.45.x`），而 Mooncake 的自动发现会给每个 GPU 分配它
NUMA 本地的 HCA（rdma0-3 → NUMA 0，rdma4-7 → NUMA 1，两节点镜像对称）。如果一个
prefill GPU 和它的 decode 对端落在互相不可路由的轨道上，RoCE QP 永远到不了 RTR，
在压力下就会超时。

Infera 本来有针对这种情况的保护（`apply_mooncake_topology_default`），但它在本集群上
**永远不会触发**，已实测确认：

```
rdma0  10.115.46.101   →  tw-eth0  10.115.46.101/31
rdma1  10.115.46.111   →  tw-eth1  10.115.46.111/31
...                       （共 8 条，每条一个独立 /31）
_active_rdma_nics() 报告的 subnets: ['10.115.46.0/24']   ← 只有 1 个
```

`_active_rdma_nics`（`infera/engine/rocm_rdma_env.py:111-133`）用
`ip.rsplit('.',1)[0] + '.0/24'` 硬编码按 /24 分桶，八张卡于是归成同一个子网，
`apply_mooncake_topology_default` 的短路条件 `len(subnets) <= 1` 命中并直接返回 `None`。
它防的是 NIC 跨真正不同网段（`172.30.x` / `10.245.x`）那种形态，而这套 fabric 的风险是
"同一个 /24 里 8 条互不可直达的 /31"——结构上覆盖不到。

**所以现在保护我们的只有脚本里写死的 `IB_DEVICE=rdma0` 这个约定，不是那个 guard。**
一旦有人不设 `IB_DEVICE` 就会退回自动发现，跨 /31 配对的风险重新出现，而且没有任何东西
会发出警告。修法见 §6。

---

## 5. 路由：round-robin → KV-aware

已改进脚本，但**尚未在硬件上验证**——这需要的 leg 重启没有执行。

Infera 只提供两种策略：`round-robin` 和 `kv-aware`
（`infera/router/policy/factory.py:56-67`）；`kv-aware` 其实是上游默认值，是这些
脚本主动把它覆盖成了 `round-robin`。

值得知道的一点：1P1D 下每个角色只有一个 worker，看起来根本没得选。其实有。
rank 多路复用的 worker 会展开成每个 DP rank 一个目标
（`infera/router/policy/target.py:45-54`），所以在 DP8 下策略是在 8 个 prefill rank
和 8 个 decode rank 之间按前缀局部性和负载做选择——`DisaggRouter.dispatch` 会带着
`role_hint` 对每条腿各调用一次 `policy.pick()`（`infera/router/disagg.py:130-136`）。
这恰恰是 agentic 流量真正在意的那个旋钮，因为连续几轮对话共享很长的前缀。

它的打分是 `overlap_weight × (request_blocks − hits) + active_blocks`，其中 hits
来自 router 侧维护的、对每个 worker 已缓存 block 哈希的镜像，由引擎通过 ZMQ 推送
KV 事件来喂（`infera/router/kv_event/client.py:79-103`）——不是 router 侧的 radix
树。所以两条腿必须开启事件推送，而我们启动时用的是
`--no-enable-kv-events --kv-events off`。

现在脚本里的改动：

- `infera_1_server.sh`：`--router-policy ${ROUTER_POLICY:-kv-aware}`，外加
  `--kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0`（prefill 权重
  高 10 倍——prefill 命中能跳过整个 prefill，decode 命中只省一点负载）。
- `infera_2_sglang_prefill.sh`、`infera_3_sglang_decode.sh`：`KV_EVENTS=1` 会把关闭
  的 flag 换成 `--enable-kv-events --kv-events on --kv-event-transport zmq`。
  默认仍是关闭。

decode 侧刻意默认保持关闭：只要开了 kv 事件，Infera 就会给 mooncake decode worker
追加 `--disaggregation-decode-enable-radix-cache`
（`infera/engine/sglang/args.py:257-263`），而 SGLang 不接受这个 flag 与投机解码
共存，会和 MTP 冲突。decode 事件关闭时 `kv_block_size` 是 `None`，decode 的命中恒为
0，decode 路由退化成纯按负载——prefill 仍然拿到完整的前缀感知路由，收益本来也在
那一侧。

启用后如何验证：`/metrics` 上的 `infera_router_pick_cache_hits` 和
`infera_router_pick_request_blocks`，router 日志里的
`pick policy=kv-aware ... cache_hits=N request_blocks=M` 那一行，以及
`GET /v1/admin/cache-view/{worker_id}?dp_rank=N` 查看每个 rank 的 block 数。

### 5.1 单机 pd-mixed 是可行的验证路径（也是唯一能和 MTP 同时跑的）

上面那个 MTP 冲突**只作用于 decode 角色**，所以单机 pd-mixed（`disaggregation_mode` 为
`"null"`）下同时开 dp-attention + MTP + kv-aware 是合法配置。两侧的判断都在 decode 分支内：
Infera 追加 flag 的条件含 `disaggregation_mode == "decode"`
（`infera/engine/sglang/args.py:257-263`），SGLang 那段 `speculative_algorithm is not None`
的拒绝也整段包在 `if server_args.disaggregation_mode == "decode":` 里
（`sglang/srt/arg_groups/pd_disaggregation_hook.py:29,41-46`）。mixed 下两者都不触发，而普通
radix cache 本来默认就开着——不需要那个 flag 就有前缀缓存和 KV 事件源。

（旁证：同一个文件 49-53 行在 decode radix cache + dp-attention 时会打
`EXPERIMENTAL: ... Requires prefix-aware DP rank routing for optimal cache hits`。上游自己
就把"按 DP rank 做前缀感知路由"当作这条路的前置条件。）

整条链在 mixed 下没有 PD 专属分叉，已逐段核对：

| 环节 | 依据 | mixed 下是否相同 |
|---|---|---|
| 引擎发 KV 事件 | `scheduler.py:447-452`（只看 `kv_events_config` + `attn_tp_rank==0`）、`1811-1817`（事件带 `attn_dp_rank`） | 是，不看 disagg 模式 |
| 注册为 MIXED | `infera/engine/sglang/worker.py:28-33` | `"null"/None → DisaggMode.MIXED` |
| 按 rank 多路复用注册 | `worker.py:122-136`（`dp_size>1` 时注册 size、`dp_rank` 留 None）、`75-77`（按 dp_size 分配端口段） | 是 |
| 展成 per-rank target | `infera/router/policy/target.py:45-54` | 是 |
| 转发 DP rank | `infera/router/mixed.py:113` → `X-Data-Parallel-Rank`；SGLang `serving_base.py:277-306` 认这个头 | 是 |

**一处必须注意：`role_hint` 走不到。** `mixed.py:87` 调 `pick()` 不传 `role_hint`，
`_base_weight_for(None)` 返回 `self._w`，即 `--kv-overlap-weight`（默认 1.0）。所以
`--kv-prefill-overlap-weight 20.0` / `--kv-decode-overlap-weight 2.0` 在 mixed 下**完全不生效**
——那两个只在 `DisaggRouter.dispatch` 对两条腿各调一次 `pick()` 时才用。mixed 下要调的是
`--kv-overlap-weight`。

能验到的：dp-attention 下事件是否真按 rank 发出、router 的 cache-view 镜像是否按 dp_rank
正确填充、`expand_targets` + 头部转发是否端到端生效，以及最关键的——agentic 形态流量下前缀
局部性路由是否真比 round-robin 命中得多。附带能拿到一个开着 MTP 的吞吐数，这是 PD 目前因
§1.2 死锁给不出来的。

验不到的：上面那条 `role_hint` 双腿分权；以及 `pop_decode_prefix_len` 和 decode 侧 radix
cache——mixed 根本没有 KV 传输，所以 §4 里"前缀命中率一高、传输量就和 prefill 算力解耦"
那个带宽问题完全碰不到。它也不解开 MTP 在 PD 上的两个结（§1.2 的死锁是独立的，decode radix
cache + 投机解码那条硬拒绝也还在）。**这是绕过，不是修掉。**

落地还差一步：`sglang_naive_engine.sh` 已经是 MTP + dp-attention 的单机配置，但它是裸 SGLang
起的，既没有 Infera 的 mixed worker 注册也没开 kv 事件，router 看不到它。需要改成走
`infera.engine.sglang`（不设 `--disaggregation-mode`）、加
`--enable-kv-events --kv-events on --kv-event-transport zmq`，再配一个
`--router-policy kv-aware --kv-overlap-weight N` 的 router。

**这条路径必须先打 §1.3 的补丁。** 这里要验的正是"dp-attention + MTP + 多轮并发"，
而 §1.3 那个 DSA indexer 行数不匹配的触发条件就是"多个 DP rank 请求数接近但不相等"——
不打补丁的话，加压的第一批 batch 就会崩，验不到路由。补丁现在在
`deploy/docker/patches/sglang_dsa/`，`bash patch_sglang.sh` 应用。

**还得显式传 `--max-running-requests`。** 一开 MTP，SGLang 就把它硬编码成 48、再除以
`attn_dp_size` 变成每 rank 6（§1.2）。6 个请求根本压不出"多个 rank 请求数接近但不相等"
之外的任何有意义的路由行为，量到的吞吐也是这个默认值的天花板而不是硬件的。DP8 下想要
每 rank N 就传 `8 × N`。

---

## 6. 未完成事项

1. **MTP 已经跑通，剩下的是三件收尾。** PD + MTP=1 现在端到端可用，正确性套件全过、
   accept len 3.78/4（§1.2）。代价是两个硬前置：`sglang_dsa` 补丁（§1.3）和
   `index_share_for_mtp_iteration=false` 覆盖（§1.2），两条腿脚本在 `MTP != 0` 时会自动
   加后者、缺前者则拒绝启动。剩下要做的：
   - **补一次公平的吞吐对比。** 现有那组数字里 MTP=0 用的 `max_running_requests=2048`、
     MTP=1 是塌缩后的 48，差 42 倍，所以只有 TPOT 那一行可比（§1.2 性能小节）。两边都
     显式传 `--max-running-requests` 重跑一遍即可，需要重启两条腿。
   - **把 aiter/HIP 那份 padded-rows 补丁提上去。** NPU（#32762）和 TRT-LLM（#32209 第 4
     项）都在修同一个 bug，HIP 这条路径上游没人做，形状照着 #32762 写（§1.3）。
   - **盯 #32209。** 它一合，上面两个前置都可以撤掉。同时值得把我们那条反例回给上游：
     作者认为"prefill 侧也开 MTP 就能避开死锁"，我们两条腿都开着仍然死锁（§1.2）——但要
     提就得重跑一轮留日志证据。
   `max_running_requests` 塌到每 rank 6 这一项**已经解释清楚并且不再是障碍**：是
   SGLang 对投机解码硬编码的 48 再除以 `attn_dp_size`，显式传
   `--max-running-requests` 即可（§1.2）。
   注意这一项和 §5 在 **PD 下**相互牵制，因为 decode 侧的 kv 事件会引出
   `--disaggregation-decode-enable-radix-cache`，而它和投机解码是硬冲突。单机 pd-mixed
   不受这个牵制（§5.1），所以"MTP + kv-aware 同时开"可以先在那边验——但那边同样要显式
   传 `--max-running-requests`，否则量到的是 48 这个默认值的天花板而不是硬件的。
2. **把 PD 传输的竞争 bug 提给上游。** §3.1 已经修好、验证，并且落成了
   `deploy/docker/patches/sglang_disagg/`（连 README 带复现材料），但还没提上去。
   复现材料很干净：mooncake + overlap 调度 + 分块 prefill，needle 放在最后一块之前，
   再加上"聚合式非 PD 同配置 9/9"这个对照实验。可以引用被自动关闭的
   [#25583](https://github.com/sgl-project/sglang/issues/25583)。上游 diff 里值得一并
   指出的是：`_early_send_wait_event` 的四个读写点里，补丁之前 mooncake 侧的消费者是
   唯一缺失的那个，`mori` 一直是对的——这让它更像"漏了一处"而不是"设计如此"。
3. **试试把 `MEM_FRAC` 调回 0.85。** §3.1 的修复验证跑在 0.80（KV pool 1,949,568）。
   当初 0.85 的 OOM 是 65536 大单块造成的，回到每 rank 16384 后应该能恢复 0.85 拿回
   完整的 2,194,496 pool，但还没重测。
4. **用真实负载形状重跑吞吐测试。** §4 全部取自 8k/128、每档 5 个 prompt、补丁之前，
   只能算一种形状。需要补的是 code agent 的形态：长输入（64k–128k）、长输出、多轮之间
   共享长前缀；每档样本量从 5 提到 ≥50，否则 p99 TTFT 没有意义；并且直接读 mooncake /
   轨道计数器测真实 KV 出口字节数，而不是从 req/s 反推。
5. **端到端验证 kv-aware**，用 agentic 形态的负载（多轮之间共享长前缀）实测命中率，
   而不是想当然。这件事和第 4 项是同一个实验：prefill 命中率一高，KV 传输量就和 prefill
   算力解耦，§4 那个 3.2% 的带宽占用才会真正受到考验。分两步走：
   - **先在单机 pd-mixed 上验**（§5.1）。MTP 冲突只作用于 decode 角色，所以 mixed 下
     dp-attention + MTP + kv-aware 可以同时开，不需要重启 PD 两条腿，而且顺带能拿到一个
     开着 MTP 的吞吐数。它覆盖事件按 rank 发出、cache-view 按 dp_rank 填充、`expand_targets`
     + `X-Data-Parallel-Rank` 端到端，以及"前缀局部性路由是否真比 round-robin 命中得多"
     这个核心问题。需要先补一个 mixed 启动脚本（§5.1 末段）。
   - **再回到 PD 验剩下两块**：`role_hint` 双腿分权（mixed 下 `pick()` 不传 role_hint，
     20.0/2.0 走不到），以及 `pop_decode_prefix_len` / decode radix cache 对传输量的削减
     ——mixed 没有 KV 传输，碰不到这块。
6. **轨道钉死：正确性上已定，性能上待复测。** 钉 `IB_DEVICE=rdma0` 的理由是每轨道 /31
   的可路由性（§4），这条与负载无关、不需要再议。但"单轨道更快"只在 c=16 / 8k 这一档
   观测到，且 c=64 差距已收窄——要在第 4 项的负载下重做 A/B，重点 c=64 和 c=256。
   另外，部署规模一旦超过 1P1D，/31 的配对问题会重新浮现。
7. **修 `apply_mooncake_topology_default` 的分桶粒度。** 它按硬编码 /24 判断 NIC 是否跨
   网段，在本集群上八条 /31 被归成一个子网，保护静默失效（§4 已实测）。应该改成按接口
   实际配置的前缀长度判断可达性；至少要在"走自动发现且存在多个点对点前缀"时打一条告警，
   否则这是个不会报错的保护缺口——目前只有脚本里的 `IB_DEVICE` 约定挡着。
8. **量化 §3.1 补丁打开的性能空间。** 补丁之前要靠 `--disable-overlap-schedule` 或 65536
   大分块来回避竞争，两者都有吞吐代价；现在 overlap 可以开着、分块大小也回归成纯性能
   参数。反过来也要如实测一下补丁自身的开销：transfer worker 里新增的那次
   `wait_event.synchronize()` 理论上会有成本。
