# 3P1D 调优计划（2026-10-13 讨论稿 v2）

依据：`REVIEW.zh-CN.md` §2（瓶颈数据链）、§6（decode 容量）、§7（调优项优先级）、§8（decode 刚重启后第一次跑偏慢 13–17%）。

v2 相对 v1 的修改（10-13 04:40 讨论）：去掉阶段 1 末尾的基线复测、去掉 R3；阶段 2a 加入 prefill mem 0.85 → 0.90；
确认阶段改为与 r2 C128 比较（理由见下）。

## 测试协议（所有阶段通用）

- 每一轮（每个变体的每个并发点）：以该轮开关重启 router → `/flush_cache` decode 与 3 个 prefill（prefill 忙时 400，重试）→
  确认之后首批 prefill `#cached-token` 为 0 → AgentX full（warmup 10/lane，3600 s，seed 42）→ 发布结果 + census。
- 引擎尽量不重启；凡重启过 decode，先跑 1 轮不计分预热。
- 漂移控制不靠重复测基线，而是每轮核对两个与 router 无关的引擎指标：按最忙 rank 对齐的 decode 每步耗时
  （`results/*/decode_step_matched_load.txt` 的做法）、prefill 每个新算 token 的 batch 耗时。与阶段 1 基线一致即视为栈未漂移。
  重启过 prefill 后，第一轮也要核对 prefill 指标，偏慢则加 1 轮预热。
- 判定：差异 < ~2% 视为噪声；≥ 3% 且机制指标方向一致则接受；介于其间或机制指标不一致时，只对该变体补测一轮。
  吞吐同时看总 tok/s/GPU、完成请求数、输出 tok/s/GPU（平均输入长度差异会抬高总吞吐）。
  机制指标：prefill 排队、3 个 P 实例均衡、TTFT、GPU/host 缓存命中、prefill 新算 tok/s、decode 有效并发、decode KV>90% 时间。

### r2 数据的用法

r2 各点之间没有 flush（prefill 日志无 flush 记录），C160/C176/C128/C112 都带着上一轮的前缀缓存起步；C144 是 decode 重启后第一轮。
所以 r2 只作参照，不作 A/B 基线：

- 阶段 1 的基线（已 flush、decode 已预热）与 r2 C144 的差，可量化 decode 刚重启的影响。
- 确认阶段的 C128：热缓存起步只会有利于 r2，所以“最优配置（已 flush）明显胜过 r2 C128”是保守成立的结论；
  只有差距落在噪声内，才补测一轮已 flush 的 C128 基线。

## 阶段

| 阶段 | 内容 | 重启范围 | 测法 | 约耗时 |
|---|---|---|---|---|
| 1 | 基线 → +R1（`INFERA_PD_PREFILL_GUARD_RELEASE=completion`）→ +R1+P 会话亲和（`INFERA_SESSION_AFFINITY=prefill` + 客户端 `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1`） | router | C144，各 1 轮 | 4.5 h |
| 1b | 最优 + R2（`INFERA_R2_DECODE_DEMAND=on`） | router | C144，1 轮 | 1.5 h |
| 确认 | 最优 @C128，对照 r2 C128（见上）；若最优在 C144 超过 C128，再测 C160 | router | 1–2 轮 | 1.5–3 h |
| 2a | prefill 前缀缓存扩容：mem 0.85 → 0.90 + HiCache 137 两实例 200 GB/rank、135 实例 400 GB/rank（`SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512`） | 仅 3 个 prefill（decode 保持预热） | 峰值附近 2 点；census 盯 OOM/MAF | 3.5 h |
| 2b | `--disaggregation-decode-allocation-policy prefill_complete`（两角色，`config/round.r3.sh`） | 整栈 | 真实接受起栈 → 两层门禁 → 切模拟接受 → 1 轮预热 → 2 点 | 5 h |
| 3 | decode mem 0.95 / decode graph bs 64 | decode + 预热 | 仅当 decode KV 成为约束 | 3.5 h |
| 4 | D radix + D 绑定（移植 PR #184 引擎补丁到 v0.5.21 + 长前缀正确性） | 整栈 | 仅当 prefill_complete 后传输/预分配仍是瓶颈 | ~1 天 |
| 5 | 增加 prefill 实例 | 另找节点 | 前述完成后仍 prefill 瓶颈 | — |

阶段 1–2b 合计约 16–17.5 h（v1 约 21 h）。

不做：R3 分层缓存评分（以前旁路观察 1,139 次决策 0 次改选；C144 host 命中只占输入 4.5%；开 P 会话亲和后续轮已被绑定，
R3 只影响新会话；还需另行确认 v0.5.21 是否上报 host 层事件）、D HiCache（−4.5%）、R4（−34%）、Dynamo 式首次评分（−1.9%）、8K chunk（−4.5%）。

每阶段以前一阶段最优为新基线；峰值移动时测试点随之移动；每阶段结束写入结论（接受/拒绝/补测）再进入下一阶段。

## 2a 的依据：prefill 显存余量（r2 栈实测）

每卡 288 GiB。prefill TP4：权重 119.8 GiB/卡，GPU KV 116.8 GiB = 2,270,656 tokens/rank（TP8 时约 3.14M，TP4 权重翻倍，KV 少约 28%）；
建池后预留 41.7 GiB（0.15 × 278 GiB）。r2 跑完 5 个点 + 门禁后，prefill 卡仍空闲 19.8–23.8 GiB（期间 prefill 未 flush，
分配器缓存未释放，可视为高水位），即预留实际最多用到 ~21.9 GiB（prefill-1 最紧）。

| mem | GPU KV/rank | 预留 | 最紧 rank 余量 |
|---|---|---|---|
| 0.85（现） | 2.27M | 41.7 GiB | ~20 GiB |
| 0.88 | ~2.43M（+7%） | 33.4 GiB | ~11.5 GiB |
| 0.90 | ~2.54M（+12%） | 27.8 GiB | ~5.9 GiB |
| 0.92 | ~2.65M（+17%） | 22.2 GiB | < 1 GiB，不用 |

- 与 P8D8 C80 的 C1（P GPU KV +8% 未见收益）不同：那里 miss 只有 ~5%；本次 C112 → C144 GPU 命中 91.1% → 74.8%，
  host 命中只 1.3% → 4.5%，理论命中 96.2%，缺口 ~17 个百分点。
- 一部分 miss 可能来自会话在 12 个 rank 间漂移，这部分扩容无效；阶段 1 的 P 会话亲和结果可以把“局部性”与“容量”分开。
- HiCache 现为 ratio 1.5，跟随 GPU KV：mem 0.90 时自动到 ~171 GB/rank；目标 ~200 GB/rank（ratio ≈ 1.75，< 256 GiB 单次注册），
  137 上 8 rank 共 ~1.6 TB（内存 2.7 TB）。
- 若出现 OOM，回退 0.88。
- 135 只有 1 个实例（4 rank），可以更大：可用内存 1,856 GB，yihou e1c 在 v0.5.21 上跑过 450 GB/rank（`CHUNK_GB=512`，单次 `hipHostRegister`）。
  DSA indexer 的 host 区按 token 数另加约 23%：400 GB/rank 实际 ~492 GB/rank，4 rank ~1.97 TB，余 ~640 GB；450 时余 ~400 GB。
  135 上还有其他租户（GPU1 vLLM）且 GPU 1/2/3/7 空闲可能被占用，取 400 GB/rank 留余量。137：200 GB/rank 实际 ~246 GB/rank，8 rank ~1.97 TB，余 ~680 GB。
- 推进方式（10-13 04:44 用户确认）：按判定标准自动推进；人工持续巡检（`scripts/tune_watch.py`），不只依赖脚本门禁。
- 同口径 decode：建池后预留 ~41 GiB，高水位空闲 30.8–32.7 GiB（实际用 ~8–10 GiB）；mem 0.95（预留 ~13.6 GiB）可行，
  但 decode KV 目前不是约束，留在阶段 3。

## 需要的脚本改动

`restart_router.sh` 支持传入 router 开关；flush + 校验脚本；harness 在会话亲和轮把 session header 环境变量写入客户端 runtime.env；
只重启 prefill 的脚本（2a）；按阶段串行执行的流水线。

## 阶段 1 后的调整（10-13 09:20）

阶段 1 结果（C144）：t1-base 19,503 → t1-r1 19,308（噪声，保留 R1）→ t1-r1aff **21,639**（+12.1%，接受）。
P 会话亲和后 prefill 缓存份额 94.0%（理论 96.3%）、新算量 −42%、排队 ~1.4；decode 每 rank 在跑 7.0、KV>90% 6%、等传输 2.5，
瓶颈从 prefill 转到 decode（ITL p90 25 → 30 ms）。据此：

1. 1b（R2）照常跑完（已开始）。
2. 确认阶段改为向高并发推进：最优配置测 C160；C160 > C144 再测 C176（ITL p50 14 ms，距 P50 interactivity 50 的 20 ms 还有空间）。
   C128 只在高并发全部低于 C144 时补测。
3. 2a（prefill 扩容）、2b（prefill_complete）在并发扫描后按新峰值处的数据决定：峰值处缓存份额明显低于理论值或 prefill 排队回升 → 2a；
   decode 等传输 / 等预分配回升 → 2b。

## 协议补充（10-14 02:50，瓶颈误判后的教训）

下"瓶颈在哪一侧"的结论之前，必须先有两类直接证据，数据互相矛盾时先查清原因，不换测试点：
1. TTFT 分段：用引擎 Prometheus 直方图在测量窗口内做差分 —— prefill `queue_time_seconds`、`kv_transfer_bootstrap_ms`、
   `kv_transfer_latency_ms`、`e2e_request_latency_seconds`；decode `kv_transfer_alloc_ms`、`queue_time_seconds`、
   `time_to_first_token_seconds`、`inter_token_latency_seconds`。
2. 进程资源：decode / prefill 主进程（HTTP + TokenizerManager）、detokenizer、各 scheduler 的 CPU；GPU 侧 KV 使用率与每步耗时。
"负载最重的一侧"不等于"瓶颈"：只有该侧资源到上限（KV 满、ITL 到约束、进程 CPU 饱和、队列持续增长）才称瓶颈。

## 借鉴 InferenceX B300 Dynamo 42k 的实验候选（10-15 06:50）

来源：`../inferencex-glm52-b300-dynamo-agg-20261015/README.zh-CN.md`（B300、4 卡 aggregated TP4 + DP4 attention，C48 = 42,047 tok/s/GPU）。
补充 InferenceX API（2026-10-15 取）里 MI355X 上 GLM-5.2 AgentX 的官方数据，作为同硬件参照（aggregated，每卡分母为实例卡数）：

| MI355X 配置 | C16 | C24 | C32 | C40 | C48 |
|---|---|---|---|---|---|
| ATOM TP4（4 卡，显存 0.95，KV fp8 + indexer fp4，无 HiCache） | 16,923 / 84.6 | 25,737 / 65.3 | 29,924 / 55.5 | 32,778 / 44.5 | 34,195 / 35.1 |
| sglang TP4（4 卡，无 DP attention，HiCache 180 GB） | 18,327 / 87.7 | — | — | — | — |

（格式：tok/s/GPU / P50 交互速度。sglang TP4 在 C12 以后 GPU 命中从 0.95 掉到 0.74：没有 DP attention 时，MLA 的 KV 在每个 TP rank 上各存一份完整副本。）
对照我们 20 卡 3P1D 的 C176 = 30,562 / 48.9：同硬件上 4 卡 aggregated 已在同一量级。

| 优先级 | 实验 | 针对我们哪个瓶颈 | 成本与前提 |
|---|---|---|---|
| 1 | **aggregated TP4 + DP4 attention 实例** | 3P1D 在 C176 卡在 decode KV 与每步耗时。PD 下每轮都要把整段上下文 KV（6.4–6.8 GB）传给 decode，decode 再存一份完整副本（每个在跑请求约 20.7 万 token），且轮与轮之间不能复用；aggregated 下会话 KV 只存一份，在跑请求的 KV 就是缓存的前缀，不需要传输 | 压测框架补一个 mixed 角色（engine.sh 不加 `--disaggregation-mode`，拓扑和 agentx_env 校验相应扩展）；router 已支持 `DisaggMode::Mixed`，`INFERA_SESSION_AFFINITY=both` 时按"实例 + DP rank"做会话亲和，并有 KV 事件（B300 那次没有）。需要停掉 3P1D 腾出 4 卡 |
| 2 | **decode `--enable-dp-lm-head`** | decode 每步耗时。DP attention 下每个 rank 本地算自己的 logits，省掉词表并行后的汇聚；MTP 每步要过 6 次 LM head（5 次草稿 + 1 次校验） | 当前未开。每卡多约 1.7 GB 复制的 LM head（mem 0.92 下余量约 9 GiB）；需确认与 EAGLE 草稿头兼容。一次 decode + prefill 重启、预热、C176，约 3 小时 |
| 3 | **prefill HiCache `write_back`（+ `direct` IO / `page_first_direct`）** | prefill 缓存容量：137 上实例总容量约 +65%、135 约 +30%，去掉持续 D2H，`direct` 用 DMA 代替 GPU kernel 搬运；也是 prefill_complete 的前提（RESULT §4.7） | 在 3P1D C176 上缓存份额已近理论值，收益小；对 2P1D（GPU 命中掉到 50%）和 C192 以上价值大。ROCm 上 write_back 的 JIT kernel 只在 CUDA 启用，会走回退实现，需功能和性能验证 |
| 4 | prefill `--enable-prefill-delayer` | 让各 DP rank 的 prefill 同步进行，减少空等 | 3P1D 的 prefill 不是瓶颈，单独做价值低；放进实验 1（aggregated 下 prefill 和 decode 同步执行，才有意义） |

不借鉴：只靠会话亲和、不用 KV 事件的路由（我们有 KV 事件 + R2，B300 那次 rank 间请求占比 18.9%–33.2% 不均）；会话亲和、MTP 模拟接受 3.61（已一致）。

**10-15 07:15 决定与复核**：用户决定先把 PD 做到极致，aggregated（实验 1）暂缓，之后再做或由同事做。对 PD 适用项的复核：
- decode 已自动开启 `enable_tp_lm_head_all_to_all`（TP8 = attention DP8 时的默认），LM head 通信已经是 all-to-all，`--enable-dp-lm-head`
  只能再省这一点，预期约 1%–2%。
- `--enable-dp-attention-local-control-broadcast`（NVIDIA 的 GLM-5.2 recipe 都开了，我们没开）：去掉每轮调度在整个 TP 组上的 gloo 同步。
  decode 调度时间里 `get_next_batch_to_run` 在 8 个 rank 上都约占 24%，py-spy（非阻塞采样）显示其中约 19% 在 DP attention 元数据 all-gather
  （CPU gloo 组）上等待，约 10% 在 `build_eagle_disagg_draft_input` 的逐请求 `torch.as_tensor(..., device=GPU)`。但 decode 8 卡 GPU 利用率 99%–100%，
  这些 CPU 开销基本被 overlap 掩盖，decode 仍是 GPU 受限，预期收益小。
- 因此这两个 decode 开关风险低、收益小，放在下一次 decode 重启时一起带上，不单独占一轮。
- PD 下一步按 C176（decode mem 0.92）结果定：若扩 KV 有收益 → prefill HiCache `write_back` + `prefill_complete`（释放等传输占的 7%–12% decode KV）；
  decode 每步 GPU 时间 → `index_share_for_mtp_iteration`（需正确性门禁）。

实验 1 的具体做法：先在 4 卡上起单个实例（v0.5.21 镜像；TP4 + DP4 attention；EAGLE 5/1/6、模拟接受 3.61；KV fp8；chunk 32768；
HiCache 200–270 GB/rank；`--enable-prefill-delayer`、`--enable-dp-lm-head`；max-running 128、cuda graph bs 64；tokenizer worker 2–4），
经 router（`INFERA_SESSION_AFFINITY=both`）测 C24 / C32 / C40 / C48，与 3P1D 的每卡吞吐和交互速度对比。单实例达到约 3 万且交互速度可接受，
再扩到 4–5 个实例（16–20 卡）。风险：ROCm 上 DP attention + MTP + HiCache 在 aggregated 下的稳定性；prefill chunk 插进 decode 步对 ITL 的影响。
