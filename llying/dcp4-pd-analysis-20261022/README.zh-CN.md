# 同事 DCP4 PD 实验包分析，以及在我们 v0.5.22 + PP4 prefill + prefill MTP 上改用 DCP decode 的指导（10-22）

分析对象：`yihou/glm52.dcp4-pd.packup_20261022/`（dev 分支提交 `2326727c`，作者 yihou，10-21 12:38Z 至 10-22 07:35Z）。下文"实验包"指这个目录，
引用的文件都是相对它的路径；"我们"指 `llying/2p1d-v0522-20261020/` 的 sweep。只读了实验包与 `/shared_nfs/yihou_glm52/dcp_pd_20261021/` 上的源码，
没有运行他们的代码。

## 0. 结论

1. **他们把 DCP4 decode 在 MI355X 的 PD 上跑通了**（1P1D 同机 4+4：prefill TP4 + DPA4、decode TP4/EP4 + DCP4、EAGLE 5/1/6、CUDA graph），
   做法是 sglang main（`fda84f80b7`）合入 PR #44322（ROCm triton 稀疏 MLA DCP）再加 31 个提交，作为 overlay 挂载到两侧容器里。
2. **但性能没有超过不用 DCP 的基线**：最佳计分 r4 A2 = 18,321 tok/s/GPU（C40，8 卡），他们自己的非 DCP PD 基线（decode TP4 + DPA4）是 22,114，
   只有 0.83 倍；最新配置 r5b 的 A1（预热轮，不计分）19,400。原因他们定位到 decode：同 4 张卡上 DCP4 每毫秒产出的 token 比 DPA4 少 9%–11%，
   为什么少还没查清（缺少同负载的 DPA4 profile 对照）。
3. **正确性只做了很小规模的验证**：DCP4 PD decode 只过了 5 个请求的 sim-off 检查（math、passkey 8K/34K/96K）；40 并发 sim-off 检查和完整 GSM8K
   都没做。相近的 DP2×DCP2 PD decode 在 CUDA graph 下 40 并发会间歇性乱码（3/200、13/240、27/320），记为已知风险、没有定位。
4. **他们的 prefill 没开 MTP**（"mode N"）。按 AUS 的发现，这会让 decode 草稿层读到未初始化的 KV（top-k 散射变慢、真实接受率偏低），
   他们的 DCP 步长和 DPA 基线都可能受此影响，这一点在实验包里没有讨论。
5. **对我们的价值**：DCP 能给 decode 更多有效 KV（不复制、无 rank 碎片），并消除 DP rank 不均衡，正好对应我们 4+4 C80 碰到的 decode KV 上限；
   但按他们的数据，DCP4 在同等负载下每步更慢，所以净收益不确定。要在我们的栈上用，需要换 sglang 版本（他们的 overlay）、补我们的 PP 修复、
   验证 PP + prefill MTP + DCP decode + 跨节点 Mooncake 这个从没跑过的组合，并补齐并发下的正确性验证（第 6 节）。

## 1. 实验包概况

| 项 | 内容 | 出处 |
|---|---|---|
| 目标 | GLM-5.2 用 DCP 在 MI355X 上跑，先混合（单机聚合）再 PD（prefill 不用 DCP、decode 用 DCP），AgentX C40 达到 20k tok/s/GPU | `spec/mission.md`，`README.md` |
| 节点 | spur 集群：054（PD r1–r4）、v3-045（混合）、v3-060（微基准、PD r5/r5b）、v3-070（E2）等；v3 节点 amdgpu 6.16.13 需 `HSA_ENABLE_SDMA=0` | `environment.md` |
| 镜像 | `infera-sglang:v0522-1020-yihou-dcp-mc4340` = `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020` + Mooncake #4340；tar 在 `/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/images/`（23 GB） | `RUN_BEST_DCP.md` §4 |
| sglang | main `fda84f80b7` + #44322 合入 + 31 个提交（分支 `dev/yihou/glm52_dcp_pd` @ `93034eba91`），以 overlay 挂到 `/sgl-workspace/sglang/python/sglang`（**两侧都挂**）；源码在 `/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-<sha>/` | `patches/README.md` |
| 客户端 | AgentX C40、3,600 s、每 lane 预热 10、InferenceX `918524ff`；A1（新起服务）→ flush → A2（计分）；全部性能轮用模拟接受率 3.61 | `RUN_BEST_DCP.md` §6 |
| KV 传输 | Mooncake 同机 HIP IPC over XGMI（`MC_TE_FILTERS` 设成不存在的设备、`MC_DISABLE_HIP_TRANSPORT=1`），**没有跑过跨节点 RDMA** | `ANALYSIS.md` §4，`HANDOFF_r5b.md` §4 |

## 2. 完成了哪些实验

| 类别 | 实验 | 结果 | 出处 |
|---|---|---|---|
| 混合（单机 4 卡） | DCP4 TP4/EP4 C40 | **26,277 tok/s/GPU**（按 4 卡），TPOT 36.9 ms；不用 DCP 的 TP4/EP4 只有 5,130（KV 不够、排队）；KV 容量 DCP4 8.89M token vs 无 DCP 2.86M（3.1 倍） | `results/RESULT.md` §3 |
| 混合正确性 | check.py（math、passkey 8K/34K/96K）eager + graph；40 并发 40/40；部分 GSM8K（sgl-eval）DCP4 0.9847 vs 参照 0.9848、错题相同 | 通过 | 同上 |
| decode 微基准 | 同一混合 server、缓存长 prompt、按并发 N 分批 | custom AR 开 vs 关 **−3.5~−3.9 ms/步**；`KEEP_KV_B_BF16` −0.4~−1.7 ms；拆分 indexer（Triton + radix 合并）34K +0.5~+1.5 ms、150K −2.6~−4.2 ms、**200K −4.4/−6.7/−9.1 ms**（N=16/24/32）；`index_share_for_mtp_iteration` 开关 ≈ 0；131K 比 34K 每步多 5%–13% | `results/RESULT.md` §5 |
| PD 正确性门 | pdBe-r1（eager）、pdBe-r5g（graph），sim off，各 5 个请求：math 29 ×2、passkey 8K/34K/96K | 通过 | `results/pd/pdBe-r1/t4/` |
| PD 性能 | r2 → r4（custom AR 关→开、关 NCCL debug） | r2 A2 17,053 → **r4 A2 18,321**（TPOT 19.7 ms、TTFT 3.93 s）；r4 C52 18,070（比 C40 低） | `results/RESULT.md` §2 |
| PD 性能 | r5b（换代码 `93034eba91` + kv_b bf16 + 拆分 Triton indexer + 关 debug agent，换到 v3 节点） | A1 **19,400**（预热轮不计分），A2 因节点被抢占丢失；r5c2/r5d/r6/r6b 都因抢占或节点故障没拿到 A2 | `ANALYSIS.md` §1 |
| E2 | PD + DP2×DCP2 decode，sim off，40 并发检查 | eager 0/40；**graph 下间歇性乱码 3/200、13/240、27/320**（前几个 token 正确、之后是垃圾），关 index_share、关拆分都没修好，未定位 | `results/RESULT.md` §6 |
| 基础设施 | 突发节点抢占、容器活得比作业久（写了 guard）、debug agent 在 6.16 上让 prefill 崩溃等 | — | `ANALYSIS.md` §6，`notes.md` |

## 3. 效果与对比

**3.1 他们的 PD 数字（C40，8 卡，模拟接受率 3.61）**：

| 配置 | tok/s/GPU | TPOT 均值 | decode 每步 @ 在跑 | 每 4 张 decode 卡的产出 |
|---|---|---|---|---|
| 非 DCP 基线（decode TP4 + DPA4，v0.5.21） | 22,114 | 12.1 ms | 每 DP rank 40.9 ms @ 3.57（4 rank 合计 14.3） | 约 1.25 tok/ms |
| DCP4 r4 A2 | 18,321 | 19.7 ms | 61.2 ms @ 18.75 | 约 1.11 tok/ms |
| DCP4 r5b A1（预热轮） | 19,400 | 17.3 ms | 52.7 ms @ 16.7 | 约 1.14 tok/ms |

- **DCP4 decode 每步按负载**（r4 A2 / r5b A1，p50 ms）：在跑 9–16 时 50.7 / 47.4，17–24 时 67.0 / 57.3，25–32 时 76.4 / 66.6（`ANALYSIS.md` §3）。
- **profile**（r2，bs 24，70.6 ms/步）：RCCL 17.8、indexer logits 15.6、MoE 10.3、coop top-k 9.3、GEMM 8.0、稀疏 MLA 3.4 ms；没有同负载的 DPA4 对照。
- **权重**：DCP4 decode 每卡少载 17.6 GB，原因是不用 DP attention 后注意力投影（BF16）按 TP4 切分；这是 attn TP4 带来的，不是 DCP 本身。
  DCP 的作用是在 attn TP4 下仍然不复制 KV（`ANALYSIS.md` §2）。
- **KV 传输**：C40 时 decode 等传输约 5 个，与基线相同，不是差距所在；C52 时等传输从 4.9 升到 18.2、TTFT 14.4 s，看起来像传输受限（未确认）。
  每个 DPA prefill rank 要把 KV 发给 4 个 decode rank（`required_dst_info_num = 4`）。

**3.2 和我们的数字对比（粗略，条件不同）**：

| 配置 | 卡 | 并发 | tok/s/GPU | TPOT / ITL | decode 在跑（合计）| decode 每步 |
|---|---|---|---|---|---|---|
| 他们 DCP4 r4 A2 | 4+4 | C40 | 18,321 | TPOT 19.7 ms | 18.75 | 61.2 ms（均值） |
| 我们 PP4 + DPA4 decode（4+4 C64） | 4+4 | C64 | 32,532 | ITL p50 18.9 ms | 38（每 rank 约 9.5，最忙中位 13） | 66–70 ms（按最忙 rank） |
| 我们 PP4 + DPA4 decode（4+4 C80） | 4+4 | C80 | 37,310 | ITL p50 20.3 ms | 52（最忙中位 17） | 71–78 ms |

- 同样约 19–20 ms 的 TPOT 下，我们的 DPA4 decode 合计扛 38 个请求，他们的 DCP4 只有约 19 个——与他们"DCP4 每步产出比 DPA4 少"的结论方向一致，
  但幅度大得多。条件差异很多，不能直接下结论：他们 C40（客户端并发限住）、prefill 不带 MTP（草稿层冷态）、r4 开着 debug agent、不同集群与驱动；
  我们带 prefill MTP、PP4 prefill、跨节点 Mooncake。
- **结论**：DCP 的收益在 KV 容量和消除 rank 不均衡，代价是每步更慢；在我们 4+4 已经 KV 顶满的高并发端（C80 以上）才可能划算，需要实测。

## 4. 他们是怎么实现的

主线上缺两块：启动时直接拒绝"DSA DCP + PD"，以及 MTP 草稿层 KV 在 ROCm 上按 DCP 切分、而 PD 重排假定草稿是复制的（`spec/design/pd_dcp_dsa.md`）。31 个提交按功能分组：

| 组 | 提交 | 作用 |
|---|---|---|
| Infera 镜像补丁重做成提交 | 01–09 | overlay 会盖掉镜像里打过补丁的 sglang，所以把 Infera 的补丁（DSA indexer 填充行、DSA DP 同步、页表行、prefill 非最后 chunk 等待事件、PD decode 跳过隐式拒绝采样、Responses 工具参数、HiCache staged write-back、kv-events 不阻塞、关闭 absorb-bmm 配置）重做成提交 |
| PD + DCP 打通 | 10–13 | 允许"DCP1 prefill → ROCm DCP decode"（只放开 decode 侧）；草稿层 KV 按 DCP 切分后跟随目标层一起重排（Mooncake 发送端）；DPA prefill 一个 rank 扇出给 4 个 DCP decode rank 的测试 |
| 拆分 decode indexer | 14、15、18、24、30、31 | 每个 DCP rank 只对 1/4 的页算 indexer 分数并取局部 top-k，再 all-gather 合并成精确的全局 top-k；Triton 打包 + radix 选择合并；默认关，由 `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER` / `_SPLIT_INDEXER_TRITON` 打开 |
| q 复制 | 22 | `SGLANG_DSA_DCP_KEEP_KV_B_BF16`：kv_b 保持 BF16，使 `--dcp-replicate-q-proj` 在 78 层都生效、省掉每层 Q 的 all-gather |
| PP prefill 相关 | 16、17 | PP prefill 各级喂给 DCP4 decode 的测试；PP（不开 overlap）下非最后一个 chunk 发送前记录 forward stream 事件。**只在 CPU 上测过，PP 从没在 GPU 上跑过** |
| 其他修复 | 19–21、23、25–29 | 客户端取消后 deferred KV 释放导致 attn-TP 各 rank 队列分叉而挂起；上游 #45483（eager IDLE batch 初始化后端）；OffloaderV2 丢 `is_shuffled`；DP2×DCP2 的三个 bug；诊断日志 |

关键机制（`spec/design/pd_dcp_dsa.md` §2、`pp_prefill.md` §2）：
- **目标层 latent KV**：decode 上按位置条带化（位置 p 放到 rank p % 4），prefill 用已有的 DCP1→DCP-N 重排 + 目的连续的打包缓冲发送，
  每个 decode rank 收到约 1/4 的 token。
- **DSA index-K**：decode 上是复制的，走通用的页传输路径，每个 decode rank 都收到完整的 index-K。
- **草稿层（NextN）KV**：ROCm 上和目标层一样条带化，新代码把它当成目标层一起重排。
- **PP prefill**：每一级是独立的发送方，按层号与 decode 的完整层列表配对；每个 decode rank 等 4 级都发完；草稿 KV 只在最后一级发。
- 每 token 传输量约 57 KB（latent 45.5 KB 条带化 + index-K 4 × 2.9 KB 复制），DPA prefill 时全部来自一个 prefill GPU。

## 5. 他们是怎么验证的，以及缺口

| 层次 | 做了什么 | 缺口 |
|---|---|---|
| CPU 单元测试 | 每个代码改动都有测试，多数是"旧代码先失败" | 不覆盖 GPU 内核与真实传输 |
| 混合正确性 | check.py eager + graph、40 并发 40/40、部分 GSM8K 与参照一致 | 只覆盖混合，不覆盖 PD |
| PD 正确性 | sim off 下 5 个请求的 math + passkey 8K/34K/96K（eager 与 graph 各一轮） | **40 并发 sim-off 检查未做、完整 GSM8K 未做、真实接受率（T4c）未见结果**；E2（DP2×DCP2）graph 下间歇乱码未定位 |
| 性能 | 微基准逐项 A/B；PD AgentX 按 A1 → flush → A2 计分 | r4 → r5b 一次改了节点、驱动、debug agent、两个开关和代码版本，收益无法归因；r5b 没有计分的 A2 |
| 场景 | 同机 4+4、C40 | 跨节点 RDMA、PP prefill（GPU）、prefill 带 MTP、prefill HiCache + PP 都没跑过 |

## 6. 在我们 v0.5.22 + PP4 prefill + prefill MTP 上把 decode 换成 DCP，需要做的事

### 6.1 软件基线：只能用他们的 overlay（或基于它重建），不能在 v0.5.22 上直接开 DCP

- v0.5.22 镜像里 DSA 的 DCP 只有 trtllm（NVIDIA）实现（我们 10-21 读源码确认过）；ROCm 需要 #44322，PD 需要他们的提交 10–13。
- 两种做法：
  1. **用他们的镜像 + overlay**（`infera-sglang:v0522-1020-yihou-dcp-mc4340` + `src/sglang-93034eba91`）。问题：他们的镜像 Mooncake 是 #4340、
     没有我们的 RDMA 补丁（dest-local rail、ACK 超时、1 GiB MR 分块），**我们是 138 → 140 跨节点 ionic RDMA，必须带这些补丁**。
  2. **把他们的 overlay 挂到我们的镜像上**（`infera-sglang:v0522-fdbafb60`，已带我们的 Mooncake 补丁）。问题：overlay 是 sglang main 的 Python 树，
     我们镜像的基础是 v0.5.22 的 10-18 构建，他们是 10-20 构建；sgl-kernel / aiter 等编译部分的版本可能对不上，要先起一个混合 server 做冒烟验证。
  - 推荐先走第 2 条（保留我们已验证的跨节点 RDMA），失败再考虑把我们的 Mooncake 补丁打进他们的镜像重建。
- 他们的 overlay 已包含 Infera 补丁的等价提交（01–09）；要逐个和我们 `deploy/docker/patch.upstream.status.md` 里 v0.5.22 的补丁清单核对，
  缺的补上（例如我们后加的补丁），多的确认不冲突。

### 6.2 必须补的修复

1. **PP 中间级的 `embed_tokens.embedding_dim`**：他们的 `deepseek_v2.py`（`93034eba91` 第 3003、3038 行）和 v0.5.22 一样会在 PP 中间级崩溃，
   要把我们的补丁（改读 `config.hidden_size`）合进去（我们的补丁文件 `2p1d-kit/v6-pp4/patches/deepseek_v2.py` 是基于 v0.5.22 的，不能直接覆盖，
   要在他们的版本上改同样两行）。
2. **PP 非最后 chunk 发送事件**：他们的提交 17（`6ee35f3617`）在 PP 下记录 forward stream 事件；我们在 v0.5.22 上跑 PP 时没有这个修复，
   smoke 6/6（含 66k 冷请求、多个 chunk）和 GSM8K 都通过；换成他们的版本后会自动带上。

### 6.3 从没跑过、需要逐项验证的组合

| 组合 | 他们的状态 | 我们的状态 | 要验证什么 |
|---|---|---|---|
| PP4 prefill → DCP4 decode | 只有 CPU 测试（提交 16），设计文档认为代码已支持 | PP4 → DPA decode 已验证 | 每级日志的层范围、`PD DCP pack buffers allocated`、没有 `geometry` / `item length` / `Rejecting state`；passkey 跨 chunk 正确 |
| prefill 带 MTP（mode S）+ DCP decode | 只是设计（草稿 KV 只在最后一级发、按目标层重排），从没跑 | prefill MTP + DPA decode 已验证（真实接受长度中位 4.5–4.6） | **真实接受长度**应和 DPA 时一样约 4.5；若掉到 1–2，说明草稿 KV 没传对 |
| 跨节点 Mooncake RDMA + DCP 扇出 | 没跑；每个 prefill 级要发给 4 个 decode rank，在 rail 隔离网络上 3/4 要跨 rail | 我们的 dest-local rail 补丁让源端选目的 rail 的网卡，DPA decode 下已经是"一个 prefill rank 发给任意 decode rank"，工作正常 | 传输失败、MR 注册（我们按 `MC_MAX_MR_SIZE` 1 GiB 分块，example 说明 8）；打包缓冲 = 4 个队列 × chunk × 45.5 KB/token，我们 chunk 8k 时每 prefill GPU 约 1.5 GB，在 KV 池分配之后才懒分配（他们的风险 R3），可能挤爆 prefill 显存 |
| PP prefill + HiCache | 没跑 | 已验证 | 无新风险，换版本后复测 |
| DCP decode + custom AR | 他们开着（−3.5 ms/步；DP2 下关掉会让 graph 捕获失败） | 我们在 DP attention + decode graph 下开 AITER custom AR，负载下会乱码（v0.5.19–v0.5.22 每个上游镜像都复现，example 说明 6），所以一直关着 | **必须先做 40 并发 sim-off 检查 + 完整 GSM8K**；他们 E2 的乱码（graph 下、负载下、开着 custom AR）和我们的现象相似，可能是同一个问题（未验证） |

### 6.4 配置要点（decode 侧）

- 并行：`--tp-size 4 --dcp-size 4`，不要 `--dp-size` / `--enable-dp-attention`；EP 他们用 4（我们 DPA decode 是 EP1），先照他们的。
- 环境：`SGLANG_DSA_DCP_KEEP_KV_B_BF16=1`、`SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1`、`SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1`；ROCm debug agent 关
  （我们一直关）；`HSA_ENABLE_SDMA=0` 只在 amdgpu 6.16 节点需要，138 / 140 都是 6.14.14（与他们 r1–r4 的 054 相同），不需要。
- 显存：他们 0.85；同事在 0.92、max-running 128、graph bs 128 时 decode 只剩 5.9 GB，运行 7 分钟后 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`。
  从 0.85–0.88 起步。
- `--cuda-graph-max-bs-decode` 要不小于实际在跑数：DCP decode 是一个 DP 组，所有请求在一个 batch 里；我们 4+4 C80 时合计在跑 52、最多约 70–80，
  需要 graph bs ≥ 96，`--max-running-requests` 相应调大（受显存限制，要和 0.85 的显存比例一起试）。
- router：decode 只有一个 DP 组，`INFERA_PD_DP_RANK_AFFINITY=false`；我们的 R2（`--decode-input-demand`，在 decode DP rank 间按 KV 均衡）不再适用。
- prefill 侧不变（PP4、chunk 8k、prefill MTP、HiCache），只是换了 sglang 版本，需要复测。

### 6.5 建议的执行顺序与验收标准

1. **软件冒烟**（不开 HiCache，失败重启快）：我们的镜像 + 他们的 overlay（加 6.2 的补丁），先起一个单机混合 DCP4 server（4 卡），
   跑他们的 check.py；通过后再起 PD。
2. **PD 正确性**（sim off、graph 开）：smoke、InferenceX GSM8K（1,319 题）、40 并发 sim-off 检查（`scripts/mixed/run_check40_pd.sh` 的做法）、
   真实接受长度。标准：GSM8K 与我们 DPA 的 0.97 一致；40 并发 0 乱码；接受长度约 4.5。**先用 custom AR 关**，乱码检查通过后再试开。
3. **打开 HiCache** 复测 smoke + GSM8K。
4. **性能**（4+4，和我们 DPA4 的点同条件对比）：C64、C80，再试 C88 / C96。标准：同交互下吞吐更高，或者能在 C88–C96 不出现等预分配
   （我们 DPA4 4+4 在 C80 已经 45% 的时间有 rank 超过 0.9、最多 25 个等预分配）。同时记录每步耗时 vs 在跑数，和 DPA4 的步长表对比。
5. 如果 4+4 有收益，再考虑 2P(PP4)+1D(TP8 + DCP8 或 TP8 + DCP4)。

### 6.6 预期（估计，需实测）

- **KV**：attn TP4 下注意力权重每卡少约 17.6 GB，同显存比例下每 rank 多约 0.4M token；加上没有 rank 碎片，有效容量粗估 +15%–30%，
  4+4 的 KV 上限可能从 C80 推到 C88–C96。
- **每步**：按他们的数据，同等负载下 DCP4 比 DPA4 慢（产出低 9%–11%）；DCP 消除不均衡（我们最忙 rank 是均值的 1.3–1.8 倍），
  在我们这边可能部分抵消。净效果 ±10% 量级，主要价值在高并发端。
- **风险最大的两项**：custom AR 与 graph 下的乱码、跨节点 RDMA 扇出，任一项不过都会卡住。

## 7. 参考

- 实验包：`README.md`、`ANALYSIS.md`、`RUN_BEST_DCP.md`、`HANDOFF_r5b.md`、`results/RESULT.md`、`patches/README.md`、`spec/design/pd_dcp_dsa.md`、
  `spec/design/pp_prefill.md`、`notes.md`。
- 外部：InferenceX PR #4064（MI355X SGLang TP4/EP4 + DCP4，依赖 #43735 + #44322）、#4045（ATOM 1P1D PP4 prefill + TP4/DCP4 decode，
  C96 42.9k/GPU @ p90 42.1）；我们的分析报告 `../gb300-atom-pd-analysis-20261021/README.zh-CN.md`。
- 我们的数据：`../2p1d-v0522-20261020/README.zh-CN.md`（4+4 C64 / C80、PP4 的修复与验证、decode KV 统计）。
