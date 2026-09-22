# GLM-5.2 PD 分离修复梳理与 2P1D 同步报告

本报告记录原环境的修复来源、本环境的同步内容，以及 2026-09-21 至 2026-09-22 的实际验证证据。测试拓扑为 Prefill 137/138、Decode 136；五档 sweep 已完成。

日期：2026-09-21。参考环境为
[`glm52.p8d8.agentx-sweep.packup_20260920`](../../yihou/glm52.p8d8.agentx-sweep.packup_20260920/README.md)，
并追溯其使用的同 rail 路由补丁、MTP 排障记录及仓库 Docker 构建补丁。

## 结论与本次同步

原环境的修复分布在 **Rust router、镜像中的 SGLang/Mooncake 路径、启动配置**
三个层面。仅复制 `config.sh` 或只指定一个上游 SGLang 镜像，都不能完整继承。

当前 2P1D 套件已经固定参考镜像并继承大部分运行配置；本次发现的主要遗漏是：
**仓库里的 Rust router 源码没有同 DP rank 约束，而参考镜像内的二进制已有该功能。**
这会导致后续从本 checkout 重建镜像时丢失修复。通用 `bench/glm5p2_pd` 的启动接线也缺失。

本次已完成：

1. 将参考的 Rust 同 DP rank 路由修复同步到当前 `rust/router/src/`，并同步
   `bench/glm5p2_pd/config{,.full}.sh` 和 `launch.sh` 的开关接线。
2. 将参考镜像额外携带的 HiCache #37152、NextN fusion 两份补丁同步到本套件
   [`patches/`](patches/README.md)，用于追溯和重建。现用镜像已经包含它们，无需重复打补丁。
3. 新增 [`scripts/verify_pd_fixes.sh`](scripts/verify_pd_fixes.sh)：在每台节点上用
   不挂 GPU、不挂模型、无网络、只读文件系统的短时 CPU 容器检查镜像内容。
   检查 router CLI 及 22 项 SGLang 代码条件，共 23 项；Python 检查先解析 AST，
   避免仅在注释中出现补丁名称就误判为已修复。
4. 将镜像检查接入 `check.sh`，在部署及 RDMA preflight 之前执行。
   一键运行时证据写入 `.tmp/results/<RUN_ID>/patch-check/`。
5. 增强每档 AgentX 的实际参数校验：逐 GPU RDMA map、GID、HCA filter、设备亲和、
   failed-session probe、按角色 HSA 设置及 MTP steps/topk/draft tokens 必须与配置一致。
   检测到已撤回的 fused-topk-off workaround、显式开启不兼容的 PD rejection sampling，
   或不同 worker 使用不同镜像时停止测试。

本次复用参考镜像，在用户授权的 136/137/138 部署 2P1D。三台镜像能力检查均通过，
两组 P→D 的 44 项 Mooncake WRITE 检查通过。实测结果见 [`results/sweep_results.md`](results/sweep_results.md)；
未将原来的 140 节点检查失败作为本轮结论。

## 实际检查到的镜像版本

| 项目 | 136 / 137 / 138 检查结果 |
|---|---|
| Image tag | `infera-sglang:v0519-yihou-0917-nextnfix-hicache` |
| Image ID | `sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35` |
| SGLang package version | `0.5.19.dev20260917+ga9fb1c3238` |
| 镜像内 SGLang checkout HEAD | `7ccbf5fd04f7ee23095fc38e49e749d58dc18282` |
| 补丁内容检查 | 三台分别 23/23 通过 |

这里的 package version 和 Git HEAD 是两个不同事实，不能将版本字符串中的
`a9fb1c3238` 直接写成实际源码 HEAD。原容器的 SGLang 工作树还包含未提交修改；
因此本次同时保存完整 Image ID、补丁条件和所检查文件的 SHA256，而不只记录版本号。
三台节点的检查均直接读取该 Image ID 启动的 CPU 容器，并非仅从原 GPU 容器推断。

原始验证 JSON、源码读取记录、编译日志和测试都保存在 [`.tmp/pd-fixes/`](.tmp/pd-fixes/)。
下文有关“原来测过”“上游状态”的描述来自有日期的仓库归档，**不是对当前上游状态的实时查询**。

## 1. 路由与 KV 传输

### 1.1 Prefill/Decode 同 DP rank 路由：必要修复

**问题。** Prefill 和 Decode 独立挑选 DP rank，会出现 P.rank1 → D.rank0。
在目标集群的隔离 RDMA rails 上，KV 的目标 GPU 与所用 rail 不匹配，导致
`transport retry counter exceeded`，并进一步触发 Mooncake session blacklist。
历史记录中 600 秒内 48 次真实失败全部是跨 rank handoff。

**修复。** 保留 Prefill 的 KV-aware 选择，再将 Decode 候选限定为相同
`effective_dp_rank`。找不到对应 rank 时返回 503，并且不先发出半个 P/D 请求。
同一约束覆盖 RoundRobin 和 KvEventAwarePolicy。

**2P1D 含义。** 第二个 Prefill 实例的 rank 3，应该交给唯一 Decode 实例的 rank 3；
rank 是实例内部的 attention DP rank，不是把两个 Prefill 拼起来后的全局编号。
新增 CPU HTTP mock 已验证这个场景，也验证了 Decode 缺少 rank 3 时返回 503。

**同步。** 源码已更新到 `config.rs`、`handlers.rs`、`main.rs`、`disagg.rs`、
`policy.rs`、`pool.rs`；原有功能测试的两处 AppState 初始化补上默认关闭字段。
新加的测试用例保存在 `.tmp/tests/pd_router_cases.rs`，在 `.tmp/` 中的源码副本运行。

来源：同 rail 归档的
[`01-router-pd-dp-rank-affinity.patch`](../../yihou/glm52-1p1d-samerail-c32-c40.packup_20260918/patches/01-router-pd-dp-rank-affinity.patch)，
追溯到 `99fa0406cc2ce3e7eedb8a3349b2626eca1339b2`。

### 1.2 开关接线与每 GPU HCA map：必须配套

只有 router 代码仍不够，功能默认关闭。启动脚本需要将 `PD_DP_RANK_AFFINITY=1`
转换成 `INFERA_PD_DP_RANK_AFFINITY=true`；Rust clap 接受的是 `true/false`，直接传
`1/0` 会报错。本套件已继承该接线，本次补齐了通用 `bench/glm5p2_pd` 的遗漏。

另一个独立条件是 `RDMA_DEVICE={"0":"ionic_0",...,"7":"ionic_7"}`。
将八个 HCA 简单写成共享列表会把设备选择交给自动发现，不能保证逐 GPU 对应。
`MC_TE_FILTERS` 则仍然需要逗号列表，不能把上述 JSON 原样传给它。

当前保留 `MC_GID_INDEX=1`、`MC_ENABLE_DEST_DEVICE_AFFINITY=1`，并在运行前检查
各节点 `ionic_i` 的 rail ID 是否一致。本轮已在 137→136 和 138→136 分别完成
Mooncake WRITE preflight，共 44 项通过；原始记录在 `.tmp/results/2p1d-20260921T140351Z/preflight-prefill-*/`。

### 1.3 Mooncake early-send 等待写入完成：KV 正确性修复

**问题。** chunked prefill 的非最终 chunk 可以在 GPU forward 尚未写完 KV 时被
RDMA 读取，传出的 KV 不完整。旧路径虽然部分地方记录了 event，但 Mooncake
transfer worker 不等待它；overlap 的一条非最终 chunk 路径甚至没有记录 event。

**修复。** `TransferKVChunk.wait_event` 传递同步事件；Prefill 在 forward stream
记录 event；Mooncake 在读取设备内存前执行 `kv_chunk.wait_event.synchronize()`。
最后一个 chunk 原有的 sampling 同步不能替代这些中间 chunk 的同步。

这不是 DSA 特有问题。归档中 MI325X/v0.5.16 的长上下文 needle 测试由 5/9、4/9
提升到 9/9；这些是历史正确性证据，不能当成本次 2P1D 的通过结果。
当前镜像中的事件生成、传递、等待三段均已检查；仓库的
[`patch_mooncake_early_send_wait_event.py`](../../deploy/docker/patches/sglang_disagg/patch_mooncake_early_send_wait_event.py)
本来就由 Dockerfile 执行，本次保留。

### 1.4 Failed-session probe：恢复能力，不替代前述修复

单次失败可使 session 被持续 blacklist；周期性探测可在连接恢复后移除黑名单。
当前镜像已有 `_failed_session_probe_loop` 和恢复逻辑，配置开启
`SGLANG_ENABLE_FAILED_SESSION_PROBE=1`、间隔 30 秒。

它不能修复跨 rail 路由、错误 KV 或死亡的 Decode rank。本次确认代码/开关存在并
增加实际环境变量校验；未做网络故障注入，不能据此宣称恢复时延已经验证。

## 2. DSA、DP attention 与 MTP

以下四项由仓库的
[`apply_sglang_dsa_patches.sh`](../../deploy/docker/scripts/apply_sglang_dsa_patches.sh)
在 MI355X 镜像构建时应用，本次在三台测试节点的参考镜像内确认它们均存在。

| 修复 | 原问题 | 实现与判断 |
|---|---|---|
| DSA padded/real rows 对齐 | HIP paged-MQA 的 query/logits 行数与 lengths 不一致；idle DP rank 下 real > padded 也会出现 | 同时处理 trim 与 clip，核心 `_p1v2_rows = min(real, padded)`；只修 real < padded 不完整 |
| DSA DP host-sync 修复 | 只有部分 rank 执行 `.max().item()` 或补 CPU mirror，破坏 collective 执行顺序，发生死锁 | 使用静态 `req_to_token.shape[1]`，去掉 DRAFT_EXTEND_V2 分支中不需要的 D2H；来自归档中的 #33973 修复 |
| MTP page-table rows 对齐 | page table 按 request 分行，top-k 按 token 分行，MTP 时行数不同 | `_glm52_match_page_table_rows` 重复/裁剪相应行，再送入 top-k transform |
| draft graph/eager 的 DP 投票 | 各 rank 根据本地输入独立决定 graph 或 eager，一部分进入另一条执行路径而挂起 | scheduler 记录本地请求，通过现有 MLP-sync collective 传播，全组取共同决定，再传到 draft graph runner；本 v0.5.19 版本使用同步槽位 8 |

来源与版本限制：[`sglang_dsa/README.md`](../../deploy/docker/patches/sglang_dsa/README.md)。
文档中同时保留 v0.5.16/v0.5.18 的历史，不能将旧 diff 的上下文直接套到任意新版。
`IndexShare=false` 可以绕过其中部分 MTP 路径，但不替代 padded-row 修复和 DP host-sync 修复。

### PD Decode 禁止隐式开启 ROCm rejection sampling

v0.5.19 会为 HIP EAGLE 自动选择 rejection sampling，但 PD handoff 没有传递
`draft_probs`。Decode 重建的 draft input 中该字段为 None，随后 `torch.stack`
失败，可能在启动 warmup 阶段就崩溃。

修复在 `speculative_hook.py` 的自动选择条件中加入
`cfg.disaggregation_mode != "decode"`。这是
[`patch_pd_disable_implicit_rocm_rejection_sampling.py`](../../deploy/docker/patches/sglang_disagg/v0519/patch_pd_disable_implicit_rocm_rejection_sampling.py)
的内容，本次确认镜像已带入。它只抑制不兼容的自动选择；显式参数仍可越过它，
因此本套件现在也检查实际 argv，拒绝在该 PD 栈强行启用此选项。

该 guard 没有实现 PD draft_probs 传输，也不是非零温度 sampling 正确性的完整修复。

## 3. HiCache 与 NextN

### 3.1 HiCache #37152：参考镜像额外携带

ROCm wave64 与实际拷贝工作分组不能混用；补丁固定逻辑 copy group 为 32 线程，
根据 element 大小选 128/64/32/16-byte 的合法 copy round，并同步 Python 侧
`_tiles_across_lanes` 的判断和 ROCm block quota。CUDA 保留原来的 128-byte 要求。
补丁还将 K-only host pool 的 JIT 可用性扩展到 HIP。

这与当前 GLM DSA/HiCache 路径有关，故保持与参考镜像一致。本套件仅 Prefill
启用 HiCache，Decode 因 MTP 约束关闭 HiCache；不能写成“两条腿都启用了该优化”。

参考归档称该 PR 当时尚未合入且 AMD ROCm CI 为红，本次没有重新核查上游状态。
保留依据是固定参考镜像与已有运行记录，而非宣称所有平台上的通用修复已经验证。
已同步原始源码 diff 到 [`patches/pr37152.sources.yihou.diff`](patches/pr37152.sources.yihou.diff)。

### 3.2 MLA staged write-back gate：保守 workaround

旧版本中 MLA pool 与 HostPoolGroup 对是否使用 staged write-back JIT 的判断不一致，
可能把 GPU indices 交给要求 host indices 的 kernel，导致首次回写崩溃。
镜像保留 `_is_cuda and can_use_write_back_jit_kernel(...)`，使 ROCm MLA 走非 JIT 路径。

这是保守开关对齐；v0.5.18 以后已有按 pool 处理 indices 的变化，因此不能直接
把 v0.5.16 的崩溃证据写成本 v0.5.19 同样必然发生。当前为保持参考行为而保留。
此开关与 #37152 的 K-only 普通 JIT 拷贝开关属于不同路径，两者并不矛盾。

### 3.3 NextN shared-experts fusion：独立配置正确性修复

GLM NextN draft 继承了 DeepSeek 的 architecture 名称，无法匹配 loader 改写后的
`GlmMoeDsaForCausalLMNextN`，导致 draft 静默丢失 shared-experts fusion。
补丁为该类补上同名 `fused_shared_experts_architecture`。

这是一项真实的独立缺陷，但原 A/B 已证明它不是 MTP 乱码的原因。归档中的
ITL ratio 0.997、acceptance 3.62/3.60 也不能支持明显加速的结论。
当前镜像已包含该修复，原始补丁同步在
[`patches/nextn-fusion-fork-4350d37c5.patch`](patches/nextn-fusion-fork-4350d37c5.patch)。

## 4. API 路径：Responses 的 PD bootstrap

旧 `ResponsesRequest` 未声明 bootstrap_host/port/room，Pydantic 忽略这些额外字段，
且 serving 层没有转交 `GenerateReqInput`，最终 worker 报
`Disaggregated request received without bootstrap room id`。

[`patch_responses_pd_bootstrap.py`](../../deploy/docker/patches/sglang_disagg/patch_responses_pd_bootstrap.py)
补齐请求模型和转发；镜像中已存在，当前保留。AgentX 主路径使用 chat，但同一个
PD 服务的 `/v1/responses` 也需要这项配套修复。
它只解决 bootstrap plumbing，不提供多 worker 的 response_store 状态共享；
也没有实现 harmony 内置工具后续轮次的新 P/D room 协调。

## 5. 保留的配置规避与不采用的项目

| 项目 | 当前处理及依据 |
|---|---|
| `index_share_for_mtp_iteration=false` | 保留稳定性 workaround。参考 IndexShare 开启时 TP8 在 1h26m 出现 GPU memory fault；关闭后完成约 13h33m sweep。它不是内核根因修复，也不等价于正确性证明 |
| `SGLANG_DSA_FUSE_TOPK=0` | 不采用，参考最终配置已撤回。恢复 fused 路径后模拟 acceptance 的乱码仍存在，因此不能断言“禁用 fused 必然导致乱码”；当前保留 fused 路径并在实际 argv/env 检查中防止误带入旧设置 |
| `--disable-custom-all-reduce` | 不默认打开。确定性的乱码 A/B 来自 P4D4/TP4，不能直接外推到本 TP8；需要真实 acceptance 下的单变量验证。现有 `DECODE_EXTRA_ARGS` 可用于明确的专项试验 |
| 模拟 acceptance 3.61 | 仅性能测量口径，不是 bug fix。强制接受长度并不保证接受的 token 正确；不能用 acceptance gauge 证明文本正确 |
| `flydsl` DSA / `aiter` DSA top-k | 不移植未经验证的 backend；当前 nightly 参数不接受相应旧 fork 选择，继续使用 tilelang/default top-k |
| `patch_hicache_rocm_host_alloc.py` | 不额外叠加。当前 Dockerfile 明确说明新基线已有 #35233 的 host→device pointer 映射修复；旧 allocator workaround 会改变 allocator 行为 |
| GLM MoE gate bias FP32 补丁 | 不纳入此次 PD 对齐。位于 `sglang_carried/`，原本没有进入镜像构建；是独立的 MoE 路由精度问题，需另开正确性/性能对照 |
| Prefill `HSA_NO_SCRATCH_RECLAIM=0`，Decode `=1` | 沿用按角色设置，作为资源管理配置，避免将其写成已闭合的 allocator/kernel 根因修复 |
| `max-running` / graph max BS=256 | 保留 sweep 的容量设置，避免最高档位测到人为调度上限；属于测试配置修正 |
| 启动前检查实际 GPU 显存，graceful stop 等待释放 | 保留。HiCache 释放可能持续数十分钟，容器退出不等于 GPU 内存已释放；不因此自动重置节点或清理其他服务 |

关键历史说明：
[`P8D8 notes`](../../yihou/glm52.p8d8.agentx-sweep.packup_20260920/notes.md)、
[`MTP 乱码 A/B`](../../yihou/glm52-mtp-garbled-decode-rootcause.packup_20260918/README.md)。

## 6. 高并发慢预热：原环境仍存在的限制

参考归档明确记录了高并发下的 `KVPoll.WaitingForInput` 超时与慢预热：
C192 约 2 小时 11 分钟，C256 约 3 小时 10 分钟。该问题没有因 IndexShare 关闭、
fused top-k 恢复而消失，不能列为“已同步修复”。见参考
[`notes.md` 第 7–8 节](../../yihou/glm52.p8d8.agentx-sweep.packup_20260920/notes.md)。

本轮 C256 也在初始预热出现 Router→Prefill POST 失败，随后部分 Decode 请求等待 KV 超时。
这类 HTTP/应用层异常不等同于此前跨 rail 引发的 RDMA retry/WQE 故障。
本轮保留原参数和超时，不通过减少 10/lane 预热或缩短正式测量来隐藏该成本。
失败发送的更底层原因仍未定位，详细证据和人工重试记录见
[`measurement_notes.md`](results/measurement_notes.md)。

## 7. 验证与后续使用

- **Rust 源码**：在 `.tmp/` 的源码副本编译，9 项 PD CPU 功能测试通过。
  覆盖原有 chat/responses/streaming/bootstrap 路径，以及新增 2P1D 同 rank、缺失 rank
  的 503 和 KV-aware 候选过滤。详细日志：`.tmp/pd-fixes/build/router-tests.log`。
- **镜像**：136、137、138 分别 23/23 通过。当前运行的逐节点记录在
  `.tmp/results/2p1d-20260921T151735Z/patch-check/`；早期 review 记录仍保留在 `.tmp/pd-fixes/`。
- **运行参数校验**：20 项 Python 离线检查通过，包括新增的错误 RDMA map、probe 被关闭、
  旧 fused-topk workaround、显式 rejection sampling 和 MTP 参数漂移用例。
  对原 137 Prefill / 136 Decode 容器只读获取 argv/env，两端均通过新增校验；
  这仍是原部署的检查，不是新的 2P1D 部署。记录在 `.tmp/pd-fixes/live-contract-check.json`，
  测试代码均在 `.tmp/tests/`。
- **三节点实测**：44 项 RDMA WRITE preflight 通过；并发 sweep 的各档时长、吞吐、延迟和错误统计见
  [`results/`](results/)。完整服务身份、启动时间、rail/GPU/OOM 计数保存在每档 before/after 审计中。
- **验证边界**：使用模拟 acceptance，未验证真实 token 正确性；未做网络故障注入或验证
  failed-session 恢复时延。测试期间出现的 HTTP/KV 等待异常单独记录在
  [`measurement_notes.md`](results/measurement_notes.md)，不能用零 rail/GPU fault 概括为“没有任何错误”。

在套件根目录可独立运行镜像检查：

```bash
bash scripts/verify_pd_fixes.sh
# 仅检查已具备 Docker 的节点：
bash scripts/verify_pd_fixes.sh 'VERIFY_NODES=crsuse2-m2m-136 crsuse2-m2m-137'
```

日后重建时，当前仓库 Dockerfile 已包含基础 DSA/PD 补丁，并将编译本次更新后的
Rust router；本套件 `patches/` 下的 HiCache/NextN 是参考镜像额外应用的两项，
不能因为通用 Dockerfile 构建成功就假定也已经包含。使用新镜像时更新
`IMAGE` 与 `EXPECTED_IMAGE_ID`，先通过能力检查，再按同一套脚本执行 GPU 验证。
