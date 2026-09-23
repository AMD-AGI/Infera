# GLM-5.2 2P1D full-configuration bring-up

## 第一章：概括

`config.sh` 是当前测过、可用于回归的配置。它保留已经验证过的保守选择：

- 2P1D、每个 worker TP8/DP8/DPA；
- Prefill/Decode HiCache 关闭；
- max-running 和 CUDA graph max BS 为 8；
- `index_share_for_mtp_iteration=false`；
- AgentX warmup lane 为 1，默认 profiling 1,200 秒。

`config.full.sh` 是需要跑通的全量目标，不是当前已验证配置。它使用同一个
`topology.tsv`，并设置：

- Prefill HiCache 开启；Decode 因 MTP 约束暂时保持 HiCache 关闭；
- AgentX C128；
- Prefill/Decode max-running 和 graph max BS 为 128；
- AgentX warmup lane 为 10；
- profiling 3,600 秒；
- 移除 `index_share_for_mtp_iteration=false` workaround；
- 打开三个单节点优化对齐项。

运行全量目标时统一显式指定：

```bash
CONFIG=./config.full.sh
```

### 推荐 TODO

| Feature | Args | TODO |
|---|---|---|
| FlyDSL DSA sparse MLA | `--dsa-prefill-backend flydsl`<br>`--dsa-decode-backend flydsl` | 先移植 AITER gfx950 sparse MLA kernels，再按 v0.5.19 当前 DSA/graph API 移植 SGLang integration；TP8 必须覆盖 Prefill/Decode 8-head padding。 |
| AITER fused DSA top-k | `--dsa-topk-backend aiter` | 移植 `dsa_topk_transform` native kernel、binding、MTP shape 支持及 SGLang dispatch/fallback；不能只增加 CLI 选项。 |
| Unified JIT grouped-topk router | `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1` | 当前 nightly 已有实现；单独开启，保持其余变量不变，完成 correctness 和 AgentX A/B。 |
| 移除 IndexShare workaround | 不再传 `--json-model-override-args '{"index_share_for_mtp_iteration":false}'` | 当前不设置该 override 会在 smoke 中出现乱码。需要定位 DSA IndexShare seed、page-table/坐标映射或 MTP handoff 的根因；乱码、GSM8K、long-context 全部通过后才能关闭 TODO。 |
| Hicache打开 | Hicache打开后的情况未知，是否有问题或者有patch需移植需check |

### 可能仍需解决的问题

以下问题来自 `llying/bench/sglang_glm5p2_agentx` 的实测。当前
`dev/pd_opt/glm_5.2_agentx` 没有完整采用对应修复，或者底层 root cause 仍未闭合。

| 问题 | 已知证据 | 当前状态 / TODO |
|---|---|---|
| Cross-rank Mooncake transfer failed 与 session blacklist | 600 秒内 48 条真实失败全部为跨 DP rank handoff，同 rank 为 0；首次失败后 session blacklist 持续放大故障。 | failed-session probe 已启用，但 router DP-rank affinity 未采用。需要移植/重做 `99fa040` 并验证 2P1D rank pool 缺失时返回 503。 |
| 镜像升级后 Decode ITL 性能回退 | 同名 tag 实际从 `c29bd17/b02ab81` 漂移到 `402df1e/2c71811`，C32--C64 ITL P50 从约 10--12 ms 升至 43--47 ms。 | 当前只校验各节点 Image ID 一致，仍需记录并核对镜像内 SGLang/AITER commit；新 nightly 要做单变量性能 A/B。 |
| Decode gfx950 fused DSA indexer memory access fault | 新栈真实 MTP GSM8K 运行 10--13 分钟后四个 Decode rank 同时 fault；禁用 fused indexer 后 1,319/1,319 通过。 | 只有绕过方案，没有 kernel-level RCA。需确认当前 v0.5.19 fused 路径是否仍受影响，并在保持 fused 真正执行时定位 fault。 |
| Prefill `HSA_STATUS_ERROR_OUT_OF_RESOURCES` | Prefill 动态 extend 时单 rank HSA allocation 失败；active allocator GC 曾把 HBM 峰值从 99% 降至 93%。 | 当前仅采用按角色 `HSA_NO_SCRATCH_RECLAIM`，未采用 allocator GC hook。config.full 的 C128+HiCache 更容易触发，需要监控后决定是否移植 `a325c0f`。 |

## 第二章：全量配置四项 TODO 展开

### 2.1 FlyDSL DSA sparse MLA

当前 v0.5.19 nightly 的 `--dsa-prefill-backend` /
`--dsa-decode-backend` 不接受 `flydsl`，当前 AITER source tree 也没有目标
FlyDSL sparse MLA modules。`AITER_USE_FLYDSL_MOE_SORTING=1` 只说明 MoE
sorting/fused-MoE 使用 FlyDSL，不等价于 DSA attention 已启用。

#### AITER 依赖

[AITER PR #11](https://github.com/xiaobochen-amd/aiter/pull/11)，merge
[`5c4095ea`](https://github.com/xiaobochen-amd/aiter/commit/5c4095eaaabbf3fdd2f4d0d59b84642605709f54)
提供：

- gfx950 FP8 sparse MLA Prefill；
- gfx950 FP8 sparse MLA Decode；
- split reducer；
- scratch/workspace contract；
- compile-cache 和 correctness tests。

#### SGLang 依赖

- [PR #31](https://github.com/xiaobochen-amd/sglang/pull/31) /
  [`eea4cc7c`](https://github.com/xiaobochen-amd/sglang/commit/eea4cc7c4c74927cfcdec3d76344d00b1087f1f9)：
  backend choice、dispatch、workspace、KV pool sizing、graph 支持；
- [PR #40](https://github.com/xiaobochen-amd/sglang/pull/40) /
  [`e95513b2`](https://github.com/xiaobochen-amd/sglang/commit/e95513b255c2cd631e69629aac693d3746e4abe8)：
  TP8 Decode 8→16 heads padding；
- [PR #41](https://github.com/xiaobochen-amd/sglang/pull/41) /
  [`70278810`](https://github.com/xiaobochen-amd/sglang/commit/70278810579ac53b94979823d1cc809aa4e8c9e7)：
  TP8 Prefill 8→16 heads padding。

默认 2P1D 每个 worker 都是 TP8，#40/#41 缺一都会让对应路径静默回退。

#### 建议操作

采用 Docker build-time patch，但必须同时 patch AITER 与 SGLang：

```text
patch AITER
→ kernel compile/correctness smoke
→ patch SGLang
→ graph capture + TP8 Prefill/Decode engage check
→ smoke/GSM8K/AgentX A/B
```

旧 v0.5.18 commit 不应直接 cherry-pick；需要针对当前 nightly 的
`arg_groups`、DSA metadata、CUDA graph 和 speculative worker 重新裁剪。

### 2.2 AITER fused DSA top-k

当前 `--dsa-topk-backend` 只接受 `sgl-kernel`、`torch`、`flashinfer`；
当前 AITER 也没有 `dsa_topk_transform`。`dsa_paged_mqa_logits_backend=auto`
虽然在 ROCm 上使用 AITER，但它负责 indexer logits，不是 fused top-k +
page-table transform。

#### AITER commits

1. [`057f6e2e`](https://github.com/xiaobochen-amd/aiter/commit/057f6e2e3c8e3dae3594d166daaaafffe830ea99)：
   cooperative top-k + page-table transform；
2. [`f4d8f973`](https://github.com/xiaobochen-amd/aiter/commit/f4d8f9730f4e623891f6690d62356bb7f1e36b59)：
   pybind/build 修复；
3. [`27ca3fc8`](https://github.com/xiaobochen-amd/aiter/commit/27ca3fc898670c5a94c51f694748812552d91d2a)：
   speculative-decode row shapes，Decode MTP 必需。

#### SGLang commits

1. [`6d867ebc`](https://github.com/xiaobochen-amd/sglang/commit/6d867ebc2e8441283f1588eca81f01351192f155)：
   AITER backend 与核心 dispatch；
2. [`74f2f91d`](https://github.com/xiaobochen-amd/sglang/commit/74f2f91dfebbe4874ff56613f128a3aa6f42ef94)：
   fused page-table path；
3. [`98b5e0fa`](https://github.com/xiaobochen-amd/sglang/commit/98b5e0fa380db6fcac3880d0a7a5b6f9ee57fc43)：
   PAGED Prefill row mapping；
4. [`67e5b9c3`](https://github.com/xiaobochen-amd/sglang/commit/67e5b9c3a48b6c8c8769cb271a7fd3d3650d7b1d)：
   ROCm 自动选择与 unsupported-shape fallback。

旧实现只在 ROCm、`index_topk == 2048` 且 AITER symbol 存在时使用该路径。
移植后必须保留 fallback，并覆盖 Decode、target verify、draft extend 和 PAGED
Prefill。旧注释中的 1.82--2.22x 是 kernel-level 结果，不代表端到端收益。

### 2.3 Unified JIT grouped-topk router

当前 nightly 已包含上游
[`6657f7d8`](https://github.com/sgl-project/sglang/commit/6657f7d8449f7c739906b6f8ebcb016dddcbceef)，
等价于 xiaobochen fork 的 [PR #36](https://github.com/xiaobochen-amd/sglang/pull/36) /
[`5d1cb45b`](https://github.com/xiaobochen-amd/sglang/commit/5d1cb45b5362e44f1ee2b229543bc7c0f5ec2cc1)。

无需移植代码或修改 AITER。`config.full.sh` 已设置：

```bash
SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1
```

验收必须使用相同镜像、topology、C128、warmup 和 duration，仅切换该环境变量。
若 correctness 不变但没有稳定性能收益，继续保持 `config.sh` 中默认关闭。

### 2.4 移除 `index_share_for_mtp_iteration=false`

`config.sh` 保留：

```bash
JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":false}'
```

`config.full.sh` 将 `JSON_MODEL_OVERRIDE_ARGS` 设为空，不再传该 workaround。
目前已知结果是 smoke 输出乱码，因此“移除 override”不是已经完成的优化，而是
需要闭合的 correctness TODO。

需要至少完成：

1. 固定同一镜像和随机输入，对比 IndexShare off/on 的 token IDs；
2. 检查 Prefill 输出的 DSA top-k seed 与 Decode 本地 page-table remap；
3. 区分 draft seed 错误、KV handoff 错误和 tokenizer/parser 层乱码；
4. 检查 CUDA graph/eager、overlap on/off 是否改变乱码；
5. smoke、GSM8K、long-context 和 AgentX warmup 全部通过；
6. 确认不是通过回退/禁用 IndexShare 隐藏问题。

## 第三章：旧 campaign 暴露的四个问题

### 3.1 Cross-rank Mooncake transfer failed 与 session blacklist

#### 现象与证据

KV-aware policy 会分别选择 Prefill 和 Decode DP rank。对称 P8/D8 DPA
部署中出现过 P7→D0、P6→D2 等跨 rank handoff；Mooncake 沿用 Prefill rank
同名 rail，而目标 KV 位于另一张 Decode GPU，发送端出现
`tx_rdma_ack_timeout`、`req_tx_retry_excd_err` 和
`transport retry counter exceeded`。

600 秒诊断保全 48 条真实失败：全部为跨 rank handoff，同 rank 为 0，其中
P7→D0 有 31 次。首次 send 失败后，整个 `mooncake_session_id` 被加入
`failed_sessions`，后续请求在真实传输前即 early failure。

#### 已知方案与缺口

- 根因修复：
  [`99fa040`](https://github.com/AMD-AGI/Infera/commit/99fa0406cc2ce3e7eedb8a3349b2626eca1339b2)
  将 Decode 限制到 Prefill 的 effective DP rank；
- 恢复增强：
  [`850d555`](https://github.com/AMD-AGI/Infera/commit/850d55536a71174aceb1d0309e4b4f184c3913c2)
  定期 probe failed session；
- 证据归档：
  [`713e4a2`](https://github.com/AMD-AGI/Infera/commit/713e4a2a2c2960a7dfc9bbebf6b6c2eb32aabf92)。

当前 baseline 只采用 failed-session probe，没有采用 router rank affinity。
需要确认 2P1D 多 Prefill pool 下 effective-rank 匹配策略，并在目标 pool 没有
该 rank 时明确返回 503，不能静默跨 rank。

另外，这个问题现象lomou给出了其他的可能原因，需要确认是否是同一问题， 文档已单独发送给yinxing。

### 3.2 镜像升级后 Decode ITL 性能回退

#### 现象与证据

tag `glm52-v518-c29bd17-b02ab81` 曾被重建覆盖：tag 名仍表示
`c29bd17/b02ab81`，实际却是 `402df1e/2c71811`。错误镜像下 C32/C48/C64
的 ITL P50 为 43.00/46.09/46.51 ms；已知良好镜像约为 9.9--11.8 ms。
C96 在同一错误镜像上恢复到 12.36 ms，说明不是简单的恒定 slowdown。

构建层根因已知：旧 build 没有转发 `SGLANG_SHA` / `AITER_SHA`，verify 又只
比较节点间 Image ID，没有验证镜像内 source commit。

#### 当前处置与缺口

- 防复发 commit：
  [`24ec6e2`](https://github.com/AMD-AGI/Infera/commit/24ec6e2e58dce7093c346d522609881777d64937)；
- 实验归档：
  [`713e4a2`](https://github.com/AMD-AGI/Infera/commit/713e4a2a2c2960a7dfc9bbebf6b6c2eb32aabf92)。

当前 baseline 使用新的 v0.5.19 nightly，此问题是否存在需要重新确认。若存在， yaocheng的修复方案https://github.com/xiaobochen-amd/aiter/pull/22。

### 3.3 Decode gfx950 fused DSA indexer memory access fault

#### 现象与证据

`402df1e/2c71811` 在 TP4/EP4、C8、真实 EAGLE MTP GSM8K 中通常运行
10--13 分钟后四个 Decode rank 同时 GPU memory access fault。关闭 Decode
CUDA graph 后约 51 分钟仍复现，所以 graph 不是必要触发条件。

同一 failing digest：

- Decode 加 `--no-enable-dsa-fused-indexer` 后 1,319/1,319 通过；
- 恢复 fused 路径后稳定复现；
- `dmesg` 为页面存在但 GPU 读权限 fault。

这只把范围收敛到 fused DSA indexer 或其改变的 metadata/workspace/buffer
生命周期，没有定位具体 faulting kernel 或指令。


#### 当前处置与缺口

旧分支没有功能性 root-cause fix；关闭 fused indexer 和固定旧镜像都只是绕过。
当前 v0.5.19 使用不同实现，需要先确认问题是否仍存在：

该问题与yanyuan未合入的pr有关，所以baseline使用的nightly镜像不一定存在该问题。如果存在或者后续为了性能需要patch进来的话，可以首先验证yanyuan之前debug过的方案， 如下：

原因找到 https://claude.ai/artifact/RNLsD3ueHWYvFk2WRT7tPd，因为你的测试镜像v0.5.18 base 不包含main上的#36714 [AMD][Spec][PD] Enable the PD DSA fused-TopK seed remap on ROCm by tianxiaojiang4 · Pull Request #36…，可以先v0.5.18 base 镜像+cherry pick 这个pr，或者用最新的sglang镜像 （包含#36714）测试 lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260911
我们的pr合入前不断rebase on main，我们最早用的v0.5.18 有些out dated，可以换到新一点的镜像

### 3.4 Prefill `HSA_STATUS_ERROR_OUT_OF_RESOURCES`

#### 现象与证据

C8/C16/C48/C96/C192 的失败发生在 Prefill 动态 extend forward：单个 DPA
scheduler 报 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`、`Available Free mem: 0 MB`
并 fatal abort。降低静态 KV fraction 只能延后问题，8 GiB scratch limit 也未
降低最坏 HBM 峰值。

根因边界是：PyTorch allocator 持有 inactive cached segments，而 ROCr/AITER
在 PyTorch 之外申请动态 scratch/临时张量；外部 HSA allocation 失败不会触发
PyTorch failure-path cache release。

#### 已知方案与缺口

旧分支 [`a325c0f`](https://github.com/AMD-AGI/Infera/commit/a325c0f43db2af41d91ffb2054daf9b6d1521b65)
提供：

- Prefill `HSA_NO_SCRATCH_RECLAIM=0`；
- Decode `HSA_NO_SCRATCH_RECLAIM=1`；
- `PREFILL_PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8`；
- `PREFILL_PYTORCH_MEMORY_FRACTION=1.0`；
- 启动时调用 `torch.cuda.set_per_process_memory_fraction()` 的 hook。

真实 GC treatment 曾把 Prefill HBM 峰值从 99% 降到 93%，并完成 96/96
warmup 和完整 1,200 秒 profiling。

#### 当前 baseline 已具备

- `config.sh` 和 `config.full.sh` 都按角色设置：
  - Prefill `HSA_NO_SCRATCH_RECLAIM=0`；
  - Decode `HSA_NO_SCRATCH_RECLAIM=1`。
- `engine.sh` 会根据 role 选择上述值，并显式传入 worker 容器。
- Prefill/Decode `mem_fraction_static` 可分别配置；当前两个 config 默认都是
  `0.85`，出现容量压力时可以单独降低 Prefill，不影响 Decode。
- `check_nodes.sh` 会在启动前检查每张 GPU 的 VRAM% 和利用率。
- worker 日志会持久化到 `OUT_DIR/server-logs/<instance>.log`，便于保留
  Prefill OOR 与 HSA 报错现场。
- AITER JIT cache 按 Image ID 持久化，可减少重复启动时的 JIT 峰值和启动时间；
  它不是运行期 allocator GC，也不能替代 OOR 修复。

#### 当前 baseline 仍缺失

- 没有按角色配置/转发 `HSA_SCRATCH_SINGLE_LIMIT_ASYNC`。
- 没有按角色配置/转发
  `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8`。
- 没有 `PREFILL_PYTORCH_MEMORY_FRACTION`，也没有调用
  `torch.cuda.set_per_process_memory_fraction()` 的启动 hook，因此 vendor
  PyTorch 的 active GC 路径当前没有被显式启用。
- 没有 benchmark 运行期间的持续 HBM/ROCr residency sampler；启动前
  `check_nodes.sh` 无法发现运行中逐步累积的 allocator cache。
- 没有把 HBM 峰值、Prefill rank OOR 和完成请求数做成自动验收 gate。

`config.full.sh` 的 C128、warmup 10、3,600 秒和 Prefill HiCache 会提高压力。
应先监控实际 HBM/ROCr 行为；若复现，再移植 allocator GC 或评估其他修复
方案，不应预先把所有 workaround 默认打开。

## 附录 A：高并发 AgentX 期间简单请求返回乱码（待交接复现）

### 状态

- 报告人：Limou。
- 当前文档维护者没有复现过该问题。
- 当前 baseline 分支尚未验证该问题是否仍然存在。
- 本附录只记录交接信息和建议复现方法，不对根因作结论。
- 该问题与 2.4 节“移除 IndexShare workaround 后 smoke 乱码”现象相似，但
  触发条件不同；在证据闭合前不能视为同一个问题。

### Limou 报告的现象

在大并发 AgentX benchmark 正在向服务持续施压时，同时向同一个服务端点发送
普通、简单的 chat completion 请求，偶发得到乱码回复。当前缺少以下原始信息：

- 首次出现问题时的 topology、镜像 digest 和完整配置；
- AgentX concurrency、warmup、持续时间及出现时间点；
- 简单请求是否 streaming、具体 prompt 和 sampling 参数；
- 原始 HTTP body、状态码、headers 和对应 worker；
- “乱码”属于非法 UTF-8、JSON 内容中的 `�`、异常 token 文本，还是模型语义错误。

### 建议接手人复现矩阵

先使用已测过的 `config.sh`，不要从 `config.full.sh` 开始，以免同时引入
FlyDSL/AITER top-k/IndexShare 等未闭合变量。

1. 服务空闲时先发送固定探针，建立 control。
2. 分别运行 C64、C96、C128 AgentX；其他配置保持不变。
3. AgentX profiling 期间每 1--5 秒发送一次非 streaming 固定请求，例如：

   ```text
   Reply with exactly: BASELINE_OK_<nonce>
   ```

4. 每次探针保存：
   - UTC 时间、nonce、HTTP 状态码和耗时；
   - 未解码的原始 response bytes；
   - JSON 解析结果、`message.content`、`reasoning_content`、finish reason；
   - `/v1/workers` 快照和可用的请求/worker 标识；
   - 同一时间窗的 router、Prefill、Decode 日志。
5. 对失败探针分别直连 router 和具体 Decode worker；只有 router 路径异常时优先
   检查路由/响应转发，两条路径都异常时再检查 engine/token/KV。
6. 问题稳定复现后，再单独切换
   `index_share_for_mtp_iteration=false`、streaming、MTP 和 KV-aware，禁止一次
   改多个变量。

### 判定与交付标准

一次探针至少区分以下结果：

- transport/timeout 错误；
- HTTP 成功但 body 不是合法 JSON；
- JSON 合法但字符串包含 U+FFFD replacement character；
- 字符串编码正常但 token 文本异常；
- 文本正常但没有遵循固定 nonce。

复现交付物应包含最小失败请求、原始 bytes、对应服务日志和可重复的触发并发。
在这些证据齐全前，不应将其归因于 tokenizer、reasoning parser、IndexShare、
Mooncake KV transfer 或 router 中的任何一层。
