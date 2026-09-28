# 开启 decode radix cache + decode KV 事件（MTP 下）的方案

日期：2026-09-24。原因分析见 [ROOT-CAUSE.zh-CN.md](ROOT-CAUSE.zh-CN.md)。

## 1. 方案概述

在 SGLang 中增加一个默认关闭的实验开关，只对 **EAGLE/NEXTN 且 `--speculative-eagle-topk 1`** 放开 decode radix cache 与投机解码的互斥。Infera 和测试脚本使用同一个环境变量，按需把开关传下去。未设置环境变量时，三处行为都与现在完全一致。

| 补丁 | 作用位置 | 内容 |
|---|---|---|
| [01-sglang-decode-radix-allow-eagle.patch](../patches/01-sglang-decode-radix-allow-eagle.patch) | 镜像内 `/sgl-workspace/sglang`，`arg_groups/pd_disaggregation_hook.py` | 当 `SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1`、算法为 EAGLE/NEXTN 且 topk 为 1 时，不再抛出 `ValueError`，改为打印 EXPERIMENTAL 警告；其他情况照旧拒绝，并在报错中说明如何开启 |
| [02-infera-decode-radix-spec-opt-in.patch](../patches/02-infera-decode-radix-spec-opt-in.patch) | `infera/engine/sglang/args.py` 及其测试 | 设置同一环境变量时，MTP 下的 decode 也追加 `--disaggregation-decode-enable-radix-cache`；模型兼容性检查（SWA/SSM、hisparse、dcp）仍然生效；新增 3 个单元测试 |
| [03-harness-decode-kv-aware.patch](../patches/03-harness-decode-kv-aware.patch) | tracing-aus 测试脚本的 `engine.sh`、`config.sh` | 新增 `DECODE_KV_AWARE`（默认 0）；为 1 时 decode 与 prefill 一样传 `--enable-kv-events`，开启 MTP 时额外设置上述环境变量 |

镜像构建使用 [docker/Dockerfile](../docker/Dockerfile)。启动覆盖项见 [config/decode-kv-aware.sh](../config/decode-kv-aware.sh)。

放开之后的完整链路是：`DECODE_KV_AWARE=1` → decode 的 Infera 追加 radix 开关 → SGLang 接受该开关，decode 使用 `UnifiedRadixCache`（GLM-5.2 只有 FULL 组件）→ 各 DP rank 发布事件（EAGLE 下为 bigram 格式）→ Router 按 rank 订阅并建立 decode 视图。Router 已支持 bigram 事件（`kv_event.rs` 中的 `as_u32_any`，有单元测试），不需要改动。

## 2. 为什么对 GLM-5.2 + EAGLE topk=1 放开是合理的

上游没有说明禁止的原因，因此下面逐项检查投机解码与 decode radix cache 的交汇点。结论是没有发现只在“PD decode + 投机解码 + radix”组合下才会出错的代码路径，但这只是静态阅读的结论，必须在 GPU 上验证（第 4 节）。

1. **radix 树本身支持 EAGLE。** `kv_cache_builder.py` 按投机算法设置 `CacheInitParams.is_eagle`（第 315 行）。`UnifiedRadixCache` 在匹配、插入和预取中都使用 bigram key。聚合部署下，EAGLE 与 radix cache 本来就是支持的组合。bigram key 要求第 i 与第 i+1 个 token 都匹配，比目标模型 KV 复用所需的条件更严格。
2. **请求结束时只插入已提交的 KV。** `release_kv_cache` 按 `effective_kv_committed_len` 插入树中，`_release_overallocated_kv_indices` 释放为 draft token 多分配的槽位（`mem_cache/common.py` 第 269-311 行）。这与聚合部署下 EAGLE 使用的是同一段代码。topk=1 时，被接受的 token 是 draft 链的前缀，所以已提交部分是连续的。
3. **decode 预分配路径没有投机解码分支。** `_match_prefix_and_lock`、`_pre_alloc` 只处理 token 长度和页对齐（`disaggregation/decode.py` 第 704-722、1260-1400、1838-1975 行）。EAGLE 的起步状态（hidden states、top-k、DSA seed）来自 metadata buffer，与前缀是否命中无关。DSA seed 重映射读取 `req_to_token`，而前缀部分的 `req_to_token` 已写入树中的页。
4. **draft KV 与目标 KV 共用索引**（`decode.py` 第 583-594 行），复用前缀页时两者一起复用。当前部署中 prefill 没有 draft 模型，prompt 位置的 draft KV 本来就不是传过来的；复用只会让这部分内容来自此前 decode 真正计算过的位置。draft 质量只影响接受长度；输出正确性由 target verify 保证。因此如果出问题，draft KV 的问题表现为接受长度下降，target KV 的问题才会表现为输出错误。
5. **GLM-5.2 不属于上游 DSV4 PR 处理的那类情况。** GLM-5.2 是 DSA（与 DeepSeek-V3.2 同类），不是 SWA/SSM，不经过 SWA tail 预分配，也没有 C4/C128 压缩状态。#31097 中为 DSV4 做的“首个 decode forward 之后再插入 prompt”等处理针对的是 DSV4 的压缩状态，这里不需要。
6. **Mooncake 支持增量传输。** `decode_prefix_len` 在 mooncake 的 conn 中已实现（#26227），prefill 从该位置开始发送（`prefill.py` 第 403-408 行）。EAGLE 下 bigram 匹配最多覆盖 N-1 个 token，再向下对齐到 64 的倍数，所以 prefill 至少会发送一页，不会出现完全不传输的情况。
7. **参数钩子的执行顺序合适。** PD 钩子在解析流程中早于投机解码钩子运行（`arg_groups/pipeline.py` 第 160、326 行），因此补丁判断的是用户传入的原始值（`EAGLE`/`NEXTN`、topk 1）。之后的钩子中没有会对 GLM-5.2 decode 重新关闭 radix 的逻辑。

## 3. 风险与未验证项

| 项目 | 说明 | 应对 |
|---|---|---|
| 没有 GPU 运行 | 目前只验证到参数层面：补丁能干净地应用；SGLang 钩子接受该组合；Infera 转发开关；单元测试通过 | 按第 4 节顺序验证 |
| overlap 调度下 prebuilt 阶段插入 | `process_prebuilt` 在首个 decode forward 前调用 `maybe_cache_unfinished_req`，把请求重新指向树中的页并释放重复页。上游非投机路径就是这样，但没有与 spec v2 overlap 一起测试过 | V1 中检查贪心输出是否与 radix 关闭时一致；关注 `KV cache is full` 断言和内存错误 |
| DP attention | SGLang 将其标为 EXPERIMENTAL。每个 attention DP rank 有独立的树，命中率取决于 rank 是否稳定 | 保持 `PD_DP_RANK_AFFINITY=1`，见下文“路由收益” |
| retraction | decode radix 下，retraction 后的重新 bootstrap 不做前缀匹配（代码中的 TODO），只是少复用，不影响正确性 | V3 压测中统计 retraction 次数 |
| decode 显存 | 已完成请求的前缀会作为可淘汰页留在显存中；准入时计为可用，分配时再淘汰 | 观察 `Eviction insufficient` 警告和 decode `token_usage` |
| 收益来源 | decode 前缀命中节省的是 prefill→decode 的 RDMA 传输量和 decode 显存中的重复前缀，**不节省 prefill 计算**：prefill 仍在本地计算完整 prompt，只发送增量（上游确认的行为，#19746 评论） | 评估指标看 TTFT、传输字节数和 decode 并发，不看 prefill 吞吐 |
| decode HiCache | 仍然关闭。脚本拒绝 decode HiCache 与 MTP 同时开启；SGLang 的 decode HiCache 要求先有 decode radix cache | 作为后续单独的任务 |
| 已知的 MTP 相关问题 | custom all-reduce、`index_share_for_mtp_iteration` 相关的已知问题（见 yihou 的两份分析）与本改动无关，但会干扰对比 | A/B 两边保持这两项一致 |

**路由收益。** 当前拓扑是 1P1D，各 DP8，并开启了 `PD_DP_RANK_AFFINITY`。Router 按 prefill 的 KV 重叠（权重 20）选 rank，decode 跟随同一 rank。因此 decode 事件在当前拓扑下不会改变任何路由决定。收益完全来自引擎侧的 decode radix cache：同一前缀会持续落在同一个 rank，decode 对应 rank 的树自然会积累这些前缀。decode 事件的作用是让 Router 能观察到 decode 的命中情况，也为以后的改进做准备。

## 4. 验证计划（需要 GPU）

> 进展（2026-09-24）：V0、V1 已在 P4D4（135 + 138）上完成，改用两轮完整 GSM8K 做统计判定。结果见 [GATE-RESULTS.zh-CN.md](GATE-RESULTS.zh-CN.md)。V3 和 V4 尚未进行。

每轮都重新启动 P/D，A 与 B 使用同一镜像（由 [docker/Dockerfile](../docker/Dockerfile) 构建），唯一差别是 `DECODE_KV_AWARE`。

**V0 启动检查。**
- decode 日志中出现 `EXPERIMENTAL: Decode radix cache with speculative decoding`、`EXPERIMENTAL: Radix cache is enabled for decode server`，不再出现 `KV cache is forced as chunk cache`。
- decode 的 `/server_info` 中 `disaggregation_decode_enable_radix_cache=true`，`kv_events_config` 不为空。
- Router 中不再出现 `not tracking decode worker`，并为 decode 的 8 个 rank 分别建立订阅。

**V1 正确性（先于任何性能测试）。**
- 使用真实接受率（不设置 `DECODE_SIMULATE_ACC_LEN`），temperature 0 的探测请求 16/16 通顺，nonce 能原样复述。
- 同一前缀连续发两轮：第二轮 decode 有前缀命中（`decode_prefix_len > 0`，`cached_tokens` 增加），贪心输出与 radix 关闭时逐 token 一致。
- 真实 `spec_accept_length` 不低于 radix 关闭时的基线。

**V2 事件与传输。**
- Router 的 `policy="kv-aware"` pick 日志中，`role=Decode` 的条目在第二轮出现 `cache_hits > 0`（与 `request_blocks` 对比）。Router 目前没有按 worker 导出已索引块数的指标，这条日志是最直接的信号；同时不应出现 `dropped store events whose parent was never seen` 的持续告警。
- 第二轮时，prefill 发送的页数相应减少（mooncake 日志或指标）。

**V3 压测。** AgentX C80，884 条 warmup，3600 秒窗口，4K chunk。对比 radix 开与关的 TTFT、decode 并发、token_usage、retraction 次数和错误率，并检查有无显存访问错误。

**V4 可选：Router 联合打分。** 在 rank 亲和模式下，把同一 rank 上 decode 的命中（乘以 `w_decode`）加到 prefill 的 rank 打分中。在 1P1D 中，这是 decode 视图唯一能影响决定的方式。此项需要先把镜像中的亲和补丁合入本仓库的 `rust/router`。

## 5. 使用方法

```bash
# 在有镜像的节点上，从本任务目录构建镜像
docker build -f docker/Dockerfile \
  --build-arg BASE=infera-sglang:v0519-yihou-0917-nextnfix-hicache \
  -t infera-sglang:v0519-yihou-0917-decode-radix-spec .

# 测试脚本打补丁（在仓库根目录）
git apply llying/decode-kv-aware-mtp-radix-20260924/patches/03-harness-decode-kv-aware.patch

# 启动时覆盖
IMAGE=infera-sglang:v0519-yihou-0917-decode-radix-spec \
  source llying/decode-kv-aware-mtp-radix-20260924/config/decode-kv-aware.sh
# 然后按原流程运行 launch.sh
```

不使用测试脚本时，只需要在 decode 容器中设置 `SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1`，并给 decode 传 `--enable-kv-events --kv-events on`。

## 6. 其他方案

| 方案 | 结论 |
|---|---|
| decode 关闭 MTP，换取上游已支持的 decode radix cache | 失去 MTP 的 decode 吞吐，不符合目标 |
| 保持现状，只用 prefill 侧 KV-aware | 可以运行，但 decode 每轮都要重新接收完整前缀，也不能在 decode 显存中共享前缀 |
| 等待上游 | 相关 PR 只针对 DSV4，且都未合入；ROCm 的 PR 明确排除投机解码 |
| 本方案：带开关放开 | 默认行为不变；在参数层面已验证；运行时正确性需要按第 4 节验证 |
