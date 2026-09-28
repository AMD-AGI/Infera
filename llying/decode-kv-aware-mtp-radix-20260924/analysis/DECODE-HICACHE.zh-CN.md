# decode 侧在 MTP + DPA + radix cache 基础上开启 HiCache

日期：2026-09-24。前置工作是 decode radix cache，见 [SOLUTION.zh-CN.md](SOLUTION.zh-CN.md) 和 [GATE-RESULTS.zh-CN.md](GATE-RESULTS.zh-CN.md)。

## 结论

**以前开不了的原因：** SGLang 中的 HiCache 必须建在 radix 树上。decode 一开 MTP，就会被强制使用 ChunkCache（没有 radix 树），于是参数检查阶段报错：

`enable-hierarchical-cache and disable-radix-cache are mutually exclusive`

测试脚本 `engine.sh` 第 76–79 行的 guard（"Decode MTP and HiCache cannot both be enabled"）就是提前挡住这个报错。

**打上 decode radix 补丁后：** 参数层面已经不缺东西。剩下的是脚本里这条 guard，以及运行时验证。SGLang 的 decode HiCache 流程和 MTP draft KV 的主机备份都已经实现，但还没有在这个组合下跑过。

## 参数层面复现（仅 CPU）

脚本：[repro_decode_hicache_args.py](../scripts/repro_decode_hicache_args.py)。输出：[evidence/repro-decode-hicache-args.txt](evidence/repro-decode-hicache-args.txt)。

做法：用 GLM-5.2 的 decode 命令行，加上与 prefill 相同的 HiCache 参数；按流水线顺序依次运行 SGLang 的 PD、HiCache 和缓存兼容性钩子。

| 镜像 | 结果 |
|---|---|
| 原镜像 | Infera 不追加 radix 开关，decode 被强制为 ChunkCache，缓存兼容性钩子报上述 `ValueError` |
| 补丁镜像，设置 `SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1` | radix 开关被追加，三个钩子全部通过 |

镜像内有两份 Infera，引擎导入的是 `/opt/infera`。复现时需要把 `/opt/infera` 放在 `PYTHONPATH` 最前面，否则导入的是 site-packages 中那份。重建后的镜像两份都已打补丁。

## SGLang 中已有的实现

1. **decode HiCache 流程完整**（`disaggregation/decode_hicache_mixin.py`，上游 #26227）。
   - decode 匹配前缀时，同时统计显存中的命中（L1）、主机内存中的命中（L2），以及外部存储中的命中（L3，本配置没有外部存储）。三者之和作为 `decode_prefix_len` 发给 prefill，prefill 只发送其后的部分。
   - L2 命中的部分由 decode 自己从主机加载回显存。`HiCacheRestoreGatedKVReceiver` 会等加载完成后，才把这个请求当作传输完成。
   - `UnifiedRadixCache` 实现了该流程需要的全部方法，包括 `init_load_back`、`is_load_back_event_done`、`query_storage_hit_length` 等。
2. **MTP 的 draft KV 会随 target 一起备份到主机。**
   - HiCache 开启时，SGLang 为 NextN MTP 选择 PACKED 方式，把 draft 层当作额外的层打包进同一个主机内存池（`speculative/base_spec_worker.py` 中的 `_build_hicache_draft_plan`）。
   - DSA 模型走 `_DsaStrategy` 和 `build_anchor_sidecar_stack`：只有带 indexer 缓冲区的 draft 池才会被打包。GLM-5.2 的 draft 架构 `GlmMoeDsaForCausalLMNextN` 属于 DSA，满足这个条件，所以 draft 层的 KV 和 indexer 都会备份和恢复。
   - 启动时可以验证：decode 日志中的 `transfer_layer_num` 应为 78+1。
3. **镜像里已有的 HiCache 补丁与 MTP 无关。** 它们是 ROCm 上的主机传输 kernel 粒度修复，以及 HIP 上 MLA 写回不使用 JIT kernel 的修改（`GLM52_ROCM_STAGED_WRITE_BACK`）。prefill 的 HiCache 已经在这些补丁上跑通。
4. **有现成的指标**（`--enable-metrics` 下默认上报，按 rank 分开）：

| 指标 | 含义 |
|---|---|
| `sglang:hicache_backup_tokens_total` | 从显存写入主机的 token 数 |
| `sglang:load_back_tokens_total` | 从主机加载回显存的 token 数 |
| `sglang:hicache_host_used_tokens` | 主机内存池已用 token 数 |
| `sglang:evicted_tokens_total` | 从显存淘汰的 token 数 |

## 需要补的地方

| 位置 | 内容 | 本次做法 |
|---|---|---|
| 测试脚本 `engine.sh` 第 76–79 行 | 拒绝 decode 同时开 HiCache 和 MTP | 不改脚本，HiCache 参数通过 `DECODE_EXTRA_ARGS` 传入（`DECODE_HICACHE` 保持 0，所以脚本打印的 `HiCache=0` 不准确）。长期可以改为：设置了 decode radix 开关时放行 |
| SGLang / Infera | 参数层面无需再改 | 使用已有的 decode radix 补丁 |

## 需要在 GPU 上验证的风险

- **draft KV 恢复是否正确：** 从主机加载回来之后，MTP 接受长度是否下降。draft 出问题只会影响接受长度；target KV 出问题才会导致输出错误。
- **write_through 的开销：** decode 每次向树中插入都会写主机内存，对 ITL 的影响未知。本次只验证正确性，不测性能。
- **主机内存：** 每 token 在主机上约占 45 KB（KV）+ 10 KB（indexer）+ 0.7 KB（draft）。按 ratio 1.5 计算，decode 满容量时每个 rank 约 3.3M token、约 187 GB，4 个 rank 约 750 GB。138 有 2.7 TB 内存。
- **decode 从主机加载的路径：** 这条路径在 ROCm 的 DSA 上还没跑过，而且带有"加载完成才放行传输"的等待。
- **retraction：** 在 HIP 上，retraction 固定使用 `cpu_tensor` 备份，不使用主机内存池，与 HiCache 独立。
- **Router：** decode 会发布主机层的事件。Router 当前的处理方式与已开 HiCache 的 prefill 相同，不需要修改。

## C 组测试设计（P+D 都开 HiCache）

**配置：** B 组 + decode HiCache。decode 的 HiCache 参数与 prefill 相同：ratio 1.5，write_through，kernel IO，page_first。

另外有两个测试专用参数，只是为了更快触发淘汰，不改变代码路径：
- `--max-total-tokens 200000`：把 decode 每个 rank 的显存 KV 从约 222 万 token 降到 20 万。
- `--hicache-size 60`：每个 rank 的主机 KV 池设为 60 GB，约 130 万 token。这样被淘汰的前缀能留在主机上，不会很快又被挤出。

配置文件：[config.decrad.c.sh](../scripts/bench-harness/results/decrad/config.decrad.c.sh)。

**测试步骤：**

1. **启动检查：**
   - decode 日志中出现 `hicache_attached=True`，并分配了主机 KV 池和 indexer 池。
   - 主机池的 `transfer_layer_num` 包含 draft 层。
   - `server_info` 中 HiCache 已开启，decode radix 已开启。
2. **固定 few-shot GSM8K：** 在淘汰压力下验证准确率。GSM8K 的共享前缀一直被正在运行的请求使用，不会被淘汰。但每道题独有的部分会持续写入主机并被淘汰。
3. **从主机加载的专项测试**（[l2_probe.py](../scripts/l2_probe.py)）：
   - **写入：** 8 份约 2 万 token 的长文档各发一个问题，decode 把它们写进树，同时写入主机。
   - **挤出：** 再发约 80 份其他长文档，每个 rank 约 40 万 token，是 decode 显存上限的 2 倍。这样先前的 8 份文档会被挤出 decode 显存，但仍留在主机上。prefill 显存很大，这些文档仍在 prefill 上，Router 会把后续请求送回同一个 rank。
   - **复用：** 对这 8 份文档各问一个新问题。decode 应在主机上命中，并从主机加载回显存。

## C 组结果（2026-09-24，135 + 138，GPU 2–5）

全部通过。证据在 `evidence/gate/` 下：`c-launch/`、`c-gsm8k-fixedshot/`、`c-metrics-*.prom`、`c-l2-probe.json`、`c-router-l2.log`。

**启动检查**
- decode 每个 rank 的显存 KV 为 200,000 token。
- 主机 KV 池 60 GB，共 1,318,592 token。日志注明 `packed MTP KV layers: target_layers=78, draft_layers=1, total_layers=79`，DSA indexer 的主机池（13.75 GB）同样是 79 层。
- `UnifiedRadixCache` 的 `hicache_attached=True`。
- 两边的 `server_info` 都显示 HiCache 已开启（write_through / kernel / page_first）；decode 的 radix 已开启。
- Router 订阅了两个 worker 的全部 4 个 rank。

**固定 few-shot GSM8K（完整 1319 题，真实接受率）**
- strict-match 0.9727 ± 0.0045，flexible-extract 0.9727，通过阈值 0.9。B 组为 0.9757，差距在误差范围内。
- 运行期间 decode 写入主机 997,952 token，从显存淘汰 199,104 token，从主机加载 0。GSM8K 的共享前缀一直在显存中，不需要从主机加载，符合预期。
- decode 日志中 MTP 接受长度约 3.5–4.0；没有 retraction。

**从主机加载的专项测试**（[l2_probe.py](../scripts/l2_probe.py)）

| 阶段 | 回答正确 | 写入主机 | 从显存淘汰 | 从主机加载 |
|---|---|---|---|---|
| 写入：8 份约 2 万 token 的文档 | 8/8 | +162,816 | +164,032 | 0 |
| 挤出：80 份其他文档，约 160 万 token | 80/80 | +1,628,096 | +1,693,120 | 0 |
| 复用：对这 8 份文档各问一个新问题 | 8/8 | +512 | +162,752 | **+162,304** |

- 复用阶段 8 个请求的 prompt 共 162,607 token，其中 162,304 个（99.8%）由 decode 从主机加载回显存。余下的部分（最后不满一页的内容和新问题）由 prefill 传输。
- Router 日志显示，复用阶段 8 个请求的 prefill 和 decode 都在同一个 rank 上。prefill 命中 317/317 块。Router 看到的 decode 命中为 0，因为 Router 只跟踪显存中的块；decode 引擎实际是在主机上命中的。
- decode 日志中没有加载失败、淘汰不足或 retraction 的记录。
- 复用阶段有 2 个请求用了约 12 秒，其余约 1 秒。**已查明：** 这是 prefill 在现场编译 tilelang 稀疏注意力内核（每个 query token 档位只编译一次），与 decode HiCache 无关。见 [PREFILL-JIT-STALL.zh-CN.md](PREFILL-JIT-STALL.zh-CN.md)。

**尚未验证：** 性能（本次 decode 显存被人为调小，不能用于性能对比）、decode 全容量下长时间运行的行为、AgentX 负载。

**通过标准：**
- 不出现乱码，没有请求报错。
- GSM8K 不低于阈值 0.9。
- 专项测试中所有回答与答案一致。
- 各阶段 decode 指标的增量：
  - 写入和挤出阶段，`hicache_backup_tokens_total` 与 `evicted_tokens_total` 增加；
  - 复用阶段，`load_back_tokens_total` 增加，量级接近 8 份文档的长度。
