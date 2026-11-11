# `dsa_page_table_rows` 补丁与 `index_share_for_mtp_iteration`：是否还需要、为什么不崩（2026-11-11）

对象：Infera `deploy/docker/patches/sglang_dsa/dsa_page_table_rows.diff`（及 `v0522/` 重切版）。
结论：**MTP 草稿 topk=1 时，当前 sglang main（以及 v0.5.22）上这个补丁不需要**；InferenceX 发布（#4629 / 后续）无需把它推上游。
代码依据为当前 main（A/B 源码树 `44ad9d69`，main `daa17184` + #48047 + B2/B6/B8/#45524）与 v0.5.22 基础提交 `8dee3f6bdb`；
实测依据为 #48047 的 MI355X A/B 栈（AUS n05-29，11-10；kit 与日志在 `/perf_apps/liyingli/bench_agentx/e1ab/`）。

## 1. 补丁解决的问题

- 现象：DSA 模型（GLM-5.x）decode 开 MTP 时，`metadata.page_table_1` 每个**请求**一行，`topk_indices` 已被补齐成每个**token** 一行；
  行数不等时 `transform_index_page_table_decode` 的 `assert page_table.shape[0] == topk_indices.shape[0]` 失败，所有 rank 一起崩。
- 修法：`repeat_interleave(n_topk // n_rows)` 把 page table 按 token 展开（比例不是整数时截断或用最后一行补齐）；补出的行是 -1，kernel 会屏蔽。
- 触发条件（补丁说明）：MTP + `index_share_for_mtp_iteration=true`，当时在 HIP/tilelang decode 路径上发现。
  `index_share_for_mtp_iteration=false` 可替代它。补丁"UPSTREAM NONE"；思路相反的上游 #32209（截短 q / top-k）已关闭未合入。
- **它是崩溃修复，不是性能优化。**间接关系：允许在会崩的路径上用模型默认的 `index_share=true`。

## 2. `index_share_for_mtp_iteration` 在我们各阶段配方里的取值与效果（`llying/` 历史）

| 时间 / 文档 | 栈 | index_share | 说明 |
|---|---|---|---|
| 9 月 – 10-12（1P1D、p8d8、session-affinity、router、3p1d 等） | v0.5.18–v0.5.21 | false | 当时去掉 override 会 smoke 乱码（`*/scripts/bench-harness/issue.md` §2.4），作为规避 |
| 10-19 / 10-20（`mix-v0522-correctness`、`2p1d-v0522`） | v0.5.22 | 测过 true，最后仍 false | 正确性无问题；性能无收益（下表） |
| 10-27 起（`prefill-opt`，yihou llr35 配方，E1 / E10 51,374） | v0.5.22 + 全套补丁 | **true**（两侧） | 沿用 llr35；乱码二分里去掉 index_share 不能消除乱码（非乱码原因） |
| 11-03 起（`dp-graph-variant-fix-aus`，最佳 53,666） | 同上 + 修法 | **true** | "config.json 默认即 true，发布时的默认值"；InferenceX 上用否"待定（#210 用 false）" |
| Infera v2 kit（`examples/sglang_pd_glm5.2_agentx`）与 #48047 A/B | v0.5.22 镜像 / 当前 main | true | v2 kit 的镜像（`deploy/docker/Dockerfile.sglang`）构建时**打了** `dsa_dp_sync` + `v0522/dsa_page_table_rows`；#48047 A/B **没打** |
| InferenceX #210 / #4629 | — | false | |

性能（只有两次单独对比，都在噪声内）：

| 测试 | 对比 | 结果 |
|---|---|---|
| 10-20 `mix-v0522-correctness`（v0.5.22，C48，同机 138） | false 33,098 → true 32,675 | −1.3%；两台机器同配置差 3.5%，噪声内；交互性 p90 +7%、TTFT p90 +15%，方向不一致 → 结论"无明确收益" |
| 10-12 `config-compare`（v0.5.21，C40） | e1c false 21,049 / e2d true 21,383 | +1.6%，但两次运行还差其它调试项；结论"C40 下无吞吐收益" |
| 最佳栈（E10 / llr35 / 53,666） | 未单独测 | 沿用 true |

参考：E10 decode trace 中 verify 的 indexer 每步 3.6 ms（一步 55.6 ms 的 6%），index_share 能省的只是其中一部分，理论上限几个百分点。

## 3. 当前 main 上为什么不崩（代码分析）

### 3.1 断言在哪、什么时候成立
`dsa_backend.py` `forward_decode`：
1. `topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])` —— 把 top-k 补齐到 q 的行数（补 -1）；top-k 行数多于 q 时直接 assert 失败，不会补。
2. `transform_index_page_table_decode(page_table=metadata.page_table_1, topk_indices=...)` —— 断言 `page_table.shape[0] == topk_indices.shape[0]`（`kernels/ops/attention/dsa/transform_index.py`）。
3. 再分发到 triton / tilelang / aiter 等实现（main 里 triton 分支在断言之后，同样会走到这里）。

所以**只要 `page_table_1` 行数 = q 行数**，补齐后 top-k 也等于 q 行数，断言必然成立。补丁修的是"page table 按请求、q/top-k 按 token，q 行数多于请求数"。

### 3.2 main 里三者的行数
- **进入 `forward_decode` 的只有 DECODE / IDLE**。target verify 与 draft extend（每请求多 token）走 `forward_extend`，其 page table 本来就按 token 建：
  replay 元数据里 `real_rows = bs * speculative_num_draft_tokens`，取 `decode_cuda_graph_metadata["page_table"][:real_rows]`。
- **MTP 草稿 decode 在 topk=1 时每请求每步 1 个 token**，q 行数 = 请求数：
  - eager：q = 真实请求数；`page_table_1` 按真实请求从 `req_to_token` 建，相等。
  - CUDA graph 回放：q = 图 batch（DP attention 下为填充后行数）；`page_table_1 = decode_cuda_graph_metadata["page_table"][:bs]` 用同一个图 batch，相等。
- **index share 打开时复用的 top-k 也是每请求一行**：
  - `eagle_worker_v2._init_dsa_index_share_state`：`index_share_for_mtp_iteration` 仅在 `self.topk == 1` 时生效，topk>1 整条复用路径关闭。
  - 种子在 draft-extend 抓取（上游 **#29787**，2026-07-07 合入，"Anchor GLM-5.2 MTP IndexShare topk on the draft-extend step"）：
    `dsa_seed_topk_select = cumsum(extend_seq_lens) - 1`（每请求最后一个被接受位置），
    `dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]` → **每请求一行**；
    后续草稿步经 `IndexTopKShareState`（`layers/attention/index_topk_share.py`）复用这份种子，行数 = 请求数 ≤ q，第 1 步只需补齐。
  - 前身 **#29654**（2026-06-29）修复了 index share 在 EAGLE MTP 里根本不生效的问题（复用的 top-k 挂到 `spec_info` 上）。

因此 `forward_decode` 中 **page table 行数 = q 行数 ≥ top-k 行数，补齐后三者相等**，断言成立；补丁处理的错位在 topk=1 时不会出现。

### 3.3 与 v0.5.22 的比较
v0.5.22 基础提交 `8dee3f6bdb` 的 `eagle_worker_v2.py` 已有完全相同的 `select_index` 选择（第 995–996、1170–1177 行），
`dsa_backend.py` 的 decode / verify page table 建法（`[:bs]` / `real_rows = bs * speculative_num_draft_tokens`）也与 main 相同。
所以**在 v0.5.22 + topk=1 上这个补丁大概率已用不上**；补丁 v0522 说明里的"STILL NEEDED ON v0.5.22"依据的是"断言和调用点仍在"，
没有核对实际行数，也没有去掉补丁复现。

### 3.4 补丁当初为什么需要（推断）
补丁的 `repeat_interleave(n_topk // n_rows)` 说明当时进入 decode 的 top-k 是"每请求 N 行"——把 verify / draft-extend 按 token 算出的整块
top-k 原样拿到 decode 复用。上游 #29654 / #29787 把这条路径改成每请求一行之后，这种错位就不再出现。
补丁最初在 AMD 自己的 v0.5.16–v0.5.18 镜像上发现，那批镜像是否已含 #29654 / #29787 未逐一核对，所以当时崩溃的确切原因未确认。

## 4. 实测
#48047 A/B 栈：当前 main（不打 `dsa_page_table_rows`）+ `index_share_for_mtp_iteration=true` + MTP（EAGLE topk=1）+ Triton DSA 后端 + DP attention decode。
门禁（GSM8K 200 题 0.975、0 乱码；长上下文；真实接受长度 3.69）与两次各 25–30 分钟的 decode 压测都没有触发该断言，与第 3 节一致。

## 5. 建议
1. InferenceX：只要 MTP 保持 topk=1（我们和 #4629 都是），`index_share=true` 不需要 `dsa_page_table_rows`，无需推上游；topk>1 时 index share 自动关闭，同样不涉及。
2. `index_share=true` 的性能收益尚无证据（两次单独对比都在噪声内），发布配方继续用 false 最稳；若想用 true（模型默认值，不算减少 FLOPs 的改动），先在最终栈上做一次只改这一项的 A/B。
3. Infera 仓库：把本结论补进补丁说明，并在 v0.5.22 之后的镜像构建里去掉该补丁或标为"仅旧版本"（待决定）。
