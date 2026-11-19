# PR U12：gfx950 上 DSA indexer decode 用 gluon paged-MQA logits kernel

> **11-19 15:05 查重结论：不开 PR。** 上游已有 open PR [#68596](https://github.com/vllm-project/vllm/pull/68596)
> （@RuibinCheung，11-19 06:11 UTC 开，非 draft）做同一件事：同样从 aiter `e7d2453` 的 preshuffle kernel 裁剪重调，文件名也相同
> （`rocm_gluon_paged_mqa_logits.py` 和同名测试），覆盖更广（indexer page 16 / 32 / 64、每请求至少 2 行（奇偶均可）、最多 256 请求 / 1,536 行，
> 不加环境变量开关）。按 vLLM `AGENTS.md` 的查重规定不能再开。用户 11-19 定：明天 review 后在 #68596 下留言，补我们的数据
> （DP4 MTP3 拉平行上的 kernel 计时、聚合 decode 每步 −2.7% 至 −3.9%、与 U13 叠加 −6.3%、门禁结果），并确认它在 ROCm 拉平 decode
> （next_n = 1、组大小从 metadata 来）下能生效——这是我们之前踩过的坑（forward 时取不到配置、组大小被缓存成 0）。下面的内容保留作素材。

- 目标：vllm-project/vllm main。分支（仓库外 `/home/liyingli/bench_agentx/upstream-20261119/vllm`）：`u12-gfx950-gluon-mqa-logits-main`
  （基于 main `f0c592bfd`，一个提交 `c63368efb`），`u12-gfx950-gluon-mqa-logits`（v0.34.0 回移版 `52d8c1318`，只给 P4D4 用）。
  11-19 按用户决定给两个提交加了 `Co-authored-by: yihou <yihou@amd.com>`（只改提交信息，代码树与测过的 `94ecc6635` / `43983d3d2` 相同，单测结论不变）。
- 只影响 ROCm：只在 gfx950 且显式打开 `VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1` 时生效（默认关）；改动在 ROCm 专用的
  `rocm_aiter_mla_sparse.py`、新文件 `rocm_gluon_paged_mqa_logits.py` 和 `envs.py` 的开关登记；新测试在非 ROCm / 非 gfx950 上整体跳过。不需要 CUDA 测试。
- 来源：kernel 移植自 sglang PR #47897（作者 yihou@amd.com，sgl-project/sglang@8a9f215c），后者改自 aiter 的
  `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`（aiter e7d2453）。kernel 文件保留 AMD 的 MIT 头并注明出处。
  PR 里的 kernel 代码与 A 线 / kf-port 测量用的覆盖层逐函数相同（11-19 按 AST 比对，只差注释、docstring 和换行）。
- 状态：代码 + 单测完成，main / v0.34.0 单测通过；端到端数据来自 v0.34.0（A 线），main 上的聚合 A/B（可选的 J2）未做；
  P4D4（J3）确认 decode 走了 gluon。**尚未提交到 GitHub。**
- 用户已定（11-19）：加 `Co-authored-by: yihou <yihou@amd.com>`（已加）；保留 `Co-authored-by: Cursor`。
  待确认：kernel 文件里 MIT 部分的许可是否需要再确认（PR-PLAN 原记"许可需确认"）。
- 提交前：上游 main 比 `f0c592bfd` 新 13 个提交，改了 `vllm/envs.py`（本 PR 也改），rebase 可能有小冲突；挪到新 main 后重跑 `main-u12`。

## 验证数据（都是单次，除非注明）

| 项 | 结果 | 出处 |
|---|---|---|
| 单测，main `f0c592bfd`（nightly `9c1d45c2` + 同步被测树的 Python 代码），MI355X | 4 个 ROCm sparse-MLA 测试文件：PR 126 通过 / 基线 108（多出的 18 个是新测试：16 个与 aiter 对比 + 1 个回落 + 1 个 CUDA graph 重放），无新增失败 | J1 `main-u12`（作业 298936，v3-057） |
| 单测，v0.34.0（回移版） | PR 72 / 基线 54 | J1 `v034-u12` |
| 数值 | 与 aiter 的 logits 在可见列上最大差 7.63e-6（logits 最大值 56–63）；新测试的门槛是 1e-5 × 行内最大值 | kf-port `results/ab2/paged_logits.txt`；测试强制 `VLLM_ROCM_USE_AITER=1` 并断言 aiter 已启用 |
| kernel 计时（每点 40 次取平均；上下文 40K–200K，MTP3 拉平行） | 4 请求（16 行）39.0 → 15.3 µs（2.54×）、8（32）96.1 → 39.9（2.41×）、16（64）163.4 → 69.3（2.36×）、26（104）247.9 → 110.8（2.24×）、32（128）324.9 → 139.1（2.34×） | kf-port `results/ab2/paged_logits.txt`（11-18，"aiter_flat" 即 vLLM 实际的拉平调用） |
| 聚合 decode A/B（v0.34.0 + 同一 kernel 与分派规则，DP4 TP1 MTP3，104 请求 × 64K 入 / 4K 出，纯 decode 窗口每步 p50） | 本 PR 50.89 / 51.45 / 51.04 ms（3 次，两组 GPU）对原版 52.86–52.98 ms（5 次，两组 GPU）：快 1.4–2.0 ms（2.7–3.9%）。trace：logits 每次 68.3 µs 对 138–146 µs，每 rank 每步 1.50 对 3.04–3.21 ms | A 线 README 2.3 / 2.4 / 2.5（v3-240） |
| 叠加 U13（同时跑，各一次） | 49.77 对 53.12 ms（−6.3%）；64 请求 41.29 对 43.62（−5.3%）；32 请求 34.45 对 36.38（−5.3%） | A 线 README 0 / 3.2（r8） |
| 门禁（打开本路径） | GSM8K 196/200、16K 长上下文 32/32 + 31/32、MTP 接受长度 3.56；同时跑的原版 197/200、32/32 + 32/32、3.44（原版 A/A 195–199）；r6 / r7 197/200 | A 线 README 2.2 / 2.5 |
| P4D4 3600 s（v0.34.0 回移版 + U13 + U26） | decode 日志有 "Using the gfx950 gluon paged-MQA logits kernel for decode."；43,269 tok/s/GPU，无对照，吞吐不归因于本 PR | J3，`results/pr-298999/pr-checks.txt` |
| P4D4，main + 本 PR（+ U13 / U26） | 排队中 | J3b（作业 299112） |

- 更正（11-19）：J1 里第一次对比失败（差 8.3e-4、最大 1.24e-3）的对照不是 aiter，而是 vLLM 的 **PyTorch** 参考回退
  `fp8_paged_mqa_logits_torch`（日志 "AITER paged-MQA logits is unavailable; using the PyTorch fallback"，原因是测试容器没设
  `VLLM_ROCM_USE_AITER=1`），HANDOFF / PR-PLAN 之前写成了"Triton 回退"，已一并更正。修好后的测试对 aiter，结论以 `main-u12` 为准。

## PR description (English)

**Title:** [ROCm][Perf] DSA indexer: gfx950 gluon paged MQA logits for decode / MTP-verify rows

### Problem

On gfx950 the DSA indexer's decode scoring, `rocm_fp8_paged_mqa_logits`, runs aiter's `deepgemm_fp8_paged_mqa_logits`.
With MTP, ROCm decode is flattened (`indexer._use_flattening`): one query per verify row, with per-row lengths and
block-table rows. For GLM-5.2 (32 indexer heads x 128, fp8, 64-token preshuffled pages) at DP4 / MTP3 on MI355X this
call takes 138-146 us and runs 22 times per decode step, 3.0-3.2 ms of a ~53 ms step on every rank.

### Change

An opt-in path, `VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1` (registered in `vllm/envs.py`, default off), scores those rows
with a re-tuned gluon kernel (new file `vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py`), ported from
sgl-project/sglang#47897, which derives from aiter's `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`:

- the KV split is decoupled from the occupancy hint, and program ids are swizzled so the pieces of one row share an
  XCD (L2);
- the MTP verify rows of one request form one longest-first dispatch group;
- sibling rows with identical page tables are scored by one program (checked on the device, so it is exact for any
  input);
- gfx950's full-rate scaled fp8 MFMA.

The group size is a request's verify rows. `rocm_aiter_sparse_attn_indexer` passes it from the indexer decode metadata
(`write_max_decode_len` when `decode_is_uniform`, else 0) as the new `flat_group_rows` argument, so it is known where the
batch is built; no config lookup happens at forward time. For a `[B, next_n]` caller the wrapper expands per-row
lengths and block-table rows (`repeat_interleave`, no host sync, CUDA-graph capturable). The path serves 1-128 rows on
the 64-token preshuffled indexer cache with an even group of 2-8 rows; every other case, including non-uniform steps,
keeps aiter.

### Tests

New `tests/kernels/attention/test_rocm_gluon_paged_mqa_logits.py` (ROCm + gfx950 + aiter only), with
`VLLM_ROCM_USE_AITER=1` forced so the reference is aiter's kernel (without it the reference would be the PyTorch
fallback):

- `test_gluon_matches_aiter`: 1, 4, 13 and 32 requests, for the `[B, next_n]` form (per-request and per-row lengths)
  and the flattened form (1-D and `[rows, 1]` lengths); the logits on every visible column match aiter within 1e-5 of
  the maximum.
- `test_unsupported_steps_keep_aiter`: 160 rows, a non-uniform step (group 0) and the switch off all keep aiter.
- `test_gluon_cuda_graph_replay`: the flattened form captured once and replayed with new queries, weights and lengths
  matches aiter.
- MI355X: the new file and the existing ROCm sparse-MLA kernel tests (`test_rocm_aiter_mla_sparse_index_remap.py`,
  `test_rocm_aiter_mla_sparse_metadata_sync.py`, `test_rocm_aiter_candidate_mask.py`): 126 passed, 108 on main (the
  difference is the new tests), no new failures.

### Validation (MI355X)

Kernel time per call, GLM-5.2 indexer shapes, MTP3 flattened rows, contexts 40K-200K (mean of 40 iterations):

| requests (rows) | aiter, flattened (current) us | this PR us | speed-up |
|---|---|---|---|
| 4 (16) | 39.0 | 15.3 | 2.54x |
| 8 (32) | 96.1 | 39.9 | 2.41x |
| 16 (64) | 163.4 | 69.3 | 2.36x |
| 26 (104) | 247.9 | 110.8 | 2.24x |
| 32 (128) | 324.9 | 139.1 | 2.34x |

Max |diff| against aiter: 7.63e-6 at logits of magnitude 56-63.

End to end: GLM-5.2-MXFP4, DP4 TP1, MTP3, fp8 KV, one 8-GPU node split into two 4-GPU servers run side by side,
104 requests x 64K input x 4K output, decode step p50 over the window where all requests are decoding:

| | decode step p50 | indexer logits per call | per step per rank |
|---|---|---|---|
| aiter (current), 5 runs | 52.86-52.98 ms | 138-146 us | 3.04-3.21 ms |
| this PR, 3 runs | 50.89 / 51.45 / 51.04 ms (-2.7% to -3.9%) | 68.3 us | 1.50 ms |

With the AITER AG/RS capture fix (separate PR) on top, side by side with the stock server (one run each): 49.77 vs
53.12 ms at 104 requests (-6.3%), 41.29 vs 43.62 ms at 64, 34.45 vs 36.38 ms at 32.

Accuracy with the path on: GSM8K 196/200, 16K long-context needle 32/32 and long-prefix GSM8K 31/32, MTP acceptance
length 3.56; the stock server run alongside scored 197/200, 32/32, 32/32, 3.44 (stock runs range 195-199 on GSM8K).

The end-to-end runs used a v0.34.0 build with the same kernel and dispatch rule; the unit tests ran on main.

### Notes

- The kernel file keeps the MIT header (AMD) of the code it adapts and names both sources (aiter, sglang#47897) in
  its header and docstring.
- Opt-in for now: it has been validated on GLM-5.2 decode shapes only. The gain grows when DP ranks have unequal
  context lengths, since the rank with the longest contexts sets every step.
- An earlier version took the group size from `get_current_vllm_config()` at forward time. Workers have no config
  context there, so it raised, the fallback returned 0 and `functools.cache` kept it: the path silently never ran.
  Passing the size from the metadata removes that failure mode.
