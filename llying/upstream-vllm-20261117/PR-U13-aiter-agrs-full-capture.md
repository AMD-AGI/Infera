# PR U13：FULL CUDA graph 捕获时放行 AITER 自定义 AG/RS（DP MoE dispatch / combine）

- 目标：vllm-project/vllm main。分支（仓库外 `/home/liyingli/bench_agentx/upstream-20261119/vllm`，fork `llying-001/vllm`）：
  `u13-aiter-agrs-full-capture-main`，11-19 15:05 rebase 到 main `8828834fb`（一个提交 `f07e10429`，无冲突）；rebase 前测过的版本留在
  `u13-aiter-agrs-full-capture-main-f0c592bfd`（`11bbd3a3e`）。`u13-aiter-agrs-full-capture` 是 v0.34.0 回移版，只给 P4D4 用。
- 提交方式（用户 11-19 定）：先在 fork 里开 PR，用户明天在那里 review，review 后再开到上游。
  **fork PR：[llying-001/vllm#1](https://github.com/llying-001/vllm/pull/1)**（11-19 15:40 开，base = fork 的 main，已同步到上游 `9a2e60358`）。
- 只影响 ROCm：改在共用的 `CudaCommunicator._can_use_aiter_ag_rs`，但 `use_aiter_ag_rs` 只在 DP 组 + `current_platform.is_rocm()` +
  AITER 自定义 allreduce 开启时才为真，CUDA 在函数第一行就返回 False。不需要 CUDA 测试。上游有 CUDA 上与 #48247 有关的回退 issue
  [#59765](https://github.com/vllm-project/vllm/issues/59765)，所以正文里把"不影响 CUDA"写得更醒目。
- 前提在 main 上仍成立（`8828834fb` 的 `vllm/v1/worker/gpu/cudagraph_utils.py`）：FULL 捕获在 `torch.cuda.graph(...)` 里调用
  `forward_fn(CUDAGraphMode.NONE)`（第 536–540 行），只有 PIECEWISE 才构造 `BatchDescriptor`（第 822–826 行）。
- 查重（11-19）：上游 open PR 里没有改 `_can_use_aiter_ag_rs` 的；[#66775](https://github.com/vllm-project/vllm/pull/66775)（QuickReduce
  reduce-scatterv，draft）只在自己的路径里判断捕获状态，[#64428](https://github.com/vllm-project/vllm/pull/64428) 不碰这个判断。
- PR 改动与 A 线测量用的 `overlay-comm/cuda_communicator.py.diff` 逐行相同。
- 状态：代码 + 单测完成；新 main 上单测与 pre-commit 都通过（J4，`precommit-u13u19` 里 17 个适用的 hook 全过）；端到端数据：v0.34.0（A 线）和
  main 上的聚合 A/B（J8，11-20）；P4D4（J3）确认走了 AITER AG/RS。
- 开到上游前要确认：正文最后一条 "the submitter reviewed every changed line and the measurements" 属实（vLLM `AGENTS.md` 要求提交人审过每一行）。

## 验证数据（都是单次，除非注明）

| 项 | 结果 | 出处 |
|---|---|---|
| 单测，main `8828834fb` + PR `f07e10429`（nightly `9c1d45c2` + 同步被测树的 Python 代码，补装 `ray`） | `tests/distributed/test_comm_ops.py -k aiter`：PR `16 passed, 43 deselected, 17 warnings in 8.13s`，main `10 passed … in 5.26s`；新增 6 个用例合计 2.41 s | J4 `main2-u13`（作业 299174，v3-221） |
| 新测试跑在 main 的代码上 | `test_aiter_ag_rs_gate` 6 个用例里 `[FULL-False-None-True]`、`[NONE-True-None-True]` 失败，其余 4 个通过（`2 failed, 4 passed, 53 deselected, 17 warnings in 3.93s`）：前者是 main 要求 `uniform` 的 batch descriptor，后者是 FULL 捕获 | J4 `newtests-on-base` 首轮的 `u13.out`（该步骤首轮因预期集合漏写 FULL 用例、U19 部分 `-k` 引号问题判了失败，已修正重跑） |
| 单测，main `f0c592bfd`（rebase 前） | PR 16 / 基线 10 | J1 `main-u13`（作业 298936） |
| 单测，v0.34.0（回移版） | PR 16 / 基线 10 | J1 `v034-u13b` |
| **聚合 decode A/B，main `8828834fb`**（nightly `9c1d45c2` + 同步 Python 代码；GPU 0–3 main + U13 `f07e10429`，GPU 4–7 main，同时跑；DP4 TP1 MTP3，104 请求 × 64K 入 / 4K 出，纯 decode 窗口每步 p50，同一对服务连跑两次） | main + U13 **51.76 / 51.21 ms** 对 main **52.83 / 52.76 ms**：快 1.0–1.6 ms（−2.0% 至 −3.1%）；只有 U13 那组日志里有 `Using AITER custom AG/RS`（1 对 0）；门禁 main + U13 GSM8K 197/200、长上下文 16/16 + 16/16、MTP 接受长度 3.51，main 196/200、16/16 + 15/16、3.476 | J8 `e2e-u13-ab`（作业 300490，v3-222，11-20） |
| 聚合 decode A/B（v0.34.0 + 同一改动，DP4 TP1 MTP3，104 请求 × 64K 入 / 4K 出，纯 decode 窗口每步 p50） | 本 PR 51.49 ms（r6）对原版 52.86–53.12 ms（6 次）：−1.4 ms（−2.7%） | A 线 README 3.2（v3-240），`results/a2-r6/summary.txt` |
| trace（dp0，每步） | 通信 6.10 ms 对原版 7.8–10.2 ms；hidden states AG 34.5 µs、RS 34.0 µs，替代每次约 50–65 µs 的 RCCL generic kernel；每步 156 次集合通信 | A 线 README 3.1 / 3.2 |
| 叠加 U12（同时跑，各一次） | 49.77 对 53.12 ms（−6.3%）；64 请求 41.29 对 43.62；32 请求 34.45 对 36.38 | A 线 `results/a2-r8/summary.txt` |
| 门禁（r6） | smoke 5/5、GSM8K 196/200、16K 长上下文 needle 32/32 + gsm8k 31/32、MTP 接受长度 3.495（原版 195–199） | A 线 `results/a2-r6/gate-agrs5.out` |
| P4D4 3600 s（v0.34.0 回移版 + U12 + U26） | decode 日志有 "Using AITER custom AG/RS"；无崩溃；吞吐无对照，不归因于本 PR | J3，`results/pr-298999/pr-checks.txt` |
| P4D4，main + 本 PR（+ U12 / U26） | 排队中 | J3b（作业 299112） |

- 不建议的变体（A 线已测）：dispatch 的三个张量用 aiter `custom_all_gatherv` 一次发出，48.0 µs 对分开的 34.5 + 2 × 4.8 µs，没有收益（单次）。
- EP + 融合共享专家 + AG/RS 结果错误（GSM8K 1/200）是另一个问题（U17），不放进本 PR。

## PR description (English, vLLM template)

**Title:** [ROCm][Perf] Let DP MoE dispatch/combine use AITER custom AG/RS inside FULL CUDA-graph capture

<!-- markdownlint-disable -->

## Overview

On ROCm, the AITER custom all-gather / reduce-scatter for DP MoE dispatch/combine (#48247) never runs with Model Runner
V2: MRV2 captures FULL graphs with `cudagraph_runtime_mode=NONE` and no batch descriptor, so `_can_use_aiter_ag_rs()`
always falls back to RCCL. This PR treats "NONE while the current stream is capturing" as a FULL capture.

## Claims

- ROCm + AITER, DP MoE models: FULL decode graphs now capture AITER AG/RS instead of RCCL's generic kernel (156
  collectives per decode step for GLM-5.2 with MTP).
- GLM-5.2-MXFP4, DP4 TP1, MTP3 on MI355X, 104 requests x 64K input x 4K output: decode step p50 **51.49 ms vs
  52.86-53.12 ms (-2.7%)**, communication per step 6.10 ms vs 7.8-10.2 ms; accuracy unchanged.
- No change on CUDA (`use_aiter_ag_rs` is only set on ROCm with AITER custom all-reduce), for PIECEWISE graphs, or for
  eager steps.

## Validation

**Unit tests** on MI355X, main `8828834fb` + this PR (`f07e10429`), in `vllm/vllm-openai-rocm:nightly-9c1d45c2` with the
tree's Python code installed over the image:

```bash
pytest -v tests/distributed/test_comm_ops.py -k aiter
```

```
PR:   ================ 16 passed, 43 deselected, 17 warnings in 8.13s ================
main: ================ 10 passed, 43 deselected, 17 warnings in 5.26s ================
```

The 6 new `test_aiter_ag_rs_gate` cases take 2.41 s in total. Run against main's code, 2 of them fail, as expected:
`[NONE-True-None-True]` (FULL capture) and `[FULL-False-None-True]` (main also requires a uniform batch descriptor):

```
FAILED tests/distributed/test_comm_ops.py::test_aiter_ag_rs_gate[FULL-False-None-True]
FAILED tests/distributed/test_comm_ops.py::test_aiter_ag_rs_gate[NONE-True-None-True]
=========== 2 failed, 4 passed, 53 deselected, 17 warnings in 3.93s ============
```

`pre-commit run --files tests/distributed/test_comm_ops.py vllm/distributed/device_communicators/cuda_communicator.py`
passes (ruff, typos, mypy, test tethering, docstring style and the other applicable hooks).

**End to end.** GLM-5.2-MXFP4 on one 8x MI355X node split into two 4-GPU servers run side by side. These runs used
vLLM v0.34.0 with this exact patch applied; the MRV2 capture path it fixes is the same on main
(`vllm/v1/worker/gpu/cudagraph_utils.py`, FULL capture at L536-L540, `BatchDescriptor` only for PIECEWISE at L822-L826).

<details>
<summary>Server, benchmark and accuracy commands</summary>

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
vllm serve GLM-5.2-MXFP4 --served-model-name glm5.2-mxfp4 --trust-remote-code \
  --data-parallel-size 4 --tensor-parallel-size 1 --kv-cache-dtype fp8_e4m3 --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --block-size 64 --reasoning-parser glm45 --tool-call-parser glm47 \
  --enable-auto-tool-choice

vllm bench serve --backend vllm --endpoint /v1/completions --model glm5.2-mxfp4 --tokenizer GLM-5.2-MXFP4 \
  --trust-remote-code --dataset-name random --random-input-len 65536 --random-output-len 4096 \
  --random-range-ratio 0 --num-prompts 104 --max-concurrency 104 --ignore-eos --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el --save-result --save-detailed
```

The decode step is the p50 of the inter-token gaps (one MTP step each) in the window where all 104 requests are
decoding: from the last request's first token to the first request's end, rebuilt from the per-request TTFT and ITLs
of `--save-detailed`, minus a 15 s torch-profiler window. Accuracy: GSM8K (200 questions), and 32 needle + 32
long-prefix GSM8K prompts at 16K tokens, through the OpenAI API.

</details>

| | decode step p50 | communication per step (rank 0 trace) |
|---|---|---|
| main (RCCL), 6 runs | 52.86-53.12 ms | 7.8-10.2 ms |
| this PR, 1 run | 51.49 ms (-2.7%) | 6.10 ms (AG 34.5 us, RS 34.0 us, top-k AGs 4.8 us) |

<details>
<summary>Raw output (this PR)</summary>

```
decode window: {"n_req": 104, "window_s": 55.4, "n_itl": 81746, "step_p50_ms": 51.49, "step_mean_ms": 51.55,
  "step_p90_ms": 51.97, "step_p99_ms": 53.07, "accept_len": 2.01}
accuracy: smoke 5/5; gsm8k n=200 correct=196/200 (0.980) garbled=0; longctx needle: n=32 correct=32;
  gsm8k: n=32 correct=31; MTP accept_len 3.495
log: INFO [cuda_communicator.py] Using AITER custom AG/RS for group dp:0.
```

Stock runs scored 195-199/200 on the same GSM8K set. Random-token prompts give a low MTP acceptance (~2); it is the
same on both sides and does not change the per-step kernel time.

</details>

With a gfx950 gluon kernel for the DSA indexer logits on top (a port of the same aiter-derived kernel that #68596
proposes), side by side with stock, one run each: 49.77 vs 53.12 ms at 104 requests (-6.3%), 41.29 vs 43.62 ms at 64,
34.45 vs 36.38 ms at 32.

## Details

- **Root cause.** `_can_use_aiter_ag_rs()` requires `cudagraph_runtime_mode == FULL` and `batch_descriptor.uniform`.
  MRV2's `CudaGraphManager` captures a FULL graph by calling `forward_fn(CUDAGraphMode.NONE)` inside
  `torch.cuda.graph(...)`; `forward_fn` builds a `BatchDescriptor` only for PIECEWISE, and `set_forward_context` fills
  in `BatchDescriptor(num_tokens=...)` with `uniform=False`. A replay runs no Python, so the RCCL choice made at capture
  is final. A diagnostic build logged `mode NONE capturing=True` for every capture-time call.
- **Fix.** Accept NONE while `torch.cuda.is_current_stream_capturing()`, the same test
  `vllm/models/deepseek_v32/attention.py` already uses, and drop the `uniform` requirement: `sizes is None` already
  means every rank of the DP group has the same token count. PIECEWISE (captured with mode PIECEWISE) and eager steps
  stay on RCCL. `is_current_stream_capturing()` is a host-side query; no GPU sync is added.
- **Alternative.** The runner could put FULL (or a "capturing FULL" flag) in the forward context while it captures.
  Other capture-time checks of `cudagraph_runtime_mode == FULL` have the same blind spot (e.g. the decode top-k bound in
  `rocm_aiter_mla_sparse.py`); this PR keeps the change local to the communicator.
- **Not pursued.** Gathering the dispatch's three tensors with one AITER `custom_all_gatherv` launch was slower (48.0 us
  vs 34.5 + 2 x 4.8 us).
- **CI.** `tests/distributed/test_comm_ops.py` runs on AMD CI. The new test mocks the forward context and the capture
  state, because the real path needs a DP group of ROCm GPUs with AITER; the end-to-end runs above cover it.
- **Duplicates.** No open PR changes this check; #66775 and #64428 add QuickReduce paths for the DP MoE combine and do
  not touch `_can_use_aiter_ag_rs`.
- **AI assistance** was used to write this change, its tests and this description; the submitter reviewed every changed
  line and the measurements.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
