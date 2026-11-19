# PR U13：FULL CUDA graph 捕获时放行 AITER 自定义 AG/RS（DP MoE dispatch / combine）

- 目标：vllm-project/vllm main。分支（仓库外 `/home/liyingli/bench_agentx/upstream-20261119/vllm`）：`u13-aiter-agrs-full-capture-main`
  （基于 main `f0c592bfd`，一个提交 `11bbd3a3e`），`u13-aiter-agrs-full-capture`（v0.34.0 回移版，只给 P4D4 用）。
- 只影响 ROCm：改在共用的 `CudaCommunicator._can_use_aiter_ag_rs`，但 `use_aiter_ag_rs` 只在 DP 组 + `current_platform.is_rocm()` +
  AITER 自定义 allreduce 开启时才为真，CUDA 在函数第一行就返回 False。不需要 CUDA 测试（判断逻辑由 CPU 上的 mock 单测覆盖）。
- 前提在 main 上仍成立（11-19 核对 `f0c592bfd` 的 `vllm/v1/worker/gpu/cudagraph_utils.py`）：FULL 捕获在 `torch.cuda.graph(...)` 里调用
  `forward_fn(CUDAGraphMode.NONE)`（第 534–540 行），只有 PIECEWISE 才构造 `BatchDescriptor`（第 821–826 行）。
- PR 改动与 A 线测量用的 `overlay-comm/cuda_communicator.py.diff` 逐行相同。
- 状态：代码 + 单测完成，main / v0.34.0 单测通过；端到端数据来自 v0.34.0（A 线），main 上的聚合 A/B（可选的 J2）未做；
  P4D4（J3）确认走了 AITER AG/RS。**尚未提交到 GitHub。**
- 提交前：上游 main 比 `f0c592bfd` 新 13 个提交，未碰本 PR 的文件；挪到新 main 后重跑 `main-u13`。`Co-authored-by: Cursor` 保留（用户 11-19 定）。

## 验证数据（都是单次，除非注明）

| 项 | 结果 | 出处 |
|---|---|---|
| 单测，main `f0c592bfd`（nightly `9c1d45c2` + 同步被测树的 Python 代码，容器里补装 `ray`） | `tests/distributed/test_comm_ops.py -k aiter`：PR 16 通过 / 基线 10（多出的 6 个是新测试的参数组合），无新增失败 | J1 `main-u13`（作业 298936，v3-057） |
| 单测，v0.34.0（回移版） | PR 16 / 基线 10 | J1 `v034-u13b` |
| 聚合 decode A/B（v0.34.0 + 同一改动，DP4 TP1 MTP3，104 请求 × 64K 入 / 4K 出，纯 decode 窗口每步 p50） | 本 PR 51.49 ms（r6）对原版 52.86–53.12 ms（6 次）：−1.4 ms（−2.7%） | A 线 README 3.2（v3-240） |
| trace（dp0，每步） | 通信 6.10 ms 对原版 7.8–10.2 ms；hidden states AG 34.5 µs、RS 34.0 µs，替代每次约 50–65 µs 的 RCCL generic kernel；每步 156 次集合通信（每个 MoE 层 2 次，含 MTP 层） | A 线 README 3.1 / 3.2 |
| 叠加 U12（同时跑，各一次） | 49.77 对 53.12 ms（−6.3%）；64 请求 41.29 对 43.62（−5.3%）；32 请求 34.45 对 36.38（−5.3%）；r7 的同类组合 49.66 ms | A 线 README 0 / 3.2（r7、r8） |
| 门禁（r6） | GSM8K 196/200、16K 长上下文 32/32 + 31/32、MTP 接受长度 3.50（原版 195–199） | A 线 `PR-U13-…md` |
| 诊断 | 未修时诊断版在每次捕获期调用都打印 `mode NONE capturing=True`（r4）；修后日志 `Using AITER custom AG/RS for group dp:0.` | A 线 README 3.1 / 3.2 |
| P4D4 3600 s（v0.34.0 回移版 + U12 + U26） | decode 日志有 "Using AITER custom AG/RS"；无崩溃；43,269 tok/s/GPU，无对照，吞吐不归因于本 PR | J3，`results/pr-298999/pr-checks.txt` |
| P4D4，main + 本 PR（+ U12 / U26） | 排队中 | J3b（作业 299112） |

- 不建议的变体（A 线已测，PR 里只在 Notes 提一句）：把 dispatch 的三个张量用 aiter `custom_all_gatherv` 一次发出，48.0 µs 对分开的 34.5 + 2 × 4.8 µs，
  没有收益（单次）。
- EP + 融合共享专家 + AG/RS 结果错误（GSM8K 1/200）是另一个问题（U17），单独提 issue / PR，不放进本 PR。

## PR description (English)

**Title:** [ROCm][Perf] Let DP MoE dispatch/combine use AITER custom AG/RS inside FULL CUDA-graph capture

### Problem

On ROCm, `CudaCommunicator` creates an AITER `CustomAllreduce` for DP groups and routes `all_gatherv` /
`reduce_scatterv` (the DP MoE dispatch/combine of `AgRsAll2AllManager`) to AITER's custom all-gather / reduce-scatter
when `_can_use_aiter_ag_rs()` holds: `sizes is None`, `cudagraph_runtime_mode == FULL` and `batch_descriptor.uniform`.

With the GPU model runner (`vllm/v1/worker/gpu/`) that never holds:

- `CudaGraphManager` captures a FULL graph by calling `forward_fn(CUDAGraphMode.NONE)` inside `torch.cuda.graph(...)`
  (`cudagraph_utils.py`), so the forward context reports `cudagraph_runtime_mode=NONE` while the graph is captured;
- `forward_fn` builds a `BatchDescriptor` only for PIECEWISE, and `set_forward_context` fills in
  `BatchDescriptor(num_tokens=...)`, whose `uniform` is False;
- a replay runs no Python, so the decision made at capture is final.

So every decode step of a DP MoE model runs its dispatch/combine on RCCL's generic kernel. GLM-5.2-MXFP4, DP4 TP1,
MTP3 on MI355X: 156 collectives per step (2 per MoE layer, MTP layer included), ~50-65 us each, 7.8-10.2 ms of a
~53 ms step.

### Change

In `_can_use_aiter_ag_rs`, treat "runtime mode NONE while the current stream is capturing" as a FULL capture, and drop
the `uniform` requirement (`sizes is None` already means every rank of the group has the same token count):

```python
full_capture = (
    ctx.cudagraph_runtime_mode == CUDAGraphMode.NONE
    and torch.cuda.is_current_stream_capturing()
)
if ctx.cudagraph_runtime_mode != CUDAGraphMode.FULL and not full_capture:
    return False
logger.info_once("Using AITER custom AG/RS for group %s.", self.unique_name)
return True
```

PIECEWISE (captured with mode PIECEWISE) and eager steps stay on RCCL, as before. The same capture test is already used
in `vllm/models/deepseek_v32/attention.py` (`cudagraph_runtime_mode == FULL or torch.cuda.is_current_stream_capturing()`).
CUDA is unaffected: `use_aiter_ag_rs` is only set on ROCm with AITER custom all-reduce enabled, and the method returns
False before reaching this check otherwise.

### Tests

- `tests/distributed/test_comm_ops.py::test_aiter_ag_rs_gate` (CPU, mocked forward context and capture state): FULL
  -> AITER; NONE while capturing -> AITER; NONE not capturing, PIECEWISE while capturing, unequal `sizes`, and no forward
  context -> RCCL.
- MI355X, `tests/distributed/test_comm_ops.py -k aiter`: 16 passed, 10 on main (the difference is the new cases), no new
  failures.

### Validation (MI355X)

GLM-5.2-MXFP4, DP4 TP1, MTP3, fp8 KV, one 8-GPU node split into two 4-GPU servers, 104 requests x 64K input x 4K output,
decode step p50 over the window where all requests are decoding:

| | decode step p50 | communication per step (rank 0 trace) |
|---|---|---|
| RCCL (current), 6 runs | 52.86-53.12 ms | 7.8-10.2 ms |
| this PR | 51.49 ms (-2.7%) | 6.10 ms (AG 34.5 us, RS 34.0 us, top-k AGs 4.8 us) |
| this PR + gfx950 gluon indexer logits (separate PR), side by side with stock | 49.77 vs 53.12 ms (-6.3%); 64 requests -5.3%, 32 requests -5.3% | |

Accuracy: GSM8K 196/200, 16K long-context needle 32/32 and long-prefix GSM8K 31/32, MTP acceptance length 3.50 (stock
runs: 195-199 on GSM8K).

The end-to-end runs used a v0.34.0 build with the identical change (v0.34.0 has the same capture path); the unit tests
ran on main.

### Notes

- A cleaner fix is for the runner to put FULL (or a "capturing FULL" flag) in the forward context while it captures.
  Other capture-time checks of `cudagraph_runtime_mode == FULL` have the same blind spot, e.g. the decode top-k bound in
  `rocm_aiter_mla_sparse.py`. This PR keeps the change local to the communicator; happy to switch if that is preferred.
- Gathering the dispatch's three tensors with one AITER `custom_all_gatherv` launch did not help (48.0 us vs
  34.5 + 2 x 4.8 us), so this PR keeps one all-gather per tensor.
