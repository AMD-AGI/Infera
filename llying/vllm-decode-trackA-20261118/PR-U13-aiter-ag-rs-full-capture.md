# [ROCm][Perf] Let DP MoE dispatch/combine use AITER custom AG/RS inside FULL CUDA-graph capture

Draft for vLLM upstream (candidate U13). Evidence: `README.zh-CN.md` §3 (track A, v3-240, 2026-11-18).

## Problem

`CudaCommunicator` creates an AITER `CustomAllreduce` for DP groups and routes `all_gatherv` / `reduce_scatterv`
(the DP MoE dispatch/combine of `AgRsAll2AllManager`) to AITER's custom all-gather / reduce-scatter when
`_can_use_aiter_ag_rs()` holds: `sizes is None`, `cudagraph_runtime_mode == FULL` and `batch_descriptor.uniform`.

With the GPU model runner (`vllm/v1/worker/gpu/`), that never holds:

- `CudaGraphManager.capture` captures a FULL graph by calling `forward_fn(CUDAGraphMode.NONE)` inside
  `torch.cuda.graph(...)`, so the forward context reports `cudagraph_runtime_mode=NONE` while the graph is captured;
- it passes no batch descriptor, and `set_forward_context` fills in `BatchDescriptor(num_tokens=...)` with
  `uniform=False`;
- replay runs no Python, so the decision made at capture is final.

So every decode step of a DP MoE model runs its dispatch/combine on RCCL's generic kernel. GLM-5.2 MXFP4, DP4 TP1, MTP3:
156 collectives per step (2 per MoE layer, MTP layer included), ~50–65 µs each, 7.8–10.2 ms of a ~53 ms step.
(A diagnostic build logged `mode NONE capturing=True` for every capture-time call.)

## Change

In `_can_use_aiter_ag_rs`, treat "runtime mode NONE while the current stream is capturing" as a FULL capture, and drop
the `uniform` requirement (`sizes is None` already guarantees equal shapes across the group):

```python
full_capture = (
    ctx.cudagraph_runtime_mode == CUDAGraphMode.NONE
    and torch.cuda.is_current_stream_capturing()
)
if ctx.cudagraph_runtime_mode != CUDAGraphMode.FULL and not full_capture:
    return False
return True
```

PIECEWISE stays on RCCL, as before. The same capture test is already used in
`vllm/models/deepseek_v32/attention.py` (`cudagraph_runtime_mode == FULL or torch.cuda.is_current_stream_capturing()`).
A cleaner fix is for the runner to put `FULL` (or a "capturing FULL" flag) in the forward context while capturing;
other capture-time checks of `cudagraph_runtime_mode == FULL` have the same problem (e.g. the decode top-k bound in
`rocm_aiter_mla_sparse.py`).

Optional (measured separately): gather the dispatch's tensors (hidden states, top-k weights, top-k ids) with AITER's
`custom_all_gatherv` — one launch and one rank barrier — instead of one `custom_all_gather` per tensor.

## Results (MI355X, GLM-5.2 MXFP4, DP4 TP1, MTP3, fp8 KV; 104 requests × 64K in × 4K out, decode-only window)

| | decode step p50 | comm per step (rank 0 trace) |
|---|---|---|
| RCCL (current) | 52.86–53.12 ms (6 runs) | 7.8–10.2 ms |
| AITER AG/RS in FULL graphs | **51.49 ms (−2.7%)** | 6.10 ms (AG 34.5 µs, RS 34.0 µs, small AGs 4.8 µs) |
| + gfx950 gluon indexer logits (U12), side by side with stock | **49.77 vs 53.12 ms (−6.3%)**; 64 req −5.3%, 32 req −5.3% | |

Correctness (GSM8K 200 / 16K long context / MTP acceptance): 196/200, 32/32 + 31/32, 3.50 (stock runs 195–199).
Batching the dispatch's three tensors into one AITER `custom_all_gatherv` launch did not help (48.0 µs vs 34.5 + 2 × 4.8).

## Related finding (separate issue)

`--enable-expert-parallel` with `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` (default with AITER) and the AG/RS
backend produces wrong output (GSM8K 1/200; 198/200 with the fusion off). `expert_map_manager` maps the fused shared
expert to a local slot on every EP rank, the AG/RS dispatch hands every rank all tokens, and the reduce-scatter sums
ep_size copies of the shared expert. The MoRI backend asserts against this combination; the AG/RS backend should too
(or compute the shared expert once per token).
