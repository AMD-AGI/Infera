# PR #47278: conflict risk and delegation handoff

Decision: escalate to the owner for a kernel agent. This is not a mechanical
rebase. No source change or push was made to the SGLang PR.

## Exact inputs and evidence

- PR head: `a5d8b3af95d7915910547073b1771825981ccb98`.
- Current upstream main checked: `73bf84e90f1cc14ccf65de37b57f4d3d8653ea85`.
- Merge base: `e33cd0c3a10474c7a0f6efeb69ce2cf4b7a55021`.
- Main kernel blob: `d4d52532aa3ac8f55db069c3f81e95755616c22c`;
  identical to the earlier November 10 preview. A fresh three-way merge again
  produced four conflicts in `python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py`.
- The earlier test-file preview merged cleanly. That is not numerical validation.
- Runtime evidence is retained at
  `runtime/ci-watch-20261103/interactive/mla-conflict-20261110/`.

## Why the conflicts affect correctness or serving performance

| Conflict | Upstream main | PR behavior to preserve | Risk from a superficial merge |
|---|---|---|---|
| KV loading | `_load_kv_group` loads 128-column groups; scaled storage has FP32 scales and a relocated BF16 tail | Tuned split-K uses `GROUP_W=256` for eligible decode shapes, custom cache loads and optional int64 page offsets | Copying the main loader directly changes operand widths/group mapping; retaining old loads ignores packed scales/tail layout. Wrong attention values, shape/compiler errors or pointer errors are possible. |
| Tile sizes | `_scaled_kv_block_k(kv, kv_scaled)` and a fixed 16-head tile | Configurable head tile and a matching padded partial-buffer contract | Dropping either side can alter memory layout, resource use or the tuned split plan. |
| Fused launch | New `KV_SCALED` argument | Controlled pipeline depth and conditional no-async-copy scope | Both must reach the actual launch. A default-false scaled flag can silently interpret packed KV incorrectly; removing lowering controls can change correctness/performance. |
| Split-K launch/reduce | New `KV_SCALED` argument | BF16 partials, `H_PADDED`, `GROUP_W`, cache controls, `DECOUPLE_DOTS`, `PARTIAL_DQ`, `LEAN_EPILOGUE` and matching reducers | Passing both sets of kwargs is insufficient: the load/compute/reduce layouts and optimization gates must agree. |

Concrete anchors: [main KV loaders](https://github.com/sgl-project/sglang/blob/73bf84e90f1cc14ccf65de37b57f4d3d8653ea85/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L100),
[PR grouped loads](https://github.com/sgl-project/sglang/blob/a5d8b3af95d7915910547073b1771825981ccb98/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L975),
[PR 256-wide dispatch](https://github.com/sgl-project/sglang/blob/a5d8b3af95d7915910547073b1771825981ccb98/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1636),
[PR partial/reduce gates](https://github.com/sgl-project/sglang/blob/a5d8b3af95d7915910547073b1771825981ccb98/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1692).

The PR already documents a measured Triton 3.7 interaction: decoupled dots plus
async copies corrupted heads 48–63 and were nondeterministic; its current scope
suppresses that combination. It also documents that the lean epilogue is only
validated with the matching partial layout. This is concrete evidence that
launch/lowering choices are not harmless text changes. It does not prove a new
merged candidate would fail; no merged candidate has been constructed or tested.

## Work for the receiving kernel agent

1. Decide whether to implement scaled-KV support in the tuned path or explicitly
   route scaled storage to an upstream-safe path. The latter can change serving
   performance and needs owner assessment; do not silently drop the optimization.
2. Reconcile 128-column scale groups with 256-column compute groups and retain
   correct tail offsets, int64 page safety, BF16 partials, DCP/LSE and matching reducers.
3. Verify ordinary/scaled FP8 and BF16, fused/split-K, tuned/fallback and partial
   layouts against a reference on the appropriate GPU. Exercise heads 48–63 and
   graph replay/repeated execution. CPU checks alone cannot qualify these changes.
4. Measure the selected deployment shapes and combined serving behavior before
   claiming that the rebased port preserves the previous e2e performance.

GPU work must use the approved machine-pool workflow. This audit allocated no
GPU, ran no application tests and made no claim of new correctness or speed.
