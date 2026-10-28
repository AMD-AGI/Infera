## Problem and fix

OffloaderV2 copies a CPU-offloaded parameter with `Tensor.to("cuda")`, producing a new tensor without the source parameter's `is_shuffled` attribute. Aiter MoE consumers use that flag to select the preshuffled weight layout. Preserve a true flag on the device copy; absent/false flags retain the existing behavior.

The implementation is scoped to OffloaderV2 `cpu` mode. Its `create_device_tensor` method has the same AST as the original deployed October 15 patch after normalizing the local variable name (`t` versus `output`). It adds no new kernel or stream behavior.

## Validation

- Archived MI355X GLM-5.2-MXFP4 full-model runs `ofs0c`, `ofg3c`, and `ofg2c` used offload groups 8, 3 and 2 with this attribute-preservation patch. Rechecked their raw `results.jsonl.gz` records: **23/23 passed in each run**. The ledger explicitly records simulated acceptance off for `ofs0c`; performance runs were recorded separately. There was no unpatched end-to-end control, so these are deployment/correctness observations, not an isolated performance comparison.
- At this PR's actual head `43dec26`, the real MI355X onload/forward regression passed: **1 test + 3 subtests**, PyTorch 2.9.1+ROCm 7.2. It covers real pinned CPU copies, alternate-stream onload, two forward cycles and host-parameter restoration for true/false/absent tags. [Execution evidence](https://github.com/sgl-project/sglang/pull/45484#issuecomment-6440222902).
- Current-main CPU regressions: **3 passed + 2 subtests**. Unpatched main: **2 expected failures, 1 control passed**. The tests exercise the actual forward wrapper as well as the copy boundary.
- Applicable local lint and upstream lint/title pass.

```bash
PYTHONPATH=python python -m pytest -q test/registered/unit/utils/test_offloader.py
PYTHONPATH=python python test/registered/model_loading/test_offloader_shuffled_weights.py
pre-commit run --files python/sglang/srt/utils/offloader.py test/registered/unit/utils/test_offloader.py test/registered/model_loading/test_offloader_shuffled_weights.py
```

## Review scope and remaining coverage

Ready for review based on the unchanged deployed behavior, current-main CPU regressions and hardware regression at this exact PR head. A further full-model rerun is not an author-side prerequisite for reviewing this narrow attribute-copy fix.

CUDA coverage is registered for CI and has not been executed locally. Maintainer-enabled CI and code review remain required; gated/skipped workflows are not passes. `shm_cpu` and `sharded_gpu` modes have a different lifecycle and are outside this fix. No throughput benefit or general stack-correctness claim is made: historical throughput used simulated acceptance, and unrelated PP-overlap/custom-all-reduce correctness issues remain separate.
