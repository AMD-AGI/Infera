## Motivation

Preserve `is_shuffled` when OffloaderV2 copies a CPU-offloaded parameter to the device. Aiter MoE kernels use this attribute to select the preshuffled weight layout, but `Tensor.to("cuda")` returns a fresh tensor without the source parameter's Python attributes. The weight bytes remain shuffled while the onloaded tensor loses the tag.

`functional_call` substitutes these device tensors for the layer parameters during forward. The affected checks include the `is_shuffled` guards in the MXFP4, FP8, and Quark MoE implementations.

## Modifications

- `_CpuParamOffloader.create_device_tensor` carries `is_shuffled=True` to the device copy. Parameters with an absent or false attribute retain their existing behavior.
- A registered CPU regression checks the tag and tensor contents, including absent/false controls. The test uses a fresh CPU copy to model the H2D copy and disables pinning; it does not allocate an accelerator tensor.
- A CPU forward regression verifies that the real `functional_call` wrapper exposes the tag to the consumer and restores the original host parameter across repeated forwards.
- An AMD/CUDA-registered GPU regression covers real pinned-memory offload, asynchronous onload, forward values/metadata, and host-parameter restoration for true/false/absent tags. No accelerator or stream operation is mocked in that test.
- The tests use `unittest.TestCase` directly to avoid importing the unrelated serving/model test harness.

This change covers OffloaderV2 `cpu` mode. `shm_cpu` and `sharded_gpu` have a different parameter-to-meta lifecycle and require separate investigation.

## Accuracy Tests

Revalidated on upstream main `449617bc94a5d4ae23ad62d0da2f842e711ca983`, using Python 3.13.13 and PyTorch 2.12.1+cpu, then repeated with the current-main PyTorch pin, 2.14.1+cpu, with identical results:

```bash
PYTHONPATH=python python -m pytest -q test/registered/unit/utils/test_offloader.py
pre-commit run --files python/sglang/srt/utils/offloader.py test/registered/unit/utils/test_offloader.py test/registered/model_loading/test_offloader_shuffled_weights.py
```

- Patched production code: **3 passed, 2 subtests passed**.
- The same test with `offloader.py` restored to unpatched main: **2 failed, 1 passed, 2 subtests passed**. Both failures detect the missing `is_shuffled` attribute, not collection or environment setup.
- All applicable repository pre-commit hooks passed.
- GPU regression command: `PYTHONPATH=python python test/registered/model_loading/test_offloader_shuffled_weights.py`. This was collected successfully but **skipped on the CPU host**; the isolated GPU requests did not obtain a node during this validation session. It is not a GPU pass. The registered AMD/CUDA CI test must execute before claiming hardware validation.

Historical context: the equivalent patch was deployed in GLM-5.2-MXFP4 prefill on MI355X. Runs `ofs0c`, `ofg3c`, and `ofg2c` (OffloaderV2 groups 8, 3, and 2 respectively) each passed the archived 23-case smoke/needle suite. All offload runs carried the patch; **there was no unpatched end-to-end control**, so those results do not isolate this patch's model-output effect. Source: the October 15 OffloaderV2 experiment archive, with the run IDs above. No new GPU or full-model validation of this rebased commit is claimed.

## Speed Tests and Profiling

No performance claim. Historical throughput used simulated MTP acceptance and bundled other changes; it is not evidence for this attribute-copy fix. A separate October 27 llr35 bisect implicated PP communication overlap and decode custom all-reduce in output corruption; fault-free or fast stack runs do not establish general model correctness.

## Checklist

- [x] Applied repository formatting and lint checks.
- [x] Added a registered CPU regression and verified failure on unpatched main.
- [x] Kept the change scoped to CPU parameter offload.
- [x] Reported validation limits; no user-facing configuration or documentation change.


## Draft blockers

Real AMD/CUDA onload regression and maintainer-enabled CI are still pending. Local CPU regression and lint pass; no hardware result is claimed.

Kept as draft until these validation/review gates are satisfied.


## GPU follow-up (October 27)

At head `43dec26`, the registered real MI355X onload/forward regression passed: **1 passed + 3 subtests**, PyTorch 2.9.1+ROCm 7.2. This supersedes the earlier GPU-pending note for that unit test only. Full-model Aiter/MXFP4, CUDA hardware and maintainer-enabled upstream CI/review remain pending. [Execution details](https://github.com/sgl-project/sglang/pull/45484#issuecomment-6440222902).
