## Problem and cause

PD DCP packing recreates and copies per-layer pointer/width/stride metadata to the GPU for every chunk. Row-index construction also uses a pageable copy on the calling thread's stream. This adds host work and synchronization to a transfer worker that already gathers the same KV layout repeatedly.

## Change

Cache one metadata layout per pack buffer, with byte prefixes independent of the current row count. Invalidate on pointer, width, stride or device changes. Stage contiguous, pinned row indices asynchronously on the gather stream, and request a high-priority stream for DCP pack buffers. Other staging users retain normal priority. Keep the producer-stream wait and completion synchronization; this PR does not introduce the historical experimental no-wait path.

## Evidence and scope

Historical GLM-5.2/MI355X runs `dcpx8c`, `dcpx9`, `dcpx10` measured mean pack time 77.6 ms, 40.9 ms and 0.6 ms respectively. Those variants bundled no-wait, caching and stream priority; cache versus priority was not isolated. They used simulated MTP acceptance and one run per variant. **These are motivation, not measured speedups for this revised PR.** No stack-throughput claim is made. Original gather equality checks covered 1/777/8192 rows; a corresponding current-tree GPU regression is included.

Archive: `yihou/glm52.dcpx-pack-fix-best.packup_20261023/{opt.readme.md,patches/tests/}` in AMD-AGI/Infera, experiment archive commit `6d45fc2d`.

## Validation

Run `s10-cpu-20261027`, base `8a455f1b9005a05091ad1825f0f3493bfd74d27d`: **8 passed, 4 subtests passed** with PyTorch 2.14.1+cpu. The GPU regression has not run.

CPU command (CPU-only PyTorch):
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m pytest test/registered/unit/disaggregation/test_dcp_pack_metadata.py test/registered/unit/disaggregation/test_dcp_pack.py -q
```

GPU command, registered for AMD and CUDA:
```bash
PYTHONPATH=python python test/registered/kernels/ops/kvcache/test_pd_dcp_gather.py
```

## Draft blockers

- Execute the real gather/PP transfer tests on AMD and CUDA; CPU tests mock the kernel boundary.
- Measure this retained-wait version independently under concurrent prefill, including cached-prefix PP sends; establish whether priority helps without harming prefill latency.
- Coordinate overlapping changes to this path in #44724, #39768 and #43782.
- Maintainer review and enabled upstream CI are pending. This is not ready to merge.
