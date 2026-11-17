Review round, 3bfe2416 (a new commit on top, so the change is easy to see):

- The one-page pin now moves on to the next RDMA device when one rejects the dma-buf MR, and only a successful pin makes a device preferred (`hip_gem_pin.h`, tested by `hip_gem_pin_test`, which needs no GPU or NIC).
- Device lookups on per-transfer paths (the HIP transport's copy device, the TCP transport's staging device) no longer `dlsym()` for the allocator; they use one that a registration already found. Processes that do not use `HipGemAllocator` therefore do no extra work per transfer.
- A non-positive size is rejected, the test polls with a sleep and a time limit, `build_wheel.sh` quotes its paths, and the render node is globbed as `render*`.

Tested on MI355X, amdgpu-dkms 6.19.14, ionic NICs, with each test run on upstream 504f2ae4 (this PR's base) and on this branch:

| Test | 504f2ae4 | This PR |
|---|---|---|
| `rdma_hip_kfd_pin_test` | 2/2 | 2/2 |
| `hip_transport_test` (two GPUs) | 5 passed, 1 skipped | 5 passed, 1 skipped |
| `hip_stream_pool_test`, `MC_HIP_STREAM_PRIORITY` unset / default / high / invalid | 2/2 each | 2/2 each |
| `transport_uint_test` | 71/71 | 71/71 |
| `hip_gem_pin_test` | – | 4/4 |
| `hip_gem_allocator_test` | – | 4/4 |

#5587 is rebased on this; its `hip_gem_allocator_test` through TENT passes 4/4.

The `tent-ci (ub-mock)` failure on the previous run was `tent_runtime_queue_dispatch_test`, which this PR does not touch (that job builds without HIP, so none of this code is compiled there). It passed on #5587, which contains this PR.
