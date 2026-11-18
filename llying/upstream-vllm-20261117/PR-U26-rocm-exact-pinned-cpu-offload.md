# [ROCm][KV offload] Allocate CPU offload tensors at their exact size instead of through the power-of-two pinned allocator

Draft for vllm-project/vllm (base: v0.34.0). Candidate U26 in `CANDIDATES.zh-CN.md`.

## Problem

On ROCm, `CPUOffloadingSpec._uses_shared_region()` is False, so `CPUOffloadingWorker` allocates one private host
tensor per KV cache tensor:

```python
cpu_tensor = torch.zeros((num_cpu_chunks, cpu_page_size_bytes), dtype=torch.int8, device="cpu", pin_memory=True)
```

PyTorch's caching host allocator rounds every pinned allocation up to the next power of two. With a few dozen KV
tensors per worker, the host memory actually pinned is up to twice `cpu_bytes_to_use`, and it jumps discontinuously
as the per-tensor size crosses a power of two:

| GLM-5.2-MXFP4, TP1 x PP4 prefill (24-27 KV tensors per worker), MI355X | `cpu_bytes_to_use` | pinned (sum over 4 workers) |
|---|---|---|
| per tensor ~15.6 GB -> 16 GiB | 1.25 TB | 1,360 GiB (1.46 TB, +17%) |
| per tensor ~20 GB -> 32 GiB | 1.6 TB | grew past 2.6 TB, host OOM (2.75 TB host), a worker was killed |

Minimal reproduction (one MI355X): two 20 GB tensors through `torch.zeros(pin_memory=True)` add 64 GiB of `Shmem`
(`exact_pin_test.py`); the same two allocated with `hipHostAlloc` directly add 37 GiB, report `is_pinned() == True`,
round-trip GPU <-> host correctly, and allocate in 6.3 s instead of 14.4 s (`exact_pin_test2.py`).

CUDA is not affected (it uses the shared mmap region, which `pin_mmap_region` registers at its exact size).

## Change

`vllm/v1/kv_offload/cpu/gpu_worker.py`:

- `_exact_pinned_host_tensor(num_rows, row_bytes)`: allocate exactly `num_rows * row_bytes` bytes of pinned host
  memory with the runtime's host allocator (`hipHostAlloc`, i.e. the same driver allocation torch uses underneath,
  without the caching allocator's rounding) and view it as an int8 tensor; returns None if unavailable or the
  allocation fails.
- `CPUOffloadingWorker.__init__`: for private (non-region) tensors, use it when pinning is enabled, falling back to
  `torch.zeros(pin_memory=True)` if it returns None.
- `CPUOffloadingWorker.shutdown`: after the transfer handlers have stopped, synchronize and free the allocations.
- `VLLM_KV_OFFLOAD_EXACT_PIN=0` restores the old allocation (to be registered in `vllm/envs.py` upstream).

For upstream, the allocator should come from `CudaRTLibrary` (`cuda_wrapper.py`): add `cudaHostAlloc` / `cudaFreeHost`
with the HIP mapping `hipHostMalloc` / `hipHostFree`, instead of the overlay's direct `ctypes` lookup of
`libamdhip64` (which keeps the overlay to one file).

The diff against v0.34.0 is `../vllm-p4d4-crusoe-20261117/overlay-cpupin/gpu_worker.py.diff` (overlay file next to it).

## Why not anonymous mmap + `hipHostRegister`

The first version of this change (used in the m10 / m12 / m13 / m14 runs, 11-18) put each tensor in an anonymous
`MAP_PRIVATE` mmap and registered it in chunks with the existing `host_register`, like `pin_mmap_region`. It sized
memory exactly too (1.9 TB requested -> 1.90 TB `AnonPages`), and m10 ran 3600 s with it at 48,572 tok/s/GPU. But
the repeat m12 (same config, same node) hung 12-14 minutes into the measurement: decode throughput fell over two
minutes to zero, then the decode EngineCore's `execute_model` RPC timed out; during the slowdown the host logged
`workqueue: evict_process_worker [amdgpu] hogged CPU for >10000us` (4 times by the stall, 64 by the end). No run
with torch's pinned allocator (s13, m6, m9b) logged that warning; m10 did not either. One occurrence does not
establish the cause, but registered user memory is the only new driver-level factor, and allocating through the
runtime keeps the same mechanism torch already uses, so the driver allocation is the lower-risk form.

## Validation

- m10 (mmap + register version, P4D4 GLM-5.2, 3600 s AgentX C104): `cpu_bytes_to_use` 1.9 TB pinned 474.5-474.9 GB
  per worker, 1.90 TB in total, host `MemAvailable` stable at ~760-810 GiB; the larger tier cut prefill
  recomputation from 6.66% to 3.92% of prompt tokens and the score to 48,572 tok/s/GPU (43,555 at 1500 s with
  1.25 TB in m6).
- `hipHostAlloc` version: allocation, `is_pinned()`, round trip, and free checked in isolation
  (`exact_pin_test2.py`, `exact_pin_test3.py` with the overlay module); full-run results to be added (P8D8 and the
  next P4D4 rounds use it).
- Unit test to add: allocate a small CPU offload worker on the platform, check the host tensors' total size equals
  `num_cpu_chunks * page_bytes` (not rounded) and `is_pinned()`; round-trip a block with `swap_blocks`.

## Notes

- With `hipHostAlloc` the memory is accounted as `Shmem` (like torch's pinned memory) and is freed at shutdown (no
  allocator cache keeping it).
- An alternative is to reuse `SharedOffloadRegion` on ROCm (one mmap per worker, `pin_mmap_region`), which would also
  give exact sizing; it was left out because the region path is disabled on ROCm for reasons not covered here, and it
  relies on registered user memory as well.
