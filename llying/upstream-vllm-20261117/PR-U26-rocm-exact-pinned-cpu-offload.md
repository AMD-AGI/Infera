# [ROCm][KV offload] Pin CPU offload tensors at their exact size instead of through the power-of-two pinned allocator

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

Minimal reproduction (one MI355X, `exact_pin_test.py`): two 20 GB tensors through `torch.zeros(pin_memory=True)` add
64 GiB of `Shmem`; the same two as anonymous mmaps registered with `host_register` (`hipHostRegister`) add 37 GiB,
report `is_pinned() == True`, and round-trip GPU <-> host correctly in the same time.

CUDA is not affected (it uses the shared mmap region, which `pin_mmap_region` registers at its exact size).

## Change

`vllm/v1/kv_offload/cpu/gpu_worker.py`:

- `_exact_pinned_host_tensor(num_rows, row_bytes)`: an anonymous `mmap` of exactly `num_rows * row_bytes` bytes,
  viewed as an int8 tensor, registered with the existing `host_register` in chunks of at most
  `MAX_HOST_REGISTER_CHUNK_BYTES` (the same bound `pin_mmap_region` uses); on any registration failure it unregisters
  what it registered and returns None.
- `CPUOffloadingWorker.__init__`: for private (non-region) tensors, use it when pinning is enabled, falling back to
  `torch.zeros(pin_memory=True)` if it returns None; keep the mmaps alive for the worker's lifetime.
- `CPUOffloadingWorker.shutdown`: unregister the ranges after the transfer handlers have stopped (with the same
  device sync on handler failure as the region path).
- `VLLM_KV_OFFLOAD_EXACT_PIN=0` restores the old allocation (to be registered in `vllm/envs.py` upstream).

The diff against v0.34.0 is `../vllm-p4d4-crusoe-20261117/overlay-cpupin/gpu_worker.py.diff` (overlay file next to it).

## Validation

- P4D4 GLM-5.2 (m10, 3600 s AgentX C104): `cpu_bytes_to_use` 1.9 TB pinned 474.5-474.9 GB per worker, 1.90 TB in total
  (`AnonPages`), host `MemAvailable` stable at ~760-810 GiB; no transfer errors; the larger tier cut prefill
  recomputation from 6.66% to 3.92% of prompt tokens and the score to 48,572 tok/s/GPU (from 43,555 at 1500 s with
  1.25 TB in m6).
- Unit test to add: allocate a small CPU offload worker on the platform, check the host tensors' total size equals
  `num_cpu_chunks * page_bytes` (not rounded) and `is_pinned()`; round-trip a block with `swap_blocks`.

## Notes

- Memory now shows as `AnonPages` instead of `Shmem`, and is released with the process (no allocator cache).
- An alternative is to reuse `SharedOffloadRegion` on ROCm (one mmap per worker, `pin_mmap_region`), which would also
  give exact sizing; it was left out because the region path is disabled on ROCm for reasons not covered here.
