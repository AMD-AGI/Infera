<!-- 已提交：https://github.com/sgl-project/sglang/pull/50938（11-17）。标题：[Feat] [AMD] mooncake: Add the AMD_GEM custom memory pool for KV caches on ROCm
     分支 llying-001:llying/mooncake-amd-gem-pool（本地 6886f1e4c7，基于 main 6de483b326）。依赖 Mooncake 第一个 PR（HipGemAllocator），与它同时提，Mooncake 发版含该 PR 后再合并。11-17 填好结果，Mooncake PR 为 #5584。 -->

## Motivation

On AMD GPUs with amdgpu-dkms 6.18.4 through at least 7.1.9, and on kernels without torvalds/linux [9131cc85d0de](https://github.com/torvalds/linux/commit/9131cc85d0dedd8ba4cae60dbe5985d081551f30) (v7.4-rc1), the KFD allocation limit counts device memory that a NIC pins twice. Once Mooncake registers a KV cache larger than the VRAM left after allocating it, every later device allocation of the process fails: the PD scheduler segfaults in `c10::hip::HIPStream::stream()` (`run_event_loop`, at `self.schedule_stream.cuda_stream == self.forward_stream.cuda_stream`) right after the KV registration. With GLM-5.2 on MI355X this hits every PD deployment on those drivers (121–160 GB of KV per prefill GPU, 142–182 GB per decode GPU).

## Modifications

- `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM` allocates the KV pools with `mooncake.allocator.HipGemAllocator` (kvcache-ai/Mooncake#5584): amdgpu GEM BOs imported into HIP, which the limit counts once. It needs a Mooncake release with that PR; with an older one the existing `ImportError` path logs a warning and keeps the default pool.
- Unlike `NVLINK` and `BAREX`, it only changes where the pools live: the KV transfer keeps the default batched path (the per-layer submission branches in `conn.py` exclude it).
- Docs: the value in the environment variable reference, and a section in the PD disaggregation guide on how to recognize the problem and what the pool needs.
- Unit test: `test/registered/unit/disaggregation/test_mooncake_amd_gem_pool.py` (CPU, mocks Mooncake and `torch.cuda.MemPool`).

## Accuracy Tests

GLM-5.2 (MXFP4) PD disaggregation, one-node 4+4 on an MI355X node with amdgpu-dkms 6.19.14, SGLang main `44ad9d69f` with this change, Mooncake main with the companion PR, `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM`:

| | Without the pool | With `AMD_GEM` |
|---|---|---|
| Start-up | all four prefill schedulers segfault in `HIPStream::stream()` (3 of 3 logged runs, another 6.19.14 node, the image's own Mooncake) | ready; KV 121.77 GB per prefill GPU, 141.74 GB per decode GPU, registered over 8 RNICs |
| Smoke | – | 6/6 |
| GSM8K probe | – | 197/200, no garbled outputs |
| 16k-token needle / GSM8K | – | 32/32, 32/32 |
| MTP acceptance | – | 3.64 |

Again with the current revision of the Mooncake PR, which also locates GEM buffers on their own GPU (HIP reports imported memory as device 0's) so each rank uses its own NICs, on another 6.19.14 node: smoke 6/6, GSM8K probe 196/200, no garbled outputs, needle 31/32, GSM8K 30/32, MTP acceptance 3.58. The same node with the earlier revision: 198/200, 32/32, 31/32, 3.57.

## Speed Tests and Profiling

The KV transfer path is unchanged. Same-NIC RDMA write bandwidth into GEM buffers equals that into `hipMalloc` buffers, and GPU copy and read bandwidth of GEM buffers match `hipMalloc` buffers (details in the Mooncake PR). On a production-like GLM-5.2 1P1D PP8 deployment (prefill on a 6.19.14 node, decode on a 6.16.13 node, an earlier version of the same allocator), total throughput at C208 was 56,298 tok/s against 56,593 for `hipMalloc` KV on another node pair, with interactivity and ITL on par; TTFT was higher on that pair. That allocator version located every rank's KV on GPU 0, so all of it went through GPU 0's NIC; in a same-node gate that mislocation took about 18 times the packets in retransmissions, and the Mooncake PR fixes it. It likely explains the TTFT; we have not re-measured it yet.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit) (isort, ruff, ruff-format and codespell pass).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests) (the new test and the neighbouring EFA allocator test: 4 passed, in the ROCm nightly image).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).
