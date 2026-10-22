# Feasibility model for offloading prefill-leg MoE expert weights to host and streaming them per layer

**Bottom line:** offloading about 50% of routed-expert weights frees about 40 GiB per GPU. That gives about +0.78 M KV tokens (about 3.21 M), roughly the measured 128-expert point (3.32 M, +13%). At the median step time, the copies would mostly hide behind compute, but they would be exposed on short or fast steps. 100% offload cannot be hidden, because one layer's copy takes longer than one layer's compute. 25% hides at every percentile, assuming at least 50 GB/s.

What rests on what:
- **Measured:** layer and expert sizes, the 119.84 GiB per-GPU weight footprint, step times, and the HiCache settings.
- **Assumed:** host-to-GPU (H2D) bandwidth. No H2D measurement exists anywhere I searched.
- **Projected:** the score gain. It is extrapolated from two measured KV points, not measured for offload.

## 1. Expert weight bytes (from safetensors headers, measured)

Config: hidden 6144, moe_intermediate 2048, 256 routed experts, top-8, 1 shared expert, 78 layers, first_k_dense_replace=3, so 75 MoE layers. Routed experts are MXFP4: U8 weights plus e8m0 per-32 scales. Attention, `lm_head` and the dense MLPs of layers 0-1 are in the quant `exclude` list (bf16). The MTP layer's experts are bf16 (18.0 GiB), but the prefill leg does not load MTP (`speculative_algorithm: None` in the pf90 prefill log).

Per expert (layer 10):

| tensor | shape | bytes |
|---|---|---|
| gate_proj / up_proj weight | [2048, 3072] U8 | 6,291,456 each |
| down_proj weight | [6144, 1024] U8 | 6,291,456 |
| scales (3 tensors) | — | 393,216 each |
| total | — | 20,054,016 (of which 1,179,648 is scales) |

| item | bytes | GiB |
|---|---|---|
| routed experts per layer, full | 5,133,828,096 | 4.781 |
| routed per layer per GPU (TP4, EP1) | 1,283,457,024 | 1.195 |
| offloadable per layer per GPU (weights only, 257 experts incl. fused shared) | 1,212,678,144 | 1.129 |
| routed, all 75 layers, full | 385,037,107,200 | 358.59 |
| routed per GPU (TP4) | — | 89.65 |
| offloadable per GPU (75 layers) | — | 84.7 |

Check against the measured 119.84 GiB per GPU:
- Routed 89.65 + attention 24.34 (replicated, attention TP=1 under DP-attention [inferred]) + shared 0.35 + dense MLP 0.37 + embedding/`lm_head` 0.9-2.7 = 115.6-117.4 GiB.
- That leaves 2-4 GiB unexplained. Extra copies made during post-load weight processing are one possible source [inferred].
- Routed experts are about 75% of resident weights.

"Shared experts fusion optimization enabled" appears in the prefill log, so the shared expert sits inside `w13_weight`.

## 2. Host-to-GPU bandwidth

- **No measured H2D or D2H number found.** I grepped `/home/yihou/dev/git/infera.glm52.pd/yihou/**/*.md` for H2D, GB/s and bandwidth. Every hit is RDMA/mooncake: 41-46 GB/s rail, 29-30 GB/s GPU over RDMA. HiCache does not log its throughput. `/home/yihou/dev/git/infera.glm52.view/` does not exist.
- **Assumption:** PCIe Gen5 x16 gives 64 GB/s theoretical, so I use about 50 GB/s effective per GPU, and show 25 GB/s as a contention case.
- How the 4 prefill GPUs (0-3, NUMA0 per CLAUDE.md) connect to host is unknown. They may share upstream switches or links [inferred].
- **Measurement that would settle it:** a 4-GPU concurrent pinned `hipMemcpyAsync` H2D test in the container, run on GPUs 0-3 while the HiCache pools are allocated.

## 3. Prefill compute per layer (pf90 prefill-0.log, measured)

Method: time between back-to-back prefill logs on the same DP rank, counting only cases where the previous log showed pending ≥ 8192 and the step was a full 8192-token chunk.

| cached tokens | n | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| 0 | 8701 | 0.64 s | 0.99 s | 1.27 s | 1.58 s | 2.10 s |
| > 0 | 733 | 0.37 s | 0.96 s | 1.42 s | 1.96 s | 3.20 s |

- Dividing by 78 layers gives about 16.3 ms per layer at p50, 12.7 ms at p25 and 8.2 ms at p5. This assumes all layers cost the same [inferred].
- 38% of per-rank prefill logs (6462/16887) are partial chunks, and 21% are under 2048 tokens. Every forward pays the full expert transfer regardless of token count, so these short steps get the least cover.
- One offloaded layer's copy at 50 GB/s takes 24.3 ms. At 64 GB/s it takes 19.0 ms; at 25 GB/s, 48.5 ms.

**How OffloaderV2 schedules copies (code read, `srt/utils/offloader.py`, `srt/models/deepseek_v2.py` in the v0.5.21 image source):**
- It is selected when `offload_group_size > 0` (offloader.py:68-83).
- It offloads layers where `index % group_size >= group_size - num_in_group` (offloader.py:255-258).
- At the end of each offloaded module's forward, it prefetches the module `prefetch_step` positions ahead in its offloaded list (offloader.py:295-300). Each copy therefore has to hide behind the compute between consecutive offloaded layers.
- In `cpu` mode it keeps a pinned host copy (offloader.py:424-430, 479-490). Each onload does `param.to("cuda", non_blocking=True)` on a separate `offload` stream (offloader.py:359-368).
- For DeepSeek-V2 models it moves only `layer.mlp.experts` → `w13_weight` and `w2_weight`; the extra scale names are listed only for nvfp4 (deepseek_v2.py:2894-2916). So the MXFP4 `*_weight_scale` tensors (about 5.3 GiB per GPU) stay on the GPU.
- Dense layers 0-2 count toward the index pattern but have an empty whitelist.
- It is untested whether these pinned-CPU params survive the AITER MXFP4 post-load weight processing [inferred risk].

## Scenario table

Assumptions:
- Prefetch_step=1, so two offloaded layers are resident at a time (about 2.26 GiB reserved).
- KV costs 55,226 B per token (125.14 GiB / 2,433,088 tokens).
- The baseline is 2.43 M tokens at mem-fraction 0.88.
- Bandwidth is 50 GB/s per GPU.

| offload | layers | GPU freed (net) | KV tokens | Δ KV | prefetch window (p50 / p5) vs 24.3 ms copy | H2D per step | exposed time / step |
|---|---|---|---|---|---|---|---|
| 25% (1 of every 4) | ~19 | ~19.2 GiB | ~2.80 M | +0.37 M (+15%) | 65 / 33 ms: hidden | 23 GB, 0.46 s | ~0 (at 25 GB/s: exposed at p5) |
| 50% (2 of every 4) | ~37.5 | ~40.1 GiB | ~3.21 M | +0.78 M (+32%) | 33 / 16 ms: hidden at p50, exposed below about p25 | 45 GB, 0.91 s | p5: ~0.30 s on a 0.64 s step (+46%) |
| 100% | 75 | ~82.4 GiB | ~4.03 M | +1.60 M (+66%) | 16 / 8 ms: exposed | 91 GB, 1.82 s | p50: ~0.6 s (+47%); each step takes at least 1.82 s |

Reference points (measured): 128 experts gave 3.32 M tokens and +13%; 8 experts gave 3.67 M and +16%. The score gain is clearly smaller from 3.32 M to 3.67 M than from the baseline to 3.32 M, so offloading beyond about 50% buys little score while costing throughput [inferred from two points].

## Contention with HiCache (same PCIe link)

- **Backup (GPU to host):** with write_through, each full 8192-token chunk sends about 452 MB per rank (8192 × 55.2 kB). That is the opposite direction to the expert onload; PCIe is full duplex [inferred].
- **Load-back (host to GPU):** this competes directly with expert onload. For example, 60k host-hit tokens is about 3.3 GB, roughly 66 ms at 50 GB/s.
- 733 of 9434 back-to-back full-chunk steps had cached > 0, but the log does not split device hits from host hits. Not found.

## 4. Host memory

- **Already committed:** the prefill HiCache allocates 450 GB × 4 ranks = 1.8 TB (log: "Allocating kv hierarchical KV host pool: 10016064 tokens, 450.00 GB host memory"). The decode leg has `enable_hierarchical_cache: False` (decode-0.a1.log).
- **Pinned expert copies needed:** each TP rank pins its own shard. The `shm_cpu` sharing mode asserts `tp_size==1` (offloader.py:224-226), so it does not apply here.
  - 25%: about 92 GB
  - 50%: about 182 GB
  - 100%: about 364 GB
- **Fit:** 1.8 TB + 0.36 TB = 2.16 TB, below 2.75-3 TB (CLAUDE.md records 3 TB on n10-29). I could not see the pf90 compute node's RAM, since spur006 is a 125 GB login node. Remaining unknowns: container memory limits, and whether the copies would be NUMA0-local (they should be [inferred]).

## Upstream PRs (from `gh search`)

- #24531 (closed): "OffloaderV2 … Support CUDA Graph and all weights offloading" — https://github.com/sgl-project/sglang/pull/24531
- #32769 (open): no-copy-engine transfer with arbitrary expert counts — https://github.com/sgl-project/sglang/pull/32769
- #19541 (merged): NPU fixes for OffloaderV2 — https://github.com/sgl-project/sglang/pull/19541

I did not look for ROCm/AITER-specific OffloaderV2 support; not found in the search above.

Key paths:
- Checkpoint: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4/` (config.json, model.safetensors.index.json, headers read)
- Offloader code: `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/utils/offloader.py`
- Model wiring: `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/models/deepseek_v2.py:2894-2916`
- Prefill log: `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pf90/server-logs/prefill-0.log`
- Decode log: `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pf90/server-logs/decode-0.a1.log`

=====VERIFY=====
# Adversarial verification: offloading prefill MoE expert weights with OffloaderV2 (sglang v0.5.21 image source)

**Verdict:** the byte arithmetic and the offloader code citations hold. Two findings change the plan:

1. **The quark MXFP4 path breaks OffloaderV2 `cpu` mode as shipped (WRONG / missing).** Two things go wrong:
   - **The weights stop being pinned.** On gfx950, `process_weights_after_loading` runs after the parameters have already been moved to pinned host memory. It then does `layer.w13_weight.data = shuffle_weight(layer.w13_weight.contiguous(), (16,16))` and the same for w2 (`srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py:938-946`, with `_is_shuffle_moe_mxfp4 = _use_aiter and _is_gfx95` at :49). Each of those lines makes a fresh CPU tensor that is not pinned.
   - **The pre-shuffled flag is lost.** The offloader swaps in `param.to("cuda")` through `functional_call`, and that plain tensor does not carry the Python attribute `is_shuffled`. `apply_weights` only forwards the flag `if hasattr(layer.w13_weight, "is_shuffled")` (quark_w4a4_mxfp4_moe.py:993-995). AITER `fused_moe` chooses its kernel family from `getattr(w1, "is_shuffled", False)` (ROCm/aiter `aiter/fused_moe.py:364,1153,1259`, upstream main). Expected result: an unshuffled-layout kernel runs on shuffled data, which would also make the perf numbers unrepresentative [inferred].
   - **Effect of losing pinning:** `non_blocking` H2D from pageable memory does not overlap with compute [inferred, PyTorch semantics].
   - **Existing hook:** sglang already has `offloader.update_param()` (`srt/utils/offloader.py:467-476`) to re-pin. Only `fp8_utils.py:1841` calls it; the quark path does not.
   - **Fix size:** three patches.
     - Use `update_param` for w13/w2 in the quark MoE method.
     - Carry `is_shuffled` onto the device tensors, either in `_CpuParamOffloader.create_device_tensor` or in `apply_weights`.
     - Check whether `shuffle_weight` behaves correctly on CPU tensors. The aiter source in the image was not extracted, so this is UNVERIFIED.
2. **The prefetch windows are overstated, and 50% offload with the default pattern is not "hidden at p50" (WRONG).**
   - **How the copy is gated.** `start_onload` does `alt_stream.wait_stream(current_stream)` (offloader.py:358-366) and runs from the experts submodule's forward-end hook (offloader.py:295-300). So the copy for the next offloaded layer starts only after the current offloaded layer's experts finish, and must be done before the next offloaded layer's experts start.
   - **25%** (`group_size=4, num_in_group=1`, layers 3, 7, 11, ...): the window is about 3 layers plus one attention block, not 4 layers. Taking the report's own per-layer averages, that is about 49 ms at p50 and about 25 ms at p5, against a 24.3 ms copy. So p5 is borderline at 50 GB/s, not "hidden at every percentile".
   - **50%** (`group_size=4, num_in_group=2`, layers 2(dense), 3, 6, 7, 10, 11, ...): offloaded layers come in adjacent pairs. With `prefetch_step=1`, the copy for layer 7 is issued at the end of layer 6's experts. It only has layer 7's attention and gate time to hide behind, so roughly 24 ms minus that attention time is exposed on every second offloaded layer, even at p50.
   - **What would work instead** [inferred]:
     - `--offload-group-size 2 --offload-num-in-group 1` gives 50% as alternating odd layers: 3, 5, ..., 77, which is 38 MoE layers. Each window is then about 1 layer plus one attention block: about 16 ms + attn at p50, against 24.3 ms. Still partly exposed.
     - `prefetch_step=2` would cover it, at the cost of about 1.1 GiB more resident memory.
   - **Missing measurement:** the attention vs MoE split of per-layer time. Without it these windows cannot be stated with confidence.

## Claim-by-claim

| # | claim | tag | evidence / correction |
|---|---|---|---|
| 1 | OffloaderV2 is selected when `offload_group_size > 0` | CONFIRMED | offloader.py:68-82. It asserts `cpu_offload_gb == 0`. Flags default to group_size -1, num_in_group 1, prefetch_step 1, mode "cpu" (`srt/arg_groups/fields/exec_.py:953-965`). |
| 2 | Layers are chosen by `index % group_size >= group_size - num_in_group` | CONFIRMED | offloader.py:257. The index runs over all decoder layers, dense ones included. |
| 3 | Each offloaded module prefetches the module `prefetch_step` ahead | CONFIRMED, missing detail | offloader.py:296-297. Indexing wraps modulo the list, so the last offloaded layer prefetches the first one for the next forward. `post_init` pre-onloads the first `prefetch_step` modules (:283-288). The `wait_stream` gating (:362) is the detail behind finding 2. |
| 4 | `cpu` mode keeps a pinned copy; onload is `param.to("cuda", non_blocking=True)` on its own stream | CONFIRMED | :424-430 and :479-490. The stream is `get_stream("offload")` (:250). |
| 5 | GLM-5.2 goes through the deepseek_v2 wiring | CONFIRMED | `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` at `srt/models/glm4_moe.py:1275`. The whitelist is `w13_weight`/`w2_weight`, plus `*_blockscale_swizzled` only for nvfp4 (deepseek_v2.py:2894-2916). The MXFP4 `*_weight_scale` tensors stay on GPU, about 5.29 GiB per GPU (1,179,648 × 257 / 4 × 75 B). |
| 6 | The hook actually fires on our path | CONFIRMED [path inferred] | The AITER path calls `self.experts(...)` (deepseek_v2.py:1159/1163, 1344/1350). Deferred-finalize and `forward_impl` bypasses need flashinfer-trtllm (`fused_moe_triton/layer.py:512-515`), so they are not taken on ROCm. |
| 7 | AITER MXFP4 post-load processing survives the offload (the report's "untested risk") | WRONG: it breaks | See finding 1. Ordering: `wrap_modules` runs during model construction (`srt/utils/common.py:1629`), so `_CpuParamOffloader.__init__` has already pinned the params. Weight loading and `process_weights_after_loading` then run on CPU params. `offloader.post_init()` runs afterwards (`model_runner.py:1262-1263`) and is a no-op for `cpu` mode. |
| 8 | `shm_cpu` asserts `tp_size==1` | CONFIRMED | offloader.py:223 and :439. `sharded_gpu` asserts the same (:544). Only `cpu` and `meta` modes are usable at TP4. |
| 9 | Guards that reject our configuration | none found | Only `--ple-offload-embedding` is rejected (`arg_groups/memory_hook.py:28-38`). The `startup_weight_load.py:376` rule only switches off the optional CUDA-only startup-overlap loader, which falls back. `forbid_copy_engine_usage` is read only by `moe_runner/deep_gemm.py:1428`, which is not on our path. No guard against DP-attention, PD, HiCache or ROCm was found. |
| 10 | Per-expert and per-layer byte table | CONFIRMED | Arithmetic re-derived: 3 × 6,291,456 + 3 × 393,216 = 20,054,016 B; × 256 = 5,133,828,096 B; /4 = 1,283,457,024 B; 257 × 18,874,368 / 4 = 1,212,678,144 B; × 75 = 84.7 GiB. Copy time 24.25 ms at 50 GB/s. |
| 11 | KV cost 55,226 B/token; 50% offload gives +0.78 M tokens | CONFIRMED (arithmetic) | The 50% default pattern gives exactly 37 MoE layers, not ~37.5, because index 2 is dense. The group_size=2 alternative gives 38. |
| 12 | Resident memory is 2 layers at `prefetch_step=1` | CONFIRMED [inferred] | One layer in use plus one prefetched. Per-step allocations come from the offload stream's caching-allocator pool. Fragmentation is not measured. These transient buffers are not reserved by mem-fraction sizing, so mem-fraction has to drop by about 2.3 GiB / 288 GiB. |
| 13 | Log facts | CONFIRMED | pf90 prefill-0.log has: `Allocating kv hierarchical KV host pool: 10016064 tokens, 450.00 GB`, `Shared experts fusion optimization enabled`, `speculative_algorithm': None`, `hicache_io_backend': 'kernel'`, `mem usage=119.84 GB`, `KV ... #tokens: 2541376, 130.71 GB` (pf90 is mem-fraction 0.90). decode-0.a1.log has `enable_hierarchical_cache': False`. |
| 14 | H2D bandwidth | UNVERIFIED | Still no measurement. Concurrent 4-GPU pinned H2D on GPUs 0-3 is the deciding experiment. |
| 15 | Step-time percentiles | UNVERIFIED (not re-derived) | The equal-cost-per-layer assumption is [inferred]. The MoE vs attention fraction is not measured and now matters (finding 2). |
| 16 | #24531 closed, #32769 open, #19541 merged | CONFIRMED | #19541 merged 2026-04-30. |
| 17 | No ROCm/AITER-specific OffloaderV2 support found | CONFIRMED (not found) | Also no quark or MXFP4 offload PR. |

## Upstream PRs the report missed

- #8034 (merged), "Overlapped weight offload": the original OffloaderV2 — https://github.com/sgl-project/sglang/pull/8034
- #43824 (open), "[NPU] Pin CPU-offloaded weights for host-to-device copies": the same pinned-vs-pageable class of bug, but in OffloaderV1 on NPU — https://github.com/sgl-project/sglang/pull/43824
- #44139 (open), "Preserve cached weights during CPU offload" — https://github.com/sgl-project/sglang/pull/44139
- #38779 (merged), "post-load staging regression breaks offload meta/sharded_gpu modes" — https://github.com/sgl-project/sglang/pull/38779
- #34558 (closed), "Preserve MXFP4 Triton weights in sharded state" — https://github.com/sgl-project/sglang/pull/34558
- #23550 (merged): GLM-5.1, skip offloader `post_init` on draft worker — https://github.com/sgl-project/sglang/pull/23550
- #29166 (merged): inline H2D during CUDA graph capture. Not relevant to us, since prefill does not capture graphs.

## Development estimate [inferred]

- **Small patch set:**
  - `update_param` plus the `is_shuffled` carry-over in the quark MoE method and the offloader.
  - Optionally offload `*_weight_scale` too, worth about +5.3 GiB.
  - Pick `group_size=2, num_in_group=1, prefetch_step=2`.
- **Experiments that must come first:**
  - The 4-GPU concurrent H2D bandwidth test.
  - A per-layer attention vs MoE timing split.
  - A smoke run checking that `is_shuffled` reaches `fused_moe` (log `getattr(w1, "is_shuffled")`).

Files:
- `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/utils/offloader.py`
- `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`
- `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/models/deepseek_v2.py`
- `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/models/glm4_moe.py`
- `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/arg_groups/memory_hook.py`
- `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/model_executor/model_runner.py`
- `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pf90/server-logs/prefill-0.log`
- `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pf90/server-logs/decode-0.a1.log`