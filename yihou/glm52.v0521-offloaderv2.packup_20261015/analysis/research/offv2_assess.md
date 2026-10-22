# OffloaderV2 for the GLM-5.2-MXFP4 prefill leg (v0.5.21 image, MI355X, P4D4): feasibility assessment

**Verdict:** we can test this with a small patch set, but whether it works depends on one number nobody has measured: host-to-GPU copy bandwidth with 4 GPUs copying at once.

- Nothing in the code rejects our config in `cpu` mode.
- One correctness blocker must be patched first: the `is_shuffled` tag is lost when weights are copied back to the GPU.
- Offloading about half the MoE layers would free about 40 GiB per GPU, for about +0.78 M KV tokens. That is close to the measured 128-expert point (3.32 M tokens, +13%).
- The copy hides behind compute at the median prefill step only if bandwidth is at least about 40-50 GB/s per GPU. Short or partial chunks will always expose it.

`S=/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt` (on spur006). [inferred] means not checked by reading code or running anything.

## (A) What OffloaderV2 is and how it works

1. It lives in `$S/utils/offloader.py`. It was merged as "Overlapped weight offload", fzyzcjy, 2025-08-23 (https://github.com/sgl-project/sglang/pull/8034). The LMSYS GB200 blog describes using weight offload to shrink prefill EP, relying on a 900 GB/s CPU-GPU link (https://github.com/lm-sys/lm-sys.github.io/blob/main/blog/2025-09-25-gb200-part-2.md). That OffloaderV2 is the mechanism behind it is [inferred].
2. It runs at model construction: `make_layers` wraps the layer list (`utils/common.py:1629`). Layer `i` is offloaded when `i % G >= G - K` (`offloader.py:257`).
3. GLM-5.2 is `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` (`models/glm4_moe.py:1275`). On this path only `layer.mlp.experts` → `w13_weight` and `w2_weight` are offloaded (`models/deepseek_v2.py:2894-2916`).
   - What stays on the GPU: attention, MXFP4 `*_weight_scale` (about 5.3 GiB per GPU), and the dense layers 0-2.
   - The fused shared expert sits inside `w13`/`w2`, so it is offloaded too.
   - The MTP layer is never offloaded (`deepseek_nextn.py:126`).
4. In `cpu` mode the parameters are moved to pinned host memory at construction (`offloader.py:424-427`).
5. Post-load processing, which is where the gfx950 aiter shuffle happens, stages each module onto the GPU, processes it there, and restores the result into new pinned host tensors. Read in code: `loader.py:1084-1085` and `:150-162`, `post_load.py:60-102`.
6. Each offloaded `experts.forward` is replaced by `functional_call` with GPU tensors (`offloader.py:309-321`).
7. At the end of offloaded layer *j*'s experts forward, it calls `start_onload` for offloaded layer *j+step* and frees *j*'s GPU copy (`:295-300`, `:381-383`).
8. `start_onload` first runs `alt_stream.wait_stream(current_stream)`, then `param.to("cuda", non_blocking=True)` on a dedicated "offload" stream, then records an event (`:358-368`, `:430`). The consumer waits on that event (`:370-380`).
9. There is no persistent device buffer. Every forward re-allocates a layer through the caching allocator, so about `step+1` layers are in flight at once.
10. The whole layer's expert shard is copied on every forward, i.e. every prefill chunk, no matter how many tokens or which experts are routed. There is no expert-level selection.

## (B) Availability: our image vs upstream

- **Image vs main:** `offloader.py` in our image is byte-identical to upstream main (md5 `1b36fe1e…`, includes #41818). `post_load.py` and the `ExecOffload` argument block are also identical to main. Main has no offloader feature that we lack.
- **Flags** (`$S/arg_groups/fields/exec_.py:941-965`):

  | flag | default | meaning |
  |---|---|---|
  | `--offload-group-size` | -1 | V2 is on when > 0 |
  | `--offload-num-in-group` | 1 | layers offloaded per group |
  | `--offload-prefetch-step` | 1 | how many offloaded layers ahead to prefetch |
  | `--offload-mode` | `cpu` | one of `meta` / `cpu` / `shm_cpu` / `sharded_gpu` (`offloader.py:391-396`) |
  | `--cpu-offload-gb` | 0 | must stay 0: V1 takes precedence, and V2 asserts it is 0 (`offloader.py:68-83`, `:74-76`) |

  The CLI spelling is confirmed for `--offload-group-size` (`memory_hook.py:36`). All four flag names appear together in the GLM-5.2 FP8 report https://github.com/sgl-project/sglang/issues/32507. Still worth running `--help` in the image.
- **Mode:** only `cpu` mode works at TP4. `shm_cpu` and `sharded_gpu` assert `tp_size == 1` (`offloader.py:223`, `:439`, `:544`).
- **Upstream maturity:**
  - No functional CI test for V2 on CUDA or ROCm. The only mode test is NPU (`test/registered/npu/.../test_npu_offload_modes.py`).
  - No ROCm, MI3xx, quark or MXFP4 report or PR found.
  - GLM use exists on CUDA:
    - #23550 (merged): skip `post_init` on the draft worker, for "the GLM-5.1 expert-offload path".
    - Issue #32507 (closed): GLM-5.2 FP8, EP16, DeepEP, crashed in `copy_to_gpu_no_ce`. Its fix #32769 is still open and is CUDA DeepGEMM only.
  - Data point from #25786: Qwen3-8B dropped from 2707 to 422 tok/s with offload, on a slow link.
  - Unmerged work: #24531 (N-buffer pool and CUDA graphs, stale), #20126 (UVM resident-expert offload, GLM-5-FP8 on 8×H20, stale).
  - Open: #43603, #44139, #36509, #43824.

## (C) Blockers for our config

| item | status | evidence |
|---|---|---|
| quark MXFP4 + aiter preshuffle: `is_shuffled` tag lost | **Needs patch (blocking)** | `create_device_tensor` returns `self._param.to("cuda")` (`offloader.py:430`), a plain tensor without the Python attribute (checked locally on torch 2.12). `apply_weights` re-tags only `if hasattr(layer.w13_weight,"is_shuffled")` (`quark_w4a4_mxfp4_moe.py:993-995`). aiter picks its kernel family by `getattr(w1,"is_shuffled",False)` (ROCm/aiter main `fused_moe.py:364,1153,1259`; the aiter copy in our image was not checked). The same failure is already documented at `mxfp4.py:1758-1760`. Fix is a few lines: copy the attribute in `create_device_tensor`, or tag unconditionally in quark `apply_weights`. Without it the wrong kernel runs, so performance is not comparable and output may be garbage. The acceptance-length waiver does not cover this. |
| pinned host weights after the shuffle | OK (code read) | One verifier claimed the shuffle produces unpinned CPU tensors. **That is wrong.** `postprocess_weights` wraps each module in `device_loading_context(module, cuda)`, the shuffle runs on the GPU, and the restore uses `_copy_data_to_device(..., pin_memory=True)` (`loader.py:1084`, `:150-162`; `post_load.py:60-102`). Not run. |
| ROCm / HIP | OK, untested | Generic `torch.cuda` stream, event and pinned memory. No ROCm guard and no ROCm code in `offloader.py`. Nobody upstream has run it on ROCm. |
| TP4, EP1, DP-attention DP4 | OK in `cpu` mode | No DP or TP assert in `cpu` mode. The MoE is TP-sharded, so every rank (including DP-idle ranks) runs experts and pays the copy. |
| PD prefill (mooncake) | OK (no guard found) | — |
| HiCache write_through, kernel io backend | OK functionally; perf risk | No guard. It shares the PCIe link: HiCache load-back H2D competes with the weight copies, write-back is D2H (full duplex) [inferred]. |
| chunked prefill 8192 | OK; perf risk | The full expert copy happens on every forward. 38% of prefill steps are partial chunks and 21% are under 2048 tokens, so those get almost no cover. |
| no CUDA graphs on prefill | OK | V2 does not force graphs off (`cuda_graph_hook.py:235-237`); PD already disables them (`:242`). |
| two-batch / single-batch overlap | OK, must stay off | The TBO path calls `run_moe_core` directly (`deepseek_v2.py:1935-1937`) and bypasses the hook. The pf90 prefill log shows `enable_two_batch_overlap: False`, `enable_single_batch_overlap: False`, `moe_a2a_backend: none`. |
| MTP draft worker | OK | The prefill log shows `speculative_algorithm: None`. Even with a draft, `post_init` is skipped (`model_runner.py:1262-1263`). |
| startup weight-load fast path | irrelevant | Already disabled by `num_dp_ranks != 1` (`startup_weight_load.py:371`). |
| transient GPU memory outside the KV pool | needs tuning | About (step+1) × 1.13 GiB is allocated per forward, outside mem-fraction sizing. Check peak headroom; mem-fraction may need to drop slightly. |

## (D) Cost model

**Measured facts:**
- Per offloaded MoE layer, per GPU: `w13`+`w2` = 257 experts × 18.87 MB / 4 = 1.21 GB (1.129 GiB), from safetensors headers.
- All 75 MoE layers: 84.7 GiB per GPU.
- KV costs 55.2 KB per token (125.14 GiB / 2.433 M).
- Full 8192-token prefill step, no cache hit: p5 0.64 s, p25 0.99 s, p50 1.27 s, p95 2.10 s. That is about 16 ms per layer at p50 and 8 ms at p5, assuming equal layer cost [inferred].

**Unmeasured:**
- H2D bandwidth. I assume 50 GB/s per GPU, with 25 GB/s as the contention case.
- The attention vs MoE split of per-layer time.

**How to read the hiding window:**
- **Latency:** a copy can start only after the previous offloaded layer's experts finish (`wait_stream`). With `G=2, step=1` that leaves about 1 layer + 1 attention block. With `step=2`, about 3 layers + attention, but copies queue serially on one stream.
- **Bandwidth floor:** total H2D per forward = offloaded layers × 1.21 GB / BW. No forward can be faster than this, whatever the prefetch setting.

| scenario | flags | MoE layers offloaded | net GiB freed per GPU | KV tokens (from 2.43 M) | H2D per forward | floor at 50 / 25 GB/s | vs p50 1.27 s step | pinned host, whole prefill leg |
|---|---|---|---|---|---|---|---|---|
| S0 smoke | `G=8 K=1 step=1` | 9-10 | about 9.0 | about 2.61 M (+7%) | 11.5 GB | 0.23 / 0.46 s | hidden [inferred] | about 46 GB |
| 25% | `G=4 K=1 step=1` | 19 | about 19.2 | about 2.80 M (+15%) | 23 GB | 0.46 / 0.92 s | hidden at p50; window about 3 layers ≈ 25 ms at p5 vs 24 ms copy, borderline | about 92 GB |
| 50% (recommended pattern) | `G=2 K=1 step=2` | 38 (layer 1 is dense, offloads nothing) | about 40.6 | about 3.22 M (+32%) | 46 GB | 0.92 / 1.84 s | hidden at p50 only if BW ≥ about 40 GB/s; exposed at p25 and below, on partial chunks, and on cache-hit steps | about 184 GB |
| 50% (bad pattern) | `G=4 K=2 step=1` | 37 | about 39.5 | about 3.20 M | 45 GB | same | adjacent offloaded pairs give about 0 ms window, so exposed on every second layer even at p50 | about 180 GB |
| 100% | `G=1 K=1` | 75 | about 82.4 | about 4.03 M | 91 GB | 1.82 / 3.64 s | always exposed: +43% or more at p50 | about 364 GB |

**What the score gain depends on:**
- The KV gain only helps if prefill throughput holds. The measured KV benefit flattens: 3.32 M gave +13%, 3.67 M gave +16%. So offloading beyond about 40 GiB per GPU buys little [inferred from two points].
- A prefill slowdown of more than about 10% probably cancels the gain [inferred].

**HiCache contention:**
- Write-through D2H is about 452 MB per full chunk per rank, in the opposite direction to the weight copies [inferred no conflict, full duplex].
- Host-hit load-back is H2D and competes directly with the copies, e.g. 60 k hit tokens ≈ 3.3 GB ≈ 66 ms. The share of host hits vs device hits is not found in the logs.

**Host memory:** HiCache already uses 1.8 TB (4 × 450 GB). Adding 184 GB stays under 3 TB on n10-29 [inferred]. Whether HiCache pins its pool, the container memory limits, and NUMA placement are unverified.

**Note on the 8-expert data point:** it gained only +1.24 M tokens, but removing about 87 GiB of experts should have given about +1.7 M. Some other cap may have applied at 8 experts. Cause unknown. It does not affect the 50% target, because the 128-expert point (about 45 GiB → +0.89 M) matches the arithmetic above.

## (E) Development plan

**Steps, estimated effort [inferred]:**
1. **H2D microbenchmark, no sglang change (about 2 h).** Run in the image container on GPUs 0-3. Pinned 1.21 GB buffer, `hipMemcpyAsync` / `tensor.to(non_blocking)`, measured four ways:
   - one GPU alone;
   - four GPUs at once;
   - four GPUs at once plus concurrent D2H;
   - with the HiCache-sized pinned pool allocated.

   **Gate:** concurrent per-GPU BW ≥ 40 GB/s to go ahead with 50%. At 25-40 GB/s, cap at 25%. Below 25 GB/s, abort.
2. **Per-layer timing (about half a day).** Profile one 8192-token chunk on the current prefill leg (torch profiler or rocprof) to get attention vs MoE time per layer. This tells us the hiding window precisely.
3. **Patch set (about 2-4 h),** kept in the kit's `patches/`, applied in the image build:
   - (a) Make `_CpuParamOffloader.create_device_tensor` carry `is_shuffled` (and any other aiter-relevant attributes) from `self._param` onto the returned tensor.
   - (b) Optional: also tag unconditionally in quark `apply_weights`.
   - (c) Add a debug log once per layer of `getattr(w13,"is_shuffled",None)` at `fused_moe` entry.
4. **Harness knobs (about 1 h):** add `PREFILL_OFFLOAD_GROUP_SIZE`, `_NUM_IN_GROUP`, `_PREFETCH_STEP` to the prefill-leg launch args only; decode stays untouched. Record each change in `notes/`.
5. **Minimal first experiment (S0):** prefill leg with `--offload-group-size 8 --offload-num-in-group 1 --offload-prefetch-step 1`, patched, same mem-fraction 0.90.
   - **Check:** it starts; the `[offloader] ... memory_allocated=` lines (`offloader.py:261`) report about 1.13 GiB per layer; `max_total_num_tokens` rises by about 0.17 M.
   - **Diff test (对拍):** fixed prompts, greedy first token / prefill logits compared with offload on vs off. Run the same comparison unpatched as well, to confirm the `is_shuffled` effect.
   - **Throughput:** prefill-only throughput (8192 chunks, cache off) vs baseline.
6. **Scale up:** try `G=4 K=1` and then `G=2 K=1 step=2`. At each point measure the KV tokens, prefill throughput, and AgentX C48 score.

**Total:** about 2-3 working days to a go/no-go answer, including the long launches (25-35 min each).

**Success criteria:**
- S0 matches the baseline on the diff test and loses at most 2% prefill throughput.
- At 50%: KV ≥ 3.1 M and AgentX C48 ≥ 24,500 (more than half of the 128-expert gain).

**Abort criteria:**
- Concurrent BW < 25 GB/s.
- The diff test still fails after the patch.
- Prefill throughput loss > 10% at 25%.
- Score ≤ 22,306 baseline at the best point.

**Alternatives if this is not feasible:**
- Offload `*_weight_scale` as well: about +5.3 GiB, small patch to the whitelist. It also adds copy volume.
- Look at attention-weight replication. Under DP-attention, the cost model counts about 24 GiB of attention weights per GPU, replicated [inferred]; I have not checked whether attention is actually replicated or sharded. Fewer DP ranks or attention TP > 1 would shard them, but would change prefill compute.
- Change the P:D split, e.g. P5D3 or P6D2, to put more prefill HBM against fewer decode GPUs.
- Tune HiCache for a higher host-hit rate, which recovers capacity without moving weights.
- UVM resident-expert offload, porting #20126 (stale, CUDA). Development cost is large.
- Expert parallelism over more GPUs is not possible inside a fixed 4-GPU prefill leg.

## (F) Open questions only an experiment answers

1. Real pinned H2D bandwidth per MI355X with 4 GPUs copying at once, and with HiCache traffic running.
2. Whether `non_blocking` H2D on the side stream actually overlaps with aiter MoE and the triton DSA kernels on ROCm, or serializes or slows compute through memory-bandwidth contention.
3. Attention vs MoE time per layer at 8192 tokens and on partial chunks, which sets the true hiding window.
4. Whether the `is_shuffled` patch is sufficient (aiter may read other attributes), and whether the image's aiter matches upstream's `getattr` selection.
5. Caching-allocator fragmentation and peak memory from the per-forward 1.13 GiB re-allocations, i.e. whether mem-fraction must drop.
6. Pinned-allocation time at startup, about 184 GB extra on top of HiCache, and host and container memory headroom.
7. Why the 8-expert point capped at 3.67 M tokens.
8. The AgentX C48 score vs offload fraction: whether the KV gain survives the measured prefill slowdown.

**Checked in this pass:**
- pf90 prefill flags: TBO and SBO are off, `moe_a2a_backend` is none, DP-attention is on, mem-fraction 0.9.
- The post-load pinned restore path (`loader.py:150-162`, `:1075-1085`; `post_load.py:40-110`) and the onload path (`offloader.py:355-432`).

Files:
- `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/utils/offloader.py`
- `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`
- `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/model_loader/loader.py`
- `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/model_loader/post_load.py`
- `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/models/deepseek_v2.py`
- `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pf90/server-logs/prefill-0.log`