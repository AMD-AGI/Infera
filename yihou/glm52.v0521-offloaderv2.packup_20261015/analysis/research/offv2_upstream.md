**Upstream status of SGLang OffloaderV2 / MoE weight offload, checked against our v0.5.21 image (2026-10-14)**

Notation: `$S` = `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang`. Scratch copies of the files I compared are in `/tmp/yihou/offload_research/`. Web search was unavailable (API error), so this is based on gh, the LMSYS blog repo via gh, and reading `$S` on spur006.

## 1. Our image vs upstream main

There is no offloader feature on main that our image lacks.
- **`srt/utils/offloader.py`:** the image copy (632 lines) is byte-identical to main (md5 `1b36fe1e…`).
  - It is newer than the `v0.5.21` tag. The tag version still uses `dp_size`; main and the image use `num_dp_ranks`, which came in with #41818 (https://github.com/sgl-project/sglang/pull/41818).
  - The last commit touching the file on main is that same #41818.
- **`srt/model_loader/post_load.py`:** identical to main. It includes the fix for offloaded parameters parked on the meta device (#38779).
- **`ExecOffload` argument block:** identical to main.
- **`deepseek_v2.py` `offloader_kwargs`:** differs from main only by one unrelated line (`stage_facts=`).
- **`memory_hook.py`:** the diff with main is unrelated (decode cuda-graph `max_bs` handling).

## 2. Where it lives in our image

- **Arguments** (`$S/srt/arg_groups/fields/exec_.py:941-965`): `--cpu-offload-gb` (V1), `--offload-group-size` (V2 is on when >0, default -1), `--offload-num-in-group` (default 1), `--offload-prefetch-step` (default 1), `--offload-mode` (default `cpu`).
- **Factory** (`$S/srt/utils/offloader.py:68-83`): V1 if `cpu_offload_gb>0`, otherwise V2 if `offload_group_size>0`.
- **How V2 works** (`offloader.py:204-385`):
  - Layer `i` is offloaded when `i % group_size >= group_size - num_in_group`.
  - Only whitelisted parameters move.
  - At the end of an offloaded module's forward, the copy for module `(i+prefetch_step)` is started on a dedicated `get_stream("offload")` stream, and module `i`'s GPU copy is dropped.
  - The forward pass runs through `torch.func.functional_call` using freshly allocated GPU tensors.
- **Modes** (`offloader.py:388-396`):
  - `meta`: debug only.
  - `cpu`: pinned host memory, a full `.to("cuda")` per layer per forward.
  - `shm_cpu`: shared host memory.
  - `sharded_gpu`: the weights are spread across DP ranks' GPUs and gathered peer-to-peer.
  - `shm_cpu` and `sharded_gpu` assert `tp_size == 1` (`offloader.py:223,439,544`).
  - **Only `cpu` mode is usable with our TP4 prefill without code changes.** The `tp_size` reading for DP-attention + TP4 is [inferred].
- **CUDA graphs:** during capture, weights are copied inline on the capture stream (#29166). We run prefill without CUDA graphs, so this does not affect us.
- **Model wiring:**
  - Only through `make_layers(offloader_kwargs=…)` (`$S/srt/utils/common.py:1629`).
  - The DeepSeek-V2 path, which GLM-5.2 reuses (`$S/srt/arg_groups/model_overrides/deepseek_v2.py:24`), offloads only `layer.mlp.experts` (routed `FusedMoE`) parameters `w13_weight` and `w2_weight`, plus nvfp4 blockscales (`$S/srt/models/deepseek_v2.py:2894-2918`).
  - For MXFP4, `w13_weight_scale`, `w2_weight_scale`, the shared experts and attention stay on the GPU.
- **Quark MXFP4 MoE compatibility (code read, not run):**
  - Serialized MXFP4 weights are created on the default (cuda) device (`$S/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py:233-236`). That satisfies the `device != cpu` precondition (`offloader.py:337`).
  - Post-load processing (the gfx950 shuffle) copies host-resident parameters to the GPU, processes them, and copies the result back into the original pinned buffer when the shape is unchanged (`$S/srt/model_loader/post_load.py:85-96`, called with `pin_memory=True` from `$S/srt/model_loader/loader.py:158-162`).
- **`forbid_copy_engine_usage`** (true in `cpu` mode) is only read by the CUDA DeepEP/DeepGEMM runner (`$S/srt/layers/moe/moe_runner/deep_gemm.py:1428`). [inferred] It has no effect on our aiter path, so on ROCm nothing stops other small host-to-GPU copies from queuing behind the large weight copies. Not measured.
- **Fast-path skip:** offload turns off the startup weight-load fast path (`$S/srt/model_executor/model_runner_components/startup_weight_load.py:374-378`).
- **No other guard found:** I did not find any argument check that rejects offload together with PD disaggregation, DP-attention or HiCache.

## 3. PR history (all `fzyzcjy` core PRs have empty descriptions: no benchmarks, models or hardware stated)

| PR | Date merged | Author | Content |
|---|---|---|---|
| https://github.com/sgl-project/sglang/pull/8466 | closed 2025-09-12 | fzyzcjy | "Several weight offload attempts", not merged; useful subsets were extracted into #8034 |
| https://github.com/sgl-project/sglang/pull/8521 | 2025-08-21 | fzyzcjy | Refactor into `offloader.py` (V1) |
| https://github.com/sgl-project/sglang/pull/8034 | 2025-08-23 | fzyzcjy | **OffloaderV2** "Overlapped weight offload": `naive_distributed`, `host_shared_memory`, server args, DeepSeek-V2 wiring |
| https://github.com/sgl-project/sglang/pull/9536 | 2025-08-25 | fzyzcjy | `sharded_gpu` mode |
| https://github.com/sgl-project/sglang/pull/9948 | 2025-09-14 | fzyzcjy | fp8 support (`update_param`, ep_moe) |
| https://github.com/sgl-project/sglang/pull/11285 | 2025-10-09 | merrymercy | File moved to `srt/utils/` |
| https://github.com/sgl-project/sglang/pull/19541 | 2026-04-30 | Hide-on-bushsh | NPU fixes; tested on deepseekv3-lite with `--tp 1 --dp 2 --offload-mode sharded_gpu --offload-group-size 4 --disable-cuda-graph` |
| https://github.com/sgl-project/sglang/pull/23550 | 2026-05-09 | zRzRzRzRzRzRzR | "Fixes blocking the **GLM-5.1 expert-offload path**": skip offloader `post_init` on draft (MTP) workers; N=32 in `copy_to_gpu_no_ce` |
| https://github.com/sgl-project/sglang/pull/25786 | 2026-05-20 | xiaobao520123 | `weight_loader` attribute-name typo fix. Qwen3-8B gsm8k 0.930→0.935, but throughput 2707→422 tok/s with offload |
| https://github.com/sgl-project/sglang/pull/29166 | 2026-07-01 | Dovis01 | Inline host-to-GPU copy during CUDA-graph capture |
| https://github.com/sgl-project/sglang/pull/38779 | 2026-09-21 | Hide-on-bushsh | Post-load staging regression from #35180 broke the meta and `sharded_gpu` modes |
| https://github.com/sgl-project/sglang/pull/40310 | 2026-09-21 | zRzRzRzRzRzRzR | Tied/shared parameter identity; keep `kv_b_proj` resident when the MLA `w_kc`/`w_vc` caches exist (`offloader.py:97-134`) |
| https://github.com/sgl-project/sglang/pull/41818 | 2026-09-30 | ch-wan | `num_dp_ranks` rename (adds `--attn-dp-size`) |

**Design and benchmark context:**
- The LMSYS GB200 Part II blog (2025-09-25, https://github.com/lm-sys/lm-sys.github.io/blob/main/blog/2025-09-25-gb200-part-2.md, repro https://github.com/sgl-project/sglang/issues/10903) says it scaled the prefill EP size down by offloading weights to host memory with prefetching, relying on GB200's 900 GB/s CPU-GPU link.
  - DeepSeek V3/R1, 4 ranks per prefill instance at FP8 and 2 at NVFP4.
  - No numbers are given for offload in isolation.
  - [inferred] OffloaderV2 was built for this; the PRs do not say so.
- No H20 benchmark for OffloaderV2 itself was found. The H20 runs are in the separate UVM PR #20126 below.

**Evidence of use on GLM:** issue https://github.com/sgl-project/sglang/issues/32507 (closed 2026-09-26).
- **GLM-5.2 FP8, 256 experts, EP16, DeepEP normal, `--offload-group-size 4 --offload-num-in-group 1 --offload-prefetch-step 1`.**
- Prefill warmup crashed with `unexpected N` in `copy_to_gpu_no_ce` (N=16).
- Fix PR https://github.com/sgl-project/sglang/pull/32769 is still OPEN. It applies to the CUDA DeepGEMM path only.

## 4. Open or closed-unmerged work

- **https://github.com/sgl-project/sglang/pull/24531** (closed as stale 2026-10-02, xiaobao520123): CUDA graph + torch.compile support, N-buffer pool, offload of all weights.
  - Claims more than 5x token throughput over the old V2 with CUDA graphs disabled.
  - Setup: 1×A100-40GB PCIe Gen4, DeepSeek-V2-Lite, `cpu` mode, group 4 / num 1 / prefetch 2.
  - It cites the vLLM design https://github.com/vllm-project/vllm/pull/29941.
  - Question it raised on #8034: "does OffloaderV2 only support dpsk v2 and only parts of it?" No answer.
- **https://github.com/sgl-project/sglang/pull/20769** (closed 2026-03-17): CUDA graph + `--offload-param-names`.
- **https://github.com/sgl-project/sglang/pull/20126** (closed as stale 2026-08-31, lichang98): **UVM MoE expert offload with all compute kept on GPU.**
  - **GLM-5-FP8 on 8×H20**, `--expert-offload-num-resident 200/210`, frequency-based selection of resident experts, docs included.
  - Not merged.
- **Open fixes:**
  - https://github.com/sgl-project/sglang/pull/43603 (public offloader factory/plugin hook; issue https://github.com/sgl-project/sglang/issues/43139)
  - https://github.com/sgl-project/sglang/pull/44139 (cached-weight residency for GDN conv)
  - https://github.com/sgl-project/sglang/pull/36509 (`tie_weights=False`)
  - https://github.com/sgl-project/sglang/pull/43086 (tests)
  - https://github.com/sgl-project/sglang/pull/43824 (NPU pinning)
- **Related but different mechanisms:**
  - SSD Expert Pack: router-selected expert streaming with a GPU cache, aimed at consumer GPUs. Blog: https://github.com/lm-sys/lm-sys.github.io/blob/main/blog/2026-08-29-sglang-ssd-expert-pack.md. Code is present in the image: `$S/srt/layers/moe/expert_pack.py`, `$S/srt/arg_groups/expert_pack_hook.py`.
  - KTransformers CPU experts: `$S/srt/layers/moe/kt_ep_wrapper.py`.
  - I did not check whether either works on ROCm.

## 5. ROCm/AMD, quantized MoE, DP attention, docs

- **ROCm/AMD:** not found. No issue or PR searches combining offload with ROCm/AMD/MI300 returned anything. No CUDA or ROCm functional CI test for V2 was found. The only mode test is `test/registered/npu/basic_function/offloading/test_npu_offload_modes.py`; the other references are argument unit tests (`test_startup_weight_load.py`, `test_server_args.py`).
- **Quantized MoE:** fp8 (#9948) and the nvfp4 blockscale whitelist exist. MXFP4/quark-specific handling or tests: not found.
- **DP attention:**
  - `sharded_gpu`/`shm_cpu` use DP ranks as their world but require TP=1.
  - `cpu` mode has no DP/TP assertion.
  - No upstream report of V2 together with DP-attention + TP>1 + PD was found.
- **Docs:** only the argument tables, with one-line descriptions:
  - `docs/docs/advanced_features/server_arguments.mdx:2309-2368`
  - Ascend's `support_features.mdx:1964` ("Offloading (must be used with `--disable-cuda-graph`)")
  - No design doc or guide found.

## 6. Points for the feasibility call (untested)

- **[inferred] Cost per offloaded layer:** `cpu` mode re-copies all of that layer's routed-expert weight shard on every forward, i.e. every prefill chunk. The cost scales with layers offloaded × per-layer expert bytes ÷ host-to-GPU bandwidth, against the per-layer compute time of a chunk. Neither the MI355X host-to-GPU bandwidth under 4 concurrent ranks nor GLM-5.2's per-layer expert bytes were measured here.
- **[inferred] Host memory:** each TP rank pins its own shard, so the extra pinned host memory is the offloaded expert bytes (all 4 TP shards together = one copy), alongside the 450 GB HiCache host pool.
- **To answer whether prefill throughput is lost:**
  1. Bring up the prefill leg with `--offload-group-size G --offload-num-in-group 1 --offload-prefetch-step 1`. Check that it starts on quark MXFP4 + aiter + ROCm, and record per-GPU weights and KV-pool size in the logs.
  2. Profile one 8192-token chunk: per-layer `start_onload` copy time vs MoE compute time.
  3. Measure single-rank and 4-rank-concurrent host-to-GPU bandwidth on MI355X.

=====VERIFY=====
**Verification of the OffloaderV2 report against `$S` (v0.5.21 image) and gh, 2026-10-14**

Overall: the report's code citations and PR facts check out. It misses two blockers that matter for our config. The first (B1 below) silently breaks quark MXFP4 + aiter as soon as offload is turned on.

## A. New, decision-relevant findings the report missed

**B1. The offloaded tensor loses the `is_shuffled` tag, so aiter will treat preshuffled MXFP4 weights as unshuffled.** Code read plus a local torch check; not run on GPU.
- `_CpuParamOffloader.create_device_tensor` returns `self._param.to("cuda", non_blocking=True)` (`offloader.py:429-430`). That tensor is passed into the forward through `functional_call` (`offloader.py:312-316`).
- `.to()` on a Parameter returns a plain `torch.Tensor` without Python attributes. Checked locally on torch 2.12.1: `hasattr(q, 'is_shuffled') == False`.
- The quark MXFP4 `apply_weights` tags the kernel inputs only when `hasattr(layer.w13_weight, "is_shuffled")` (`$S/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py:993-995`). Inside `functional_call`, `layer.w13_weight` is the untagged substitute.
- Upstream aiter chooses the kernel from `getattr(w1, "is_shuffled", False)`: `aiter/fused_moe.py:364`, `1153`, `1259-1260` and `1319-1320` on ROCm/aiter main. The aiter copy in our image was not checked.
- Meanwhile `process_weights_after_loading` did shuffle the stored data (`quark_w4a4_mxfp4_moe.py:846-867` or `:938-946`).
- Expected result: a wrong kernel family or layout (wrong data or a crash) and different performance. This is not covered by our correctness waiver, because it changes which kernel runs, so performance would not be comparable.
- Fix size: a few lines, e.g. copy the `is_shuffled` attribute onto the device tensor in `_CpuParamOffloader.create_device_tensor`, or tag it unconditionally in `apply_weights`.

**B2. The hook only fires through `self.experts(...)` (FusedMoE `forward`).**
- `forward_deferred_finalize` and `forward_impl` (`$S/srt/models/deepseek_v2.py:1153,1157`) bypass it. Both are gated on flashinfer-trtllm (`$S/srt/layers/moe/fused_moe_triton/layer.py:512-515`, `deepseek_v2.py:1062`).
- On our aiter path the normal call `self.experts(hidden_states, topk_output)` is taken (`deepseek_v2.py:1159-1163`). OK for us [inferred from the flags].
- Risk [inferred]: the hook assumes exactly one experts forward per layer per model forward, in ring order (`offloader.py:295-306`, assert at `:374`). Anything that calls experts twice per layer (TBO or two-chunk overlap) or skips a layer would trip it. Whether our prefill config enables TBO was not checked.

**B3. Cost-model numbers the report left open.** Computed from `/shared_nfs/lingzha/GLM-5.2-MXFP4/config.json`: hidden 6144, `moe_intermediate_size` 2048, 256 routed experts, 78 layers, `first_k_dense_replace=3`, so 75 MoE layers.
- **Per layer, per TP4 rank:** `w13`+`w2` = 256 × 3 × 6144 × 2048 × 0.5 B / 4 = 1.21 GB (1.125 GiB). This is exactly what moves; the scales stay on the GPU.
- **All 75 MoE layers:** about 84 GiB of weights plus about 5 GiB of scales per rank. That is consistent with the measured 119.84 GiB total weights.
- **KV cost per token:** 125.1 GiB / 2.43 M ≈ 55.3 KB.
- **To match the 128-expert point (+0.89 M tokens):** we need about 46 GiB freed, i.e. about 41 offloaded layers.
- **`G=2` (`num_in_group=1`):** offloads module indices `i%2==1`. That is 39 modules; one of them is dense layer 1, which has an empty whitelist (`deepseek_v2.py:2897-2918`). So 38 MoE layers ≈ 42.75 GiB, minus about 2 in-flight buffers of 1.125 GiB, ≈ +40 GiB ≈ +0.78 M tokens [inferred].
- **Hiding condition:** with `prefetch_step=1`, the next offloaded module's copy overlaps about G layers of compute. So 1.21 GB / H2D_BW ≤ G × t_layer(8192-token chunk). At a hypothetical 50 GB/s that is 24 ms per copy, which needs t_layer ≥ 12 ms. Neither BW nor t_layer has been measured. HiCache `write_through` H2D/D2H traffic shares the same link.
- **Pinned host memory for G=2:** about 38 × 5.1 GB ≈ 195 GB per node for the prefill leg, alongside the 450 GB HiCache pool [inferred].

**B4. Draft (MTP) worker.**
- `ModelRunner.__init__` calls `set_offloader(create_offloader())` for every runner, including the draft (`$S/srt/model_executor/model_runner.py:453`). The draft's instance replaces the global one.
- `post_init` is skipped for draft workers (`model_runner.py:1262-1263`, #23550).
- The NextN model has no `make_layers` or offloader wiring (no match in `deepseek_nextn.py`). So the MTP layer's experts stay resident.
- The target's hooks hold their own `offloaders` list, so I expect no functional problem [inferred].

## B. Tagged claims

1. **Arguments at `exec_.py:941-965`, with defaults -1/1/1/cpu:** CONFIRMED.
2. **Factory at `offloader.py:68-83`:** CONFIRMED. V2 also asserts `cpu_offload_gb==0` (line 74).
3. **Layer selection rule, whitelist-only, `get_stream("offload")`, `functional_call`:** CONFIRMED (`offloader.py:240-280, 309-321`).
4. **`tp_size==1` asserts at 223/439/544 for `shm_cpu`/`sharded_gpu`; `cpu` mode has no TP/DP assert:** CONFIRMED.
5. **Whitelist is `w13_weight`/`w2_weight` (+nvfp4 blockscales) on `layer.mlp.experts`; the MXFP4 scales stay on the GPU:** CONFIRMED (`deepseek_v2.py:2894-2918`).
6. **GLM-5.2 reuses the DeepSeek-V2 path:** CONFIRMED via `model_overrides/deepseek_v2.py:24`. The model class is registered in `glm4_moe.py` / `deepseek_weight_loader.py`; there is no `glm_moe_dsa.py`.
7. **Quark weights are created on the default device (`quark_w4a4_mxfp4_moe.py:233-236`):** CONFIRMED.
8. **Post-load "copies back into the original pinned buffer when the shape is unchanged": PARTLY WRONG.**
   - The in-place path needs the same `data_ptr` as well as the same shape and stride (`post_load.py:43-51, 89-96`).
   - The shuffle assigns a new tensor to `.data`, so the restore allocates a new pinned host tensor instead (`_copy_data_to_device` with `pin_memory=True`, `post_load.py:60-70, 98-102`; `loader.py:158-162`).
   - The net result, a pinned CPU tensor, is the same.
9. **`forbid_copy_engine_usage` is read only at `deep_gemm.py:1428`:** CONFIRMED. It is the only external reader of `get_offloader()` besides `model_runner` and `common.py`.
10. **The startup fast path is disabled by offload (`startup_weight_load.py:374-378`):** CONFIRMED, but irrelevant. The same gate already disables it for `num_dp_ranks != 1`, which is our case (DP4).
11. **CUDA-graph inline copy:** CONFIRMED (`offloader.py:358-362, 375-379`).
12. **PR table:** all dates, states and authors CONFIRMED with `gh pr view`.
    - #40310's actual title is "Support GLM-5.3-Flash hybrid attention CPU offload and PD index mapping". The report's description matches the code at `offloader.py:97-134`; the title was not given.
    - #25786: 2707.19 → 422.47 tok/s and gsm8k 0.930 → 0.935 CONFIRMED.
    - #24531: A100-PCIE-40GB, DeepSeek-V2-Lite, group 4 / num 1 / prefetch 2, ">5x", cites vllm#29941. CONFIRMED.
    - #32769 is OPEN; #32507 (GLM-5.2 FP8, EP16, DeepEP normal, `unexpected N`) was closed 2026-09-26. Both CONFIRMED.
13. **Only `cpu` mode is usable with TP4:** CONFIRMED by the asserts. "Usable" here means no assert fires; B1 still has to be fixed.
14. **No ROCm or MXFP4 upstream evidence; no CUDA/ROCm functional CI for V2:** UNVERIFIED. I did not repeat the searches.
15. **The CLI flag spelling (`--offload-group-size` etc.) maps to these msgspec fields:** CONFIRMED indirectly. #32507 uses those flags on a recent build.

## C. Before any experiment

1. Patch B1: carry `is_shuffled` onto the onloaded tensor.
2. Confirm TBO / two-chunk overlap is off on the prefill leg (B2).
3. Measure MI355X H2D bandwidth from pinned memory, single-rank and 4-rank concurrent with HiCache idle.
4. Measure per-layer forward time for one 8192-token chunk on the current prefill leg.
5. Plug those two numbers into the hiding condition in B3.

Key files:
- `$S/srt/utils/offloader.py`
- `$S/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`
- `$S/srt/models/deepseek_v2.py`
- `$S/srt/model_loader/post_load.py`
- `$S/srt/model_executor/model_runner.py`

where `S=/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang`.