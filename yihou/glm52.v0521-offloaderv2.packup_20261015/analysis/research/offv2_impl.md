# OffloaderV2 in sglang v0.5.21: what it is, how it works, and whether it accepts our GLM-5.2 prefill config

Source root is `S=/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang`, read-only on spur006. Anything marked [inferred] was not checked by reading code or running anything.

## 1. Answer in one paragraph

OffloaderV2 is in `$S/srt/utils/offloader.py`. It is a layer-group weight offloader with a ring prefetch, first merged upstream as "Overlapped weight offload" (https://github.com/sgl-project/sglang/pull/8034, fzyzcjy, merged 2025-08-23). For DeepSeek-V2-family models, which includes `GlmMoeDsaForCausalLM` (`srt/models/glm4_moe.py:1275`, a subclass of `DeepseekV2ForCausalLM`), it offloads only the routed-MoE weights `w13_weight` and `w2_weight` of selected layers to pinned host memory. Before each offloaded layer runs, it copies those weights back with an async H2D copy on its own stream. No code guard rejects our prefill config (TP4, DP4 with DP-attention, EP1, PD, HiCache, no prefill CUDA graphs) in mode `cpu`. Two things are unproven on our stack:
- quark MXFP4 / aiter preshuffled weights: there is a concrete `is_shuffled` attribute-loss risk (§5).
- whether H2D bandwidth can keep up with prefill compute (§7). This is the deciding question and nobody has measured it.

## 2. Flags (`srt/arg_groups/fields/exec_.py:940-965`, class `ExecOffload`, namespace `exec.offload`)

| field | default | help |
|---|---|---|
| `cpu_offload_gb` | 0 | V1: GBs of RAM to reserve |
| `offload_group_size` | -1 | "Number of layers per group" (> 0 enables V2) |
| `offload_num_in_group` | 1 | layers offloaded per group |
| `offload_prefetch_step` | 1 | how many offloaded layers ahead to prefetch |
| `offload_mode` | `"cpu"` | plain `str`, no choices list. Valid values come from the dict at `offloader.py:391-396`: `meta`, `cpu`, `shm_cpu`, `sharded_gpu` |

- **CLI spelling:**
  - `--offload-group-size` is confirmed by the error string at `memory_hook.py:36` and by PR #24531's example.
  - `--offload-num-in-group`, `--offload-prefetch-step` and `--offload-mode` follow the same naming convention [inferred]. Check with `--help` in the image.
- **Enabling it:** `--offload-group-size G --offload-num-in-group K [--offload-prefetch-step 1] [--offload-mode cpu]`, with `--cpu-offload-gb` left at 0.
- **Selection order** (`offloader.py:68-83`): V1 wins if `cpu_offload_gb > 0`. V2 asserts `cpu_offload_gb == 0` with the message "V2 offload does not support cpu_offload_gb yet". The offloader is created in `ModelRunner.__init__` (`model_executor/model_runner.py:453`, `set_offloader(create_offloader())`).

## 3. What gets offloaded

- **Layer choice:** `make_layers` (`utils/common.py:1585-1640`) passes the layer generator through `get_offloader().wrap_modules(...)` while the model is being constructed, before weights load.
- **Rule** (`offloader.py:257`): layer `i` is offloaded if `i % group_size >= group_size - num_in_group`. For example, G=2, K=1 offloads every odd layer.
- **What the DeepseekV2 wiring picks** (`models/deepseek_v2.py:2894-2916`):
  - submodule = `layer.mlp.experts` if the layer is a `DeepseekV2MoE`, else `layer.mlp`;
  - whitelist = `["w13_weight","w2_weight"]` (plus `w13_blockscale_swizzled`/`w2_blockscale_swizzled` for nvfp4) only if the module is a `FusedMoE`, else `[]`.
- **Consequences:**
  - Dense layers fall in the group pattern but offload nothing.
  - Attention, shared-expert (unless fused into the experts) and scale tensors (`w13_weight_scale`, `w2_weight_scale`) stay on the GPU.
  - With aiter shared-expert fusion, the fused shared expert lives inside `w13_weight`/`w2_weight`, so it is offloaded too [inferred from `num_fused_shared_experts`, `deepseek_v2.py:948`].
- **Scope:** only the target model's `make_pp_layers` goes through this path. The MTP/NextN layer is built directly as `DeepseekV2DecoderLayer(` at `models/deepseek_nextn.py:126`, so it is not offloaded. `model_runner.py:1262-1263` skips `post_init()` for draft workers (upstream PR #23550, "skip offloader post_init on draft worker", merged for the GLM-5.1 expert-offload path).

## 4. Prefetch mechanics (mode `cpu`)

- **Placement at construction:** `_CpuParamOffloader.__init__` calls `_move_param_to_cpu(param, pin_memory=True)` (`offloader.py:424-430`), so weights load straight into pinned host tensors.
- **Post-load processing:** `process_weights_after_loading` stages each module back on the GPU and then restores it to CPU, using pinned memory for any replaced data (`model_loader/post_load.py:109-230`, `loader.py:152-162`). The aiter shuffle therefore runs on GPU and the result ends up pinned on the host [inferred from reading the code, not run].
- **Stream:** a dedicated `get_stream("offload")` stream (`offloader.py:251`, `runtime_context.py:967`) creates a fresh `torch.cuda` stream, which is HIP on ROCm.
- **Schedule** (`offloader.py:295-321`): the forward of each offloaded submodule is replaced by `functional_call(module, device_tensors, ...)`.
  - At the end of the forward, `_on_forward_end` calls `offloaders[(index+prefetch_step) % N].start_onload()` and then `offloaders[index].offload()`. This is a ring over the offloaded layers only.
  - `post_init` pre-starts the first `prefetch_step` loads (`:283-288`).
- **Copy** (`:358-385`, `:430`):
  - `start_onload`: `alt_stream.wait_stream(current)`, then `param.to("cuda", non_blocking=True)` for each whitelisted param inside `torch.cuda.stream(alt_stream)`, then records an event.
  - `wait_and_get_device_tensors` waits on that event.
  - `offload()` simply drops the GPU tensors (`_device_tensors = None`) and they return to the caching allocator.
- **Memory:** there is no persistent staging buffer. Each forward re-allocates a full layer's MoE weights. Peak transient ≈ (prefetch_step+1) layers [inferred].
- **CUDA-graph capture:** the copy is done inline on the capture stream (`:359-362`, `:375-379`; upstream PR #29166, merged 2026-07-01).
- **`forbid_copy_engine_usage`:** true for mode `cpu` (`:290-292`). It is only consulted by `layers/moe/moe_runner/deep_gemm.py:1428`, which is CUDA-only.

## 5. Quantized MXFP4 (quark, aiter) compatibility

- **Hook path:** the offloader hooks `experts.forward`. On our path DeepseekV2MoE calls `self.experts(hidden_states, topk_output)` (`deepseek_v2.py:1159-1163`), so the hook applies. The flashinfer `forward_impl` / `forward_deferred_finalize` paths at `:1153`/`:1157` would bypass it, but they are not used on ROCm.
- **Shuffle step:** quark MoE `process_weights_after_loading` (`layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py:869-946`) rewrites `.data` in place. For silu on gfx950 that is `shuffle_weight(...,(16,16))` at `:938-946`, plus `layer.w13_weight.is_shuffled = True`.
- **Risk:** `apply_weights` (`:993-995`) does `if hasattr(layer.w13_weight, "is_shuffled"): w13_weight.is_shuffled = True`.
  - Under `functional_call`, `layer.w13_weight` is the tensor returned by `Parameter.to("cuda")`, which is a new tensor that does not carry the Python attribute [inferred].
  - So `is_shuffled` would be lost and aiter `fused_moe` may pick the non-preshuffled kernel family.
  - `is_shuffled` is not read anywhere in sglang `srt`; it is consumed inside aiter (not read).
  - Fix size: a few lines, for example copying the attribute in `create_device_tensor` [inferred]. A one-layer smoke test would show it.
- **Scales:** the scale params stay resident on GPU. That is fine, because `functional_call` only substitutes the two whitelisted names.

## 6. Guards and asserts that apply to our config (quoted)

- `offloader.py:74-76`: `assert get_exec().offload.cpu_offload_gb == 0, "V2 offload does not support cpu_offload_gb yet"`. Not hit.
- `offloader.py:222-223` (modes `sharded_gpu`/`shm_cpu` only): `assert get_parallel().tp_size == 1, "not yet support tp_size!=1"`. Repeated at `:439` and `:544`. **This rejects those two modes at TP4. Mode `cpu` has no such assert.**
- `offloader.py:219`: `run_id = os.environ["SGLANG_RUN_ID"]`. It is set by `entrypoints/engine.py:1865` in `_set_envs_and_config`, so not a problem under `launch_server`.
- `offloader.py:337-339`: `assert self.device != torch.device("cpu"), "not handled device=cpu case yet ..."`. Not hit on GPU.
- `offloader.py:345-347`: whitelist names must exist in `named_parameters()`. They do for quark FusedMoE (registered at `quark_w4a4_mxfp4_moe.py:274,295`).
- `offloader.py:246`: `assert len(self.offloaders) == 0, "should only call wrap_modules once"`.
- `arg_groups/memory_hook.py:31-38`: raises if `--ple-offload-embedding` is combined with `--cpu-offload-gb or --offload-group-size`. Not relevant to us.
- `model_executor/model_runner_components/startup_weight_load.py:374-378`: startup weight-load overlap is refused with "CPU offload is not supported" / "layer-group offloading is not supported". It only disables that optional startup feature.
- `arg_groups/cuda_graph_hook.py:235-237`: CUDA graphs are disabled for `cpu_offload_gb > 0 or enable_hierarchical_cache`. V2's `offload_group_size` is not in that list, so V2 does not force-disable graphs. Our prefill has no graphs anyway, and PD at `:242` already disables them.
- **Not found:** any guard on DP-attention, EP, PD disaggregation, torch.compile, ROCm/HIP, or `disaggregation_mode` for V2.
- **Not found:** any ROCm-specific code in `offloader.py`. It is generic `torch.cuda`/pinned memory.

## 7. Older and alternative mechanisms

- **OffloaderV1** (`--cpu-offload-gb N`, `offloader.py:137-201`):
  - Per-parameter greedy fill of whole decoder layers (attention included) until N GB is reached.
  - On every forward it does a synchronous `functional_call` with `.to(device, non_blocking=True)` of the whole module state, with no prefetch or overlap.
  - Forces CUDA graphs off (`cuda_graph_hook.py:236`).
  - Unsuitable for keeping prefill throughput [inferred].
- **KTransformers CPU experts** (`kt_weight_path`, `kt_method` default `"AMXINT4"`, `kt_cpuinfer`, `kt_threadpool_count`, `kt_num_gpu_experts`, `kt_max_deferred_experts_per_token`; `exec_.py:893-916`, `layers/moe/kt_ep_wrapper.py`):
  - Experts are computed on the CPU (AMX), which is different from weight streaming.
  - It needs separately quantized CPU weights.
  - Not checked on ROCm or EPYC.
- **Upstream activity:**
  - #24531 (closed, unmerged): OffloaderV2 with a CUDA-graph N-buffer pool and all-weights offload.
  - #43603 (open): public offloader factory and post-load hook.
  - #44139 (open): preserve cached weights.
  - #36509 (open): tie_weights in `functional_call`.
  - #23550 (merged): Z.ai used V2 for GLM-5.1 "expert-offload" on CUDA, which needed `copy_to_gpu_no_ce` N=32.
  - Issue #32507 (closed): V2 with DeepEP, `copy_to_gpu_no_ce` sizes; CUDA-only.
  - Not found: any ROCm or MI3xx report of V2 use.

## 8. Feasibility arithmetic for our case [inferred, unmeasured]

- **Per-layer size:** about 1.2-1.3 GiB of MoE weight per GPU per layer, from 119.84 GiB of weights per GPU spread over roughly 75 MoE layers plus attention. The exact `w13`+`w2` bytes per layer should be read from the `[offloader] offload module_index=... memory_allocated=` log line at `offloader.py:261`.
- **Layers needed:** reaching the 128-expert KV (3.32 M tokens, +0.89 M) at about 51.5 KiB/token (125.1 GiB / 2.43 M) needs about 46 GiB freed, i.e. roughly 36-38 offloaded layers (G=2, K=1).
- **Copy cost:** about 46 GiB of H2D per GPU per forward chunk. At an assumed ~50 GB/s PCIe Gen5 x16 that is about 0.9-1 s per forward.
- **What has to hold:** with G=2 and prefetch_step=1, each layer's copy must hide under about 2 layers of compute. So no throughput loss requires about ≥12 ms of forward time per layer per 8192-token chunk.
- **Contention:** HiCache write_through D2H uses the same PCIe link. DP-attention idle ranks still run the MoE forward, so they still pay the copy.
- **Measurements needed:**
  - per-chunk prefill forward time from the prefill leg's logs or a profile;
  - real H2D GB/s from pinned memory per MI355X while all 4 prefill GPUs copy at once;
  - a smoke run with `--offload-group-size 2 --offload-num-in-group 1` to check the `is_shuffled` risk and the resulting `max_total_num_tokens`.
- **Host memory:** about 4 × 46 GiB pinned, plus the 450 GB HiCache pool, on a 3 TB host. It fits [inferred]. Pinned-allocation time at startup has not been measured.

=====VERIFY=====
# Adversarial verification: OffloaderV2 report (sglang v0.5.21 image source)

All paths are under `$S/srt` on spur006 (read-only). `[inferred]` means I did not run it or read it in code.

## Bottom line

- No code guard rejects our config in mode `cpu`. I re-read the code paths and this holds.
- The `is_shuffled` risk is real, and the evidence is stronger than the report says. The fix is mandatory before any run.
- The cost model's window is too optimistic. The prefetch for an offloaded layer cannot start until the previous offloaded layer's MoE has finished on the GPU, so the copy has to hide under roughly 1.5 layers of compute, not 2.
- The two expert-reduction data points disagree about how big the routed-expert weights are. The "~46 GiB / 36-38 layers" figure is therefore not solid until the real per-layer size is read from the logs.

## Claim checks

### 1. Location, upstream PR #8034, class `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` at `glm4_moe.py:1275` — CONFIRMED
- PR #8034 "Overlapped weight offload", fzyzcjy, merged 2025-08-23T09:06Z.

### 2. Flags — CONFIRMED
- `exec_.py:948-965`: the defaults are 0, -1, 1, 1 and `"cpu"`, and `offload_mode` is a plain `str`.
- The mode dict is at `offloader.py:391-396`.
- Selection order is at `offloader.py:68-83`.
- The offloader is created at `model_runner.py:453`.
- The CLI spelling of the other three flags is still UNVERIFIED. Run `--help` in the image.

### 3. Offload rule and wiring — CONFIRMED
- Rule at `offloader.py:257`.
- Wiring at `deepseek_v2.py:2894-2916`.
- `make_layers` wraps the layer generator at `utils/common.py:1629`.
- The NextN layer is built directly at `deepseek_nextn.py:126`.
- The draft-worker skip is at `model_runner.py:1262-1263`.
- PR #23550: merged 2026-05-09. Its body says it was needed by "the GLM-5.1 expert-offload path". That it ran on CUDA is [inferred] from the sgl-kernel `copy.cu` change.

### 4. Prefetch mechanics — CONFIRMED, with one fact the cost model misses
- `start_onload` (`offloader.py:365`) does `alt_stream.wait_stream(torch.cuda.current_stream())` before the H2D copy.
- `_on_forward_end` (`:296-298`) calls this right after the module's `functional_call`, so the copy waits for everything already queued on the compute stream, including this layer's MoE GEMMs.
- Result: with G=2, K=1, step=1, the copy for layer i+2 can only start after layer i's MoE has finished. It must then finish before layer i+2's MoE starts, so the window is about one layer of compute plus one attention block, not two full layers [inferred from code read].
- Raising `prefetch_step` widens the window, but costs one more layer of transient GPU memory each step.
- `get_stream` is at `srt/runtime_context.py:967`, not under `utils/`. It creates a new `torch.get_device_module(device).Stream()`.

### 5. Post-load staging — CONFIRMED
- `DefaultModelLoader.postprocess_weights` (`loader.py:1084-1085`) wraps every quant module in `device_loading_context`, which stages it onto the GPU.
- The restore step (`post_load.py:81-105`, `:59-78`) puts each tensor back on its original device, which is CPU here, as freshly pinned memory.
- Quark replaces `layer.w13_weight` with a new Parameter in some branches (`quark_w4a4_mxfp4_moe.py:576`, `:780`, `:821`). The slot key still maps it back to CPU, and the offloader looks up `_param` by name with `getattr`, so it follows the new object.
- So the shuffled weights should end up pinned on the host, and the `is_shuffled` attribute survives on the host Parameter [inferred, not run].
- Side effect [inferred]: during load, each MoE module is staged onto the GPU one at a time, so peak load-time memory goes up by about one layer.

### 6. `is_shuffled` risk — CONFIRMED, and stronger than stated
- Experiment (local torch 2.12): `Parameter(...).to(<other device or dtype>)` returns a plain `torch.Tensor` with `hasattr(r, 'is_shuffled') == False`. So `_CpuParamOffloader.create_device_tensor` (`offloader.py:430`) drops the tag.
- `quark_w4a4_mxfp4_moe.py:993-995` only re-tags the weights `if hasattr(layer.w13_weight, "is_shuffled")`. Under `functional_call` this is False, so the tag is never set.
- WRONG: the report says "`is_shuffled` is not read anywhere in sglang srt". It is read at `fp8.py:3318`, `mxfp4.py:1761`, `weight_checker.py:351`. None of these is on our quark path.
- The more important new evidence:
  - `mxfp4.py:1758-1760` says: "`.view()` creates a fresh tensor that drops the `is_shuffled` marker ... Re-tag it so the downstream aiter.fused_moe selects preshuffle_on kernels."
  - `quark_w8a8_fp8_moe.py:240` says aiter.fused_moe "selects on via the is_shuffled tag".
  - So the tag does pick the aiter kernel. Without a fix, preshuffled data would go to the non-preshuffle kernel family. The aiter source is not in the extracted image, so this is UNVERIFIED in aiter itself.
- Fix: copy the attribute in `create_device_tensor`, or change the hasattr check in `apply_weights`.
- This is not covered by "correctness waived". That waiver covers the forced acceptance count, not garbage MoE output, and the wrong kernel family may also run at different speed [inferred].

### 7. Hook coverage — CONFIRMED, with an addition
- The hook covers `self.experts(...)` at `deepseek_v2.py:1159`, `:1163`, `:1344`, `:1350`, `:1416`, `:1666`.
- ADD: the two-batch-overlap op path `op_experts` (`deepseek_v2.py:1935-1937`) calls `self.experts.run_moe_core(...)` directly, which bypasses the hooked `forward`. With offload on, those weights would never be loaded onto the GPU [inferred].
- So TBO must stay off on the prefill leg. I have not checked the launch args to confirm it is off.
- The flashinfer paths at `:1153` and `:1157` are CUDA-only, as the report says.

### 8. Guards — CONFIRMED
- Quoted guards: `offloader.py:74-76`, `:222-223`, `:219`, `:337-339`, `:345-347`, `:246`, `memory_hook.py:31-38`, `cuda_graph_hook.py:235-237` and `:242`.
- `startup_weight_load.py:370-378`: our DP4 already disqualifies startup overlap through `options.num_dp_ranks != 1` (`:371`), so the offload rows add nothing.
- `SGLANG_RUN_ID` is set at `entrypoints/engine.py:1865`.
- `get_offloader()` has only three external callers: `deep_gemm.py:1428` (CUDA-only), `model_runner.py:1263` and `common.py:1629`.
- Not found: any DP-attention, EP, PD, HiCache or ROCm guard. I re-ran the grep.

### 9. Upstream PR and issue facts — CONFIRMED
- #29166: merged 2026-07-01.
- #24531: closed, unmerged.
- #43603, #44139, #36509: open.
- Issue #32507: closed, DeepEP/`copy_to_gpu_no_ce`.
- ADD: open issue #43139 "Let a general plugin provide the weight offloader, with a post-load step", which goes with #43603.
- `gh search prs "offloader rocm"`: nothing found. A search for "offload-group-size" issues found no ROCm or MI3xx report.

### 10. Cost model — PARTLY WRONG / UNVERIFIED
- Per-token KV: 125.1 GiB / 2.43 M = 54.0 KiB if M = 10^6, or 51.5 KiB if M = 2^20. The report's 51.5 KiB implies M = 2^20. Either way, +0.89 M tokens needs about 45.8 GiB. CONFIRMED arithmetically.
- WRONG / unflagged inconsistency between the two expert-reduction points:
  - 128 experts (half the routed experts removed) gave +0.89 M tokens, about 46 GiB. That implies the routed experts total about 92 GiB.
  - 8 experts (31/32 removed) gave only +1.24 M tokens, about 64 GiB. That implies a total of about 66 GiB.
  - These cannot both hold. Either something else capped the 8-expert pool, or the synthetic reduction changed more than the routed weights. I don't know which.
  - So "36-38 layers" is unsupported. Read the real per-layer bytes from the `memory_allocated=` lines that `offloader.py:261` logs, or from the weight-load log.
- Per-layer copy time: about 1.24 GiB at an assumed 50 GB/s is about 27 ms. With the roughly 1.5-layer window from item 4, each layer needs about ≥18 ms of compute per 8192-token chunk, not ≥12 ms [inferred].
- The PCIe Gen5 bandwidth, per-GPU link topology (whether GPUs share a switch) and concurrent 4-GPU H2D throughput on MI355X are all UNVERIFIED and unmeasured.
- The pinned-host-memory fit is plausible. That HiCache also pins its 450 GB pool is UNVERIFIED.

## Missing facts to add
- No persistent device buffer: each forward re-allocates through the caching allocator on the offload stream. That stream waits on the compute stream, so reuse is ordered [inferred]. Allocator fragmentation interacting with the KV pool sizing is unmeasured.
- `max_total_num_tokens` is fixed at startup after the weights load, so freed MoE memory goes to the KV pool only if the offloaded params are already on the host when memory profiling runs. That should be true, because the params move at construction (`offloader.py:424-427`), but I have not checked it against the profiling code.

## Measurements that decide it
1. Smoke run with `--offload-group-size 2 --offload-num-in-group 1`, with and without the `is_shuffled` patch. Read the per-layer bytes and `max_total_num_tokens` from it.
2. Pinned H2D GB/s with all 4 prefill GPUs copying at once.
3. Per-layer prefill forward time at chunk 8192.