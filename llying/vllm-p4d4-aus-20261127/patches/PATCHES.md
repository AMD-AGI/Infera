# vLLM patch set for GLM-5.2 P4D4 on nightly `de7cf7f8` (2026-11-27)

Base: `vllm/vllm-openai-rocm:nightly-rocm100-de7cf7f8cb5dc47d2e65207e0fabbe2c8f0d73b3`, i.e. vLLM main
`de7cf7f8cb5dc47d2e65207e0fabbe2c8f0d73b3`. Checked in the worktree
`/perf_apps/liyingli/bench_agentx/upstream/vllm-nightly-de7cf7f8` (detached at that commit, left clean).

Rule: each patch is byte-identical to `gh pr diff <N>` at the PR head listed below. `vllm/*.diff` files are
the raw `gh pr diff` output; `meta/<N>.json` / `meta/<N>.diff` hold the `gh pr view` metadata and the same
diff. `vllm/proposed-*.diff` is **not a PR yet** (draft content for a new fork PR).
PR heads were read on 2026-11-27 (UTC 15:40). If a PR is force-pushed, fetch it again and re-check.

"Applies" = `git apply --check` on de7cf7f8. The stacking test (all patches in the order below, applied in
sequence in the worktree, then reset) passed with the `--exclude` / `--3way` flags noted.

## Table

| PR | Title | Author | State | Head sha | Files (non-test) | Applies to de7cf7f8 | Needed? | Notes |
|---|---|---|---|---|---|---|---|---|
| [#61985](https://github.com/vllm-project/vllm/pull/61985) | [ROCm][Perf] MXFP4 indexer K cache for GLM-5 / DeepSeek-V3.2 DSA | whx-sjtu | open | a7ec12ed965d | models/deepseek_v32/amd/rocm.py, models/deepseek_v32/common/kernels.py, v1/attention/backends/mla/indexer.py, v1/attention/backends/mla/rocm_paged_mxfp4_indexer.py | All source files apply. **Only the test** `tests/v1/attention/test_rocm_paged_mxfp4_indexer_plan.py` fails (hunk @290: the test it appends after `test_decode_row_shard_skips_steps_with_prefill` now has different trailing lines on main, from the ShardQ decode-row work in #67363/#68074). Apply with `--exclude=tests/v1/attention/test_rocm_paged_mxfp4_indexer_plan.py` | Yes (not in main) | Turns MXFP4 on for the **MTP draft too**: `DeepseekV32MTPModel` is listed in `_ROCM_MXFP4_V32_ARCHS`, and `DeepseekV32ROCmIndexerCache` decides through the global `dsa_indexer_uses_fp4(vllm_config)`. You cannot exclude the draft through config. A supplementary draft PR is needed (see item 1). GLM-5.2: `num_hidden_layers=78`, `num_nextn_predict_layers=1`, so the NextN layer is `model.layers.78`. |
| (new, fork) | Keep the MTP draft's DSA indexer K cache FP8 when the target uses MXFP4 | ours | to open | - | models/deepseek_v32/amd/rocm.py (+test) | stacks on #61985 | Yes | See item 1. Logic already exists in our v0.34 backport (`vllm-pd-opt-20261123/scripts/make_idxfp4_overlay.py`, "MTP draft excluded"). |
| [#69540](https://github.com/vllm-project/vllm/pull/69540) | [KV Offload] Remove O(n) heap compaction stalls in CPU LRU policy | dorado269 | open | f49666688523 | v1/kv_offload/cpu/policies/lru.py | Clean | Yes | Head moved since our overlay: f1793481 (vendored in `overlay-kvoffload/lru-pr69540-f1793481.py`) no longer exists. The new head has 2 more commits that add `_EvictionIndex` (an O(1) tail plus sorted sublists) and make non-positive evictions no-ops. Use the current head; our vendored file is stale. |
| [#68845](https://github.com/vllm-project/vllm/pull/68845) | [Perf][KV Offload] Coalesce adjacent hit source ranges | Alex-ai-future | draft | 665b9aaed4a2 | distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py | Clean | Yes | Code hunk is identical to our string edit (`COALESCE` in `make_kvoffload_overlay.py`). No overlap with #69540 (different files). |
| [#68560](https://github.com/vllm-project/vllm/pull/68560) | [Core][PP] Chunk a lone long prefill under pipeline parallelism | ChengYM5685 | open | 69aa66d84f8c | config/scheduler.py, v1/core/sched/scheduler.py | scheduler.py and the test apply. `vllm/config/scheduler.py` @107 fails: docstring-only. Its trailing context is `max_num_queued_reqs`, but on main `pp_microbatch_min_tokens` (from #68614) now follows | Note only (not in a49) | Needs a trivial rebase by the author before it can be used verbatim. |
| [#68614](https://github.com/vllm-project/vllm/pull/68614) | [Core][PP] Admit new requests past idle empty steps, and size prefill microbatches from the backlog | njhill | merged 11-25 | merge 249e6529 | - | In de7cf7f8 (ancestor) | No (in main) | - |
| [#66799](https://github.com/vllm-project/vllm/pull/66799) | [Perf][MoE] Mask DP padding rows in grouped top-k routing by default | Pb314314 | merged 11-14 | merge fee96868 | - | In de7cf7f8 | No (in main) | **No effect on our ROCm AITER path.** GLM-5.2 has `n_group=1, topk_group=1` + bias, so `router_factory` degenerates to `FusedTopKBiasRouter`, whose `masks_padding` is `not rocm_aiter_ops.is_fused_moe_enabled()` = False. `GroupedTopKRouter.masks_padding` is also False on AITER unless the model opts in with `skip_padding` under DeepEP v2. Separately, under naive DP dispatch `get_moe_padding_mask` returns None because the router sees all-gathered rows. Padding rows are still routed. |
| [#67656](https://github.com/vllm-project/vllm/pull/67656) | [Perf][KV Connector][MooncakeConnector] Submit Mooncake transfer descriptors as arrays | zhou-yuhan | merged 11-16 | merge 919025c8 | - | In de7cf7f8 | No (in main) | Needs `batch_transfer_sync_write_arrays` (mooncake-transfer-engine >= 0.3.14). The image pins `mooncake-transfer-engine-rocm==0.3.14` (see item 13). |
| [#69616](https://github.com/vllm-project/vllm/pull/69616) | [ROCm][Bugfix][MoE] Keep fused shared experts computed once per token under DP (U17) | llying-001 | open | d9dd5161c9a9 | fused_moe/experts/rocm_aiter_moe.py, fused_moe/prepare_finalize/naive_dp_ep.py | Clean | Yes | Adds `_localize_shared_expert_ids` before the modular `prepare()` return. **Semantic interaction with the FP4-dispatch pack port** (item 12): its `return packed` early exit would skip the localize. The port must apply it too. |
| [#69617](https://github.com/vllm-project/vllm/pull/69617) | [ROCm][KV Offload] Register only each worker's slots of a shared offload region (U19) | llying-001 | open | 8bf00c65f51c | v1/kv_offload/cpu/gpu_worker.py | Clean | Only if the shared mmap region is used on ROCm (i.e. with #69254) | On main, ROCm always uses private per-tensor buffers (#57160), so the shared region path that #69617 fixes runs only with #69254 or on CUDA. |
| [#67232](https://github.com/vllm-project/vllm/pull/67232) | [Bugfix][ROCm] Fix fp8 NoPE NaN, fp16 crash and a decode-path sync in ROCM_AITER_MLA_SPARSE | tjtanaa | open | b22ebda8c2e2 | backends/mla/rocm_aiter_mla_sparse.py, ops/rocm_aiter_mla_sparse.py | **No**: backend file @467 and ops file @1449 conflict (3-way: conflicts), and the test @315 conflicts | No | Its "decode-path sync" is the `torch.cuda.current_stream().synchronize()` after `get_mla_metadata_v1`, which **main already removed**. Its ops hunk (`skip_kv_gather`) is also already in main. The rest (NoPE fp8 / fp16 guards) does not affect GLM-5.2 (bf16, has RoPE). It does **not** touch the `seq_lens` D2H sync. |
| [#58710](https://github.com/vllm-project/vllm/pull/58710) | [ROCm][Perf] Skip the seq_lens D2H sync in the AITER sparse MLA metadata builder | NickLucche | open (stalled, base 10-25) | 6331e45c1fe4 | backends/mla/rocm_aiter_mla_sparse.py | Direct apply fails (backend @457 context). `--3way` applies the **source cleanly**; only the test `test_rocm_aiter_mla_sparse_metadata_sync.py` @1 conflicts | Yes (sync still on main, line 914-915) | Covers exactly the remaining half of our sync-skip overlay: it reads `seq_lens_cpu_upper_bound`, with slack `2*num_spec*max_concurrent_batches` where ours is a fixed 32. Our other half (the post-metadata synchronize) is already gone on main. You need a verbatim-applicable diff: author rebase, or a supplementary rebased draft PR on the fork (item 7). |
| [#68596](https://github.com/vllm-project/vllm/pull/68596) | [ROCm][Perf] Add gfx950 gluon paged-MQA logits kernel for DSA indexer | RuibinCheung | open | 801cbb6820d3 | ops/rocm_aiter_mla_sparse.py, ops/rocm_gluon_paged_mqa_logits.py (new) | Clean (also stacks after #61985 and #58710) | Not with #61985 (see item 8) | Same lineage as our U12 port (aiter gluon kernel, pairwise sibling-row scoring, LPT, XCD swizzle). Reduced to 673 lines (ours: 2026). Supports 16/32/64-token pages. **No env var**: it is on automatically for FP8-cache MTP verify rows (`rows_per_request >= 2`) on gfx950. With MXFP4 target indexers, verify rows no longer reach `rocm_fp8_paged_mqa_logits`, and draft rows have `rows_per_request=1`, so the kernel never runs. |
| [#68006](https://github.com/vllm-project/vllm/pull/68006) | [Refactor] Use `FULL` instead of `NONE` during MR V2 full-graph capture | yewentao256 | merged 11-20 | merge 1934f186 | - | In de7cf7f8 | No (in main) | Fixes the "mode NONE during FULL capture" half of U13 (see item 9). |
| proposed-U13 (`vllm/proposed-U13-aiter-agrs-full-graph.diff`) | Let AITER AG/RS run in FULL graphs without a uniform batch descriptor | ours | **not a PR** | - | distributed/device_communicators/cuda_communicator.py | Clean | Yes | On main, `_can_use_aiter_ag_rs` still requires `ctx.batch_descriptor.uniform`. FULL capture passes `batch_descriptor=None` (`cudagraph_utils.py:839`), FULL replay sets no context, and nothing in the tree sets `BatchDescriptor.uniform=True`. So the AITER AG/RS path never runs. The diff drops that check (`sizes is None` already guarantees equal tokens per rank). |
| [llying-001#1](https://github.com/llying-001/vllm/pull/1) | [ROCm][Perf] Let DP MoE dispatch/combine use AITER custom AG/RS inside FULL CUDA-graph capture | llying-001 | open (fork) | f07e10429614 | cuda_communicator.py | Clean | Superseded by proposed-U13 | Its extra `NONE + is_current_stream_capturing()` branch is dead after #68006. Its test still covers the gate. Suggest updating fork #1 to the proposed-U13 content and keeping a reduced test. |
| [#60909](https://github.com/vllm-project/vllm/pull/60909) | [Bugfix] Pin long-lived host buffers at their exact size | lobo235 | open | eafac2f29528 | distributed/device_communicators/cuda_wrapper.py, model_loader/utils.py, offloader/prefetch.py, utils/pinned_memory.py (new) | `model_executor/model_loader/utils.py` @38 fails: import context only (main split the `vllm.model_executor.utils` / `torch_utils` imports into multi-line). The other files apply | Prerequisite for U26 | **The KV-offload half was dropped** in the last commit ("Drop the KV-offload half: #69254 answers it"). As it stands, it does not change the CPU KV tier. Its earlier commit 15b5108c4 had the `gpu_worker.py` change (`empty_pinned(shape, torch.int8).zero_()`). |
| [#69254](https://github.com/vllm-project/vllm/pull/69254) | [Bugfix][ROCm][KV Offload] Check the CPU offload region against the amdkfd system memory limit | yuzhouo7 | open | a546f48b2346 | v1/kv_offload/cpu/spec.py, v1/kv_offload/tiering/spec.py | `cpu/spec.py` hunk @211 fails, and `docs/features/kv_offloading_usage.md` fails. tiering/spec.py and the test apply | Optional / alternative | On ROCm it returns to the shared mmap region (hipHostRegister) when it fits the amdkfd limit. Our U26 deliberately used hipHostAlloc instead of registered user memory, which avoids KFD evict/revalidate (see `kfd_evict_summary.py`). Needs an author rebase. |
| [llying-001#7](https://github.com/llying-001/vllm/pull/7) | [Core][DP] Optional KV cache usage weight in the internal DP load balancer | llying-001 | draft (fork) | 941527ab86ac | config/parallel.py, engine/arg_utils.py, v1/engine/core_client.py | Clean | Yes | Gets the KV capacity from the engine: `cache_config.kv_cache_size_tokens`, which the client copies per engine from the handshake response (`core_client.py:925`, set by `update_kv_cache_capacity`). `VPD_DP_LB_KV_CAP_TOKENS` is no longer needed. Score and increment are identical to `make_dplb_kv_overlay.py`. To use it verbatim, open it against vllm-project/vllm. |

## Items with no upstream PR (need new draft PRs)

| Item | Generator(s) | Does the generator still apply on de7cf7f8? | Changed anchors / semantics | Porting effort |
|---|---|---|---|---|
| FP4 dispatch (`VPD_FP4_DISPATCH`) | `make_fp4dispatch_overlay.py` | Yes: all anchors match once (naive_dp_ep.py `_quantize_and_setup_dispatch(...)` + LoRA comment, the modular `return a1q, a1q_scale, None, topk_ids, topk_weights`, rocm_aiter_moe.py `packed_fp4_input = (...use_mori_epv1_kernels...)`). It also applies on top of #69616 | None textually. `dp_metadata.get_chunk_sizes_across_dp_rank`, `AiterExperts.workspace_shapes` K*2 logic and the MoRI path are unchanged | Small (about 1 day): turn the env switch into config, drop `_vpd_` names, add a test |
| FP4 dispatch pack, one all-gather (`VPD_FP4_DISPATCH_PACK`) | `make_fp4dispatch_pack_overlay.py` | Yes, including on top of #69616 | `AgRsAll2AllManager._get_comm_group/_get_sizes`, `get_ep_group()`, `self._comm_region()` still exist. **Semantic:** with #69616 the packed early return must also call `_localize_shared_expert_ids(topk_ids)` | Medium (2-3 days): Triton pack/unpack kernels, PR-quality tests, graph-capture validation |
| Triton decode top-k v3 (`VPD_TRITON_TOPK`) | `make_tritontopk_overlay.py` + `vpd_triton_topk_decode.py` | Yes on `ops/rocm_aiter_mla_sparse.py` (`HEAD_OLD`, `CALL_OLD` match once, also after #68596) | `rocm_aiter_sparse_attn_indexer` still has `k_cache_prefix`, `compress_ratio`, `decode_metadata.seq_lens`, `dsv4_indexer_prefers_native_top_k`. **But with #61985 the target layers' decode top-k moves to `ops/rocm_paged_mxfp4_indexer.py::_dense_decode -> _topk`**, so this hook would only reach the FP8 MTP draft layer. Our idxfp4 overlay's extra hook does not apply cleanly: `OPS_1..3` match, but `OPS_4` does not. Main now calls `_dense_decode(_decode_rank_rows(layer), max_model_len, candidate_write)` then `_exchange_decode_rows(layer)` (ShardQ decode-row sharding), and `_topk` gained a `step_rows` argument | Medium (3-4 days): the kernel file (688 lines) plus hooks in both top-k sites; respect `step_rows` and ShardQ row exchange |

## Item notes

1. **#61985 and the MTP draft.** The PR applies to the MTP draft. `DeepseekV32MTPModel` is in the allowed
   arch set, and `DeepseekV32ROCmIndexerCache.__init__` uses only the global `dsa_indexer_uses_fp4()`.
   No config excludes the draft. Minimal supplementary change, about 25 lines plus a test, on
   `vllm/models/deepseek_v32/amd/rocm.py`. In `DeepseekV32ROCmIndexerCache.__init__`, take
   `mxfp4 = dsa_indexer_uses_fp4(vllm_config) and not is_draft`. Here
   `is_draft = "DeepseekV32MTPModel" in model_config.architectures or extract_layer_index(prefix) >= hf_text_config.num_hidden_layers`
   (78 for GLM-5.2). An opt-in such as an attention-config field (e.g. `indexer_kv_dtype_draft`, default
   follows the PR) keeps the PR's behaviour available. Everything downstream already follows this per-layer
   flag: `self._indexer_mxfp4`, the `SparseAttnIndexer` vs `RocmSparseAttnIndexer` choice, `index_q_mxfp4`
   in `fused_q`, and `indexer_cache_mxfp4` in `fused_norm_rope`. The KV spec head size also follows it
   (68 B vs 132 B per token). The only global left is `DeepseekV32IndexerMetadataBuilder.indexer_uses_fp4`
   (indexer.py:1059). On ROCm it only disables `fused_mqa_topk` prefill chunk planning and changes a log
   line, so it is harmless for the FP8 draft group.
2. #69540 + #68845: no conflict (different files); both clean.
7. **Host syncs on de7cf7f8.** `rocm_aiter_mla_sparse.py:914-915` still does
   `seq_lens[:num_reqs].cpu()` under `gpu_sync_allowed()`. The stream synchronize after
   `get_mla_metadata_v1` is gone. #58710 covers exactly the remaining sync. #67232 does not.
9. **U13.** #68006 changed MRV2 FULL capture to `cudagraph_runtime_mode=FULL`, so the mode check passes
   during capture. The `batch_descriptor.uniform` requirement remains and is never satisfied (see table).
   Remaining diff: `vllm/proposed-U13-aiter-agrs-full-graph.diff`.
10. **U26.** On main ROCm, the CPU tier uses private `torch.zeros(pin_memory=True)` tensors
    (`gpu_worker.py:880`), so the power-of-two rounding bug is back. Recommended: #60909 (helper; needs an
    author rebase for the one-line import conflict) plus a supplementary fork draft PR that restores its dropped
    `gpu_worker.py` hunk (from commit 15b5108c4: `empty_pinned(shape, torch.int8).zero_()`). That is
    functionally our U26 (hipHostMalloc exact size, not registered memory). #69254 is the alternative
    (shared registered region, with #69617). Do not combine the two approaches for the same tier.
13. **Image.** `requirements/rocm.txt` pins `mooncake-transfer-engine-rocm == 0.3.14`, installed in
    `docker/Dockerfile.rocm` (runtime stage, line ~325), so **no need to build Mooncake**. AITER comes from
    the base stack, `docker/Dockerfile.rocm_base` `AITER_BRANCH="v0.1.29.post1"`, unless the bake used
    `AITER_NIGHTLY=1` / `AITER_SPEC` (both empty by default). Also MORI `v1.2.3.post1`, Triton `669b31ac`
    (3.8), torch `2.13.0+rocm10.0.0`, Python 3.14. On a compute node, confirm with `/app/versions.txt` and
    `pip show amd-aiter mooncake-transfer-engine-rocm`.

## Apply order (on de7cf7f8)

1. `61985` (`--exclude=tests/v1/attention/test_rocm_paged_mxfp4_indexer_plan.py`), then the MTP-draft FP8 supplementary PR (to open)
2. `58710` rebased (supplementary PR or author rebase; `--3way` of the current diff is clean for source)
3. `69540`, `68845`
4. `69616`, then FP4 dispatch / pack new PR (after 69616; must localize ids on the packed path)
5. `proposed-U13` (or the updated fork #1)
6. `llying-001#7` (DP LB KV weight)
7. `60909` (rebased) + U26 supplementary (gpu_worker.py), **or** `69254` (rebased) + `69617`
8. Triton top-k new PR (hooks `ops/rocm_aiter_mla_sparse.py` and `ops/rocm_paged_mxfp4_indexer.py`)
9. Not applied: 67232 (not needed), 68596 (moot with 61985; clean if you skip 61985), 68560 (not in recipe)

## Supplementary / new fork PRs opened 2026-11-27 (llying-001/vllm, draft, base `base/nightly-de7cf7f8` = de7cf7f8 unless noted)

| Fork PR | Patch file | What |
|---|---|---|
| [#11](https://github.com/llying-001/vllm/pull/11) | `llying-001-11-58710-rebased-de7cf7f8.diff` | vllm-project/vllm#58710 rebased onto de7cf7f8 (author's commit kept; only the test conflict resolved: main's new `test_build_skips_metadata_aiter_ignores` now asserts no device read) |
| [#12](https://github.com/llying-001/vllm/pull/12) | `llying-001-12-61985-mtp-draft-fp8.diff` | on #61985 (base `base/pr-61985-a7ec12ed`): MTP draft layers keep the FP8 indexer K cache |
| [#13](https://github.com/llying-001/vllm/pull/13) | `llying-001-13-aiter-agrs-full-graph.diff` | U13 on main: drop the `batch_descriptor.uniform` check that keeps AITER AG/RS from ever running (nothing sets `uniform` on main) |
| [#14](https://github.com/llying-001/vllm/pull/14) | `llying-001-14-60909-with-kv-offload.diff` | vllm-project/vllm#60909 at 15b5108c4 (with the CPU KV-offload half), cherry-picked onto de7cf7f8 without conflicts |

`series-C0.txt` stacks the 9 available patches; checked in order on a clean de7cf7f8 checkout with
`git apply --include='vllm/*'` (the Dockerfile's command): all apply.
