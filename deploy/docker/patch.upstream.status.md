# Patch ↔ upstream status

Every patch under `deploy/docker/patches/`, and where it stands relative to the
project it patches. Kept here so "why do we still carry this?" has one answer
per row, and so a patch that upstream has since merged gets dropped instead of
quietly outliving its reason.

**Verified with `gh` on 2026-08-01**, except the `patches/sglang_rocm/` section:
2026-08-03 for the host allocator, 2026-08-04 for the staged write-back.
State drifts; re-check before relying on a row. `gh search`
matches titles and bodies, **not diff content**, so "no upstream PR" means "none
found by search", not "none exists" — where a row could be checked by reading
upstream source instead, it says so.

Column meanings:

- **ours?** — was the upstream PR opened by a contributor of this repo?
  `yes` = us; `no` = a third party; `—` = no PR.
- **PR state** — of the upstream PR named in the same row.

## sglang — `patches/sglang_dsa/` (baked by `Dockerfile.sglang` and `Dockerfile.sglang.gfx942`, `APPLY_SGLANG_DSA_PATCHES=1`)

Only patch 01 is baked by both: it is an anchor script, while 02 and 04 are
`--fuzz=0` diffs pinned to v0.5.15.post1 and cannot apply to the gfx942 image's
v0.5.16 base. That image substitutes 02b and 04 at runtime with
`--json-model-override-args '{"index_share_for_mtp_iteration":false}'` and does
not address 02a at all — `patches/sglang_dsa/README.md` carries the reasoning.

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py` | HIP/aiter paged-MQA sizes its output from DP-padded rows while `lengths` is sized to real rows → `Expected lengths.size(0) == B` | none found | [sglang#33059](https://github.com/sgl-project/sglang/pull/33059) | **yes** (`dorado269`) | OPEN, `REVIEW_REQUIRED` |
| ″ (same bug class, other platform) | — | none found | [sglang#32762](https://github.com/sgl-project/sglang/pull/32762) `[NPU] Fix DSA eager padding mismatch` — our diff is written in its shape | no (`stellaxcpeng`) | OPEN |
| ″ (anchor collision, **not** a fix) | — | — | [sglang#32738](https://github.com/sgl-project/sglang/pull/32738) pads heads for DeepGEMM at the same two aiter call sites; [#31480](https://github.com/sgl-project/sglang/pull/31480) extracts the paged-MQA backend and restructures the `is_aiter()` dispatch | no | both OPEN (re-read 2026-08-03) |
| `sglang_dsa/dsa_backend_dp_sync_and_page_table_rows.diff` (2a) | `seq_lens.max().item()` is a host sync on a branch only *some* DP ranks take → DP collectives desync → deadlock | none found | none found | — | — |
| `sglang_dsa/dsa_backend_dp_sync_and_page_table_rows.diff` (2b) | page table has one row per **request**, top-k one per **token** under MTP → `assert page_table.shape[0] == topk_indices.shape[0]` | none found | [sglang#32209](https://github.com/sgl-project/sglang/pull/32209) solves the same row mismatch by **trimming q/top-k**; porting that half here fails at conc=32 and is unresolved | no (`HZY-Wade`) | OPEN, `REVIEW_REQUIRED` |
| `sglang_dsa/draft_cuda_graph_dp_vote.diff` | draft graph/eager choice is per-rank and diverges on the PD decode leg → group deadlock | [sglang#32527](https://github.com/sgl-project/sglang/issues/32527) (independent report, 8× Blackwell) | [sglang#32209](https://github.com/sgl-project/sglang/pull/32209) — same defect, same strategy; **this diff adopts its placement** | no (`HZY-Wade`) | OPEN, `REVIEW_REQUIRED` |

Prerequisite for the set, applied earlier in the same Dockerfile and **asserted**
by `scripts/apply_sglang_dsa_patches.sh`:

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `sglang/patch_glm52_nextn_quark_exclude.py` | GLM-5.2 MTP `eh_proj` is bf16 but the quark-exclude check probes the bare layer prefix → draft weight-load dies `3072 vs 6144` | none found | [sglang#30265](https://github.com/sgl-project/sglang/pull/30265) `[AMD] Fix GLM-5.2 MTP Quark excludes` — a **superset** (dedicated `GlmMoeDsaForCausalLMNextN`); ours is a narrow backport | no (`wangjiaxin99`) | **MERGED** 2026-07-08 |

> Our base `v0.5.15.post1` (`0b3bb0c`) predates #30265 — the release line was cut
> without it. **Drop this patch** when the base sglang carries #30265; the anchor
> disappears and the script no-ops.

Background, already present in the base and **not** patched by us:
[sglang#30378](https://github.com/sgl-project/sglang/pull/30378) /
[#30427](https://github.com/sgl-project/sglang/pull/30427) (MERGED) clamp padded-row
seq_lens **values**; our patch 01 fixes the HIP-side row **count**.
[#30839](https://github.com/sgl-project/sglang/pull/30839) /
[#31083](https://github.com/sgl-project/sglang/pull/31083) (MERGED 2026-07-14)
introduced the guard patch 04 repairs — so that deadlock is a regression in this
baseline, not a legacy wart.
[#32722](https://github.com/sgl-project/sglang/pull/32722) (OPEN) adds a test for
PD + DP-attention + MTP, i.e. **no CI covers this topology today**.

## sglang PD — `patches/sglang_disagg/` (baked by `Dockerfile.sglang` and `Dockerfile.sglang.gfx942`)

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `sglang_disagg/patch_mooncake_early_send_wait_event.py` | `mooncake/conn.py` never waits on the forward's completion event and the overlap path records none, so chunked prefill hands a non-final chunk to the decode leg while the forward writing those pages is still running → prompts longer than one chunk come back **partially wrong**, with nothing in any log | [sglang#25583](https://github.com/sgl-project/sglang/issues/25583) reports the same corruption shape on GLM-5, but **aggregated** — no PD, no mooncake — so a shared root cause is unestablished; closed **inactive** 2026-07-18, no follow-up | none found | — | — |

> `prefill.py` already records the completion event this needs; only the `mori`
> backend ever read it, and the patch mirrors what `mori` does. **Drop it** once a
> base sglang synchronizes on that event itself — the script then reports "already
> present" and no-ops.
>
> Unlike the rest of this page, this row was **not** verified with `gh`: the issue
> state was read from the web UI on 2026-08-03, and no upstream PR search was run.
> "none found" here is weaker than elsewhere on the page.

## sglang ROCm — `patches/sglang_rocm/` (baked by `Dockerfile.sglang`, `Dockerfile.sglang.gfx942`)

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `sglang_rocm/patch_hicache_rocm_host_alloc.py` | hicache allocates host pools with `mmap` + `hipHostRegister`, which on ROCm maps the pages at a device address ≠ the host VA, but the pools hand raw host `data_ptr()`s to GPU kernels via device-side pointer tables → `Memory access fault by GPU node-N on address <host VA>` on the first kvd write-back | none found | none found | — | — |
| `sglang_rocm/patch_hicache_rocm_staged_write_back.py` | `pool_host/mla.py` enables the staged write-back JIT on HIP while `DSAIndexerPoolHost` — in the same `HostPoolGroup` for any DSA model — still gates it on `_is_cuda`. The group ANDs the flag, so the controller puts the destination indices on the GPU; the anchor MLA pool then reads its own flag and launches the JIT anyway → `Tensor match failed … device=rocm:0 … allowed options: [cpu, rocm_host]`, scheduler exit −3 on the first write-back | none found | [sglang#28534](https://github.com/sgl-project/sglang/pull/28534) `[AMD] Enable JIT staged HiCache write-back and fix CPU-index crash` — added the HIP enablement and aligned `cache_controller.py`, `memory_pool_host.py`, `pool_host/mha.py`; **never touched `pool_host/mla.py`** | no (`AMD-yanfeiwang`) | **MERGED** 2026-07-09 |

**`patch_hicache_rocm_host_alloc.py`** — the fault is **gfx950-only so far**: MI300X
(amdgpu 6.14.14, ROCm 7.2.0) measures the two addresses equal, so
`Dockerfile.sglang.gfx942` carries this one preventively rather than to fix a crash.
Don't read its row above as evidence the fault was seen on both arches.

> Verified on 2026-08-03 by **reading upstream `main` directly** (contents API,
> `pool_host/common.py`): `ALLOC_MEMORY_FUNCS` still overrides only `"npu"` and
> `"musa"`, with no HIP entry — so main is affected too, and this is stronger
> than the usual "no search hit". [sglang#23361](https://github.com/sgl-project/sglang/pull/23361)
> (**MERGED**, MUSA) is the same one-line dispatch override for the same reason
> and is the shape this patch copies.
>
> **No PR of ours has been filed** — it should be. **Drop this patch** when a
> base sglang routes HIP to `alloc_with_pin_memory`; the anchor stops matching
> and the script exits non-zero, so the drop is not silent. Note
> [#32503](https://github.com/sgl-project/sglang/pull/32503) /
> [#32792](https://github.com/sgl-project/sglang/pull/32792) (OPEN, Intel XPU
> HiCache) touch this same dict — expect an anchor conflict, not a fix.

**`patch_hicache_rocm_staged_write_back.py`** — **not** preventive: without it the
v0.5.16 gfx942 base kills the prefill scheduler on the first reused prefix, so that
image needs it to run kvd at all. A **no-op on the mi35x image**, which has nothing to
fix — every `can_use_write_back_jit` gate at v0.5.15.post1 is still `_is_cuda` (MHA,
MLA, both V4 pools and `DSAIndexerPoolHost`, all in `memory_pool_host.py` before
MLA/MHA moved into `pool_host/`), and `_is_hip` appears only in the kernel import
guard, so the group's AND and its anchor agree on False. #28534 introduced the
disagreement after that tag; the absent `pool_host/mla.py` is only how the script
notices. Both `_is_cuda or _is_hip` in `pool_host/mla.py` and the CUDA-only gates on
the other pools are still on upstream `main` (read from the raw files, so stronger
than a search miss), i.e. **main is affected** — and #28534's own reasoning argues for
the opposite repair, teaching the remaining pools the JIT rather than gating MLA down,
so expect upstream to close this differently than we did.

> **That repair is already in flight (checked 2026-08-04):**
> [sglang#30350](https://github.com/sgl-project/sglang/pull/30350) `Add HiCache JIT
> test and benchmark for ROCm/HIP CI support` (**OPEN**, `Emmanuel0612`) adds
> `_is_cuda_alike = _is_cuda or _is_hip` and flips exactly the three CUDA-only gates
> (`DSAIndexerPoolHost`, `DeepSeekV4PagedHostPool`, `DeepSeekV4StateHostPool`), so the
> group AND stops reading False on ROCm — **including the V4 stack our patch does not
> cover**. It also teaches `staged_write_back.cuh` to accept kDLROCM/kDLROCMHost (the
> TensorMatcher check that emits our crash) and adds an AMD CI lane for the HiCache
> JIT; the author reports 47/47 on MI355X. Stalled rather than rejected: amd-bot
> called its AMD suites green on 07-09 alongside a merge conflict, conflicts were
> cleared 07-13, nothing since 07-16, and #28534 landed in between. **Our leverage is
> a gfx942 reproduction on that thread, not a competing PR** — it has no MI300X
> datapoint.
>
> **Anchor drift cannot be the drop signal.** #30350 never touches
> `pool_host/mla.py`, so our anchor would keep matching and the patch would keep
> applying on top of the fix — not a crash, since both gates read False again, but a
> silent forfeit of the staged kernel #30350 enables. `check_group_still_poisoned()`
> checks the *precondition* instead: once `DSAIndexerPoolHost` stops gating on
> `_is_cuda` alone, the script refuses and exits 1 telling the operator to drop it.
>
> **Exercised 2026-08-04** against the v0.5.16 *and* current-`main` copies of both
> files, on throwaway trees. Stock: applies, exit 0. Re-run: "already applied", exit
> 0. #30350 simulated by flipping the three CUDA-only gates: refuses, exit 1, with
> `mla.py` byte-identical to pristine. `pool_host/mla.py` removed (the v0.5.15.post1
> shape): tolerated, exit 0. Anchor or pool renamed: exit 1.
>
> Scope: `DSAIndexerPoolHost` is not the only CUDA-only member that can poison the
> group's AND, and `build_deepseek_v4_hicache_stack` puts `DeepSeekV4PagedHostPool`
> in a group anchored by `LogicalHostPool` (flag unconditionally True). Expect the
> same crash on a V4 hicache stack on gfx942; **this patch gates `mla.py` only**. No
> V4 stack runs on this branch, so that gate would be untested — the script's SCOPE
> section records it for whoever gets there.

## Mooncake C++ — `patches/mooncake_cpp/` (built by `Dockerfile.sglang`, `Dockerfile.vllm`, `Dockerfile.atom`)

Pinned to Mooncake `main @ 747003c`; `git apply` fails loudly on ref drift.

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `mooncake_cpp/rdma_auto_chunk_mr_2017.diff` | buffers over the device `max_mr_size` are silently truncated by `ibv_reg_mr` while `BufferDesc.length` advertises the full size → `IBV_WC_REM_ACCESS_ERR` past the boundary | [Mooncake#2017](https://github.com/kvcache-ai/Mooncake/issues/2017) | [Mooncake#2644](https://github.com/kvcache-ai/Mooncake/pull/2644) | **yes** (`jiejingzhangamd`) | **MERGED** 2026-07-28 |
| `mooncake_cpp/rdma_transport_dmabuf_cmake.diff` | `USE_HIP_DMABUF` is defined only on the `transfer_engine` target, so the `ibv_reg_dmabuf_mr` branch compiles **out** of `rdma_transport` — where it is actually called → GPU buffers fall back to bare `ibv_reg_mr`, which cannot pin VRAM without `ib_peer_mem` | none found | none found | — | — |
| `mooncake_cpp/transfer_engine_impl.diff` | `installTransport("hip")` runs unconditionally, so GPU buffers become intra-node HIP IPC segments a cross-node peer cannot open (`Corrupted segment descriptor hipbuffer`) | none found | none found | — | — |

> #2644 is merged upstream but **not** in the pinned `747003c` tree, so the local
> diff is still applied. Drop it when the pin advances past the merge.

## vLLM — `patches/vllm/` (baked by `Dockerfile.vllm`)

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `vllm/patch_defer_kv_register.py` | registering the Mooncake KV pool before warmup trips a decode-boot crash at high util (`compile_or_warm_up_model` returns None → `AttributeError: 'NoneType' … language_model`); defer to the end of warmup | none found | none found | — | — |
| `vllm/patch_moriio_pagelen.py` | MoRIIO MLA derives per-block transfer size/stride from tensor **shape**, so block-scaled fp8 MLA + DSA transfers the wrong geometry → PD output is wrong while direct prefill is correct | none found | none found | — | — |
| `vllm/patch_moriio_write.py` | in WRITE (push) mode the decode addresses **itself** (`is_producer=True`), the consumer handler asserts, the notify thread dies and the request hangs | `AMD-AGI/Infera#67` | fixed on vLLM `main` / `v0.22.1rc0` **source**, but no AMD ROCm image ships it | no | n/a (source-only) |
| `vllm/patch_sched_guard.py` | decode EngineCore dies on `assert req_id in self.requests` when a KV-xfer-finished event arrives for an already-removed request | `AMD-AGI/Infera#69` | none found | — | — |
| `vllm/patch_vllm_mooncake_blocksize.py` | backends that force a kernel block size of 1 make the connector index logical pages at kernel granularity → RDMA moves empty rows, decode attends over zeros | none found | none found | — | — |
| `vllm/patch_vllm_mooncake_prom_metrics.py` | `MooncakeConnector` lacks `build_prom_metrics`, so `MultiKVConnectorPromMetrics.observe` asserts and kills the engine under `MultiConnector` | none found | internal PR #178 (the sibling fix for `InferaKvdConnector`) | **yes** | **MERGED** |

> The `AMD-AGI/Infera#NN` references above are quoted verbatim from the patches'
> own docstrings, which is the citation form this repo already uses. They resolve
> against an internal tracker, so treat a `404` as expected rather than as a
> stale number.

## vLLM DSv4 — `patches/vllm-dsv4/` (baked by `Dockerfile.vllm`)

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `vllm-dsv4/patch_aiter_flydsl_moe_memref_bufres.py` | aiter 0.1.16's `fx.ptrtoint` rejects flydsl memrefs → `MLIRError` in the 2-stage MoE GEMM (Kimi-K2.6 int4 W4A16) | none found | none found | — | — |
| `vllm-dsv4/patch_moriio_dsv4_hybrid_blocksize.py` | a global `block_size` equality check kills the prefill worker at KV registration, though DSv4 registers per-layer caches with different block sizes and offsets already use the per-layer map | none found | none found | — | — |
| `vllm-dsv4/patch_moriio_dsv4_noncontig_register.py` | `register_torch_tensor` rejects DSv4's 576B-aligned non-contiguous fp8_ds_mla KV view, and `.contiguous()` would detach from the buffer the forward writes into | none found | none found | — | — |
| `vllm-dsv4/patch_moriio_dsv4_sparse_backend.py` | the generic ROCm selector returns `ROCM_AITER_MLA_SPARSE` (no `fp8_ds_mla`) and raises, killing the prefill worker, though `backend_name` is only a P/D handshake tag | none found | none found | — | — |

### `patches/vllm-dsv4/legacy/` — archival, **not executed by any Dockerfile**

Kept for provenance; both are no-ops on the current verified stack
(vLLM `0.23.1rc1.dev748`, `amd-aiter 0.1.16.post2`).

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `vllm-dsv4/legacy/patch_dsv4_aiter_moe.py` | DSv4 MXFP4 MoE gate-mode plumbing, before aiter carried it | none found | [aiter#3123](https://github.com/ROCm/aiter/pull/3123) `[MoE] Align Swiglu MXFP4 fused quant paths` | no (`XiaobingSuper`) | **MERGED** 2026-05-12 |
| `vllm-dsv4/legacy/patch_dsv4_mhc_aiter.py` + `.diff` | `mhc_pre_gemm_sqrsum_kernel` store race → EngineCore dies | none found | [aiter#3033](https://github.com/ROCm/aiter/pull/3033) `Fix sqrsum store race condition` | no (`kkHuang-amd`) | **MERGED** 2026-05-06 |

## ATOM — `patches/atom/` (baked by `Dockerfile.atom`)

ATOM is an internal engine; there is no public upstream to file against.

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `atom/patch_gdn_pd_state_transfer.py` | hybrid GDN models never transfer the GatedDeltaNet recurrent state in PD, so decode starts from a zero state and cannot recall prompt context (5/5 mixed vs 0/5 PD) | n/a — internal engine | n/a | — | — |
| `atom/patch_minimax_m2_qknorm_rope.py` | stock `minimax_m2.py` fails to load: `get_rope()` rejects `dtype=`, and the TP fused QK-norm kernel needs a batch guard | n/a | n/a | — | — |
| `atom/patch_mooncake_consumer_slot.py` | `UnboundLocalError: consumer_staging_pool_idx` crashes the decode worker on the first PD request for models with slot state but no `slot_regions` | n/a | n/a | — | — |

## hipFile — `patches/hipfile_async/` (baked by `Dockerfile.vllm`)

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `hipfile_async/hipfile_async.patch` (+ `file_async_fragment.py`, `patch_hipfile_async.sh`) | the ROCm hipFile binding exposes no async API; adds `write_async`/`read_async`, `Stream`, `supports_async()`. Cython stack-locals passed as `&local` die before the driver dereferences them → `bytes_done` reads 0 and intermittent `HipFileException 5022`, so the wrapper calls `libhipfile.so` via ctypes with heap-allocated slots | none found | none found | — | — |

## Not patches

Every file under `patches/` is covered above except these, which carry no fix of
their own: `mooncake_cpp/apply_mooncake_cpp_patches.sh` (applies the three
Mooncake diffs), `sglang_dsa/README.md`, `sglang_disagg/README.md` and
`vllm-dsv4/legacy/README.md`.

## Maintenance

A row is ready to delete when its upstream PR is **merged and present in the
pinned base**. Merged-but-not-in-base still needs the local patch — sglang#30265
and Mooncake#2644 are exactly that case. The staged write-back row is the
exception: its `MERGED` PR (sglang#28534) is what *introduced* the defect, so that
patch drops on sglang#30350 instead.

When adding a patch, add its row here in the same commit, and put the full
argument — evidence, alternatives, how it differs from our own upstream PR — in
the patch's own header. This table is the index, not the record.
