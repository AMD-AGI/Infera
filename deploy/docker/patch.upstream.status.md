# Patch ↔ upstream status

Every patch under `deploy/docker/patches/`, and where it stands relative to the
project it patches. Kept here so "why do we still carry this?" has one answer
per row, and so a patch that upstream has since merged gets dropped instead of
quietly outliving its reason.

**Verified with `gh` on 2026-08-01.** State drifts; re-check before relying on a
row. `gh search` matches titles and bodies, **not diff content**, so "no upstream
PR" means "none found by search", not "none exists".

Column meanings:

- **ours?** — was the upstream PR opened by a contributor of this repo?
  `yes` = us; `no` = a third party; `—` = no PR.
- **PR state** — of the upstream PR named in the same row.

## sglang — `patches/sglang_dsa/` (baked by `Dockerfile.sglang`, `APPLY_SGLANG_DSA_PATCHES=1`)

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `sglang_dsa/dsa_indexer_hip_dp_padded_rows.diff` | HIP/aiter paged-MQA sizes its output from DP-padded rows while `lengths` is sized to real rows → `Expected lengths.size(0) == B` | none found | [sglang#33059](https://github.com/sgl-project/sglang/pull/33059) | **yes** (`dorado269`) | OPEN, `REVIEW_REQUIRED` |
| ″ (same bug class, other platform) | — | none found | [sglang#32762](https://github.com/sgl-project/sglang/pull/32762) `[NPU] Fix DSA eager padding mismatch` — our diff is written in its shape | no (`stellaxcpeng`) | OPEN |
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
pinned base**. Merged-but-not-in-base still needs the local patch — the two
`MERGED` rows above (sglang#30265, Mooncake#2644) are exactly that case.

When adding a patch, add its row here in the same commit, and put the full
argument — evidence, alternatives, how it differs from our own upstream PR — in
the patch's own header. This table is the index, not the record.
