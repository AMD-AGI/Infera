# `patches/atom_nightly/`: build steps for an ATOM nightly base

`Dockerfile.atom` runs these only with `ATOM_NIGHTLY=1`, which says the base is
an ATOM nightly rather than the default release base. With it the build skips
the legacy `patches/atom/` loop (written for the release base; on the nightly
its edits are duplicate or dead code), runs the scripts here, and pins
`protobuf<7` (the nightly's OpenTelemetry needs it). Verified on
`rocm/atom-dev:nightly_202610131505` (ATOM `661ef766`, aiter `9205e1059`,
LMCache `0.5.6.dev139+gf22dec28`). The build for this stack is:

```bash
docker build -f deploy/docker/Dockerfile.atom \
  --build-arg ATOM_BASE_IMAGE=rocm/atom-dev:nightly_202610131505 \
  --build-arg BUILD_MOONCAKE=0 \
  --build-arg ATOM_NIGHTLY=1 \
  -t <tag> .
```

`BUILD_MOONCAKE=0` keeps the base's multi-protocol Mooncake (see the Mooncake
comment in `Dockerfile.atom`).

## Scripts

Each script's docstring has the full rationale.

| script | files | does |
|---|---|---|
| `patch_lmcache_rocm_pinned_alloc.py` | `lmcache/v1/memory_management.py` | on ROCm with `HSA_USERPTR_FOR_PAGED_MEM=0`, backs LMCache's NUMA CPU pool with exact-size `hipHostMalloc` (GTT) instead of `hipHostRegister` (KFD USERPTR) |
| `patch_atom_dcp_sparse_pad_rows.py` | `atom/model_ops/attention_mla.py` | in sparse DCP decode, gives the rows an eager step padded past the scheduled ones an empty KV range; their stale offsets made the decode kernel read garbage slot ids (prefill `Memory access fault`) |
| `patch_atom_api_gc_stalls.py` | `atom/model_engine/engine_core_mgr.py`, `atom/metrics/scheduler.py` | keeps the API process's heap flat, so its gen-2 GC passes stop growing with every request served: finished sequences no longer go on `CoreManager.outputs_queue` when serving online (nothing reads it), and the per-request context gauges are opt-in (`ATOM_METRICS_PER_REQUEST_CONTEXT=1`) |
| `patch_atom_dcp_topk_custom_ag.py` | `atom/model_ops/dcp_ops.py` | the sparse indexer's DCP top-k exchange goes through `dcp_all_gather` (aiter's IPC all-gather) instead of pynccl |
| `patch_aiter_custom_alltoall.py` | aiter `csrc/include/custom_all_reduce.cuh`, `dist/device_communicators/custom_all_reduce.py`; rebuilds `module_custom_all_reduce` | two kernels on aiter's custom all-reduce buffers: an all-to-all (`custom_all_to_all`) and the DCP merge in one kernel, all-to-all + LSE combine + per-token FP8 quant (`custom_a2a_combine_quant`) |
| `patch_atom_dcp_custom_a2a.py` | `atom/model_ops/dcp_ops.py` | the DCP attention merge exchanges its partials over `custom_all_to_all` where it applies, RCCL otherwise |
| `patch_atom_dcp_custom_a2a_combine.py` | `atom/model_ops/dcp_ops.py` | a DCP merge that also quantizes (the GLM-5.2 decode) packs the split layout and runs `custom_a2a_combine_quant` where it applies |
| `patch_aiter_mla_bmm_configs.py` | aiter's gfx950 Triton configs of `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`, N=512 K=192 and N=256 K=512 | one config bucket per decode M up to 512 for the MLA absorbed bmm; the stock buckets take a masked path at 2-2.5x the time whenever M is not a multiple of 64 |
| `patch_aiter_glm52_decode_gemm_rows.py` | new `configs/model_configs/a8w8_bpreshuffle_tuned_gemm_glm5.2_decode.csv` | 27 tuned FP8 GEMM rows for the GLM-5.2 decode's M 144-240 |
| `prebuild_aiter_mha_varlen.py` | AITER JIT dir | builds an AITER kernel into the image (a build step, not a source patch) |
| `check_dcp_merge.py` | none | GPU check of the `patch_aiter_custom_alltoall.py` kernels, run by hand (below) |

### ATOM per-request context gauges

`patch_atom_api_gc_stalls.py` changes behavior for every model served from the
image: `atom:prefill_request_context_tokens` and
`atom:decode_request_context_tokens` are no longer recorded unless the engine
starts with `ATOM_METRICS_PER_REQUEST_CONTEXT=1`. Each request added a series
that was never removed, and every `/metrics` render of the API process re-read
all of them. Set it where those series are needed, such as profiling runs.

### DCP merge kernel check

The anchors only show that the source text did not drift, not that the two
aiter kernels still agree with RCCL once aiter's sync primitives or buffer
registration change. After a base move, run `check_dcp_merge.py` in the new
image on 2, 4 or 8 GPUs; it exits 1 on any mismatch:

```bash
docker run --rm --network host --ipc host --device /dev/kfd --device /dev/dri \
  --group-add video --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -v "$PWD/deploy/docker/patches/atom_nightly:/check:ro" <image> \
  torchrun --nproc-per-node 4 /check/check_dcp_merge.py
```

### LMCache CPU pool on ROCm

The patch takes effect only when the engine process starts with
`HSA_USERPTR_FOR_PAGED_MEM=0` (see the script docstring for why). That must be
set before HIP initializes, so pass it in the container environment, not from
Python. The image does not set it: it turns every pinned host buffer of the
process into GTT, and GTT has a machine-wide limit (about half of host memory),
so set it only where an LMCache pool is allocated. The GTT pool lands on the
NUMA node of the first GPU the process can open, not on LMCache's `numa_id`.
Hugepage pools are left unchanged. On success LMCache logs
`LMCache CPU pool uses hipHostMalloc (GTT`; without `=0` it logs a warning.

### AITER kernel prebuild

Builds one `mha_varlen_fwd` variant the base lacks, so its first call does not
JIT-compile for ~30 s under load; see the script docstring. On gfx950 this
base's prefill loads the ASM `fmha_v3_varlen_fwd` module instead and never
loads it; it is kept for other archs (gfx942).

## Contract

Source patches use `anchor_patch.py`. All files of a patch resolve under one
importable package root (`lmcache`, `atom` or `aiter`; aiter's C++ sources are
reached as `../csrc` from its package). Every anchor must occur exactly once
before any file is written; a missing or repeated anchor, or a missing file,
exits 1 and leaves the package unchanged. A file that already holds the
patch's marker is skipped, so a rerun is a no-op.

The two aiter table patches edit data and check it themselves:
`patch_aiter_mla_bmm_configs.py` requires both files to hold the stock buckets,
and `patch_aiter_glm52_decode_gemm_rows.py` requires that its table is new or
identical and that none of its keys is in a table aiter merges it with (aiter
would raise at engine start). `patch_aiter_custom_alltoall.py` also rebuilds
`module_custom_all_reduce` from the patched sources, as the runtime JIT would;
that needs no GPU, and the archs come from `GPU_ARCHS` (default
`PYTORCH_ROCM_ARCH`).

No script depends on another's edits, so the order the build globs them in
does not matter. Unlike `patches/atom/`, which swallows failures, a base that
drifted from the one above fails the build.

## Dropping them

Re-check each script whenever `ATOM_BASE_IMAGE` moves, and run
`check_dcp_merge.py` in the new image. Upstream status:
[`patch.upstream.status.md`](../../patch.upstream.status.md).

- `patch_lmcache_rocm_pinned_alloc.py`: drop once LMCache's non-MP CPU pool
  (`_resolve_pinned_alloc_free`) no longer uses `hipHostRegister` on ROCm. As
  of 2026-10-14 LMCache `dev` has no such fix (LMCache#5591 covers only the MP
  server's L1 pool).
- `patch_atom_dcp_sparse_pad_rows.py`: drop once ATOM's fused DCP candidate
  exchange writes (or clamps) `dcp_sparse_kv_indptr_buffer` for the padded
  rows itself, or `_forward_decode` clamps them. Not in `661ef766`; not filed
  upstream yet.
- `patch_atom_api_gc_stalls.py`: drop once ATOM, serving online, no longer
  queues finished sequences on `outputs_queue` and no longer keeps one gauge
  series per request (removes it when the request finishes, or uses a
  histogram). Not in `661ef766`; not filed upstream yet.
- `patch_atom_dcp_topk_custom_ag.py`: drop once ATOM's DCP top-k exchange uses
  `dcp_all_gather` (or another custom all-gather) itself.
- `patch_aiter_custom_alltoall.py`, `patch_atom_dcp_custom_a2a.py`,
  `patch_atom_dcp_custom_a2a_combine.py`: one change; drop the three together
  once aiter offers an IPC all-to-all and a fused merge and ATOM's `cp_lse_a2a`
  uses them. Not filed upstream yet.
- `patch_aiter_mla_bmm_configs.py`: drop once aiter's gfx950 configs for these
  two shapes bucket M finely below 512 (a file with `M_BOUNDS` fails the
  build); re-sweep if the stock entries change.
- `patch_aiter_glm52_decode_gemm_rows.py`: drop once aiter's own GLM-5.2
  a8w8_bpreshuffle table covers M 144-240 (a shared key fails the build).
- `prebuild_aiter_mha_varlen.py`: drop once the base prebuilds that variant
  itself, or once ATOM no longer loads it.
