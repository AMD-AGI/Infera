# `patches/atom_nightly/`: build steps for an ATOM nightly base

`Dockerfile.atom` runs these only with `ATOM_NIGHTLY=1`, which says the base is
an ATOM nightly rather than the default release base. With it the build skips
the legacy `patches/atom/` loop (written for the release base; on the nightly
its edits are duplicate or dead code), runs the scripts here, and pins
`protobuf<7` (the nightly's OpenTelemetry needs it). Verified on
`rocm/atom-dev:nightly_202610131505` (ATOM `661ef766`, LMCache
`0.5.6.dev139+gf22dec28`). The build for this stack is:

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
| `patch_atom_dcp_sparse_pad_rows.py` | `atom/model_ops/attention_mla.py` | gives padded rows (q_len 0) an empty KV range before the sparse DCP persistent MLA plan is built; their stale offsets made the decode kernel read garbage slot ids (prefill `Memory access fault`) |
| `prebuild_aiter_mha_varlen.py` | AITER JIT dir | builds an AITER kernel into the image (a build step, not a source patch) |

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

Patch scripts use `anchor_patch.py`. All files of a patch resolve under
one importable package root (`lmcache` or `atom`). Every anchor must occur exactly
once before any file is written; a missing or repeated anchor, or a missing
file, exits 1 and leaves the package unchanged. A file that already holds the
patch's marker is skipped, so a rerun is a no-op. Unlike `patches/atom/`, which
swallows failures, a base that drifted from the one above fails the build.

## Dropping them

Re-check each script whenever `ATOM_BASE_IMAGE` moves.

- `patch_lmcache_rocm_pinned_alloc.py`: drop once LMCache's non-MP CPU pool
  (`_resolve_pinned_alloc_free`) no longer uses `hipHostRegister` on ROCm. As
  of 2026-10-14 LMCache `dev` has no such fix (LMCache#5591 covers only the MP
  server's L1 pool).
- `patch_atom_dcp_sparse_pad_rows.py`: drop once ATOM's fused DCP candidate
  exchange writes (or clamps) `dcp_sparse_kv_indptr_buffer` for the padded
  rows itself, or `_rebuild_sparse_dcp_persistent_metadata` clamps them. Not in
  `661ef766`; not filed upstream yet.
- `prebuild_aiter_mha_varlen.py`: drop once the base prebuilds that variant
  itself, or once ATOM no longer loads it.
