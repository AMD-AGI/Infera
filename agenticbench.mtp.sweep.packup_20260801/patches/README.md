# Patches

Two files, one fix. Both are **uncommitted working-tree changes** in the image
source repo at the time of this run, kept that way by operator decision until the
experiment concludes.

## `patch_hicache_rocm_host_alloc.py` + `dockerfile_add_rocm_hicache_layer.diff`

**What.** Routes SGLang's hierarchical-cache host allocator to `pin_memory`
(`hipHostMalloc`) on HIP, instead of `mmap` + `hipHostRegister`. The `.diff` adds
the patch layer to `deploy/docker/Dockerfile.sglang` so the fix is **baked into
the image**; the `.py` is the self-locating, idempotent patch script it runs.

**Why.** On ROCm, `hipHostRegister` maps host pages at a device virtual address
that **differs** from the host VA. SGLang's hicache stores raw **host**
`data_ptr()`s in a device-side pointer table that a GPU kernel dereferences:

    pool_host/common.py    ALLOC_MEMORY_FUNCS -> alloc_with_host_register()
                             buffer = alloc_mmap(...)           # anonymous mmap
                             cudaHostRegister(buffer.data_ptr(), n, 0)

    memory_pool_host.py    DSAIndexerPoolHost.init_kv_buffer
                             self.index_k_data_ptrs = torch.tensor(
                                 [x.data_ptr() for x in ...],   # HOST VAs
                                 dtype=torch.uint64, device=<gpu>)
                           -> transfer_kv_all_layer_mla(dst_layers=..., ...)

On CUDA this is fine — `cudaHostRegister` maps at the *same* address. On ROCm the
kernel is handed an address that is not mapped on the device, and since `gfx950`
is `xnack-` there is no page-migration fallback, so the process aborts:

    Memory access fault by GPU node-N (Agent handle: ...) on address <host VA>.

The fault address equals the host pointer exactly — that is the fingerprint.

`pin_memory` (`hipHostMalloc`) returns memory whose device pointer **is** the
host pointer, which is what the pointer-table design requires. This is the same
reason the `"npu"` and `"musa"` entries are already routed there.

**How applied.** As a Dockerfile layer, at image build time — **not** by patching
a running container. `deploy/docker/patches/sglang_rocm/` is copied in and every
`*.py` in it is executed, under `set -eu` so a drifted anchor **fails the build**
rather than silently shipping an engine that GPU-faults.

Verified in the built image from **freshly compiled bytecode**, not from source
(`scripts/start_ctr.sh`):

    ROCm hicache host alloc   GLM52_ROCM_HOST_ALLOC   pyc_hits=1   OK
    ALLOC_MEMORY_FUNCS dispatch  is_hip=True -> alloc_with_pin_memory   OK

The second line is the load-bearing one: the marker proves the file was edited,
the dispatch check proves the edit does what it claims.

**Context.** The fix originates from `kvd.rocm.hostalloc.packup_20260731`, where
it was root-caused and validated on this same cluster. It is **not on the merged
branch** — the branch was validated on vultr at `--context-length 32768` with
short prompts, a regime that never triggers the fault. This experiment runs
120K–235K-token prompts with kvd on, where it triggers immediately.

Result here: `Memory access fault` **0** on both legs across the entire 70-minute
sweep, including the conc=128 × 155K-token point.

**Status.** Deliberately **not committed** to the branch. Whether it should be
adopted there is a decision the operator deferred until after the experiment.
Recorded here so the image is reproducible either way.

## Applying them to a fresh checkout

    cd <repo>                      # at b92a1e8
    git apply <KIT>/patches/dockerfile_add_rocm_hicache_layer.diff
    mkdir -p deploy/docker/patches/sglang_rocm
    cp <KIT>/patches/patch_hicache_rocm_host_alloc.py deploy/docker/patches/sglang_rocm/

Then build as in `REPRODUCE.md` §2. The patch is idempotent and re-running it
prints `already applied`.

## Not a patch, but worth naming: the debug probe

`scripts/probe_prefetch.py` instruments every early return of SGLang's
`prefetch_from_storage`. It was **installed, used to diagnose one round, and
removed before any measured window** — verified absent from bytecode afterwards.
It is a diagnostic instrument, not part of the deployment, and it logs on a hot
path. Do not leave it installed. See `notes/kvd_serving_proof.md`.
