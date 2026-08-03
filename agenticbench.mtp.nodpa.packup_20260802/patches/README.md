# Patches

Three things had to be applied for this run to work. **None is on the branch.**

## 1. `0001-dockerfile-rocm-hicache-hostalloc.patch` + `sglang_rocm/`

**What.** Modifies `deploy/docker/Dockerfile.sglang` to apply
`sglang_rocm/patch_hicache_rocm_host_alloc.py` at build time, which points
`ALLOC_MEMORY_FUNCS["cuda"]` at `alloc_with_pin_memory` (`hipHostRegister`) under
HIP.

**Why.** `gfx950` is `xnack-` — there is no page-migration fallback, so a
host-pointer access from the GPU is a hard fault, not a slow path. Without this,
kvd + long prompts kill the prefill leg with
`Memory access fault by GPU node-N on address <host VA>`.

**How.** Apply to a clean `e56e975` checkout **before** building:

```bash
git apply patches/0001-dockerfile-rocm-hicache-hostalloc.patch
cp -r patches/sglang_rocm deploy/docker/patches/
bash scripts/build_image.sh <job>
```

**Context.** It was never needed on the vultr sibling because that validation ran
`--context-length 32768` with small prompts. These are 35K–260K with kvd on.
`Dockerfile.sglang.AS-BUILT` is the exact file the reported image was built from.

**Status: deliberately uncommitted**, by standing operator instruction (reaffirmed
at this packup). Both experiments that needed it are now complete.

Verify it reached the running image *behaviourally*, not by listing files:

```bash
docker exec agbench_mtp python3 -c "
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS, alloc_with_pin_memory
assert ALLOC_MEMORY_FUNCS['cuda'] is alloc_with_pin_memory"
```

## 2. `apply_p1v3.py` — `GLM52_P1V3`, decode container only

**What.** Fixes the DSA indexer's handling of the reversed IDLE-rank case.

**Why.** Without it the run dies under MTP draft-extend with
`Expected lengths.size(0) == B`.

**How.** Applied **inside the running decode container**, not in the image:

```bash
docker exec agbench_mtp bash -c '
  F=/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
  md5sum $F                      # expect 632f17acd38737459b43f830ee60ee89
  cp $F /tmp/dsa_indexer.py.orig # for a revert-based A/B
  python3 apply_p1v3.py'
```

Then verify the **loaded module**, not the file — a stale `__pycache__` silently
running unpatched bytecode has invalidated experiments on this stack twice:

```bash
docker exec agbench_mtp python3 -c \
  "import sglang.srt.layers.attention.dsa.dsa_indexer as m, inspect; print(inspect.getsource(m).count('GLM52_P1V3'))"
# expect: 3
```

**Context.** The md5 was re-confirmed on this run's freshly built image, so the
patch still targets the intended source. `/tmp/dsa_indexer.py.orig` is preserved
inside the container.

## 3. The leg-script corrections — inline, not a .patch

`scripts/glm52_leg_spur_mtp.sh` differs from the branch's version by **three
functional lines**, all inside the `DPA` branch, and they are load-bearing for
*this* experiment rather than for the engine:

- `--ep-size $TP` kept when `DPA=0` (was dropped, collapsing MoE expert
  parallelism at the same time as attention — a two-variable change)
- `CHUNK` caller-overridable when `DPA=0` (was hardcoded 8192)

Called out here per the checklist's "if the fix is inline in a script rather than
a patch, it's called out". Full reasoning with the source read for each:
`../notes/nodpa_design.md`.
