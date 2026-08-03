# Patches

**Three artifacts, all load-bearing.** Without any one of them this experiment
does not produce a valid number: two prevent a crash, and the third is what makes
the image reproducible at all.

| # | what | where it lives | applied |
|---|---|---|---|
| 0001 | ROCm hicache host-allocator fix | build time, into the image | **uncommitted working-tree change** |
| 0002 | prefill `--mem-fraction-static` 0.88 → 0.80 | `../scripts/glm52_leg_spur_mtp.sh` | inline in the leg script |
| 0003 | `GLM52_P1V3` DSA indexer fix | `apply_p1v3.py`, run in the decode container | runtime, in-container |

---

## 0001 — ROCm hicache host allocator (`hipHostRegister` → `hipHostMalloc`)

**Files:** `0001-dockerfile-rocm-hicache-hostalloc.patch` (the `git diff`),
`sglang_rocm/patch_hicache_rocm_host_alloc.py` (the patch it invokes), and
`Dockerfile.sglang.AS-BUILT` (the full file as built, for reference).

### What
Routes `ALLOC_MEMORY_FUNCS["cuda"]` to `alloc_with_pin_memory` (`hipHostMalloc`)
under HIP, exactly as the existing `"npu"` and `"musa"` entries already do.

### Why
`hipHostRegister` maps host pages at a **device VA that differs from the host
VA**, but sglang's hicache stores raw host `data_ptr()`s in device-side pointer
tables that a GPU kernel dereferences. The kernel then dereferences an unmapped
address:

    Memory access fault by GPU node-N on address <host VA>

**gfx950 is `xnack-`** — there is no page-migration fallback, so this is a hard
abort, not a slow path.

### How
Applied at **build time** by `deploy/docker/Dockerfile.sglang`. Unlike the
disagg patch loop above it in the Dockerfile, this one is **not** failure-
tolerant: an image that GPU-faults the moment kvd writes back is worse than a
failed build.

Prove it in the **bytecode**, not the source:

```bash
docker exec agbench_mtp python3 -c "
from sglang.srt.mem_cache.memory_pool_host import ALLOC_MEMORY_FUNCS, alloc_with_pin_memory
assert ALLOC_MEMORY_FUNCS['cuda'] is alloc_with_pin_memory
print('hostalloc OK')"
```

### Context — why it is uncommitted, and why that is deliberate
This fix is **not on the branch** (`e56e975`). It exists only as two
working-tree changes, kept uncommitted on the operator's explicit instruction:
decide after the experiment whether it enters the branch.

It never mattered on vultr because that validation ran `--context-length 32768`
with small prompts. On spur, Case A prompts are 74K–235K tokens **with kvd on**,
which is exactly the path that faults.

To reproduce: apply `0001-*.patch` to `deploy/docker/Dockerfile.sglang`, ensure
`deploy/docker/patches/sglang_rocm/` exists, and rebuild.

**Upstream status.** Not filed. The `"npu"`/`"musa"` precedent in the same dict
suggests upstream would take it.

---

## 0002 — prefill `--mem-fraction-static` 0.88 → 0.80

**Not a separate file** — it is inline in `../scripts/glm52_leg_spur_mtp.sh`
(the `GMU` default), with the full rationale in the comment block there.

### What
`if [ "$ROLE" = "prefill" ]; then GMU="${GMU:-0.80}"; else GMU="${GMU:-0.85}"; fi`
Decode is deliberately untouched — changing both would make it a two-variable fix.

### Why
At 0.88 the prefill leg aborts under Case A's prompt lengths:

    rocdevice.cpp:3582 HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 78 MB
    Fatal Python error: Aborted

**Not KV exhaustion** — `token usage` reads 0.01–0.05 at the abort. It is
DP-attention runtime **activation** memory.

### How / effect

| | 0.88 | 0.80 |
|---|---:|---:|
| per-rank free after init | 33.8 GB | **284.0 GB** |
| KV pool (tokens/rank) | 3,260,992 | 2,939,264 (−10 %) |

**The direction is the trap.** Prefill activation OOM is fixed by *lowering*
mem-fraction-static — the opposite of the decode-side retract fix.

### Context
Cost three attempts to find, and it was already documented in the vultr sibling
kit (`../../caseA.glm52.fullfeature.packup_20260801/`, patch 0002) before the
first attempt. Three wrong root causes were published along the way; all three
and their refutations are in `../notes/notes.config.md`.

**Upstream status.** N/A — a deployment tuning value, not a defect.

---

## 0003 — `GLM52_P1V3`: the reversed padding case in the aiter paged-MQA trim

**File:** `apply_p1v3.py`. Runs **inside the decode container**. Idempotent;
anchors on exact source text; exits non-zero if the image differs.

### What
Three edits in `_get_topk_paged`: `_p1v2_rows = min(_p1v2_real, _p1v2_padded)`
becomes the single source of truth; a new `_p1v2_clip` flag passes
`ke_offset=metadata.get_seqlens_expanded()[:_p1v2_rows]` when rows were clipped;
the restore-padding assert keys off `_p1v2_rows`.

### Why
The image already ships `GLM52_P1V2`, which guards only one direction —
`_p1v2_real < _p1v2_padded`, i.e. it assumes DP padding always makes `q_fp8`
*longer*. On a DP-attention **IDLE** rank under MTP draft-extend the inequality
inverts. Captured live on vultr with the image's own `SGLANG_DEBUG_DSA_ROWS=1`:

    mode=IDLE q_fp8=(1,32,128) q_offset=2 ntnp=0 agree=False lengths=(2,) -> mqa_q=(1,32,128)

`q_offset` (2) exceeds the rows present (1), no trim runs, and `fast_topk_v2`
gets 1 score row against 2 lengths entries:

    RuntimeError: Expected lengths.size(0) == B to be true, but got false.

**A trim cannot fix it** — there are *fewer* query rows than lengths entries, so
both sides must be reconciled down to `min(real, padded)`.

### How

```bash
# 1. confirm the image matches what the patch anchors on
docker exec agbench_mtp md5sum \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
# expect 632f17acd38737459b43f830ee60ee89

# 2. back up, apply
docker exec agbench_mtp cp <target> /tmp/dsa_indexer.py.orig
docker cp apply_p1v3.py agbench_mtp:/tmp/ && docker exec agbench_mtp python3 /tmp/apply_p1v3.py
# expect: patched OK - GLM52_P1V3 occurrences: 3

# 3. clear bytecode, verify the LOADED module, relaunch the leg
```

### Context
Needs MTP **and** DP-attention **and** an idle rank simultaneously. Vultr
reproduced it twice (125 s and 766 s into two independent Case A runs); Phase 1's
660 fixed-shape requests never hit it, because `--random-range-ratio 1.0` keeps
batch shapes homogeneous and ranks rarely go idle mid-flight.

**On this cluster it was applied pre-emptively** on the strength of the vultr
evidence plus the md5 match. Result: **0 occurrences of `Expected lengths.size`
across the full 4,006 s window.** So this kit does *not* independently reproduce
the bug — it inherits the diagnosis and confirms the fix is harmless and the
symptom absent.

### Scope and risk, stated plainly
- This modifies **engine code inside the image under test**. Every number in this
  kit is `merged-mtp + P1V3`, not stock.
- The **prefill leg is unpatched** — the bug is on the MTP draft path, decode only.
- `ke_offset` is an existing parameter of `DSAIndexerMetadata.topk_transform`,
  so the fix rides an intended extension point rather than reaching into internals.
- The original is preserved at `/tmp/dsa_indexer.py.orig` in the container for a
  revert-based A/B.

**Upstream status.** Not filed. It should be — the image already ships the
instrumentation that proves the bug (`SGLANG_DEBUG_DSA_ROWS`, `dsa_indexer.py:63`)
and a comment conceding the two bookkeeping sources were never verified to agree
on this path. Patch comments reference upstream **#32762** (NPU, same bug class).
