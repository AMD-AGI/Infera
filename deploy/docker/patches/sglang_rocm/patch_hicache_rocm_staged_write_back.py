#!/usr/bin/env python3
"""Stop the hicache staged write-back JIT kernel from being used on ROCm.

THE BUG
-------
Two gates decide one thing and disagree on ROCm.

`mem_cache/pool_host/mla.py` opts HIP into the staged (layer_first -> page_first)
write-back JIT kernel:

    # The staged write-back JIT kernel builds with hipcc and has a ROCm
    # path, so enable it on HIP too (consistent with the CUDA path).
    self.can_use_write_back_jit = (
        _is_cuda or _is_hip
    ) and can_use_write_back_jit_kernel(...)

`pool_host/mha.py` carries the identical HIP enablement -- #28534 did both -- but
the pools that can share a `HostPoolGroup` with them do not. Three still gate the
flag on `_is_cuda` alone and therefore read False on ROCm: `DSAIndexerPoolHost`,
which a DSA model such as GLM-5.2 always instantiates alongside the MLA pool, plus
`DeepSeekV4PagedHostPool` and `DeepSeekV4StateHostPool`. (`MambaPoolHost` and
`LogicalHostPool` set it True unconditionally, so they are harmless here -- see
SCOPE.) `HostPoolGroup` ANDs the flag over its entries, so one CUDA-only member
makes the group's flag False on ROCm while its anchor entry -- the MLA pool --
still says True.

The two gates feed different decisions:

  * `hybrid_cache_controller.start_writing()` reads the GROUP flag. False means
    "the JIT is not going to run", so it calls `move_hybrid_indices()` and the
    destination host indices end up on the GPU.
  * `HostPoolGroup.backup_from_device_all_layer()` delegates to the ANCHOR pool,
    whose own flag is True, so the MLA pool launches the JIT kernel after all.

The JIT kernel is then handed GPU-resident `dst_indices` where it requires
host-resident ones, and the first write-back after the first reusable prefix
aborts the scheduler:

    File "mem_cache/pool_host/mla.py", line 403, in backup_from_device_all_layer
      jit_transfer_hicache_all_layer_mla_staged_lf_pf(
    tvm.error.InternalError: Tensor match failed for
      Tensor<1152>[strides=<1>, dtype=int64, device=rocm:0]
      at jit_kernel/csrc/kvcacheio/staged_write_back.cuh:248
    - Root cause: Device value [rocm:0] not in the allowed options: [cpu, rocm_host]
    Subprocess scheduler_0 crashed with exit code -3

Measured on MI300X (gfx942, ROCm 7.2.0, sglang v0.5.16) with GLM-5.2-FP8, DSA
attention, dp8, `--hicache-mem-layout page_first --hicache-io-backend kernel`
(both defaults). It is not specific to kvd: any hierarchical cache on this path
dies the moment it writes a page back, so the engine survives startup and then
crashes on the first real request.

THE FIX
-------
Make the anchor agree with the group: gate the staged JIT on CUDA only, exactly as
the three CUDA-only pools above already spell it. The MLA pool then
falls through to `transfer_kv_all_layer_mla_lf_pf` -- the non-JIT kernel in the
same branch, which asserts its destination indices ARE on the GPU, which is where
the controller just put them. Both gates now read False on ROCm and the two
decisions match.

Cost, on any model that reaches this crash: nothing that worked before. Because
`DSAIndexerPoolHost` is CUDA-gated, the group flag is already False on ROCm for
every DSA model, so the staged JIT was unreachable here regardless of this patch --
the `_is_hip` in mla.py did not buy a faster path, it only desynchronized the two
gates.

It is not free everywhere, though. An MLA model in a group whose every other member
sets the flag True unconditionally (`MambaPoolHost`, `LogicalHostPool`) had both
gates reading True on ROCm, so it really did run the staged kernel, and this patch
moves it back to the non-JIT one. That is a throughput question on the write-back
path, not a correctness one -- the non-JIT kernel is what ROCm ran before #28534 --
and no such model is deployed here. Restoring the staged kernel for the DSA path
instead means flipping the CUDA-only pools and proving the JIT correct for them,
which is a much larger change than this one; see SCOPE for who should own it.

WHY NOT A LAUNCH FLAG
---------------------
Two flags dodge the crash and neither is a good trade here.
`--hicache-io-backend direct` leaves the kernel branch altogether for per-layer
`transfer_kv_direct` copies; `--hicache-mem-layout layer_first` is #28473's
workaround, which trips the `layout != "page_first"` early return that zeroes the
flag. Both change how every transfer is done in order to route around one
mis-set boolean, and both leave the disagreement in place for the next model to
find. This patch changes the boolean.

SCOPE -- WHAT THIS DOES NOT FIX
------------------------------
The disagreement is a property of the group, not of MLA, so gating one pool closes
one instance of it. The other reachable instance today is the DeepSeek-V4 hicache
stack: `build_deepseek_v4_hicache_stack` anchors on `LogicalHostPool`, whose flag is
an unconditional True, and hangs `DeepSeekV4PagedHostPool` (CUDA-only, False on
ROCm) off the same group. Same AND, same False group flag, same True anchor, so the
same crash shape should be expected there on gfx942 -- and this patch will not
prevent it, because it only touches mla.py. Left alone deliberately: no V4 stack
runs on this branch, so a gate here would be untested code guarding an untested
path. Named so the next person recognizes the failure instead of re-deriving it.

The mirror case -- an MHA anchor over a CUDA-only member -- is not reachable today:
the CUDA-only pools appear in groups anchored by MLA (DSA sidecar) or by
`LogicalHostPool` (V4), never by `pool_host/mha.py`.

UPSTREAM STATUS (2026-08-03)
  `main` still carries the `_is_cuda or _is_hip` gate (read from the raw file at
  refs/heads/main), so main is affected, not just this base.

  The HIP enablement comes from sglang#28534 "[AMD] Enable JIT staged HiCache
  write-back and fix CPU-index crash" (MERGED 2026-07-09), which fixed the mirror
  image of this crash -- CPU indices reaching a kernel that wanted GPU ones -- and
  deliberately chose parity with CUDA over the #28473 `layer_first` workaround.
  That PR taught `cache_controller.py`, `memory_pool_host.py` and `pool_host/mha.py`
  to agree; it never touched `pool_host/mla.py`, and it predates
  `DSAIndexerPoolHost` joining the same pool group. So this is #28534's fix left
  incomplete on the MLA + DSA path rather than a fresh defect.

  No PR of ours filed yet; it should be, and it should say which side upstream
  wants. There are three repairs, not two: teach the CUDA-only pools the JIT
  (parity, #28534's intent), gate MLA like them (this patch), or make the
  sidecar pools stop poisoning the AND. The third is what upstream already does
  elsewhere -- `MambaPoolHost` sets the flag True unconditionally under the comment
  "Must be True: HostPoolGroup computes can_use_write_back_jit as AND of all pools.
  When True, start_writing() keeps indices on CPU, which MLA's staged write-back
  kernel requires", and routes its own backup by layout + io_backend instead. A
  `DSAIndexerPoolHost` following that precedent would keep the staged kernel on ROCm
  and cost nothing, which makes it the strongest thing to open the PR with; we did
  not take it here only because the patch has to be a one-anchor edit that fails
  loudly, not a behavior change inside a pool we do not otherwise touch.

  Drop this patch once a base sglang makes the two gates agree by construction: the
  anchor stops matching and the build fails, so the drop cannot be silent.

BASES WITHOUT THIS CODE
  If `can_use_write_back_jit` does not appear in mla.py at all, this base predates
  the staged write-back and there is nothing to gate: the patch says so and exits
  0. That is the ONLY tolerated miss. A file that has the flag in an unexpected
  shape exits non-zero, because then the gate silently would not be applied.

Idempotent and self-locating. Run inside the container, then delete stale .pyc.
"""

import importlib.util
import os
import sys

MARKER = "GLM52_ROCM_STAGED_WRITE_BACK"

OLD = """        self.can_use_write_back_jit = (
            _is_cuda or _is_hip
        ) and can_use_write_back_jit_kernel("""

NEW = f"""        # {MARKER}: WHY the ROCm build of this kernel declares dst_indices
        # host-resident while the caller passes a GPU tensor -> "Tensor match
        # failed ... device=rocm:0" kills the scheduler on the first write-back.
        # HOW gate on CUDA like every other host pool; HIP falls through to
        # transfer_kv_all_layer_mla_lf_pf. See infera
        # patch_hicache_rocm_staged_write_back.py.
        {MARKER} = "applied"  # a literal, so `strings *.pyc` can prove it
        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel("""


def find_mla_py() -> str:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("cannot locate the sglang package")
    root = list(spec.submodule_search_locations)[0]
    path = os.path.join(root, "srt", "mem_cache", "pool_host", "mla.py")
    if not os.path.isfile(path):
        print(f"[patch] no {path} on this base — no staged write-back to gate")
        raise SystemExit(0)
    return path


def main() -> int:
    path = find_mla_py()
    src = open(path).read()

    if MARKER in src:
        print(f"[patch] already applied: {path}")
        return 0

    if "can_use_write_back_jit" not in src:
        print(f"[patch] {path} has no staged write-back JIT — nothing to gate")
        return 0

    if OLD not in src:
        print(
            "[patch] ERROR: the can_use_write_back_jit assignment is not in the "
            "expected shape. It is present, so this base DOES have the path that "
            "crashes on ROCm -- refusing to guess. Inspect:"
        )
        print(f"        {path}")
        for i, line in enumerate(src.splitlines(), 1):
            if "can_use_write_back_jit" in line:
                print(f"        {i}: {line}")
        return 1

    open(path, "w").write(src.replace(OLD, NEW, 1))
    print(f"[patch] applied to {path}")

    pyc = os.path.join(os.path.dirname(path), "__pycache__")
    n = 0
    if os.path.isdir(pyc):
        for f in os.listdir(pyc):
            if f.startswith("mla."):
                os.remove(os.path.join(pyc, f))
                n += 1
    print(f"[patch] removed {n} stale .pyc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
