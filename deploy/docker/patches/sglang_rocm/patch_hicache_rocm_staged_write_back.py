#!/usr/bin/env python3
"""Stop the hicache staged write-back JIT kernel from being used on ROCm.

THE BUG
-------
Two gates decide one thing and disagree on ROCm.

`mem_cache/pool_host/mla.py` opts HIP into the staged (layer_first -> page_first)
write-back JIT kernel:

    self.can_use_write_back_jit = (
        _is_cuda or _is_hip
    ) and can_use_write_back_jit_kernel(...)

`pool_host/mha.py` carries the identical HIP enablement -- sglang#28534 did both --
but the pools that can share a `HostPoolGroup` with them do not. Three still gate
the flag on `_is_cuda` alone and therefore read False on ROCm: `DSAIndexerPoolHost`,
which a DSA model such as GLM-5.2 always instantiates alongside the MLA pool, plus
`DeepSeekV4PagedHostPool` and `DeepSeekV4StateHostPool`. `HostPoolGroup` ANDs the
flag over its entries, so one CUDA-only member makes the group's flag False on ROCm
while its anchor entry -- the MLA pool -- still says True.

The two gates feed different decisions:

  * `hybrid_cache/hybrid_cache_controller.py :: start_writing()` reads the GROUP
    flag. False means "the JIT is not going to run", so it calls
    `move_hybrid_indices()` and the destination host indices end up on the GPU.
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
(both defaults). Not specific to kvd: any hierarchical cache on this path dies the
moment it writes a page back, so the engine survives startup and then crashes on
the first real request.

THE FIX
-------
Make the anchor agree with the group: gate the staged JIT on CUDA only, exactly as
the three CUDA-only pools above already spell it. The MLA pool then falls through
to `transfer_kv_all_layer_mla_lf_pf` -- the non-JIT kernel in the same branch,
which asserts its destination indices ARE on the GPU, which is where the controller
just put them. Both gates now read False on ROCm and the two decisions match.

On any model that reaches this crash the fix costs nothing that worked before:
`DSAIndexerPoolHost` is CUDA-gated, so the group flag was already False on ROCm for
every DSA model and the staged JIT was unreachable regardless -- the `_is_hip` in
mla.py bought no faster path, it only desynchronized the two gates. It is not free
everywhere, though. An MLA model grouped only with pools that set the flag True
unconditionally (`MambaPoolHost`, `LogicalHostPool`) had both gates reading True on
ROCm and really did run the staged kernel; this patch moves it back to the non-JIT
one. That is a throughput question on the write-back path, not a correctness one --
the non-JIT kernel is what ROCm ran before #28534 -- and no such model is deployed
here.

Two launch flags dodge the crash instead and neither is a good trade:
`--hicache-io-backend direct` leaves the kernel branch altogether for per-layer
`transfer_kv_direct` copies, and `--hicache-mem-layout layer_first` trips the
`layout != "page_first"` early return that zeroes the flag. Both change how every
transfer is done in order to route around one mis-set boolean, and both leave the
disagreement in place for the next model to find.

SCOPE -- WHAT THIS DOES NOT FIX
-------------------------------
The disagreement is a property of the group, not of MLA, so gating one pool closes
one instance of it. The other instance reachable today is the DeepSeek-V4 hicache
stack: `build_deepseek_v4_hicache_stack` anchors on `LogicalHostPool`, whose flag is
an unconditional True, and hangs `DeepSeekV4PagedHostPool` (CUDA-only, False on
ROCm) off the same group -- same AND, same False group flag, same True anchor, so
expect the same crash there on gfx942. Left alone deliberately: no V4 stack runs on
this branch, so a gate here would be untested code guarding an untested path. Named
so the next person recognizes the failure instead of re-deriving it.

UPSTREAM STATUS (2026-08-04)
  The `_is_cuda or _is_hip` gate is still on `main` (read from the raw file), so
  main is affected, not just this base.

  It comes from sglang#28534 "[AMD] Enable JIT staged HiCache write-back and fix
  CPU-index crash" (MERGED 2026-07-09), which fixed the mirror image of this crash
  -- CPU indices reaching a kernel that wanted GPU ones -- and deliberately chose
  parity with CUDA. That PR taught `cache_controller.py`, `memory_pool_host.py` and
  `pool_host/mha.py` to agree; it never touched `pool_host/mla.py`, and it predates
  `DSAIndexerPoolHost` joining the same pool group. So this is #28534 left
  incomplete on the MLA + DSA path rather than a fresh defect.

  A fix is in flight and it is the parity repair done thoroughly: sglang#30350 "Add
  HiCache JIT test and benchmark for ROCm/HIP CI support" (OPEN, Emmanuel0612) adds
  `_is_cuda_alike = _is_cuda or _is_hip` and flips exactly the three CUDA-only gates
  named above, so the group AND stops reading False on ROCm. It also teaches
  `staged_write_back.cuh` to accept kDLROCM / kDLROCMHost -- the TensorMatcher check
  that emits the crash above -- and adds an AMD CI lane (author reports 47/47 on
  MI355X). It therefore also covers the V4 stack that SCOPE leaves open. Stalled
  rather than rejected: amd-bot called its AMD suites green on 07-09 alongside a
  merge conflict, the author cleared conflicts on 07-13, nothing since 07-16, and
  #28534 landed in between. So the useful contribution is a gfx942 reproduction on
  that thread, not a competing PR; file our own only if #30350 dies.

  Anchor drift cannot be the drop signal, which is why this script checks a
  precondition instead. #30350 repairs the OTHER pools and never touches
  `pool_host/mla.py`, so the anchor would keep matching and this patch would keep
  applying on top of the fix -- without crashing, because both gates read False
  again, and therefore without anyone noticing that the staged kernel #30350 enabled
  on ROCm had been given back. `check_group_still_poisoned` closes that: it reads
  `DSAIndexerPoolHost` in `memory_pool_host.py` and refuses to apply once that pool
  stops gating on `_is_cuda` alone (line 1777 at v0.5.16, 1830 on main).

THE TWO IMAGES THAT RUN THIS
  `Dockerfile.sglang.gfx942` (MI300X / MI325X, base v0.5.16) is the one that needs
  it: mla.py carries the `_is_cuda or _is_hip` gate and `DSAIndexerPoolHost` is
  still CUDA-only, so both checks below find what they expect.

  `Dockerfile.sglang` (MI355X, base v0.5.15.post1) exits 0 at the first check and
  has no bug to exit from. Every write-back gate on that base is still `_is_cuda` --
  MHA (196), MLA (1405), both V4 pools (2478, 2887) and `DSAIndexerPoolHost` (3387),
  all in `memory_pool_host.py` before MLA and MHA moved into `pool_host/` -- and
  `_is_hip` appears only in the kernel import guard. So the group's AND and its
  anchor both read False, they agree, and the non-JIT kernel runs. #28534 introduced
  the disagreement after that tag; the absent `pool_host/mla.py` is just how this
  script notices.

EXIT CODES
  0  applied; already applied; or nothing to gate -- no `pool_host/mla.py`, or an
     mla.py with no `can_use_write_back_jit` (a base predating the staged
     write-back). Those are the ONLY tolerated misses.
  1  everything else, all of it meaning "a human has to look":
       * the anchor is present but in an unexpected shape -- the crashing path IS on
         this base and the gate would silently not be applied;
       * `DSAIndexerPoolHost` no longer gates on `_is_cuda` alone -- upstream fixed
         it, so this patch must be dropped rather than applied;
       * that pool, or `memory_pool_host.py`, cannot be found to check at all.

Idempotent and self-locating. Run inside the container, then delete stale .pyc.
"""

import importlib.util
import os
import sys

MARKER = "GLM52_ROCM_STAGED_WRITE_BACK"

# Swallows upstream's comment as well as the assignment: it argues for the HIP
# enablement this patch removes, so leaving it behind would contradict the code.
OLD = """        # The staged write-back JIT kernel builds with hipcc and has a ROCm
        # path, so enable it on HIP too (consistent with the CUDA path).
        self.can_use_write_back_jit = (
            _is_cuda or _is_hip
        ) and can_use_write_back_jit_kernel("""

NEW = f"""        # {MARKER}: WHY the ROCm build of this kernel wants
        # host-resident dst_indices while the caller passes a GPU tensor ->
        # "Tensor match failed ... device=rocm:0" kills the scheduler on the
        # first write-back. HOW gate on CUDA like every other host pool; HIP
        # falls through to transfer_kv_all_layer_mla_lf_pf. See infera
        # patch_hicache_rocm_staged_write_back.py.
        {MARKER} = "applied"  # a literal, so `strings *.pyc` can prove it
        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel("""

# The precondition this patch exists for: a CUDA-only pool in the same
# HostPoolGroup dragging the group's AND to False on ROCm. Checked against the
# pool that does it for every DSA model, in the file it lives in.
MPH_REL = ("srt", "mem_cache", "memory_pool_host.py")
GROUP_MEMBER = "class DSAIndexerPoolHost"
POISON_GATE = "self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel("


def find_pkg_root() -> str:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("cannot locate the sglang package")
    return list(spec.submodule_search_locations)[0]


def find_mla_py(root: str) -> str:
    path = os.path.join(root, "srt", "mem_cache", "pool_host", "mla.py")
    if not os.path.isfile(path):
        print(f"[patch] no {path} on this base — no staged write-back to gate")
        raise SystemExit(0)
    return path


def check_group_still_poisoned(root: str) -> int:
    """Verify the disagreement this patch resolves is still there.

    The OLD anchor cannot be the drop signal on its own: sglang#30350 repairs the
    other side and never touches `pool_host/mla.py`, so the anchor would keep
    matching and this patch would keep applying on top of the fix. That does not
    crash; it silently gates the anchor back down, makes the group read False
    again, and forfeits the staged kernel #30350 just enabled on ROCm.
    """
    path = os.path.join(root, *MPH_REL)
    if not os.path.isfile(path):
        print(
            "[patch] ERROR: mla.py carries the staged write-back but there is no\n"
            f"        {path}\n"
            "        to check the group's other members against. Refusing to\n"
            "        apply a gate whose precondition cannot be verified."
        )
        return 1

    src = open(path).read()
    start = src.find(GROUP_MEMBER)
    if start < 0:
        print(
            f"[patch] ERROR: no `{GROUP_MEMBER}` to check. It has moved or been\n"
            "        renamed, so whether the HostPoolGroup AND still reads False on\n"
            "        ROCm is unknown. Re-derive before shipping — see this script's\n"
            "        UPSTREAM STATUS section.\n"
            f"        Checked: {path}"
        )
        return 1

    end = src.find("\nclass ", start + 1)
    block = src[start:] if end < 0 else src[start:end]
    if POISON_GATE not in block:
        print(
            "[patch] ERROR: DSAIndexerPoolHost no longer gates can_use_write_back_jit\n"
            "        on _is_cuda alone, so it no longer forces the HostPoolGroup AND\n"
            "        to False on ROCm and the two gates already agree without us.\n"
            "        This is what sglang#30350 does.\n"
            "\n"
            "        DROP THIS PATCH — remove it from Dockerfile.sglang.gfx942.\n"
            "\n"
            "        Do NOT re-anchor it instead. #30350 leaves pool_host/mla.py\n"
            "        alone, so the anchor below still matches and applying it would\n"
            "        gate the anchor back down, put the group at False again, and\n"
            "        silently give up the staged write-back kernel on ROCm.\n"
            f"        Checked: {path}"
        )
        return 1

    return 0


def main() -> int:
    root = find_pkg_root()
    path = find_mla_py(root)
    src = open(path).read()

    if "can_use_write_back_jit" not in src:
        print(f"[patch] {path} has no staged write-back JIT — nothing to gate")
        return 0

    # Checked only once the two exits above have established there is something
    # to patch, so a base without the JIT path never reports a drifted anchor.
    rc = check_group_still_poisoned(root)
    if rc:
        return rc

    if MARKER in src:
        print(f"[patch] already applied: {path}")
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
