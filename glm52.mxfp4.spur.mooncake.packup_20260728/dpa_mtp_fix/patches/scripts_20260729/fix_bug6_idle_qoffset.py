#!/usr/bin/env python3
"""Bug 6 fix: the Bug-1 slice must also fire when q_offset == 0 (DP-idle rank).

Crash (job 9006, 2026-07-29 13:41, 64 concurrent requests, #running-req reached 8):

    eager_runner.py:409  _execute_idle          <- an IDLE DP rank
      -> deepseek_nextn.py:271                  <- the MTP draft model
      -> forward_mla.py:413  forward_absorb_prepare
      -> dsa_indexer.py:1941 forward_cuda
      -> dsa_indexer.py:989  _get_topk_paged
      -> dsa_backend.py:353  topk_transform
      -> dsa_topk_backend.py:89 -> sgl_kernel fast_topk_v2
    RuntimeError: Expected lengths.size(0) == B to be true, but got false.

## Same defect family as Bug 1 and Bug 5: padded rows vs real rows

The Bug 1 fix slices the aiter MQA-logits inputs down to the real (unpadded)
row count so that `logits` and `lengths` agree:

    _q_mqa, _w_mqa = q_fp8, weights
    if 0 < q_offset < q_fp8.shape[0]:      # <-- the bug is the `0 <`
        _q_mqa = q_fp8[:q_offset]
        _w_mqa = weights[:q_offset]

`q_offset = sum(metadata.get_dsa_extend_len_cpu())` is the real q length. On a
**DP-idle rank there are no requests, so q_offset == 0** — while `q_fp8` still
carries the DP-padded rows. The `0 <` lower bound makes the slice a no-op in
exactly that case, so aiter sizes `logits` from `q_fp8.shape[0]` (padded) while
`lengths` has 0 rows, and the kernel's `lengths.size(0) == B` check fails.

The lower bound was defensive — an attempt to avoid an empty tensor — but
`q_fp8[:0]` is a perfectly legal empty slice and is precisely what an idle rank
should feed the kernel. The CUDA path reaches the same state by always slicing
to `q_offset`, with no lower bound.

This is a **shape** defect, not a data or reachability defect:
  * the error raised is itself a shape assertion (`lengths.size(0) == B`);
  * it fires on `_execute_idle`, i.e. the one path where q_offset is
    structurally 0;
  * it is the same padded-vs-real row mismatch as Bug 1 and Bug 5.

## Fix

Drop the `0 <` lower bound, so the slice fires whenever the tensor is padded
relative to the real row count, including the q_offset == 0 idle case.

The padding-restore below (`if q_offset < q_fp8.shape[0] and
topk_result.shape[0] == q_offset`) already has no lower bound and so already
handles q_offset == 0 correctly; this change makes the two guards consistent.

Idempotent; handles the stale-.pyc trap; verifies by import before returning.
"""

import os
import py_compile
import shutil
import subprocess
import sys

DSA = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"
SUFFIX = ".fix_bug6_orig"

OLD = """            _q_mqa, _w_mqa = q_fp8, weights
            if 0 < q_offset < q_fp8.shape[0]:"""

NEW = """            _q_mqa, _w_mqa = q_fp8, weights
            # GLM52_BUG6: no lower bound on q_offset. A DP-IDLE rank has no
            # requests, so q_offset == 0 while q_fp8 still carries DP padding.
            # The old `0 < q_offset` guard skipped the slice in exactly that
            # case, leaving aiter to size `logits` from the padded row count
            # while `lengths` had 0 rows -> "lengths.size(0) == B" failed in
            # fast_topk_v2. q_fp8[:0] is a legal empty slice and is what an idle
            # rank should pass; the CUDA path likewise slices unconditionally.
            if q_offset < q_fp8.shape[0]:"""


def bump_and_purge(path):
    """copy2 preserves mtime, so CPython can reuse a .pyc built from the
    UNPATCHED source. This already invalidated one full experiment."""
    os.utime(path, None)
    pycache = os.path.join(os.path.dirname(path), "__pycache__")
    if os.path.isdir(pycache):
        base = os.path.basename(path).rsplit(".", 1)[0]
        for fn in os.listdir(pycache):
            if fn.startswith(base + "."):
                os.remove(os.path.join(pycache, fn))


def main():
    bak = DSA + SUFFIX
    if "--revert" in sys.argv:
        if os.path.exists(bak):
            shutil.copy2(bak, DSA)
            bump_and_purge(DSA)
            print(f"reverted {DSA}")
        else:
            print("no backup")
        return

    if os.path.exists(bak):
        shutil.copy2(bak, DSA)
        print("restored from backup first")
    else:
        shutil.copy2(DSA, bak)
        print("backed up")

    with open(DSA) as f:
        src = f.read()

    n = src.count(OLD)
    if n != 1:
        print(f"FAIL: anchor found {n} times, expected 1")
        sys.exit(1)
    src = src.replace(OLD, NEW)

    with open(DSA, "w") as f:
        f.write(src)

    bump_and_purge(DSA)

    try:
        py_compile.compile(DSA, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"FAIL syntax: {e}")
        sys.exit(1)

    # py_compile only proves it parses. Import it too -- a previous patch parsed
    # fine but landed between a decorator and its class, killing every scheduler
    # at import. A 10-minute boot is too expensive a way to find that out.
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '/sgl-workspace/sglang/python');"
            "import sglang.srt.layers.attention.dsa.dsa_indexer as m;"
            "print('IMPORT OK')",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if "IMPORT OK" not in check.stdout:
        print("FAIL import check:")
        for ln in (check.stdout + check.stderr).strip().splitlines()[-8:]:
            print("   ", ln)
        sys.exit(1)

    print(f"OK applied + syntax + import: {DSA}")
    with open(DSA) as f:
        lines = f.readlines()
    i = next(i for i, ln in enumerate(lines) if "GLM52_BUG6" in ln)
    for j in range(i - 1, min(len(lines), i + 10)):
        print(f"{j + 1}: {lines[j].rstrip()}")


if __name__ == "__main__":
    main()
