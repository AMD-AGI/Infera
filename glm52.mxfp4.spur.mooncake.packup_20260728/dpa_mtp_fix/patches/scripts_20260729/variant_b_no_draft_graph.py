#!/usr/bin/env python3
"""Variant B: disable the DRAFT cuda graph only, keep every other graph.

Context: `--disable-cuda-graph` (the eager control pair) turns off the target
decode graph, the draft graph AND the draft-extend graph at once, so it proves
"a graph is involved" without saying WHICH one. This variant isolates the draft
graph, which is the one selected by the rank-divergent guard at
eagle_worker_v2.py:517.

Implementation: force `can_cuda_graph = False` unconditionally in `draft()`, so
every rank always takes the eager draft path. This is deliberately blunt -- the
point is to localize, not to be fast.

Distinguishes:
  * hang GONE  -> the draft graph is the culprit; the fix belongs at the guard,
                  and target/draft-extend graphs can stay (most perf retained).
  * hang STAYS -> the draft graph is NOT the culprit; look at the target decode
                  graph or the draft-extend graph instead.

Note the guard already had its `not is_idle()` term removed by
fix_bug2b_uniform_graph.py and that FAILED to stop the hang -- the remaining
terms (`can_cuda_graph` from prepare_for_draft, and `dsa_topk_indices is None`)
are themselves rank-divergent. Forcing the whole decision to a constant removes
every rank-dependent term at once.

Must be applied ON TOP of the existing patches. Handles the .pyc trap.
"""

import os
import py_compile
import shutil
import sys

EAGLE = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
SUFFIX = ".variant_b_orig"

MARKER = "GLM52_VARIANT_B"


def bump_and_purge(path):
    """Defeat the stale-bytecode trap: copy2 preserves mtime, so CPython can
    reuse a .pyc compiled from the UNPATCHED source. This already invalidated
    one full experiment."""
    os.utime(path, None)
    pycache = os.path.join(os.path.dirname(path), "__pycache__")
    if os.path.isdir(pycache):
        base = os.path.basename(path).rsplit(".", 1)[0]
        for fn in os.listdir(pycache):
            if fn.startswith(base + "."):
                os.remove(os.path.join(pycache, fn))


def main():
    bak = EAGLE + SUFFIX
    if "--revert" in sys.argv:
        if os.path.exists(bak):
            shutil.copy2(bak, EAGLE)
            bump_and_purge(EAGLE)
            print(f"reverted {EAGLE}")
        else:
            print("no backup")
        return

    if os.path.exists(bak):
        shutil.copy2(bak, EAGLE)
        print("restored from backup first")
    else:
        shutil.copy2(EAGLE, bak)
        print("backed up")

    with open(EAGLE) as f:
        src = f.read()

    # Anchor on the line right after the guard block, which the probe added.
    anchor = "        _sp_log(\n            \"DRAFT_GRAPH\","
    if src.count(anchor) != 1:
        # Probe not installed; fall back to anchoring on n_inner.
        anchor = "        n_inner = self.speculative_num_steps - 1"
    if src.count(anchor) != 1:
        print(f"FAIL: anchor not unique (count={src.count(anchor)})")
        sys.exit(1)

    inject = (
        "        # " + MARKER + ": force the draft path fully eager on EVERY rank.\n"
        "        # Isolates the draft cuda graph from the target-decode and\n"
        "        # draft-extend graphs, which --disable-cuda-graph turns off together.\n"
        "        can_cuda_graph = False\n"
    )
    src = src.replace(anchor, inject + anchor)

    with open(EAGLE, "w") as f:
        f.write(src)

    bump_and_purge(EAGLE)

    try:
        py_compile.compile(EAGLE, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"FAIL syntax: {e}")
        sys.exit(1)

    print(f"OK applied + syntax clean: {EAGLE}")
    with open(EAGLE) as f:
        lines = f.readlines()
    i = next(i for i, ln in enumerate(lines) if MARKER in ln)
    for j in range(max(0, i - 2), min(len(lines), i + 6)):
        print(f"{j + 1}: {lines[j].rstrip()}")


if __name__ == "__main__":
    main()
