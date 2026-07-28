#!/usr/bin/env python3
"""Bug 2 candidate fix: make the draft-extend init_forward_metadata branch
collective-uniform across DP ranks.

ROOT CAUSE (proven 2026-07-28 by rank-tagged instrumentation, 56/56 samples):

In `base_spec_worker.prepare_for_draft_extend`:

    can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(fb)
    if not batch.forward_mode.is_idle() and not can_cuda_graph:
        draft_model_runner.attn_backend.init_forward_metadata(forward_batch)

`can_cuda_graph` is always False on this HIP build (the draft-extend CUDA graph
is never captured -- see eagle_worker_v2.py:441-482), so the branch reduces to
`if not is_idle()`.

Measured invariant, zero mismatches over 56 probe samples:

    is_idle(rank) == (global_num_tokens_cpu[rank] == 0)

So when the PD decode leg has work for only SOME DP ranks -- e.g.
global_num_tokens_cpu = [0,0,4,0,0,0,0,0] -- exactly one rank enters
init_forward_metadata and the other seven skip it. init_forward_metadata is not
collective-free: it reaches DSA indexer metadata construction and a
`.max().item()` device sync. The busy rank blocks there while its peers race
ahead into the NEXT collective, so the ranks end up in three different places:

    DP2 (the only busy rank): init_forward_metadata (dsa_backend.py:746)
    DP1,3,4,5,7:              all_gather_into_tensor
    DP0,6:                    broadcast

-> ragged collective -> hard deadlock on the first routed request.
Confirmed reproducible: the stuck rank follows whichever rank owns the work
(DP1 in the first capture, DP2 in the second), so it is not a fixed-rank issue.

THE FIX: under DP-attention the branch must be taken by all ranks or none.
Every rank already knows the global picture via `forward_batch.global_num_tokens_cpu`,
so no extra communication is needed -- derive the predicate from the GLOBAL state
instead of the LOCAL one:

    any_rank_needs_metadata = max(global_num_tokens_cpu) > 0

An idle rank calling init_forward_metadata is safe: it builds metadata for a
zero-row batch (bs=0), which is exactly what the padded/IDLE path already does
elsewhere in the DP-attention flow.

Idempotent. --revert restores.
"""
import argparse
import os
import shutil
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/speculative/base_spec_worker.py"
BACKUP_SUFFIX = ".bug2fix_orig"
MARKER = "GLM52_BUG2_UNIFORM_BRANCH"

ANCHOR = """        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
"""

REPLACEMENT = '''        # ''' + MARKER + ''': make this branch collective-uniform across DP ranks.
        #
        # `init_forward_metadata` is not collective-free (DSA indexer metadata +
        # a .max().item() device sync). Under DP-attention the per-rank predicate
        # `not is_idle()` is TRUE only on ranks that happen to hold work for this
        # step -- measured invariant: is_idle(rank) == (global_num_tokens_cpu[rank] == 0).
        # When a PD decode step has work for a subset of ranks (e.g. gnt =
        # [0,0,4,0,0,0,0,0]) the busy rank enters this call while the idle ranks
        # skip it and run ahead into the next collective, leaving the group split
        # across broadcast / all_gather / init_forward_metadata -> deadlock.
        #
        # Every rank already has the global token counts, so agreeing costs no
        # extra communication. Idle ranks build zero-row metadata, which the
        # DP-padded/IDLE path already handles.
        _gnt = getattr(forward_batch, "global_num_tokens_cpu", None)
        if _gnt:
            _needs_metadata = max(_gnt) > 0
        else:
            _needs_metadata = not batch.forward_mode.is_idle()
        if _needs_metadata and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(TARGET):
        sys.exit(f"FAIL: target not found: {TARGET}")

    backup = TARGET + BACKUP_SUFFIX
    src = open(TARGET).read()

    if args.revert:
        if not os.path.exists(backup):
            sys.exit(f"FAIL: no backup at {backup}")
        shutil.copyfile(backup, TARGET)
        print(f"OK: reverted {TARGET}")
        return

    if MARKER in src:
        print("OK: fix already present (no-op)")
        return

    n = src.count(ANCHOR)
    if n != 1:
        sys.exit(f"FAIL: anchor matched {n} times, expected 1. Source drifted "
                 f"(is the divergence probe still installed? revert it first).")

    if not os.path.exists(backup):
        shutil.copyfile(TARGET, backup)
        print(f"OK: backup -> {backup}")

    out = src.replace(ANCHOR, REPLACEMENT, 1)
    open(TARGET, "w").write(out)

    import py_compile
    try:
        py_compile.compile(TARGET, doraise=True)
    except Exception as e:
        shutil.copyfile(backup, TARGET)
        sys.exit(f"FAIL: broke syntax, reverted. {e}")

    print(f"OK: fix installed in {TARGET}")


if __name__ == "__main__":
    main()
