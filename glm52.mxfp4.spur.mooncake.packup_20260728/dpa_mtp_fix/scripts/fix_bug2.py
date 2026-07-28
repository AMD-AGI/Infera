#!/usr/bin/env python3
"""Bug 2 fix: make the draft-extend init_forward_metadata branch collective-uniform
across DP ranks, and make the DSA backend tolerate the zero-row batch that idle
ranks then bring in.

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
collective-free (DSA indexer metadata + a `.max().item()` device sync). The busy
rank blocks there while its peers race ahead into the NEXT collective, so the
ranks end up in three different places:

    DP2 (the only busy rank): init_forward_metadata (dsa_backend.py:746)
    DP1,3,4,5,7:              all_gather_into_tensor
    DP0,6:                    broadcast

-> ragged collective -> hard deadlock on the first routed request.
The stuck rank follows whichever rank owns the work (DP1 in the first capture,
DP2 in the second), so it is not a fixed-rank issue.

THE FIX, two parts:

(1) base_spec_worker: derive the predicate from the GLOBAL state that every rank
    already has, so agreeing costs no extra communication:

        any_rank_needs_metadata = max(global_num_tokens_cpu) > 0

(2) dsa_backend.init_forward_metadata: part (1) means idle ranks now enter with a
    zero-row batch (bs=0, seq_lens empty). Stock code does
    `forward_batch.seq_lens_cpu.max()` unguarded, which raises

        RuntimeError: max(): Expected reduction dim to be specified for
                      input.numel() == 0

    (observed: DP6 crashed exactly here on the first attempt at this fix). Guard
    both max() sites with an empty check -- max_seqlen_k is meaningless for a
    zero-row batch and 0 is the correct degenerate value.

Idempotent. --revert restores both files.
"""
import argparse
import os
import shutil
import sys

SPEC = "/sgl-workspace/sglang/python/sglang/srt/speculative/base_spec_worker.py"
DSA = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"
BACKUP_SUFFIX = ".bug2fix_orig"
MARKER = "GLM52_BUG2_UNIFORM_BRANCH"
MARKER_DSA = "GLM52_BUG2_EMPTY_BATCH_GUARD"

SPEC_ANCHOR = """        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
"""

SPEC_REPLACEMENT = '''        # ''' + MARKER + ''': make this branch collective-uniform across DP ranks.
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
        # extra communication. Idle ranks build zero-row metadata; the DSA backend
        # is guarded for numel()==0 (see GLM52_BUG2_EMPTY_BATCH_GUARD).
        _gnt = getattr(forward_batch, "global_num_tokens_cpu", None)
        if _gnt:
            _needs_metadata = max(_gnt) > 0
        else:
            _needs_metadata = not batch.forward_mode.is_idle()
        if _needs_metadata and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
'''

DSA_ANCHOR = """        if forward_batch.seq_lens_cpu is not None:
            max_seqlen_k = int(
                forward_batch.seq_lens_cpu.max().item() + draft_token_num
            )
        else:
"""

DSA_REPLACEMENT = '''        if forward_batch.seq_lens_cpu is not None:
            # ''' + MARKER_DSA + ''': under the collective-uniform draft-extend
            # branch, DP-idle ranks reach here with a zero-row batch. .max() on an
            # empty tensor raises; 0 is the correct degenerate width.
            max_seqlen_k = (
                int(forward_batch.seq_lens_cpu.max().item() + draft_token_num)
                if forward_batch.seq_lens_cpu.numel() > 0
                else 0
            )
        else:
'''

DSA_ANCHOR2 = """            max_seqlen_k = int(forward_batch.seq_lens.max().item()) + draft_token_num
"""

DSA_REPLACEMENT2 = '''            max_seqlen_k = (
                int(forward_batch.seq_lens.max().item()) + draft_token_num
                if forward_batch.seq_lens.numel() > 0
                else 0
            )
'''


def patch_file(path, pairs, marker):
    backup = path + BACKUP_SUFFIX
    src = open(path).read()
    if marker in src:
        print(f"OK: {os.path.basename(path)} already patched (no-op)")
        return
    for i, (a, _) in enumerate(pairs):
        n = src.count(a)
        if n != 1:
            sys.exit(f"FAIL: {os.path.basename(path)} anchor#{i} matched {n} times, "
                     f"expected 1. Source drifted (probe still installed?).")
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
        print(f"OK: backup -> {backup}")
    out = src
    for a, r in pairs:
        out = out.replace(a, r, 1)
    open(path, "w").write(out)
    import py_compile
    try:
        py_compile.compile(path, doraise=True)
    except Exception as e:
        shutil.copyfile(backup, path)
        sys.exit(f"FAIL: broke syntax in {path}, reverted. {e}")
    print(f"OK: patched {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    for p in (SPEC, DSA):
        if not os.path.exists(p):
            sys.exit(f"FAIL: target not found: {p}")

    if args.revert:
        for p in (SPEC, DSA):
            b = p + BACKUP_SUFFIX
            if os.path.exists(b):
                shutil.copyfile(b, p)
                print(f"OK: reverted {p}")
            else:
                print(f"skip: no backup for {p}")
        return

    patch_file(SPEC, [(SPEC_ANCHOR, SPEC_REPLACEMENT)], MARKER)
    patch_file(DSA, [(DSA_ANCHOR, DSA_REPLACEMENT),
                     (DSA_ANCHOR2, DSA_REPLACEMENT2)], MARKER_DSA)


if __name__ == "__main__":
    main()
