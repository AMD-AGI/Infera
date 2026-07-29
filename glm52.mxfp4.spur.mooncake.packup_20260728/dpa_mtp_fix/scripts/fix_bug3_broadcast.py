#!/usr/bin/env python3
"""Bug 3 fix: broadcast the finalized EAGLE verify decision on the greedy path too.

Backport of the `eagle_utils.py` hunk of upstream PR #31683
("[ROCm][MI35X] Enable GLM-5.2-MXFP4 MTP speculative decoding", OPEN, base main),
which itself carries the fix from PR #31071 (not merged on any branch).

THE BUG
-------
`eagle_sample()` picks accepted draft tokens on one of two paths:

    if sampling_info.is_all_greedy or _is_npu or _is_hip or _is_xpu:   # line 620
        target_predict = torch.argmax(next_token_logits, dim=-1)       # PER-RANK argmax
        ... verify_tree_greedy_func(...)
    else:
        ... tree_speculative_sampling_target_only(...)
        # Sync sampling results across TP ranks ...                    # lines 718-728
        tp_group.broadcast(predict, src=0)
        tp_group.broadcast(accept_index, src=0)
        tp_group.broadcast(num_correct_drafts, src=0)

`_is_hip` unconditionally forces the greedy path, and the broadcast sits nested
inside the `else:` (sampling) branch -- so on AMD it is NEVER executed. The greedy
path derives the accept decision from a per-rank `torch.argmax` over per-rank
logits. When those logits differ slightly between ranks (non-deterministic
reductions, or ROCm argmax tie-breaking, which differs from CUDA), a near-tie
makes different ranks accept a DIFFERENT NUMBER of draft tokens.

Downstream, seq_lens and batch shapes are derived from the accept count, so the
ranks diverge in shape and their next collective can never pair up. Observed
signature on our cluster (8x MI355, TP8+DPA8+EP8, PD decode leg, EAGLE steps=3):

    ranks with work:  draft() -> _execute_decode
    ranks idle:       _draft_extend_for_decode -> _execute_idle
    all 8 wedged in the same all_gather (dp_gather_replicate), GPUs split 4x100%/4x0%

PR #31071's own commit message describes the same failure verbatim:

    "a near-tie makes argmax pick a different token per rank, so ranks accept a
     different number of drafts, committed seq_lens/batch shapes diverge, and the
     next TP collective deadlocks (both ranks wedge in resolve_seq_lens_cpu ->
     Event.synchronize; /health still 200 until the watchdog fires)."

THE FIX
-------
Move the broadcast OUT of the `else:` branch to after BOTH paths converge -- and
after the SIMULATE_ACC_LEN block, which can re-derive `accept_index` from the same
per-rank argmax. Every path then commits an identical accept decision.

BASELINE DRIFT vs upstream
--------------------------
Upstream's hunk (written against `main`) uses:
    get_parallel().attn_tp_group if is_dp_attention_enabled() else get_tp_group()
Our baseline (0b3bb0cb, release/v0.5.15) has no `get_parallel()`; it uses:
    get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
This backport keeps OUR spelling, so the import set is unchanged. Everything else
matches upstream, including the placement after SIMULATE_ACC_LEN.

COST
----
Three extra small broadcasts per verify step on the greedy path. On AMD that path
is mandatory, so this is a real (small) cost -- and it is exactly what the sampling
path has always paid. Correctness first; see CLAUDE.md ("prefer the naive fix").

Idempotent. --revert restores. py_compile-checked.
"""
import argparse
import os
import shutil
import sys

EU = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_utils.py"
BACKUP_SUFFIX = ".fix_bug3_orig"
MARKER = "GLM52_BUG3_BROADCAST"

# --- 1. remove the broadcast from inside the `else:` (sampling) branch ---
ANCHOR_REMOVE = """        # Sync sampling results across TP ranks: different GPUs may
        # produce slightly different target_probs due to floating-point
        # non-determinism in softmax/top_k/top_p, causing different
        # sampled tokens. Broadcast from rank 0 to ensure consistency.
        tp_group = (
            get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
        )
        if tp_group.world_size > 1:
            tp_group.broadcast(predict, src=0)
            tp_group.broadcast(accept_index, src=0)
            tp_group.broadcast(num_correct_drafts, src=0)

    if SIMULATE_ACC_LEN > 0:
"""

REPLACEMENT_REMOVE = """    if SIMULATE_ACC_LEN > 0:
"""

# --- 2. re-add it after BOTH paths (and after SIMULATE_ACC_LEN) converge ---
ANCHOR_ADD = """    # `num_correct_drafts` stays drafts-only inside this function; the returned
    # tensor includes the trailing/bonus token via out-of-place +1 so the
    # name no longer flips semantics mid-function (naming doc C2).
    return predict, num_correct_drafts + 1, accept_index
"""

REPLACEMENT_ADD = """    # """ + MARKER + """: sync the finalized verify decision across TP ranks.
    # Both the greedy path (per-rank argmax) and the sampling path
    # (softmax/top_k/top_p) can pick different tokens on different ranks when
    # per-rank logits differ from non-deterministic reductions, or from ROCm's
    # argmax tie-breaking. If ranks accept a different number of drafts, committed
    # seq_lens/batch shapes diverge and the next TP collective deadlocks.
    # `_is_hip` forces the greedy path unconditionally, so before this change AMD
    # never broadcast at all. Broadcasting here -- after the accept decision is
    # finalized (including SIMULATE_ACC_LEN, which re-derives from per-rank argmax)
    # and before the worker consumes it -- keeps every path consistent.
    # Backport of the eagle_utils hunk of upstream PR #31683 (carries PR #31071).
    tp_group = (
        get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
    )
    if tp_group.world_size > 1:
        tp_group.broadcast(predict, src=0)
        tp_group.broadcast(accept_index, src=0)
        tp_group.broadcast(num_correct_drafts, src=0)

    # `num_correct_drafts` stays drafts-only inside this function; the returned
    # tensor includes the trailing/bonus token via out-of-place +1 so the
    # name no longer flips semantics mid-function (naming doc C2).
    return predict, num_correct_drafts + 1, accept_index
"""

PATCHES = [
    ("remove broadcast from else-branch", ANCHOR_REMOVE, REPLACEMENT_REMOVE),
    ("re-add broadcast after both paths", ANCHOR_ADD, REPLACEMENT_ADD),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(EU):
        sys.exit(f"FAIL: {EU} not found")

    backup = EU + BACKUP_SUFFIX
    src = open(EU).read()

    if args.revert:
        if not os.path.exists(backup):
            sys.exit("FAIL: no fix_bug3 backup to revert to")
        shutil.copyfile(backup, EU)
        print(f"OK: reverted {EU}")
        return

    if MARKER in src:
        print("OK: Bug 3 broadcast fix already present (no-op)")
        return

    for name, anchor, _ in PATCHES:
        n = src.count(anchor)
        if n != 1:
            sys.exit(f"FAIL: {name}: anchor matched {n} times, expected 1. "
                     "Source drifted -- re-derive the anchor.")

    if not os.path.exists(backup):
        shutil.copyfile(EU, backup)
        print(f"OK: backup -> {backup}")

    out = src
    for name, anchor, repl in PATCHES:
        out = out.replace(anchor, repl, 1)
        print(f"OK: applied {name}")

    open(EU, "w").write(out)

    import py_compile
    try:
        py_compile.compile(EU, doraise=True)
    except Exception as e:
        shutil.copyfile(backup, EU)
        sys.exit(f"FAIL: broke syntax, reverted. {e}")

    chk = open(EU).read()
    # exactly one broadcast trio must remain, and it must be at function scope
    if chk.count("tp_group.broadcast(predict, src=0)") != 1:
        shutil.copyfile(backup, EU)
        sys.exit("FAIL: expected exactly 1 broadcast site after patch, reverted")
    if "        if tp_group.world_size > 1:" in chk:
        shutil.copyfile(backup, EU)
        sys.exit("FAIL: an 8-space-indented broadcast guard survived, reverted")
    print(f"OK: Bug 3 broadcast fix installed in {EU}")


if __name__ == "__main__":
    main()
