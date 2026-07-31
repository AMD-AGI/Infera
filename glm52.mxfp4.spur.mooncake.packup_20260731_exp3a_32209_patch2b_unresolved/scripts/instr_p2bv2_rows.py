#!/usr/bin/env python3
"""E3a diagnostic #2: is the patch-2b trim CAUSING the row divergence, or
merely coexisting with it?

WHAT THE FIRST ROUND ESTABLISHED (arm e3a, decode.log, 2026-07-30 11:23:19,
94 GLM52_E3INSTR records):

    rank=0..5  global=(32,6144) local=(4,6144) ratio=8.0 pad_mode=1 fwd_mode=2
    rank=6,7   global=(32,6144) local=(6,6144) ratio=5.33 pad_mode=1 fwd_mode=2
    -> ValueError: output tensor size must be equal to world_size times input

Global buffer is 32 rows = 8 ranks x 4. Ranks 6 and 7 hand the collective SIX
rows into a slot planned for FOUR. Two of the three earlier candidates die on
this record:

  C2 (ranks disagree on DpPaddingMode -- charter H3) is REFUTED for this
     crash: `pad_mode=1` (MAX_LEN) on all eight ranks, including 6 and 7.
     The modes agree; the row counts do not.

  "the draft graph/eager vote diverged" is REFUTED: all eight ranks reached
     this eager Python site on the same iteration, so patch 4's vote held.
     What diverged is the row count, which the vote does not cover.

  C1 (trim/restore leaks) is WEAKENED but not dead. A leak makes the tensor
     SHORTER by pad_rows; here it is LONGER than the planned slot. A leak
     cannot add rows. But the restore CAN: it re-pads to the physical count,
     so if `physical` is itself wrong the restore faithfully reproduces a
     wrong row count.

THE REMAINING QUESTION, and it is binary:

  Q. On the crashing iteration, did the trim fire at all?

     If pad_rows == 0 on every layer, patch 2b is INERT on that iteration and
     the 6-vs-4 divergence is upstream of it -- the batch grew between the
     MLP-sync all-gather (which planned max_len=4) and the draft forward
     (which ran 6 rows). Patch 2b would then be a bystander, and the real
     defect is a scheduler-time staleness that #32209 addresses on its own
     path via `_slice_draft_output_to_local_tokens` -- a hunk this port
     deliberately did NOT carry (verified absent: grep = 0 in
     eagle_worker_v2.py on 14321).

     If pad_rows > 0, the trim is live on the crashing iteration and the
     restore is what materializes the 6 rows. Patch 2b is then causal.

NOTE ON THE FREE CONTROL ALREADY IN HAND. The merged e3 arm ran the SAME
patch-2b port and logged 3200 GLM52_E3INSTR records with ZERO non-8.0 ratios.
The difference between the arms is patch 4 only: e3 forced every rank eager
(0.0% draft-graph usage, measured), e3a votes and mostly permits the graph.
So the divergence needs the graph path to be live. That is a fact about when
it appears, not about what produces the extra rows -- hence this probe.

Logs per (rank, call) at the trim site: physical q rows, the page-table row
count the trim targets, resulting pad_rows, and the forward mode. Pairs with
GLM52_E3INSTR by proximity in the log.

Rank-tagged, capped, flushed so a crash cannot swallow the tail.
Idempotent; anchor asserted unique; invalidates the .pyc.
"""
import os
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"
MARK = "GLM52_P2BROWS"
CAP = 4000

HELPER = '''
# GLM52_P2BROWS -- see /shared_nfs/yihou_exp3way/e3/instr_p2bv2_rows.py
_P2BROWS_N = [0]
_P2BROWS_CAP = {cap}


def _p2brows_log(physical, real, pad_rows, forward_batch):
    """Record whether the patch-2b trim fired, and by how much."""
    if _P2BROWS_N[0] >= _P2BROWS_CAP:
        return
    _P2BROWS_N[0] += 1
    try:
        import torch.distributed as _d

        rank = _d.get_rank() if _d.is_initialized() else -1
        fmode = getattr(forward_batch, "forward_mode", None)
        bs = getattr(forward_batch, "batch_size", None)
        print(
            "GLM52_P2BROWS rank={{}} physical={{}} real={{}} pad_rows={{}} "
            "fwd_mode={{}} bs={{}}".format(rank, physical, real, pad_rows, fmode, bs),
            flush=True,
        )
    except Exception as _e:  # never let the probe break the run
        print("GLM52_P2BROWS probe-error {{}}".format(_e), flush=True)

'''.format(
    cap=CAP
)

HELPER_ANCHOR = "def _p2bv2_trim_decode_dp_padding("

# The trim block published by patch2b_32209_style.py. Anchored on the
# assignment that publishes the count to the wrapper, so this probe sees the
# final value regardless of which branch produced it.
ANCHOR = """        # Published for the wrapper installed around forward_decode; see the
        # module docstring of patch2b_32209_style.py.
        self._p2bv2_pad_rows = _p2bv2_pad_rows
"""

REPL = """        # Published for the wrapper installed around forward_decode; see the
        # module docstring of patch2b_32209_style.py.
        self._p2bv2_pad_rows = _p2bv2_pad_rows
        # GLM52_P2BROWS: did the trim fire on this call, and by how much?
        _p2brows_log(
            (q_all if q_all is not None else q_nope).shape[0] + _p2bv2_pad_rows,
            (
                metadata.page_table_1.shape[0]
                if metadata.page_table_1 is not None
                else None
            ),
            _p2bv2_pad_rows,
            forward_batch,
        )
"""


def die(msg):
    print("FAIL: {}".format(msg), file=sys.stderr)
    sys.exit(1)


def main():
    src = open(TARGET).read()

    if MARK in src:
        print("already patched ({} present); nothing to do".format(MARK))
        return
    if "_p2bv2_trim_decode_dp_padding" not in src:
        die(
            "patch 2b (#32209 style) is not applied; this probe only measures "
            "that patch and is meaningless without it"
        )

    n = src.count(ANCHOR)
    if n != 1:
        die("trim-publish anchor matched {} times, expected 1".format(n))
    n = src.count(HELPER_ANCHOR)
    if n != 1:
        die("helper anchor matched {} times, expected 1".format(n))

    src = src.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    src = src.replace(ANCHOR, REPL, 1)

    open(TARGET, "w").write(src)
    # CLAUDE.md: a restored/rewritten .py can keep an mtime that matches the
    # cached bytecode, and CPython then runs the UNPATCHED .pyc. Touch and purge.
    os.utime(TARGET, None)
    pyc = os.path.join(
        os.path.dirname(TARGET), "__pycache__", "dsa_backend.cpython-310.pyc"
    )
    if os.path.exists(pyc):
        os.remove(pyc)
    print("patched {} with {}".format(TARGET, MARK))


if __name__ == "__main__":
    main()
