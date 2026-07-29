#!/usr/bin/env python3
"""FALSIFICATION EXPERIMENT (not a shippable fix): hoist the HIP publish_ready
sync in FutureMap.resolve_seq_lens_cpu above every early return, so that ALL DP
ranks execute it unconditionally.

WHY
---
With Fix A/A2 installed, the DSA deadlock is gone (no rank is in
init_forward_metadata any more, and PD warmup now passes on all 8 dp_ranks), but
a routed request still hangs with the SAME 7-vs-1 shape one layer down:

    DP2:       resolve_seq_lens_cpu   (overlap_utils.py:295)  publish_ready.synchronize()
    DP0,1,3-7: process_batch_result_idle (batch_result_processor.py:623) copy_done.synchronize()

i.e. DP2 is still in run_batch launching, the other seven have already moved on to
pop_and_process for the previous batch. Different phases of the same event-loop
iteration -> the collectives inside the forward cannot pair up.

resolve_seq_lens_cpu has two rank-divergent early returns before the sync:

    if draft_input is None:            return      # <- idle ranks
    if fi is None:                     return      # <- idle ranks
    if self.publish_ready is not None:
        if _is_hip: self.publish_ready.synchronize()   # <- only busy ranks reach

and publish() -- the only place publish_ready is ever created -- itself returns
early for idle ranks:

    if indices.shape[0] == 0:
        return  # DP idle
    ...
    if self.publish_ready is None:
        self.publish_ready = ...Event()
    self.publish_ready.record()

So on an idle rank publish_ready is *never even constructed*: it stays None
forever, the `is not None` guard is False, and the rank sails past while the busy
rank blocks. Same defect class as Bug 2, one layer up the stack.

WHAT THIS DOES
--------------
Moves the sync to the top of the function, before the early returns, and drops the
`is not None` guard's ability to skip it by making idle ranks wait on the event too
(when it exists). This is the "dumbest possible" uniform-entry fix:

  * It costs idle ranks one event wait per step. Under DP attention those ranks
    are already obliged to wait at the next collective, so the expected throughput
    cost is ~zero -- which is the user's point, and why this is worth testing.
  * It is NOT the fix to ship: a rank that never created publish_ready still has
    None and cannot wait on it, so uniformity is only restored for ranks that have
    published at least once. If the hang moves or persists, that asymmetry is the
    next thing to look at.

PURPOSE: falsify or confirm the "rank-divergent sync" hypothesis class as a whole.
  * Hang clears  -> hypothesis class CONFIRMED; design a real uniform-entry fix.
  * Hang persists at a NEW site -> class still alive, keep peeling.
  * Hang persists HERE unchanged -> hypothesis is wrong, rethink.

Idempotent. --revert restores. py_compile-checked.
"""
import argparse
import os
import shutil
import sys

OU = "/sgl-workspace/sglang/python/sglang/srt/managers/overlap_utils.py"
BACKUP_SUFFIX = ".hoist_sync_orig"
MARKER = "GLM52_HOIST_SYNC"

ANCHOR = """        draft_input = batch.spec_info
        if draft_input is None:
            return

        fi = draft_input.future_indices
        if fi is None:
            return
        if self.publish_ready is not None:
            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
        batch.seq_lens = self.new_seq_lens_buf[fi]
"""

REPLACEMENT = '''        # ''' + MARKER + ''': EXPERIMENT -- uniform entry for the publish_ready sync.
        # Hoisted ABOVE the two early returns below so that every DP rank executes
        # it, not just the ranks that happen to hold work this step. Under DP
        # attention the idle ranks must wait at the next collective anyway, so the
        # extra event wait should be ~free. See module docstring.
        if self.publish_ready is not None:
            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()

        draft_input = batch.spec_info
        if draft_input is None:
            return

        fi = draft_input.future_indices
        if fi is None:
            return
        batch.seq_lens = self.new_seq_lens_buf[fi]
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(OU):
        sys.exit(f"FAIL: {OU} not found")

    backup = OU + BACKUP_SUFFIX
    src = open(OU).read()

    if args.revert:
        if not os.path.exists(backup):
            sys.exit("FAIL: no hoist_sync backup to revert to")
        shutil.copyfile(backup, OU)
        print(f"OK: reverted {OU}")
        return

    if MARKER in src:
        print("OK: hoist already present (no-op)")
        return

    n = src.count(ANCHOR)
    if n != 1:
        sys.exit(f"FAIL: anchor matched {n} times, expected 1. Source drifted.")

    if not os.path.exists(backup):
        shutil.copyfile(OU, backup)
        print(f"OK: backup -> {backup}")

    open(OU, "w").write(src.replace(ANCHOR, REPLACEMENT, 1))

    import py_compile
    try:
        py_compile.compile(OU, doraise=True)
    except Exception as e:
        shutil.copyfile(backup, OU)
        sys.exit(f"FAIL: broke syntax, reverted. {e}")
    print(f"OK: hoisted publish_ready sync in {OU}")


if __name__ == "__main__":
    main()
