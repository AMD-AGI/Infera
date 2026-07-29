#!/usr/bin/env python3
"""Bug 4 fix (EXPERIMENT, two parts): make the publish_ready event exist and be
waited on by EVERY DP rank, not only the ranks that hold work.

THE BUG
-------
`FutureMap.publish()` early-returns for DP-idle ranks BEFORE it lazily constructs
the event:

    def publish(self, future_indices, new_seq_lens):
        indices = future_indices
        if indices.shape[0] == 0:
            return                                  # <-- DP idle leaves here
        self.new_seq_lens_buf[indices] = ...
        if self.spec_algo.is_some():
            if self.publish_ready is None:
                self.publish_ready = ...Event()     # <-- never reached when idle
            self.publish_ready.record()

So on an idle rank `self.publish_ready` stays **None forever**. Then in
`resolve_seq_lens_cpu` the guard `if self.publish_ready is not None:` is False on
exactly those ranks, so they skip the wait entirely while busy ranks block in a
HIP-only *host-blocking* `Event.synchronize()`. Busy ranks stall; idle peers sail
into the next collective. Classic rank-divergent sync.

Verified by the upstream survey (see ECOSYSTEM_glm52_amd_mtp.md): twelve PRs touch
this file, **none** changes the `if indices.shape[0] == 0: return  # DP idle` early
return, and the HIP `synchronize()` branch is unchanged in main and v0.5.16. This
appears to be unreported.

WHY THE EARLIER HOIST EXPERIMENT FAILED
---------------------------------------
`hoist_sync.py` moved the sync above the early returns in resolve_seq_lens_cpu, so
every rank *executed* the guard -- but the guard tests `publish_ready is not None`,
and on an idle rank it is still None. Moving the sync did nothing because **the
guard itself is the divergence**. Both parts below are therefore required.

PART 1 (publish) -- record the event on every rank
--------------------------------------------------
Restructure so the buffer write stays gated on having work, while the event is
created and recorded unconditionally. Ordering is preserved: the record still
happens AFTER any write, so waiting on the event still guarantees the write is
visible. Recording an event on an idle rank is harmless -- it just marks a point
in that rank's stream, which carries the IDLE forward.

PART 2 (resolve) -- wait on it before the early returns
--------------------------------------------------------
Same change hoist_sync.py made, now meaningful because Part 1 guarantees the event
exists on idle ranks too.

COST
----
Idle ranks now pay one host-blocking event sync per step. Under DP attention those
ranks are already obliged to wait at the next collective, so the expected
throughput cost is near zero -- that is the premise being tested.

STATUS: experiment, not shippable as-is.
A proper fix would likely avoid HIP's host-blocking `synchronize()` (CUDA uses the
non-blocking `Event.wait()`; the HIP branch is a TPOT workaround from #26672)
rather than making every rank block. Ship only after measuring TPOT.

Idempotent. --revert restores. py_compile-checked.
"""
import argparse
import os
import shutil
import sys

OU = "/sgl-workspace/sglang/python/sglang/srt/managers/overlap_utils.py"
BACKUP_SUFFIX = ".fix_bug4_orig"
MARKER = "GLM52_BUG4_UNIFORM_EVENT"

# ---- Part 1: publish() records the event on every rank ----
ANCHOR_PUBLISH = """        indices = future_indices
        if indices.shape[0] == 0:
            return  # DP idle
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        # Only spec_v2 needs the event; it gates the seq_lens D2H on the private stream.
        if self.spec_algo.is_some():
            if self.publish_ready is None:
                self.publish_ready = torch.get_device_module(self.device).Event()
            self.publish_ready.record()
"""

REPLACEMENT_PUBLISH = """        indices = future_indices
        # """ + MARKER + """ (part 1/2): a DP-idle rank used to return here, before
        # the event was lazily constructed, leaving self.publish_ready None on that
        # rank forever. resolve_seq_lens_cpu then skipped its wait while busy ranks
        # blocked in the HIP host-side Event.synchronize() -- a sync on a branch only
        # some ranks take, which desynchronizes the DP collectives and deadlocks.
        # The buffer write stays gated on actually having work; only the event is
        # made unconditional. The record still happens AFTER any write, so waiting on
        # the event still guarantees that write is visible.
        if indices.shape[0] != 0:  # DP idle ranks have nothing to publish
            self.new_seq_lens_buf[indices] = new_seq_lens.to(
                self.new_seq_lens_buf.dtype
            )
        # Only spec_v2 needs the event; it gates the seq_lens D2H on the private stream.
        if self.spec_algo.is_some():
            if self.publish_ready is None:
                self.publish_ready = torch.get_device_module(self.device).Event()
            self.publish_ready.record()
"""

# ---- Part 2: resolve_seq_lens_cpu waits before the early returns ----
ANCHOR_RESOLVE = """        draft_input = batch.spec_info
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

REPLACEMENT_RESOLVE = """        # """ + MARKER + """ (part 2/2): hoisted above the two early returns
        # below so every DP rank waits, not just the ones holding work. This is only
        # meaningful together with part 1 -- previously an idle rank reached this
        # guard with publish_ready still None and skipped the wait anyway.
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
"""

PATCHES = [
    ("part 1: publish() records on every rank", ANCHOR_PUBLISH, REPLACEMENT_PUBLISH),
    ("part 2: resolve() waits before early returns", ANCHOR_RESOLVE, REPLACEMENT_RESOLVE),
]


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
            sys.exit("FAIL: no fix_bug4 backup to revert to")
        shutil.copyfile(backup, OU)
        print(f"OK: reverted {OU}")
        return

    if MARKER in src:
        print("OK: Bug 4 uniform-event fix already present (no-op)")
        return

    for name, anchor, _ in PATCHES:
        n = src.count(anchor)
        if n != 1:
            sys.exit(f"FAIL: {name}: anchor matched {n} times, expected 1. "
                     "Source drifted (is hoist_sync.py still applied?).")

    if not os.path.exists(backup):
        shutil.copyfile(OU, backup)
        print(f"OK: backup -> {backup}")

    out = src
    for name, anchor, repl in PATCHES:
        out = out.replace(anchor, repl, 1)
        print(f"OK: applied {name}")

    open(OU, "w").write(out)

    import py_compile
    try:
        py_compile.compile(OU, doraise=True)
    except Exception as e:
        shutil.copyfile(backup, OU)
        sys.exit(f"FAIL: broke syntax, reverted. {e}")

    chk = open(OU).read()
    if "            return  # DP idle" in chk.split("def stash")[0]:
        shutil.copyfile(backup, OU)
        sys.exit("FAIL: publish()'s DP-idle early return survived, reverted")
    if chk.count("self.publish_ready.record()") != 1:
        shutil.copyfile(backup, OU)
        sys.exit("FAIL: expected exactly one record() site, reverted")
    print(f"OK: Bug 4 uniform-event fix installed in {OU}")


if __name__ == "__main__":
    main()
