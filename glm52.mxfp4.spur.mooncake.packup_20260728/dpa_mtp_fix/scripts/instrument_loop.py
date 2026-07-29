#!/usr/bin/env python3
"""Instrument the PD-decode event loop to see HOW the 8 DP ranks diverge.

WHY
---
With Fix A/A2 (dsa_backend) the DSA deadlock is gone, and the hoist experiment on
overlap_utils showed the sync's *position* is not the problem. The surviving hang
has a stable shape, sampled repeatedly:

    DP<one>:   resolve_seq_lens_cpu   (overlap_utils.py:293)   <- inside run_batch
    DP<seven>: process_batch_result_idle (batch_result_processor.py:623)

Note WHERE these are in event_loop_overlap_disagg_decode:

    batch  = self.get_next_disagg_decode_batch_to_run()      # 1838
    ...
    if batch:
        batch_result = self.run_batch(batch)                 # 1848  <- the ONE rank
    ...
    if self.last_batch:
        if not disable_overlap_for_batch:
            pop_and_process()                                # 1856  <- the SEVEN

So the seven are a full half-iteration AHEAD: they are processing the previous
batch while the one is still launching the current one. The question this probe
answers is *why the loop lets them get there* -- i.e. which of the per-rank
predicates differs:

    batch is None                  -> skips run_batch entirely
    self.last_batch is None        -> skips pop_and_process
    disable_overlap_for_batch      -> reorders pop_and_process vs run_batch
    is_extend_in_batch / forward_mode
    global_num_tokens (the DP all-gathered work distribution)

`maybe_prepare_mlp_sync_batch` (decode.py:1908) is supposed to make every rank
agree on shape; if it is doing its job, `batch is None` should be rank-uniform and
the divergence must come from `last_batch` / `disable_overlap_for_batch` instead.

WHAT IT LOGS
------------
One line per rank per iteration, at the top of the loop body right after the
overlap decision is computed, plus a line just before pop_and_process and before
run_batch. Tagged so they can be grepped and aligned by iteration number.

This is READ-ONLY instrumentation -- it changes no control flow.

Idempotent. --revert restores. py_compile-checked.
"""
import argparse
import os
import shutil
import sys

DEC = "/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py"
BACKUP_SUFFIX = ".instr_loop_orig"
MARKER = "GLM52_LOOP_PROBE"

ANCHOR = """            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch
            # overlap + spec + grammar is unsupported (would desync DP ranks).
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            if disable_overlap_for_batch and self.last_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                self.on_idle()
"""

REPLACEMENT = '''            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch
            # overlap + spec + grammar is unsupported (would desync DP ranks).
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            # ''' + MARKER + '''  (debug only, read-only)
            try:
                import logging as _lg
                _it = getattr(self, "_glm52_it", 0) + 1
                self._glm52_it = _it
                _b, _lb = batch, self.last_batch
                def _d(x):
                    if x is None:
                        return "None"
                    return "mode=%s|bs=%s|ext_in_batch=%s|gnt=%s" % (
                        getattr(x, "forward_mode", "?"),
                        len(getattr(x, "reqs", []) or []),
                        getattr(x, "is_extend_in_batch", "?"),
                        getattr(x, "global_num_tokens", None),
                    )
                _lg.getLogger(__name__).warning(
                    "[''' + MARKER + '''] it=%s dp=%s batch=(%s) last=(%s) "
                    "disable_overlap=%s qlen=%s -> will_run=%s will_pop=%s",
                    _it, getattr(getattr(self, "ps", None), "dp_rank", "?"),
                    _d(_b), _d(_lb), disable_overlap_for_batch,
                    len(self.result_queue),
                    bool(_b), bool(_lb),
                )
            except Exception:
                pass

            if disable_overlap_for_batch and self.last_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                self.on_idle()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(DEC):
        sys.exit(f"FAIL: {DEC} not found")

    backup = DEC + BACKUP_SUFFIX
    src = open(DEC).read()

    if args.revert:
        if not os.path.exists(backup):
            sys.exit("FAIL: no instr_loop backup to revert to")
        shutil.copyfile(backup, DEC)
        print(f"OK: reverted {DEC}")
        return

    if MARKER in src:
        print("OK: loop probe already present (no-op)")
        return

    n = src.count(ANCHOR)
    if n != 1:
        sys.exit(f"FAIL: anchor matched {n} times, expected 1. Source drifted.")

    if not os.path.exists(backup):
        shutil.copyfile(DEC, backup)
        print(f"OK: backup -> {backup}")

    open(DEC, "w").write(src.replace(ANCHOR, REPLACEMENT, 1))

    import py_compile
    try:
        py_compile.compile(DEC, doraise=True)
    except Exception as e:
        shutil.copyfile(backup, DEC)
        sys.exit(f"FAIL: broke syntax, reverted. {e}")
    print(f"OK: loop probe installed in {DEC}")


if __name__ == "__main__":
    main()
