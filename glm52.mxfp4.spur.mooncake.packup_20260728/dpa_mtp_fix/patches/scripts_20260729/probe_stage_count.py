#!/usr/bin/env python3
"""Probe: measure WHY DP ranks reach different MTP pipeline stages.

Every prior fix addressed how to RE-synchronize after divergence.  None
measured why divergence happens.  This probe measures it directly.

The observed hang (py-spy, 2026-07-29 11:2x) is:

    DP0, DP1 : draft_forward            (eagle_worker_v2.py:697)  _execute_decode
    DP3, DP5 : _draft_extend_for_decode (eagle_worker_v2.py:965)  _execute_idle
    ALL 8    : blocked in the SAME collective --
               logits_processor.py:930 -> tensor_model_parallel_all_gather
               (the LM-head vocab all-gather, over the FULL tp_group)

A single collective that all ranks enter from different call sites means the
per-rank *count* of preceding collectives diverged: rank A is issuing its Nth
all-gather while rank B is issuing its (N+1)th.  NCCL/RCCL matches by order,
not by identity, so they deadlock.

So the question is not "which stage is each rank in" (py-spy already told us)
but "how many LM-head all-gathers has each rank issued, and at which stage did
the counts drift apart".

Instruments:
  (1) LogitsProcessor._get_logits -- a global monotonic counter of LM-head
      all-gathers per rank, logged with the current MTP stage.  This is the
      collective whose count must agree.  If the counts differ, we see exactly
      which iteration and which stage introduced the skew.
  (2) EAGLEWorkerV2.forward_batch_generation -- stage entry/exit, with the
      batch shape that drives the branch decisions (forward_mode, bs,
      spec_num_steps).  Tells us WHICH branch produced the extra/missing
      collective.
  (3) EagleDraftWorker.draft_forward -- the per-step loop trip count.  If
      speculative_num_steps ever differs per rank, the draft loop issues a
      different number of forwards -> different collective count.

Everything is written to /tmp/stage_probe_dp{rank}.log, one line per event,
so the logs can be diffed rank-against-rank.

Idempotent: re-running restores from the .stage_probe_orig backups first.
"""

import os
import re
import shutil
import sys

SGL = "/sgl-workspace/sglang/python/sglang/srt"
LOGITS = f"{SGL}/layers/logits_processor.py"
EAGLE = f"{SGL}/speculative/eagle_worker_v2.py"

SUFFIX = ".stage_probe_orig"


def backup_or_restore(path):
    """Return the pristine content, keeping a one-time backup."""
    bak = path + SUFFIX
    if os.path.exists(bak):
        shutil.copy2(bak, path)
        print(f"  restored {path} from backup")
    else:
        shutil.copy2(path, bak)
        print(f"  backed up {path}")
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Shared probe runtime, injected into logits_processor.py (imported early and
# by everything else).
# ---------------------------------------------------------------------------
RUNTIME = '''
# ===================== STAGE PROBE RUNTIME (debug only) =====================
import os as _sp_os
import threading as _sp_threading

_SP_STATE = {
    "agc": 0,          # LM-head all-gather count on this rank
    "stage": "boot",   # current MTP pipeline stage
    "iter": 0,         # scheduler iteration
    "fh": None,
    "rank": None,
    "lock": _sp_threading.Lock(),
}


def _sp_rank():
    """Resolve this process's DP rank.

    The scheduler sets its process title to "sglang::scheduler_DP<n>_TP<n>_EP<n>"
    (scheduler.py:4262), which is authoritative and always present -- unlike
    SGLANG_DP_RANK, which this deployment does not export.  Falls back to the
    global distributed rank, then to -1.
    """
    if _SP_STATE["rank"] is None:
        import re as _sp_re

        r = None
        # setproctitle rewrites argv, so the full title is in cmdline.
        # (/proc/self/comm is capped at 15 bytes and would truncate the DP id.)
        try:
            with open("/proc/self/cmdline", "rb") as _f:
                _title = _f.read().replace(b"\\x00", b" ").decode(errors="replace")
            m = _sp_re.search(r"DP(\\d+)", _title)
            if m:
                r = m.group(1)
        except Exception:
            pass
        if r is None:
            try:
                import setproctitle as _spt

                m = _sp_re.search(r"DP(\\d+)", _spt.getproctitle())
                if m:
                    r = m.group(1)
            except Exception:
                pass
        if r is None:
            try:
                import torch.distributed as _d

                r = _d.get_rank() if _d.is_initialized() else -1
            except Exception:
                r = -1
        _SP_STATE["rank"] = int(r)
    return _SP_STATE["rank"]


def _sp_log(event, **kw):
    """Append one event line.  Never raise -- a probe must not change control flow."""
    try:
        with _SP_STATE["lock"]:
            if _SP_STATE["fh"] is None:
                path = f"/tmp/stage_probe_dp{_sp_rank()}.log"
                _SP_STATE["fh"] = open(path, "a", buffering=1)
            kws = " ".join(f"{k}={v}" for k, v in kw.items())
            _SP_STATE["fh"].write(
                f"it={_SP_STATE['iter']} agc={_SP_STATE['agc']} "
                f"stage={_SP_STATE['stage']} {event} {kws}\\n"
            )
    except Exception:
        pass


def _sp_set_stage(stage):
    _SP_STATE["stage"] = stage


def _sp_bump_iter():
    _SP_STATE["iter"] += 1
# =================== END STAGE PROBE RUNTIME ================================
'''


def patch_logits():
    """Count every LM-head all-gather -- the collective that deadlocks."""
    print(f"[1/3] {LOGITS}")
    src = backup_or_restore(LOGITS)

    # Inject the runtime after the last top-level import block.
    anchor = "class LogitsProcessorOutput"
    idx = src.index(anchor)
    # walk back to the start of that line's preceding decorator/comment block
    line_start = src.rindex("\n@", 0, idx)
    src = src[:line_start] + "\n" + RUNTIME + src[line_start:]

    # Instrument the all-gather itself.
    old = """        if self.do_tensor_parallel_all_gather:
            if self.use_attn_tp_group:
                logits = self._gather_attn_tp_logits(logits)
            else:
                logits = self._logits_gatherer(logits)"""
    new = """        if self.do_tensor_parallel_all_gather:
            _SP_STATE["agc"] += 1
            _sp_log(
                "AG_ENTER",
                shape=tuple(logits.shape),
                attn_tp=self.use_attn_tp_group,
            )
            if self.use_attn_tp_group:
                logits = self._gather_attn_tp_logits(logits)
            else:
                logits = self._logits_gatherer(logits)
            _sp_log("AG_EXIT", shape=tuple(logits.shape))"""
    assert src.count(old) == 1, f"logits all-gather anchor count={src.count(old)}"
    src = src.replace(old, new)

    with open(LOGITS, "w") as f:
        f.write(src)
    print("  + runtime, + AG_ENTER/AG_EXIT counter")


def patch_eagle():
    """Tag the MTP pipeline stages and the draft loop trip count."""
    print(f"[2/3] {EAGLE} (stages)")
    src = backup_or_restore(EAGLE)

    imp = "from sglang.srt.layers.logits_processor import _sp_log, _sp_set_stage, _sp_bump_iter, _SP_STATE\n"
    # place the import right before the first class definition
    first_class = src.index("\nclass ")
    src = src[: first_class + 1] + imp + src[first_class + 1 :]

    # -- stage: draft ------------------------------------------------------
    old = """                    verify_input: EagleVerifyInput = self.draft_worker.draft(batch)"""
    new = """                    _sp_set_stage("draft")
                    _sp_log(
                        "STAGE_ENTER",
                        mode=str(batch.forward_mode),
                        bs=batch.seq_lens.shape[0],
                        steps=self.speculative_num_steps,
                    )
                    verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
                    _sp_log("STAGE_EXIT")"""
    assert src.count(old) == 1, f"draft anchor count={src.count(old)}"
    src = src.replace(old, new)

    # -- iteration boundary + stage: verify --------------------------------
    old = """            batch_output = self.verify(batch)"""
    new = """            _sp_set_stage("verify")
            _sp_log("STAGE_ENTER")
            batch_output = self.verify(batch)
            _sp_log("STAGE_EXIT")"""
    assert src.count(old) == 1, f"verify anchor count={src.count(old)}"
    src = src.replace(old, new)

    # -- stage: draft_extend ----------------------------------------------
    old = """                    self.draft_worker._draft_extend_for_decode(batch, batch_output)"""
    new = """                    _sp_set_stage("draft_extend")
                    _sp_log("STAGE_ENTER")
                    self.draft_worker._draft_extend_for_decode(batch, batch_output)
                    _sp_log("STAGE_EXIT")"""
    assert src.count(old) == 1, f"draft_extend anchor count={src.count(old)}"
    src = src.replace(old, new)

    # -- iteration boundary: top of the decode branch ----------------------
    old = """            self.activate_step_by_batch(batch.seq_lens.shape[0])"""
    new = """            _sp_bump_iter()
            _sp_set_stage("iter_top")
            _sp_log(
                "ITER",
                mode=str(batch.forward_mode),
                bs=batch.seq_lens.shape[0],
                steps=self.speculative_num_steps,
                spec_info=type(batch.spec_info).__name__,
            )
            self.activate_step_by_batch(batch.seq_lens.shape[0])"""
    assert src.count(old) == 1, f"iter anchor count={src.count(old)}"
    src = src.replace(old, new)

    # -- draft loop trip count --------------------------------------------
    print(f"[3/3] {EAGLE} (draft loop)")
    old = """        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens("""
    new = """        _sp_log(
            "DRAFT_LOOP",
            n=self.speculative_num_steps,
            idle=forward_batch.forward_mode.is_idle(),
        )
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens("""
    assert src.count(old) == 1, f"draft loop anchor count={src.count(old)}"
    src = src.replace(old, new)

    with open(EAGLE, "w") as f:
        f.write(src)
    print("  + STAGE_ENTER/EXIT x3, + ITER, + DRAFT_LOOP")


def main():
    if "--revert" in sys.argv:
        for p in (LOGITS, EAGLE):
            bak = p + SUFFIX
            if os.path.exists(bak):
                shutil.copy2(bak, p)
                print(f"reverted {p}")
        return

    patch_logits()
    patch_eagle()

    # Syntax-check both files before anyone tries to boot on them.
    import py_compile

    for p in (LOGITS, EAGLE):
        try:
            py_compile.compile(p, doraise=True)
            print(f"OK  syntax {p}")
        except py_compile.PyCompileError as e:
            print(f"FAIL syntax {p}: {e}")
            sys.exit(1)

    print("\nprobe installed. logs -> /tmp/stage_probe_dp{0..7}.log")


if __name__ == "__main__":
    main()
