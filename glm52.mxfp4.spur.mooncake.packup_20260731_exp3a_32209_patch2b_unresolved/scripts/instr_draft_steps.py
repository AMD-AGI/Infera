#!/usr/bin/env python3
"""E3a diagnostic #3: which draft step is rank 4 missing, and why?

MEASUREMENT ONLY. This script changes no fix and no behaviour -- it adds
print statements. It exists to settle one binary question, not to try a
candidate fix.

WHAT ROUNDS 1-2 ESTABLISHED (arm e3a, conc=32, both rounds crash identically
with `output tensor size must be equal to world_size times input tensor size`
in dp_gather_replicate):

  On the crashing iteration (bs=3, buflen=24, plan=[3,3,3,3,3,3,3,3],
  orig_global_num_tokens_cpu=[1,1,1,1,2,1,3,3]):

    rank=0,1,2,3,5,6,7   local=(3,6144)   ratio=8.0   <- correct
    rank=4               local=(4,6144)   ratio=6.0   <- the offender

  and at the patch-2b trim site, same iteration:

    rank=0,1,2,3,5   physical=3 real=1 pad_rows=2   entered TWICE
    rank=4           physical=3 real=2 pad_rows=1   entered ONCE
    rank=6,7         physical=3 real=3 pad_rows=0   (no trim needed)

REFUTED by that data, and NOT to be re-litigated:

  - DpPaddingMode divergence (charter H3): pad_mode=1 (MAX_LEN) on all eight
    ranks, both rounds. The modes agree.
  - patch-2b trim/restore leaking rows: five ranks trimmed 2 rows and still
    delivered the correct 3. Trim+restore is row-neutral. It does not create
    the extra row.
  - patch-4's graph/eager vote diverging at the top level: all eight ranks
    reached the same eager Python site on the same iteration.

THE REMAINING QUESTION, binary:

  Rank 4 entered the DSA decode path ONE FEWER TIME than its peers on the
  crashing iteration. MTP runs speculative_num_steps=3 with the last step
  skipped, so peers make 2 inner forwards. Rank 4 made 1. Why?

    A. Rank 4 replayed the draft CUDA graph for one step while peers ran
       eager. A replayed graph executes NO Python, so the trim probe is
       structurally blind to it -- "one fewer record" is exactly the
       signature. This would mean the graph/eager split is per-STEP, below
       the granularity patch 4's vote operates at.

    B. Rank 4 broke out of the step loop early, or took a branch that skips
       the DSA decode call (e.g. page_table_1 is None), so the step ran but
       the probe site was never reached.

  A and B need completely different fixes, so this measures rather than
  guesses. The instrumentation is deliberately placed to separate them:

    GLM52_DSTEP vote=...   -- once per draft(), the acted-on graph/eager
                              decision AFTER patch 4's all-reduce. If rank 4
                              says graph=1 while peers say graph=0 on the
                              crashing iteration, that is A, decided.
    GLM52_DSTEP step=...   -- once per inner eager forward, with the step
                              index and the row count. Counting these per
                              rank shows exactly which step is absent, and
                              the rows carried into it.

  Under A the vote line diverges and the step lines are simply fewer.
  Under B the vote line agrees and a step index is missing from the sequence.

Rank-tagged, capped, flushed so a crash cannot swallow the tail.
Idempotent; anchors asserted unique; invalidates the .pyc.
"""
import os
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
MARK = "GLM52_DSTEP"
CAP = 8000

HELPER = '''
# GLM52_DSTEP -- see /shared_nfs/yihou_exp3way/e3/instr_draft_steps.py
_DSTEP_N = [0]
_DSTEP_CAP = {cap}


def _dstep_log(kind, **kw):
    """Record one draft-loop event, rank-tagged."""
    if _DSTEP_N[0] >= _DSTEP_CAP:
        return
    _DSTEP_N[0] += 1
    try:
        import torch.distributed as _d

        rank = _d.get_rank() if _d.is_initialized() else -1
        body = " ".join("{{}}={{}}".format(k, v) for k, v in kw.items())
        print("GLM52_DSTEP {{}} rank={{}} {{}}".format(kind, rank, body), flush=True)
    except Exception as _e:  # never let the probe break the run
        print("GLM52_DSTEP probe-error {{}}".format(_e), flush=True)

'''.format(
    cap=CAP
)

HELPER_ANCHOR = "class EagleDraftWorker(EagleDraftWorkerBase):"

# --- 1. the acted-on graph/eager decision, AFTER patch 4's all-reduce --------
# This anchor is our patch 4's own text (arm e3a runs OUR patch 4), so it is
# present and unique by construction. Logging AFTER the vote is the point:
# the pre-vote local value is already known to diverge; what matters is what
# each rank actually did.
VOTE_ANCHOR = """        if _needs_eager_local:
            can_cuda_graph = False
"""

VOTE_REPL = """        if _needs_eager_local:
            can_cuda_graph = False
        # GLM52_DSTEP: the decision each rank ACTS on, post-vote. If this
        # diverges across ranks on the crashing iteration, hypothesis A holds
        # (a rank replayed the graph and its Python-level probes went silent).
        _dstep_log(
            "vote",
            graph=int(bool(can_cuda_graph)),
            bs=getattr(forward_batch, "batch_size", None),
            fwd_mode=getattr(forward_batch, "forward_mode", None),
            idle=int(bool(forward_batch.forward_mode.is_idle())),
        )
"""

# --- 2. every inner eager draft forward, with its step index ----------------
# Placed immediately before the forward so a step that ENTERS but dies is
# still recorded. `input_ids` is the row count actually fed to the draft
# model, which is the quantity that ends up in the all-gather.
STEP_ANCHOR = """                logits_output = self.draft_runner.forward(forward_batch).logits_output
"""

STEP_REPL = """                _dstep_log(
                    "step",
                    i=i,
                    rows=input_ids.shape[0],
                    ocl=forward_batch.out_cache_loc.shape[0],
                )
                logits_output = self.draft_runner.forward(forward_batch).logits_output
"""


def die(msg):
    print("FAIL: {}".format(msg), file=sys.stderr)
    sys.exit(1)


def main():
    src = open(TARGET).read()

    if MARK in src:
        print("already patched ({} present); nothing to do".format(MARK))
        return

    for name, anchor, count in (
        ("helper", HELPER_ANCHOR, 1),
        ("vote", VOTE_ANCHOR, 1),
        ("step", STEP_ANCHOR, 1),
    ):
        n = src.count(anchor)
        if n != count:
            die("{} anchor matched {} times, expected {}".format(name, n, count))

    src = src.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    src = src.replace(VOTE_ANCHOR, VOTE_REPL, 1)
    src = src.replace(STEP_ANCHOR, STEP_REPL, 1)

    open(TARGET, "w").write(src)
    # CLAUDE.md: a rewritten .py can keep an mtime matching the cached
    # bytecode, and CPython then runs the UNPATCHED .pyc. Touch and purge.
    os.utime(TARGET, None)
    pyc = os.path.join(
        os.path.dirname(TARGET), "__pycache__", "eagle_worker_v2.cpython-310.pyc"
    )
    if os.path.exists(pyc):
        os.remove(pyc)
    print("patched {} with {}".format(TARGET, MARK))


if __name__ == "__main__":
    main()
