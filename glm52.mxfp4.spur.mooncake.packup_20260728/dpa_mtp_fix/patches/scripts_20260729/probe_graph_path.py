#!/usr/bin/env python3
"""Probe 2: is the draft stage taking the CUDA-graph path or the eager path?

Probe 1 measured per-rank LM-head all-gather counts and found idle iterations
issue 1 while decode iterations issue 3.  But that reading has a hole: a
*replayed* CUDA graph executes no Python, so the AG_ENTER hook inside
_get_logits never fires for graph-path forwards.  "0 all-gathers on the idle
path" is therefore ambiguous:

  (a) the idle path really issues no collective            -> a genuine skip
  (b) the idle path replays a graph containing the collective  -> invisible to probe 1

These have opposite fixes, so the ambiguity has to be resolved by measurement.

There is a concrete reason to suspect (b) matters here.  eagle_worker_v2.py:517
disables the graph path on a condition that is *rank-divergent by construction*:

    if (can_cuda_graph
        and not forward_batch.forward_mode.is_idle()   # <-- differs per DP rank
        and self.seed_dsa_topk_from_draft_extend
        and draft_input.dsa_topk_indices is None):
        can_cuda_graph = False

`seed_dsa_topk_from_draft_extend` is True for GLM-5.2 (index_share_for_mtp_iteration
and index_topk=2048).  So a busy rank can be forced to eager while an idle rank
keeps the graph -- and graph vs eager need not issue the same collectives.

This probe logs, per draft() call: forward_mode, can_cuda_graph both before and
after that guard, and whether dsa_topk_indices was None.  Diffing the resulting
per-rank logs answers (a) vs (b) directly.

Also instruments _draft_extend_for_decode's can_cuda_graph, which is chosen by
a separate `can_run_graph(forward_batch)` predicate that is batch-size
dependent -- and DP ranks have different batch sizes.
"""

import os
import py_compile
import shutil
import sys

EAGLE = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
SUFFIX = ".graph_probe_orig"


def main():
    if "--revert" in sys.argv:
        bak = EAGLE + SUFFIX
        if os.path.exists(bak):
            shutil.copy2(bak, EAGLE)
            print(f"reverted {EAGLE}")
        return

    bak = EAGLE + SUFFIX
    if os.path.exists(bak):
        shutil.copy2(bak, EAGLE)
        print("restored from backup first")
    else:
        shutil.copy2(EAGLE, bak)
        print("backed up")

    with open(EAGLE) as f:
        src = f.read()

    # -- draft(): log the graph/eager decision and the guard's inputs --------
    old = """        if (
            can_cuda_graph
            and not forward_batch.forward_mode.is_idle()
            and self.seed_dsa_topk_from_draft_extend
            and draft_input.dsa_topk_indices is None
        ):
            can_cuda_graph = False"""
    new = """        _sp_cg_before = can_cuda_graph
        if (
            can_cuda_graph
            and not forward_batch.forward_mode.is_idle()
            and self.seed_dsa_topk_from_draft_extend
            and draft_input.dsa_topk_indices is None
        ):
            can_cuda_graph = False
        _sp_log(
            "DRAFT_GRAPH",
            mode=str(forward_batch.forward_mode),
            cg_before=_sp_cg_before,
            cg_after=can_cuda_graph,
            seed=self.seed_dsa_topk_from_draft_extend,
            topk_none=(draft_input.dsa_topk_indices is None),
        )"""
    assert src.count(old) == 1, f"draft graph anchor count={src.count(old)}"
    src = src.replace(old, new)

    # -- _draft_extend_for_decode(): same question, different predicate ------
    old2 = """        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
        )"""
    new2 = """        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
        )
        _sp_log(
            "EXTEND_GRAPH",
            mode=str(forward_batch.forward_mode),
            cg=bool(can_cuda_graph),
            ntok=forward_batch.input_ids.shape[0],
        )"""
    assert src.count(old2) == 1, f"extend graph anchor count={src.count(old2)}"
    src = src.replace(old2, new2)

    with open(EAGLE, "w") as f:
        f.write(src)

    try:
        py_compile.compile(EAGLE, doraise=True)
        print(f"OK syntax {EAGLE}")
    except py_compile.PyCompileError as e:
        print(f"FAIL syntax: {e}")
        sys.exit(1)

    print("probe 2 installed: DRAFT_GRAPH + EXTEND_GRAPH")


if __name__ == "__main__":
    main()
