#!/usr/bin/env python3
"""Bug 2b fix: make the draft graph/eager decision uniform across the DP group.

## The defect (measured in r01, not inferred)

`eagle_worker_v2.py::draft()` decides per rank whether to replay the draft CUDA
graph or run the multi-step draft eagerly:

    if (can_cuda_graph
        and not forward_batch.forward_mode.is_idle()
        and self.seed_dsa_topk_from_draft_extend
        and draft_input.dsa_topk_indices is None):
        can_cuda_graph = False

Under DP-attention the ranks run this in lockstep, but the terms are not
rank-invariant, so the *decision* is not either. Measured on the PD decode leg
at the exact iteration that deadlocked (r01_instrument/RESULT.md):

    it=9  dp2  DECODE bs=1  t2=True  t4=True  -> EAGER   <-- the only busy rank
          dp0,1,3,4,5,6,7  IDLE bs=0  t2=False -> GRAPH

py-spy at that moment found dp2 blocked in the eager `init_forward_metadata`
(`dsa_backend.py:785`) while the other seven sat in `all_gather`/`broadcast`.
Graph and eager do not issue the same host-side collective sequence, so once the
ranks split they never re-converge.

Term 4 is the one that flips: on the PD decode leg `dsa_topk_indices` is built
from RDMA-shipped per-request payloads (`eagle_disaggregation.py:54-59`), so it
is None for a request that just arrived and non-None once
`_draft_extend_for_decode` has seeded it. Term 2 is what makes the flip
*asymmetric* -- idle ranks short-circuit before ever looking at term 4.

Single-node mix never hits this because there is no disagg path: the seed is
always produced locally, by the same code, on every rank.

## Why the previous attempt (drop `not is_idle()`) failed

Replaying the two measured iterations without term 2:

    it=9  every rank has t1 & t3 & t4 -> all eager -> uniform (fixed)
    it=5  dp6 idle, t4=True -> eager;  busy ranks t4=False -> graph -> DIVERGENT

It fixes one iteration and breaks another, which is exactly the observed
"failure moved earlier, into warmup" (WARMUP_MATRIX runs 5/6).

## This fix

Vote on the *local need for eager*, then apply the group answer:

    needs_eager_local = t1 and t2 and t3 and t4        # unchanged predicate
    needs_eager_group = OR over the attention TP group
    can_cuda_graph    = t1 and not needs_eager_group

Properties:
  * An idle rank contributes False, so it can never drag the group to eager --
    the failure mode of the naive fix.
  * If any rank genuinely needs eager (its top-k seed is missing), *all* ranks go
    eager, which is correct: that is what the guard is for.
  * When nobody needs eager the graph is still used, so unlike Variant B this
    keeps the draft graph's speedup.

Checked against both measured iterations:
    it=9  dp2 local=True, others False  -> OR=True  -> all eager  (uniform)
    it=5  all local=False               -> OR=False -> all graph  (uniform)

## Cost and safety of the added collective

One 1-element all-reduce per draft() call, on the **cpu_group** (gloo), so it
adds no GPU sync and cannot serialize the compute stream.

All ranks provably reach this line every iteration: the r01 probe recorded 9
draft() entries on all 8 ranks including the idle ones, with equal counts. A
collective here is therefore safe -- it is on a branch every rank takes, which
is precisely the property the buggy code lacked.

Idempotent, anchor-verified, purges .pyc. `--revert` supported.
"""
import os
import shutil
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
MARKER = "GLM52_BUG2B_UNIFORM"

ANCHOR = """        if (
            can_cuda_graph
            and not forward_batch.forward_mode.is_idle()
            and self.seed_dsa_topk_from_draft_extend
            and draft_input.dsa_topk_indices is None
        ):
            can_cuda_graph = False
"""

REPLACEMENT = '''        # GLM52_BUG2B_UNIFORM: the graph/eager choice must be the SAME on every DP
        # rank. Graph replay and the eager multi-step loop do not issue the same
        # host-side collective sequence, so a per-rank choice desynchronizes the
        # group and deadlocks (measured: one busy rank eager in
        # init_forward_metadata while seven idle ranks sat in all_gather).
        #
        # The terms below are all rank-dependent under DP-attention:
        #   is_idle()            -- occupancy, differs by construction
        #   dsa_topk_indices     -- on the PD decode leg this comes from
        #                           RDMA-shipped per-request payloads, so it is
        #                           None only for freshly-arrived requests
        # so we vote on the *local need for eager* and apply the group answer.
        # An idle rank contributes False and can never force the group eager.
        _needs_eager_local = (
            can_cuda_graph
            and not forward_batch.forward_mode.is_idle()
            and self.seed_dsa_topk_from_draft_extend
            and draft_input.dsa_topk_indices is None
        )
        if can_cuda_graph and self.seed_dsa_topk_from_draft_extend:
            # NOT get_attention_tp_group(): under DPA8 on tp8 the attention TP
            # group is attn_tp_size = tp/dp = 1 rank wide, so voting on it is a
            # no-op. The group that must agree is the full TP group -- the same
            # one the scheduler's MLP-sync all-gather spans (dp_attn.py:91).
            from sglang.srt.distributed.parallel_state import (
                get_tp_group as _get_tp_group,
            )

            _grp = _get_tp_group()
            if _grp is not None and _grp.world_size > 1:
                # 1-element gloo all-reduce on the CPU group: no GPU sync, and
                # every rank reaches this line every iteration (verified by
                # instrumentation), so the collective is safe here.
                _vote = torch.tensor(
                    [1 if _needs_eager_local else 0], dtype=torch.int32
                )
                torch.distributed.all_reduce(
                    _vote,
                    op=torch.distributed.ReduceOp.MAX,
                    group=_grp.cpu_group,
                )
                _needs_eager_local = bool(_vote.item())
        if _needs_eager_local:
            can_cuda_graph = False
'''


def purge_pyc(path):
    """Stale bytecode silently reverts a patch; this already invalidated one run."""
    os.utime(path, None)
    d, f = os.path.split(path)
    pyc_dir = os.path.join(d, "__pycache__")
    if os.path.isdir(pyc_dir):
        stem = f[:-3]
        for name in os.listdir(pyc_dir):
            if name.startswith(stem + "."):
                os.remove(os.path.join(pyc_dir, name))


def main():
    bak = TARGET + ".bug2bbak"
    if "--revert" in sys.argv:
        if not os.path.exists(bak):
            print("no backup")
            return 1
        shutil.copyfile(bak, TARGET)
        os.remove(bak)
        purge_pyc(TARGET)
        print("reverted")
        return 0

    src = open(TARGET).read()
    if MARKER in src:
        print("already applied")
        return 0
    n = src.count(ANCHOR)
    if n != 1:
        print(f"ANCHOR matched {n} times (want 1) -- refusing to patch")
        return 2
    if "\nimport torch" not in src:
        print("torch not imported -- refusing")
        return 3

    if not os.path.exists(bak):
        shutil.copyfile(TARGET, bak)
    open(TARGET, "w").write(src.replace(ANCHOR, REPLACEMENT, 1))
    purge_pyc(TARGET)

    import py_compile

    py_compile.compile(TARGET, doraise=True)
    print(f"applied to {TARGET}; markers in source = {open(TARGET).read().count(MARKER)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
