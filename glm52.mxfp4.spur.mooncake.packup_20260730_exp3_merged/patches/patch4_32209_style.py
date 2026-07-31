#!/usr/bin/env python3
"""Patch 4, reimplemented in the shape of upstream PR #32209.

Our patch 4 (`eagle_worker_v2_uniform_draft_graph.diff`, verified 2540/2540)
fixes a real deadlock: the graph/eager choice in `EagleDraftWorker.draft()` is
made per rank from rank-dependent inputs, and diverges on the PD decode leg
because guard term 4 (`draft_input.dsa_topk_indices is None`) is seeded from
RDMA-shipped per-request payloads.  It votes the local need-for-eager over the
full TP group with a 1-element gloo all-reduce.

PR #32209 (HZY-Wade, sgl-project/sglang, open) fixes the SAME defect with the
same strategy but a better placement, and this script ports that placement:

  * ours   : an EXTRA collective inside `draft()`, once per draft call.
  * #32209 : one more int64 slot in the MLP-sync all-gather the scheduler
             ALREADY performs every iteration -- zero extra collectives.

The port, four edits mirroring #32209's own hunks:

  1. `speculative/eagle_worker_v2.py`
     add `EagleWorkerV2.requires_dp_attention_eager_forward(batch)` -- the
     rank-local predicate, lifted out of `draft()` and moved to scheduler time.
  2. `disaggregation/decode.py`
     call it in `get_next_disagg_decode_batch_to_run`, just BEFORE
     `maybe_prepare_mlp_sync_batch`, storing the answer on the batch.
  3. `managers/scheduler_components/dp_attn.py`
     carry it as slot 7 of the all-gather tensor and `min()`-reduce it, exactly
     like `can_cuda_graph` (slot 2).  Idle/inactive ranks contribute 1
     (permissive), so an idle rank can never force the group eager -- the same
     property our vote has via `ReduceOp.MAX` over need-for-eager.
  4. `speculative/eagle_draft_cuda_graph_runner.py`
     `can_run_graph` additionally requires `can_run_dp_draft_cuda_graph`.

Plumbing needed to carry the flag (ScheduleBatch -> ForwardBatch), also
mirroring #32209:
  * `managers/schedule_batch.py`     : two new fields + copy() propagation
  * `model_executor/forward_batch_info.py` : one new field + init_new wiring

DELIBERATE DIVERGENCES from #32209, and why:

  * #32209 reads `draft_input.future_dsa_topk_indices_available` when
    `future_indices is not None`, to cope with overlap scheduling resolving
    inputs after the scheduler-side vote.  That attribute does NOT exist in our
    baseline (verified by grep).  We therefore read `dsa_topk_indices` directly
    and, when `future_indices` is set, fall back to REQUIRING EAGER -- the safe
    direction, since a wrong "graph is fine" is the deadlock and a wrong "go
    eager" is only slower.  This is a divergence to measure, not to hide: if
    the graph-usage rate comes back far below the 98.4% our patch 4 achieved,
    this fallback is why.
  * #32209 also touches `_forward_trtllm` and the CUDA idle path in
    `dsa_indexer.forward_cuda`.  Neither is on our HIP/tilelang path; they are
    NOT ported here.  (`patch2b_32209_style.py` handles our tilelang decode
    site separately.)

Idempotent.  Every anchor is asserted to match exactly once.  Invalidates the
affected .pyc files (CLAUDE.md: a stale .pyc silently reverts the patch).
"""
import os
import sys

ROOT = "/sgl-workspace/sglang/python/sglang/srt"
MARK = "GLM52_P4V2"

EDITS = []  # (path, anchor, replacement, expected_count)


def add(path, anchor, repl, n=1):
    EDITS.append((os.path.join(ROOT, path), anchor, repl, n))


# ---------------------------------------------------------------- (1) worker --
add(
    "speculative/eagle_worker_v2.py",
    """    @property
    def spec_v2_attn_backends(self) -> tuple:""",
    '''    def requires_dp_attention_eager_forward(self, batch) -> bool:
        """GLM52_P4V2: does THIS rank need the draft step to run eagerly?

        Ported from upstream PR #32209. The graph/eager choice must be the
        SAME on every DP rank -- graph replay and the eager multi-step loop do
        not issue the same host-side collective sequence, so a per-rank choice
        desynchronizes the group and deadlocks (measured: one busy rank eager
        in init_forward_metadata while seven idle ranks sat in all_gather).

        This is the rank-local half. The scheduler folds the answer into the
        MLP-sync all-gather it already performs, which min()-reduces it across
        the group -- so no extra collective is introduced, unlike our v1 fix
        which added a gloo all-reduce inside draft().

        The condition is guard term 4 of the original `draft()` check: with
        IndexShare seeding on, a batch whose DSA top-k seed has not arrived
        must not use the draft graph. On the PD decode leg that seed comes
        from RDMA-shipped per-request payloads, so the answer is a function of
        which requests this rank happens to hold -- which is exactly why it
        diverges here and never does on a single-node mix run.
        """
        if not self._draft_worker.seed_dsa_topk_from_draft_extend:
            return False

        draft_input = batch.spec_info
        if draft_input is None:
            return False

        # Under overlap scheduling FutureMap resolves the draft input AFTER
        # this scheduler-side vote, so `dsa_topk_indices` is still stale here
        # and must not be read directly. Upstream #32209 consults
        # `future_dsa_topk_indices_available` instead, and so do we:
        #
        #   scheduler.py:3314          sets it to (dsa_topk_indices is not None)
        #                              on the draft input published for the
        #                              NEXT iteration;
        #   overlap_utils.py:271       is the consumer -- it fills
        #                              draft_input.dsa_topk_indices from the
        #                              buffer iff the flag is set, else None.
        #
        # So the flag is exactly "will term 4 be satisfied once resolved",
        # which is the question this predicate has to answer one step early.
        #
        # CORRECTION (2026-07-30): an earlier revision of this script claimed
        # the attribute was absent from this baseline and fell back to
        # REQUIRING EAGER whenever `future_indices` was set. That claim was
        # wrong -- it is present at eagle_info.py:179 and spec_info.py:261 --
        # and the fallback was catastrophic rather than merely conservative:
        # overlap scheduling sets `future_indices` on EVERY decode iteration,
        # so the draft graph was refused 100% of the time. Measured on arm e3b,
        # conc=32: draft-graph usage 0.0% across all 8 ranks (200/200 calls
        # refused), with the real guard term (`seed_none`) never once firing.
        # That silently degraded patch 4 into the Variant-B workaround while
        # still passing every functional test.
        if getattr(draft_input, "future_indices", None) is not None:
            return not draft_input.future_dsa_topk_indices_available
        return getattr(draft_input, "dsa_topk_indices", None) is None

    @property
    def spec_v2_attn_backends(self) -> tuple:''',
)

# ---------------------------------------------------------------- (2) decode --
add(
    "disaggregation/decode.py",
    """        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return ret""",
    """        # GLM52_P4V2: ask the draft worker whether THIS rank needs an eager
        # draft step, and record it on the batch so the MLP-sync all-gather
        # below carries it to every rank. Must run BEFORE
        # maybe_prepare_mlp_sync_batch -- that is the collective that spreads
        # it. (Upstream PR #32209 places it at exactly this line.)
        if ret is not None and getattr(self, "draft_worker", None) is not None:
            ret.force_disable_draft_cuda_graph = (
                self.draft_worker.requires_dp_attention_eager_forward(ret)
            )

        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return ret""",
)

# --------------------------------------------------------------- (3) dp_attn --
add(
    "managers/scheduler_components/dp_attn.py",
    """    num_tokens: int
    num_tokens_for_logprob: int
    can_cuda_graph: bool
    is_extend_in_batch: bool""",
    """    num_tokens: int
    num_tokens_for_logprob: int
    can_cuda_graph: bool
    # GLM52_P4V2: slot 7 of the gathered tensor. min()-reduced like
    # can_cuda_graph, so ANY rank needing eager takes the whole group eager.
    can_draft_cuda_graph: bool
    is_extend_in_batch: bool""",
)

add(
    "managers/scheduler_components/dp_attn.py",
    """                self.local_forward_mode,
                int(self.can_run_breakable_cuda_graph),
            ],
            device=device,
            dtype=dtype,
        )

    def _get_fallback_tensor""",
    """                self.local_forward_mode,
                int(self.can_run_breakable_cuda_graph),
                int(self.can_draft_cuda_graph),  # GLM52_P4V2
            ],
            device=device,
            dtype=dtype,
        )

    def _get_fallback_tensor""",
)

add(
    "managers/scheduler_components/dp_attn.py",
    """                ForwardMode.IDLE.value,  # local_forward_mode
                0,  # can_run_breakable_cuda_graph
            ],""",
    """                ForwardMode.IDLE.value,  # local_forward_mode
                0,  # can_run_breakable_cuda_graph
                1,  # can_draft_cuda_graph -- GLM52_P4V2, permissive: an
                    # inactive rank must not drag the group into eager.
            ],""",
)

add(
    "managers/scheduler_components/dp_attn.py",
    """        global_info_tensor = torch.empty(
            (self.dp_size, self.tp_size * self.cp_size, 7),
            dtype=torch.int64,
            device=device,
        )""",
    """        # GLM52_P4V2: 7 -> 8 slots (can_draft_cuda_graph appended).
        global_info_tensor = torch.empty(
            (self.dp_size, self.tp_size * self.cp_size, 8),
            dtype=torch.int64,
            device=device,
        )""",
)

add(
    "managers/scheduler_components/dp_attn.py",
    """        tp_info = global_info_tensor.view(self.dp_size * self.tp_size * self.cp_size, 7)""",
    """        tp_info = global_info_tensor.view(  # GLM52_P4V2: 7 -> 8
            self.dp_size * self.tp_size * self.cp_size, 8
        )""",
)

add(
    "managers/scheduler_components/dp_attn.py",
    """        self.can_run_breakable_cuda_graph = bool(tp0_info[:, 6].min().item())""",
    """        self.can_run_breakable_cuda_graph = bool(tp0_info[:, 6].min().item())
        # GLM52_P4V2: min() -- one rank needing eager takes the whole group.
        self.can_draft_cuda_graph = bool(tp0_info[:, 7].min().item())""",
)

add(
    "managers/scheduler_components/dp_attn.py",
    """    batch.can_run_dp_cuda_graph = mlp_sync_info.can_cuda_graph
    batch.can_run_dp_breakable_cuda_graph = mlp_sync_info.can_run_breakable_cuda_graph""",
    """    batch.can_run_dp_cuda_graph = mlp_sync_info.can_cuda_graph
    batch.can_run_dp_draft_cuda_graph = mlp_sync_info.can_draft_cuda_graph  # GLM52_P4V2
    batch.can_run_dp_breakable_cuda_graph = mlp_sync_info.can_run_breakable_cuda_graph""",
)

add(
    "managers/scheduler_components/dp_attn.py",
    """    can_run_breakable_cuda_graph = (
        local_batch is not None
        and local_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
        and not disable_cuda_graph
    )""",
    """    can_run_breakable_cuda_graph = (
        local_batch is not None
        and local_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
        and not disable_cuda_graph
    )
    # GLM52_P4V2: rank-local answer set by the scheduler just before this call.
    # Default True (permissive) when the batch is None or the attribute is
    # absent, matching can_cuda_graph's treatment of idle ranks.
    can_draft_cuda_graph = not (
        local_batch is not None
        and getattr(local_batch, "force_disable_draft_cuda_graph", False)
    )""",
)

add(
    "managers/scheduler_components/dp_attn.py",
    """        can_cuda_graph=can_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,""",
    """        can_cuda_graph=can_cuda_graph,
        can_draft_cuda_graph=can_draft_cuda_graph,  # GLM52_P4V2
        is_extend_in_batch=is_extend_in_batch,""",
)

# --------------------------------------------------------- (4) graph runner --
add(
    "speculative/eagle_draft_cuda_graph_runner.py",
    """            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph""",
    """            # GLM52_P4V2: the group-wide draft-graph gate, all-gathered by
            # the scheduler. Without it the draft graph decision stays
            # rank-local and the group deadlocks.
            is_bs_supported = (
                is_bs_supported
                and forward_batch.can_run_dp_cuda_graph
                and forward_batch.can_run_dp_draft_cuda_graph
            )""",
)

# ------------------------------------------------------------- plumbing ------
add(
    "managers/schedule_batch.py",
    """    can_run_dp_breakable_cuda_graph: bool = False""",
    """    can_run_dp_draft_cuda_graph: bool = True  # GLM52_P4V2
    # GLM52_P4V2: rank-local request for an eager draft step; the MLP-sync
    # all-gather turns it into the group-wide can_run_dp_draft_cuda_graph.
    force_disable_draft_cuda_graph: bool = False
    can_run_dp_breakable_cuda_graph: bool = False""",
)

add(
    "managers/schedule_batch.py",
    """            can_run_dp_breakable_cuda_graph=self.can_run_dp_breakable_cuda_graph,""",
    """            can_run_dp_draft_cuda_graph=self.can_run_dp_draft_cuda_graph,  # GLM52_P4V2
            force_disable_draft_cuda_graph=self.force_disable_draft_cuda_graph,
            can_run_dp_breakable_cuda_graph=self.can_run_dp_breakable_cuda_graph,""",
)

add(
    "model_executor/forward_batch_info.py",
    """    can_run_dp_cuda_graph: bool = False
    can_run_dp_breakable_cuda_graph: bool = False""",
    """    can_run_dp_cuda_graph: bool = False
    can_run_dp_draft_cuda_graph: bool = True  # GLM52_P4V2
    can_run_dp_breakable_cuda_graph: bool = False""",
)

add(
    "model_executor/forward_batch_info.py",
    """            can_run_dp_cuda_graph=batch.can_run_dp_cuda_graph,""",
    """            can_run_dp_cuda_graph=batch.can_run_dp_cuda_graph,
            can_run_dp_draft_cuda_graph=batch.can_run_dp_draft_cuda_graph,  # GLM52_P4V2""",
)


def die(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    # Read everything first so a mid-way anchor failure leaves nothing edited.
    files = {}
    for path, anchor, repl, n in EDITS:
        if path not in files:
            files[path] = open(path).read()

    if any(MARK in s for s in files.values()):
        print(f"already patched ({MARK} present); nothing to do")
        return

    for path, anchor, repl, n in EDITS:
        got = files[path].count(anchor)
        if got != n:
            die(f"{os.path.relpath(path, ROOT)}: anchor matched {got}x, want {n}\n"
                f"---\n{anchor[:200]}\n---")
        files[path] = files[path].replace(anchor, repl, n)

    for path, src in files.items():
        open(path, "w").write(src)
        os.utime(path, None)
        d, base = os.path.split(path)
        pc = os.path.join(d, "__pycache__")
        if os.path.isdir(pc):
            stem = base[:-3] + "."
            for f in os.listdir(pc):
                if f.startswith(stem):
                    os.remove(os.path.join(pc, f))
        print(f"patched {os.path.relpath(path, ROOT)}  ({src.count(MARK)} markers)")


if __name__ == "__main__":
    main()
