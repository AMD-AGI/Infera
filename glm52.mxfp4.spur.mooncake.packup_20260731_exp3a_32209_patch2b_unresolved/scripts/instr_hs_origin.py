#!/usr/bin/env python3
"""Trace where the over-long `hidden_states` row count comes from.

ONE BOOT, FIVE SITES. Round 5 established the fact this probe chases:

    seq=8, plan=[4]*8, buffer=32, orig=[0,1,2,2,4,3,3,3]
      rank=0,1,4   local_rows=4  inp_rows=4   OK
      rank=2,3,5,6,7  local_rows=6  inp_rows=4   -> ValueError

  `inp_rows` (this forward's own input_ids) is 4 on ALL EIGHT ranks and equals
  the agreed plan. The tensor handed to the all-gather has 6. It therefore did
  not come from this forward -- it was carried in. 6 is not any rank's real
  token count (orig max is 4), so it is not a stale copy of a peer's load
  either; it is a count produced by some *combination*.

  Also settled in round 5, do not re-test:
    - not num_draft_tokens (ndt=1, and the bad value was 4 one run, 6 the next)
    - not TARGET_VERIFY (fwd=2 DECODE on every faulting record)
    - not the page-table/top-k mismatch of upstream PR #31760: measured
      pt_rows == topk_rows on 100% of transform calls here, because our
      patch 2b already reconciles them before the call.

CANDIDATE PRODUCERS OF A CARRIED ROW COUNT, all instrumented together:

  S1 merge_batch cat        -- `self.hidden_states = cat([self, spec_info])`.
                               A concatenation is the one operation in this
                               file that can produce a row count larger than
                               either input batch's own size. 4+2 = 6 fits the
                               observed value exactly, so this is the primary
                               suspect, not a hunch to test in isolation.
  S2 merge_batch idle-stub  -- the `len(topk_index)==0` branch adopts the
                               other batch's tensor wholesale.
  S3 filter_batch           -- both the `[:len]` and the `[new_indices]` arm
                               resize; a wrong arm leaves a stale length.
  S4 draft-extend select    -- `hidden_states[select_index]` after verify,
                               the value that becomes next_draft_input.
  S5 draft() entry          -- the row count actually present when the draft
                               forward begins, i.e. the end of the carry
                               chain. Pairs with GLM52_MULTI's inp_rows.

  S1-S4 are the writers; S5 is the reader. Logging all five in one run means
  the carry chain is read off directly instead of bisected over five boots.

Rank-tagged, capped, flushed. No behaviour change.
Idempotent; anchors asserted unique; invalidates the .pyc.
"""
import os
import sys

MARK = "GLM52_HSO"
CAP = 6000

INFO = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_info.py"
WORKER = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"

HELPER = '''
# GLM52_HSO -- see /shared_nfs/yihou_exp3way/e3/instr_hs_origin.py
_HSO_N = [0]
_HSO_CAP = {cap}


def _hso_log(site, **kw):
    """Record one hidden_states row-count event."""
    if _HSO_N[0] >= _HSO_CAP:
        return
    _HSO_N[0] += 1
    try:
        import torch.distributed as _d

        rank = _d.get_rank() if _d.is_initialized() else -1
        body = " ".join("{{}}={{}}".format(k, v) for k, v in kw.items())
        print("GLM52_HSO {{}} rank={{}} {{}}".format(site, rank, body), flush=True)
    except Exception as _e:
        print("GLM52_HSO probe-error {{}}".format(_e), flush=True)


def _hso_rows(t):
    return None if t is None else t.shape[0]

'''.format(
    cap=CAP
)

# ---------------------------------------------------------------- eagle_info
# Anchor on the decorator+class pair, NOT the bare name: "class EagleDraftInput"
# also occurs inside a comment at eagle_info.py:141, and inserting the helper
# there lands it in the middle of the class body -> SyntaxError.
INFO_HELPER_ANCHOR = "@dataclass\nclass EagleDraftInput(SpecInput):"

# S2: the idle-stub adoption arm of merge_batch.
S2_ANCHOR = """        if len(self.topk_index) == 0:
            self.hidden_states = spec_info.hidden_states
"""
S2_REPL = """        if len(self.topk_index) == 0:
            _hso_log(
                "merge_idle",
                self_rows=_hso_rows(self.hidden_states),
                other_rows=_hso_rows(spec_info.hidden_states),
            )
            self.hidden_states = spec_info.hidden_states
"""

# S1: the concatenation arm -- the only site that can grow beyond both inputs.
S1_ANCHOR = """        if self.hidden_states is not None and spec_info.hidden_states is not None:
            self.hidden_states = torch.cat(
                [self.hidden_states, spec_info.hidden_states], axis=0
            )
"""
S1_REPL = """        if self.hidden_states is not None and spec_info.hidden_states is not None:
            _hso_log(
                "merge_cat",
                self_rows=_hso_rows(self.hidden_states),
                other_rows=_hso_rows(spec_info.hidden_states),
                result_rows=(
                    self.hidden_states.shape[0] + spec_info.hidden_states.shape[0]
                ),
            )
            self.hidden_states = torch.cat(
                [self.hidden_states, spec_info.hidden_states], axis=0
            )
"""

# S3: both filter arms.
S3A_ANCHOR = """            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[: len(new_indices)]
"""
S3A_REPL = """            if self.hidden_states is not None:
                _hso_log(
                    "filter_prefix",
                    before=_hso_rows(self.hidden_states),
                    after=len(new_indices),
                )
                self.hidden_states = self.hidden_states[: len(new_indices)]
"""

S3B_ANCHOR = """            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[new_indices]
"""
S3B_REPL = """            if self.hidden_states is not None:
                _hso_log(
                    "filter_index",
                    before=_hso_rows(self.hidden_states),
                    after=len(new_indices),
                )
                self.hidden_states = self.hidden_states[new_indices]
"""

# ---------------------------------------------------------------- worker
WORKER_HELPER_ANCHOR = "class EagleDraftWorker(EagleDraftWorkerBase):"

# S4: the draft-extend select that produces next_draft_input.hidden_states.
S4_ANCHOR = """        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
"""
S4_REPL = """        if draft_logits_output.hidden_states is not None:
            _hso_log(
                "extend_select",
                before=_hso_rows(draft_logits_output.hidden_states),
                sel=select_index.shape[0],
            )
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
"""

# S5: the row count present when draft() begins -- the end of the carry chain.
S5_ANCHOR = """    def draft(self, batch: ScheduleBatch):
        draft_input: EagleDraftInput = batch.spec_info
"""
S5_REPL = """    def draft(self, batch: ScheduleBatch):
        draft_input: EagleDraftInput = batch.spec_info
        _hso_log(
            "draft_entry",
            hs_rows=_hso_rows(getattr(draft_input, "hidden_states", None)),
            topk_rows=_hso_rows(getattr(draft_input, "topk_index", None)),
            bs=batch.seq_lens.shape[0] if batch.seq_lens is not None else None,
            fwd=batch.forward_mode,
        )
"""


def die(msg):
    print("FAIL: {}".format(msg), file=sys.stderr)
    sys.exit(1)


def patch(path, helper_anchor, pairs, pyc_name):
    src = open(path).read()
    if MARK in src:
        print("already patched: {}".format(path))
        return
    n = src.count(helper_anchor)
    if n != 1:
        die("helper anchor in {} matched {} times, expected 1".format(path, n))
    for name, anchor, _ in pairs:
        c = src.count(anchor)
        if c != 1:
            die("{} anchor in {} matched {} times, expected 1".format(name, path, c))
    src = src.replace(helper_anchor, HELPER + helper_anchor, 1)
    for _, anchor, repl in pairs:
        src = src.replace(anchor, repl, 1)
    open(path, "w").write(src)
    os.utime(path, None)
    pyc = os.path.join(os.path.dirname(path), "__pycache__", pyc_name)
    if os.path.exists(pyc):
        os.remove(pyc)
    print("patched {} with {}".format(path, MARK))


def main():
    patch(
        INFO,
        INFO_HELPER_ANCHOR,
        [
            ("S2 merge_idle", S2_ANCHOR, S2_REPL),
            ("S1 merge_cat", S1_ANCHOR, S1_REPL),
            ("S3a filter_prefix", S3A_ANCHOR, S3A_REPL),
            ("S3b filter_index", S3B_ANCHOR, S3B_REPL),
        ],
        "eagle_info.cpython-310.pyc",
    )
    patch(
        WORKER,
        WORKER_HELPER_ANCHOR,
        [
            ("S4 extend_select", S4_ANCHOR, S4_REPL),
            ("S5 draft_entry", S5_ANCHOR, S5_REPL),
        ],
        "eagle_worker_v2.cpython-310.pyc",
    )


if __name__ == "__main__":
    main()
