#!/usr/bin/env python3
"""Pin the +1 row: instrument the DP padding site itself, plus its consumers.

THE RULE TO EXPLAIN (measured e3c round 3, 72 gather records, no exceptions):

    local_rows == plan[rank] + 1     on 3/3 faulting records
    orig[rank]  <  plan[rank]        on 3/3 faulting records
    local_rows == plan[rank]         on all 69 correct records

  i.e. exactly the ranks whose real token count is BELOW the group-agreed
  plan deliver one row too many, and always exactly one. Ranks at or above
  the plan, and idle ranks, are fine.

  Example iteration: plan=[3]*8, orig=[3,3,0,3,2,2,2,3]
    ranks 0,1,3,7 (orig=3)  -> 3 rows  OK
    rank  2       (idle)    -> 3 rows  OK
    ranks 4,5,6   (orig=2)  -> 4 rows  FAULT

ALREADY REFUTED, do not re-test (all measured, not argued):
  DpPaddingMode divergence; patch-4 vote failure; a rank replaying the graph;
  a rank skipping a draft step; trim/restore leaking rows; missing output
  slicing in the draft loop; num_draft_tokens leaking (ndt=1); TARGET_VERIFY
  contamination (fwd=2 throughout); hidden_states carried stale across
  iterations (draft_entry showed hs_rows == bs on 456/456 records);
  merge_batch/filter_batch resizing (0 calls); and upstream PR #31760's
  page_table-vs-topk mismatch (pt_rows == topk_rows on 100% of calls here,
  because our patch 2b reconciles them before the transform).

WHERE THE +1 MUST BE BORN. `_pad_inputs_to_size(model_runner, num_tokens, bs)`
pads input_ids/positions/out_cache_loc to `num_tokens` and the per-request
tensors to `bs`, then the spec block pads `spec_info.hidden_states` to
`num_tokens`. For a draft (DECODE) batch under MTP the two are related by
`num_tokens = bs * num_tokens_per_req`, and `bs` is recomputed at
forward_batch_info.py:1213/1257 as `num_tokens // num_tokens_per_req`. An
integer-division round trip on a rank whose real count is below the plan is
precisely the shape of an off-by-one -- but that is a hypothesis, and the last
four hypotheses I formed from reading code were wrong. So: measure.

SITES, all in one boot:

  P1 pad_enter  -- num_tokens and bs as passed in, the pre-pad row counts of
                   input_ids / positions / hidden_states, and the local vs
                   planned token counts. This is where the +1 either is or
                   is not introduced.
  P2 pad_exit   -- the same tensors' row counts after padding. P1 vs P2 shows
                   whether padding produced a consistent set.
  P3 hs_pad     -- the spec_info.hidden_states pad specifically, since that is
                   the tensor the faulting all-gather receives.
  P4 unpad      -- post_forward_mlp_sync_batch's restore, which uses
                   `self.hidden_states_backup.shape[0]` for DECODE. If the
                   backup was taken from an already-padded tensor, the restore
                   returns the padded count and the next forward inherits it.

  P1/P2/P3 catch a pad-side off-by-one; P4 catches a restore-side one. Both
  are live candidates and neither is assumed.

Rank-tagged, capped, flushed. No behaviour change.
Idempotent; anchors asserted unique; invalidates the .pyc.
"""
import os
import sys

MARK = "GLM52_PAD"
CAP = 8000

FBI = "/sgl-workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py"

HELPER = '''
# GLM52_PAD -- see /shared_nfs/yihou_exp3way/e3/instr_pad.py
_PAD_N = [0]
_PAD_CAP = {cap}


def _pad_log(site, **kw):
    if _PAD_N[0] >= _PAD_CAP:
        return
    _PAD_N[0] += 1
    try:
        import torch.distributed as _d

        rank = _d.get_rank() if _d.is_initialized() else -1
        body = " ".join("{{}}={{}}".format(k, v) for k, v in kw.items())
        print("GLM52_PAD {{}} rank={{}} {{}}".format(site, rank, body), flush=True)
    except Exception as _e:
        print("GLM52_PAD probe-error {{}}".format(_e), flush=True)


def _pad_rows(t):
    try:
        return None if t is None else t.shape[0]
    except Exception:
        return None

'''.format(
    cap=CAP
)

# Anchor on the decorator+class pair. ForwardBatch is a @dataclass, so
# inserting before the bare `class` line splits the decorator from its class
# and the module fails to compile (hit once already, on eagle_info.py).
HELPER_ANCHOR = "@dataclass\nclass ForwardBatch(ForwardBatchDeepSeekMHAMixin):"

# --- P1 / P2: the pad entry point -------------------------------------------
P12_ANCHOR = """    def _pad_inputs_to_size(self, model_runner: ModelRunner, num_tokens, bs):
        # padding
        self.input_ids = self._pad_tensor_to_size(self.input_ids, num_tokens)
"""

P12_REPL = """    def _pad_inputs_to_size(self, model_runner: ModelRunner, num_tokens, bs):
        # GLM52_PAD P1: the requested target sizes and the pre-pad reality.
        _spec = self.spec_info
        _pad_log(
            "enter",
            num_tokens=num_tokens,
            bs=bs,
            inp=_pad_rows(self.input_ids),
            pos=_pad_rows(self.positions),
            hs=_pad_rows(getattr(_spec, "hidden_states", None)),
            ntpr=getattr(_spec, "num_tokens_per_req", None),
            fwd=self.forward_mode,
            plan=self.global_num_tokens_cpu,
            orig=self.original_global_num_tokens_cpu,
        )
        # padding
        self.input_ids = self._pad_tensor_to_size(self.input_ids, num_tokens)
"""

# The exit point: the last statement of the method body. Anchored on the
# mamba_track_seqlens block that closes the per-request padding, immediately
# before the mrope block.
P2_ANCHOR = """        if self.mrope_positions is not None:
            self.mrope_positions = torch.cat(
"""

P2_REPL = """        _pad_log(
            "exit",
            num_tokens=num_tokens,
            bs=bs,
            inp=_pad_rows(self.input_ids),
            pos=_pad_rows(self.positions),
            ocl=_pad_rows(self.out_cache_loc),
            seq=_pad_rows(self.seq_lens),
        )
        if self.mrope_positions is not None:
            self.mrope_positions = torch.cat(
"""

# --- P3: the spec hidden_states pad ------------------------------------------
P3_ANCHOR = """            spec_info.hidden_states = self._pad_tensor_to_size(
                spec_info.hidden_states, num_tokens
            )
"""

P3_REPL = """            _pad_log(
                "hs_pad",
                before=_pad_rows(spec_info.hidden_states),
                target=num_tokens,
                bs=bs,
                backup=_pad_rows(self.hidden_states_backup),
                fwd=self.forward_mode,
            )
            spec_info.hidden_states = self._pad_tensor_to_size(
                spec_info.hidden_states, num_tokens
            )
"""

# --- P4: the restore ---------------------------------------------------------
P4_ANCHOR = """            if self.forward_mode.is_decode():  # draft
                num_tokens = self.hidden_states_backup.shape[0]
"""

P4_REPL = """            if self.forward_mode.is_decode():  # draft
                num_tokens = self.hidden_states_backup.shape[0]
                _pad_log(
                    "unpad",
                    backup_rows=num_tokens,
                    bs=bs,
                    pos=_pad_rows(self.positions),
                    logits=_pad_rows(
                        getattr(logits_output, "next_token_logits", None)
                    ),
                    hs=_pad_rows(getattr(logits_output, "hidden_states", None)),
                )
"""


def die(msg):
    print("FAIL: {}".format(msg), file=sys.stderr)
    sys.exit(1)


def main():
    src = open(FBI).read()
    if MARK in src:
        print("already patched ({} present)".format(MARK))
        return

    pairs = [
        ("helper", HELPER_ANCHOR, HELPER + HELPER_ANCHOR),
        ("P1 enter", P12_ANCHOR, P12_REPL),
        ("P2 exit", P2_ANCHOR, P2_REPL),
        ("P3 hs_pad", P3_ANCHOR, P3_REPL),
        ("P4 unpad", P4_ANCHOR, P4_REPL),
    ]
    for name, anchor, _ in pairs:
        n = src.count(anchor)
        if n != 1:
            die("{} anchor matched {} times, expected 1".format(name, n))
    for _, anchor, repl in pairs:
        src = src.replace(anchor, repl, 1)

    open(FBI, "w").write(src)
    os.utime(FBI, None)
    pyc = os.path.join(
        os.path.dirname(FBI), "__pycache__", "forward_batch_info.cpython-310.pyc"
    )
    if os.path.exists(pyc):
        os.remove(pyc)
    print("patched {} with {}".format(FBI, MARK))


if __name__ == "__main__":
    main()
