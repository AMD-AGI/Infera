#!/usr/bin/env python3
"""E3 diagnostic: why does the MLP hidden-states all-gather size-mismatch?

Observed (arm E3, conc=32, 2026-07-30 09:48 and 10:00, both legs of the run):

    File ".../layers/communicator.py", line 1095, in _gather_hidden_states_and_residual
      dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
    ...
    ValueError: output tensor size must be equal to world_size times input
                tensor size

Two candidate causes, and they need different fixes, so guessing is not on:

  C1 -- the patch-2b trim/restore leaks. The trim drops DP-padding rows before
        attention and the wrapper pads them back; if some return path skips the
        restore, hidden_states is short by exactly `pad_rows`.
        AGAINST C1: the trim asserts `real <= physical`, and that assert never
        fired, so the leading dim only ever shrinks by a known amount that the
        wrapper re-adds.

  C2 -- ranks disagree on DpPaddingMode. MAX_LEN sizes the global buffer as
        dp_size * max_tokens and uses all_gather_into_tensor; SUM_LEN sizes it
        as sum(tokens). If one rank plans MAX_LEN and another SUM_LEN, the
        gather buffer is inconsistent -- which is precisely this ValueError.
        Patch 4 (P4V2) changes WHICH ranks run the draft CUDA graph, and the
        captured-graph path fixes the padding mode at capture time while the
        eager path recomputes it per step. This is hypothesis H3 from the
        project charter, which was never confirmed nor refuted because
        `dp_padding_mode` read as None on every record last time.

This records both at the gather site: the local/global row counts actually
handed to the collective, plus the padding mode and forward mode. Whichever of
C1/C2 is true is then a fact, not a story.

Rank-tagged, capped, and flushed so a crash cannot swallow the tail.
Idempotent; anchors asserted unique; invalidates the .pyc.
"""
import os
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/layers/dp_attention.py"
MARK = "GLM52_E3INSTR"
CAP = 400

PROBE = '''
# GLM52_E3INSTR -- see /shared_nfs/yihou_exp3way/e3/instr_e3.py
_E3I_N = [0]
_E3I_CAP = {cap}


def _e3i_log(global_tokens, local_tokens, forward_batch):
    """Record the shapes the all-gather is about to be given."""
    if _E3I_N[0] >= _E3I_CAP:
        return
    _E3I_N[0] += 1
    try:
        import torch.distributed as _d

        rank = _d.get_rank() if _d.is_initialized() else -1
        g = tuple(global_tokens.shape)
        l = tuple(local_tokens.shape)
        ws = get_attention_tp_size() if False else None
        mode = getattr(forward_batch, "dp_padding_mode", None)
        fmode = getattr(forward_batch, "forward_mode", None)
        gnt = getattr(forward_batch, "global_num_tokens", None)
        print(
            "GLM52_E3INSTR rank={{}} global={{}} local={{}} ratio={{}} "
            "pad_mode={{}} fwd_mode={{}} global_num_tokens={{}}".format(
                rank,
                g,
                l,
                (g[0] / l[0]) if l and l[0] else None,
                mode,
                fmode,
                gnt,
            ),
            flush=True,
        )
    except Exception as _e:  # never let the probe break the run
        print("GLM52_E3INSTR probe-error {{}}".format(_e), flush=True)

'''.format(
    cap=CAP
)

PROBE_ANCHOR = "def _dp_gather_via_all_gather(\n"

# The traceback points at the `get_attention_tp_size() == 1` early-return
# branch -- which is exactly our case, since DPA8 makes the attention TP group
# one rank wide. That branch is 8-space indented, and its body is what
# distinguishes it from the other all_gather_into_tensor call in this function.
CALL_ANCHOR = """    if get_attention_tp_size() == 1:
        get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return"""

CALL_REPL = """    if get_attention_tp_size() == 1:
        _e3i_log(global_tokens, local_tokens, forward_batch)  # GLM52_E3INSTR
        get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return"""


def main():
    src = open(TARGET).read()
    if MARK in src:
        print("already instrumented")
        return 0

    assert src.count(PROBE_ANCHOR) == 1, (
        "probe anchor not unique: %d" % src.count(PROBE_ANCHOR)
    )
    assert src.count(CALL_ANCHOR) == 1, (
        "call anchor not unique: %d" % src.count(CALL_ANCHOR)
    )

    src = src.replace(PROBE_ANCHOR, PROBE + PROBE_ANCHOR, 1)
    src = src.replace(CALL_ANCHOR, CALL_REPL, 1)

    open(TARGET, "w").write(src)
    # PITFALL: a restored/rewritten .py can keep an mtime that matches the
    # cached bytecode, so CPython silently runs the UNPATCHED .pyc.
    os.utime(TARGET, None)
    d = os.path.dirname(TARGET)
    pd = os.path.join(d, "__pycache__")
    if os.path.isdir(pd):
        for f in os.listdir(pd):
            if f.startswith("dp_attention."):
                os.remove(os.path.join(pd, f))
    print("instrumented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
