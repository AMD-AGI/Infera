#!/usr/bin/env python3
"""The half of PR #32209's patch 2b that the first port left out.

WHY THIS EXISTS. `patch2b_32209_style.py` carried #32209's trim/restore around
the DSA decode call but NOT this hunk, on the judgement that it was
CUDA-specific. Three measured rounds of arm e3a say that judgement was wrong:
the trim half alone crashes at conc=32, every time, with

    ValueError: output tensor size must be equal to world_size times input
                tensor size      (dp_gather_replicate -> _dp_gather_via_all_gather)

WHAT THE THREE ROUNDS ESTABLISHED, and what they ruled out:

  Round 3, crashing iteration (bs=3, buffer 24 = 8 ranks x 3 rows,
  plan global_num_tokens_cpu=[3]*8, real orig=[2,3,0,1,3,2,3,2]):

    GLM52_DSTEP vote  rank=0..7  graph=0        <- IDENTICAL on all 8 ranks
    GLM52_DSTEP step  rank=0 i=0 rows=2
                      rank=1 i=0 rows=3
                      rank=2 i=0 rows=0 (idle)
                      rank=3 i=0 rows=1
                      rank=4 i=0 rows=3
                      rank=5 i=0 rows=2
                      rank=6 i=0 rows=3
                      rank=7 i=0 rows=2         <- matches `orig` position-wise
    GLM52_E3INSTR     rank=0,5,7  local=(4,...) ratio=6.0   <- the three orig==2 ranks

  RULED OUT (do not re-litigate):
    - DpPaddingMode divergence (charter H3): pad_mode=1 (MAX_LEN) on all eight
      ranks, all three rounds.
    - patch 4's graph/eager vote failing: the vote is uniform, 51 graph / 7
      eager, identical on every rank.
    - a rank replaying the draft graph so Python probes go blind (hypothesis A):
      every rank voted graph=0 on this iteration.
    - a rank skipping the DSA site (hypothesis B): all eight ranks logged both
      i=0 and i=1.
    - the trim creating rows: five ranks trimmed and still delivered correctly;
      trim+restore is row-neutral.

  WHAT IS LEFT, and what this hunk addresses. The three ranks whose REAL token
  count is 2 (while the group planned 3) are exactly the three that fault, and
  they hand the collective 4 rows -- neither their real 2 nor the planned 3.
  A row count that is neither the local nor the agreed value is a carried-over
  one: `hidden_states` is threaded from each draft step into the next
  (`spec_info.hidden_states = hidden_states`) and, on the first step, in from
  the previous decode iteration. Nothing on our path ever trims it back to the
  local token count, so a padded/stale row count survives across steps and
  across iterations, and the gather inside the NEXT forward sees it.

  #32209 closes exactly that carry: it slices next_token_logits, hidden_states
  and positions back to `num_local_tokens = input_ids.shape[0]` after every
  draft step, so nothing padded is ever threaded forward.

HONEST SCOPE. This is upstream's own code, applied where upstream applies it.
It is NOT independently derived by us, and this script does not by itself prove
the fix -- arm e3a at conc=32 does. Two earlier claims about this same patch
("page_table_1.shape[0] is the fix", "rank 4 misses a trim") were reported and
then retracted; nothing here is to be called a fix until conc=32 is green with
the graph provably in use.

DELIBERATE DIVERGENCES from #32209's text, both mechanical:
  - `tuple[...]` -> `Tuple[...]`: this baseline runs Python 3.10 but the module
    has no `from __future__ import annotations`, and the helper's annotations
    are evaluated at def time.
  - upstream also rewrites the topk==1 CUDA branch to use the fused
    `draft_topk1_postprocess`; that branch is `not _is_hip`-gated and dead on
    our HIP path, so only the row-slicing behaviour is ported.

Requires patch2b_32209_style.py (the trim half) to be present -- the two are
halves of one change. Refuses to run otherwise.

Idempotent. Anchors asserted to match exactly once. Invalidates the .pyc.
"""
import os
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
MARK = "GLM52_P2BSLICE"

HELPER = '''def _slice_draft_output_to_local_tokens(
    next_token_logits,
    hidden_states,
    positions,
    num_local_tokens,
):
    """GLM52_P2BSLICE: discard DP-attention padding rows before eager draft
    postprocessing. Ported verbatim from upstream PR #32209.

    Without this, a padded or stale row count is threaded into the next draft
    step through `spec_info.hidden_states`, and the DP all-gather inside that
    step's forward sees a local row count that matches neither this rank's real
    tokens nor the count the MLP-sync all-gather agreed on. Measured on arm e3a
    (2026-07-30, three rounds): the ranks whose real token count was below the
    group plan delivered 4 rows into a 3-row slot and faulted in
    dp_gather_replicate.
    """
    for name, tensor in (
        ("next_token_logits", next_token_logits),
        ("hidden_states", hidden_states),
        ("positions", positions),
    ):
        if tensor is not None and tensor.shape[0] < num_local_tokens:
            raise RuntimeError(
                f"EAGLE draft {name} has {tensor.shape[0]} rows, "
                f"but {num_local_tokens} local tokens need postprocessing"
            )

    return (
        next_token_logits[:num_local_tokens],
        hidden_states[:num_local_tokens] if hidden_states is not None else None,
        positions[:num_local_tokens],
    )


'''

HELPER_ANCHOR = "class EagleDraftWorker(EagleDraftWorkerBase):"

# --- capture the local token count for this step ----------------------------
NLOCAL_ANCHOR = """            # Set inputs
            forward_batch.input_ids = input_ids
"""

NLOCAL_REPL = """            # Set inputs
            # GLM52_P2BSLICE: the row count this rank actually owns this step.
            num_local_tokens = input_ids.shape[0]
            forward_batch.input_ids = input_ids
"""

# --- slice the step output, and use the sliced tensors downstream -----------
# The baseline body between the forward and the end of the loop is replaced as
# one block so every consumer reads the sliced tensors. Anchored on the exact
# baseline text (verified present on 2026-07-30); if the baseline shifts, the
# uniqueness assert below fails loudly rather than silently half-applying.
BODY_ANCHOR = """                logits_output = self.draft_runner.forward(forward_batch).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            maybe_detect_inf(logits_output.next_token_logits, f"draft_forward step {i}")
            if self.server_args.speculative_use_rejection_sampling:
                probs = renorm_draft_probs(
                    logits_output.next_token_logits,
                    forward_batch.sampling_info,
                    self.server_args.speculative_use_rejection_sampling,
                )
                topk_p, topk_index = fast_sample(probs, num_samples=1)
                draft_probs_list.append(probs)
            elif self.topk == 1 and not _is_hip:
                topk_index = torch.argmax(
                    logits_output.next_token_logits, dim=-1, keepdim=True
                )
                topk_p = torch.ones_like(topk_index, dtype=torch.float32)
            else:
                probs = renorm_draft_probs(
                    logits_output.next_token_logits,
                    forward_batch.sampling_info,
                    self.server_args.speculative_use_rejection_sampling,
                )
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states
            forward_batch.positions.add_(1)
"""

BODY_REPL = '''                logits_output = self.draft_runner.forward(forward_batch).logits_output
            # GLM52_P2BSLICE: drop DP-attention padding rows before ANY
            # postprocessing, so nothing padded is threaded into the next step
            # via hidden_states. Upstream PR #32209.
            next_token_logits, next_hidden_states, local_positions = (
                _slice_draft_output_to_local_tokens(
                    logits_output.next_token_logits,
                    logits_output.hidden_states,
                    forward_batch.positions,
                    num_local_tokens,
                )
            )
            maybe_detect_nan(next_token_logits, f"draft_forward step {i}")
            maybe_detect_inf(next_token_logits, f"draft_forward step {i}")
            if self.server_args.speculative_use_rejection_sampling:
                probs = renorm_draft_probs(
                    next_token_logits,
                    forward_batch.sampling_info,
                    self.server_args.speculative_use_rejection_sampling,
                )
                topk_p, topk_index = fast_sample(probs, num_samples=1)
                draft_probs_list.append(probs)
            elif self.topk == 1 and not _is_hip:
                topk_index = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                topk_p = torch.ones_like(topk_index, dtype=torch.float32)
            else:
                probs = renorm_draft_probs(
                    next_token_logits,
                    forward_batch.sampling_info,
                    self.server_args.speculative_use_rejection_sampling,
                )
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = next_hidden_states
            # positions advances on the SLICED view; it aliases the first
            # num_local_tokens rows of forward_batch.positions, so the real rows
            # still advance and the padding rows are simply left alone.
            local_positions.add_(1)
'''


def die(msg):
    print("FAIL: {}".format(msg), file=sys.stderr)
    sys.exit(1)


def main():
    src = open(TARGET).read()

    if MARK in src:
        print("already patched ({} present); nothing to do".format(MARK))
        return
    if "_p2bv2_trim_decode_dp_padding" not in open(
        "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"
    ).read():
        die(
            "the trim half (patch2b_32209_style.py) is not applied; these are "
            "two halves of one change and this half alone is not meaningful"
        )

    for name, anchor in (
        ("helper", HELPER_ANCHOR),
        ("num_local_tokens", NLOCAL_ANCHOR),
        ("loop body", BODY_ANCHOR),
    ):
        n = src.count(anchor)
        if n != 1:
            die("{} anchor matched {} times, expected 1".format(name, n))

    src = src.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    src = src.replace(NLOCAL_ANCHOR, NLOCAL_REPL, 1)
    src = src.replace(BODY_ANCHOR, BODY_REPL, 1)

    open(TARGET, "w").write(src)
    # CLAUDE.md: a rewritten .py can keep an mtime matching the cached bytecode,
    # and CPython then runs the UNPATCHED .pyc. Touch and purge.
    os.utime(TARGET, None)
    pyc = os.path.join(
        os.path.dirname(TARGET), "__pycache__", "eagle_worker_v2.cpython-310.pyc"
    )
    if os.path.exists(pyc):
        os.remove(pyc)
    print("patched {} with {}".format(TARGET, MARK))


if __name__ == "__main__":
    main()
