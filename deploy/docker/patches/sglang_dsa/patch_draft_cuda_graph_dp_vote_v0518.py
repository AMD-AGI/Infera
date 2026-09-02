#!/usr/bin/env python3
"""Patch 04's one v0.5.18-drifted edit in dp_attn.py, carried by anchor.

`draft_cuda_graph_dp_vote.diff` carries seven files. On the v0.5.18 base six
files apply clean, and `dp_attn.py` applies 6 of its 7 hunks -- only hunk 4, the
min()-reduce, rejects. It rejects for a boring reason: upstream turned the
per-field device reads into one D2H copy and renamed the tensor with it, so
every context line in that hunk is stale.

    v0.5.17                                         v0.5.18
    tp0_info[:, 6].min().item()                     tp0_info_cpu[:, 6].min()

Nothing about the mechanism changed, so this is an anchor script rather than a
re-cut diff -- it edits by unique source text and therefore survives that kind
of rename, the same reasoning that makes patch 01 a script.

WHY THIS ONE HUNK IS THE WHOLE PATCH
    Losing it does NOT leave patch 04 partly applied in a way that degrades
    gracefully. The other six edits declare the dataclass field, contribute the
    rank-local answer to the all-gather, and consume the result -- but hunk 4 is
    the only place the gathered column is min()-reduced back onto
    `self.can_run_draft_cuda_graph`. Without it that attribute keeps the value
    the constructor put there, which is THIS RANK's answer. The vote silently
    never happens, every consumer downstream reads a rank-local bool, and the DP
    group splits across graph-replay and eager exactly as it did unpatched.

    That is the failure mode the apply script's header warns about: "an inert 04
    looks exactly like a working one until load." It is also why this cannot be
    left to the `dp_attn.py:can_run_draft_cuda_graph` marker -- that identifier
    is present after six hunks, so the marker passes on an inert patch. The
    assertions below are the real gate.

WHAT (unchanged from the diff -- see its header for the full analysis)
    `EagleDraftWorker.draft()` picks graph-replay vs eager PER RANK, and two of
    the guard's four terms are rank-dependent by construction. The two paths do
    not issue the same host-side collective sequence, so the DP group diverges
    and deadlocks on the first routed request. Fix: carry the choice as one more
    int64 slot in the MLP-sync all-gather the scheduler already performs,
    min()-reduced, so any rank needing eager takes the whole group eager.

SCOPE
    Wired into the `full` arm only, which is the v0.5.18 gfx950 image
    (`Dockerfile.sglang`). The `indexer` arm (gfx942 / v0.5.16) does not carry
    04 at all and substitutes it at runtime with
    `--json-model-override-args '{"index_share_for_mtp_iteration":false}'`; its
    unused sibling port is `patch_draft_cuda_graph_dp_vote_v0516.py`, which
    carries all seven edits because on that base the whole file rejects.

    DELETE THIS FILE when the diff is re-cut against a newer base, or when 04
    lands upstream. It is a one-line bridge, not a second source of truth.

All-or-nothing: the anchor must match exactly once and all six sibling edits
must already be in place, or nothing is written and this exits non-zero -- so a
drifted base fails the build instead of shipping a patch that is present,
inert, and indistinguishable from a working one.
"""

from __future__ import annotations

import os
import sys

REL = "python/sglang/srt/managers/scheduler_components/dp_attn.py"

# The edit. Anchor is rewritten to itself plus the new line, so this is a pure
# insertion, and it is placed beside the two gates it is reduced like.
ANCHOR = "        self.can_run_prefill_cuda_graph = bool(tp0_info_cpu[:, 6].min())\n"
INSERT = (
    "        self.can_run_draft_cuda_graph = bool(tp0_info_cpu[:, 7].min())\n"
)

# Proof that hunk 4 specifically landed. Distinct from the bare identifier,
# which six hunks already satisfy.
MARKER = "self.can_run_draft_cuda_graph = bool(tp0_info_cpu[:, 7]"

# The six edits the diff itself applies. If any is missing this script is being
# run somewhere it does not belong -- inserting the reduce alone would then read
# a column nothing writes.
SIBLINGS: list[tuple[str, str]] = [
    ("dataclass field", "    can_run_draft_cuda_graph: bool\n"),
    ("all-gather slot", "                int(self.can_run_draft_cuda_graph),\n"),
    ("inactive-rank slot", "                1,  # can_run_draft_cuda_graph\n"),
    (
        "publish onto batch",
        "    batch.can_run_dp_draft_cuda_graph = mlp_sync_info.can_run_draft_cuda_graph\n",
    ),
    ("rank-local answer", "    can_run_draft_cuda_graph = not (\n"),
    (
        "constructor arg",
        "        can_run_draft_cuda_graph=can_run_draft_cuda_graph,\n",
    ),
]

# Slot 7 needs no width edit -- the all-gather derives its width from the tensor
# itself (`info_width = local_info_tensor.numel()`). These two readers are the
# ones that could have hard-coded the old width; assert they survive.
SLICE_READERS = ("tp0_info_cpu[:, 4:6]", "tp0_info_cpu[:, 5]")


def main() -> int:
    root = os.environ.get("SGLANG_DIR", "/sgl-workspace/sglang")
    path = os.path.join(root, REL)
    if not os.path.isfile(path):
        print(f"[draft-dp-vote-v0518] MISSING {path}", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print("[draft-dp-vote-v0518] already present — skipping")
        return 0

    for name, text in SIBLINGS:
        if text not in src:
            print(
                f"[draft-dp-vote-v0518] sibling edit missing ({name}) — "
                "draft_cuda_graph_dp_vote.diff did not apply here, refusing to write",
                file=sys.stderr,
            )
            return 1

    n = src.count(ANCHOR)
    if n != 1:
        print(
            f"[draft-dp-vote-v0518] anchor matched {n} times (want 1) — base drifted, "
            "refusing to write",
            file=sys.stderr,
        )
        return 1

    out = src.replace(ANCHOR, ANCHOR + INSERT, 1)

    for slice_expr in SLICE_READERS:
        if slice_expr not in out:
            print(
                f"[draft-dp-vote-v0518] expected {slice_expr} to survive — "
                "check slot layout",
                file=sys.stderr,
            )
            return 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(out)

    print(f"[draft-dp-vote-v0518] applied the min()-reduce to {REL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
