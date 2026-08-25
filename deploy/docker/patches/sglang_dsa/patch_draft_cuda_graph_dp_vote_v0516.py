#!/usr/bin/env python3
"""Patch 04's dp_attn.py half, ported to the v0.5.16 base by anchor.

`draft_cuda_graph_dp_vote.diff` carries seven files. Six apply to v0.5.16 under
`--fuzz=0`; only `dp_attn.py` fails, and it fails for a boring reason -- v0.5.16
renamed the two neighbouring gates, so every context line in those hunks is
stale:

    v0.5.15.post1 / v0.5.17        v0.5.16
    can_run_decode_cuda_graph      can_cuda_graph
    can_run_prefill_cuda_graph     can_run_breakable_cuda_graph

Nothing about the mechanism changed, so this is an anchor script rather than a
re-cut diff: it edits by unique source text and therefore survives that kind of
rename, the same reasoning that makes patch 01 a script.

WHAT (unchanged from the diff -- see its header for the full analysis)
    `EagleDraftWorker.draft()` picks graph-replay vs eager PER RANK, and two of
    the guard's four terms are rank-dependent by construction. The two paths do
    not issue the same host-side collective sequence, so the DP group diverges.
    Fix: carry the choice as one more int64 slot in the MLP-sync all-gather the
    scheduler already performs, min()-reduced, so any rank needing eager takes
    the whole group eager. Inactive ranks contribute 1 (permissive).

WIRED INTO NO ARM, AND NOT KNOWN TO BE NEEDED ON v0.5.16
    Both `DSA_PATCH_SET` arms leave this out -- `PORT_SCRIPTS` in
    `apply_sglang_dsa_patches.sh` is empty. It is kept, verified in-image, only
    so that 04 does not have to be re-cut should it ever be genuinely wanted
    here. The mechanism above is real and measured, but on the gfx950 /
    v0.5.17 arm: revert 04 there and the first routed request deadlocks at the
    120 s timeout, patched it answers in 2.3 s.

    The low-concurrency argument for wanting it here -- idle ranks flipping the
    guard's occupancy term asymmetrically -- does not hold up. A 2 x 8 MI300X
    1P1D deployment with dp8, IndexShare-off and MTP(5,1,6), on a base matched
    to its host driver and carrying patch 01 alone, ran clean through warmup,
    conc 1 and conc 8/16/32 -- and conc 1 is exactly where that argument
    predicts divergence. See the driver precondition in README.md.

ON THIS BASE 04 IS SUBSTITUTED AT RUNTIME
    `--json-model-override-args '{"index_share_for_mtp_iteration":false}'`,
    which the gfx942 recipe passes on every leg. It removes the seed that makes
    the guard's inputs diverge, so no vote is needed.

The seven edits mirror the diff's seven dp_attn.py insertions one-for-one.
All-or-nothing: every anchor must match exactly once, or nothing is written and
this exits non-zero, so a drifted base fails the build instead of shipping a
half-applied fix.
"""

from __future__ import annotations

import os
import sys

REL = "python/sglang/srt/managers/scheduler_components/dp_attn.py"
MARKER = "can_run_draft_cuda_graph"

# (anchor, replacement). Anchors are chosen to be unique in the file; each is
# rewritten to itself plus the new line(s), so the edits are pure insertions.
EDITS: list[tuple[str, str]] = [
    # 1. dataclass field, beside the two gates it is reduced like. No default:
    #    a missed call site should be a TypeError, not a silent permissive True.
    (
        "    local_forward_mode: int\n"
        "    can_run_breakable_cuda_graph: bool\n",
        "    local_forward_mode: int\n"
        "    can_run_breakable_cuda_graph: bool\n"
        "    # min()-reduced like the two above, so any rank needing eager takes\n"
        "    # the whole group eager.\n"
        "    can_run_draft_cuda_graph: bool\n",
    ),
    # 2. contribute the rank-local answer to the all-gather (new slot 7)
    (
        "                int(self.can_run_breakable_cuda_graph),\n"
        "            ],\n",
        "                int(self.can_run_breakable_cuda_graph),\n"
        "                int(self.can_run_draft_cuda_graph),\n"
        "            ],\n",
    ),
    # 3. inactive-rank slot: permissive, so an idle peer cannot force eager
    (
        "                0,  # can_run_breakable_cuda_graph\n",
        "                0,  # can_run_breakable_cuda_graph\n"
        "                # Permissive: an inactive rank must not drag the group\n"
        "                # into eager.\n"
        "                1,  # can_run_draft_cuda_graph\n",
    ),
    # 4. min()-reduce the new slot out of the gathered tensor
    (
        "        self.can_run_breakable_cuda_graph = bool(tp0_info[:, 6].min().item())\n",
        "        self.can_run_breakable_cuda_graph = bool(tp0_info[:, 6].min().item())\n"
        "        self.can_run_draft_cuda_graph = bool(tp0_info[:, 7].min().item())\n",
    ),
    # 5. publish the group decision onto the batch
    (
        "    batch.can_run_dp_breakable_cuda_graph = mlp_sync_info.can_run_breakable_cuda_graph\n",
        "    batch.can_run_dp_breakable_cuda_graph = mlp_sync_info.can_run_breakable_cuda_graph\n"
        "    batch.can_run_dp_draft_cuda_graph = mlp_sync_info.can_run_draft_cuda_graph\n",
    ),
    # 6. compute this rank's answer from what the scheduler recorded on the batch
    (
        "    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False\n",
        "    # Rank-local answer, recorded on the batch by the scheduler just before\n"
        "    # this call. A None batch is permissive, as it is for the two above.\n"
        "    can_run_draft_cuda_graph = not (\n"
        "        local_batch is not None and local_batch.force_disable_draft_cuda_graph\n"
        "    )\n"
        "\n"
        "    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False\n",
    ),
    # 7. pass it to the constructor
    (
        "        can_run_breakable_cuda_graph=can_run_breakable_cuda_graph,\n",
        "        can_run_breakable_cuda_graph=can_run_breakable_cuda_graph,\n"
        "        can_run_draft_cuda_graph=can_run_draft_cuda_graph,\n",
    ),
]


def main() -> int:
    root = os.environ.get("SGLANG_DIR", "/sgl-workspace/sglang")
    path = os.path.join(root, REL)
    if not os.path.isfile(path):
        print(f"[draft-dp-vote] MISSING {path}", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print("[draft-dp-vote] already present — skipping")
        return 0

    # Verify every anchor first; write nothing unless all seven are unambiguous.
    for i, (anchor, _) in enumerate(EDITS, 1):
        n = src.count(anchor)
        if n != 1:
            print(
                f"[draft-dp-vote] anchor {i} matched {n} times (want 1) — base drifted, "
                "refusing to write",
                file=sys.stderr,
            )
            return 1

    out = src
    for anchor, repl in EDITS:
        out = out.replace(anchor, repl, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(out)

    # The all-gather width is derived from the tensor itself
    # (`info_width = local_info_tensor.numel()`), so slot 7 needs no other edit;
    # assert the two slice readers that could have hard-coded the old width.
    for slice_expr in ("tp0_info[:, 4:6]", "tp0_info[:, 5]"):
        if slice_expr not in out:
            print(
                f"[draft-dp-vote] expected {slice_expr} to survive — check slot layout",
                file=sys.stderr,
            )
            return 1

    print(f"[draft-dp-vote] applied 7 edits to {REL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
