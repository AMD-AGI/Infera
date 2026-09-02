"""`handoff.version_dir` and `env_mgr.fs.layout.handoff_version_dir` are two
writers of one path shape, and this is the price of that.

`handoff` design §6.2 says **"exactly one function computes a path and the
on-disk shape is private"**, with Bazel #23576 as the reason: a path-shape change
survived there only because consumers use `file.path` rather than composing
strings. `env_mgr` composes the string anyway — `grants.py` and `meta.py` need
`<root>/<hid>/v<N>/` to grant access to it.

**It is duplicated by construction, not by carelessness.** `docs/interfaces.md`
§4.6 permits `env_mgr` to import `task_graph` and nothing else of ours, so it
cannot call `handoff.version_dir` even if it wanted to. The alternatives were a
new package edge for one function, or a shared constant with no honest home —
a path layout belongs to neither `task_graph` nor `spec_loader`, and
`engineer_principle.md` §2 says an unowned concept reported beats an owned
concept in the wrong place.

So the shape stays declared twice and something checks it, which is exactly
`test_pushable.py`'s bargain one directory over. `docs/interfaces.md` §8 states
the terms: *"The test is not a nicety attached to the decision; it is the
decision's price."* **A test may import both packages**, because tests are not
under §4's import rule.

Found by the `handoff` implementer while grepping for callers of `resolve`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from env_mgr.fs.layout import handoff_version_dir
from handoff.store import version_dir

CASES = [
    ("/srv/handoffs", "h-abc123", 0),
    ("/srv/handoffs", "h-abc123", 1),
    ("/srv/handoffs", "h-abc123", 42),
    ("/tmp/store", "0191f0c2-9e5a-7c31-a4d1-2b8e6f3a5c47", 7),
    ("relative/root", "h-1", 0),
]


@pytest.mark.parametrize(("root", "hid", "version"), CASES)
def test_the_two_writers_of_the_layout_agree(root: str, hid: str, version: int) -> None:
    """The day the layout moves, one of the two will not move with it.

    `handoff.version_dir` returns a `Path` and `handoff_version_dir` returns a
    `str`; the shape is what has to agree, so compare the shape.
    """
    assert Path(handoff_version_dir(root, hid, version)) == version_dir(Path(root), hid, version)


def test_the_shape_is_root_then_id_then_v_number() -> None:
    """Pin the shape itself, so a *matching* change to both still gets read.

    Without this, the pair could agree on something neither design describes.
    """
    assert version_dir(Path("/r"), "h-9", 3) == Path("/r/h-9/v3")
    assert handoff_version_dir("/r", "h-9", 3) == "/r/h-9/v3"


def test_the_two_spellings_of_the_granted_subdirectories_agree() -> None:
    """`interfaces.md` §4.14 grants a producing agent two subtrees of `v<N>/`,
    and **each package spells their names itself**.

    Same bargain as the path shape above, for the same reason: §4.6 lets
    `env_mgr` import `task_graph` and nothing else of ours, so it cannot read
    `handoff`'s constants even though it must name the directories to grant
    them. `handoff.allocate` **creates** exactly the directories `env_mgr`
    **grants**, and a granted path that does not exist either raises in
    `prepare` or evaporates silently — so a divergence here is a dispatch
    failure, not a cosmetic one.
    """
    from env_mgr.grants import CLAIM_DIR as env_claim
    from env_mgr.grants import CONTENT_DIR as env_content
    from handoff.store import CLAIM_DIR as handoff_claim
    from handoff.store import CONTENT_DIR as handoff_content

    assert handoff_content == env_content == "content"
    assert handoff_claim == env_claim == "claim"
