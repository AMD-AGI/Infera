"""Criterion 13's second half — design §12 and O1.

**Criterion 13 splits, and only half of it is an `xfail`.** "Stored in Claude
Code's canonical format" is testable now and passes — `test_spec.py`'s
`test_material_stored_canonically`. "Converts … losslessly for what both
support" is the half with no testable formulation.

Marking the whole row `xfail` would have been the opposite error: hiding a
satisfied requirement behind an unsatisfiable one.
"""

from __future__ import annotations

import pytest

REASON = (
    "O1: 'losslessly for what both support' requires knowing the intersection of "
    "two harnesses' feature sets, and no converter computes it — everyone "
    "hand-maintains a table, and both reference implementations' tables fail "
    "invisibly. pandoc classifies a dropped block as INFO, so --fail-if-warnings "
    "exits 0 with the content gone; kompose's 25-entry unsupported-key table has "
    "no production caller. The criterion needs to name the artefact that defines "
    "'what both support' and the test that keeps it honest, and to separate "
    "*unsupported* from *unknown*, which it conflates."
)


@pytest.mark.xfail(strict=True, reason=REASON)
def test_transform_lossless() -> None:
    """There is no transform helper, and design §3.5 places it outside this
    package deliberately: it converts between two third parties' formats, needs
    neither `AgentSpec` nor `AgentBackend`, and its release cadence is the union
    of the harnesses' — `rulesync`, which does exactly this job for ~30
    harnesses, has **299 releases and seven major versions inside two weeks**.

    **`strict=True` on purpose.** An expected failure that passes is a failure:
    if somebody makes this green without answering O1, that is the thing to
    find out about.
    """
    from agent import transform  # noqa: F401 — the module does not exist

    raise AssertionError("unreachable")
