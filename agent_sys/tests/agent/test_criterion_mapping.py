"""The criterion → test mapping is the deliverable, so something checks it.

The first condition for done is *"every
acceptance criterion maps to a **named test that exists and passes**"*, and
`agent/docs/design.md` carries that mapping. A name in it that no longer resolves is
the mapping quietly ceasing to be true — and it happened: a test removed when
`set_task` moved to the scheduler stayed in the table for four commits, which
is the same shape as a comment outliving the fact beneath it.

Four of the sixteen criteria are structural and one is a strict `xfail`, so the
mapping cannot be derived from the suite. It has to be written, and therefore
checked.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "agent" / "docs" / "design.md"


def _section() -> str:
    text = MAPPING.read_text()
    start = text.index("## 16. Criterion → test")
    nxt = text.find("\n## ", start + 1)
    return text[start : nxt if nxt != -1 else len(text)]


def _named() -> set[str]:
    return set(re.findall(r"`(test_[a-z0-9_]+)`", _section()))


def _defined() -> set[str]:
    found: set[str] = set()
    for path in (ROOT / "tests" / "agent").glob("test_*.py"):
        found |= set(re.findall(r"^def (test_[a-z0-9_]+)", path.read_text(), re.M))
    return found


def test_every_test_the_mapping_names_exists() -> None:
    missing = sorted(_named() - _defined())
    assert not missing, (
        f"design.md names tests that no longer exist: {missing}. "
        f"The mapping is the definition of done; a dangling name makes it untrue."
    )


def test_every_criterion_has_a_row() -> None:
    """Sixteen, and the count is `spec.md` §8's. A criterion that loses its row
    loses the only place its test is named."""
    numbered = [row for row in re.findall(r"^\| (\d+) \|", _section(), re.M)]
    assert numbered == [str(n) for n in range(1, 17)]
