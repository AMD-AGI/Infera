# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criteria 3, 4, 5 — at the **userspace** layer. Design §14.5.

Which layer a test targets is not a detail here. Measured, the kernel already
denies all three documented ``startswith`` defeats with no userspace check
involved, so a test that satisfies criterion 3 by unit-testing a comparison
function proves nothing about confinement. `test_confine.py` is the other half,
and says so.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from env_mgr.fs.path import canonical_here, canonical_syntax, contained, resolve_strict


@pytest.fixture
def zone(tmp_path: Path) -> str:
    z = tmp_path / "zone"
    (z / "inner").mkdir(parents=True)
    (z / "inner" / "file.txt").write_text("in")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("out")
    return str(z)


# --------------------------------------------------------------- criterion 3


def test_sibling_prefix_denied(zone: str, tmp_path: Path) -> None:
    """CVE-2025-54794's shape: `zone-EVIL` shares the zone's name prefix.

    A bare `startswith` passes it. The trailing separator is what stops it, and
    the assertion below is that both statements are true at once — otherwise a
    passing test could mean the fixture never built the sibling.
    """
    evil = tmp_path / "zone-EVIL"
    evil.mkdir()
    target = evil / "x.txt"
    target.write_text("owned")
    assert str(target).startswith(zone), "the fixture did not build the defeat"
    assert contained(target, zone) is False


def test_symlink_out_denied(zone: str, tmp_path: Path) -> None:
    link = Path(zone) / "escape"
    link.symlink_to(tmp_path / "outside")
    assert contained(link / "secret.txt", zone) is False


def test_dotdot_denied(zone: str) -> None:
    assert contained(os.path.join(zone, "inner", "..", "..", "outside"), zone) is False


def test_inside_is_allowed(zone: str) -> None:
    """The positive control. Without it, a `contained` that always returned
    False would pass every test above."""
    assert contained(os.path.join(zone, "inner", "file.txt"), zone) is True
    assert contained(zone, zone) is True


# --------------------------------------------------------------- criterion 4


def test_broken_symlink_denied(zone: str) -> None:
    link = Path(zone) / "broken"
    link.symlink_to(Path(zone) / "does-not-exist")
    assert contained(link, zone) is False


def test_symlink_loop_denied(zone: str) -> None:
    a = Path(zone) / "loop_a"
    b = Path(zone) / "loop_b"
    a.symlink_to(b)
    b.symlink_to(a)
    assert contained(a, zone) is False


def test_nonstrict_resolve_is_not_used(zone: str) -> None:
    """The trap, stated as a test.

    ``os.path.realpath`` and ``Path.resolve()`` at its default ``strict=False``
    both return a partly-resolved path for a broken symlink **without raising**,
    which is what an implementation reaches for. `resolve_strict` must not.
    """
    broken = Path(zone) / "nope-link"
    broken.symlink_to(Path(zone) / "nope")
    assert os.path.realpath(broken)  # does not raise, returns something
    assert Path(broken).resolve() is not None  # nor does this
    assert resolve_strict(broken) is None


# --------------------------------------------------------------- criterion 5


def test_nul_byte_rejected(zone: str) -> None:
    assert contained(zone + "\x00/etc/passwd", zone) is False
    assert canonical_syntax("/tmp/a\x00b") is False


def test_valueerror_is_caught_too(zone: str) -> None:
    """Rule 4 raises `ValueError`, **not** `OSError`.

    A handler written the natural way — ``except OSError: return False`` — lets
    a NUL through to whatever runs next, which makes rule 4 dead code. One line,
    and it is the difference between rules 3 and 4 composing.
    """
    with pytest.raises(ValueError):
        Path(zone + "\x00").resolve(strict=True)
    assert resolve_strict(zone + "\x00") is None


# ------------------------------------------- design D4: the canonical grammar


@pytest.mark.parametrize(
    "path",
    ["/usr", "/usr/lib/x", "/"],
)
def test_canonical_syntax_accepts_canonical_forms(path: str) -> None:
    assert canonical_syntax(path) is True


@pytest.mark.parametrize(
    "path",
    ["usr", "/usr/", "/usr/.", "/usr/../etc", "/usr//lib", "/usr/*", "/a\x00b", ""],
)
def test_canonical_syntax_rejects_the_rest(path: str) -> None:
    """Including the wildcard, which is not an anticipation of a feature: it is
    the closed side of the covering grammar. If the schema never admits ``*``,
    no component can be the one that gives it meaning."""
    assert canonical_syntax(path) is False


def test_canonical_here_rejects_a_symlink(tmp_path: Path) -> None:
    """Exact equality and realpath disagree on a symlink, in the direction
    `closure` §6.3 forbids: ``covers()`` says "not covered" while this module
    would grant. Requiring canonical form makes them agree by construction."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert canonical_here(str(real)) is True
    assert canonical_here(str(link)) is False
