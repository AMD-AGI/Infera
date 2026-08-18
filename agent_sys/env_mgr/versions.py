# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Version constraint handling. Thin wrapper over `packaging`."""

from __future__ import annotations

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


def _as_specifier(constraint: str) -> SpecifierSet:
    """A bare version ('1.2.3') means '>=1.2.3'; operators pass through."""
    c = constraint.strip()
    if c and c[0].isdigit():
        c = f">={c}"
    return SpecifierSet(c)


def satisfies(actual: str | None, constraint: str | None) -> bool:
    if constraint is None:
        return True
    if actual is None:
        return False
    try:
        return Version(actual) in _as_specifier(constraint)
    except (InvalidVersion, InvalidSpecifier):
        return False


def constraints_conflict(a: str | None, b: str | None) -> bool:
    """True if two non-empty constraints are NOT semantically equivalent.

    Uses the same bare-version normalization as `satisfies` (via
    `_as_specifier`), so "0.5" and ">=0.5" are equal, not a conflict. If either
    constraint is unparseable, fall back to a textual comparison.
    """
    if a is None or b is None:
        return False
    try:
        return _as_specifier(a) != _as_specifier(b)
    except InvalidSpecifier:
        return a.strip() != b.strip()
