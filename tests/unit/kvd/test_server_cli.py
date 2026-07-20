###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for `python -m infera.kvd`'s CLI parsing.

We don't spawn a subprocess — just call into the argparse builder
directly so we can assert on the resolved namespace.
"""

from __future__ import annotations

import pytest

from infera.kvd.server import _parse_size, _parse_size_or_auto


def test_parse_size_accepts_suffixes():
    assert _parse_size("32G") == 32 * 1024**3
    assert _parse_size("512M") == 512 * 1024**2
    assert _parse_size("128K") == 128 * 1024
    assert _parse_size("1T") == 1024**4
    assert _parse_size("1024") == 1024  # plain int


def test_parse_size_or_auto_passes_auto_through():
    """The `auto` sentinel must come back unchanged so main() can
    resolve it after parsing other args. Mixed-case is tolerated."""
    assert _parse_size_or_auto("auto") == "auto"
    assert _parse_size_or_auto("AUTO") == "auto"
    assert _parse_size_or_auto("  auto  ") == "auto"


def test_parse_size_or_auto_falls_through_to_size():
    """Anything not `auto` is parsed like a size."""
    assert _parse_size_or_auto("64G") == 64 * 1024**3
    assert _parse_size_or_auto("0") == 0


def test_parse_size_or_auto_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_size_or_auto("not-a-size")
