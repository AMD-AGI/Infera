# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criterion 1 — a domain is registered, reloaded idempotently, and its kind
determines its layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from env_mgr.fs.domain import Domain, DomainKind, DomainRegistry, subdir_for


def test_register_idempotent(tmp_path: Path) -> None:
    reg = DomainRegistry()
    first = reg.register("store", str(tmp_path / "store"), DomainKind.HANDOFF_STORAGE)
    second = reg.register("store", str(tmp_path / "store"), DomainKind.HANDOFF_STORAGE)
    assert first == second
    assert isinstance(first, Domain)
    assert len(reg) == 1


def test_register_rejects_a_changed_root(tmp_path: Path) -> None:
    """A different root or kind for a live name is an error, not an update.

    A registry that quietly updated would have two answers to *where does this
    domain live* and no way to tell which one the zones on disk used.
    """
    reg = DomainRegistry()
    reg.register("store", str(tmp_path / "a"), DomainKind.HANDOFF_STORAGE)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("store", str(tmp_path / "b"), DomainKind.HANDOFF_STORAGE)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("store", str(tmp_path / "a"), DomainKind.PLAYGROUND)


def test_reload_preserves_playground(tmp_path: Path) -> None:
    """Registration touches nothing on disk when the name is already live —
    which is what lets a playground survive a restart (spec §6.2)."""
    root = tmp_path / "play"
    root.mkdir()
    (root / "half-finished.txt").write_text("scratch")

    DomainRegistry().register("play", str(root), DomainKind.PLAYGROUND)
    # A fresh process, the same declaration.
    DomainRegistry().register("play", str(root), DomainKind.PLAYGROUND)
    assert (root / "half-finished.txt").read_text() == "scratch"


def test_kind_decides_layout() -> None:
    assert subdir_for(DomainKind.HANDOFF_STORAGE) == "handoffs"
    assert subdir_for(DomainKind.WORKSPACE) == "workspace"
    assert subdir_for(DomainKind.PLAYGROUND) == "playground"


def test_get_names_the_candidates(tmp_path: Path) -> None:
    """Following `env_mgr/registry.py`, which already does exactly this."""
    reg = DomainRegistry()
    reg.register("store", str(tmp_path / "s"), DomainKind.HANDOFF_STORAGE)
    with pytest.raises(KeyError, match=r"have \['store'\]"):
        reg.get("typo")


def test_storage_root_needs_exactly_one(tmp_path: Path) -> None:
    """Two would be two answers to *where does a zone go*, and the layout may
    only have one."""
    reg = DomainRegistry()
    with pytest.raises(ValueError, match="exactly one"):
        reg.storage_root()
    reg.register("a", str(tmp_path / "a"), DomainKind.HANDOFF_STORAGE)
    assert reg.storage_root() == str((tmp_path / "a").resolve())
    reg.register("b", str(tmp_path / "b"), DomainKind.HANDOFF_STORAGE)
    with pytest.raises(ValueError, match="exactly one"):
        reg.storage_root()
