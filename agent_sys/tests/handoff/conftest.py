"""Fixtures. Every test builds its own store against `tmp_path`; nothing is
process-global, and no test depends on another's leftovers."""

from __future__ import annotations

from pathlib import Path

import pytest

from handoff import content as content_mod
from handoff.protocols import HandoffKind
from handoff.store import FilesystemStore, KindSource
from task_graph.ids import HandoffId

__all__ = ["FixedKind", "make_content", "make_kind", "open_kind"]


class FixedKind:
    """A `KindSource` satisfied with one mapping. Satisfies the Protocol in
    this package's own tests rather than importing a neighbour."""

    def __init__(self, mapping: dict[HandoffId, HandoffKind] | None = None) -> None:
        self._m = mapping or {}

    def kind_for(self, hid: HandoffId) -> HandoffKind | None:
        return self._m.get(hid)


def make_kind(**over) -> HandoffKind:
    """A `reproducible` kind that passes every load-time check."""
    spec = {
        "name": "trace",
        "content_type": "reproducible",
        "items_schema": {
            "type": "object",
            "properties": {"script": {}, "result": {}, "env": {}},
            "required": ["env"],
            "additionalProperties": False,
        },
        "validators": ("check_trace_shape",),
        "scope": "fixed.required",
        "version": None,
    }
    spec.update(over)
    return HandoffKind(**spec)  # type: ignore[arg-type]


def open_kind() -> HandoffKind:
    """A kind whose `items_schema` permits runtime keys.

    Spec §3.1's second source of `items` keys: a kind may declare some and
    leave room for others, and `additionalProperties` is the whole mechanism.
    """
    return make_kind(
        items_schema={
            "type": "object",
            "properties": {"script": {}, "result": {}, "env": {}},
            "required": ["env"],
            "additionalProperties": True,
        }
    )


def make_content(
    root: Path,
    *,
    sections: tuple[str, ...] = ("Purpose", "How to run", "Result", "Environment", "Watch out"),
    files: dict[str, str] | None = None,
    data: dict[str, object] | None = None,
) -> Path:
    """A valid `content/` directory: a filled-in README plus items."""
    root = Path(root)
    (root / content_mod.ITEMS_DIR).mkdir(parents=True, exist_ok=True)
    body = ["# Trace", ""]
    for name in sections:
        body += [f"## {name}", "", f"Real prose for {name}.", ""]
    (root / "README.md").write_text("\n".join(body), encoding="utf-8")

    for key, text in (files or {"script": "echo hi\n", "result": "42\n"}).items():
        (root / content_mod.ITEMS_DIR / key).write_text(text, encoding="utf-8")
    content_mod.write_items_json(
        root / content_mod.ITEMS_DIR, data if data is not None else {"env": {"gpu": "MI300X"}}
    )
    return root


@pytest.fixture
def store(tmp_path: Path) -> FilesystemStore:
    """A **read-only** store: no `KindSource`, so `put` refuses.

    Named that way deliberately. Any test that needs to publish takes
    `kinded_store` instead, and a test that reaches for this one and then calls
    `put` gets a message telling it so — rather than a pass that came from the
    wrong refusal.
    """
    return FilesystemStore(tmp_path / "handoffs")


@pytest.fixture
def kinded_store(tmp_path: Path) -> tuple[FilesystemStore, HandoffId]:
    hid = HandoffId.new()
    source: KindSource = FixedKind({hid: make_kind()})
    return FilesystemStore(tmp_path / "handoffs", kinds=source), hid
