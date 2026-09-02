"""YAML text to a tree that still knows where every value came from.

This is `render.py`'s successor and it is a much smaller thing: no runtime seam,
no fallback binding, no thread pool. What it keeps is the one property the old
module had for a different reason — **a fault names a place**.

## Why `ruamel.yaml`, and why round-trip mode specifically

Main spec §7 adopts it, and the reason is that PyYAML's `safe_load` throws the
positions away. Four facts, all measured here rather than taken from §7's web
research, because §7 was written before anything in this tree parsed a
hand-written document — `scratch/ui-yaml-2026-08/w3/probe_ruamel_positions.py`
and `probe_ruamel_semantics.py`:

| | |
|---|---|
| **Positions survive nesting** | `lc.item(i)` on a sequence at the root, `lc.key(k)` / `lc.value(k)` on a mapping three levels down. Both 0-based, so everything here adds one. The nested case is the new one: an inline definition has no file of its own and a diagnostic that could only name the file would name the wrong thing |
| **Syntax errors carry `problem_mark`** | line and column, 0-based, for all three malformed documents probed |
| **Duplicate keys are rejected**, with a position | The trap §7 flagged when jsonnet's static rejection went away is closed by the library. `{name: a, name: b}` raises `DuplicateKeyError`; PyYAML `safe_load` silently keeps the last |
| **The tree is `dict` and `list`** | `CommentedMap` and `CommentedSeq` subclass them, so `jsonschema` validates the position-carrying tree directly and `err.json_path` is correct. Nothing is converted, and therefore nothing is lost on the way to `validate` |

The fourth is what makes this a thin wrapper rather than a layer.

## YAML 1.2, and the trap that is now closed rather than avoided

`spec_loader/validate.py` used to read specs with PyYAML and argue that *"neither
the YAML 1.1 `norway: NO` trap nor the duplicate-key trap can reach us — jsonnet
quotes every string and rejects a duplicate field statically"*. Main spec §7
rev. 10 records that the argument's premise is gone and hands the question here.

Measured, both parsers over the same eight scalars:

    norway: NO        ruamel -> 'NO'     PyYAML -> False
    affirmative: yes  ruamel -> 'yes'    PyYAML -> True
    switch: on        ruamel -> 'on'     PyYAML -> True
    sexagesimal: 12:30  ruamel -> '12:30'  PyYAML -> 750
    thousand: 1e3     ruamel -> 1000.0   PyYAML -> '1e3'

`ruamel.yaml`'s round-trip loader is **YAML 1.2**, so the trap does not need
avoiding — the values a package author writes mean what they look like. The two
parsers disagree on real documents, which is why exactly one of them may touch a
package document: see `validate`, which no longer parses at all.

## What is deliberately not here

**Nothing writes YAML.** The round-trip *dumper* is the other half of what this
library is famous for and no part of this system emits a spec, so it is not
imported. If something ever does, this is the module it belongs in.

**An unknown tag is preserved, not refused.** Round-trip mode does not resolve
`!!python/object/apply:os.system [...]` — nothing is constructed and nothing
runs, measured — but it does not reject it either; the value arrives as an
ordinary `CommentedSeq` carrying a tag the schema cannot see. `typ='safe'`
refuses it and has no `lc`. Recorded in `README.md` as open rather than guarded,
because the exposure is a hypothetical downstream unsafe round trip and this
package performs none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import MarkedYAMLError

from .protocols import Problem

__all__ = ["Position", "position_of", "read_yaml"]


@dataclass(frozen=True)
class Position:
    """A 1-based place in a file. `ruamel.yaml` reports 0-based; this is the
    only module that knows that, and it adds one exactly once."""

    line: int
    column: int

    @classmethod
    def from_ruamel(cls, pair: tuple[int, int] | None) -> Position | None:
        """`None` in, `None` out — a position is never guessed."""
        if pair is None:
            return None
        return cls(line=pair[0] + 1, column=pair[1] + 1)


def read_yaml(path: Path, *, origin: str) -> tuple[Any, list[Problem]]:
    """Parse one file. Returns `(tree, [])` or `(None, [one Problem])`.

    A fault is returned rather than raised, for `load_package`'s reason: one
    unreadable file must not hide the other nine. There is at most one problem
    because the parser stops at the first fault it meets — this reports what
    happened, and does not invent a recovery the library does not offer.

    `origin` is the label; the path is opened but never printed from here, so a
    caller that labels a document differently from its file gets its own label
    in the message.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [Problem(origin=origin, path="$", keyword="unreadable", message=str(exc))]
    except UnicodeDecodeError as exc:
        # Not an `OSError`. A binary file with a `.yaml` name is a fault in the
        # package and belongs in the report beside the others, not as a
        # traceback out of a scan.
        return None, [Problem(origin=origin, path="$", keyword="unreadable", message=str(exc))]

    try:
        tree = YAML().load(text)
    except MarkedYAMLError as exc:
        # `DuplicateKeyError` is one of these, and it is the one worth naming:
        # it is the trap that jsonnet's static rejection used to close.
        at = Position.from_ruamel(
            (exc.problem_mark.line, exc.problem_mark.column) if exc.problem_mark else None
        )
        return None, [
            Problem(
                origin=origin,
                path="$",
                keyword="parse",
                message=_syntax_message(exc),
                line=at.line if at else None,
                column=at.column if at else None,
            )
        ]
    except Exception as exc:  # noqa: BLE001
        # The library raises `YAMLError` subclasses that are not `Marked`, and a
        # `ValueError` from a scalar constructor. Breadth here matches
        # `load_package`'s admit loop and for the same reason: an unparseable
        # file is a fault in the *package*, and letting one shape of it escape
        # would abort the whole multi-package load.
        return None, [
            Problem(
                origin=origin,
                path="$",
                keyword="parse",
                message=f"{type(exc).__name__}: {exc}",
            )
        ]

    return tree, []


def position_of(node: Any, key: str | int | None = None) -> Position | None:
    """Where `node` — or the value it holds under `key` — was written.

    `None` whenever the tree does not carry it, which happens for a plain `dict`
    a test built by hand and for the root of a document with no keys. The caller
    puts `None` in the diagnostic rather than a guess; that is the discipline the
    deleted `RenderError` documented, and the reason it is worth keeping is that
    `google/jsonnet#786` — a location a runtime *would not* report, guessed at by
    the layer above — is what it was written for.
    """
    lc = getattr(node, "lc", None)
    if lc is None:
        return None
    if key is None:
        line, col = getattr(lc, "line", None), getattr(lc, "col", None)
        return Position.from_ruamel((line, col)) if line is not None else None
    if isinstance(node, Mapping):
        return Position.from_ruamel(lc.key(key)) if key in node else None
    if isinstance(node, Sequence) and isinstance(key, int):
        return Position.from_ruamel(lc.item(key))
    return None


def _syntax_message(exc: MarkedYAMLError) -> str:
    """The library's own words, on one line.

    `str(exc)` is four lines with a caret diagram, and `report.py` prints one
    problem per line. The parts are kept in the order the library composes them
    so that a reader who searches for the text finds the library's own
    documentation of it.
    """
    parts = [p for p in (exc.context, exc.problem) if p]
    return ", ".join(parts) or str(exc).splitlines()[0]
