# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Recipe parsing: YAML -> Target + [Item], with validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

IMPORTANCE = ("required", "strongly-suggested", "suggested")

# Keys the CLI understands directly; everything else goes into Item.spec.
_CLI_KEYS = {"installer", "importance", "tags", "version"}

#: Migration guard for the removed layer model: a stale author-written
#: `layer:` key is rejected rather than silently passed through as an
#: ordinary spec key. Can be dropped once none remain in circulation.
_LAYER_REMOVED = (
    "{where}: 'layer' was removed on 2026-09-04; the destination of an install "
    "is derived, not declared — see env_mgr/docs/spec.md §9.1"
)


class RecipeError(Exception):
    pass


@dataclass
class Target:
    kind: str
    name: str
    path: str
    parent: dict[str, str] = field(default_factory=dict)


@dataclass
class Item:
    installer: str
    importance: str
    tags: list[str] = field(default_factory=list)
    version: str | None = None
    spec: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        provides = self.spec.get("provides")
        if not isinstance(provides, str):
            provides = None
        return self.spec.get("name") or provides or self.installer


def _parse_item(raw: dict[str, Any], idx: int) -> Item:
    where = f"items[{idx}]"
    for key in ("installer", "importance"):
        if key not in raw:
            raise RecipeError(f"{where}: missing required field '{key}'")
    importance = raw["importance"]
    if importance not in IMPORTANCE:
        raise RecipeError(f"{where}: bad importance {importance!r} (expected {IMPORTANCE})")
    if "layer" in raw:
        raise RecipeError(_LAYER_REMOVED.format(where=where))
    spec = {k: v for k, v in raw.items() if k not in _CLI_KEYS}
    if raw["installer"] == "oneline":
        run = spec.get("run", "")
        if isinstance(run, str) and "\n" in run.rstrip("\n"):
            raise RecipeError(f"{where}: oneline.run must be exactly one line")
    return Item(
        installer=raw["installer"],
        importance=importance,
        tags=list(raw.get("tags", [])),
        version=raw.get("version"),
        spec=spec,
    )


def load_recipe(path: str | Path) -> tuple[Target, list[Item]]:
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise RecipeError(f"cannot load recipe {path}: {e}") from e
    if not isinstance(data, dict):
        raise RecipeError("recipe root must be a mapping")
    t = data.get("target")
    if not isinstance(t, dict) or "kind" not in t or "path" not in t:
        raise RecipeError("target must have at least 'kind' and 'path'")
    target = Target(
        kind=t["kind"],
        name=t.get("name", ""),
        path=t["path"],
        parent=dict(t.get("parent", {})),
    )
    raw_items = data.get("items", [])
    if not isinstance(raw_items, list):
        raise RecipeError("items must be a list")
    items = [_parse_item(r, i) for i, r in enumerate(raw_items)]
    return target, items
