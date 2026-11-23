"""o11y_sources — read-only source adapters shared by o11y_log and o11y_facts (design §8.0).

Each adapter reads one external format (Claude Code's registry and histories, the machine pool, KernelForge,
e2e-flow, the optional OTel copy) defensively, returns plain values, and reports missing or mistyped fields
through `Problems`, never by guessing. Nothing here writes protocol files; cursors are kept in a directory the
caller owns. Pure functions shared by both consumers (usage totals, pool attribution) live here so each rule
has one home. Standard library only in this module and in `otel`, so bin/o11y-launch can import them."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Problem:
    source: str  # claude_registry | transcript | pool | forge | e2e_flow | otel
    path: str
    field: str
    detail: str = ""

    @property
    def key(self) -> str:
        return f"{self.source}|{self.path}|{self.field}"


@dataclass
class Problems:
    items: list[Problem] = field(default_factory=list)

    def add(self, source: str, path: str, fld: str, detail: str = "") -> None:
        self.items.append(Problem(source, str(path), fld, detail))

    def require(
        self, source: str, path: str, obj: dict, fields: list[str], types: dict | None = None
    ) -> bool:
        """Check dotted required fields (present; the value may be null unless `types` says otherwise) and,
        for fields named in `types`, the value's type (a tuple of Python types; include type(None) to allow null)."""
        ok = True
        for f in fields:
            cur = obj
            for part in f.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    self.add(source, path, f, "missing required field")
                    ok = False
                    break
                cur = cur[part]
            else:
                want = (types or {}).get(f)
                if want is not None and (
                    not isinstance(cur, want) or (isinstance(cur, bool) and bool not in want)
                ):
                    self.add(source, path, f, f"wrong type {type(cur).__name__}")
                    ok = False
        return ok
