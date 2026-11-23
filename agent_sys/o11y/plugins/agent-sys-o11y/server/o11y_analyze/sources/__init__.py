"""Source adapters. Each adapter reads one external format defensively and reports missing required
fields through `Problems` (-> o11y.source.schema_mismatch events), never by guessing."""

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
