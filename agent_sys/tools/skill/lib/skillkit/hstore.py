"""Handoffs on disk — agent_sys's own `handoff.FilesystemStore`, with the
package's kinds plugged in.

    <store>/handoff.<kind-slug>.<uuid>/v<N>/content/{README.md, items/...}
                                           /claim/            (allocate's, unused here)
                                           /manifest.yaml     written by the store's seal = published
                                           /validation.yaml   created empty by the seal, verdicts appended

Allocation, the seal (README and items admission, digest, manifest) and the
verdict record are the store's and `handoff.verdict`'s; the executable-bit test
is `agent.gate.runnable`. What this module adds is only what a run outside the
runtime has no other source for: which kind an id holds (a `.kind` file beside
the version directories — the runtime keeps that in `task_graph`), and a
listing of the store by kind.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agent.gate import EXECUTABLE_ITEMS, runnable
from handoff import content as content_mod
from handoff import kind as kind_mod
from handoff import readme as readme_mod
from handoff import verdict as verdict_mod
from handoff.digest import tree_digest
from handoff.errors import Malformed
from handoff.protocols import Verdict
from handoff.store import FilesystemStore, handoff_dir, handoff_dirname
from task_graph.ids import HandoffId, TaskId

KIND_FILE = ".kind"


@dataclass
class Slot:
    kind: str
    hid: str
    version: int
    root: Path  # .../handoff.<slug>.<hid>

    @property
    def vdir(self) -> Path:
        return self.root / f"v{self.version}"

    @property
    def content(self) -> Path:
        return self.vdir / "content"

    @property
    def published(self) -> bool:
        return (self.vdir / "manifest.yaml").is_file()


class _Kinds:
    """`handoff.store.KindSource` over the package's kind specs: hid -> kind via
    the `.kind` file (or an id just registered), kind -> `HandoffKind` via
    `handoff.kind.from_spec`, the same constructor the runtime's registry uses."""

    def __init__(self, root: Path, specs: Mapping[str, Mapping]) -> None:
        self.root, self.specs = root, specs
        self.pending: dict[str, str] = {}

    def name_for(self, hid: object) -> str | None:
        if str(hid) in self.pending:
            return self.pending[str(hid)]
        side = handoff_dir(self.root, HandoffId(str(hid))) / KIND_FILE
        return side.read_text().strip() if side.is_file() else None

    def kind_for(self, hid: object):  # -> HandoffKind | None
        name = self.name_for(hid)
        if name is None or name not in self.specs:
            return None
        return kind_mod.from_spec(self.specs[name], origin=f"package kind {name}")


def store(root: Path, specs: Mapping[str, Mapping] | None = None) -> tuple[FilesystemStore, _Kinds]:
    kinds = _Kinds(Path(root), specs or {})
    return FilesystemStore(Path(root), kinds=kinds), kinds


# -- listing ---------------------------------------------------------------- #


def slots(root: Path, kind: str | None = None) -> list[Slot]:
    """Every version directory under `root`, oldest first (published or not)."""
    out: list[Slot] = []
    if not root.is_dir():
        return out
    for d in sorted(root.glob("handoff.*")):
        hid = d.name.rsplit(".", 1)[-1]
        label = (d / KIND_FILE).read_text().strip() if (d / KIND_FILE).is_file() else "?"
        if kind is not None and label != kind:
            continue
        versions = sorted(int(v.name[1:]) for v in d.glob("v*") if v.name[1:].isdigit())
        out += [Slot(kind=label, hid=hid, version=n, root=d) for n in versions]
    return out


def latest_published(root: Path, kind: str) -> Slot | None:
    """The newest published version of `kind` — the store's `latest`, per id."""
    fs, _ = store(root)
    best: Slot | None = None
    for d in sorted(root.glob("handoff.*")) if root.is_dir() else []:
        if not ((d / KIND_FILE).is_file() and (d / KIND_FILE).read_text().strip() == kind):
            continue
        hid = d.name.rsplit(".", 1)[-1]
        n = fs.latest(HandoffId(hid))
        if n is not None:
            cand = Slot(kind=kind, hid=hid, version=n, root=d)
            if (
                best is None
                or (cand.vdir / "manifest.yaml").stat().st_mtime
                >= (best.vdir / "manifest.yaml").stat().st_mtime
            ):
                best = cand
    return best


def find_digest(root: Path, kind: str, sha256: str) -> Slot | None:
    """A published version of `kind` whose manifest digest is `sha256`."""
    fs, _ = store(root)
    for s in slots(root, kind):
        if (
            s.published
            and fs.get_manifest(HandoffId(s.hid), s.version).digest.get("sha256") == sha256
        ):
            return s
    return None


# -- writing ---------------------------------------------------------------- #


def allocate(root: Path, kind: str, specs: Mapping[str, Mapping], hid: str | None = None) -> Slot:
    """The store's `allocate`: a fresh `v<N>/content/` for `kind` (a new version
    of the kind's existing handoff id, as a re-dispatch would get)."""
    if hid is None:
        existing = slots(root, kind)
        hid = existing[-1].hid if existing else str(uuid.uuid4())
    fs, kinds = store(root, specs)
    kinds.pending[hid] = kind
    n = fs.allocate(HandoffId(hid))
    d = handoff_dir(root, HandoffId(hid), kind)
    (d / KIND_FILE).write_text(kind + "\n")
    return Slot(kind=kind, hid=hid, version=n, root=d)


def import_content(root: Path, kind: str, specs: Mapping[str, Mapping], src: Path) -> Slot:
    """Copy an existing content dir into a freshly allocated version (modes and links kept)."""
    if not (src / "README.md").is_file():
        raise Malformed(f"{src} is not a handoff content dir (no README.md)")
    slot = allocate(root, kind, specs)
    from env_mgr.fs.layout import copy_out  # noqa: PLC0415 — the runtime's copy, links kept

    copy_out(str(src), str(slot.content))
    return slot


def seal(slot: Slot, specs: Mapping[str, Mapping], *, producer: str) -> str | None:
    """The store's seal: admission checks, digest, manifest, empty validation.yaml.
    Returns None when published, else the store's reason."""
    fs, _ = store(slot.root.parent, specs)
    return fs.seal(HandoffId(slot.hid), slot.version, producer=TaskId(task_id(producer)))


def record_verdicts(slot: Slot, rows: list[dict]) -> None:
    """`handoff.verdict.append`, one row each, onto a published version."""
    path = slot.vdir / verdict_mod.VERDICT_FILE
    for row in rows:
        verdict_mod.append(
            path,
            Verdict(
                validator=row["validator"],
                result=bool(row["result"]),
                strength=row["strength"],
                dimension=row["dimension"],
                task_id=TaskId(row["task_id"]),
                agent_id=None,
                environment=row["environment"],
                at=dt.datetime.fromisoformat(row["at"]),
            ),
        )


def verdicts(slot: Slot) -> list[dict]:
    path = slot.vdir / verdict_mod.VERDICT_FILE
    return [verdict_mod.to_row(v) for v in verdict_mod.read(path)] if path.is_file() else []


# -- checks ----------------------------------------------------------------- #


def precheck(content_dir: Path, content_type: str, items_schema: dict | None) -> list[str]:
    """The store's seal admission and the gate's executable test, as a list of
    problems ([] = it would publish), so they can be fixed before sealing — a
    sealed version is immutable."""
    problems: list[str] = []
    if not content_dir.is_dir() or not any(content_dir.iterdir()):
        return [f"{content_dir}: content is empty (the store refuses to seal it)"]
    try:
        ctype = content_mod.content_type(content_type)
    except Malformed as exc:
        return [str(exc)]
    try:
        readme_mod.check(content_dir, content_mod.required_sections(ctype))
    except Malformed as exc:
        problems.append(f"README: {exc}")
    try:
        items = content_mod.load(content_dir).items
        content_mod.check_items(content_mod.load(content_dir), ctype, items_schema or {})
    except Malformed as exc:
        problems.append(f"items: {exc}")
        items = {}
    for key in sorted(set(items) & EXECUTABLE_ITEMS):
        if not runnable(items[key]):
            problems.append(
                f"items/{key} is not an executable file (the gate refuses it): chmod +x, "
                "and not an items.json value"
            )
    return problems


def template(content_dir: Path, content_type: str, items_schema: dict | None = None) -> list[Path]:
    """Write a README skeleton (`handoff.readme.template` — every body is the
    placeholder the seal rejects) and the required items' placeholders.
    Never overwrites an existing file."""
    ctype = content_mod.content_type(content_type)
    created: list[Path] = []
    content_dir.mkdir(parents=True, exist_ok=True)
    readme = content_dir / "README.md"
    if not readme.exists():
        readme.write_text(readme_mod.template(content_mod.required_sections(ctype)))
        created.append(readme)
    items = content_dir / "items"
    items.mkdir(exist_ok=True)
    wanted = set(ctype.required_items) | set((items_schema or {}).get("required") or [])
    # One member of each one-of group, so the skeleton shows where it goes.
    wanted |= {"reproducible": {"command"}, "structured_text": {"text.json"}}.get(ctype.name, set())
    for name in sorted(wanted):
        p = items / name
        if p.exists():
            continue
        if name in ("codes", "env", "logs", "result"):
            p.mkdir()
        elif name in EXECUTABLE_ITEMS:
            p.write_text(
                "#!/bin/sh\n# Reproduce items/result from scratch: the exact commands, in order.\nset -eu\n"
            )
            p.chmod(0o755)
        else:
            p.write_text("")
        created.append(p)
    return created


def digest(content_dir: Path) -> str:
    return tree_digest(os.fsencode(content_dir)).hex()


def task_id(label: str) -> str:
    """A `TaskId` for `label`. agent_sys parses `producer` and a verdict's
    `task_id` as UUIDs, so the readable label goes into `state.json` and the
    verdict's `environment` instead. Deterministic: one attempt, one id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent_sys-skillkit:{label}"))


def describe(slot: Slot) -> dict:
    return {
        "kind": slot.kind,
        "hid": slot.hid,
        "version": slot.version,
        "content": str(slot.content),
        "published": slot.published,
        "verdicts": verdicts(slot),
    }


def dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, default=str)


def kind_label(handoff_root: Path) -> str | None:
    side = handoff_root / KIND_FILE
    return side.read_text().strip() if side.is_file() else None


__all__ = [
    "Slot",
    "allocate",
    "digest",
    "find_digest",
    "handoff_dirname",
    "import_content",
    "latest_published",
    "precheck",
    "record_verdicts",
    "seal",
    "slots",
    "store",
    "task_id",
    "template",
    "verdicts",
]
