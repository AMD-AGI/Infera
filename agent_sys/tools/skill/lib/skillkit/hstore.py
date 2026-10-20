"""Handoffs on disk, in agent_sys's store layout, checked by agent_sys's code.

A store here is the same tree `handoff/store.py` writes, so a reader (or
`replay`-style tooling) cannot tell the two apart:

    <store>/handoff.<kind-slug>.<uuid>/v<N>/content/{README.md, items/...}
                                           /validation.yaml   verdict rows
                                           /manifest.yaml     written LAST = published

`precheck` is exactly the store's seal test (README sections, items) plus the
gate's executable-bit test, both through agent_sys's own functions. Nothing
the runtime does not enforce is enforced here.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from handoff import content as content_mod
from handoff import readme as readme_mod
from handoff.digest import tree_digest
from handoff.errors import Malformed
from handoff.store import handoff_dirname

#: `agent/gate.py`: items with these names must carry the executable bit.
EXECUTABLE_ITEMS = ("script", "command", "entry")
ALGORITHM = "agent_sys.handoff.tree.v1"


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


# -- the store -------------------------------------------------------------- #


def slots(store: Path, kind: str | None = None) -> list[Slot]:
    """Every version directory under `store`, oldest first."""
    out = []
    if not store.is_dir():
        return out
    for d in sorted(store.glob("handoff.*")):
        hid = d.name.rsplit(".", 1)[-1]
        label = kind_of(d)
        if kind is not None and label != kind:
            continue
        for v in sorted(
            d.glob("v*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1
        ):
            if v.name[1:].isdigit():
                out.append(Slot(kind=label or "?", hid=hid, version=int(v.name[1:]), root=d))
    return out


def kind_of(handoff_root: Path) -> str | None:
    """The kind, from the newest manifest, else from the `.kind` sidecar."""
    for m in sorted(handoff_root.glob("v*/manifest.yaml"), reverse=True):
        return str(yaml.safe_load(m.read_text()).get("kind"))
    side = handoff_root / ".kind"
    return side.read_text().strip() if side.is_file() else None


def latest_published(store: Path, kind: str) -> Slot | None:
    pub = [s for s in slots(store, kind) if s.published]
    return pub[-1] if pub else None


def find_digest(store: Path, kind: str, sha256: str) -> Slot | None:
    """A published version of `kind` whose manifest digest is `sha256`."""
    for s in slots(store, kind):
        m = s.vdir / "manifest.yaml"
        if (
            m.is_file()
            and (yaml.safe_load(m.read_text()) or {}).get("digest", {}).get("sha256") == sha256
        ):
            return s
    return None


def allocate(store: Path, kind: str, hid: str | None = None) -> Slot:
    """A fresh `v<N>/content/` for `kind` (a new handoff id unless `hid`)."""
    if hid is None:
        existing = [s for s in slots(store, kind)]
        hid = existing[-1].hid if existing else str(uuid.uuid4())
    root = store / handoff_dirname(hid, kind)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".kind").write_text(kind + "\n")
    n = 0
    while (root / f"v{n}").exists():
        n += 1
    slot = Slot(kind=kind, hid=hid, version=n, root=root)
    slot.content.mkdir(parents=True)
    return slot


def import_content(store: Path, kind: str, src: Path) -> Slot:
    """Copy an existing content dir into a fresh slot, modes and links kept."""
    if not (src / "README.md").is_file():
        raise Malformed(f"{src} is not a handoff content dir (no README.md)")
    slot = allocate(store, kind)
    slot.content.rmdir()
    shutil.copytree(src, slot.content, symlinks=True)
    return slot


# -- checks ----------------------------------------------------------------- #


def precheck(content_dir: Path, content_type: str, items_schema: dict | None) -> list[str]:
    """The store's seal test and the gate's +x test. Returns problems ([] = sealable)."""
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
        content_mod.check_items(content_mod.load(content_dir), ctype, items_schema or {})
    except Malformed as exc:
        problems.append(f"items: {exc}")
    try:
        items = content_mod.load(content_dir).items
    except Malformed:
        items = {}
    for name in EXECUTABLE_ITEMS:  # `agent/gate.py` `_executable`, the same test
        item = items.get(name)
        if item is None:
            continue
        if item.path is None:
            problems.append(
                f"items/{name} is a value in items.json; the gate needs an executable file"
            )
        elif not os.access(item.path, os.X_OK):
            problems.append(
                f"items/{name} is not executable (the gate refuses it): chmod +x {item.path}"
            )
    return problems


def template(content_dir: Path, content_type: str, items_schema: dict | None = None) -> list[Path]:
    """Write a README skeleton and the required items' placeholders.

    The README is `handoff.readme.template` — every body is the placeholder the
    seal rejects, so an unfilled skeleton cannot pass by accident. Returns the
    paths created. Never overwrites an existing file.
    """
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


# -- verdicts and the manifest --------------------------------------------- #


def record_verdicts(slot: Slot, rows: list[dict]) -> None:
    """Append rows to `validation.yaml` (the store's file; outside the digest)."""
    path = slot.vdir / "validation.yaml"
    data = yaml.safe_load(path.read_text()) if path.is_file() else None
    data = data or {"verdicts": []}
    data["verdicts"] = list(data.get("verdicts") or []) + rows
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def verdicts(slot: Slot) -> list[dict]:
    path = slot.vdir / "validation.yaml"
    if not path.is_file():
        return []
    return list((yaml.safe_load(path.read_text()) or {}).get("verdicts") or [])


def task_id(label: str) -> str:
    """A `TaskId` for `label`. agent_sys parses `producer` and a verdict's
    `task_id` as UUIDs (`task_graph.ids`), so a readable label would make the
    store unreadable to it; the label goes into `state.json` and the verdict's
    `environment` instead. Deterministic, so one attempt keeps one id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent_sys-skillkit:{label}"))


def seal(slot: Slot, *, producer: str) -> dict:
    """Write `manifest.yaml` — the publish step. Caller has checked first.

    `producer` is a label; the manifest carries its `task_id`."""
    manifest = {
        "digest": {"sha256": digest(slot.content)},
        "algorithm": ALGORITHM,
        "kind": slot.kind,
        "producer": task_id(producer),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    tmp = slot.vdir / ".manifest.yaml.pending"
    tmp.write_text(yaml.safe_dump(manifest, sort_keys=False))
    tmp.rename(slot.vdir / "manifest.yaml")
    return manifest


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
