"""Record persistence.

Full CRUD, keyed by (kind, key). The store never sees a model — managers pass
``model_dump(mode="json")`` and validate on the way back, which is what lets one
implementation serve every kind.

``sqlite3`` is the named upgrade path: it is stdlib and would supply the
cross-manager transaction this design leaves open. ``StoreMgr`` is a Protocol so
that swap is one file.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

__all__ = ["StoreMgr", "MemoryStoreMgr", "JsonFileStoreMgr"]


class StoreMgr(Protocol):
    def create(self, kind: str, key: str, record: dict) -> None:
        """Write a new record. Raises KeyError if one is already present."""

    def read(self, kind: str, key: str) -> dict | None: ...

    def read_all(self, kind: str) -> list[dict]: ...

    def update(self, kind: str, key: str, record: dict) -> None:
        """Replace an existing record. Raises KeyError if absent."""

    def delete(self, kind: str, key: str) -> None:
        """Remove a record. Raises KeyError if absent."""

    def exists(self, kind: str, key: str) -> bool: ...


class MemoryStoreMgr:
    """``dict[kind][key] -> deepcopy(record)``.

    Survives a manager restart because the store object does; that is exactly
    what makes recovery testable. The deep copy is not caution — without it a
    caller holds a live reference into the store and a reload returns objects
    nobody ever wrote.
    """

    def __init__(self) -> None:
        self._kinds: dict[str, dict[str, dict]] = {}

    def create(self, kind: str, key: str, record: dict) -> None:
        bucket = self._kinds.setdefault(kind, {})
        if key in bucket:
            raise KeyError(f"{kind}/{key} already exists")
        bucket[key] = deepcopy(record)

    def read(self, kind: str, key: str) -> dict | None:
        record = self._kinds.get(kind, {}).get(key)
        return deepcopy(record) if record is not None else None

    def read_all(self, kind: str) -> list[dict]:
        return [deepcopy(r) for r in self._kinds.get(kind, {}).values()]

    def update(self, kind: str, key: str, record: dict) -> None:
        bucket = self._kinds.setdefault(kind, {})
        if key not in bucket:
            raise KeyError(f"{kind}/{key} does not exist")
        bucket[key] = deepcopy(record)

    def delete(self, kind: str, key: str) -> None:
        bucket = self._kinds.get(kind, {})
        if key not in bucket:
            raise KeyError(f"{kind}/{key} does not exist")
        del bucket[key]

    def exists(self, kind: str, key: str) -> bool:
        return key in self._kinds.get(kind, {})


class JsonFileStoreMgr:
    """``<root>/<kind>/<quoted-key>.json``, one file per record.

    Records stay readable with ``cat`` while the schema is still moving, which
    is the whole reason for choosing files over sqlite today.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, kind: str, key: str) -> Path:
        # Keys are opaque strings that become filenames; callers should not
        # have to know that.
        return self.root / kind / f"{quote(key, safe='')}.json"

    def _write(self, path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
        tmp.replace(path)  # atomic on POSIX — per-record atomicity for free

    def create(self, kind: str, key: str, record: dict) -> None:
        path = self._path(kind, key)
        if path.exists():
            raise KeyError(f"{kind}/{key} already exists")
        self._write(path, record)

    def read(self, kind: str, key: str) -> dict | None:
        path = self._path(kind, key)
        return json.loads(path.read_text()) if path.exists() else None

    def read_all(self, kind: str) -> list[dict]:
        directory = self.root / kind
        if not directory.is_dir():
            return []
        return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]

    def update(self, kind: str, key: str, record: dict) -> None:
        path = self._path(kind, key)
        if not path.exists():
            raise KeyError(f"{kind}/{key} does not exist")
        self._write(path, record)

    def delete(self, kind: str, key: str) -> None:
        path = self._path(kind, key)
        if not path.exists():
            raise KeyError(f"{kind}/{key} does not exist")
        path.unlink()

    def exists(self, kind: str, key: str) -> bool:
        return self._path(kind, key).exists()
