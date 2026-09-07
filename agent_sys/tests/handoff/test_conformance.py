"""`FilesystemStore` against the conformance suite.

One line of subclassing, and it is the point: when a second backend arrives it
adds the same line and finds its own leaks at implementation time.
"""

from __future__ import annotations

from pathlib import Path

from handoff import FilesystemStore
from tests.handoff.conformance import StoreConformance
from tests.handoff.conftest import FixedKind


class TestFilesystemStoreConformance(StoreConformance):
    def make_store(self, tmp_path: Path, kinds: FixedKind) -> FilesystemStore:
        return FilesystemStore(tmp_path / "store", kinds=kinds)
