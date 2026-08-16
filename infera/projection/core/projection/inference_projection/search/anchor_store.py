###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Anchor store — a directory of benchmark artifacts indexed by regime.

The store is the backbone of the sub-scale search: a handful of measured
anchors (the JSON artifacts ``benchmark_vllm.py`` emits) indexed by their
:func:`regime signature <regime.regime_signature>`, with nearest-anchor lookup
in *regime space*.  Reconstruction (see :mod:`reconstruct`) then transports the
nearest in-regime anchor across the continuous axes to any target recipe.

Design choices (v1, deliberately simple and greppable):

* The index is a plain ``index.json`` manifest — no database.  Each entry
  records the artifact path, its regime signature + axes, and the *transport
  coverage* it measured (parallelism, layer counts, batches) so a lookup can
  tell whether a target is interpolated or extrapolated.
* Lookup distance is Hamming over the regime axes; ties are broken by
  closeness on the transport axes (prefer the anchor whose measured
  parallelism / depth is nearest the target, i.e. the least restore).
* The store is typically **per model** — you harvest anchors for the model you
  are tuning — so the ``model`` axis is only used to *filter* when both sides
  name a model, never to inflate distance otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from . import regime


def _axes_version() -> str:
    """Short fingerprint of the regime axis set an index was written under."""
    return hashlib.sha256(",".join(regime.REGIME_AXES).encode()).hexdigest()[:12]


class AnchorStore:
    """A directory of benchmark artifacts with nearest-in-regime lookup."""

    #: How deep below the root to look for artifacts nobody indexed. Benchmark
    #: harnesses write into a per-run subdirectory, so the artifacts sit one or
    #: two levels down; beyond that a store root has been pointed somewhere too
    #: broad and walking it is a cost, not a feature.
    DISCOVERY_DEPTH = 3

    def __init__(self, root: str, discover: bool = True):
        self.root = os.path.abspath(root)
        self.index_path = os.path.join(self.root, "index.json")
        self._entries: List[Dict[str, Any]] = []
        self._load_index()
        if discover:
            self.discover()

    # -- persistence -----------------------------------------------------------

    def discover(self) -> int:
        """Index any artifact under the root that nothing has indexed yet.

        Without this the store only ever sees what some caller remembered to
        hand it, which makes a directory full of perfectly good measurements
        look empty and silently downgrades every projection to uncalibrated.
        That failure is invisible at the call site -- you get an answer, just a
        worse one -- so the store reads its own directory instead of trusting
        someone else to have kept the manifest current.

        Returns the number of newly indexed artifacts.
        """
        known = {e.get("path") for e in self._entries}
        found = 0
        for path in self._walk_artifacts():
            if path in known:
                continue
            try:
                with open(path) as f:
                    art = json.load(f)
            except (OSError, ValueError):
                continue
            if not self._looks_like_artifact(art):
                continue
            self._entries.append(self._make_entry(path, art))
            found += 1
        if found:
            self._save_index()
        return found

    def _walk_artifacts(self):
        """Absolute paths of candidate JSON files within the depth limit."""
        root_depth = self.root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(self.root):
            if dirpath.count(os.sep) - root_depth >= self.DISCOVERY_DEPTH:
                dirnames[:] = []
            for name in filenames:
                if name == "index.json" or not name.endswith(".json"):
                    continue
                yield os.path.abspath(os.path.join(dirpath, name))

    @staticmethod
    def _looks_like_artifact(art: Any) -> bool:
        """Whether a parsed JSON file is a benchmark artifact we can anchor on.

        Deliberately strict. A store root will contain reports, configs and
        other JSON, and indexing one of those produces an anchor with no curve
        that then loses a lookup in a way nobody can trace back to here.
        """
        return (
            isinstance(art, dict)
            and isinstance(art.get("meta"), dict)
            and isinstance(art.get("sweep"), list)
            and any(
                isinstance(e, dict) and e.get("decode_ms") for e in art["sweep"]
            )
        )

    def _load_index(self) -> None:
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path) as f:
                doc = json.load(f)
        except (OSError, ValueError):
            self._entries = []
            return
        self._entries = doc.get("anchors", [])
        # Signatures and per-entry regime dicts are only comparable within one
        # definition of the regime axes. When an axis is added, an index written
        # by the older definition describes each anchor on fewer axes than a
        # fresh recipe does, so it is re-derived from the artifacts rather than
        # silently compared against the new axis set.
        if doc.get("axes_version") != _axes_version():
            self._rebuild_entries()

    def _rebuild_entries(self) -> None:
        """Re-derive every entry from its artifact under the current axis set."""
        rebuilt = []
        for entry in self._entries:
            path = entry.get("path")
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    art = json.load(f)
            except (OSError, ValueError):
                continue
            # A signature stamped by an older axis set is stale by definition.
            art.get("meta", {}).pop("regime_signature", None)
            rebuilt.append(self._make_entry(os.path.abspath(path), art))
        self._entries = rebuilt
        self._save_index()

    def _save_index(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(self.index_path, "w") as f:
            json.dump(
                {"axes_version": _axes_version(), "anchors": self._entries},
                f,
                indent=2,
            )

    # -- ingestion -------------------------------------------------------------

    def add_artifact(self, artifact_path: str) -> Dict[str, Any]:
        """Index a benchmark artifact JSON already on disk.  Idempotent by path
        (re-adding refreshes the entry)."""
        path = os.path.abspath(artifact_path)
        with open(path) as f:
            art = json.load(f)
        entry = self._make_entry(path, art)
        self._entries = [e for e in self._entries if e.get("path") != path]
        self._entries.append(entry)
        self._save_index()
        return entry

    def add_result(self, result: Dict[str, Any], artifact_path: str) -> Dict[str, Any]:
        """Write a benchmark result dict to ``artifact_path`` and index it."""
        path = os.path.abspath(artifact_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f)
        return self.add_artifact(path)

    def _make_entry(self, path: str, art: Dict[str, Any]) -> Dict[str, Any]:
        meta = art.get("meta", {})
        recipe = regime.recipe_from_meta(meta)
        sig = meta.get("regime_signature") or regime.regime_signature(recipe)
        sweep = art.get("sweep") or []
        batches = sorted({int(e["batch"]) for e in sweep if e.get("batch") is not None})
        return {
            "path": path,
            "regime_signature": sig,
            "regime": {k: recipe.get(k) for k in regime.REGIME_AXES},
            "model": meta.get("model"),
            "transport": {
                "tp": meta.get("benchmark_tp") or meta.get("tp"),
                "ep": meta.get("benchmark_ep") or meta.get("ep"),
                "pp": meta.get("benchmark_pp") or meta.get("pp"),
                "target_tp": meta.get("tp"),
                "target_ep": meta.get("ep"),
                "target_pp": meta.get("pp"),
                "num_layers": meta.get("num_hidden_layers"),
                "full_layers": (meta.get("restore") or {}).get("full_layers")
                if meta.get("restore")
                else None,
                "batches": batches,
                "input_len": meta.get("input_len"),
            },
        }

    # -- query -----------------------------------------------------------------

    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def load_artifact(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        with open(entry["path"]) as f:
            return json.load(f)

    def nearest(
        self, recipe: Dict[str, Any], *, model: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """Return ``(entry, regime_distance)`` for the best anchor, or
        ``(None, None)`` if the store is empty.  Distance 0 => same regime
        (fully transportable).  Candidates are optionally filtered to a model;
        ties on regime distance are broken by transport closeness (least
        restore/extrapolation)."""
        cands = self._entries
        if model:
            named = [e for e in cands if e.get("model") in (None, model)]
            cands = named or cands
        if not cands:
            return None, None

        def transport_gap(e: Dict[str, Any]) -> float:
            t = e.get("transport", {})
            gap = 0.0
            # Prefer anchors whose measured parallelism matches the target
            # (smaller restore) and whose depth is closer (less extrapolation).
            for axis in ("tp", "ep", "pp"):
                tv, rv = t.get(axis), recipe.get(axis)
                if tv and rv:
                    gap += abs(float(tv) - float(rv))
            nl, rnl = t.get("num_layers"), recipe.get("num_layers")
            if nl and rnl:
                gap += abs(float(nl) - float(rnl)) / max(1.0, float(rnl))
            return gap

        scored = [
            (regime.regime_distance(recipe, {**e["regime"]}), transport_gap(e), e)
            for e in cands
        ]
        scored.sort(key=lambda s: (s[0], s[1]))
        best = scored[0]
        return best[2], best[0]
