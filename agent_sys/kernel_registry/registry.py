# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Read ordinary catalog files, check their links, and summarize observations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs, parse_constant=invalid)


def _local(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"expected a relative catalog path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or ".." in Path(relative).parts:
        raise ValueError(f"path escapes catalog: {relative}")
    if not path.is_file():
        raise ValueError(f"file does not exist: {relative}")
    return path


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(root, relative_path):
    """Describe existing bytes; do not copy, execute or modify the file."""
    path = _local(Path(root).resolve(), relative_path)
    return {"path": relative_path, "sha256": _sha(path), "size": path.stat().st_size}


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON number")


def _positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def metric_ratio(metric):
    """Compute exactly the declared aggregation, with no cross-run ranking."""
    if metric.get("direction") not in ("lower", "higher"):
        raise ValueError("metric direction must be lower or higher")
    for key in ("name", "unit"):
        if not isinstance(metric.get(key), str) or not metric[key]:
            raise ValueError(f"metric needs {key}")

    def ratio(pair):
        baseline, candidate = pair.get("baseline"), pair.get("candidate")
        if not _positive(baseline) or not _positive(candidate):
            raise ValueError("metric needs positive finite baseline and candidate values")
        return baseline / candidate if metric["direction"] == "lower" else candidate / baseline

    aggregation = metric.get("aggregation")
    if aggregation == "ratio":
        return ratio(metric)
    if aggregation == "mean_case_ratio":
        cases = metric.get("cases")
        if not isinstance(cases, list) or not cases or not all(isinstance(c, dict) for c in cases):
            raise ValueError("mean_case_ratio needs nonempty case pairs")
        return sum(ratio(case) for case in cases) / len(cases)
    if aggregation == "reported" and _positive(metric.get("reported_ratio")):
        return metric["reported_ratio"]
    raise ValueError("metric needs ratio, mean_case_ratio, or reported aggregation and values")


def environment_gaps(environment):
    gaps = list(environment.get("missing", []))
    for group, keys in {
        "container": ("image", "digest"),
        "hardware": ("device", "architecture", "count", "driver"),
        "system": ("os", "kernel", "python"),
    }.items():
        values = environment.get(group, {})
        for key in keys:
            if not values.get(key):
                gaps.append(f"{group}.{key}")
    digest = environment.get("container", {}).get("digest")
    if digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        gaps.append("container.digest is not a full SHA256 identity")
    if not environment.get("packages"):
        gaps.append("package inventory")
    if not environment.get("repositories"):
        gaps.append("source repository revisions")
    for repo in environment.get("repositories", []):
        if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", repo.get("commit") or ""):
            gaps.append(f"full source commit: {repo.get('url', 'unknown repository')}")
        if repo.get("dirty") is None or (repo.get("dirty") and not repo.get("patch")):
            gaps.append(f"source dirty state/patch: {repo.get('url', 'unknown repository')}")
    return list(dict.fromkeys(gaps))


def _run_gaps(run, environments):
    gaps = list(run.get("reproducibility", {}).get("missing", []))
    for key in ("timestamp", "sources", "baseline", "commands", "configuration", "evidence"):
        if not run.get(key):
            gaps.append(key)
    workload = run.get("workload", {})
    if not workload:
        gaps.append("workload")
    if not workload.get("input_identity"):
        gaps.append("workload.input_identity (input digest or deterministic generation recipe)")
    if run.get("run_type") == "e2e":
        if not workload.get("model_identity"):
            gaps.append("workload.model_identity")
        if not run.get("integration"):
            gaps.append("integration snapshot")
    for key in ("environment", "baseline_environment"):
        name = run.get(key)
        if name in environments:
            gaps.extend(f"{key}: {gap}" for gap in environment_gaps(environments[name]))
        else:
            gaps.append(key)
    return list(dict.fromkeys(gaps))


def _file_lists(kind, record):
    keys = {
        "kernel": ("original", "reference", "documentation"),
        "environment": ("evidence",),
        "run": ("sources", "baseline", "integration", "evidence"),
    }[kind]
    lists = [record[key] for key in keys if key in record]
    if kind == "kernel":
        lists.extend(version.get("files", []) for version in record.get("versions", []))
        lists.append(record.get("integration", {}).get("files", []))
    for entries in lists:
        if not isinstance(entries, list):
            raise ValueError("file descriptor fields must be lists")
        for entry in entries:
            if not isinstance(entry, dict) or not {"sha256", "size"}.issubset(entry):
                raise ValueError("file descriptor needs path/uri, sha256 and size")
            if not ({"path", "uri"} & entry.keys()):
                raise ValueError("file descriptor needs path or uri")


class Registry:
    """A reader for a customer-owned directory, not another database."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    def _load(self):
        documents = {"kernel": {}, "environment": {}, "run": {}}
        errors = []
        for kind, pattern in (
            ("kernel", "kernels/*/kernel.json"),
            ("environment", "environments/*.json"),
            ("run", "runs/*.json"),
        ):
            for path in sorted(self.root.glob(pattern)):
                label = str(path.relative_to(self.root))
                try:
                    _local(self.root, label)
                    data = read_json(path)
                    if not isinstance(data, dict) or data.get("kind") != kind:
                        raise ValueError(f"expected kind={kind}")
                    name = data.get("name")
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("name must be a nonempty string")
                    if name in documents[kind]:
                        raise ValueError(f"duplicate {kind} name: {name}")
                    documents[kind][name] = data
                except (OSError, ValueError) as exc:
                    errors.append(f"{label}: {exc}")
        return documents, errors

    def validate(self):
        documents, errors = self._load()
        warnings, external = [], set()
        checked = {}
        if not self.root.is_dir():
            errors.append(f"catalog does not exist: {self.root}")
        elif not documents["kernel"]:
            warnings.append("catalog has no kernel descriptions")
        for kind, records in documents.items():
            for name, record in records.items():
                label = f"{kind} {name}"
                try:
                    _file_lists(kind, record)
                    for item in _walk(record):
                        if not ({"path", "uri"} & item.keys() and {"sha256", "size"} & item.keys()):
                            continue
                        if ("path" in item) == ("uri" in item):
                            raise ValueError("file descriptor needs exactly one of path or uri")
                        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
                            raise ValueError(f"file descriptor needs full SHA256: {item}")
                        size = item.get("size")
                        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                            raise ValueError("file descriptor needs nonnegative integer size")
                        if "uri" in item:
                            if not isinstance(item["uri"], str) or not urlsplit(item["uri"]).scheme:
                                raise ValueError("external file needs a URI scheme")
                            external.add(item["uri"])
                            continue
                        path = _local(self.root, item["path"])
                        if path not in checked:
                            checked[path] = (_sha(path), path.stat().st_size)
                        if checked[path] != (item["sha256"], size):
                            raise ValueError(f"file digest/size mismatch: {item['path']}")
                    self._check_record(kind, name, record, documents, warnings)
                except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
                    errors.append(f"{label}: {exc}")
        if external:
            warnings.append(
                f"{len(external)} external artifacts: identity recorded; availability not checked"
            )
        return {
            "ok": not errors,
            "counts": {kind: len(records) for kind, records in documents.items()},
            "files_checked": len(checked),
            "external_artifacts": len(external),
            "errors": errors,
            "warnings": warnings,
        }

    def _check_record(self, kind, name, record, documents, warnings):
        def warn(message):
            warnings.append(f"{kind} {name}: {message}")

        if not record.get("description"):
            warn("missing description")
        if kind == "environment":
            gaps = environment_gaps(record)
            if gaps:
                warn("missing reproduction facts: " + "; ".join(gaps))
            return
        if kind == "kernel":
            for key in ("original", "reference", "documentation", "contract", "versions"):
                if not record.get(key):
                    warn(f"missing {key}")
            if record.get("reference_status") != "complete":
                warn("naive reference completeness is not established")
            versions = record.get("versions", [])
            by_id = {}
            for version in versions:
                vid = version.get("id")
                if not isinstance(vid, str) or not vid or vid in by_id:
                    raise ValueError(f"missing or duplicate version id: {vid}")
                if not version.get("files"):
                    raise ValueError(f"version {vid} has no sources")
                by_id[vid] = version
            for vid, version in by_id.items():
                seen = {vid}
                parent = version.get("parent")
                while parent:
                    if parent not in by_id:
                        raise ValueError(f"version {vid} has unknown parent {parent}")
                    if parent in seen:
                        raise ValueError(f"version lineage cycle at {parent}")
                    seen.add(parent)
                    parent = by_id[parent].get("parent")
            integration = record.get("integration", {})
            for key in ("method", "files", "activation", "rollback", "compatibility"):
                if not integration.get(key):
                    warn(f"missing integration.{key}")
            if integration.get("version") and integration["version"] not in by_id:
                raise ValueError(f"integration names unknown version {integration['version']}")
            return
        if record.get("run_type") not in ("microbench", "e2e"):
            raise ValueError("run_type must be microbench or e2e")
        kernels = record.get("kernels")
        if not isinstance(kernels, dict) or not kernels:
            raise ValueError("run must name at least one kernel version")
        expected = set()
        for kernel_name, version_id in kernels.items():
            if kernel_name not in documents["kernel"]:
                raise ValueError(f"unknown kernel: {kernel_name}")
            versions = documents["kernel"][kernel_name].get("versions", [])
            selected = next((v for v in versions if v.get("id") == version_id), None)
            if selected is None:
                raise ValueError(f"unknown version: {kernel_name}@{version_id}")
            expected.update((f.get("path", f.get("uri")), f["sha256"]) for f in selected["files"])
        if record.get("sources"):
            actual = {(f.get("path", f.get("uri")), f["sha256"]) for f in record["sources"]}
            if not expected.issubset(actual):
                raise ValueError("run source snapshot does not match the named kernel versions")
        for key in ("environment", "baseline_environment"):
            if record.get(key) and record[key] not in documents["environment"]:
                raise ValueError(f"unknown {key}: {record[key]}")
        correctness = record.get("correctness", {})
        if correctness.get("status", "not_tested") not in ("passed", "failed", "not_tested"):
            raise ValueError("correctness.status must be passed, failed or not_tested")
        if not correctness.get("scope") or not correctness.get("summary"):
            warn("missing correctness scope/summary")
        for metric in record.get("metrics", []):
            metric_ratio(metric)
        comparison = record.get("comparison", {})
        if comparison.get("controlled") and comparison.get("confounders"):
            warn("comparison has confounders despite controlled=true")
        gaps = _run_gaps(record, documents["environment"])
        if gaps:
            warn("missing reproduction facts: " + "; ".join(gaps))
            if record.get("reproducibility", {}).get("status") == "complete":
                warn("complete reproduction claim is not supported by recorded facts")

    def history(self, kernel=None):
        report = self.validate()
        if report["errors"]:
            raise ValueError("invalid catalog: " + "; ".join(report["errors"]))
        documents, _ = self._load()
        if kernel is not None and kernel not in documents["kernel"]:
            raise ValueError(f"unknown kernel: {kernel}")
        result = []
        for run in sorted(
            documents["run"].values(), key=lambda r: (r.get("timestamp") or "", r["name"])
        ):
            if kernel is not None and kernel not in run["kernels"]:
                continue
            correctness = run.get("correctness", {})
            qualifiers = list(run.get("limitations", []))
            if correctness.get("status") != "passed":
                qualifiers.append("correctness is not passed for the stated scope")
            comparison = run.get("comparison", {})
            if not comparison.get("controlled") or comparison.get("confounders"):
                qualifiers.append("comparison is not controlled")
            qualifiers.extend(comparison.get("confounders", []))
            gaps = _run_gaps(run, documents["environment"])
            if gaps or run.get("reproducibility", {}).get("status") != "complete":
                qualifiers.append("reproduction information is incomplete")
            if run["run_type"] == "e2e" and run.get("activation") != "confirmed":
                qualifiers.append("integration activation is not confirmed")
            metrics = []
            for metric in run.get("metrics", []):
                notes = list(qualifiers)
                if metric["aggregation"] == "reported":
                    notes.append("reported ratio; raw paired timings unavailable")
                metrics.append({**metric, "ratio": metric_ratio(metric), "qualifiers": notes})
            result.append(
                {
                    "name": run["name"],
                    "timestamp": run.get("timestamp"),
                    "run_type": run["run_type"],
                    "kernels": run["kernels"],
                    "environment": run.get("environment"),
                    "baseline_environment": run.get("baseline_environment"),
                    "correctness": correctness,
                    "metrics": metrics,
                    "observed_metrics": run.get("observed_metrics", []),
                    "qualifiers": qualifiers,
                    "missing": gaps,
                }
            )
        return result
