# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import shutil
from importlib.resources import files

import pytest

from kernel_registry import Registry, file_record
from kernel_registry.cli import main
from kernel_registry.registry import metric_ratio


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "catalog"
    shutil.copytree(str(files("kernel_registry") / "templates"), root)
    return root


def edit(root, relative, change):
    path = root / relative
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


def test_template_is_valid_but_never_measured(catalog):
    report = Registry(catalog).validate()
    assert report["ok"] and report["warnings"]
    assert report["counts"] == {"kernel": 1, "environment": 1, "run": 2}
    rows = Registry(catalog).history()
    assert {r["run_type"] for r in rows} == {"microbench", "e2e"}
    assert all(r["correctness"]["status"] == "not_tested" and r["qualifiers"] for r in rows)


def test_payload_corruption_is_detected(catalog):
    (catalog / "payloads/candidate.py").write_text("corrupted\n")
    report = Registry(catalog).validate()
    assert not report["ok"]
    assert any("mismatch" in error for error in report["errors"])
    with pytest.raises(ValueError, match="invalid catalog"):
        Registry(catalog).history()


def test_run_snapshot_detects_changed_named_version(catalog):
    path = catalog / "payloads/candidate.py"
    path.write_text(path.read_text() + "# new version\n")
    descriptor = file_record(catalog, "payloads/candidate.py")
    edit(
        catalog,
        "kernels/example/kernel.json",
        lambda r: r["versions"][0].update(files=[descriptor]),
    )
    assert not Registry(catalog).validate()["ok"]


@pytest.mark.parametrize("target", ["../outside.txt", "/etc/passwd"])
def test_paths_cannot_escape(catalog, target):
    edit(catalog, "kernels/example/kernel.json", lambda r: r["original"][0].update(path=target))
    assert not Registry(catalog).validate()["ok"]


def test_symlink_escape_rejected(catalog):
    outside = catalog.parent / "outside.txt"
    outside.write_text("content")
    (catalog / "payloads/link.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        file_record(catalog, "payloads/link.txt")


def test_external_artifact_is_not_fetched(catalog):
    edit(
        catalog,
        "kernels/example/kernel.json",
        lambda r: r.update(
            fixtures=[{"uri": "s3://private/fixture", "sha256": "a" * 64, "size": 42}]
        ),
    )
    report = Registry(catalog).validate()
    assert report["ok"] and report["external_artifacts"] == 1


def test_unknown_version_is_error(catalog):
    edit(
        catalog, "runs/example-microbench.json", lambda r: r.update(kernels={"example": "missing"})
    )
    assert any("unknown version" in e for e in Registry(catalog).validate()["errors"])


def test_lineage_cycle_is_error(catalog):
    edit(catalog, "kernels/example/kernel.json", lambda r: r["versions"][0].update(parent="v1"))
    assert any("cycle" in e for e in Registry(catalog).validate()["errors"])


@pytest.mark.parametrize("text", ['{"kind":"run","name":"a","name":"b"}', '{"x":NaN}', "[]"])
def test_malformed_documents_rejected(catalog, text):
    (catalog / "runs/broken.json").write_text(text)
    assert not Registry(catalog).validate()["ok"]


def test_missing_file_descriptor_fields_rejected(catalog):
    edit(catalog, "kernels/example/kernel.json", lambda r: r.update(original=[{"path": "missing"}]))
    assert not Registry(catalog).validate()["ok"]


def test_unknown_environment_is_error(catalog):
    edit(catalog, "runs/example-e2e.json", lambda r: r.update(environment="unknown"))
    assert any("unknown environment" in e for e in Registry(catalog).validate()["errors"])


def metric(**fields):
    return {"name": "latency", "unit": "us", "direction": "lower", "aggregation": "ratio", **fields}


def test_ratios_preserve_direction_and_case_aggregation():
    assert metric_ratio(metric(baseline=12, candidate=4)) == 3
    assert metric_ratio(metric(direction="higher", baseline=4, candidate=12)) == 3
    cases = [{"baseline": 10, "candidate": 2}, {"baseline": 100, "candidate": 50}]
    assert metric_ratio(metric(aggregation="mean_case_ratio", cases=cases)) == 3.5
    assert metric_ratio(metric(aggregation="reported", reported_ratio=2)) == 2


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, None])
def test_invalid_timings_are_not_ratios(value):
    with pytest.raises(ValueError):
        metric_ratio(metric(baseline=value, candidate=1))


def test_failed_fast_run_keeps_evidence_and_qualifiers(catalog):
    def change(run):
        run["correctness"]["status"] = "failed"
        run["comparison"] = {"controlled": False, "confounders": ["different nodes"]}
        run["metrics"] = [metric(baseline=10, candidate=1)]

    edit(catalog, "runs/example-microbench.json", change)
    rows = Registry(catalog).history("example")
    row = next(r for r in rows if r["run_type"] == "microbench")
    assert row["metrics"][0]["ratio"] == 10
    assert "different nodes" in row["metrics"][0]["qualifiers"]
    assert "correctness is not passed for the stated scope" in row["qualifiers"]


def test_false_complete_claim_warns_instead_of_blocking_import(catalog):
    edit(
        catalog,
        "runs/example-e2e.json",
        lambda r: r.update(reproducibility={"status": "complete", "missing": []}),
    )
    report = Registry(catalog).validate()
    assert report["ok"]
    assert any("complete reproduction claim" in w for w in report["warnings"])
    assert Registry(catalog).history()[0]["missing"]


def test_unpaired_observation_does_not_invent_ratio(catalog):
    edit(
        catalog,
        "runs/example-microbench.json",
        lambda r: r.update(observed_metrics=[{"name": "tpot", "value": 11.2, "unit": "ms"}]),
    )
    row = next(r for r in Registry(catalog).history() if r["run_type"] == "microbench")
    assert row["observed_metrics"][0]["value"] == 11.2
    assert row["metrics"] == []


def test_cli_workflow_and_existing_destination(tmp_path, capsys):
    root = tmp_path / "new-catalog"
    assert main(["init", str(root)]) == 0
    assert main(["validate", str(root)]) == 0
    assert main(["history", str(root), "--kernel", "example"]) == 0
    assert main(["history", str(root), "--json"]) == 0
    assert main(["file", str(root), "payloads/candidate.py"]) == 0
    assert main(["init", str(root)]) == 1
    assert "already exists" in capsys.readouterr().err
