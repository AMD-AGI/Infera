# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
from env_mgr.recipe import Item, Target
from env_mgr.runner import Filters, detect_conflicts, run, select


def _items():
    return [
        Item(
            installer="oneline",
            importance="required",
            tags=["lsp"],
            spec={"name": "a", "check_cmd": "true", "run": "true"},
        ),
        Item(
            installer="oneline",
            importance="suggested",
            tags=["agent"],
            spec={"name": "b", "check_cmd": "false", "run": "true"},
        ),
    ]


def test_select_by_tag():
    got = select(_items(), Filters(tags=["lsp"]))
    assert [i.name for i in got] == ["a"]


def test_select_by_installer_and_importance():
    got = select(_items(), Filters(importance="required"))
    assert [i.name for i in got] == ["a"]


def test_detect_conflicts_flags_differing_versions():
    items = [
        Item(
            installer="uv",
            importance="required",
            version=">=0.5",
            spec={"name": "uv"},
        ),
        Item(
            installer="uv",
            importance="required",
            version=">=0.6",
            spec={"name": "uv"},
        ),
    ]
    outs = detect_conflicts(items)
    assert any(o.level == "fail" and "uv" in o.message for o in outs)


def test_conflict_detail_lists_every_conflicting_version():
    # The layer model is gone, and with it the key this detail used to be a
    # mapping on (`{item.layer: item.version}`). A mapping keyed on anything
    # still available would drop one of the two versions whenever the two items
    # agreed on that key, so the detail is a list and both constraints survive.
    # Asserted on the *value*, not on the type: the property that matters is
    # that a reader of the Outcome can see what conflicted with what.
    items = [
        Item(installer="uv", importance="required", version=">=0.5", spec={"name": "uv"}),
        Item(installer="uv", importance="required", version=">=0.6", spec={"name": "uv"}),
    ]
    (out,) = detect_conflicts(items)
    assert out.details["versions"] == [">=0.5", ">=0.6"]
    assert "layer" not in out.message


def test_detect_conflicts_ignores_installer_fallback_names():
    items = [
        Item(
            installer="oneline",
            importance="suggested",
            version=">=1.0",
            spec={"run": "true"},
        ),
        Item(
            installer="oneline",
            importance="suggested",
            version=">=2.0",
            spec={"run": "true"},
        ),
    ]
    assert detect_conflicts(items) == []


def test_detect_conflicts_none_when_same():
    items = [
        Item(
            installer="uv",
            importance="required",
            version=">=0.5",
            spec={"name": "uv"},
        ),
        Item(
            installer="uv",
            importance="required",
            version=">=0.5",
            spec={"name": "uv"},
        ),
    ]
    assert detect_conflicts(items) == []


def test_run_check_rolls_status(tmp_path):
    target = Target(kind="repo", name="x", path=str(tmp_path))
    outs, status = run(target, _items(), "check", Filters())
    assert status in ("OK", "WARN", "FAIL")
    # item b's check_cmd is false + suggested -> info, not fail
    assert status in ("OK", "WARN")


def test_run_dry_run_conflict_fails(tmp_path):
    target = Target(kind="repo", name="x", path=str(tmp_path))
    items = [
        Item(
            installer="uv",
            importance="required",
            version=">=0.5",
            spec={"name": "uv", "ref": "pyproject.toml"},
        ),
        Item(
            installer="uv",
            importance="required",
            version=">=0.6",
            spec={"name": "uv", "ref": "pyproject.toml"},
        ),
    ]
    outs, status = run(target, items, "dry-run", Filters(), on_conflict="fail")
    assert status == "FAIL"


def test_run_install_conflict_fail_does_not_mutate(tmp_path):
    # two same-named items with conflicting versions -> fatal conflict under fail.
    # install must NOT run the oneline 'run' (which would touch the marker).
    marker = tmp_path / "marker"
    items = [
        Item(
            installer="oneline",
            importance="required",
            version=">=1.0",
            spec={"name": "x", "run": f"touch {marker}"},
        ),
        Item(
            installer="oneline",
            importance="required",
            version=">=2.0",
            spec={"name": "x", "run": f"touch {marker}"},
        ),
    ]
    target = Target(kind="repo", name="t", path=str(tmp_path))
    outs, status = run(target, items, "install", Filters(), on_conflict="fail")
    assert status == "FAIL"
    assert not marker.exists()


def test_run_install_conflict_weak_proceeds(tmp_path):
    # under weak the conflict is a no-op: install proceeds and the marker appears.
    marker = tmp_path / "marker"
    items = [
        Item(
            installer="oneline",
            importance="required",
            version=">=1.0",
            spec={"name": "x", "run": f"touch {marker}"},
        ),
        Item(
            installer="oneline",
            importance="required",
            version=">=2.0",
            spec={"name": "x", "run": f"touch {marker}"},
        ),
    ]
    target = Target(kind="repo", name="t", path=str(tmp_path))
    outs, status = run(target, items, "install", Filters(), on_conflict="weak")
    assert marker.exists()
