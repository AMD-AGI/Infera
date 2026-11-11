"""Persistent CPU regression contracts for the live Forge integration.

These protect result verdicts, backend selection, resume arguments, external KB
delivery, portable measurement drivers and collection of reviewed source bytes.
They exercise the current runner and generated artifacts; they do not migrate
repositories or replay the historical campaign.
"""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

AGENT_SYS = Path(__file__).resolve().parents[2]
FLOW = AGENT_SYS / "examples/llm_e2e_performance_optimization/e2e-flow"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize("improved,rounds", [(True, 3), (False, 4), (False, 0)])
def test_native_final_verdict_and_round_count(tmp_path, improved, rounds):
    adapter = _module(
        "forge_result_adapter", FLOW / "assets/optimize_kernel.task/steps/normalize_forge_result.py"
    )
    native = {
        "total_improved": improved,
        "iteration_count": rounds,
        "next_iteration": 99,
        "mean_case_speedup": 1.2 if improved else 1.0,
    }
    (tmp_path / "forge_native_result.json").write_text(json.dumps(native))
    result = adapter.normalize(tmp_path, tmp_path / "workspace", "op")
    assert result["improved"] is improved
    assert result["iteration_count"] == rounds
    assert result["forge_native"]["--result-json"] == native
    assert result["snr_db"] is None


@pytest.mark.parametrize("native", [{"error": "preparation failed"}, {"mean_case_speedup": 1.3}])
def test_error_or_missing_verdict_is_not_no_improvement(tmp_path, native):
    adapter = _module(
        "forge_result_adapter", FLOW / "assets/optimize_kernel.task/steps/normalize_forge_result.py"
    )
    (tmp_path / "forge_native_result.json").write_text(json.dumps(native))
    with pytest.raises(ValueError):
        adapter.normalize(tmp_path, tmp_path, "op")


def test_stdout_fallback_preserves_legitimate_no_improvement(tmp_path):
    adapter = _module(
        "forge_result_adapter", FLOW / "assets/optimize_kernel.task/steps/normalize_forge_result.py"
    )
    native = {"total_improved": False, "next_iteration": 5, "mean_case_speedup": 1.0}
    (tmp_path / "forge_stdout.log").write_text(
        "log\n__FORGE_RESULT__" + json.dumps(native) + "__FORGE_RESULT__\n"
    )
    result = adapter.normalize(tmp_path, tmp_path, "op")
    assert result["improved"] is False
    assert result["iteration_count"] == 4


def test_generated_command_repeats_external_kb_on_resume(tmp_path):
    exporter = _module("forge_export_adapter", FLOW / "assets/lib/forge_export.py")
    fixture = AGENT_SYS / "examples/03_kernelforge_one_kernel/assets/fixture/workset/items/codes"
    workset = tmp_path / "workset with spaces"
    shutil.copytree(fixture, workset)
    exporter.export_all(workset)
    operator = yaml.safe_load((workset / "workset.yaml").read_text())["operators"][0]
    workspace = tmp_path / "workspace with spaces"
    target = workspace / exporter._kernel_rel(operator)
    target.parent.mkdir(parents=True)
    target.write_text("# Candidate path only; no GPU work in this test.\n")
    kb = tmp_path / "external AKLR" / "harness/skills/aklr-kb"
    (kb / "triton").mkdir(parents=True)
    (kb / "SKILL.md").write_text("External test reference\n")
    (kb / "triton/SKILL.md").write_text("Language test reference\n")
    probe = tmp_path / "argv-probe"
    probe.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    probe.chmod(0o755)
    env = dict(os.environ, AKLR_REPO=str(kb.parents[2]), FORGE_PYTHON=str(probe))
    command = ["sh", str(workset / operator["forge"]["one_line"]), "--workspace", str(workspace)]
    fresh = json.loads(subprocess.check_output(command, env=env, text=True))
    resumed = json.loads(subprocess.check_output(command + ["--resume"], env=env, text=True))
    assert fresh[:3] == ["-m", "kernelforge.cli", "forge-loop"]
    assert fresh[fresh.index("--kernel-backend") + 1] == "triton"
    assert fresh[fresh.index("--workspace") + 1] == str(workspace)
    assert "--driver" not in resumed and "--kernel" not in resumed
    assert "--kernel-backend" not in resumed
    for argv in (fresh, resumed):
        dirs = [argv[i + 1] for i, value in enumerate(argv) if value == "--skill-dir"]
        assert dirs == [str(kb), str(kb / "triton")]


def test_unknown_backend_is_refused_instead_of_falling_back(monkeypatch):
    exporter = _module("forge_export_adapter", FLOW / "assets/lib/forge_export.py")
    monkeypatch.delenv("E2E_FORGE_BACKEND", raising=False)
    monkeypatch.delenv("E2E_FORGE_FELLOW", raising=False)
    assert exporter._fellow({}, {"tags": ["intellikit"]}) == ""
    assert exporter._fellow({}, {"tags": ["gluon"]}) == "gluon"


def test_staged_driver_measures_its_own_detached_workspace(tmp_path, monkeypatch):
    exporter = _module("forge_export_adapter", FLOW / "assets/lib/forge_export.py")
    driver_path = tmp_path / "driver.py"
    driver_path.write_text(exporter._driver_source("op", ["one"], "kernel.py"))
    driver = _module("portable_forge_driver", driver_path)
    monkeypatch.setenv("E2E_FORGE_IMPL", "/outside/yihou/wrong-tree/kernel.py")
    monkeypatch.setattr(sys, "argv", [str(driver_path), "--bench-mode"])
    measured = []

    def instrument(script, case_id, impl, *extra):
        measured.append(impl)
        return 0, {"operators": [{"shapes": [{"case_id": "one", "weighted_mean_ms": 1.0}]}]}

    monkeypatch.setattr(driver, "_run", instrument)
    for tree in ("best", "ablation"):
        monkeypatch.setattr(driver, "ROOT", tmp_path / tree / ".infera_forge_workset")
        assert driver.main() == 0
    assert measured == [str(tmp_path / tree / "kernel.py") for tree in ("best", "ablation")]


def test_generated_driver_matches_formatted_fixture():
    exporter = _module("forge_export_adapter", FLOW / "assets/lib/forge_export.py")
    fixture = AGENT_SYS / "examples/03_kernelforge_one_kernel/assets/fixture/workset/items/codes"
    operator = yaml.safe_load((fixture / "workset.yaml").read_text())["operators"][0]
    case_ids = [shape["case_id"] for shape in operator["shapes"]]
    generated = exporter._driver_source(
        operator["operator_id"], case_ids, exporter._kernel_rel(operator)
    )
    assert generated == (fixture / operator["forge"]["driver"]).read_text()


def test_generated_driver_preserves_quoted_and_unicode_parameters():
    exporter = _module("forge_export_adapter", FLOW / "assets/lib/forge_export.py")
    operator_id = 'op"""\\\n\x00'
    case_ids = ['quote"', "apostrophe'", "slash\\newline\n", "unicode-\U0001f600"]
    kernel_rel = 'path/quote"and\\slash.py'
    tree = ast.parse(exporter._driver_source(operator_id, case_ids, kernel_rel))
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"OPERATOR", "CASE_IDS", "KERNEL_REL"}
    }
    assert values == {"OPERATOR": operator_id, "CASE_IDS": case_ids, "KERNEL_REL": kernel_rel}


@pytest.mark.parametrize(
    "flags",
    [
        ["--kernel-backend", "intellikit"],
        ["--kernel-backend=intellikit"],
        ["--resume", "--kernel", "other.py"],
        ["--resume", "--driver=other.py"],
    ],
)
def test_generated_wrapper_refuses_backend_substitution_and_fresh_flags_on_resume(tmp_path, flags):
    exporter = _module("forge_export_adapter", FLOW / "assets/lib/forge_export.py")
    fixture = AGENT_SYS / "examples/03_kernelforge_one_kernel/assets/fixture/workset/items/codes"
    workset = tmp_path / "workset"
    shutil.copytree(fixture, workset)
    exporter.export_all(workset)
    operator = yaml.safe_load((workset / "workset.yaml").read_text())["operators"][0]
    result = subprocess.run(
        ["sh", str(workset / operator["forge"]["one_line"]), "--dry-run", *flags],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "conflicts" in result.stderr or "resume keeps" in result.stderr


def test_collection_refreshes_reviewed_bytes_and_rejects_harness_edits(tmp_path):
    collector = _module(
        "forge_output_collector", FLOW / "assets/optimize_kernel.task/steps/collect_forge_output.py"
    )
    workdir = tmp_path / "forge"
    workspace = workdir / "engine_src"
    protected = workspace / ".infera_forge_workset/driver.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("# Protected instrument\n")
    source = workspace / "kernel.py"
    source.write_text("value = 0\n")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(workspace), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("add", ".")
    git("commit", "-qm", "baseline")
    (workdir / "engine_baseline.sha").write_text(git("rev-parse", "HEAD"))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps({"edit_target": {"source_file": "kernel.py", "repo_root_var": "SGLANG_ROOT"}})
    )
    collector.collect(inputs, workdir)
    assert (workdir / "engine.patch").read_text() == ""
    source.write_text("value = 1\n")
    git("commit", "-qam", "kept candidate")
    collector.collect(inputs, workdir)
    assert (workdir / "optimized_kernel.py").read_text() == "value = 1\n"
    source.write_text("value = 2\n")
    git("commit", "-qam", "reviewed candidate")
    collector.collect(inputs, workdir)
    assert (workdir / "optimized_kernel.py").read_text() == "value = 2\n"
    assert "+value = 2" in (workdir / "engine.patch").read_text()
    protected.write_text("# Changed instrument\n")
    with pytest.raises(ValueError, match="protected measurement harness"):
        collector.collect(inputs, workdir)
