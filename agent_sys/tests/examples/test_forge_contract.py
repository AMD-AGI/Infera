"""Persistent CPU regression contracts for the live Forge integration.

These protect result verdicts, backend selection, resume arguments, external KB
delivery, portable measurement drivers and collection of reviewed source bytes.
They exercise the current runner and generated artifacts; they do not migrate
repositories or replay the historical campaign.
"""

import ast
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jsonschema import validate

from cli.package import task_package

AGENT_SYS = Path(__file__).resolve().parents[2]
FLOW = AGENT_SYS / "examples/llm_e2e_performance_optimization/e2e-flow"


@pytest.mark.parametrize("provider,rounds", [(None, None), ("codex", "2")])
def test_real_launch_passes_explicit_provider_and_round_limit(tmp_path, provider, rounds):
    launcher = tmp_path / "capture.sh"
    launcher.write_text("printf '%s\\0' \"$@\"\n")
    env = dict(os.environ)
    env.pop("KFO_AGENT_BACKEND", None)
    env.pop("KFO_MAX_ROUNDS", None)
    env.update(
        WS=str(tmp_path / "engine workspace"),
        WORKDIR=str(tmp_path / "campaign work"),
        RUN=str(tmp_path),
        ONELINE=launcher.name,
        OPERATOR="sample",
        KFO_MAX_HOURS="1",
        KFO_FORGE_MODEL="model with spaces",
    )
    if provider is not None:
        env["KFO_AGENT_BACKEND"] = provider
    if rounds is not None:
        env["KFO_MAX_ROUNDS"] = rounds
    helper = FLOW / "assets/optimize_kernel.task/steps/_shlib.sh"
    output = subprocess.check_output(
        ["sh", "-c", '. "$1"; kfo_run_forge', "sh", str(helper)], env=env
    )
    argv = [part.decode() for part in output.rstrip(b"\0").split(b"\0")]
    assert argv[argv.index("--agent-backend") + 1] == (provider or "claude")
    assert argv[argv.index("--max-rounds") + 1] == (rounds or "0")
    assert argv[argv.index("--model") + 1] == "model with spaces"
    assert argv[argv.index("--workspace") + 1] == env["WS"]


@pytest.mark.parametrize(
    "overrides,provider,rounds,container",
    [
        ({}, "claude", "0", ""),
        (
            {
                "forge_agent_backend": "codex",
                "forge_max_rounds": "2",
                "measure_container": "yihou-measure",
            },
            "codex",
            "2",
            "yihou-measure",
        ),
    ],
)
def test_package_forwards_provider_round_and_measurement_container(
    overrides, provider, rounds, container
):
    variables = dict(
        container="yihou-test",
        EXPORT_DIR="/data/yihou/export",
        jobid="local",
        node="localhost",
        node_ip="127.0.0.1",
        model_name="test",
        model_path="/data/yihou/model",
        image="test:latest",
        work_root="/data/yihou/work",
        scratch_root="/data/yihou/scratch",
        **overrides,
    )
    contents = task_package(FLOW, **variables).documents()
    assert not contents.problems
    agent = next(
        doc.doc
        for doc in contents.documents
        if doc.kind == "agent" and doc.doc["name"] == "e2e_kernel_optimizer"
    )
    assert agent["env"]["KFO_AGENT_BACKEND"] == provider
    assert agent["env"]["KFO_MAX_ROUNDS"] == rounds
    assert agent["env"]["KFO_MEASURE_CONTAINER"] == container


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize("package", [FLOW, AGENT_SYS / "examples/03_kernelforge_one_kernel"])
def test_staged_package_embeds_skill_and_runs_mock_adapter(tmp_path, package):
    from env_mgr.fs.layout import stage_package

    staged = Path(stage_package(str(package), SimpleNamespace(root=str(tmp_path / "zone"))))
    steps = staged / "assets/optimize_kernel.task/steps"
    embedded_skill = staged / "assets/kernel_forge_skill"
    for name in ("SKILL.md", "references/workset.md", "references/campaign.md"):
        assert (embedded_skill / name).is_file()
        assert (embedded_skill / name).resolve().is_relative_to(staged)
    for name in (
        "30_run_forge.sh",
        "_shlib.sh",
        "normalize_forge_result.py",
        "prepare_forge_workspace.py",
        "collect_forge_output.py",
    ):
        assert (steps / name).resolve().is_relative_to(staged)
    assert (staged / "assets/lib/workset_program.md").resolve().is_relative_to(staged)
    exporter = _module("staged_forge_export", staged / "assets/lib/forge_export.py")
    fixture = AGENT_SYS / "examples/03_kernelforge_one_kernel/assets/fixture/workset/items/codes"
    workset = tmp_path / "workset"
    shutil.copytree(fixture, workset)
    exporter.export_all(workset)
    operator = yaml.safe_load((workset / "workset.yaml").read_text())["operators"][0]
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "workset_root": str(workset),
                "operator_id": operator["operator_id"],
                "forge_one_line": operator["forge"]["one_line"],
                "definition": operator["definition"],
            }
        )
    )
    workdir = tmp_path / "forge"
    command = ["sh", str(steps / "30_run_forge.sh")]
    if package != FLOW:
        command = [
            "sh",
            str(embedded_skill / "scripts/workset/30_run_forge.sh"),
            "--adapter-steps",
            str(steps),
        ]
    subprocess.run(
        [*command, "--inputs", str(inputs), "--workdir", str(workdir)],
        env=dict(
            os.environ,
            KFO_MOCK="1",
            KFO_PYTHON=sys.executable,
            KFO_SCRATCH_ROOT=str(tmp_path / "scratch"),
            TMPDIR=str(tmp_path / "tmp"),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads((workdir / "forge_result.json").read_text())
    assert result["mock"] is True and result["ran"] is False
    definition = json.loads((workset / operator["definition"]).read_text())
    assert definition["baseline"] in (workdir / "optimized_kernel.py").read_text()


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


@pytest.mark.parametrize("stdout_result", [False, True])
def test_native_summary_passes_locality_without_rewriting_raw_records(tmp_path, stdout_result):
    from handoff import locality
    from handoff.errors import Malformed

    adapter = _module(
        "forge_result_adapter", FLOW / "assets/optimize_kernel.task/steps/normalize_forge_result.py"
    )
    native = {
        "total_improved": True,
        "iteration_count": 2,
        "mean_case_speedup": 1.25,
        "best_commit": "a" * 40,
        "termination_reason": "round_cap_reached",
        "best_manifest": "/var/tmp/yihou/forge-run/best/manifest.json",
        "optimization_report": "/var/tmp/yihou/forge-run/optimization_report.md",
        "optimization_history": "/var/tmp/yihou/forge-run/optimization_history.md",
        "agent_relay": {"profiles": [{"log": r"C:\Users\worker\forge\agent.log"}]},
        "future_field": {"diagnostic": "see /home/worker/campaign/agent.log"},
    }
    name = "forge_stdout.log" if stdout_result else "forge_native_result.json"
    raw = tmp_path / name
    text = json.dumps(native)
    if stdout_result:
        text = "__FORGE_RESULT__" + text + "__FORGE_RESULT__\n"
    raw.write_text(text)
    best = tmp_path / "engine/forge_experiments/best_result.json"
    best.parent.mkdir(parents=True)
    best.write_text(
        json.dumps(
            {
                "snr_db": 37.5,
                "commit_hash": "a" * 40,
                "validation_path": "/var/tmp/yihou/forge-run/validation.txt",
            }
        )
    )
    before = {path: path.read_bytes() for path in (raw, best)}
    original = tmp_path / "raw-handoff"
    original.mkdir()
    (original / "native.json").write_text(json.dumps(native))
    with pytest.raises(Malformed, match="local path"):
        locality.check(original, oracles=locality.Oracles())

    record = adapter.normalize(tmp_path, tmp_path / "engine", "sample")
    exported = tmp_path / "handoff/results"
    exported.mkdir(parents=True)
    (exported / "forge_result.json").write_text(json.dumps(record, indent=2))
    locality.check(exported.parent, oracles=locality.Oracles(playground_root=tmp_path))
    origin = "__FORGE_RESULT__" if stdout_result else "--result-json"
    assert record["forge_native"][origin]["best_commit"] == "a" * 40
    assert record["forge_native"][origin]["termination_reason"] == "round_cap_reached"
    assert (record["mean_case_speedup"], record["snr_db"], record["iteration_count"]) == (
        1.25,
        37.5,
        2,
    )
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("field", ["mean_case_speedup", "snr_db"])
def test_native_measurement_cannot_smuggle_a_host_path(tmp_path, field):
    adapter = _module(
        "forge_result_adapter", FLOW / "assets/optimize_kernel.task/steps/normalize_forge_result.py"
    )
    (tmp_path / "forge_native_result.json").write_text(
        json.dumps({"total_improved": True, field: "/var/tmp/yihou/forge-run/value"})
    )
    with pytest.raises(ValueError, match="must be numeric"):
        adapter.normalize(tmp_path, tmp_path, "sample")


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
    program = Path(fresh[fresh.index("--program-md-file") + 1])
    assert fresh.count("--program-md-file") == 1
    assert program == workset / operator["forge"]["program"] and program.is_file()
    assert len(program.read_text()) < 3000
    assert "--driver" not in resumed and "--kernel" not in resumed
    assert "--kernel-backend" not in resumed
    assert "--program-md-file" not in resumed
    for argv in (fresh, resumed):
        dirs = [argv[i + 1] for i, value in enumerate(argv) if value == "--skill-dir"]
        assert dirs == [str(kb), str(kb / "triton")]


@pytest.mark.parametrize("inline_option", [False, True])
def test_explicit_caller_program_is_preserved_without_duplicate_default(tmp_path, inline_option):
    exporter = _module("forge_export_adapter", FLOW / "assets/lib/forge_export.py")
    fixture = AGENT_SYS / "examples/03_kernelforge_one_kernel/assets/fixture/workset/items/codes"
    workset = tmp_path / "workset"
    shutil.copytree(fixture, workset)
    exporter.export_all(workset)
    operator = yaml.safe_load((workset / "workset.yaml").read_text())["operators"][0]
    supplied = tmp_path / "caller brief.md"
    supplied.write_text("Caller-owned workset and external KB requirements.\n")
    flag = (
        [f"--program-md-file={supplied}"] if inline_option else ["--program-md-file", str(supplied)]
    )
    result = subprocess.check_output(
        ["sh", str(workset / operator["forge"]["one_line"]), "--dry-run", *flag], text=True
    )
    argv = shlex.split(result)
    program_options = [
        value
        for value in argv
        if value == "--program-md-file" or value.startswith("--program-md-file=")
    ]
    assert len(program_options) == 1
    if inline_option:
        assert f"--program-md-file={supplied}" in argv
    else:
        assert str(supplied) in argv
    assert str(workset / operator["forge"]["program"]) not in argv


def test_generated_program_survives_native_campaign_snapshot_and_resume(tmp_path):
    native = pytest.importorskip("kernelforge.cli.forge_loop_command")
    campaign = pytest.importorskip("kernelforge.loop.campaign_config")
    exporter = _module("forge_export_adapter", FLOW / "assets/lib/forge_export.py")
    fixture = AGENT_SYS / "examples/03_kernelforge_one_kernel/assets/fixture/workset/items/codes"
    workset = tmp_path / "workset"
    shutil.copytree(fixture, workset)
    exporter.export_all(workset)
    operator = yaml.safe_load((workset / "workset.yaml").read_text())["operators"][0]
    program = workset / operator["forge"]["program"]
    context = program.read_text()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def run():\n    return None\n")
    (workspace / "driver.py").write_text("# Protected CPU fixture; never executed.\n")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(workspace), *args], text=True).strip()

    git("init", "-q")
    (workspace / ".git/info/exclude").write_text("/forge_experiments/\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "baseline",
    )
    git("checkout", "-qb", "test-forge")
    inputs = campaign.CampaignInputs(
        kernel="kernel.py",
        driver="driver.py",
        kernel_backend="triton",
        gpu_target="gfx942",
        program_md_file=str(program),
    )
    config = campaign.create_campaign_config(str(workspace), inputs)
    store = campaign.CampaignConfigStore(str(workspace))
    fresh = SimpleNamespace(save_deferred=True, program_text=context, campaign=config)
    assert (
        native._program_context(
            fresh, store, resume=False, experience_kb=False, workspace_dir=str(workspace)
        )
        == context
    )
    store.save(config, program_md=context)
    assert config.program_md_sha256 == hashlib.sha256(context.encode()).hexdigest()
    assert (workspace / "forge_experiments/program.md").read_text() == context
    program.write_text("Modified source brief after campaign start.\n")
    resumed = SimpleNamespace(save_deferred=False, campaign=config)
    assert (
        native._program_context(
            resumed, store, resume=True, experience_kb=False, workspace_dir=str(workspace)
        )
        == context
    )
    assert git("status", "--porcelain") == ""


def test_unknown_backend_is_refused_instead_of_falling_back(monkeypatch):
    exporter = _module("forge_export_adapter", FLOW / "assets/lib/forge_export.py")
    monkeypatch.delenv("E2E_FORGE_BACKEND", raising=False)
    monkeypatch.delenv("E2E_FORGE_FELLOW", raising=False)
    assert exporter._kernel_backend({}, {"tags": ["intellikit"]}) == ""
    assert exporter._kernel_backend({}, {"tags": ["gluon"]}) == "gluon"
    assert exporter._kernel_backend({}, {"tags": ["triton-fellow"]}) == "triton"


def test_export_writes_kernel_backend_and_schema_reads_historical_alias(tmp_path):
    exporter = _module("forge_export_adapter", FLOW / "assets/lib/forge_export.py")
    fixture = AGENT_SYS / "examples/03_kernelforge_one_kernel/assets/fixture/workset/items/codes"
    workset = tmp_path / "workset"
    shutil.copytree(fixture, workset)
    exporter.export_all(workset)
    operator = yaml.safe_load((workset / "workset.yaml").read_text())["operators"][0]
    forge = operator["forge"]
    assert forge["kernel_backend"] == "triton" and "fellow" not in forge
    task = yaml.safe_load((workset / forge["task"]).read_text())
    invocation = json.loads((workset / forge["invocation_spec"]).read_text())
    assert task["kernel_backend"] == invocation["implementation"]["kernel_backend"] == "triton"
    assert "fellow" not in task and "fellow" not in invocation["implementation"]
    schema = json.loads((FLOW / "assets/schemas/workset.schema.json").read_text())
    forge_schema = schema["$defs"]["operator"]["properties"]["forge"]
    validate(forge, forge_schema)
    historical = dict(forge)
    historical["fellow"] = historical.pop("kernel_backend") + "-fellow"
    validate(historical, forge_schema)


def test_taxonomy_emits_kernel_backend_for_current_and_legacy_rules():
    taxonomy = _module("forge_taxonomy", FLOW / "assets/lib/taxonomy.py")
    current = {
        "kernel_backends": [
            {"kernel_backend": "triton", "language": "triton", "patterns": ["sample"]}
        ]
    }
    legacy = {
        "fellows": [{"fellow": "triton-fellow", "language": "triton", "patterns": ["sample"]}]
    }
    for rules in (current, legacy):
        assert taxonomy.kernel_backend_of("sample", rules) == {
            "kernel_backend": "triton",
            "language": "triton",
        }


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
        operator["operator_id"], case_ids, exporter._kernel_rel(operator), example=True
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
        ["--resume", "--program-md-file", "other.md"],
        ["--resume", "--program-md-file=other.md"],
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


@pytest.mark.parametrize("canonical", [False, True])
def test_collection_refreshes_reviewed_bytes_and_rejects_harness_edits(tmp_path, canonical):
    adapter_steps = FLOW / "assets/optimize_kernel.task/steps"
    entry = adapter_steps / "collect_forge_output.py"
    if canonical:
        entry = (
            AGENT_SYS / "tools/skill/agent-sys-kernel-forge/scripts/workset/collect_forge_output.py"
        )
    collector = _module("forge_output_collector", entry)

    def collect():
        collector.collect(inputs, workdir, adapter_steps if canonical else None)

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
    collect()
    assert (workdir / "engine.patch").read_text() == ""
    source.write_text("value = 1\n")
    git("commit", "-qam", "kept candidate")
    collect()
    assert (workdir / "optimized_kernel.py").read_text() == "value = 1\n"
    source.write_text("value = 2\n")
    git("commit", "-qam", "reviewed candidate")
    collect()
    assert (workdir / "optimized_kernel.py").read_text() == "value = 2\n"
    assert "+value = 2" in (workdir / "engine.patch").read_text()
    protected.write_text("# Changed instrument\n")
    with pytest.raises(ValueError, match="protected measurement harness"):
        collect()


@pytest.mark.parametrize("operator_id", ["sample", "Sample-v1.2"])
def test_workspace_setup_tracks_spec_and_excludes_only_root_runtime_state(tmp_path, operator_id):
    setup = _module(
        "forge_workspace_setup",
        FLOW / "assets/optimize_kernel.task/steps/prepare_forge_workspace.py",
    )
    workspace = tmp_path / "engine workspace"
    operator = workspace / ".infera_forge_workset/operators" / operator_id
    driver = operator / "scripts/forge_driver.py"
    driver.parent.mkdir(parents=True)
    driver.write_text("# Protected instrument\n")
    source = operator / "invocation_spec.json"
    payload = {"z": "Unicode \U0001f600", "a": {"cases": ["one"]}}
    original = json.dumps(payload) + "\n"
    source.write_text(original)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(workspace), *args], text=True).strip()

    git("init", "-q")
    setup.prepare(workspace, operator_id)
    adjacent = driver.with_name("invocation_spec.json")
    canonical = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert adjacent.read_text() == canonical
    assert source.read_text() == original
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "baseline",
    )
    assert git("ls-files", "--error-unmatch", str(adjacent.relative_to(workspace)))
    exclude = workspace / ".git/info/exclude"
    exclude_before = exclude.read_bytes()
    setup.prepare(workspace, operator_id)
    assert exclude.read_bytes() == exclude_before and adjacent.read_text() == canonical
    runtime = workspace / "forge_experiments"
    runtime.mkdir()
    (runtime / "llm_usage.json").write_text("{}\n")
    (workspace / ".stop").write_text("")
    assert git("status", "--porcelain") == ""
    source_file = workspace / "new_kernel.py"
    source_file.write_text("# Unapproved new kernel source\n")
    nested = workspace / "other/forge_experiments/output.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}\n")
    for path in (source_file, nested, driver):
        result = subprocess.run(
            ["git", "-C", str(workspace), "check-ignore", "--quiet", str(path)], check=False
        )
        assert result.returncode == 1
    driver.write_text("# Changed instrument\n")
    assert "forge_driver.py" in git("diff", "--name-only")
    adjacent.write_text('{"different": true}\n')
    with pytest.raises(ValueError, match="differs from the authoritative workset"):
        setup.prepare(workspace, operator_id)
    assert source.read_text() == original


@pytest.mark.parametrize("operator_id", ["", ".", "..", "../outside", "a/b", "a\\b", "/outside"])
def test_workspace_setup_rejects_paths_instead_of_operator_components(tmp_path, operator_id):
    setup = _module(
        "forge_workspace_setup",
        FLOW / "assets/optimize_kernel.task/steps/prepare_forge_workspace.py",
    )
    with pytest.raises(ValueError, match="safe path component"):
        setup.prepare(tmp_path, operator_id)
