"""Pinned source/runtime identity and external knowledge delivery, without a GPU."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

HELPER = (
    Path(__file__).resolve().parents[2]
    / "tools/skill/agent-sys-kernel-forge/scripts/check_forge.py"
)
spec = importlib.util.spec_from_file_location("forge_preflight", HELPER)
forge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forge)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    hyperloom, aklr = tmp_path / "hyperloom", tmp_path / "aklr"
    package = hyperloom / "src/kernelforge"
    (package / "kernel_backends").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (hyperloom / "pyproject.toml").write_text('[project]\nname = "hyperloom-inference_optimizer"\n')
    (package / "kernel_backends/constants.py").write_text(
        'KERNEL_BACKEND_PROMPT_MODULES = {"triton": "a", "gluon": "b", "hip": "c"}\n'
    )
    kb = aklr / "harness/skills/aklr-kb"
    (kb / "triton").mkdir(parents=True)
    (kb / "SKILL.md").write_text("External KB entry\n")
    (kb / "triton/SKILL.md").write_text("External language entry\n")
    state = {
        "module": package / "__init__.py",
        "commands": forge.COMMANDS,
        "loop_options": forge.LOOP_OPTIONS,
        "calls": [],
    }

    def run(argv):
        state["calls"].append(argv)
        if argv[0] == "git":
            if "rev-parse" in argv:
                return (
                    forge.HYPERLOOM_REVISION if Path(argv[2]) == hyperloom else forge.AKLR_REVISION
                )
            if "status" in argv:
                return ""
            if "ls-files" in argv:
                return "\0".join(
                    [
                        "pyproject.toml",
                        "src/kernelforge/__init__.py",
                        "src/kernelforge/kernel_backends/constants.py",
                    ]
                )
        if argv[1] == "-c":
            return json.dumps(
                {"module": str(state["module"]), "backends": ["triton", "gluon", "hip"]}
            )
        if argv[-2:] == ["forge-loop", "--help"]:
            return " ".join(state["loop_options"])
        if argv[-1] == "--help":
            return " ".join(state["commands"])
        pytest.fail(f"unexpected preflight command: {argv}")

    monkeypatch.setattr(forge, "_run", run)
    return hyperloom, aklr, kb, state


def test_source_only_has_no_runtime_dependency_and_routes_gluon_to_external_triton(sources):
    hyperloom, aklr, kb, state = sources
    report = forge.check(hyperloom, aklr, "gluon", source_only=True)
    assert report["skill_dirs"] == [str(kb), str(kb / "triton")]
    assert report["installed"] is None
    assert report["aklr_knowledge"]["revision"] == forge.AKLR_REVISION
    assert all(argv[0] == "git" for argv in state["calls"])
    with pytest.raises(forge.PreflightError, match="unsupported kernel backend"):
        forge.check(hyperloom, aklr, "triton-fellow", source_only=True)


def test_rejects_import_from_another_checkout_even_when_cli_looks_compatible(sources):
    hyperloom, aklr, _, state = sources
    state["module"] = hyperloom.parent / "old_forge/kernelforge/__init__.py"
    with pytest.raises(forge.PreflightError, match="outside"):
        forge.check(hyperloom, aklr, "triton")


def test_rejects_missing_aklr_branch_capabilities(sources):
    hyperloom, aklr, _, state = sources
    state["commands"] = ("forge-loop",)
    with pytest.raises(forge.PreflightError, match="missing commands"):
        forge.check(hyperloom, aklr, "triton")
    state["commands"] = forge.COMMANDS
    state["loop_options"] = ("--kernel-backend", "--resume")
    with pytest.raises(forge.PreflightError, match="missing options"):
        forge.check(hyperloom, aklr, "triton")


def test_scratch_editable_install_must_match_pinned_source_bytes(sources):
    hyperloom, aklr, _, state = sources
    installed = hyperloom.parent / "installed_copy"
    shutil.copytree(hyperloom, installed)
    state["module"] = installed / "src/kernelforge/__init__.py"
    assert forge.check(hyperloom, aklr, "triton", installed_source=installed)["installed"][
        "source_path"
    ] == str(installed)
    (installed / "src/kernelforge/__init__.py").write_text("# changed\n")
    with pytest.raises(forge.PreflightError, match="differs from pinned"):
        forge.check(hyperloom, aklr, "triton", installed_source=installed)


def test_rejects_unpinned_revision_before_loading_runtime(sources, monkeypatch):
    hyperloom, aklr, _, state = sources
    previous = forge._run

    def other_revision(argv):
        return "different-revision" if "rev-parse" in argv else previous(argv)

    monkeypatch.setattr(forge, "_run", other_revision)
    with pytest.raises(forge.PreflightError, match="expected revision"):
        forge.check(hyperloom, aklr, "triton")
    assert state["calls"] == []
