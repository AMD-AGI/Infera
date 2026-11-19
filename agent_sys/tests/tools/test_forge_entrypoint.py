"""Compatibility launchers recheck explicit editable-source identity before exec."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[2] / "tools/skill/agent-sys-kernel-forge"
CHECK = SKILL / "scripts/check_forge.py"
ENTRYPOINT = SKILL / "scripts/run_entrypoint.sh"


@pytest.fixture
def runtime(tmp_path):
    """Use real Git, subprocess probes and preflight; only campaign exec is inert."""
    spec = importlib.util.spec_from_file_location("entrypoint_preflight", CHECK)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source, aklr, installed = (tmp_path / name for name in ("source", "aklr", "installed"))
    package = source / "src/kernelforge"
    (package / "kernel_backends").mkdir(parents=True)
    (source / "pyproject.toml").write_text('[project]\nname = "forge-entrypoint-test"\n')
    (package / "__init__.py").write_text("")
    (package / "kernel_backends/constants.py").write_text(
        'KERNEL_BACKEND_PROMPT_MODULES = {"triton": "test"}\nKERNEL_BACKENDS = ["triton"]\n'
    )
    (package / "cli.py").write_text(
        "# CPU capability fixture; campaign execution is never invoked.\n"
        f"print({' '.join((*helper.COMMANDS, *helper.LOOP_OPTIONS))!r})\n"
    )
    kb = aklr / "harness/skills/aklr-kb"
    (kb / "triton").mkdir(parents=True)
    (kb / "SKILL.md").write_text("---\nname: aklr-kb\n---\nExternal knowledge\n")
    (kb / "triton/SKILL.md").write_text("External language knowledge\n")
    revisions = []
    for repository in (source, aklr):
        for arguments in (
            ["init", "-q"],
            ["config", "user.name", "Forge test"],
            ["config", "user.email", "forge-test@example.invalid"],
            ["add", "."],
            ["commit", "-qm", "Source fixture"],
        ):
            subprocess.run(["git", "-C", str(repository), *arguments], check=True)
        revisions.append(
            subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
            ).strip()
        )
    shutil.copytree(source, installed)
    selected = {
        "hyperloom_repo": str(source),
        "aklr_repo": str(aklr),
        "hyperloom_revision": revisions[0],
        "aklr_revision": revisions[1],
        "aklr_knowledge": {"path": str(kb)},
        "resolution": {"aklr_requested_ref": "fixture", "fallback_reasons": []},
    }
    session, report, capture = (
        tmp_path / name for name in ("session.json", "checked.json", "argv.json")
    )
    session.write_text(json.dumps(selected))
    campaign = tmp_path / "generated campaign.sh"
    campaign.write_text(
        "#!/bin/sh\n# Generated kernelforge.cli entrypoint contract\n"
        'exec "$FORGE_PYTHON" - "$@" <<\'CAPTURE\'\n'
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({'argv': sys.argv[1:], "
        "'env': {name: os.environ.get(name) for name in "
        "['HYPERLOOM_REPO', 'AKLR_REPO', 'AKLR_KB_ROOT', 'FORGE_INSTALLED_SOURCE']}}))\n"
        "CAPTURE\n"
    )
    campaign.chmod(0o755)
    environment = {
        **os.environ,
        "INFERA_REPO": str(SKILL.parents[3]),
        "HYPERLOOM_REPO": str(source),
        "AKLR_REPO": str(aklr),
        "FORGE_BACKEND": "triton",
        "FORGE_ENTRYPOINT": str(campaign),
        "FORGE_PREFLIGHT_JSON": str(report),
        "FORGE_SESSION_JSON": str(session),
        "FORGE_PYTHON": sys.executable,
        "PYTHONPATH": str(installed / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CAPTURE": str(capture),
    }
    environment.pop("FORGE_INSTALLED_SOURCE", None)
    return {
        "source": source,
        "installed": installed,
        "aklr": aklr,
        "kb": kb,
        "session": session,
        "report": report,
        "capture": capture,
        "env": environment,
    }


def preflight(runtime, output):
    return subprocess.run(
        [
            sys.executable,
            str(CHECK),
            "--session",
            str(runtime["session"]),
            "--installed-source",
            str(runtime["installed"]),
            "--backend",
            "triton",
            "--out",
            str(output),
        ],
        env=runtime["env"],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("identity", ["explicit", "saved-report", "same-session"])
@pytest.mark.parametrize("arguments", [["--model", "model with space"], ["--resume", "--gpu", "0"]])
def test_verified_editable_copy_launches_with_original_arguments(runtime, identity, arguments):
    if identity == "same-session":
        runtime["env"]["FORGE_PREFLIGHT_JSON"] = str(runtime["session"])
        output = runtime["session"]
    elif identity == "saved-report":
        output = runtime["report"]
    else:
        output = runtime["report"].with_name("direct.json")
        runtime["env"]["FORGE_INSTALLED_SOURCE"] = str(runtime["installed"])
    direct = preflight(runtime, output)
    assert direct.returncode == 0, direct.stderr
    result = subprocess.run(
        ["sh", str(ENTRYPOINT), *arguments], env=runtime["env"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    launched = json.loads(runtime["capture"].read_text())
    assert launched["argv"] == arguments
    assert launched["env"] == {
        "HYPERLOOM_REPO": str(runtime["source"]),
        "AKLR_REPO": str(runtime["aklr"]),
        "AKLR_KB_ROOT": str(runtime["kb"]),
        "FORGE_INSTALLED_SOURCE": str(runtime["installed"]),
    }
    checked = json.loads(Path(runtime["env"]["FORGE_PREFLIGHT_JSON"]).read_text())
    assert checked["installed"]["source_path"] == str(runtime["installed"])


@pytest.mark.parametrize("failure", ["tampered", "wrong-origin", "unidentified"])
def test_editable_copy_still_requires_byte_and_import_identity(runtime, failure):
    assert preflight(runtime, runtime["report"]).returncode == 0
    if failure == "tampered":
        (runtime["installed"] / "src/kernelforge/__init__.py").write_text("# changed\n")
        expected = "differs from pinned"
    elif failure == "wrong-origin":
        other = runtime["installed"].with_name("other")
        shutil.copytree(runtime["source"], other)
        runtime["env"]["FORGE_INSTALLED_SOURCE"] = str(other)
        expected = "outside"
    else:
        runtime["env"]["FORGE_PREFLIGHT_JSON"] = str(runtime["report"].with_name("fresh.json"))
        expected = "outside"
    result = subprocess.run(
        ["sh", str(ENTRYPOINT), "--resume"], env=runtime["env"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert expected in result.stderr
    assert not runtime["capture"].exists()


def test_explicit_source_overrides_saved_identity_but_is_reverified(runtime):
    assert preflight(runtime, runtime["report"]).returncode == 0
    saved = json.loads(runtime["report"].read_text())
    saved["installed"]["source_path"] = str(runtime["source"])
    runtime["report"].write_text(json.dumps(saved))
    runtime["env"]["FORGE_INSTALLED_SOURCE"] = str(runtime["installed"])
    result = subprocess.run(
        ["sh", str(ENTRYPOINT)], env=runtime["env"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(runtime["capture"].read_text())["env"]["FORGE_INSTALLED_SOURCE"] == str(
        runtime["installed"]
    )


def test_reuses_session_installed_identity_when_writing_a_new_report(runtime):
    assert preflight(runtime, runtime["session"]).returncode == 0
    assert not runtime["report"].exists()
    result = subprocess.run(
        ["sh", str(ENTRYPOINT), "--resume"], env=runtime["env"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(runtime["report"].read_text())["installed"]["source_path"] == str(
        runtime["installed"]
    )
