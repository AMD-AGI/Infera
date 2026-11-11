"""Session source resolution, runtime identity and external KB delivery without a GPU."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
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
    (kb / "SKILL.md").write_text("---\nname: aklr-kb\n---\nExternal KB entry\n")
    (kb / "triton/SKILL.md").write_text("External language entry\n")
    state = {
        "module": package / "__init__.py",
        "commands": forge.COMMANDS,
        "loop_options": forge.LOOP_OPTIONS,
        "calls": [],
        "hyperloom_revision": "a" * 40,
        "aklr_revision": "b" * 40,
    }
    (aklr / "requirements.txt").write_text(f'HYPERLOOM_REVISION="{state["hyperloom_revision"]}"\n')

    def run(argv):
        state["calls"].append(argv)
        if argv[0] == "git":
            if argv[1] == "ls-remote":
                return state["aklr_revision"] + "\trefs/heads/main"
            if "rev-parse" in argv:
                return (
                    state["hyperloom_revision"]
                    if Path(argv[2]) == hyperloom
                    else state["aklr_revision"]
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
    assert report["aklr_knowledge"]["revision"] == state["aklr_revision"]
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


def test_rejects_revision_drift_inside_existing_session_before_loading_runtime(
    sources, monkeypatch
):
    hyperloom, aklr, _, state = sources
    selected = forge.resolve_sources(hyperloom, aklr, aklr.parent / "sources")
    state["calls"].clear()
    previous = forge._run

    def other_revision(argv):
        return "different-revision" if "rev-parse" in argv else previous(argv)

    monkeypatch.setattr(forge, "_run", other_revision)
    with pytest.raises(forge.PreflightError, match="expected revision"):
        forge.check(hyperloom, aklr, "triton", resolution=selected)
    assert state["calls"] == []


def test_follows_current_aklr_main_and_its_declared_branch(sources):
    hyperloom, aklr, _, state = sources
    state["aklr_revision"] = "c" * 40
    (aklr / "requirements.txt").write_text("")
    contract = aklr / "harness/scripts/forge_task/README.md"
    contract.parent.mkdir(parents=True)
    contract.write_text("Install KernelForge from Hyperloom's AKLR branch.\n")
    previous = forge._run

    def remote_branch(argv):
        if argv[1] == "ls-remote" and argv[2] == forge.HYPERLOOM_REPOSITORY:
            return state["hyperloom_revision"] + "\trefs/heads/AKLR"
        return previous(argv)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(forge, "_run", remote_branch)
        report = forge.check(hyperloom, aklr, source_only=True)
    assert report["aklr_revision"] == "c" * 40
    assert report["hyperloom_revision"] == state["hyperloom_revision"]
    assert report["resolution"]["hyperloom_declaration"] == {
        "path": "harness/scripts/forge_task/README.md",
        "ref": "AKLR",
        "kind": "branch",
    }
    assert report["resolution"]["fallback_reasons"] == []


def test_discovered_moved_knowledge_requires_explicit_adoption(sources):
    hyperloom, aklr, kb, _ = sources
    moved = aklr / "harness/skills/kernel-knowledge"
    kb.rename(moved)
    with pytest.raises(forge.KnowledgeMoved, match="--accept-kb-path"):
        forge.resolve_sources(hyperloom, aklr, aklr.parent / "sources")
    selected = forge.resolve_sources(
        hyperloom, aklr, aklr.parent / "sources", accept_kb_path="harness/skills/kernel-knowledge"
    )
    report = forge.check(hyperloom, aklr, "gluon", source_only=True, resolution=selected)
    assert report["skill_dirs"] == [str(moved), str(moved / "triton")]
    assert report["aklr_knowledge"]["relative_path"] == "harness/skills/kernel-knowledge"


def test_adoption_rejects_another_skill_or_a_path_outside_aklr(sources):
    _, aklr, _, _ = sources
    other = aklr / "harness/skills/other"
    other.mkdir()
    (other / "SKILL.md").write_text("---\nname: other-skill\n---\n")
    outside = aklr.parent / "external-kb"
    outside.mkdir()
    (outside / "SKILL.md").write_text("---\nname: aklr-kb\n---\n")
    for name in ("harness/skills/other", "../external-kb"):
        with pytest.raises(forge.PreflightError, match="inside the selected AKLR"):
            forge._knowledge_path(aklr, name)


def test_later_preflight_reuses_session_without_resolving_main(sources, monkeypatch, tmp_path):
    hyperloom, aklr, _, state = sources
    selected = forge.resolve_sources(hyperloom, aklr, tmp_path / "sources")
    session, output = tmp_path / "session.json", tmp_path / "checked.json"
    session.write_text(json.dumps(selected))
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_forge.py", "--session", str(session), "--out", str(output), "--source-only"],
    )
    monkeypatch.setattr(
        forge, "resolve_sources", lambda *a, **k: pytest.fail("session resolved main twice")
    )
    state["calls"].clear()
    assert forge.main() == 0
    assert json.loads(output.read_text())["resolution"] == selected["resolution"]
    assert not any("ls-remote" in argv for argv in state["calls"])


@pytest.mark.parametrize("mode", ["unavailable", "missing-kb", "adaptation-failed"])
def test_fallback_materializes_suggested_aklr_without_changing_caller(tmp_path, monkeypatch, mode):
    hyperloom, aklr = tmp_path / "hyperloom", tmp_path / "aklr"

    def git(root, *args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    for root in (hyperloom, aklr):
        root.mkdir()
        git(root, "init", "-q")
        git(root, "config", "user.name", "Test")
        git(root, "config", "user.email", "test@example.invalid")
        git(root, "config", "commit.gpgsign", "false")
    (hyperloom / "pyproject.toml").write_text("# runtime\n")
    kb = aklr / forge.DEFAULT_KB_PATH
    kb.mkdir(parents=True)
    (kb / "SKILL.md").write_text("---\nname: aklr-kb\n---\nSuggested KB\n")
    for root in (hyperloom, aklr):
        git(root, "add", ".")
        git(root, "commit", "-qm", "suggested source")
    fallback = git(aklr, "rev-parse", "HEAD")
    monkeypatch.setattr(forge, "SUGGESTED_AKLR_REVISION", fallback)
    monkeypatch.setattr(forge, "SUGGESTED_HYPERLOOM_REVISION", git(hyperloom, "rev-parse", "HEAD"))
    (kb / "SKILL.md").write_text("---\nname: unrelated\n---\nNo kernel KB in this main revision\n")
    git(aklr, "commit", "-qam", "main changed its structure")
    main_head = git(aklr, "rev-parse", "HEAD")

    def remote_revision(repository, ref):
        if mode == "unavailable":
            raise forge.PreflightError("upstream unavailable")
        if mode == "adaptation-failed":
            pytest.fail("explicit suggested fallback should not resolve upstream main")
        return main_head

    monkeypatch.setattr(forge, "_remote_revision", remote_revision)
    selected = forge.resolve_sources(
        hyperloom,
        aklr,
        tmp_path / "sources",
        use_suggested="new campaign adapter cannot serve this workset"
        if mode == "adaptation-failed"
        else "",
    )
    actual = Path(selected["aklr_repo"])
    assert actual != aklr and git(actual, "rev-parse", "HEAD") == fallback
    assert (actual / forge.DEFAULT_KB_PATH / "SKILL.md").read_text().endswith("Suggested KB\n")
    assert git(aklr, "rev-parse", "HEAD") == main_head
    assert git(aklr, "status", "--porcelain") == ""
    assert selected["resolution"]["fallback_reasons"]


def test_hyperloom_ref_lookup_failure_selects_suggested_source(sources, monkeypatch):
    hyperloom, aklr, _, state = sources
    monkeypatch.setattr(forge, "SUGGESTED_HYPERLOOM_REVISION", state["hyperloom_revision"])
    previous = forge._remote_revision
    (aklr / "requirements.txt").write_text('HYPERLOOM_REF="unavailable-branch"\n')

    def remote_revision(repository, ref):
        if repository == forge.HYPERLOOM_REPOSITORY:
            raise forge.PreflightError("Hyperloom ref unavailable")
        return previous(repository, ref)

    # The real detached fallback checkout is covered above; this isolates the
    # declaration lookup decision without requiring a network or installed Git tree.
    monkeypatch.setattr(forge, "_remote_revision", remote_revision)
    monkeypatch.setattr(
        forge, "_checkout", lambda repository, revision, candidate, destination, **kw: candidate
    )
    selected = forge.resolve_sources(hyperloom, aklr, aklr.parent / "sources")
    assert selected["hyperloom_revision"] == state["hyperloom_revision"]
    assert selected["resolution"]["hyperloom_declaration"]["ref"] == "unavailable-branch"
    assert "Hyperloom ref could not be resolved" in selected["resolution"]["fallback_reasons"][-1]


def test_incomplete_owned_fetch_retries_in_fresh_directory(tmp_path):
    original, destination = tmp_path / "source", tmp_path / "sources/aklr-incomplete"
    original.mkdir()
    subprocess.run(["git", "init", "-q", str(original)], check=True)
    (original / "source.txt").write_text("Original source remains untouched\n")
    subprocess.run(["git", "-C", str(original), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(original),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "source",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(original), "rev-parse", "HEAD"], text=True
    ).strip()
    destination.parent.mkdir()
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    selected = forge._checkout(
        forge.AKLR_REPOSITORY, revision, original, destination, reuse_candidate=False
    )
    assert selected != destination and selected.name.endswith("-retry-1")
    assert (selected / "source.txt").read_text() == (original / "source.txt").read_text()
    assert destination.is_dir()


def test_cross_uid_source_trust_is_scoped_and_preserves_dirty_checks(tmp_path, monkeypatch):
    repositories = [tmp_path / "selected-source", tmp_path / "unrelated-source"]
    revisions = []
    for root in repositories:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        (root / "source.txt").write_text("original\n")
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "source",
            ],
            check=True,
        )
        revisions.append(
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
        )
    global_config = tmp_path / "unwritten-global.gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    monkeypatch.delenv("GIT_CONFIG_PARAMETERS", raising=False)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    selected, unrelated = repositories
    refusal = subprocess.run(
        ["git", "-C", str(selected), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if refusal.returncode == 0:
        if os.geteuid() != 0:
            pytest.skip(
                "Git lacks the ownership test flag and this user cannot change fixture ownership"
            )
        # Older Git lacks this flag. In a root test container, reproduce the
        # real UID boundary on these two disposable repositories only.
        for root in repositories:
            os.chown(root, 1, -1)
            os.chown(root / ".git", 1, -1)
        refusal = subprocess.run(
            ["git", "-C", str(selected), "rev-parse", "HEAD"], capture_output=True, text=True
        )
    assert refusal.returncode == 128 and "dubious ownership" in refusal.stderr
    assert forge._source(selected, revisions[0], ["source.txt"]) == revisions[0]
    still_untrusted = subprocess.run(
        ["git", "-C", str(unrelated), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    assert still_untrusted.returncode == 128 and "dubious ownership" in still_untrusted.stderr
    assert not global_config.exists()
    (selected / "source.txt").write_text("changed\n")
    with pytest.raises(forge.PreflightError, match="local changes"):
        forge._source(selected, revisions[0], ["source.txt"])
