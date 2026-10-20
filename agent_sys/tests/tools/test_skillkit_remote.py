"""The knowledge repo, search, trigger mode, the hint hook and the installer —
against a local bare repo, so no network is needed."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

AGENT_SYS = Path(__file__).resolve().parents[2]
SKILL = AGENT_SYS / "tools" / "skill"
sys.path.insert(0, str(SKILL / "lib"))

from skillkit import remote, search, settings  # noqa: E402

HELLO = AGENT_SYS / "examples" / "01_hello_world"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A bare 'remote' with one commit, and an isolated config/settings."""
    for k, v in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "INFERA_AGENT_SYS_CONFIG": str(tmp_path / "cfg" / "config.json"),
        "INFERA_AGENT_SYS_SETTINGS": str(tmp_path / "cfg" / "settings.json"),
    }.items():
        monkeypatch.setenv(k, v)
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(bare), str(seed)], check=True)
    (seed / "README.md").write_text("knowledge\n")
    subprocess.run(["git", "-C", str(seed), "add", "."], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-q", "-m", "init"], check=True)
    subprocess.run(["git", "-C", str(seed), "push", "-q", "origin", "HEAD:main"], check=True)
    return remote.init(str(bare), user="alice", clone=str(tmp_path / "clone"))


def _handoff(tmp_path: Path) -> Path:
    c = tmp_path / "h" / "content"
    (c / "items").mkdir(parents=True)
    (c / "README.md").write_text(
        "# h\n\n## Purpose\nsglang decode throughput on MI300X: 1234 tok/s.\n"
    )
    (c / "items" / "content").write_text("ran sglang with --cuda-graph-max-bs 256\n")
    return c


def test_backup_promote_search_and_release_guard(repo, tmp_path):
    out = remote.backup(repo, HELLO, name="hello", summary="hello world env check", tags=["env"])
    assert out["relative"] == "alice/draft/tasks/hello"
    again = remote.backup(repo, HELLO, name="hello", summary="again")
    assert again["relative"] == "alice/draft/tasks/hello.v2"  # never overwritten
    h = remote.backup(
        repo,
        _handoff(tmp_path),
        name="sglang_decode",
        summary="sglang decode tput on MI300X",
        context={"engine": "sglang"},
    )
    meta = json.loads((Path(h["path"]) / remote.META).read_text())
    assert meta["type"] == "handoff" and len(meta["digest"]) == 64
    assert meta["sealed"] is False and meta["context"] == {"engine": "sglang"}
    assert remote.can_push(repo)[0]

    remote.promote(repo, "handoff", "sglang_decode")
    tiers = {(i.type, i.name): i.tier for i in remote.items(repo)}
    assert tiers[("handoff", "sglang_decode")] == "staging"

    # The remote has every commit: a fresh clone sees the same items.
    log = subprocess.run(
        ["git", "-C", repo.clone, "log", "--oneline", "origin/main"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "promote(alice)" in log and log.count("backup(alice/draft)") == 3

    res = search.search(repo, "优化 sglang 解码 吞吐 MI300X", local=False)
    hits = res.hits
    assert (
        res.searched == 3 and hits and hits[0].name == "sglang_decode" and hits[0].tier == "staging"
    )
    assert "decode" in hits[0].matched and hits[0].sealed is False
    assert not any(sn.startswith("{") for sn in hits[0].snippets)  # META is not dumped raw
    local = search.search(None, "kernelforge optimise one kernel", limit=3)
    assert local.hits and local.hits[0].tier == "local"

    with pytest.raises(remote.RemoteError, match="promote it first"):
        remote.release_pr(repo, "task", "hello")
    out = remote.release_pr(repo, "handoff", "sglang_decode")
    branch = out["branch"]
    files = subprocess.run(
        ["git", "-C", repo.clone, "ls-tree", "-r", "--name-only", f"origin/{branch}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "release/handoffs/sglang_decode/content/README.md" in files
    main = subprocess.run(
        ["git", "-C", repo.clone, "ls-tree", "-r", "--name-only", "origin/main"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "release/" not in main and not (Path(repo.clone) / "release").exists()
    with pytest.raises(remote.RemoteError, match="proposed already"):
        remote.release_pr(repo, "handoff", "sglang_decode")


def test_two_clones_same_name_and_bad_meta_never_wedge(repo, tmp_path):
    other = remote.Config(url=repo.url, clone=str(tmp_path / "clone2"), user="alice")
    subprocess.run(["git", "clone", "-q", repo.url, other.clone], check=True)
    a = remote.backup(repo, HELLO, name="same")
    b = remote.backup(other, HELLO, name="same")  # chosen against the remote: no add/add conflict
    assert {a["relative"], b["relative"]} == {"alice/draft/tasks/same", "alice/draft/tasks/same.v2"}
    remote.sync(repo)
    remote.sync(other)
    (Path(repo.clone) / "alice" / "draft" / "tasks" / "same" / remote.META).write_text("{not json")
    names = {i.name: i.meta for i in remote.items(repo)}
    assert "meta_error" in names["same"] and "same.v2" in names


def test_task_backup_audits_symlink_targets(repo, tmp_path):
    pkg = tmp_path / "pkg"
    subprocess.run(["cp", "-r", str(HELLO), str(pkg)], check=True)
    outside = tmp_path / "outside.env"
    outside.write_text("HF_TOKEN=hf_AAAAAAAAAAAAAAAAAAAAAAAAAA\n")
    (pkg / "assets" / "creds.env").symlink_to(outside)
    with pytest.raises(remote.Refused, match="secret"):
        remote.backup(repo, pkg)


def test_backup_refuses_secrets_and_big_files(repo, tmp_path):
    c = _handoff(tmp_path)
    (c / "items" / "content").write_text("export HF_TOKEN=hf_abcdefghijklmnopqrstuvwx\n")
    with pytest.raises(remote.Refused, match="secret"):
        remote.backup(repo, c)
    (c / "items" / "content").write_text("fine\n")
    with (c / "items" / "big.bin").open("wb") as fh:
        fh.truncate(5 << 20)
    with pytest.raises(remote.Refused, match="4 MB"):
        remote.backup(repo, c)


def test_not_configured_and_trigger_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERA_AGENT_SYS_CONFIG", str(tmp_path / "none.json"))
    monkeypatch.setenv("INFERA_AGENT_SYS_SETTINGS", str(tmp_path / "s.json"))
    with pytest.raises(remote.NotConfigured):
        remote.load()
    assert settings.engaged("优化 sglang") and not settings.engaged("what is the capital of France")
    assert {"qwen", "mi300x"} <= set(settings.matches("tune qwen3.6 on mi300x"))
    assert settings.matches("优化一下网站的性能和延迟") == [] and settings.matches("PD分离部署")
    assert settings.named("/agent-sys-prior-work") and not settings.matches("an eagle flew")
    settings.save(trigger="off")  # alias of explicit
    assert settings.load()["trigger"] == "explicit"
    assert not settings.engaged("优化 sglang")
    for named in (
        "用 agent_sys 优化 sglang",
        "用agentsys优化",
        "run the agent-sys task",
        "/infera-agent-sys",
    ):
        assert settings.engaged(named), named
    assert not settings.engaged("look at /x/agent_sys/foo.py")


def test_gated_skills_render_per_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("INFERA_AGENT_SYS_SETTINGS", str(tmp_path / "s.json"))
    for trigger in ("auto", "explicit"):
        for p in settings.render_gated(SKILL, trigger):
            text = p.read_text()
            desc = next(ln for ln in text.splitlines() if ln.startswith("description:"))
            assert ("ONLY" in desc) == (trigger == "explicit"), desc
            assert str(SKILL) in text and "${CLAUDE_SKILL_DIR}" not in text


def _hook(prompt: str, env: dict) -> str:
    proc = subprocess.run(
        [sys.executable, str(SKILL / "hooks" / "prompt_hint.py")],
        input=json.dumps({"prompt": prompt}),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return proc.stdout


def test_hint_hook(tmp_path):
    env = {**os.environ, "INFERA_AGENT_SYS_SETTINGS": str(tmp_path / "s.json")}
    out = json.loads(_hook("帮我优化 sglang 在 MI300X 上的 decode 吞吐", env))
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "infera-agent-sys" in out["hookSpecificOutput"]["additionalContext"]
    assert _hook("优化一下我的网页", env) == ""
    assert _hook("the atomic counter overflows", env) == ""
    (tmp_path / "s.json").write_text(json.dumps({"trigger": "explicit"}))
    assert _hook("帮我优化 sglang", env) == ""
    assert _hook("用 agent_sys 优化 sglang", env) != ""


def test_installer_links_skills_and_adds_hook_idempotently(tmp_path):
    claude = tmp_path / "claude"
    claude.mkdir()
    (claude / "settings.json").write_text(json.dumps({"env": {"X": "1"}}))
    env = {
        **os.environ,
        "CLAUDE_CONFIG_DIR": str(claude),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        "INFERA_AGENT_SYS_CONFIG": str(tmp_path / "none.json"),
        "INFERA_AGENT_SYS_SETTINGS": str(tmp_path / "s.json"),
    }
    for _ in range(2):
        subprocess.run(
            ["sh", str(SKILL / "install.sh"), "--hook"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    names = sorted(p.name for p in (claude / "skills").iterdir())
    assert names == sorted(p.parent.name for p in SKILL.glob("*/SKILL.md"))
    gated = set(settings.GATED)
    assert all((claude / "skills" / n).resolve() == SKILL / n for n in names if n not in gated)
    assert all(
        (claude / "skills" / n).resolve() == tmp_path / "xdg" / "infera-agent-sys" / "skills" / n
        for n in gated
    )
    subprocess.run(
        ["sh", str(SKILL / "install.sh"), "--mode", "explicit"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        "ONLY" in (claude / "skills" / "infera-agent-sys" / "SKILL.md").read_text().split("\n")[2]
    )
    data = json.loads((claude / "settings.json").read_text())
    assert data["env"] == {"X": "1"} and len(data["hooks"]["UserPromptSubmit"]) == 1
    assert len(list(claude.glob("settings.json.bak-astk-*"))) == 1
    hook_cmd = data["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    probe = subprocess.run(
        ["sh", "-c", hook_cmd.replace("prompt_hint.py", "gone.py")],
        input="{}",
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0  # a moved checkout never blocks a prompt
