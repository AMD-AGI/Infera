"""The knowledge repo and its search, against a local bare repo (no network)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

AGENT_SYS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_SYS / "tools" / "skill" / "lib"))

from skillkit import remote, search  # noqa: E402

HELLO = AGENT_SYS / "examples" / "01_hello_world"


def _bare(tmp_path: Path, files: dict[str, str | Path] | None = None) -> Path:
    """A bare remote with one commit; `files` maps a path to text, or to a Path
    meaning 'a symlink to this'."""
    bare, seed = tmp_path / "remote.git", tmp_path / "seed"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    subprocess.run(["git", "clone", "-q", str(bare), str(seed)], check=True)
    for rel, what in {"README.md": "knowledge\n", **(files or {})}.items():
        (seed / rel).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(what, Path):
            os.symlink(what, seed / rel)
        else:
            (seed / rel).write_text(what)
    for cmd in (
        ["add", "-A"],
        ["commit", "-q", "-m", "init"],
        ["push", "-q", "origin", "HEAD:main"],
    ):
        subprocess.run(["git", "-C", str(seed), *cmd], check=True)
    return bare


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{k}_NAME", "t")
        monkeypatch.setenv(f"GIT_{k}_EMAIL", "t@example.com")
    monkeypatch.setenv("INFERA_AGENT_SYS_CONFIG", str(tmp_path / "cfg" / "config.json"))
    monkeypatch.setenv("INFERA_AGENT_SYS_SETTINGS", str(tmp_path / "cfg" / "settings.json"))


@pytest.fixture
def repo(tmp_path, env):
    return remote.init(str(_bare(tmp_path)), user="alice", clone=str(tmp_path / "clone"))


def handoff(tmp_path: Path, name: str = "h") -> Path:
    c = tmp_path / name / "content"
    (c / "items").mkdir(parents=True)
    (c / "README.md").write_text("# h\n\n## Purpose\nsglang decode throughput on MI300X.\n")
    return c


def remote_files(repo: remote.Config, ref: str = "main") -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", repo.url, "ls-tree", "-r", "-z", "--name-only", ref]
    )
    return [p.decode() for p in out.split(b"\0") if p]


def test_backup_promote_search_and_release(repo, tmp_path):
    assert remote.backup(repo, HELLO, name="hello")["relative"] == "alice/draft/tasks/hello"
    again = remote.backup(repo, HELLO, name="hello")
    assert again["relative"] == "alice/draft/tasks/hello.v2"  # never overwritten
    c = handoff(tmp_path)
    names = ["结果.md", "a b.txt", 'q"uote.txt', "tab\there.txt", "back\\slash.txt"]
    for n in names:
        (c / "items" / n).write_text(f"{n}\n")
    h = remote.backup(repo, c.parent, name="sglang_decode", context={"engine": "sglang"})
    meta = json.loads((Path(h["path"]) / remote.META).read_text())
    assert meta["type"] == "handoff" and len(meta["digest"]) == 64 and meta["sealed"] is False
    assert meta["context"] == {"engine": "sglang"} and remote.can_push(repo)[0]
    # no local path or host name is published, only placeholders
    assert meta["source"] == f"<source-dir>/{c.parent.name}" and meta["source_host"] == "<host>"
    assert str(tmp_path) not in (Path(h["path"]) / remote.META).read_text()

    remote.promote(repo, "handoff", "sglang_decode")
    files = remote_files(repo)
    base = "alice/staging/handoffs/sglang_decode/content/items/"
    assert all(base + n in files for n in names), files  # paths survive byte for byte
    assert not any(f.startswith("alice/draft/handoffs/") for f in files)

    res = search.search(repo, "优化 sglang 解码 吞吐 MI300X", local=False)
    top = res.hits[0]
    assert res.searched == 3 and (top.name, top.tier, top.sealed) == (
        "sglang_decode",
        "staging",
        False,
    )
    assert "decode" in top.matched and not any(sn.startswith("{") for sn in top.snippets)
    assert search.search(None, "kernelforge optimise one kernel", limit=3).hits[0].tier == "local"

    with pytest.raises(remote.RemoteError, match="promote it first"):
        remote.release_pr(repo, "task", "hello")
    branch = remote.release_pr(repo, "handoff", "sglang_decode")["branch"]
    assert "release/handoffs/sglang_decode/content/items/tab\there.txt" in remote_files(
        repo, branch
    )
    assert not any(f.startswith("release/") for f in remote_files(repo))
    with pytest.raises(remote.RemoteError, match="proposed already"):
        remote.release_pr(repo, "handoff", "sglang_decode")


def test_two_clones_same_name_and_bad_meta_never_wedge(repo, tmp_path):
    other = remote.Config(url=repo.url, clone=str(tmp_path / "clone2"), user="alice")
    subprocess.run(["git", "clone", "-q", repo.url, other.clone], check=True)
    a = remote.backup(repo, HELLO, name="same")
    b = remote.backup(other, HELLO, name="same")  # chosen against the remote: no add/add conflict
    assert {a["relative"], b["relative"]} == {"alice/draft/tasks/same", "alice/draft/tasks/same.v2"}
    remote.sync(repo)
    (Path(repo.clone) / "alice/draft/tasks/same" / remote.META).write_text("{not json")
    names = {i.name: i.meta for i in remote.items(repo)}
    assert "meta_error" in names["same"] and "same.v2" in names


def test_not_configured_and_reserved_user_names(tmp_path, env):
    with pytest.raises(remote.NotConfigured):
        remote.load()
    bare = _bare(tmp_path)
    for user in ("release", "Release", "staging", "draft"):
        with pytest.raises(remote.RemoteError, match="reserved"):
            remote.init(str(bare), user=user, clone=str(tmp_path / "c1"))
    cfg = remote.init(str(bare), user="alice", clone=str(tmp_path / "c1"))
    remote.config_path().write_text(json.dumps({**cfg.__dict__, "user": "release"}))
    with pytest.raises(remote.RemoteError, match="reserved"):
        remote.load()


def test_hostile_items_cannot_break_or_abuse_search(tmp_path, env):
    """Anybody can push: links out of an item are not followed, a malformed
    META is normalised for every reader."""
    victim = tmp_path / "victim_secret"
    victim.write_text("sglang deploy password=Sup3rS3cretValue99\n")
    (tmp_path / "vdir").mkdir()
    (tmp_path / "vdir" / "leak.md").write_text("sglang deploy vdir-leak\n")
    bad = {"summary": {"a": 1}, "tags": 5, "context": {"engine": "sglang", "n": {}}, "sealed": "y"}
    task = {"main.yaml": "x: 1\n", "assets/k": "k"}
    bare = _bare(
        tmp_path,
        {
            **{f"m/draft/tasks/evil/{k}": v for k, v in task.items()},
            "m/draft/tasks/evil/README.md": victim,
            "m/draft/handoffs/ev/content/README.md": Path("/dev/zero"),
            "m/draft/handoffs/ev/content/items/linked": tmp_path / "vdir",
            "m/draft/handoffs/ev/content/items/note.md": "sglang deploy note\n",
            "m/staging": tmp_path / "vdir",
            **{f"m/draft/tasks/t1/{k}": v for k, v in task.items()},
            f"m/draft/tasks/t1/{remote.META}": json.dumps(bad),
            f"m/draft/tasks/t2/{remote.META}": json.dumps({"tags": "sglang", "summary": 3}),
            f"m/draft/tasks/t3/{remote.META}": "[" * 100000,
        },
    )
    cfg = remote.init(str(bare), user="alice", clone=str(tmp_path / "clone"))
    metas = {(i.tier, i.name): i.meta for i in remote.items(cfg)}
    assert set(metas) == {("draft", n) for n in ("evil", "ev", "t1", "t2", "t3")}
    t1, t2 = metas[("draft", "t1")], metas[("draft", "t2")]
    assert (t1["summary"], t1["tags"], t1["context"]) == ("", [], {"engine": "sglang"})
    assert "sealed" not in t1 and (t2["tags"], t2["summary"]) == (["sglang"], "3")
    assert "meta_error" in metas[("draft", "t3")]
    res = search.search(cfg, "sglang deploy", local=False)
    snippets = [s for h in res.hits for s in h.snippets]
    assert not any("Sup3rS3cret" in s or "vdir-leak" in s for s in snippets), snippets
    assert any("note" in s for s in snippets) and {"t1", "t2"} <= {h.name for h in res.hits}
