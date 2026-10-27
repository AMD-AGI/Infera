"""The rules page and the knowledge page (knowledge.py): rules read and written, lesson candidates adopted /
turned into rules / discarded, the libraries and their INDEX.md, and lessons reaching the next task, played
through with a replay."""

import os
import re
import time

import pytest
from conftest import DONE, T, error, get, new, ok, replay, run_home, wait, ws
from fastapi.testclient import TestClient

from agent_sys_v2 import knowledge, launch
from agent_sys_v2.model import plain, split_frontmatter
from agent_sys_v2.server import create_app

PITFALL = (
    "---\nname: ssh-nohup-hang\ndescription: ssh 远程 nohup cmd & 会挂住；用 setsid -f nohup cmd </dev/null\n"
    "type: pitfall\nsource: develop/learn\n---\n**做法**：…  **原因**：…\n"
)
RULE = (
    "---\nname: always-tee-logs\ndescription: 以后长命令的输出都 tee 到 log/\ntype: rule\nsource: develop/learn\n---\n"
    "你在对话里说：以后都把长命令的输出 tee 到 log/。\n"
)


@pytest.fixture
def home(tmp_path):
    home = run_home(tmp_path, "autorun: false\n")
    (home / "rules").mkdir()
    (home / "rules" / "task_rules.md").write_text(
        "# 全局规则\n\n- 先写 progress.md\n", encoding="utf-8"
    )
    (home / "knowledge" / "entries").mkdir(parents=True)
    return home


def lessons(home, name, files):
    d = ws(home, name) / "lessons"
    d.mkdir(parents=True, exist_ok=True)
    for f, text in files.items():
        (d / f).write_text(text, encoding="utf-8")


def sync(c):
    c.app.state.store.sync()


def ended(t):
    """The task's run is over (a replay that writes no progress.md ends in 待确认)."""
    return t["runs"] and not t["runs"][0]["live"] and t["status"] != "running"


# ---------------------------------------------------------------- rules


def test_rules_are_read_written_and_sized_as_launch_md_lists_them(client, home):
    new(client, "a", "做 A")
    r = ok(client.get("/api/rules?project=r"))
    assert [f["name"] for f in r["global"]["files"]] == ["task_rules.md"] and r["global"][
        "dir"
    ] == "rules"
    assert [f["name"] for f in r["project"]["files"]] == ["project.md"] and r["project"][
        "dir"
    ] == "projects/r/rules"
    assert r["protocol"]["name"] == "protocol.md" and "任务约定" in r["protocol"]["content"]
    assert [(p["kind"], p["title"]) for p in r["injected"]["parts"]] == [
        ("protocol", "agent_sys 约定"),
        ("rules", "全局规则"),
        ("project_rules", "项目规则"),
        # the shipped skills: a project without `skills:` is offered all of them
        ("skills", "skills 索引"),
    ]
    assert r["injected"]["bytes"] == sum(p["bytes"] for p in r["injected"]["parts"])
    assert ok(client.get("/api/rules"))["project"] is None

    f = r["global"]["files"][0]
    saved = ok(
        client.put(
            "/api/rules",
            json={
                "scope": "global",
                "name": "task_rules.md",
                "base_mtime": f["mtime"],
                "content": f["content"] + "- 临时文件放在工作区里\n",
            },
        )
    )
    assert saved["path"] == "rules/task_rules.md" and saved["content"].endswith(
        "- 临时文件放在工作区里\n"
    )
    assert "已被修改" in error(
        client.put(
            "/api/rules",
            json={
                "scope": "global",
                "name": "task_rules.md",
                "content": "x",
                "base_mtime": "2026-01-01T00:00:00Z",
            },
        ),
        409,
    )
    # a new file with that name
    assert "已存在" in error(
        client.put("/api/rules", json={"scope": "global", "name": "task_rules.md", "content": "x"}),
        409,
    )
    assert "不合法" in error(
        client.put(
            "/api/rules",
            json={"scope": "project", "project": "r", "name": "../x.md", "content": "x"},
        ),
        400,
    )
    assert "不存在" in error(
        client.put(
            "/api/rules",
            json={"scope": "project", "project": "nope", "name": "x.md", "content": "x"},
        ),
        404,
    )
    ok(
        client.put(
            "/api/rules",
            json={
                "scope": "project",
                "project": "r",
                "name": "cluster.md",
                "content": "# 集群\n\n- 用 /path/to/clusters/ 的说明",
            },
        )
    )
    r = ok(client.get("/api/rules?project=r"))
    assert [f["name"] for f in r["project"]["files"]] == ["cluster.md", "project.md"]

    # what launch.md lists for a run of this project is what the rules page says
    ok(client.post(f"{T}/a/actions/start"))
    t = wait(client, "a", ended)
    md = (ws(home, "a") / ".agent_sys/runs" / t["runs"][0]["id"] / "launch.md").read_text(
        encoding="utf-8"
    )
    table = re.findall(r"^\| \d+ \| (.+?) \| `(.+?)` \| (.+?) \|$", md, re.M)
    assert table == [
        (
            p["title"],
            str(home / p["path"]) if not p["path"].startswith("/") else p["path"],
            launch.kb(p["bytes"]),
        )
        for p in r["injected"]["parts"]
    ]
    assert "临时文件放在工作区里" in md or "临时文件放在工作区里" in (
        ws(home, "a") / ".agent_sys/runs" / t["runs"][0]["id"] / "system.md"
    ).read_text(encoding="utf-8")


# ---------------------------------------------------------------- candidates


def test_candidates_are_listed_before_the_task_is_done_broken_ones_too(client, home):
    new(client, "learn", "做事", mode="draft")
    lessons(
        home,
        "learn",
        {
            "ssh.md": PITFALL,
            "no-front.md": "只有正文，没有 frontmatter\n",
            "broken.md": "---\nname: [x\n---\n正文\n",
            "half.md": "---\nname: half\ntype: tip\n---\nx\n",
        },
    )
    (ws(home, "learn") / "lessons" / ".done").mkdir()
    (ws(home, "learn") / "lessons" / ".done" / "old.md").write_text(PITFALL, encoding="utf-8")
    k = ok(client.get("/api/knowledge?project=r"))
    by = {c["file"]: c for c in k["candidates"]}
    assert sorted(by) == ["broken.md", "half.md", "no-front.md", "ssh.md"]  # .done/ is not listed
    c = by["ssh.md"]
    assert c["id"] == "r:develop/learn/lessons/ssh.md" and (
        c["task"],
        c["task_title"],
        c["task_status"],
    ) == ("develop/learn", "learn", "draft")
    assert (c["name"], c["type"], c["problems"], c["conflict"]) == (
        "ssh-nohup-hang",
        "pitfall",
        [],
        [],
    )
    assert c["path"] == "projects/r/workspaces/develop/learn/lessons/ssh.md"
    assert (by["no-front.md"]["name"], by["no-front.md"]["problems"]) == (
        "no-front",
        ["缺少 frontmatter", "缺少 name", "缺少 description", "缺少 type"],
    )
    assert by["broken.md"]["name"] == "broken" and by["broken.md"]["problems"][0].startswith(
        "frontmatter 无法解析"
    )
    assert by["half.md"]["problems"] == ["缺少 description", "type「tip」不认识"]
    assert [lib["scope"] for lib in k["libraries"]] == ["project", "global"]
    assert [lib["scope"] for lib in ok(client.get("/api/knowledge"))["libraries"]] == ["global"]
    # the global page lists every project's
    assert len(ok(client.get("/api/knowledge"))["candidates"]) == 4


def test_adopt_to_rule_and_discard(client, home):
    new(client, "learn", "做事", mode="draft")
    lessons(
        home,
        "learn",
        {
            "ssh.md": PITFALL,
            "tee.md": RULE,
            "other.md": PITFALL.replace("ssh-nohup-hang", "x1"),
            "p.md": PITFALL.replace("ssh-nohup-hang", "proj-one"),
        },
    )
    cid = lambda f: f"r:develop/learn/lessons/{f}"  # noqa: E731

    ok(
        client.post(
            "/api/knowledge/candidates/adopt",
            json={
                "id": cid("ssh.md"),
                "scope": "global",
                "description": "  ssh 远程后台启动会挂住：用 setsid -f  ",
            },
        )
    )
    entry = home / "knowledge/entries/ssh-nohup-hang.md"
    fm, body = split_frontmatter(entry.read_text(encoding="utf-8"))
    assert plain(fm) == {
        "name": "ssh-nohup-hang",
        "description": "ssh 远程后台启动会挂住：用 setsid -f",
        "type": "pitfall",
        "source": "r:develop/learn",
        "approved": time.strftime("%Y-%m-%d"),
    }
    # a bare date
    assert f"\napproved: {time.strftime('%Y-%m-%d')}\n" in entry.read_text(encoding="utf-8")
    assert body == "**做法**：…  **原因**：…\n"
    assert (home / "knowledge/INDEX.md").read_text(
        encoding="utf-8"
    ) == "- ssh-nohup-hang — ssh 远程后台启动会挂住：用 setsid -f\n"
    assert (ws(home, "learn") / "lessons/.done/ssh.md").is_file() and not (
        ws(home, "learn") / "lessons/ssh.md"
    ).exists()

    ok(client.post("/api/knowledge/candidates/adopt", json={"id": cid("p.md"), "scope": "project"}))
    fm, _ = split_frontmatter(
        (home / "projects/r/knowledge/entries/proj-one.md").read_text(encoding="utf-8")
    )
    assert fm["source"] == "develop/learn"
    assert (
        (home / "projects/r/knowledge/INDEX.md")
        .read_text(encoding="utf-8")
        .startswith("- proj-one — ssh 远程")
    )

    ok(
        client.post(
            "/api/knowledge/candidates/to_rule",
            json={
                "id": cid("tee.md"),
                "scope": "global",
                "file": "task_rules.md",
                "text": "- 长命令的输出都 tee 到 log/",
            },
        )
    )
    assert (home / "rules/task_rules.md").read_text(encoding="utf-8") == (
        "# 全局规则\n\n- 先写 progress.md\n\n- 长命令的输出都 tee 到 log/\n"
    )
    ok(
        client.post(
            "/api/knowledge/candidates/to_rule",
            json={
                "id": cid("other.md"),
                "scope": "project",
                "file": "new_rules.md",
                "text": "- 项目里的规则",
            },
        )
    )
    assert (home / "projects/r/rules/new_rules.md").read_text(
        encoding="utf-8"
    ) == "- 项目里的规则\n"
    assert sorted(p.name for p in (ws(home, "learn") / "lessons/.done").iterdir()) == [
        "other.md",
        "p.md",
        "ssh.md",
        "tee.md",
    ]
    assert "已处理" in error(
        client.post("/api/knowledge/candidates/discard", json={"id": cid("tee.md")}), 404
    )
    assert "不存在" in error(
        client.post("/api/knowledge/candidates/discard", json={"id": "r:../x/lessons/a.md"}), 404
    )

    # moving a handled candidate back makes it a candidate again; discarding a second one with the same file name
    # keeps both in .done/
    os.rename(ws(home, "learn") / "lessons/.done/tee.md", ws(home, "learn") / "lessons/tee.md")
    assert [c["file"] for c in ok(client.get("/api/knowledge?project=r"))["candidates"]] == [
        "tee.md"
    ]
    ok(client.post("/api/knowledge/candidates/discard", json={"id": cid("tee.md")}))
    lessons(home, "learn", {"tee.md": RULE})
    ok(client.post("/api/knowledge/candidates/discard", json={"id": cid("tee.md")}))
    assert len(list((ws(home, "learn") / "lessons/.done").glob("tee*.md"))) == 2
    assert ok(client.get("/api/knowledge?project=r"))["candidates"] == []


def test_same_name_is_reported_and_replaced(client, home):
    new(client, "learn", "做事", mode="draft")
    lessons(home, "learn", {"a.md": PITFALL})
    (home / "knowledge/entries/ssh-nohup-hang.md").write_text(
        PITFALL.replace("会挂住", "旧的说法"), encoding="utf-8"
    )
    c = ok(client.get("/api/knowledge?project=r"))["candidates"][0]
    assert c["conflict"] == ["global"]
    assert "经验名称" in error(
        client.post(
            "/api/knowledge/candidates/adopt",
            json={"id": c["id"], "scope": "global", "name": "不合法 的名字"},
        ),
        400,
    )
    assert "描述" in error(
        client.post(
            "/api/knowledge/candidates/adopt",
            json={"id": c["id"], "scope": "global", "description": " "},
        ),
        400,
    )
    # the same name, kept or given while editing: refused unless the page asked and you said replace
    (home / "knowledge/entries/other.md").write_text(PITFALL.replace("ssh-nohup-hang", "other"))
    for name in (None, "other"):
        assert "已有" in error(
            client.post(
                "/api/knowledge/candidates/adopt",
                json={"id": c["id"], "scope": "global", **({"name": name} if name else {})},
            ),
            409,
        )
    assert "旧的说法" in (home / "knowledge/entries/ssh-nohup-hang.md").read_text(encoding="utf-8")
    ok(
        client.post(
            "/api/knowledge/candidates/adopt",
            json={"id": c["id"], "scope": "global", "replace": True},
        )
    )
    assert "会挂住" in (home / "knowledge/entries/ssh-nohup-hang.md").read_text(encoding="utf-8")
    assert len(list((home / "knowledge/entries").iterdir())) == 2
    # the entry replaced is kept as a deleted one is
    assert "旧的说法" in (home / "knowledge/.trash/ssh-nohup-hang.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- libraries and INDEX.md


def test_entries_edited_by_hand_or_through_the_page_remake_the_index(client, home):
    entries = home / "knowledge/entries"
    (entries / "b-second.md").write_text(
        PITFALL.replace("ssh-nohup-hang", "b-second"), encoding="utf-8"
    )
    (entries / "a-first.md").write_text(
        RULE.replace("always-tee-logs", "a-first"), encoding="utf-8"
    )
    sync(client)  # what the server's 2 s loop does: entries/ changed by hand
    index = (home / "knowledge/INDEX.md").read_text(encoding="utf-8")
    assert index == (
        "- a-first — 以后长命令的输出都 tee 到 log/\n"
        "- b-second — ssh 远程 nohup cmd & 会挂住；用 setsid -f nohup cmd </dev/null\n"
    )
    lib = ok(client.get("/api/knowledge"))["libraries"][0]
    assert [e["name"] for e in lib["entries"]] == ["a-first", "b-second"]
    assert (lib["index"]["lines"], lib["index"]["bytes"], lib["index"]["limit"]) == (
        2,
        len(index.encode()),
        60,
    )

    e = lib["entries"][0]
    ok(
        client.put(
            "/api/knowledge/entries",
            json={
                "scope": "global",
                "name": "a-first",
                "description": "长命令 tee 到 log/",
                "type": "method",
                "body": "新的正文",
                "base_mtime": e["mtime"],
            },
        )
    )
    fm, body = split_frontmatter((entries / "a-first.md").read_text(encoding="utf-8"))
    # source stays
    assert (fm["type"], fm["source"], body) == (
        "method",
        "develop/learn",
        "新的正文\n",
    )
    assert (
        (home / "knowledge/INDEX.md")
        .read_text(encoding="utf-8")
        .startswith("- a-first — 长命令 tee 到 log/\n")
    )
    assert "已被修改" in error(
        client.put(
            "/api/knowledge/entries",
            json={
                "scope": "global",
                "name": "a-first",
                "description": "x",
                "type": "method",
                "body": "",
                "base_mtime": "2026-01-01T00:00:00Z",
            },
        ),
        409,
    )
    lib = ok(
        client.post("/api/knowledge/entries/delete", json={"scope": "global", "name": "b-second"})
    )
    assert [e["name"] for e in lib["entries"]] == ["a-first"] and (
        home / "knowledge/.trash/b-second.md"
    ).is_file()
    assert (home / "knowledge/INDEX.md").read_text(
        encoding="utf-8"
    ) == "- a-first — 长命令 tee 到 log/\n"
    assert "不存在" in error(
        client.post("/api/knowledge/entries/delete", json={"scope": "global", "name": "b-second"}),
        404,
    )


def test_a_new_server_remakes_an_index_written_while_none_ran(home):
    (home / "knowledge/entries/x.md").write_text(
        PITFALL.replace("ssh-nohup-hang", "x"), encoding="utf-8"
    )
    (home / "knowledge/INDEX.md").write_text("- 手写的旧索引\n", encoding="utf-8")
    with TestClient(create_app(home)):
        assert (
            (home / "knowledge/INDEX.md").read_text(encoding="utf-8").startswith("- x — ssh 远程")
        )


def test_index_lines(tmp_path):
    kdir = tmp_path / "knowledge"
    (kdir / "entries").mkdir(parents=True)
    # no entries: nothing to write
    assert not knowledge.refresh_index(kdir) and not (kdir / "INDEX.md").exists()
    (kdir / "INDEX.md").write_text("- gone — 已删除的条目\n")
    assert knowledge.refresh_index(kdir) and (kdir / "INDEX.md").read_text() == ""
    (kdir / "entries" / "z.md").write_text("---\nname: z\ntype: method\n---\n", encoding="utf-8")
    (kdir / "entries" / "y.md").write_text(
        '---\nname: y\ndescription: "多行\n  描述"\ntype: method\n---\n', encoding="utf-8"
    )
    # written once, then nothing changes
    assert knowledge.refresh_index(kdir) and not knowledge.refresh_index(kdir)
    assert (kdir / "INDEX.md").read_text(encoding="utf-8") == "- y — 多行 描述\n- z\n"
    # no entries/: an index written by hand is left alone
    assert not knowledge.refresh_index(tmp_path / "none")


# ---------------------------------------------------------------- lessons reach the next task, with a replay


def test_lessons_adopted_reach_the_next_task(client, home):
    """A task writes 2 lessons (one is your "以后都……" rule); one is adopted into the global library, the other becomes
    a rule; the next new task's launch.md has the index line and its system.md the rule."""
    writes = {
        "progress.md": DONE,
        "report.md": "## 结论\n\n- 做完了\n",
        "lessons/ssh-nohup-hang.md": PITFALL,
        "lessons/always-tee-logs.md": RULE,
    }
    new(client, "learn", "做事", agent=replay("message_mid_command", speed=40, writes=writes))
    ok(client.post(f"{T}/learn/actions/start"))
    wait(client, "learn", lambda t: t["status"] == "done")
    cands = {c["name"]: c for c in ok(client.get("/api/knowledge?project=r"))["candidates"]}
    assert (
        sorted(cands) == ["always-tee-logs", "ssh-nohup-hang"]
        and cands["always-tee-logs"]["type"] == "rule"
    )

    ok(
        client.post(
            "/api/knowledge/candidates/adopt",
            json={"id": cands["ssh-nohup-hang"]["id"], "scope": "global"},
        )
    )
    ok(
        client.post(
            "/api/knowledge/candidates/to_rule",
            json={
                "id": cands["always-tee-logs"]["id"],
                "scope": "global",
                "file": "task_rules.md",
                "text": "- 长命令的输出都 tee 到工作区的 log/（develop/learn 里你说的）",
            },
        )
    )
    assert ok(client.get("/api/knowledge?project=r"))["candidates"] == []

    new(client, "next", "下一个任务")
    ok(client.post(f"{T}/next/actions/start"))
    t = wait(client, "next", ended)
    run = ws(home, "next") / ".agent_sys/runs" / t["runs"][0]["id"]
    md = (run / "launch.md").read_text(encoding="utf-8")
    assert "### 全局经验索引（1 行）" in md
    assert "- ssh-nohup-hang — ssh 远程 nohup cmd & 会挂住；用 setsid -f nohup cmd </dev/null" in md
    assert "- 长命令的输出都 tee 到工作区的 log/（develop/learn 里你说的）" in (
        run / "system.md"
    ).read_text(encoding="utf-8")
    # system.md 全文, folded in launch.md
    assert "- 长命令的输出都 tee 到工作区的 log/（develop/learn 里你说的）" in md
    assert get(client, "learn")["status"] == "done"
