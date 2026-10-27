"""The REST API on the fixture AGENT_SYS_HOME, through FastAPI's test client."""

import os
import time

from conftest import error, ok

from agent_sys_v2.model import new_state, read_state, split_frontmatter, write_state

TASK = "/api/projects/llama70b/tasks"


def test_home_board_and_task(client):
    home = ok(client.get("/api/home"))
    # longest waiting first
    assert [t["id"] for t in home["in_review"]] == [
        "baseline/c64",
        "experiment/fp8_kv",
    ]
    assert [t["id"] for t in home["needs_you"]] == ["baseline/profile"]  # unmanaged, progress 阻塞
    assert [t["id"] for t in home["running"]] == ["experiment/tp4"]
    assert home["projects"][0]["counts"]["total"] == 7 and home["archived_projects"] == []
    assert [(t["id"], t["types"]) for t in home["templates"]] == [
        ("blank", ["develop"]),
        (
            "infera-perf",
            [
                "baseline",
                "close",
                "develop",
                "experiment",
                "intake",
                "plan",
                "profile",
                "review",
                "survey",
            ],
        ),
    ]
    assert home["templates"][1]["tasks"][1] == "intake/r1"

    board = ok(client.get("/api/projects/llama70b/board"))
    assert [ty["id"] for ty in board["types"]] == [
        "baseline",
        "experiment",
        "intake",
        "todo",
    ] and len(board["edges"]) == 4
    assert all(e["blocking"] for e in board["edges"])  # the UI does not know which kinds block
    assert board["running"] == {"count": 0, "limit": 3, "global_count": 0, "global_limit": 4}

    t = ok(client.get(f"{TASK}/experiment/combined"))
    assert (
        t["status"] == "draft"
        and t["prompt"]["path"] == "projects/llama70b/tasks/experiment/combined.md"
    )
    assert [(u["task"], u["satisfied"], u["blocking"]) for u in t["upstream"]] == [
        ("experiment/fp8_kv", False, True),
        ("experiment/tp4", False, True),
    ]
    assert t["runs"] == [] and t["questions"] is None and t["drafts"] == []
    assert t["progress"] is None and ok(client.get(f"{TASK}/baseline/c64"))["progress"]["done"]
    # a task's drafts are what 全部纳入托管 joins (ops.drafts_of): the UI does not work them out itself
    assert ok(client.get(f"{TASK}/experiment/fp8_kv"))["drafts"] == ["experiment/combined"]
    assert error(client.get(f"{TASK}/develop/nope"), 404) == "任务「develop/nope」不存在"
    assert error(client.get("/api/projects/nope/board"), 404) == "项目「nope」不存在"
    assert "接口不存在" in error(client.get("/api/nope"), 404)


def test_new_projects_from_both_templates(client, home):
    p = ok(
        client.post(
            "/api/projects", json={"name": "perf-q4", "template": "infera-perf", "title": "Q4"}
        )
    )
    assert (p["id"], p["title"], p["template"], p["counts"]["total"]) == (
        "perf-q4",
        "Q4",
        "infera-perf",
        5,  # the first round's tasks come with the template
    )
    assert [ty["id"] for ty in ok(client.get("/api/projects/perf-q4/board"))["types"]] == [
        "intake",
        "survey",
        "baseline",
        "profile",
        "plan",
        "experiment",
        "close",
        "develop",
        "review",
    ]
    ok(client.post("/api/projects", json={"name": "blank1", "template": "blank"}))
    assert [p["id"] for p in ok(client.get("/api/home"))["projects"]] == [
        "blank1",
        "llama70b",
        "perf-q4",
    ]
    assert "已存在" in error(client.post("/api/projects", json={"name": "blank1"}), 409)
    assert "不合法" in error(client.post("/api/projects", json={"name": "中文"}), 400)
    assert not (home / "projects/perf-q4/rules/protocol.md").exists()


def test_new_type_and_tasks(client, home):
    ty = ok(client.post("/api/projects/llama70b/types", json={"id": "perf", "review": True}))
    assert ty == {
        "id": "perf",
        "review": True,
        "checks": [],
        "skeleton": ty["skeleton"],
        "count": 0,
    }
    # the form's preview: prompt paths and chosen upstream tasks, as the scan will see them
    prompt = "## 任务描述\n\n- 按workspaces/experiment/tp4/report.md 补测\n"
    pv = ok(
        client.post(
            f"{TASK}/preview",
            json={
                "type": "perf",
                "name": "c128",
                "prompt": prompt,
                "depends_on": ["intake/goal"],
            },
        )
    )
    assert pv["name_error"] == "" and [(u["task"], u["source"]) for u in pv["upstream"]] == [
        ("experiment/tp4", "prompt"),
        ("intake/goal", "frontmatter"),
    ]
    assert (
        "不合法"
        in ok(client.post(f"{TASK}/preview", json={"type": "perf", "name": "c.v2"}))["name_error"]
    )
    assert (
        "已存在"
        in ok(client.post(f"{TASK}/preview", json={"type": "baseline", "name": "c64"}))[
            "name_error"
        ]
    )

    t = ok(
        client.post(
            TASK,
            json={
                "type": "perf",
                "name": "c128",
                "mode": "queue",
                "prompt": prompt,
                "depends_on": [{"task": "intake/goal", "kind": "input"}],
                "review": True,
                "agent": {"budget_usd": 150},
            },
        )
    )
    assert (t["status"], t["detail"], t["managed"]) == (
        "queued",
        {"kind": "upstream", "tasks": ["experiment/tp4"]},
        True,
    )
    assert t["agent"]["budget_usd"] == 150 and t["agent"]["model"] == "opus" and t["session_id"]
    text = (home / "projects/llama70b/tasks/perf/c128.md").read_text()
    assert text.startswith(
        "---\ndepends_on:\n  - intake/goal\nreview: true\nagent: {budget_usd: 150}\n---\n"
    )
    d = ok(
        client.post(TASK, json={"type": "perf", "name": "later", "mode": "draft", "prompt": "x\n"})
    )
    assert (
        d["status"] == "draft" and not (home / "projects/llama70b/workspaces/perf/later").exists()
    )
    assert "已存在" in error(
        client.post(TASK, json={"type": "perf", "name": "later", "mode": "draft"}), 409
    )


def test_update_rename_and_delete(client, home):
    t = ok(client.get(f"{TASK}/todo/spec_decode"))
    base = t["prompt"]["mtime"]
    t = ok(
        client.put(
            f"{TASK}/todo/spec_decode",
            json={
                "frontmatter": {"summary": None, "review": True},
                "prompt": "新正文\n",
                "base_mtime": base,
            },
        )
    )
    assert (t["summary"], t["review"], t["prompt"]["body"]) == ("", True, "新正文\n")
    path = home / "projects/llama70b/tasks/todo/spec_decode.md"
    os.utime(path, (time.time() + 5,) * 2)  # someone else wrote the file after we read it
    assert "已被修改" in error(
        client.put(f"{TASK}/todo/spec_decode", json={"prompt": "x", "base_mtime": base}), 409
    )

    r = ok(
        client.post(f"{TASK}/todo/spec_decode/rename", json={"type": "experiment", "name": "spec"})
    )
    assert r["id"] == "experiment/spec" and not path.exists()
    assert "正在运行" in error(
        client.post(f"{TASK}/experiment/tp4/rename", json={"name": "tp4b"}), 409
    )
    status = ok(client.get(f"{TASK}/baseline/c64"))["status"]
    # finished: may change type
    r = ok(client.post(f"{TASK}/baseline/c64/rename", json={"type": "intake"}))
    assert r["id"] == "intake/c64" and r["status"] == status
    assert (home / "projects/llama70b/workspaces/intake/c64/progress.md").is_file()

    lanes = ok(client.put("/api/projects/llama70b/lanes", json={"order": ["todo", "experiment"]}))[
        "types"
    ]
    assert [ty["id"] for ty in lanes] == ["todo", "experiment", "baseline", "intake"]
    assert [ty["id"] for ty in ok(client.get("/api/projects/llama70b/board"))["types"]] == [
        "todo",
        "experiment",
        "baseline",
        "intake",
    ]

    d = ok(client.post(f"{TASK}/experiment/spec/actions/delete"))
    assert d["deleted"] == "experiment/spec" and d["trash"].startswith("projects/llama70b/.trash/")
    assert (home / d["trash"] / "tasks/experiment/spec.md").is_file()


def test_join_approve_cancel_and_refusals(client, home):
    # approving an unmanaged task keeps it unmanaged and lets its downstream go
    t = ok(client.post(f"{TASK}/baseline/c64/actions/approve"))["task"]
    assert (t["status"], t["managed"]) == ("done", False)
    fm, _ = split_frontmatter((home / "projects/llama70b/tasks/baseline/c64.md").read_text())
    assert fm["approved_at"].endswith("Z")
    assert ok(client.get(f"{TASK}/experiment/tp4"))["upstream"][0]["satisfied"]

    t = ok(client.post(f"{TASK}/experiment/combined/actions/join"))["task"]  # a draft -> queued
    assert (t["status"], t["managed"], t["detail"]["kind"]) == ("queued", True, "upstream")
    # blocked -> needs_you / joined
    t = ok(client.post(f"{TASK}/baseline/profile/actions/join"))["task"]
    assert (t["status"], t["detail"]) == ("needs_you", {"kind": "joined"})
    assert "已纳入托管" in error(client.post(f"{TASK}/baseline/profile/actions/join"), 409)
    assert (
        ok(client.post(f"{TASK}/experiment/combined/actions/cancel"))["task"]["status"]
        == "cancelled"
    )
    assert (
        read_state(home / "projects/llama70b/workspaces/experiment/combined")["status"]
        == "cancelled"
    )
    for tid in ("experiment/combined", "experiment/tp4"):  # cancelled already; unmanaged
        assert "无法取消" in error(client.post(f"{TASK}/{tid}/actions/cancel"), 409)
    assert (
        error(client.post(f"{TASK}/baseline/profile/actions/start"), 409)
        == "任务当前为「待确认」，无法执行「开始」"
    )
    assert (
        error(client.post(f"{TASK}/baseline/profile/messages", json={"text": " "}), 400)
        == "消息内容不能为空"
    )
    assert "未托管" in error(
        client.post(f"{TASK}/experiment/tp4/messages", json={"text": "x"}), 409
    )
    assert "不认识" in error(client.post(f"{TASK}/baseline/profile/actions/fly"), 404)


def test_archive(client, home):
    p = ok(client.post("/api/projects/llama70b/archive"))
    assert p["archived"] and p["archived_at"].endswith("Z")
    home_data = ok(client.get("/api/home"))
    assert home_data["projects"] == [] and home_data["in_review"] == []
    assert [p["id"] for p in home_data["archived_projects"]] == ["llama70b"]
    assert "已归档" in error(client.post(TASK, json={"type": "develop", "name": "x"}), 409)
    assert not ok(client.post("/api/projects/llama70b/unarchive"))["archived"]
    # comments kept
    assert "# 上游完成后自动启动下游" in (home / "projects/llama70b/project.yaml").read_text()


def test_broken_project_yaml_only_hides_that_project(client, home):
    ok(client.post("/api/projects", json={"name": "other"}))
    (home / "projects/llama70b/project.yaml").write_text("title: [unclosed\n")
    client.app.state.store.sync()
    projects = {p["id"]: p for p in ok(client.get("/api/home"))["projects"]}
    assert "无法解析" in projects["llama70b"]["error"] and "counts" in projects["other"]
    assert "无法解析" in error(client.get("/api/projects/llama70b/board"), 422)


def test_files(client, home):
    base = f"{TASK}/baseline/c64"
    root = ok(client.get(f"{base}/files"))
    assert [(e["name"], e["type"]) for e in root["entries"]] == [
        ("log", "dir"),
        ("progress.md", "file"),
        ("report.md", "file"),
    ]
    assert (
        ok(client.get(f"{base}/files", params={"dir": "log"}))["entries"][0]["path"]
        == "log/bench.out"
    )
    f = ok(client.get(f"{base}/file", params={"path": "report.md"}))
    assert f["kind"] == "md" and f["content"].startswith("## 结论")
    saved = ok(
        client.put(
            f"{base}/file",
            params={"path": "report.md"},
            json={"content": "## 结论\n\n- 改了\n", "base_mtime": f["mtime"]},
        )
    )
    assert saved["path"] == "report.md"
    ws = home / "projects/llama70b/workspaces/baseline/c64"
    os.utime(ws / "report.md", (time.time() + 5,) * 2)
    assert "已被修改" in error(
        client.put(
            f"{base}/file",
            params={"path": "report.md"},
            json={"content": "x", "base_mtime": saved["mtime"]},
        ),
        409,
    )
    # out of the workspace, agent_sys's own files, and through a symlink: refused
    assert "不在任务工作区内" in error(
        client.get(f"{base}/file", params={"path": "../../experiment/fp8_kv/report.md"}), 400
    )
    assert ".agent_sys/" in error(
        client.put(f"{base}/file", params={"path": ".agent_sys/state.json"}, json={"content": ""}),
        400,
    )
    (ws / "outside").symlink_to(home / "projects/llama70b/workspaces/experiment/fp8_kv")
    # reading follows
    assert ok(client.get(f"{base}/file", params={"path": "outside/report.md"}))["content"]
    assert "软链接" in error(
        client.put(f"{base}/file", params={"path": "outside/report.md"}, json={"content": ""}), 400
    )
    (ws / ".agent_sys").mkdir()
    (ws / "state").symlink_to(ws / ".agent_sys")
    assert ".agent_sys/" in error(
        client.put(f"{base}/file", params={"path": "state/x.json"}, json={"content": ""}), 400
    )
    # long text: its tail; images and markup through /raw, sandboxed
    (ws / "log/big.log").write_text("".join(f"line {i} " + "x" * 80 + "\n" for i in range(5000)))
    big = ok(client.get(f"{base}/file", params={"path": "log/big.log"}))
    assert big["truncated"] and big["content"].splitlines()[-1].startswith("line 4999")
    (ws / "page.html").write_text('<script>fetch("/api/home")</script>')
    raw = client.get(f"{base}/raw", params={"path": "page.html"})
    assert (
        raw.headers["content-security-policy"] == "sandbox"
        and raw.headers["x-content-type-options"] == "nosniff"
    )
    (ws / "curve.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    img = ok(client.get(f"{base}/file", params={"path": "curve.png"}))
    assert img["kind"] == "image" and "url" not in img  # the UI makes the /raw URL (api.rawUrl)


def test_settings(client, home):
    s = ok(client.get("/api/settings", params={"project": "llama70b"}))
    assert s["global"]["values"] == {
        "max_running": 4,
        "language": "en",
        "protocol_language": "zh",
        "claude_config": "isolated",
    }
    assert s["project"]["path"] == "projects/llama70b/project.yaml"
    assert s["project"]["values"]["agent"]["budget_usd"] == 200
    out = ok(
        client.put(
            "/api/settings",
            json={
                "scope": "project",
                "project": "llama70b",
                "values": {"max_running": 5, "agent": {"model": "sonnet"}},
            },
        )
    )
    assert out["values"]["max_running"] == 5 and out["values"]["agent"]["effort"] == "high"
    text = (home / "projects/llama70b/project.yaml").read_text()
    assert "model: sonnet" in text and "# 任务默认值" in text
    # the global settings page sends project: null
    ok(
        client.put(
            "/api/settings", json={"scope": "global", "project": None, "values": {"max_running": 6}}
        )
    )
    assert (home / "agent_sys.yaml").read_text() == "max_running: 6\n"
    assert ok(client.get("/api/projects/llama70b/board"))["running"]["global_limit"] == 6
    assert "正整数" in error(
        client.put("/api/settings", json={"scope": "global", "values": {"max_running": 0}}), 400
    )
    for path in ("/api/settings", "/api/rules"):  # scope project without the project: the same 400
        assert "scope 应为" in error(client.put(path, json={"scope": "project", "values": {}}), 400)
    assert "语言应为" in error(
        client.put(
            "/api/settings", json={"scope": "global", "values": {"protocol_language": "fr"}}
        ),
        400,
    )
    ok(client.put("/api/settings", json={"scope": "global", "values": {"protocol_language": "en"}}))
    assert ok(client.get("/api/rules"))["protocol"]["language"] == "en"
    assert ok(client.get("/api/settings"))["projects"][0]["id"] == "llama70b"


def test_changes_reach_listeners(client, home):
    store, seen = client.app.state.store, []
    store.listeners.add(seen.append)
    ok(client.post(f"{TASK}/experiment/combined/actions/join"))
    assert {"project": "llama70b", "task": "experiment/combined"} in seen
    (home / "projects/llama70b/workspaces/experiment/tp4/progress.md").write_text("状态: 已完成\n")
    store.sync()  # what the 2 s poll does
    assert seen[-1] == {"project": "llama70b", "task": "experiment/tp4"}
    assert ok(client.get(f"{TASK}/experiment/tp4"))["status"] == "done"


def test_a_scan_that_fails_unexpectedly_shows_on_its_row(client, home, monkeypatch):
    from agent_sys_v2 import server

    ok(client.post("/api/projects", json={"name": "other"}))
    real = server.load_project

    def broken(pdir):
        if pdir.name == "llama70b":
            raise PermissionError("denied")
        return real(pdir)

    monkeypatch.setattr(server, "load_project", broken)
    (home / "projects/llama70b/project.yaml").write_text("title: 换个标题\n")
    client.app.state.store.sync()
    projects = {p["id"]: p for p in ok(client.get("/api/home"))["projects"]}
    assert "扫描项目失败" in projects["llama70b"]["error"] and "counts" in projects["other"]
    monkeypatch.setattr(server, "load_project", real)
    client.app.state.store.sync()  # scanned again at the next poll, without a new change
    assert ok(client.get("/api/projects/llama70b/board"))["project"]["title"] == "换个标题"


def test_a_file_needs_a_name(client):
    for path in ("", "."):
        assert "请指定" in error(
            client.put(f"{TASK}/baseline/c64/file", params={"path": path}, json={"content": "x"}),
            400,
        )


def test_saving_a_file_keeps_its_symlink_and_mode(client, home):
    ws = home / "projects/llama70b/workspaces/baseline/c64"
    (ws / "reports").mkdir()
    (ws / "reports/003.md").write_text("旧\n")
    (ws / "latest.md").symlink_to("reports/003.md")
    (ws / "run.sh").write_text("echo old\n")
    (ws / "run.sh").chmod(0o755)
    base = f"{TASK}/baseline/c64/file"
    ok(client.put(base, params={"path": "latest.md"}, json={"content": "新\n"}))
    assert (ws / "latest.md").is_symlink() and (ws / "reports/003.md").read_text() == "新\n"
    ok(client.put(base, params={"path": "run.sh"}, json={"content": "echo new\n"}))
    assert (ws / "run.sh").stat().st_mode & 0o777 == 0o755


def test_a_directory_that_cannot_be_read_still_lists(client, home):
    ws = home / "projects/llama70b/workspaces/baseline/c64"
    (ws / "locked").mkdir()
    (ws / "many").mkdir()
    for i in range(1001):
        (ws / f"many/{i}").touch()
    (ws / "many/.hidden").touch()
    (ws / "locked").chmod(0)
    try:
        entries = {e["name"]: e for e in ok(client.get(f"{TASK}/baseline/c64/files"))["entries"]}
    finally:
        (ws / "locked").chmod(0o755)
    if os.geteuid():  # root reads it anyway
        assert entries["locked"]["count"] is None
    assert (entries["many"]["count"], entries["many"]["more"]) == (1000, True)
    assert (entries["log"]["count"], entries["log"]["more"]) == (1, False)


def test_other_sites_and_other_host_names_are_refused(client):
    """Through `ssh -L` the browser reaches the server as localhost: another site's page may not make it do
    anything (a form or a no-cors fetch needs no preflight), nor may a page that got another name to point here
    (DNS rebinding); nor may another site show it in a frame."""
    draft = f"{TASK}/experiment/combined"
    cross = (
        {"origin": "https://evil.example", "host": "localhost:8765"},
        {"sec-fetch-site": "cross-site", "host": "localhost:8765"},
        {"origin": "null", "host": "localhost:8765"},  # a sandboxed page, such as /raw
    )
    for headers in cross:
        r = client.post(f"{draft}/actions/delete", headers=headers)
        assert r.status_code == 403 and "其他网站" in r.json()["error"], headers
    assert "不接受 Host" in error(
        client.get("/api/home", headers={"host": "evil.example:8765"}), 403
    )
    assert ok(client.get(draft))["status"] == "draft"  # nothing was done
    # the UI's own requests, and clients that are no browser (curl, this test client)
    same = (
        {
            "origin": "http://localhost:8765",
            "host": "localhost:8765",
            "sec-fetch-site": "same-origin",
        },
        {"origin": "http://127.0.0.1:9000", "host": "127.0.0.1:9000"},
        {"host": "[::1]:8765", "sec-fetch-site": "none"},
    )
    for headers in same:
        ok(client.post(f"{draft}/actions/mark_read", headers=headers))
    assert "frame-ancestors 'none'" in client.get("/").headers["content-security-policy"]


def test_bodies_of_the_wrong_shape_are_400s(client, home):
    for method, url, body in (
        ("post", "/api/mark_read", {"tasks": ["llama70b/baseline/c64"]}),
        ("put", "/api/settings", {"scope": "global", "values": ["max_running"]}),
        ("post", "/api/projects", {"name": "x", "template": 5}),
        ("put", f"{TASK}/todo/spec_decode", {"frontmatter": ["title"]}),
        ("post", f"{TASK}/baseline/c64/questions", {"answers": {"1": 5}}),
    ):
        assert "请求格式不正确" in error(getattr(client, method)(url, json=body), 400), url
    # a batch names projects by id only: nothing outside projects/ is read
    outside = home.parent / "outside"
    (outside / "tasks/develop").mkdir(parents=True)
    (outside / "project.yaml").write_text("title: outside\n")
    write_state(
        home / "projects/llama70b/workspaces/experiment/combined",
        new_state("done", finished="2026-10-22T04:00:00Z"),
    )
    client.app.state.store.sync()
    batch = [
        {"project": "../../outside", "task": "develop/x"},
        {"project": "llama70b", "task": "experiment/combined"},
    ]
    r = ok(client.post("/api/mark_read", json={"tasks": batch}))
    assert r["marked"] == ["llama70b/experiment/combined"] and "不存在" in r["errors"][0]["error"]
    assert [p["id"] for p in ok(client.get("/api/home"))["projects"]] == ["llama70b"]


def test_save_and_notify_saves_even_when_the_agent_cannot_be_told(client, home):
    ws = home / "projects/llama70b/workspaces/experiment/combined"
    write_state(ws, {**new_state("cancelled"), "runs": 1})
    t = ok(client.get(f"{TASK}/experiment/combined"))
    d = ok(
        client.put(
            f"{TASK}/experiment/combined",
            json={"prompt": "改过\n", "base_mtime": t["prompt"]["mtime"], "send_to_agent": True},
        )
    )
    assert d["prompt"]["body"] == "改过\n" and "已取消" in d["notify_error"]


def test_the_board_shows_the_global_limit_the_scheduler_uses(client, home):
    (home / "agent_sys.yaml").write_text("max_running: 0\n")  # not a limit: the default
    client.app.state.store.sync()
    assert ok(client.get("/api/projects/llama70b/board"))["running"]["global_limit"] == 4


def test_routes_nobody_calls_are_gone(client):
    for path in ("/api/projects", "/api/templates"):
        assert "接口不存在" in error(client.get(path), 404)


def test_every_project_is_scanned_again_now_and_then(client, monkeypatch):
    """Reminders that come with time alone (progress.md not updated for hours, today's cost) need no file change."""
    from agent_sys_v2 import scan, server

    store = client.app.state.store
    monkeypatch.setattr(scan, "STALE_HOURS", 0.001)  # as if a day went by: tp4 still says 进行中
    store.poll()
    assert ok(client.get(f"{TASK}/experiment/tp4"))["warnings"] == []  # nothing changed on disk
    monkeypatch.setattr(server, "FULL_SCAN_SECONDS", 0)
    store.poll()
    assert "小时未更新" in ok(client.get(f"{TASK}/experiment/tp4"))["warnings"][0]
