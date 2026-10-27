import json

import pytest

from agent_sys_v2.model import (
    AgentSysError,
    Edge,
    Progress,
    check_name,
    classify_progress,
    cycle_tasks,
    join_frontmatter,
    local_time_iso,
    new_state,
    plain,
    read_state,
    read_task_file,
    split_frontmatter,
    write_state,
    write_task_file,
)

DESIGN_EXAMPLE = """---
title: C64 性能优化迭代
summary: 按 profiling 结论在 C64 上迭代优化，每轮记录性能变化
depends_on:
  - develop/profile_method                           # 简写 = 普通依赖（input）
  - {task: develop/fix_router, kind: code}
parent: develop/optimize                             # 可选，父任务
review: true                                         # 完成后需要我审；默认值来自类型骨架，创建时可改
agent: {budget_usd: 300}                             # 可选，覆盖项目默认值
owner: someone                                       # 不认识的字段
---
## 任务描述
- （原来的自由文本 prompt，格式不变）
"""


def round_trips(files):
    for path in files:
        text = path.read_text(encoding="utf-8")
        fm, body = split_frontmatter(text)
        assert join_frontmatter(fm, body) == text, path


def test_frontmatter_round_trip_is_byte_identical(proj):
    round_trips(proj.glob("tasks/*/*.md"))
    fm, body = split_frontmatter(DESIGN_EXAMPLE)
    assert join_frontmatter(fm, body) == DESIGN_EXAMPLE


def test_frontmatter_edit_keeps_body_comments_order_and_unknown_fields(tmp_path):
    path = tmp_path / "a.md"
    path.write_text(DESIGN_EXAMPLE, encoding="utf-8")
    fm, body = read_task_file(path)
    fm["title"] = "新标题"
    fm["depends_on"].append("develop/x")
    write_task_file(path, fm, body)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("---\n## 任务描述\n- （原来的自由文本 prompt，格式不变）\n")
    assert "# 简写 = 普通依赖（input）" in text and "{task: develop/fix_router, kind: code}" in text
    assert "owner: someone" in text and "  - develop/x\n" in text
    keys = list(split_frontmatter(text)[0])
    assert keys == ["title", "summary", "depends_on", "parent", "review", "agent", "owner"]


def test_no_frontmatter_and_bad_frontmatter():
    fm, body = split_frontmatter("## 任务描述\n\n- x\n")
    assert fm == {} and body == "## 任务描述\n\n- x\n"
    assert join_frontmatter({}, body) == body
    with pytest.raises(ValueError):
        split_frontmatter("---\ntitle: [unclosed\n---\nbody\n")


def test_frontmatter_with_bom_crlf_or_only_comments():
    # written on Windows: read as frontmatter, so a write does not put a second one on top
    assert split_frontmatter("\ufeff---\r\ntitle: A\r\n---\r\nbody\r\n") == (
        {"title": "A"},
        "body\n",
    )
    # a prompt that opens with a --- rule and a heading: prompt text, not an empty frontmatter to drop
    text = "---\n# 背景\n---\n正文\n"
    assert split_frontmatter(text) == ({}, text)


def test_progress_head_rule():
    p = Progress.parse(
        "- **状态**：已完成（C64 部分）\n**下一步**: 无\n完成时间: 2026-10-21 17:10\n"
        "- [x] a\n- [~] b\n- [ ] c\n  * [X] d\n",
        1.0,
    )
    assert (p.status, p.next, p.finished) == ("已完成（C64 部分）", "无", "2026-10-21 17:10")
    assert p.checks == {"done": 2, "doing": 1, "todo": 1} and p.done
    assert Progress.parse("Status: done\n", 1.0).done
    assert Progress.parse("# 没有头部\n", 1.0).status == ""


@pytest.mark.parametrize(
    "head",
    [
        "**状态:** 已完成",  # the colon inside the bold: how a model most often writes it
        "**Status:** Done",
        "- **状态**: **已完成**",
        "status: done",
        "\ufeff状态: 已完成",
    ],
)
def test_progress_head_in_its_common_forms(head):
    p = Progress.parse(f"{head}\n下一步: 无\n", 1.0)
    assert p.done and p.to_json()["done"], p.status


@pytest.mark.parametrize(
    "status,review,expected",
    [
        ("进行中", False, ("running", {"kind": "unmanaged"})),
        ("阻塞（等用户）", False, ("needs_you", {"kind": "unmanaged"})),
        ("已完成", False, ("done", {})),
        ("已完成", True, ("in_review", {"kind": "review"})),
        ("暂停", False, ("unknown", {"raw": "暂停"})),
        ("", False, ("unknown", {"raw": ""})),
    ],
)
def test_classify_unmanaged(status, review, expected):
    assert classify_progress(Progress.parse(f"状态: {status}\n", 1.0), review) == expected


def test_classify_without_progress_is_draft():
    assert classify_progress(None, True) == ("draft", {})


def test_classify_approved_unmanaged_task():
    # review: true and done: in_review until approved; a newer finish in progress.md asks again
    p = Progress.parse("状态: 已完成\n完成时间: 2026-10-21 17:10\n", 1.0)
    assert classify_progress(p, True, p.finished_iso) == ("done", {})
    assert classify_progress(p, True, "2026-01-01T00:00:00Z") == ("in_review", {"kind": "review"})
    assert classify_progress(p, True, "") == ("in_review", {"kind": "review"})
    assert classify_progress(p, False, "") == ("done", {})


def test_names_are_ascii():
    for ok in ("m1c_server", "restest-c96", "A1", "9x"):
        check_name(ok)
    for bad in ("中文任务", "restest_c96.v2", "_x", "-x", "a/b", "a b", ""):
        with pytest.raises(AgentSysError):
            check_name(bad)


def test_cycle_tasks():
    e = lambda up, down, kind="input": Edge(up, down, kind)
    assert cycle_tasks([e("a", "b"), e("b", "c")]) == set()
    assert cycle_tasks(
        [e("a", "b"), e("b", "c", "code"), e("c", "a"), e("c", "d"), e("x", "y"), e("y", "x")]
    ) == {"a", "b", "c", "x", "y"}
    assert (
        cycle_tasks([e("a", "b"), e("b", "a", "review"), e("b", "c"), e("c", "b", "subtask")])
        == set()
    )


def test_local_time_iso():
    assert local_time_iso("2026-10-21 17:10（补）").endswith("Z")
    assert local_time_iso("昨天") == ""
    # in the form, but not a date or a time: no time, not an error that fails the project's scan
    assert local_time_iso("2026-09-31 10:00") == local_time_iso("2026-10-21 24:00") == ""


def test_state_schema(tmp_path):
    s = new_state("queued")
    assert set(s) == {
        "version",
        "status",
        "detail",
        "since",
        "session_id",
        "runs",
        "run_id",
        "cost_usd",
        "session_cost_usd",
        "created",
        "started",
        "finished",
        "seen_at",
    }
    assert s["runs"] == 0 and s["run_id"] is None and len(s["session_id"]) == 36
    assert new_state("queued")["session_id"] != s["session_id"]
    write_state(tmp_path, s)
    assert read_state(tmp_path) == s
    (tmp_path / ".agent_sys/state.json").write_text('{"status": "paused"}')
    with pytest.raises(ValueError):
        read_state(tmp_path)
    assert read_state(tmp_path / "none") is None


def test_yaml_dates_become_strings():
    fm, _ = split_frontmatter(
        "---\napproved: 2026-10-21\nat: 2026-10-22T08:00:00+08:00\nn: [1.5, {d: 2026-01-02}]\n---\n"
    )
    assert json.loads(json.dumps(plain(fm))) == {
        "approved": "2026-10-21",
        "at": "2026-10-22T00:00:00Z",
        "n": [1.5, {"d": "2026-01-02"}],
    }


def test_yaml_across_threads():
    """The server parses YAML in its request threads and its polling thread at once: no document may get another's
    values (ruamel's YAML instance keeps its parser state)."""
    import threading

    from agent_sys_v2.model import load_yaml

    docs = [f"title: t{i}\nreview: true\nagent: {{budget_usd: {i}}}\n" for i in range(2)]
    wrong = []

    def parse(i):
        for _ in range(400):
            data = load_yaml(docs[i])
            if data["title"] != f"t{i}" or data["agent"]["budget_usd"] != i:
                wrong.append(dict(data))

    threads = [threading.Thread(target=parse, args=(i,)) for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert wrong == []
