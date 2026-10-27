"""agent_sys's language (i18n.py): every message has its English and every English is used, nothing a person or the
agent sees is left out of _() / translate(), English is spoken by default, and the language setting reaches the page
and the server's messages. Also the model choices (/api/models)."""

import ast
import re
from pathlib import Path

import pytest
from conftest import get, new, ok, run_home
from fastapi.testclient import TestClient

from agent_sys_v2 import i18n
from agent_sys_v2.i18n import EN, PARAM, _, status_word, translate
from agent_sys_v2.model import AgentSysError, check_name
from agent_sys_v2.server import create_app

SOURCES = Path(__file__).resolve().parents[1] / "src" / "agent_sys_v2"
CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")  # with the full-width punctuation
REAL_USE = i18n.use  # conftest keeps every other test in Chinese

# Chinese that is not for people and not translated: the skeleton of a new type, and the words agent_sys reads in what
# agents write; also texts translated where they are used (KIND_ZH). (file, the function or module constant it is in,
# the literal or None for any).
AGENT_FACING = {
    ("launch.py", "KIND_ZH", None),
    ("model.py", "is_done", None),
    ("model.py", "classify_progress", None),
    ("model.py", "parse", None),
    ("model.py", "field_value", None),
    ("ops.py", "DEFAULT_SKELETON", None),
    ("questions.py", "ANSWER", None),
    ("questions.py", "ANSWER_COLUMN", None),
    ("questions.py", "submit", None),
    ("runs.py", "resume_interrupted", None),
    ("runs.py", "add_budget", None),
    ("perfcheck.py", "PLACEHOLDER", None),
    ("perfcheck.py", "NO_REFERENCE", None),
    ("perfcheck.py", "check_intake", None),
    ("perfcheck.py", "check_survey", None),
    ("scan.py", "HANDOFF_HEAD", None),
}


def sources():
    return [f for f in sorted(SOURCES.glob("*.py")) if f.name != "i18n.py"]


# what is translated: the argument at this position of a call of the function of this name
TRANSLATED = {"_": 0, "translate": 0, "say_for": 1}


def messages(tree):
    """The literal texts of the _(), translate() and say_for() calls."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name = getattr(n.func, "id", getattr(n.func, "attr", ""))
            pos = TRANSLATED.get(name)
            if pos is not None and len(n.args) > pos and isinstance(n.args[pos], ast.Constant):
                out.append(n.args[pos])
    return out


def owned(tree):
    """[(node, the function / class / module constant it is in)]."""
    out = []

    def walk(node, owner):
        for child in ast.iter_child_nodes(node):
            o = owner
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                o = child.name
            elif (
                isinstance(child, ast.Assign)
                and owner is None
                and isinstance(child.targets[0], ast.Name)
            ):
                o = child.targets[0].id
            out.append((child, o))
            walk(child, o)

    walk(tree, None)
    return out


def test_every_message_has_its_english_with_the_same_placeholders():
    missing, mismatched = [], []
    for f in sources():
        for node in messages(ast.parse(f.read_text(encoding="utf-8"))):
            key = node.value
            if key not in EN:
                missing.append(f"{f.name}:{node.lineno} {key!r}")
            elif set(PARAM.findall(key)) != set(PARAM.findall(EN[key])):
                mismatched.append(f"{f.name}:{node.lineno} {key!r}")
    assert not missing and not mismatched, (missing, mismatched)
    assert not [
        v for v in EN.values() if CJK.search(v.replace("状态:", "").replace("完成时间:", ""))
    ]


def test_every_english_is_used():
    """An EN key that no source has as a string (a message, or a text translated where it is used) is dead."""
    used = set()
    for f in sources():
        used |= {
            n.value
            for n in ast.walk(ast.parse(f.read_text(encoding="utf-8")))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
    assert [k for k in EN if k not in used] == []


def test_no_chinese_left_outside_messages():
    """A Chinese string literal outside _(), a docstring and AGENT_FACING would show untranslated in English."""
    left = []
    for f in sources():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        skip = {id(n) for m in messages(tree) for n in ast.walk(m)}
        skip |= {
            id(n.body[0].value)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
        }
        for node, owner in owned(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and CJK.search(node.value)
                and id(node) not in skip
                and (f.name, owner, None) not in AGENT_FACING
                and (f.name, owner, node.value) not in AGENT_FACING
            ):
                left.append(f"{f.name}:{node.lineno} ({owner}) {node.value[:40]!r}")
    assert not left, left


def test_english_and_chinese(monkeypatch):
    monkeypatch.setattr(i18n, "LANG", "en")
    assert _("任务「{task}」不存在", task="develop/a") == 'Task "develop/a" does not exist'
    assert _("不在字典里的话 {x}", x=1) == "不在字典里的话 1"  # missing: shown in Chinese
    # a name without a value stays
    assert _("任务「{task}」不存在") == 'Task "{task}" does not exist'
    # another language than the UI's (what the agent is told)
    assert translate("继续", "zh") == "继续" and translate("继续", "en") == "Continue"
    assert status_word("needs_you") == "needs you"
    with pytest.raises(AgentSysError, match='Invalid task name "a b"'):
        check_name("a b", _("任务名"))
    i18n.set_language("zh-CN")
    assert i18n.LANG == "zh" and status_word("needs_you") == "待确认"
    assert (
        _("预算已用尽（${cost} / ${budget}），请先追加预算", cost="1.00", budget="2.00")
        == "预算已用尽（$1.00 / $2.00），请先追加预算"
    )
    assert [i18n.normalize(x) for x in ("zh", "中文", "en", "fr", None)] == [
        "zh",
        "zh",
        "en",
        "en",
        "en",
    ]


def test_the_page_and_the_messages_follow_the_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(i18n, "use", REAL_USE)
    home = run_home(tmp_path)
    with TestClient(create_app(home)) as c:
        # no language in agent_sys.yaml: English
        assert i18n.LANG == "en" and '<html lang="en">' in c.get("/").text
        r = c.get("/api/projects/nope/board")
        assert r.status_code == 404 and r.json()["error"] == 'Project "nope" does not exist'
        assert (
            c.put(
                "/api/settings", json={"scope": "global", "values": {"language": "fr"}}
            ).status_code
            == 400
        )
        ok(c.put("/api/settings", json={"scope": "global", "values": {"language": "zh"}}))
        assert "language: zh" in (home / "agent_sys.yaml").read_text(encoding="utf-8")
        assert (
            '<html lang="zh">' in c.get("/").text
            and '<html lang="zh">' in c.get("/index.html").text
        )
        assert c.get("/api/projects/nope/board").json()["error"] == "项目「nope」不存在"
        new(c, "a", mode="draft")
        (home / "projects/r/tasks/develop/a.md").write_text(
            "---\nparent: develop/none\n---\n做事\n", encoding="utf-8"
        )
        c.app.state.store.sync(also=[("r", "develop/a")])
        assert get(c, "a")["warnings"] == ["parent「develop/none」不存在"]
        ok(c.put("/api/settings", json={"scope": "global", "values": {"language": "en"}}))
        # scanned again in English
        assert get(c, "a")["warnings"] == ['Parent "develop/none" does not exist']


def test_models(tmp_path):
    with TestClient(create_app(run_home(tmp_path))) as c:
        m = ok(c.get("/api/models"))
        ids = [x["id"] for x in m["models"]]
        assert {
            "opus",
            "sonnet",
            "haiku",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
        } <= set(ids)
        assert all(set(x) == {"id", "label", "resolves"} for x in m["models"])
