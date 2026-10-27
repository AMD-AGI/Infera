import re
import shutil
from pathlib import Path

import pytest

from agent_sys_v2 import checks, cli, ops
from agent_sys_v2.model import AgentSysError, read_text, split_frontmatter
from agent_sys_v2.scan import load_project, project_ids

PROTOCOL = ops.TEMPLATES / "_common/rules/protocol.md"


def test_protocol_covers_the_conventions():
    text = PROTOCOL.read_text(encoding="utf-8")
    for item in (
        "状态: 进行中",
        "下一步:",
        "已完成",
        "完成时间:",
        "report.md",
        "## 结论",
        "issue.md",
        "questions.md",
        "你的意见",
        "questions_1022.md",
        "已按默认",
        "替你做的决定",
        "run_in_background",
        "timeout",
        "reached its background time limit",
        "log/",
        "不要写别的任务的工作区",
        "parent",
        "summary",
        "不要自己开始执行",
        "push",
        "lessons/",
        "type: rule",
        "date '+%F %H:%M'",
        "临时文件也放在本任务的工作区里",
    ):
        assert item in text, item
    assert len(text.splitlines()) < 60
    # progress.md: the head, then a list of facts and a checklist; the language of the work is the prompt's
    for item in (
        "- [x]",
        "- [ ]",
        "无序列表",
        "工作语言跟我写的任务 prompt 一致",
        "`@<类型>/<任务名>`",
    ):
        assert item in text, item


def test_protocol_names_every_variable_the_agent_and_the_checks_get():
    names = {f"${k}" for k in checks.env(Path("p"), "t/a", Path("w"), [], Path("s"))}
    for path in (PROTOCOL, PROTOCOL.with_name("protocol.en.md")):
        assert names == set(re.findall(r"\$AGENT_SYS_\w+", path.read_text(encoding="utf-8"))), path


def test_protocol_in_english_says_the_same():
    zh, en = (
        p.read_text(encoding="utf-8") for p in (PROTOCOL, PROTOCOL.with_name("protocol.en.md"))
    )
    heads = lambda text: [x for x in text.splitlines() if x.startswith("#")]  # noqa: E731
    assert len(heads(en)) == len(heads(zh)) and len(en.splitlines()) == len(zh.splitlines())
    # what agent_sys reads in progress.md and questions.md, in the English forms
    for item in (
        "Status: In progress",
        "Next:",
        '"Done"',
        "Finished: YYYY-MM-DD HH:MM",
        '"Blocked"',
        "## Conclusion",
        "Your answer",
        "questions_1022.md",
        "run_in_background",
        "reached its background time limit",
        "lessons/",
        "type: rule",
        "date '+%F %H:%M'",
        "- [x]",
        "- [ ]",
        "`@<type>/<name>`",
    ):
        assert item in en, item


def test_blank_template(tmp_path):
    blank = ops.list_templates()[0]
    assert (blank["id"], blank["title"], [t["id"] for t in blank["types"]]) == (
        "blank",
        "空白项目",
        ["develop"],
    )
    assert blank["description"] and "## 任务描述" in blank["types"][0]["skeleton"]
    pdir = ops.new_project(tmp_path, "demo", "blank", title="演示项目")
    assert pdir == tmp_path / "projects/demo"
    # the agent_sys protocol is not copied: every run injects it from the code
    assert sorted(str(p.relative_to(pdir)) for p in pdir.rglob("*")) == [
        "knowledge",
        "knowledge/INDEX.md",
        "knowledge/entries",
        "project.yaml",
        "rules",
        "rules/project.md",
        "task_types",
        "task_types/develop.md",
        "tasks",
        "tasks/develop",
        "workspaces",
    ]
    config = read_text(pdir / "project.yaml")
    assert "description" not in config  # it describes the template
    assert config.startswith("title: 演示项目\ntemplate: blank\nautorun: true")
    assert "# 任务默认值，新建任务时可改" in config  # comments of the template survive
    p = load_project(pdir)
    assert (p.title, [t.id for t in p.types], p.tasks) == ("演示项目", ["develop"], {})
    assert p.config["agent"]["budget_usd"] == 200 and not p.types[0].review
    assert project_ids(tmp_path) == ["demo"]
    # a new task of the type starts from its skeleton
    ops.new_task(pdir, "develop", "first", p.types[0].skeleton, title="第一个任务")
    t = load_project(pdir).tasks["develop/first"]
    assert "## 任务描述" in t.body and t.status == "queued" and t.agent["model"] == "opus"


def test_template_with_rules_knowledge_and_types(tmp_path):
    """The shape of infera-perf: rules/, knowledge/ with entries, several types."""
    templates = tmp_path / "templates"
    shutil.copytree(ops.TEMPLATES / "_common", templates / "_common")
    src = templates / "infera-perf"
    for path, text in {
        "project.yaml": "title: Infera 性能优化\nautorun: true\nagent: {model: opus, budget_usd: 200}\n",
        "rules/cluster.md": "# 集群使用\n",
        "knowledge/INDEX.md": "- skip-bad-nodes — 坏节点记录后跳过\n",
        "knowledge/entries/skip-bad-nodes.md": "---\nname: skip-bad-nodes\n---\n正文\n",
        "task_types/develop.md": "## 任务描述\n",
        "task_types/codereview.md": "---\nreview: true\n---\n## 评审对象\n",
    }.items():
        (src / path).parent.mkdir(parents=True, exist_ok=True)
        (src / path).write_text(text, encoding="utf-8")
    assert [t["id"] for t in ops.list_templates(templates)] == ["infera-perf"]

    pdir = ops.new_project(tmp_path / "home", "perf-q4", "infera-perf", templates_dir=templates)
    assert sorted(p.name for p in (pdir / "rules").iterdir()) == ["cluster.md"]
    assert read_text(pdir / "knowledge/INDEX.md").startswith("- skip-bad-nodes")
    assert (pdir / "knowledge/entries/skip-bad-nodes.md").is_file()
    p = load_project(pdir)
    assert p.title == "perf-q4" and p.config["template"] == "infera-perf"
    assert {t.id: t.review for t in p.types} == {"codereview": True, "develop": False}
    # copies are independent of the template
    (src / "rules/cluster.md").write_text("改了\n")
    assert read_text(pdir / "rules/cluster.md") == "# 集群使用\n"


def test_infera_perf_template(tmp_path):
    t = {x["id"]: x for x in ops.list_templates()}["infera-perf"]
    assert t["description"] and {x["id"]: x["review"] for x in t["types"]} == {
        "baseline": False,
        "close": False,
        "develop": False,
        "experiment": False,
        "intake": True,
        "plan": True,
        "profile": True,
        "review": True,
        "survey": False,
    }
    assert t["tasks"] == ["baseline/r1", "intake/r1", "plan/r1", "profile/r1", "survey/r1"]
    pdir = ops.new_project(tmp_path, "perf-q4", "infera-perf", title="Q4 性能")
    p = load_project(pdir)
    # the lanes in the order of a round; the ledger's curves, machines and skills are the project's to fill in
    assert p.title == "Q4 性能" and [ty.id for ty in p.types] == [
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
    assert p.config["ledger"]["primary"] == "input_tput_per_gpu" and p.config["machines"] == []
    assert p.config["ledger"]["x"] == "p90_intvty" and p.config["round"] == 1
    assert "infera-experiment" in p.config["skills"] and "infera-profile" in p.config["skills"]
    # the template's own project rules
    assert "## 目标与部署" in (pdir / "rules/project.md").read_text(encoding="utf-8")
    # a task written from each skeleton as is: no edges; it gets its type's acceptance checks
    for ty in p.types:
        ops.new_task(pdir, ty.id, f"first_{ty.id}", ty.skeleton, review=ty.review, queue=False)
    p = load_project(pdir)
    assert not [e for e in p.edges if "first_" in e.up or "first_" in e.down]
    assert p.tasks["profile/first_profile"].review
    assert all("## 任务描述" in ty.skeleton and "report.md" in ty.skeleton for ty in p.types)
    checks = {tid: [c["run"] for c in t.checks] for tid, t in p.tasks.items()}
    assert any("perfcheck plan" in c for c in checks["plan/first_plan"])
    assert any("perfcheck close" in c for c in checks["close/first_close"])
    assert checks["review/first_review"] == []
    assert any("ledger verify" in c for c in checks["experiment/first_experiment"])
    assert any("agent_sys_v2.ledger check" in c for c in checks["baseline/first_baseline"])


def test_new_project_refusals(tmp_path):
    ops.new_project(tmp_path, "demo")
    with pytest.raises(AgentSysError) as e:
        ops.new_project(tmp_path, "demo")
    assert e.value.status == 409
    for name, template in (("x", "nosuch"), ("x", "_common"), ("bad name", "blank")):
        with pytest.raises(AgentSysError):
            ops.new_project(tmp_path, name, template)
    assert not any(p.name.startswith(".") for p in (tmp_path / "projects").iterdir())


def test_cli_new(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_SYS_HOME", str(tmp_path / "env_home"))
    assert cli.main(["new", "a"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "env_home/projects/a")
    assert (
        cli.main(
            [
                "new",
                "b",
                "--template",
                "blank",
                "--title",
                "标题",
                "--home",
                str(tmp_path / "arg_home"),
            ]
        )
        == 0
    )
    fm, _ = split_frontmatter(
        "---\n" + read_text(tmp_path / "arg_home/projects/b/project.yaml") + "---\n"
    )
    assert fm["title"] == "标题"
    assert cli.main(["new", "a"]) == 1
    assert "已存在" in capsys.readouterr().err


def test_every_template_skill_and_web_file_is_package_data():
    """pyproject's package-data reaches every file of the templates, the skills and the web pages but the web's tests
    (a wheel install has them)."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")

    def globs(table):
        return re.findall(r'"([^"]+)"', text.split(f"[tool.setuptools.{table}]")[1].split("[")[1])

    pkg = root / "src/agent_sys_v2"
    shipped = {p for g in globs("package-data") for p in pkg.glob(g) if p.is_file()}
    shipped -= {p for g in globs("exclude-package-data") for p in pkg.glob(g)}
    files = {
        p
        for sub in ("templates", "skills", "web")
        for p in (pkg / sub).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and "test" not in p.relative_to(pkg).parts[:2]
    }
    assert sorted(str(p.relative_to(pkg)) for p in files - shipped) == []
    assert not any("web/test" in str(p) for p in shipped)
