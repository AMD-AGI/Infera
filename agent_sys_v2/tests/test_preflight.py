"""preflight.py: the check before a start. Every check finds what it is about and nothing in a project without it; the
three ways in (开始运行 and a template task's first 开始, 全部纳入托管, `agent-sys-v2 check`) refuse on an error with the
whole list."""

import pytest
from conftest import DONE, T, error, manage, new, ok, run_home, set_status
from fastapi.testclient import TestClient

from agent_sys_v2 import cli, ops, preflight, runs
from agent_sys_v2.model import AgentSysError, read_state
from agent_sys_v2.scan import load_project
from agent_sys_v2.server import create_app

CONFIG = (
    "title: t\n"
    "machines: [smci355-a, {name: smci355-b, gpus: 8}]\n"
    "skills: [infera-bench, infera-deploy]\n"
)


def check(pdir, tasks=None):
    return preflight.preflight(load_project(pdir), tasks)


def brief(problems):
    """(task, level, message) of each problem."""
    return [(p["task"], p["level"], p["message"]) for p in problems]


def task(fm="", body="做事\n"):
    return f"---\n{fm}---\n{body}" if fm else body


def checks_of(*runs_):
    return "checks:\n" + "".join(
        f"  - {r!r}\n" if '"' in r or "#" in r else f"  - {r}\n" for r in runs_
    )


# ---------------------------------------------------------------- a clean project


@pytest.mark.parametrize("template", ["infera-perf", "blank"])
def test_a_new_project_has_no_problems(tmp_path, template):
    """The template's tasks under review: no machines yet, rules/project.md's placeholders, empty goal and working
    point are what intake fills in, not problems."""
    pdir = ops.new_project(tmp_path, "p", template)
    assert check(pdir) == []
    if template == "infera-perf":
        assert "<待 intake 填写>" in (pdir / "rules/project.md").read_text(encoding="utf-8")
        assert "machines: []" in (pdir / "project.yaml").read_text(encoding="utf-8")
        assert ops.start_problems(load_project(pdir)) == []


def test_which_tasks_are_checked(make_project):
    """By default every task not done, in review or cancelled, drafts too; tasks= picks some; project.yaml's problems
    go with any."""
    bad = task("machines: [nope]\n")
    pdir = make_project(
        {f"develop/{n}": bad for n in ("draft", "queued", "done", "review", "cancelled")},
        progress={"develop/done": DONE},
        config=CONFIG.replace("infera-deploy", "infera-deplyo"),
    )
    manage(pdir, "develop/queued")
    manage(pdir, "develop/review", status="in_review")
    manage(pdir, "develop/cancelled", status="cancelled")
    tasks = lambda ps: sorted({p["task"] for p in ps})  # noqa: E731
    assert tasks(check(pdir)) == ["", "develop/draft", "develop/queued"]
    assert tasks(check(pdir, ["develop/queued", "develop/missing"])) == ["", "develop/queued"]
    # 开始运行 lets the managed ones run: a draft waits to be joined
    assert tasks(ops.start_problems(load_project(pdir))) == ["", "develop/queued"]


# ---------------------------------------------------------------- the graph: depends_on, parent, cycles


def test_dependencies_and_parent(make_project):
    pdir = make_project(
        {
            "develop/alpha": "",
            "develop/b": task("depends_on: [develop/alpah]\n"),
            "develop/c": task("depends_on: [{task: develop/alpha, kind: inptu}]\n"),
            "develop/d": task("depends_on: [[x]]\nparent: develop/alhpa\n"),
            "develop/e": task("parent: [x]\ndepends_on: [develop/zzz]\n"),
            "develop/f": task(
                "depends_on: [{task: develop/alpha, kind: code}]\nparent: develop/alpha\n"
            ),
        }
    )
    problems = check(pdir)
    assert brief(problems) == [
        ("develop/b", "error", "depends_on 中的任务「develop/alpah」不存在"),
        (
            "develop/c",
            "error",
            "depends_on 中「develop/alpha」的 kind「inptu」不认识（可用 input/code/children/review/subtask）",
        ),
        (
            "develop/d",
            "error",
            "depends_on 中的「['x']」看不懂：应写成 类型/任务名 或 {task, kind, label}",
        ),
        ("develop/d", "error", "parent「develop/alhpa」不存在"),
        ("develop/e", "error", "depends_on 中的任务「develop/zzz」不存在"),
        ("develop/e", "error", "parent「['x']」看不懂：应写成 类型/任务名"),
    ]
    fixes = [p["fix"] for p in problems]
    assert fixes[0] == "最接近的已有任务是「develop/alpha」：是不是写错了？"
    assert fixes[1] == "是不是「input」？"
    assert "develop/a 或 {task: develop/a, kind: code}" in fixes[2]
    assert fixes[3] == "最接近的已有任务是「develop/alpha」：是不是写错了？"
    assert fixes[4] == "写成已有任务的 类型/任务名，或删掉这一项"  # nothing near
    # the card says the same (scan.lint)
    assert load_project(pdir).tasks["develop/b"].warnings == [
        "depends_on 中的任务「develop/alpah」不存在"
    ]


def test_a_cycle_is_told_once_with_its_tasks(make_project):
    pdir = make_project(
        {
            "develop/x": task("depends_on: [develop/z]\n"),
            "develop/y": task("depends_on: [develop/x]\n"),
            "develop/z": task("depends_on: [develop/y]\n"),
            "develop/w": task("depends_on: [develop/x]\n"),  # waits on the cycle, is not on it
            "develop/r": task("depends_on: [{task: develop/s, kind: review}]\n"),
            "develop/s": task(
                "depends_on: [develop/r]\n"
            ),  # a review edge does not block: no cycle
        }
    )
    problems = check(pdir)
    assert [(p["task"], p["level"]) for p in problems] == [("develop/x", "error")]
    assert problems[0]["message"].endswith("：develop/x → develop/y → develop/z → develop/x")
    assert "review" in problems[0]["fix"]
    # every task on it says so on its card
    p = load_project(pdir)
    assert (
        p.tasks["develop/z"].warnings[0].endswith("develop/z → develop/x → develop/y → develop/z")
    )
    assert not p.tasks["develop/w"].warnings


def test_a_task_that_cannot_be_read_says_only_that(make_project):
    """A frontmatter or checks that cannot be parsed: that one error, not what follows from it."""
    pdir = make_project(
        {
            "develop/a": "---\ntitle: [oops\n---\n做事\n",
            "develop/b": task("machines: [nope]\nchecks: [{name: x}]\ndepends_on: [develop/zzz]\n"),
        },
        config=CONFIG,
    )
    problems = check(pdir)
    assert [(p["task"], p["level"]) for p in problems] == [
        ("develop/a", "error"),
        ("develop/b", "error"),
    ]
    assert problems[0]["message"].startswith("frontmatter 无法解析")
    assert problems[1]["message"] == "checks 的第 1 项应为一条命令，或 {name, run, timeout}"
    assert "frontmatter" in problems[1]["fix"]


# ---------------------------------------------------------------- machines and skills


def test_machines_project_yaml_does_not_declare(make_project):
    pdir = make_project(
        {
            "develop/ok": task("machines: [smci355-a, smci355-b]\n"),
            "develop/typo": task("machines: [smci355-c]\n"),
            "develop/far": task("machines: smc-a16-19\n"),
            "develop/none": "",
        },
        config=CONFIG,
    )
    assert brief(check(pdir)) == [
        (
            "develop/far",
            "error",
            "machines 中的「smc-a16-19」不在 project.yaml 的 machines 中（有：smci355-a, smci355-b）",
        ),
        (
            "develop/typo",
            "error",
            "machines 中的「smci355-c」不在 project.yaml 的 machines 中（有：smci355-a, smci355-b）",
        ),
    ]
    assert [p["fix"] for p in check(pdir)] == [
        "先在 project.yaml 的 machines 中加上它，或改成已声明的机器",
        "是不是「smci355-b」？",
    ]
    # none declared yet: a task that names one is wrong, one that names none is not
    (pdir / "project.yaml").write_text("title: t\nmachines: []\n", encoding="utf-8")
    assert [p["message"] for p in check(pdir, ["develop/far", "develop/none"])] == [
        "machines 中的「smc-a16-19」不在 project.yaml 的 machines 中（有：还没有声明机器）"
    ]


def test_skills_that_do_not_exist(make_project):
    pdir = make_project({"develop/a": ""}, config=CONFIG)
    assert check(pdir) == []
    (pdir / "project.yaml").write_text(
        "title: t\nskills: [infera-bnech, mine, nothing-like-it]\n", encoding="utf-8"
    )
    (pdir / "skills/mine").mkdir(parents=True)
    (pdir / "skills/mine/SKILL.md").write_text(
        "---\nname: mine\ndescription: 项目自己的\n---\n", encoding="utf-8"
    )
    problems = check(pdir)
    assert [(p["task"], p["level"]) for p in problems] == [("", "error"), ("", "error")]
    assert problems[0]["message"].startswith(
        "project.yaml 的 skills 中的「infera-bnech」不存在（有：infera-bench, "
    )
    assert "mine" in problems[0]["message"] and problems[0]["fix"] == "是不是「infera-bench」？"
    assert problems[1]["fix"] == "删掉它，或在项目的 skills/nothing-like-it/SKILL.md 写一个"
    # no skills: = all of them
    (pdir / "project.yaml").write_text("title: t\n", encoding="utf-8")
    assert check(pdir) == []


# ---------------------------------------------------------------- what a check runs


PY = '"$AGENT_SYS_PYTHON" -m agent_sys_v2'


def test_modules_and_kinds_a_check_runs(make_project):
    good = checks_of(
        f'{PY}.perfcheck plan "$AGENT_SYS_TASK"',
        f'test ! -d "$AGENT_SYS_WORKSPACE/results" || {PY}.ledger check "$AGENT_SYS_WORKSPACE/results"',
        f'{PY}.ledger verify "$AGENT_SYS_WORKSPACE/results"',
        f"{PY}.ledger gsm8k --help",
        f'{PY}.perfcheck "$KIND" "$AGENT_SYS_TASK"',  # what the shell fills in is not checked
        f"{PY}.cli --version",
        f"{PY}.perfcheck plan; echo",
        f"{PY}.ledger;echo",
        f"true # {PY}.nope",
    )
    bad = checks_of(
        f'{PY}.perfchek survey "$AGENT_SYS_TASK"',
        f'{PY}.perfcheck plna "$AGENT_SYS_TASK"',
        f'{PY}.ledger veriffy "$AGENT_SYS_WORKSPACE/results"',
        f"{PY}.zzzzzz",
    )
    pdir = make_project({"develop/good": task(good), "develop/bad": task(bad)})
    problems = check(pdir)
    assert [(p["task"], p["level"]) for p in problems] == [("develop/bad", "error")] * 4
    assert [p["message"] for p in problems] == [
        f'验收检查「{PY}.perfchek survey "$AGENT_SYS_TASK"」运行的模块 agent_sys_v2.perfchek 不存在',
        f'验收检查「{PY}.perfcheck plna "$AGENT_SYS_TASK"」的 perfcheck 类型「plna」不认识（可用 intake/survey/baseline/plan/close）',
        f'验收检查「{PY}.ledger veriffy "$AGENT_SYS_WORKSPACE/results"」的 ledger 类型「veriffy」不认识（可用 check/verify/agentx/gsm8k）',
        f"验收检查「{PY}.zzzzzz」运行的模块 agent_sys_v2.zzzzzz 不存在",
    ]
    assert [p["fix"] for p in problems] == [
        "是不是 agent_sys_v2.perfcheck？",
        "是不是「plan」？",
        "是不是「verify」？",
        "改成存在的模块，或去掉这条检查",
    ]


def test_paths_a_check_uses(make_project):
    """$AGENT_SYS_SKILLS/ is shipped code: it is there or the check is wrong. A script the check runs from the project
    must be there; other project files may be an upstream task's output (a warning); a workspace's files and what
    test looks at are the check's subject; what the shell would expand is not read."""
    good = checks_of(
        '"$AGENT_SYS_PYTHON" "$AGENT_SYS_SKILLS/infera-experiment/scripts/ab_judge.py" --help',
        "bash ${AGENT_SYS_PROJECT}/tools/gate.sh",
        "$AGENT_SYS_PROJECT/tools/gate.sh && echo ok",
        'test -s "$AGENT_SYS_PROJECT/stack.json"',
        '[ -f "$AGENT_SYS_PROJECT/later.json" ]',
        '! [[ -e "$AGENT_SYS_PROJECT/lock" ]]',
        'grep x "$AGENT_SYS_PROJECT/workspaces/develop/a/report.md"',
        'cat "$AGENT_SYS_PROJECT/results/$AGENT_SYS_TASK.json" $AGENT_SYS_PROJECT/logs/*.log',
        # the shell's operators stuck to a word, a comment, a keyword or ! leading a command, a redirection's target
        "bash $AGENT_SYS_PROJECT/tools/gate.sh; echo done",
        "$AGENT_SYS_PROJECT/tools/gate.sh&&echo ok|tail -1",
        "true # bash $AGENT_SYS_PROJECT/tools/commented.sh",
        "test ! -e $AGENT_SYS_SKILLS/lock",
        "if test -f $AGENT_SYS_PROJECT/x; then echo; fi",
        "echo x > $AGENT_SYS_PROJECT/out.log 2>&1",
    )
    bad = checks_of(
        'python3 "$AGENT_SYS_SKILLS/infera-experiment/scripts/ab_judeg.py"',
        'cat "$AGENT_SYS_SKILLS/infera-bench/nothing.md"',
        "bash $AGENT_SYS_PROJECT/tools/gate2.sh",
        "$AGENT_SYS_PROJECT/tools/nope",
        'python "$AGENT_SYS_PROJECT/tools/gate.py" "$AGENT_SYS_PROJECT/stack.json"',
    )
    pdir = make_project({"develop/good": task(good), "develop/bad": task(bad)})
    (pdir / "tools").mkdir()
    for name in ("gate.sh", "gate.py"):
        (pdir / "tools" / name).write_text("exit 0\n", encoding="utf-8")
    problems = check(pdir)
    assert [(p["task"], p["level"]) for p in problems] == [("develop/bad", "error")] * 4 + [
        ("develop/bad", "warn")
    ]
    words = [p["message"].split("用到的 ")[1] for p in problems]
    assert words == [
        "$AGENT_SYS_SKILLS/infera-experiment/scripts/ab_judeg.py 不存在",
        "$AGENT_SYS_SKILLS/infera-bench/nothing.md 不存在",
        "$AGENT_SYS_PROJECT/tools/gate2.sh 不存在",
        "$AGENT_SYS_PROJECT/tools/nope 不存在",
        "$AGENT_SYS_PROJECT/stack.json 不存在（如果它由上游任务生成，可以忽略）",
    ]
    assert problems[0]["fix"] == "是不是 $AGENT_SYS_SKILLS/infera-experiment/scripts/ab_judge.py？"
    assert problems[2]["fix"] == "是不是 $AGENT_SYS_PROJECT/tools/gate.sh？"
    assert problems[3]["fix"] == "改正路径，或去掉这条检查"


def test_a_command_that_cannot_be_split_is_a_warning(make_project):
    pdir = make_project({"develop/a": task('checks:\n  - echo "oops\n')})
    assert [(p["level"], p["fix"]) for p in check(pdir)] == [("warn", "检查引号是否配对")]


# ---------------------------------------------------------------- warnings


def test_lanes_and_a_cancelled_upstream_are_warnings(make_project):
    pdir = make_project(
        {
            "develop/a": "",
            "develop/b": task("depends_on: [develop/a]\n"),
            "review/c": task("depends_on: [{task: develop/a, kind: review}]\n"),
            "develop/d": "",
        },
        config="title: t\nlanes: [develop, review]\n",
    )
    assert check(pdir) == []
    (pdir / "project.yaml").write_text("title: t\nlanes: [develop]\n", encoding="utf-8")
    manage(pdir, "develop/a", "develop/b", "review/c")
    set_status(pdir, "develop/a", "cancelled")
    problems = check(pdir)
    assert brief(problems) == [
        ("", "warn", "类型「review」不在 project.yaml 的 lanes 中：它的泳道排在最后"),
        ("develop/b", "warn", "依赖的任务「develop/a」已取消：本任务会一直等它"),
        ("review/c", "warn", "依赖的任务「develop/a」已取消"),
    ]
    # warnings do not stop 开始运行
    (pdir / "project.yaml").write_text(
        "title: t\nlanes: [develop]\ntemplate_review: true\n", encoding="utf-8"
    )
    ops.start_project(pdir)
    assert not load_project(pdir).template_review
    # no lanes: types in any order
    (pdir / "project.yaml").write_text("title: t\n", encoding="utf-8")
    assert [p["task"] for p in check(pdir)] == ["develop/b", "review/c"]


def test_errors_come_first_and_read_as_lines(make_project):
    pdir = make_project(
        {
            "develop/a": "",
            "develop/b": task("depends_on: [develop/a]\n"),
            "develop/c": task("machines: [n2]\n"),
        },
        config="title: t\nmachines: [n1]\n",
    )
    manage(pdir, "develop/a", "develop/b")
    set_status(pdir, "develop/a", "cancelled")
    problems = check(pdir)
    assert [p["level"] for p in problems] == ["error", "warn"]
    assert preflight.text(problems) == [
        "error  develop/c: machines 中的「n2」不在 project.yaml 的 machines 中（有：n1）",
        "       → 先在 project.yaml 的 machines 中加上它，或改成已声明的机器",
        "warn   develop/b: 依赖的任务「develop/a」已取消：本任务会一直等它",
        "       → 去掉这条依赖，或改为依赖接替它的任务",
    ]


# ---------------------------------------------------------------- 开始运行


def template_project(make_project, **tasks):
    pdir = make_project(
        {"develop/a": "", "develop/b": task("depends_on: [develop/a]\n"), **tasks},
        config="title: t\ntemplate_review: true\nmachines: [n1]\n",
    )
    manage(pdir, *[f"develop/{n}" for n in ("a", "b", *[t.split("/")[1] for t in tasks])])
    return pdir


def test_start_project_is_refused_with_the_whole_list(make_project):
    pdir = template_project(
        make_project, **{"develop/c": task("machines: [n2]\ndepends_on: [develop/zz]\n")}
    )
    with pytest.raises(AgentSysError) as e:
        ops.start_project(pdir)
    assert e.value.status == 409
    assert str(e.value).startswith("有 2 个问题要先改正，才能开始运行：\nerror  develop/c: ")
    assert [p["task"] for p in e.value.data["problems"]] == ["develop/c", "develop/c"]
    assert e.value.data["project"] == "p"
    assert load_project(pdir).template_review  # nothing changed
    (pdir / "tasks/develop/c.md").write_text(task("machines: [n1]\n"), encoding="utf-8")
    ops.start_project(pdir)
    assert not load_project(pdir).template_review


def test_the_api_refuses_to_start_a_template_with_an_error(tmp_path):
    """开始运行 and the first 开始 of one of its tasks (which ends the review too): 409 with the problems; the board lists
    them while the template waits."""
    home = run_home(tmp_path, "template_review: true\nmachines: [n1]\n")
    with TestClient(create_app(home)) as client:
        new(client, "a")
        new(client, "b", machines=["n9"])
        board = ok(client.get("/api/projects/r/board"))
        assert [(p["task"], p["level"]) for p in board["preflight"]] == [("develop/b", "error")]
        r = client.post("/api/projects/r/start")
        assert r.status_code == 409
        assert r.json()["problems"] == board["preflight"] and r.json()["project"] == "r"
        assert "才能开始运行" in r.json()["error"]
        assert "开始运行" in error(client.post(f"{T}/a/actions/start"), 409)
        assert ok(client.get(f"{T}/a"))["status"] == "queued"
        assert ok(client.get("/api/home"))["projects"][0]["template_review"] is True
        # fixed: the board is clear and the project starts
        ok(client.put(f"{T}/b", json={"frontmatter": {"machines": ["n1"]}}))
        assert ok(client.get("/api/projects/r/board"))["preflight"] == []
        assert ok(client.post(f"{T}/a/actions/start"))["task"]["status"] == "running"
        assert ok(client.get("/api/home"))["projects"][0]["template_review"] is False
        ok(client.post(f"{T}/a/actions/stop"))


def test_a_start_after_the_review_is_not_checked_again(make_project, monkeypatch):
    """Once the project runs, 开始 of a task is the task's business: what it cannot read stops it (runs.start_run),
    the check before a start is not run again."""
    pdir = make_project({"develop/a": ""})
    manage(pdir, "develop/a")
    monkeypatch.setattr(runs, "start_run", lambda *a, **kw: None)
    monkeypatch.setattr(preflight, "preflight", lambda *a, **kw: pytest.fail("checked"))
    runs.start(pdir.parent.parent, pdir, "develop/a")


# ---------------------------------------------------------------- 全部纳入托管


def plan_with_drafts(make_project, **more):
    return make_project(
        {
            "plan/r1": "",
            "experiment/e1": task("parent: plan/r1\nmachines: [n1]\n"),
            "experiment/e2": task("parent: plan/r1\nmachines: [n1]\n"),
            "close/r1": task("depends_on: [{task: plan/r1, kind: children}]\n"),
            **more,
        },
        progress={"plan/r1": DONE},
        config="title: t\nmachines: [n1]\n",
    )


def test_join_children_is_refused_on_an_error(make_project):
    pdir = plan_with_drafts(make_project)
    (pdir / "tasks/experiment/e2.md").write_text(
        task("parent: plan/r1\nmachines: [n2]\n"), encoding="utf-8"
    )
    with pytest.raises(AgentSysError) as e:
        ops.join_children(pdir, "plan/r1")
    assert e.value.status == 409 and "才能纳入托管" in str(e.value)
    assert brief(e.value.data["problems"]) == [
        ("experiment/e2", "error", "machines 中的「n2」不在 project.yaml 的 machines 中（有：n1）")
    ]
    assert not any(t.managed for t in load_project(pdir).tasks.values())  # none of them
    (pdir / "tasks/experiment/e2.md").write_text(
        task("parent: plan/r1\nmachines: [n1]\n"), encoding="utf-8"
    )
    assert sorted(ops.join_children(pdir, "plan/r1")) == [
        "close/r1",
        "experiment/e1",
        "experiment/e2",
    ]


def test_join_children_checks_only_the_drafts(make_project):
    """Another task's problem is not the drafts' (project.yaml's are)."""
    pdir = plan_with_drafts(make_project, **{"develop/x": task("machines: [n9]\n")})
    assert [p["task"] for p in check(pdir)] == ["develop/x"]
    assert len(ops.join_children(pdir, "plan/r1")) == 3


def test_the_api_lists_what_join_children_would_refuse(make_project):
    pdir = plan_with_drafts(make_project)
    (pdir / "tasks/experiment/e1.md").write_text(
        task("parent: plan/r1\nmachines: [n1]\n" + checks_of(f"{PY}.perfcheck plna x")),
        encoding="utf-8",
    )
    with TestClient(create_app(pdir.parent.parent)) as client:
        detail = ok(client.get("/api/projects/p/tasks/plan/r1"))
        assert [p["task"] for p in detail["drafts_problems"]] == ["experiment/e1"]
        r = client.post("/api/projects/p/tasks/plan/r1/actions/join_children")
        assert r.status_code == 409 and r.json()["problems"] == detail["drafts_problems"]
        assert ok(client.get("/api/projects/p/tasks/experiment/e1"))["status"] == "draft"
        assert ok(client.get("/api/projects/p/tasks/close/r1"))["drafts_problems"] == []


# ---------------------------------------------------------------- agent-sys-v2 check


def test_the_command_line(make_project, capsys, monkeypatch):
    pdir = make_project(
        {"develop/a": "", "develop/b": task("depends_on: [develop/aa]\n")},
        config="title: t\n",
    )
    home = pdir.parent.parent
    assert cli.main(["check", "p", "--home", str(home)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "error  develop/b: depends_on 中的任务「develop/aa」不存在",
        "       → 最接近的已有任务是「develop/a」：是不是写错了？",
    ]
    # by its directory (what a run has: $AGENT_SYS_PROJECT), as a run writes nothing
    (pdir / "tasks/develop/b.md").write_text(task("depends_on: [develop/a]\n"), encoding="utf-8")
    assert cli.main(["check", str(pdir)]) == 0
    assert capsys.readouterr().out == "没有发现问题\n"
    monkeypatch.chdir(pdir)  # "." from the project directory
    assert cli.main(["check", "."]) == 0 and capsys.readouterr().out == "没有发现问题\n"
    # warnings only: exit 0
    manage(pdir, "develop/a", "develop/b")
    set_status(pdir, "develop/a", "cancelled")
    assert cli.main(["check", str(pdir)]) == 0
    assert capsys.readouterr().out.startswith("warn   develop/b: ")
    # a project.yaml that cannot be read, a project that is not there
    (pdir / "project.yaml").write_text("title: [\n", encoding="utf-8")
    assert cli.main(["check", str(pdir)]) == 1
    assert capsys.readouterr().out.startswith("error  project.yaml: p/project.yaml 无法解析")
    assert cli.main(["check", "nope", "--home", str(home)]) == 1
    assert "不存在" in capsys.readouterr().err
    assert not (home / "projects/nope").exists()


def test_a_draft_written_by_an_agent_is_checked(make_project):
    """plan, intake and close draft tasks and run `check` before they finish: drafts are in it."""
    pdir = plan_with_drafts(make_project)
    (pdir / "tasks/experiment/e2.md").write_text(
        task("parent: plan/r1\nmachines: [n3]\n"), encoding="utf-8"
    )
    assert load_project(pdir).tasks["experiment/e2"].status == "draft"
    assert [p["task"] for p in check(pdir)] == ["experiment/e2"]
    assert read_state(pdir / "workspaces/experiment/e2") is None


# ---------------------------------------------------------------- what the independent review found


def test_a_path_that_cannot_be_looked_at_is_not_checked(make_project):
    """A name too long, a directory you cannot read: neither a problem nor a crash (the board and 开始运行 run it)."""
    long = "x" * 300
    pdir = make_project(
        {
            "develop/a": task(checks_of(f"bash $AGENT_SYS_PROJECT/{long}/run.sh")),
            "develop/b": task(checks_of("bash $AGENT_SYS_PROJECT/locked/inner/run.sh")),
        }
    )
    (pdir / "locked/inner").mkdir(parents=True)
    (pdir / "locked").chmod(0)
    try:
        assert check(pdir) == []
    finally:
        (pdir / "locked").chmod(0o755)


def test_joining_one_draft_is_checked_too(make_project):
    """纳入托管 of a single draft would get round 全部纳入托管's refusal."""
    pdir = plan_with_drafts(make_project)
    (pdir / "tasks/experiment/e2.md").write_text(
        task("parent: plan/r1\nmachines: [n2]\n"), encoding="utf-8"
    )
    with pytest.raises(AgentSysError) as e:
        ops.join(pdir, "experiment/e2")
    assert e.value.status == 409 and e.value.data["problems"][0]["task"] == "experiment/e2"
    assert load_project(pdir).tasks["experiment/e2"].status == "draft"
    ops.join(pdir, "experiment/e1")  # the other one is fine
    assert load_project(pdir).tasks["experiment/e1"].status == "queued"
    # a task with work done is joined as it is: it does not start by itself
    (pdir / "workspaces/experiment/e2").mkdir(parents=True, exist_ok=True)
    (pdir / "workspaces/experiment/e2/progress.md").write_text(DONE, encoding="utf-8")
    ops.join(pdir, "experiment/e2")


def test_a_drafts_checks_are_its_types_when_it_has_none(make_project):
    """A draft without checks gets its type's when it is joined: those are the ones to check."""
    pdir = plan_with_drafts(make_project)
    (pdir / "task_types").mkdir()
    (pdir / "task_types/experiment.md").write_text(
        "---\n" + checks_of(f'{PY}.ledger verfy "$AGENT_SYS_WORKSPACE/results"') + "---\n",
        encoding="utf-8",
    )
    problems = check(pdir)
    assert [(p["task"], p["fix"]) for p in problems] == [
        ("experiment/e1", "是不是「verify」？"),
        ("experiment/e2", "是不是「verify」？"),
    ]
    with pytest.raises(AgentSysError):
        ops.join_children(pdir, "plan/r1")
    # a draft that writes its own checks keeps them; a managed task has what it has
    (pdir / "tasks/experiment/e1.md").write_text(
        task("parent: plan/r1\nmachines: [n1]\nchecks: [echo ok]\n"), encoding="utf-8"
    )
    manage(pdir, "experiment/e2")
    assert check(pdir) == []
    # a type whose checks cannot be read
    (pdir / "tasks/experiment/e3.md").write_text(task("parent: plan/r1\n"), encoding="utf-8")
    (pdir / "task_types/experiment.md").write_text(
        "---\nchecks: [{name: x}]\n---\n", encoding="utf-8"
    )
    [p] = check(pdir)
    assert p["task"] == "experiment/e3" and p["fix"] == "改正 task_types/experiment.md 的 checks"


def test_two_cycles_through_one_task_are_both_told(make_project):
    pdir = make_project(
        {
            "develop/a": task("depends_on: [develop/b]\n"),
            "develop/b": task("depends_on: [develop/a, develop/c]\n"),
            "develop/c": task("depends_on: [develop/b]\n"),
        }
    )
    assert sorted(p["message"].rsplit("：", 1)[1] for p in check(pdir)) == [
        "develop/a → develop/b → develop/a",
        "develop/c → develop/b → develop/c",
    ]


def test_a_start_by_hand_is_checked_once(make_project, monkeypatch):
    """开始 of a template's task: the check runs before the run starts, not again after it (a run that started is not
    refused)."""
    pdir = template_project(make_project)
    calls = []
    real = preflight.preflight
    monkeypatch.setattr(preflight, "preflight", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    monkeypatch.setattr(runs, "start_run", lambda *a, **kw: None)
    runs.start(pdir.parent.parent, pdir, "develop/a")
    assert len(calls) == 1 and not load_project(pdir).template_review


def test_the_command_line_for_one_tasks_drafts(make_project, capsys):
    """--drafts-of: what 全部纳入托管 on that task would refuse (a plan checks its own experiments)."""
    pdir = plan_with_drafts(make_project, **{"develop/x": task("machines: [n9]\n")})
    assert cli.main(["check", str(pdir)]) == 1
    capsys.readouterr()
    assert cli.main(["check", str(pdir), "--drafts-of", "plan/r1"]) == 0
    assert capsys.readouterr().out == "没有发现问题\n"
    (pdir / "tasks/experiment/e1.md").write_text(
        task("parent: plan/r1\nmachines: [n7]\n"), encoding="utf-8"
    )
    assert cli.main(["check", str(pdir), "--drafts-of", "plan/r1"]) == 1
    assert capsys.readouterr().out.startswith("error  experiment/e1: ")
    assert cli.main(["check", str(pdir), "--drafts-of", "plan/zz"]) == 1
    assert "plan/zz" in capsys.readouterr().out
