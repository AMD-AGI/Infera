"""skills.py: which skills a project's tasks are offered, and where the index says they come from."""

import pytest

from agent_sys_v2 import launch, skills
from agent_sys_v2.ops import new_task
from agent_sys_v2.scan import load_project


@pytest.fixture
def shipped(tmp_path, monkeypatch):
    root = tmp_path / "shipped"
    for name in ("infera-bench", "infera-deploy"):
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: shipped {name}\n---\n"
        )
    monkeypatch.setattr(skills, "SHIPPED", root)
    return root


def own_skill(pdir, name):
    (pdir / "skills" / name).mkdir(parents=True)
    (pdir / "skills" / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: own {name}\n---\n"
    )


def test_one_name_is_a_list_of_one(shipped, make_project):
    p = load_project(make_project({}, config="title: t\nskills: infera-deploy\n"))
    assert [s["name"] for s in skills.available(p)] == ["infera-deploy"]


def test_a_project_skill_takes_the_place_of_a_shipped_one(shipped, make_project):
    pdir = make_project({})
    own_skill(pdir, "infera-bench")
    p = load_project(pdir)
    assert [(s["name"], s["description"]) for s in skills.available(p)] == [
        ("infera-deploy", "shipped infera-deploy"),
        ("infera-bench", "own infera-bench"),
    ]
    assert skills.source(p) == f"{shipped}, {pdir / 'skills'}"


def test_the_source_is_where_the_skills_are(shipped, make_project):
    pdir = make_project({"develop/a": ""}, config="title: t\nskills: []\n")
    own_skill(pdir, "mine")
    p = load_project(pdir)
    parts, _text = launch.system(pdir.parents[1], p, p.tasks["develop/a"])
    assert [x["path"] for x in parts if x["kind"] == "skills"] == [str(pdir / "skills")]


def test_unknown_names_are_listed_in_launch_md(shipped, make_project, tmp_path):
    pdir = make_project(
        {}, config="title: t\nskills: [infera-bench, nosuch, mine]\n", types=["develop"]
    )
    own_skill(pdir, "mine")
    new_task(pdir, "develop", "a", "做 A", queue=True)
    p = load_project(pdir)
    assert skills.unknown(p) == ["nosuch"]
    home, a = pdir.parents[1], p.tasks["develop/a"]
    parts, text = launch.system(home, p, a)
    launch.prepare(
        home,
        p,
        a,
        run_dir=tmp_path / "run",
        kind="sdk",
        parts=parts,
        system_text=text,
        message="m",
        resume=False,
        max_budget_usd=1,
        cost_start_usd=0.0,
        window="a",
        cli_path="/x/claude",
    )
    assert (
        "project.yaml 的 skills 中这些名字不存在：nosuch\n"
        in (tmp_path / "run/launch.md").read_text()
    )


def test_skills_index(tmp_path, make_project, monkeypatch):
    shipped = tmp_path / "shipped"
    for name, desc in (
        ("infera-bench", "measure a point"),
        ("infera-deploy", "bring a kit up"),
        ("draft", ""),
    ):
        (shipped / name).mkdir(parents=True)
        (shipped / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n"
        )
    monkeypatch.setattr(skills, "SHIPPED", shipped)
    pdir = make_project(
        {"develop/a": ""}, config="title: t\nskills: [infera-deploy, infera-bench, nosuch]\n"
    )
    (pdir / "skills/own").mkdir(parents=True)
    (pdir / "skills/own/SKILL.md").write_text("---\ndescription: this project's own\n---\n")
    p = load_project(pdir)
    assert [s["name"] for s in skills.available(p)] == ["infera-deploy", "infera-bench", "own"]
    index = skills.index(p)
    assert (
        index.splitlines()[0]
        == f"- infera-deploy — bring a kit up `{shipped}/infera-deploy/SKILL.md`"
    )
    # no list: every shipped skill with a description
    (pdir / "project.yaml").write_text("title: t\n")
    p = load_project(pdir)
    assert [s["name"] for s in skills.available(p)] == ["infera-bench", "infera-deploy", "own"]
    parts, text = launch.system(tmp_path / "home", p, p.tasks["develop/a"])
    assert [x["title"] for x in parts if x["kind"] == "skills"] == ["skills 索引"]
    assert "一行一个 skill" in text and "infera-deploy — bring a kit up" in text


def test_the_shipped_skills_are_indexed():
    assert sorted(s["name"] for s in skills.found(skills.SHIPPED)) == [
        "infera-bench",
        "infera-deploy",
        "infera-diagnose",
        "infera-experiment",
        "infera-hygiene",
        "infera-profile",
        "infera-quality-gate",
    ]
