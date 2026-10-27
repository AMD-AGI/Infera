"""Field checks of the infera-perf template's round (intake → survey ∥ baseline → profile → plan → experiments → close):
what a task of the round has to leave behind before it counts as done, as its acceptance check runs it.

  "$AGENT_SYS_PYTHON" -m agent_sys_v2.perfcheck intake|survey|plan|close "$AGENT_SYS_TASK"

It runs in the project directory (the checks do), or the one $AGENT_SYS_PROJECT names, and prints one line per item
(ok / FAIL); exit 1 when one fails. The round is the number in the task's name: intake/r2, plan/r2, close/r2.

  intake   project.yaml parses and has goal, machines, round, ledger (primary, x, working_point); rules/project.md
           has no placeholder left; the questionnaire was asked (questions*.md); this round's tasks that need machines
           have declared ones; from round 2, this round's plan is drafted, unless report.md says 项目结束 (the
           questionnaire's last question ends the project)
  survey   a reference result (role reference) in results/, or report.md says there is no public reference
  plan     every subtask names declared machines and expect: {delta_pct, level L0-L5, falsify}; a task waits for the
           plan's subtasks ({task: <plan>, kind: children}) and has machines
  close    stack.json says version and sealed; a sealed round has its curve (that series at 2+ concurrencies) and is
           the version and the ledger's baseline series in project.yaml; the next round's intake is drafted, after
           this task

The messages are for the agent, which gets them back when a check fails.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from . import ledger, machines
from .model import AgentSysError, plain, read_text
from .scan import load_project

PLACEHOLDER = "<待 intake 填写>"  # the template's rules/project.md marks what intake fills in
LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")
MACHINE_TYPES = ("baseline", "profile", "experiment", "close")  # the types that run on nodes
NO_REFERENCE = re.compile(r"无公开对标|no public reference", re.I)


def round_of(name):
    """r2 / r2-spf / r12_x -> 2 / 2 / 12; None when the name has no round."""
    m = re.match(r"r(\d+)(?![0-9A-Za-z])", name)
    return int(m[1]) if m else None


class Report:
    def __init__(self):
        self.lines, self.failed = [], 0

    def item(self, ok, text):
        self.lines.append(f"{'ok  ' if ok else 'FAIL'}  {text}")
        self.failed += not ok
        return ok


def machine_problems(project, t):
    """What is wrong with a task's machines: none, or names project.yaml does not declare."""
    if not t.machines:
        return "no machines in its frontmatter"
    known = {m["name"] for m in machines.declared(project.config)}
    if unknown := [m for m in t.machines if m not in known]:
        return f"machines not in project.yaml: {', '.join(unknown)}"
    return ""


def check_intake(project, t, r):
    n = round_of(t.name)
    cfg = plain(project.config)
    r.item(bool(str(cfg.get("goal") or "").strip()), "project.yaml goal is set")
    r.item(bool(machines.declared(project.config)), "project.yaml machines lists the nodes")
    if n is not None:
        r.item(cfg.get("round") == n, f"project.yaml round is {n} (it is {cfg.get('round')!r})")
    lc = ledger.settings(project)
    r.item(bool(lc["primary"]), "project.yaml ledger.primary (Y) is set")
    r.item(bool(lc["x"]), "project.yaml ledger.x (X of the curves) is set")
    r.item(bool(lc["working_point"]), "project.yaml ledger.working_point is {conc: C} or {x: X}")
    rules = read_text(project.path / "rules" / "project.md")
    r.item(
        bool(rules.strip()) and PLACEHOLDER not in rules,
        f"rules/project.md is written (no {PLACEHOLDER} left)",
    )
    r.item(any(t.workspace.glob("questions*.md")), "the questionnaire was asked (questions.md)")
    if n is None:
        return
    for other in project.tasks.values():
        if (
            other.type in MACHINE_TYPES
            and round_of(other.name) == n
            and other.status not in ("done", "cancelled")
        ):
            problem = machine_problems(project, other)
            r.item(
                not problem, f"{other.id} has its machines" + (f": {problem}" if problem else "")
            )
    if n >= 2:
        ends = re.search(r"项目结束|the project ends", read_text(t.workspace / "report.md"), re.I)
        r.item(
            f"plan/r{n}" in project.tasks or bool(ends),
            f"this round's plan is drafted (tasks/plan/r{n}.md), or report.md says 项目结束",
        )


def check_survey(project, t, r):
    rows, _bad = ledger.rows(project)
    refs = [x for x in rows if x["task"] == t.id and x["role"] == "reference"]
    said = NO_REFERENCE.search(read_text(t.workspace / "report.md"))
    r.item(
        bool(refs or said),
        f"results/ has reference curves ({len(refs)} points), or report.md says 无公开对标",
    )


def check_plan(project, t, r):
    children = [project.tasks[c] for c in project.children(t.id)]
    r.item(True, f"{len(children)} subtasks: {', '.join(c.id for c in children) or 'none'}")
    for c in children:
        problem = machine_problems(project, c)
        r.item(not problem, f"{c.id} has its machines" + (f": {problem}" if problem else ""))
        expect = plain(c.fm.get("expect"))
        expect = expect if isinstance(expect, dict) else {}
        delta = expect.get("delta_pct")
        r.item(
            ledger.number(delta)
            or (
                isinstance(delta, list) and len(delta) == 2 and all(ledger.number(v) for v in delta)
            ),
            f"{c.id} expect.delta_pct is a number or [low, high]",
        )
        r.item(
            expect.get("level") in LEVELS,
            f"{c.id} expect.level is one of {'/'.join(LEVELS)} (the fast check it uses)",
        )
        r.item(
            bool(str(expect.get("falsify") or "").strip()),
            f"{c.id} expect.falsify says what result would refute it",
        )
    closers = [project.tasks[e.down] for e in project.downstream(t.id) if e.kind == "children"]
    if r.item(
        bool(closers),
        "a task waits for the plan's subtasks (depends_on {task: " + t.id + ", kind: children})",
    ):
        for c in closers:
            problem = machine_problems(project, c)
            r.item(not problem, f"{c.id} has its machines" + (f": {problem}" if problem else ""))


def check_close(project, t, r):
    n = round_of(t.name)
    try:
        stack = json.loads(read_text(t.workspace / "stack.json") or "null")
    except json.JSONDecodeError as e:
        stack = None
        r.item(False, f"stack.json is JSON: {e}")
    if not r.item(
        isinstance(stack, dict) and "version" in stack and isinstance(stack.get("sealed"), bool),
        'stack.json is {"version": ..., "sealed": true|false, ...}',
    ):
        stack = {}
    if stack.get("sealed"):
        version = str(stack.get("version") or "")
        rows, _bad = ledger.rows(project)
        points = [
            x
            for x in rows
            if x["task"] == t.id
            and x["series"] == version
            and x["status"] == "valid"
            and x["conc"] is not None
        ]
        r.item(
            len({x["conc"] for x in points}) >= 2,
            f"results/ has the integration curve of series {version!r} (2+ concurrencies, it has "
            f"{len({x['conc'] for x in points})})",
        )
        cfg = plain(project.config)
        r.item(cfg.get("version") == version, f"project.yaml version is {version!r}")
        r.item(
            ledger.settings(project)["baseline_series"] == version,
            f"project.yaml ledger.baseline_series is {version!r}",
        )
    if n is not None:
        nxt = project.tasks.get(f"intake/r{n + 1}")
        r.item(nxt is not None, f"the next round's intake is drafted (tasks/intake/r{n + 1}.md)")
        if nxt is not None:
            r.item(
                any(e.up == t.id and e.blocking for e in project.upstream(nxt.id)),
                f"intake/r{n + 1} depends on {t.id}",
            )


CHECKS = {"intake": check_intake, "survey": check_survey, "plan": check_plan, "close": check_close}


def run(kind, task_id, project_dir):
    """(passed, lines)."""
    r = Report()
    try:
        project = load_project(project_dir)
    except AgentSysError as e:
        r.item(False, f"the project can be read: {e}")
        return False, r.lines
    if task_id not in project.tasks:
        r.item(False, f"task {task_id} exists")
        return False, r.lines
    CHECKS[kind](project, project.tasks[task_id], r)
    return r.failed == 0, r.lines


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in CHECKS:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    ok, lines = run(argv[0], argv[1], Path(os.environ.get("AGENT_SYS_PROJECT") or "."))
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
