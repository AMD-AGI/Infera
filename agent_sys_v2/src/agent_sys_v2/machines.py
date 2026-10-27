"""Machines (design §3.2): the nodes a project may use, which task holds which, and who holds one idle.

  project.yaml   machines: [{name: smc-a16-19, gpus: 8, note: MI300X}, smc-a16-20]
  task           machines: [smc-a16-19]

A managed task holds the machines in its frontmatter while it runs and while it waits for you (its deployment is
usually still up), until it is in review, done or cancelled; a done task that runs again (a follow-up question)
holds them again. The scheduler starts no task whose machines another task holds (sched.plan), and no run starts on
them by hand either: 开始, a message, 继续, 追加预算 or a takeover (runs.start_run, check_free). Machines are named by host
name, so the rule holds across projects. A holder that has waited for you longer than IDLE_MINUTES is shown as
holding its machines idle: the node is up and nobody is using it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .i18n import _
from .model import AgentSysError, plain

HOLDING = ("running", "needs_you")
IDLE_MINUTES = 30


def declared(config):
    """project.yaml machines -> [{name, gpus, note}], names unique, in the order written."""
    items = plain(config.get("machines")) or []
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        m = {"name": item} if isinstance(item, str) else item if isinstance(item, dict) else {}
        name = str(m.get("name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            gpus = m.get("gpus")
            out.append(
                {
                    "name": name,
                    "gpus": gpus if isinstance(gpus, int) and not isinstance(gpus, bool) else None,
                    "note": str(m.get("note") or ""),
                }
            )
    return out


def task_machines(fm):
    """A task's frontmatter machines -> [name]: a list, or one name."""
    value = plain(fm.get("machines"))
    items = value if isinstance(value, list) else [value] if value else []
    return list(dict.fromkeys(str(x).strip() for x in items if str(x or "").strip()))


def holds(t):
    return t.managed and t.status in HOLDING and bool(t.machines)


def holders(projects):
    """{machine name: [(project, task)]} over the tasks that hold machines; more than one only if the rule was broken
    (a machine added to a task that waits for you, say), and then all of them are shown."""
    out = {}
    for p in projects:
        for t in p.tasks.values():
            if holds(t):
                for name in t.machines:
                    out.setdefault(name, []).append((p, t))
    return out


def minutes_since(iso_time, now=None):
    try:
        then = datetime.strptime(iso_time or "", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return 0
    return max(0, int(((now or datetime.now(timezone.utc)) - then).total_seconds() // 60))


def holder_json(p, t, now=None):
    waited = minutes_since(t.since, now) if t.status == "needs_you" else 0
    return {
        "project": p.id,
        "task": t.id,
        "title": t.title,
        "status": t.status,
        "detail": t.detail,
        "since": t.since,
        "idle_minutes": waited if waited >= IDLE_MINUTES else 0,
    }


def table(projects, only=None, now=None):
    """The machines as the board (one project: only) or the home page (all projects) shows them:
    [{name, gpus, note, projects, holders}], holders [] when free. A held machine nobody declared is listed too."""
    held = holders(projects)
    rows = {}
    for p in projects:
        if only is not None and p.id != only.id:
            continue
        for m in declared(p.config):
            row = rows.setdefault(m["name"], {**m, "projects": []})
            row["projects"].append(p.id)
    for name, pts in held.items():
        ids = list(dict.fromkeys(p.id for p, _t in pts))
        if only is None or only.id in ids:
            rows.setdefault(name, {"name": name, "gpus": None, "note": "", "projects": ids})
    for name, row in rows.items():
        row["holders"] = [holder_json(p, t, now) for p, t in held.get(name, [])]
    return list(rows.values())


def busy(project, t, held):
    """t's machines that another task holds: [(name, holder's project, holder)]. held: holders()."""
    return [
        (name, p, h)
        for name in t.machines
        for p, h in held.get(name, [])
        if (p.id, h.id) != (project.id, t.id)
    ]


def check_free(projects, project, t):
    """Refuse a run of t while another task holds one of its machines (runs.start_run; the scheduler waits)."""
    taken = busy(project, t, holders(projects))
    if taken:
        raise AgentSysError(
            _(
                "机器 {machines} 正被 {tasks} 占用：等它完成或取消，或修改本任务的机器",
                machines=_("、").join(name for name, _p, _h in taken),
                tasks=_("、").join(sorted({f"{p.id}/{h.id}" for _n, p, h in taken})),
            ),
            409,
        )


def wait_detail(project, t, held):
    """The detail of a queued task whose machines are held: {kind: machines, machines: [{name, project, task, title}]}
    or None when they are all free."""
    taken = [
        {"name": name, "project": p.id, "task": h.id, "title": h.title}
        for name, p, h in busy(project, t, held)
    ]
    return {"kind": "machines", "machines": taken} if taken else None
