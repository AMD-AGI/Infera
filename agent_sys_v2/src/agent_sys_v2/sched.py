"""The scheduler: every 2 s, right after the scan, the queued tasks that wait for nothing start, the
longest waiting first, as long as their project and agent_sys as a whole have a free slot. It runs in the server
(server.Store.poll): with no server nothing is scheduled, and a restarted server goes on from the files.

Why a queued task waits, the first reason that holds (shown on its card):
  项目已归档 > prompt 有错误 > 阻塞类上游未完成 > 等子任务 > 等你审核模板 > 已关闭自动开始
                                                     scan.queued_reason: the project alone
  > 机器被占用 (machines, with their holders)         plan(): across projects
  > 项目名额已满 > 全局名额已满 (concurrency, with scope and limit)
A running task holds a slot, also while it waits in the background or is taken over in a terminal; an unmanaged one
(progress 进行中) does not. A task holds its machines while it runs and while it waits for you (machines.py). Runs
start through runs.start_run, as 开始 does, and only on this host. A start that fails puts the task in 待确认: on its
budget (budget), else 运行中断 with the reason; it is not tried again.
"""

from __future__ import annotations

import logging

from . import machines, runs
from .i18n import _
from .model import GLOBAL_DEFAULTS, AgentSysError, load_config, now_iso, positive, write_state

log = logging.getLogger("agent_sys")


def global_limit(home):
    return positive(load_config(home).get("max_running"), GLOBAL_DEFAULTS["max_running"])


def plan(projects, limit):
    """The queued tasks that can start now, [(project, task)], the longest waiting first. A task whose machines are
    held gets detail machines; one that waits only for a slot gets concurrency (the project's limit before the global
    one). A task that can start is counted as running, and as holding its machines, for the tasks after it."""
    running = {p.id: p.running() for p in projects}
    total = sum(running.values())
    held = machines.holders(projects)
    ready = sorted(
        (
            (p, t)
            for p in projects
            for t in p.tasks.values()
            if t.managed
            and t.status == "queued"
            and t.detail.get("kind") in (None, "concurrency", "machines")
        ),
        key=lambda pt: (pt[1].since, pt[0].id, pt[1].id),
    )
    go = []
    for p, t in ready:
        if busy := machines.wait_detail(p, t, held):
            t.detail = busy
        elif running[p.id] >= p.max_running:
            t.detail = {"kind": "concurrency", "scope": "project", "limit": p.max_running}
        elif total >= limit:
            t.detail = {"kind": "concurrency", "scope": "global", "limit": limit}
        else:
            t.detail = {}
            go.append((p, t))
            running[p.id] += 1
            total += 1
            for name in t.machines:
                held.setdefault(name, []).append((p, t))
    return go


def annotate(projects, limit):
    """Give the queued tasks of a fresh scan their machine and concurrency reasons. Returns the tasks whose reason changed,
    [(project id, task id)]: a slot freed in one project changes the cards of another."""
    queued = [
        (p, t) for p in projects for t in p.tasks.values() if t.managed and t.status == "queued"
    ]
    before = [dict(t.detail) for _, t in queued]
    plan(projects, limit)
    return [(p.id, t.id) for (p, t), b in zip(queued, before) if t.detail != b]


def start(home, pdir, tid):
    """Start a task the plan picked if a fresh scan still says it waits for nothing. True if its run started."""
    try:
        project, t = runs.fresh(pdir, tid)
    except AgentSysError:  # deleted meanwhile
        return False
    if not t.managed or t.status != "queued" or t.detail:
        return False
    try:
        runs.start_run(home, project, t)
        return True
    except Exception as e:  # noqa: BLE001 - whatever stopped it goes on the task's card
        log.exception("auto start of %s/%s failed", project.id, tid)
        project, t = runs.fresh(pdir, tid)
        detail = (
            {"kind": "budget"}
            if runs.budget_left(t) <= 0
            else {"kind": "interrupted", "error": _("自动开始失败：{error}", error=e)}
        )
        write_state(
            t.workspace, {**t.state, "status": "needs_you", "detail": detail, "since": now_iso()}
        )
        return False


def tick(store):
    """One round: start what plan() picks, one task at a time under the write lock (the lock is let go in between,
    and each start rescans), until nothing more can start. Returns the tasks started, [(project id, task id)]."""
    started, tried = [], set()
    while True:
        with store.lock:
            todo = [
                (p, t)
                for p, t in plan(store.live(), global_limit(store.home))
                if (p.id, t.id) not in tried
            ]
            if not todo:
                return started
            p, t = todo[0]
            tried.add((p.id, t.id))
            if start(store.home, p.path, t.id):
                started.append((p.id, t.id))
            store.sync(also=[(p.id, t.id)])
