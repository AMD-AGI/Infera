"""agent_sys server: the REST API and the change stream for the web UI, which it serves too.

  agent-sys-v2 serve    (starts uvicorn on /tmp/agent_sys-$UID/agent_sys.sock in tmux -L $AGENT_SYS_TMUX; see cli.py)

Every project is scanned once into memory and rescanned when it changes: the Watcher polls stat every 2 s, and
each write through the API syncs right away. Reads (home page, board, task detail) come from memory. Writes go
through one lock, because an operation checks on a fresh scan and then writes: two interleaved requests would
get round the check. Paths in the API are relative to AGENT_SYS_HOME, errors are {"error": <text for the user>}.
Runs are started, stopped and taken over through runs.py; a 0.5 s loop (runs.Monitor) follows the running tasks'
events and moves a task on when its run is over. The scheduler (sched.py) runs after each 2 s scan and starts the
queued tasks that wait for nothing, their machines free. Each rescan reads the project's ledger (ledger.py) for the
board's head and the cards' result badges; the rules and knowledge pages are knowledge.py, and a knowledge library's
INDEX.md is made again whenever its entries change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator
from starlette.exceptions import HTTPException

from . import checks, files, i18n, knowledge, launch, ledger, machines, ops, questions, runs, sched
from .events import read_event, read_events
from .i18n import _
from .model import (
    PROJECTS,
    AgentSysError,
    Project,
    check_scope,
    project_path,
    read_text,
    runs_dir,
)
from .scan import Watcher, load_project, preview_edges, project_dir, project_ids

WEB = Path(__file__).resolve().parent / "web"
POLL_SECONDS = 2
# every project is scanned again this often: reminders that come with time alone
FULL_SCAN_SECONDS = 600
COALESCE_SECONDS = 0.15  # changes this close together go out as one push
# The UI's scripts come only from here; components set some styles inline; images may be data: or blob:; no other
# site may show the UI in a frame.
CSP = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"
# The Host names the server answers to (more in $AGENT_SYS_HOSTS, comma-separated): a page that got another name to
# point here (DNS rebinding) is refused.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
RUN_POLL = 0.5  # running tasks' events and run records
log = logging.getLogger("agent_sys")


class Store:
    """The scans of all projects, kept current. listeners get {"project", "task"} for each change
    (task None = the project itself), and {"project", "task", "run", "seq"} when a running task's run has new events
    (seq: the last); they may be called from any thread."""

    def __init__(self, home):
        self.home = Path(home)
        self.lock = threading.RLock()
        self.projects: dict[str, Project | AgentSysError] = {}  # a project that cannot be read: why
        self.listeners = set()
        self.watcher = Watcher(self.home)
        self.monitor = runs.Monitor(self)
        self.scanned = time.monotonic()  # when every project was last scanned
        # (pid, None): every index too, for the entries written while no server ran (by hand)
        self.sync(also=[(pid, None) for pid in ["", *project_ids(self.home)]])

    def knowledge_dir(self, pid):
        return knowledge.knowledge_dir(self.home, pid or None)

    def scan(self, pid):
        """A fresh scan of a project with what its runs are doing and its ledger; why it failed; None if it is gone."""
        pdir = project_dir(self.home, pid)
        if not (pdir / "project.yaml").is_file():
            return None
        try:
            p = load_project(pdir)
            self.monitor.annotate(p)  # running tasks: what their runs are doing
            full = ledger.ledger(p)  # the board's head and the cards' result badges
            p.ledger = ledger.head(full)
            for tid, brief in ledger.task_briefs(full).items():
                p.tasks[tid].results = brief
            return p
        except AgentSysError as e:
            return e
        # a file that changed under the scan, a bad moment on NFS: shown on the project's row, and scanned again at
        # the next poll
        except Exception as e:
            error = AgentSysError(_("扫描项目失败：{error}", error=repr(e)), 500)
            if str(self.projects.get(pid)) != str(error):
                log.exception("scanning project %s failed", pid)
            self.watcher.forget(pid)
            return error

    def sync(self, also=()):
        """Rescan the projects that changed on disk, plus `also` (changes the Watcher does not look at, such as
        any file of a workspace written through the API), and tell the listeners. The new scans get their queue
        reasons before they are published: a read without the lock never sees one half done."""
        with self.lock:
            changes = set(self.watcher.poll()) | set(also)
            # agent_sys.yaml or AGENT_SYS_HOME's rules: in another language, the cards' reminders are made again
            if ("", None) in changes and i18n.LANG != i18n.use(self.home):
                changes |= {(pid, None) for pid in self.projects}
            projects = dict(self.projects)
            for pid in {pid for pid, _ in changes if pid}:
                projects[pid] = self.scan(pid)
            projects = {pid: p for pid, p in projects.items() if p is not None}
            if changes:  # a slot taken or freed anywhere: which queued tasks wait for one
                live = [p for p in projects.values() if isinstance(p, Project)]
                changes |= set(sched.annotate(live, sched.global_limit(self.home)))
            self.projects = projects
            # entries edited by hand, or added: a new index
            for pid in {pid for pid, tid in changes if tid is None}:
                knowledge.refresh_index(self.knowledge_dir(pid))
        self.notify(changes)

    def poll(self):
        """Every 2 s: rescan what changed on disk (every project now and then: a progress.md not updated for hours,
        today's cost), then start the queued tasks that can start."""
        everything = time.monotonic() - self.scanned > FULL_SCAN_SECONDS
        if everything:
            self.scanned = time.monotonic()
        self.sync(also=[(pid, None) for pid in self.projects] if everything else ())
        sched.tick(self)

    def notify(self, changes):
        """Tell the listeners, without a rescan (a running task's sub-state changed in memory)."""
        for pid, tid in sorted(set(changes), key=lambda c: (c[0], c[1] or "")):
            for fn in list(self.listeners):
                fn({"project": pid, "task": tid})

    def tell(self, change):
        for fn in list(self.listeners):
            fn(change)

    def write(self, fn, *args, also=(), **kw):
        """Run a write operation under the lock, then sync, so the reply and the next read agree with the disk."""
        with self.lock:
            result = fn(*args, **kw)
            self.sync(also)
        return result

    def project(self, pid) -> Project:
        p = self.projects.get(pid)
        if p is None:
            raise AgentSysError(_("项目「{project}」不存在", project=pid), 404)
        if isinstance(p, AgentSysError):
            raise p
        return p

    def live(self):
        return [p for p in self.projects.values() if isinstance(p, Project)]


# ---------------------------------------------------------------- requests


def allowed_hosts():
    """LOCAL_HOSTS and the names in $AGENT_SYS_HOSTS."""
    more = os.environ.get("AGENT_SYS_HOSTS", "").split(",")
    return {*LOCAL_HOSTS, *(h.strip().lower() for h in more if h.strip())}


def refusal(request, hosts):
    """Why a request is refused, or "". The Host must be one of hosts (DNS rebinding); a request that changes
    something must come from this server's own page: its Origin the Host, else Sec-Fetch-Site same-origin or none
    (typed by you). Neither header: not a browser (curl), let through."""
    host = request.headers.get("host", "")
    if re.sub(r":\d+$", "", host).lower() not in hosts:
        return _(
            "不接受 Host「{host}」：请通过 localhost 访问，或将它加入环境变量 AGENT_SYS_HOSTS",
            host=host,
        )
    if request.method in SAFE_METHODS:
        return ""
    origin, site = request.headers.get("origin"), request.headers.get("sec-fetch-site")
    if origin is not None:
        same = urlsplit(origin).netloc.lower() == host.lower()
    else:
        same = site in (None, "same-origin", "none")
    return "" if same else _("拒绝来自其他网站的请求")


class Body(BaseModel):
    """A request's JSON body: wrong types are refused with 400 (bad_request); fields left out, or null (the UI's
    "none"), take these defaults."""

    @model_validator(mode="before")
    @classmethod
    def nulls_left_out(cls, data):
        return {k: v for k, v in data.items() if v is not None} if isinstance(data, dict) else data


class TaskRef(Body):
    project: str
    task: str


class TaskList(Body):
    tasks: list[TaskRef] = []


class NewProject(Body):
    name: str = ""
    template: str = "blank"
    title: str = ""


class NewType(Body):
    id: str = ""
    review: bool = False


class Lanes(Body):
    order: list[str]


class NewTask(Body):
    type: str = ""
    name: str = ""
    title: str = ""
    prompt: str = ""
    depends_on: list = []
    review: bool = False
    agent: dict | None = None
    mode: str = "queue"  # or draft
    machines: list[str] | None = None
    checks: list | None = None  # None: the type's


class TaskEdit(Body):
    frontmatter: dict | None = None
    prompt: str | None = None
    base_mtime: str = ""
    send_to_agent: bool = False


class Rename(Body):
    type: str = ""
    name: str = ""


class Action(Body):
    usd: float | None = None  # add_budget
    text: str = ""  # add_budget: what the agent is told


class Message(Body):
    text: str = ""


class Answers(Body):
    answers: dict[str, str] = {}
    base_mtime: str = ""


class FileText(Body):
    content: str = ""
    base_mtime: str = ""


class Scoped(Body):
    scope: str = ""  # global, or project with project
    project: str = ""


class Rule(Scoped):
    name: str = ""
    content: str = ""
    base_mtime: str = ""


class Entry(Scoped):
    name: str = ""
    description: str = ""
    type: str = ""
    body: str = ""
    base_mtime: str = ""


class Candidate(Body):
    id: str = ""
    scope: str = "global"
    name: str | None = None  # adopt: the entry, edited or (None) as the candidate has it
    description: str | None = None
    type: str | None = None
    body: str | None = None
    replace: bool = False  # adopt: over an entry of the same name
    file: str = ""  # to_rule
    text: str = ""


class Settings(Scoped):
    values: dict = {}


# ---------------------------------------------------------------- JSON


def task_summary(t):
    """TaskSummary; a managed task to review also says what its last run cost (本轮花费)."""
    out = {**t.to_json(), "last_message": runs.last_message(t)}
    if t.managed and t.status == "in_review":
        out["round_cost_usd"] = runs.round_cost(t)
    return out


def edge_end(p, e, other):
    o = p.tasks[other]
    return {
        "task": other,
        "title": o.title,
        "status": o.status,
        "kind": e.kind,
        "label": e.label,
        "evidence": e.evidence,
        "source": e.source,
        "blocking": e.blocking,
        "satisfied": p.edge_satisfied(e),
    }


def task_detail(store, p, t):
    run_id = t.managed and t.state.get("run_id")
    follow = store.monitor.follows.get(runs_dir(t.workspace) / run_id) if run_id else None
    run_list = runs.list_runs(t, follow) if t.managed else []
    return {
        **t.detail_json(),
        "last_message": runs.last_message(t),
        "upstream": [edge_end(p, e, e.up) for e in p.upstream(t.id)],
        "downstream": [edge_end(p, e, e.down) for e in p.downstream(t.id)],
        "drafts": ops.drafts_of(p, t.id),  # what 全部纳入托管 joins
        "questions": questions.round_questions(t),
        "check_runs": checks.load(runs_dir(t.workspace) / run_id) if run_id else None,
        "project_machines": machines.declared(p.config),
        "runs": run_list,
    }


def project_json(store, pid):
    p = store.projects[pid]
    if isinstance(p, AgentSysError):
        return {
            "id": pid,
            "title": pid,
            "path": project_path(pid),
            "archived": False,
            "error": str(p),
        }
    return p.to_json()


@cache  # the templates come with the code
def templates_brief():
    return [{**t, "types": [ty["id"] for ty in t["types"]]} for t in ops.list_templates()]


def home_json(store):
    """Everything that needs you, across the projects that are not archived."""
    live = [p for p in store.live() if not p.archived]
    tasks = [t for p in live for t in p.tasks.values()]
    pick = lambda status: sorted((t for t in tasks if t.status == status), key=lambda t: t.since)
    archived = sorted(
        (p for p in store.live() if p.archived),
        key=lambda p: str(p.to_json()["archived_at"]),
        reverse=True,
    )
    return {
        "needs_you": [task_summary(t) for t in pick("needs_you")],  # longest waiting first
        "in_review": [task_summary(t) for t in pick("in_review")],
        "unread": [
            task_summary(t)
            for t in sorted((t for t in tasks if t.unread), key=lambda t: t.finished, reverse=True)
        ],
        "running": [task_summary(t) for t in pick("running")],
        "queued": sum(t.status == "queued" for t in tasks),
        "cost_today_usd": round(
            sum(t.cost_today for p in store.live() for t in p.tasks.values()), 4
        ),
        "projects": [
            project_json(store, pid)
            for pid in sorted(store.projects)
            if pid not in {p.id for p in archived}
        ],
        "archived_projects": [p.to_json() for p in archived],
        "machines": [m for m in machines.table(live) if m["holders"]],
        "templates": templates_brief(),
        "projects_path": PROJECTS,  # where a new project is made
    }


def board_json(store, p):
    return {
        "project": p.to_json(),
        "types": [ty.to_json() for ty in p.types],
        "tasks": [task_summary(t) for t in p.tasks.values()],
        "edges": [p.edge_json(e) for e in p.edges],
        "machines": machines.table(store.live(), only=p),
        "running": {
            "count": p.running(),
            "limit": p.max_running,
            "global_count": sum(q.running() for q in store.live()),
            "global_limit": sched.global_limit(store.home),
        },
    }


# ---------------------------------------------------------------- the app


def create_app(home):
    i18n.use(home)
    store = Store(home)
    hosts = allowed_hosts()

    @asynccontextmanager
    async def lifespan(app):
        async def poll(fn, seconds):
            while True:
                await asyncio.sleep(seconds)
                try:
                    await run_in_threadpool(fn)
                except Exception:  # a bad moment on NFS must not end the polling
                    log.exception("polling failed")

        tasks = [
            asyncio.create_task(poll(store.poll, POLL_SECONDS)),
            asyncio.create_task(poll(store.monitor.poll, RUN_POLL)),
        ]
        yield
        for task in tasks:
            task.cancel()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.store = store

    @app.exception_handler(AgentSysError)
    async def refused(request, e: AgentSysError):
        return JSONResponse({"error": str(e)}, status_code=e.status)

    @app.exception_handler(HTTPException)
    async def http_error(request, e: HTTPException):
        text = (
            _("接口不存在：{method} {path}", method=request.method, path=request.url.path)
            if e.status_code == 404
            else str(e.detail)
        )
        return JSONResponse({"error": text}, status_code=e.status_code)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request, e: RequestValidationError):
        where = _("；").join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        return JSONResponse({"error": _("请求格式不正确（{where}）", where=where)}, status_code=400)

    # the run output, boards and scripts are text that packs 3-6 times smaller: what makes a page slow over a tunnel
    # is its size (the SSE streams are left as they are: text/event-stream is not compressed)
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)

    @app.middleware("http")
    async def same_site(request, call_next):
        """Through `ssh -L` the browser reaches the server as localhost: refuse other sites' pages (refusal)."""
        if why := refusal(request, hosts):
            return JSONResponse({"error": why}, status_code=403)
        return await call_next(request)

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", CSP)
        response.headers.setdefault("Cache-Control", "no-cache")
        # one line per page load and per server error, to see which browser reached this server and what failed
        if request.url.path == "/" or response.status_code >= 500:
            ua = request.headers.get("user-agent", "-")
            print(
                f"{time.strftime('%H:%M:%S')} {response.status_code} {request.method} {request.url.path} {ua}",
                flush=True,
            )
        return response

    def task_of(pid, type_, name):
        p = store.project(pid)
        return p, p.task(f"{type_}/{name}")

    def task_write(pid, tid, fn, *args):
        """fn(*args) under the write lock, then sync: (what it returned, the task's summary as it is now)."""
        result = store.write(fn, *args, also=[(pid, tid)])
        return result, task_summary(store.project(pid).task(tid))

    # ---- home and projects

    @app.get("/api/home")
    def home():
        return home_json(store)

    def each_task(body, fn):
        """fn(project, task) for each task of a batch action on the home page: the ones it did ("<project>/<task>"),
        and the ones refused with why."""
        done, errors = [], []
        for ref in body.tasks:
            try:
                p = store.project(ref.project)
                if fn(p, p.task(ref.task)):
                    done.append(f"{ref.project}/{ref.task}")
            except AgentSysError as e:
                errors.append({"task": f"{ref.project}/{ref.task}", "error": str(e)})
        return done, errors

    @app.post("/api/resume")
    def resume_all(body: TaskList):
        """继续全部中断任务: every listed task whose run was interrupted goes on (继续); others are
        left as they are."""
        done, errors = each_task(
            body,
            lambda p, t: store.write(
                runs.resume_interrupted, store.home, p.path, t.id, also=[(p.id, t.id)]
            ),
        )
        return {"resumed": done, "errors": errors}

    @app.post("/api/mark_read")
    def mark_all_read(body: TaskList):
        """全部标为已查看: clear the unread mark of every listed task; one scan per project, one rescan."""
        listed = {}  # project id -> (its directory, [task ids])
        _done, errors = each_task(
            body, lambda p, t: listed.setdefault(p.id, (p.path, []))[1].append(t.id)
        )

        def mark():
            return [
                f"{pid}/{tid}"
                for pid, (pdir, tids) in listed.items()
                for tid in ops.mark_read(pdir, *tids)
            ]

        also = [(pid, tid) for pid, (pdir, tids) in listed.items() for tid in tids]
        return {"marked": store.write(mark, also=also), "errors": errors}

    @app.post("/api/projects")
    def create_project(body: NewProject):
        name = body.name.strip()
        store.write(ops.new_project, store.home, name, body.template, body.title.strip() or None)
        return project_json(store, name)

    @app.post("/api/projects/{pid}/types")
    def create_type(pid: str, body: NewType):
        store.write(ops.new_type, store.project(pid).path, body.id.strip(), body.review)
        return next(ty.to_json() for ty in store.project(pid).types if ty.id == body.id.strip())

    @app.put("/api/projects/{pid}/lanes")
    def set_lanes(pid: str, body: Lanes):
        """The board's lanes from left to right: {"order": [type id, ...]} (project.yaml lanes)."""
        store.write(ops.set_lanes, store.project(pid).path, body.order)
        return {"types": [ty.to_json() for ty in store.project(pid).types]}

    @app.post("/api/projects/{pid}/archive")
    def archive(pid: str):
        store.write(ops.set_archived, store.project(pid).path, True)
        return project_json(store, pid)

    @app.post("/api/projects/{pid}/unarchive")
    def unarchive(pid: str):
        store.write(ops.set_archived, store.project(pid).path, False)
        return project_json(store, pid)

    @app.post("/api/projects/{pid}/start")
    def start_project(pid: str):
        """开始运行: the template's tasks are reviewed; they start as any other project's do."""
        store.write(ops.start_project, store.project(pid).path)
        return project_json(store, pid)

    @app.get("/api/projects/{pid}/board")
    def board(pid: str):
        return board_json(store, store.project(pid))

    @app.get("/api/projects/{pid}/ledger")
    def get_ledger(pid: str):
        """The project's measured results: primary metric, baseline, best, every row and the files in error."""
        return ledger.ledger(store.project(pid))

    # ---- tasks

    @app.post("/api/projects/{pid}/tasks")
    def create_task(pid: str, body: NewTask):
        type_, name = body.type, body.name.strip()
        store.write(
            ops.new_task,
            store.project(pid).path,
            type_,
            name,
            body.prompt,
            title=body.title.strip(),
            depends_on=body.depends_on,
            review=body.review,
            agent=body.agent,
            queue=body.mode != "draft",
            machines=body.machines,
            checks=body.checks,
        )
        p = store.project(pid)
        return task_detail(store, p, p.task(f"{type_}/{name}"))

    @app.post("/api/projects/{pid}/tasks/preview")
    def preview(pid: str, body: NewTask):
        """What the new-task form shows while you type: the dependencies the prompt and the chosen upstream tasks
        give, and whether the name can be used."""
        p = store.project(pid)
        type_, name = body.type, body.name.strip()
        error = ""
        if name:
            try:
                ops.task_paths(p, type_, name)
            except AgentSysError as e:
                error = str(e)
        edges = preview_edges(p, type_, name, body.prompt, body.depends_on)
        return {"name_error": error, "upstream": [edge_end(p, e, e.up) for e in edges]}

    @app.get("/api/projects/{pid}/tasks/{type_}/{name}")
    def get_task(pid: str, type_: str, name: str):
        return task_detail(store, *task_of(pid, type_, name))

    @app.put("/api/projects/{pid}/tasks/{type_}/{name}")
    def put_task(pid: str, type_: str, name: str, body: TaskEdit):
        """Only the fields given; '' or null removes a frontmatter key. send_to_agent: a task that has run is told
        about the new prompt by a message; when it cannot be told now, the edit is saved all the same and
        notify_error says why."""
        p, t = task_of(pid, type_, name)
        store.write(ops.update_task, p.path, t.id, body.frontmatter, body.prompt, body.base_mtime)
        t, error = store.project(pid).task(t.id), ""
        if body.send_to_agent and t.managed and t.state.get("runs"):
            text = launch.say_for(
                store.home,
                "任务 prompt 已更新（`tasks/{task}.md`），请重新阅读并按新要求调整。",
                task=t.id,
            )
            try:
                task_write(pid, t.id, runs.message, store.home, p.path, t.id, text)
            except AgentSysError as e:
                error = str(e)
        return {**task_detail(store, *task_of(pid, type_, name)), "notify_error": error}

    @app.post("/api/projects/{pid}/tasks/{type_}/{name}/rename")
    def rename(pid: str, type_: str, name: str, body: Rename):
        p, t = task_of(pid, type_, name)
        new = store.write(ops.rename, p.path, t.id, body.type or t.type, body.name or t.name)
        p = store.project(pid)
        return task_detail(store, p, p.task(new))

    @app.post("/api/projects/{pid}/tasks/{type_}/{name}/actions/{action}")
    def act(pid: str, type_: str, name: str, action: str, body: Action | None = None):
        """A task's buttons: the task as it is now, and what the action returned under its key (takeover: how to
        attach; resume: when the message is read; ...). 删除 returns where the task went instead."""
        p, t = task_of(pid, type_, name)
        home, pdir, tid, body = store.home, p.path, t.id, body or Action()
        if action == "delete":
            trash = store.write(ops.delete, pdir, tid)
            return {"deleted": tid, "trash": str(trash.relative_to(home))}
        actions = {  # action -> (the reply's key for what it returns, the call)
            "start": (None, lambda: runs.start(home, pdir, tid)),
            "stop": (None, lambda: runs.stop(home, pdir, tid)),
            "takeover": ("takeover", lambda: runs.takeover(home, pdir, tid)),
            "end_takeover": (None, lambda: runs.end_takeover(home, pdir, tid)),
            # 继续, in the agent's language; a message of yours goes to …/messages
            "resume": (
                "delivery",
                lambda: runs.message(home, pdir, tid, launch.say_for(home, "继续")),
            ),
            "add_budget": (
                "budget_usd",
                lambda: runs.add_budget(home, pdir, tid, body.usd, body.text),
            ),
            "cancel": (None, lambda: runs.cancel(home, pdir, tid)),
            "join": (None, lambda: ops.join(pdir, tid)),
            "join_children": ("joined", lambda: ops.join_children(pdir, tid)),
            "approve": (None, lambda: ops.approve(pdir, tid)),
            "mark_read": (None, lambda: ops.mark_read(pdir, tid)),
        }
        if action not in actions:
            raise AgentSysError(_("不认识的操作「{action}」", action=action), 404)
        key, call = actions[action]
        result, task = task_write(pid, tid, call)
        return {**({key: result} if key else {}), "task": task}

    @app.post("/api/projects/{pid}/tasks/{type_}/{name}/messages")
    def message(pid: str, type_: str, name: str, body: Message):
        """Your message: into the running session, or a new round of it (resume)."""
        p, t = task_of(pid, type_, name)
        delivery, task = task_write(pid, t.id, runs.message, store.home, p.path, t.id, body.text)
        return {"delivery": delivery, "task": task}

    # ---- questions.md

    @app.get("/api/projects/{pid}/tasks/{type_}/{name}/questions")
    def get_questions(pid: str, type_: str, name: str):
        return questions.read(task_of(pid, type_, name)[1])

    @app.post("/api/projects/{pid}/tasks/{type_}/{name}/questions")
    def answer_questions(pid: str, type_: str, name: str, body: Answers):
        p, t = task_of(pid, type_, name)
        args = (store.home, p.path, t.id, body.answers, body.base_mtime)
        out, task = task_write(pid, t.id, questions.submit, *args)
        return {**out, "task": task}

    # ---- run output

    def run_dir_of(pid, type_, name, run):
        t = task_of(pid, type_, name)[1]
        path = runs_dir(t.workspace) / run
        if not t.managed or run.startswith(".") or "/" in run or not path.is_dir():
            raise AgentSysError(_("运行记录「{run}」不存在", run=run), 404)
        return t, path

    @app.get("/api/projects/{pid}/tasks/{type_}/{name}/runs/{run}/events")
    def run_events(
        pid: str,
        type_: str,
        name: str,
        run: str,
        after: int = -1,
        before: int | None = None,
        tail: int | None = None,
    ):
        """The events the run view shows, slim (events.slim): those after seq `after` (what is new), or the last
        `tail` before seq `before` (a long run opens at its end; earlier pages when you scroll up)."""
        t, path = run_dir_of(pid, type_, name, run)
        return {
            "run": runs.run_json(t, runs.record(path), store.monitor.follows.get(path)),
            **read_events(path, after, before, tail),
        }

    @app.get("/api/projects/{pid}/tasks/{type_}/{name}/runs/{run}/events/{seq}")
    def run_event(pid: str, type_: str, name: str, run: str, seq: int):
        """One event whole: what an opened tool line or thinking block shows of what was cut."""
        event = read_event(run_dir_of(pid, type_, name, run)[1], seq)
        if event is None:
            raise AgentSysError(_("运行记录「{run}」中没有第 {seq} 条事件", run=run, seq=seq), 404)
        return event

    @app.get("/api/projects/{pid}/tasks/{type_}/{name}/runs/{run}/launch")
    def run_launch(pid: str, type_: str, name: str, run: str):
        return {"markdown": read_text(run_dir_of(pid, type_, name, run)[1] / "launch.md")}

    # ---- workspace files

    @app.get("/api/projects/{pid}/tasks/{type_}/{name}/files")
    def list_files(pid: str, type_: str, name: str, dir: str = ""):
        return files.list_dir(task_of(pid, type_, name)[1].workspace, dir)

    @app.get("/api/projects/{pid}/tasks/{type_}/{name}/file")
    def read_file(pid: str, type_: str, name: str, path: str):
        return files.read_file(task_of(pid, type_, name)[1].workspace, path)

    @app.put("/api/projects/{pid}/tasks/{type_}/{name}/file")
    def write_file(pid: str, type_: str, name: str, path: str, body: FileText):
        t = task_of(pid, type_, name)[1]
        args = (t.workspace, path, body.content, body.base_mtime)
        return store.write(files.write_file, *args, also=[(pid, t.id)])

    @app.get("/api/projects/{pid}/tasks/{type_}/{name}/raw")
    def raw(pid: str, type_: str, name: str, path: str):
        target = files.resolve(task_of(pid, type_, name)[1].workspace, path)
        if not target.is_file():
            raise AgentSysError(_("文件「{path}」不存在", path=path), 404)
        return FileResponse(target, headers=files.raw_headers())

    # ---- rules and knowledge

    def scope_of(body):
        """The project id of a rules / knowledge / settings write, None for global; a project must exist."""
        pid = check_scope(body.scope, body.project)
        if pid:
            store.project(pid)
        return pid

    @app.get("/api/rules")
    def get_rules(project: str = ""):
        return knowledge.rules(store.home, store.project(project) if project else None)

    @app.put("/api/rules")
    def put_rule(body: Rule):
        pid = scope_of(body)
        args = (store.home, pid, body.name, body.content, body.base_mtime)
        return store.write(knowledge.save_rule, *args, also=[(pid or "", None)])

    @app.get("/api/knowledge")
    def get_knowledge(project: str = ""):
        """Candidates of one project (all projects that are not archived on the global page) and the libraries."""
        with store.lock:
            projects = (
                [store.project(project)] if project else [p for p in store.live() if not p.archived]
            )
            return {
                "candidates": knowledge.candidates(store.home, projects),
                "libraries": [
                    *([knowledge.library(store.home, project)] if project else []),
                    knowledge.library(store.home),
                ],
                "types": list(knowledge.ENTRY_TYPES),
            }

    @app.post("/api/knowledge/candidates/{action}")
    def handle_candidate(action: str, body: Candidate):
        p, t, path = knowledge.find_candidate(store.project, body.id)
        if action == "adopt":
            fields = (body.name, body.description, body.type, body.body)
            args = (knowledge.adopt, store.home, p, t, path, body.scope, *fields, body.replace)
        elif action == "to_rule":
            args = (knowledge.to_rule, store.home, p, path, body.scope, body.file, body.text)
        elif action == "discard":
            args = (knowledge.discard, path)
        else:
            raise AgentSysError(_("不认识的操作「{action}」", action=action), 404)
        library = p.id if body.scope == "project" else ""
        store.write(*args, also=[(p.id, t.id), (library, None)])
        return {"ok": True}

    @app.put("/api/knowledge/entries")
    def put_entry(body: Entry):
        pid = scope_of(body)
        fields = (body.name, body.description, body.type, body.body, body.base_mtime)
        store.write(knowledge.edit_entry, store.home, pid, *fields, also=[(pid or "", None)])
        return knowledge.library(store.home, pid)

    @app.post("/api/knowledge/entries/delete")
    def delete_entry(body: Entry):
        pid = scope_of(body)
        args = (store.home, pid, body.name, body.base_mtime)
        store.write(knowledge.delete_entry, *args, also=[(pid or "", None)])
        return knowledge.library(store.home, pid)

    # ---- settings

    @app.get("/api/settings")
    def get_settings(project: str = ""):
        if project:
            store.project(project)
        return ops.settings(store.home, project or None)

    @app.put("/api/settings")
    def put_settings(body: Settings):
        """A new language or global limit reaches the cards with the sync: ('', None) is agent_sys.yaml."""
        pid = scope_of(body)
        args = (store.home, body.scope, body.values, pid)
        return store.write(ops.save_settings, *args, also=[(pid or "", None)])

    @app.get("/api/models")
    def get_models():
        """The models offered in the forms, and what the aliases stand for here (launch.models)."""
        return {"models": launch.models(store.home)}

    # ---- changes

    @app.get("/api/events")
    async def events():
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        listener = lambda change: loop.call_soon_threadsafe(queue.put_nowait, change)  # noqa: E731
        store.listeners.add(listener)

        async def stream():
            try:
                yield "retry: 3000\n\n"
                while True:
                    try:
                        first = await asyncio.wait_for(queue.get(), 15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    await asyncio.sleep(COALESCE_SECONDS)
                    batch = [first] + [queue.get_nowait() for _ in range(queue.qsize())]
                    for change, c in {json.dumps(c, ensure_ascii=False): c for c in batch}.items():
                        yield f"event: {'run' if 'run' in c else 'changed'}\ndata: {change}\n\n"
            finally:
                store.listeners.discard(listener)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/")
    @app.get("/index.html")
    def index():
        """The page in the language of agent_sys.yaml: the scripts read it from <html lang> before they draw anything.
        It names every module up front, so the browser asks for them all at once instead of one import level per
        round trip."""
        page = read_text(WEB / "index.html").replace(
            '<html lang="en">', f'<html lang="{i18n.LANG}">', 1
        )
        modules = sorted(
            p.relative_to(WEB).as_posix()
            for p in [*WEB.glob("*.js"), *WEB.glob("vendor/*.*js")]
            if p.name != "boot.js"
        )
        links = "".join(f'<link rel="modulepreload" href="{m}">\n' for m in modules)
        return HTMLResponse(page.replace("</head>", f"{links}</head>", 1))

    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app
