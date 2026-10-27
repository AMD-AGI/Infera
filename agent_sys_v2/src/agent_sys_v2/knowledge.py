"""Rules and knowledge: the rules page and the knowledge page.

  AGENT_SYS_HOME/rules/*.md, projects/<p>/rules/*.md          global and project rules, injected into every run
  AGENT_SYS_HOME/knowledge/{INDEX.md, entries/<name>.md}        global knowledge library
  projects/<p>/knowledge/{INDEX.md, entries/<name>.md}     project knowledge library
  projects/<p>/workspaces/<type>/<task>/lessons/<file>.md  candidates the agent wrote; handled ones move to
                                                            lessons/.done/ (moving one back makes it a candidate again)

An entry (and a candidate) is a markdown file with frontmatter name / description / type (ENTRY_TYPES), and in a
library also source and approved. INDEX.md is made from the entries, one line each, `- name — description`, sorted
by name; it is made again whenever entries/ changes, through agent_sys or by hand (the server's Watcher). Only the two
INDEX.md files go into a run; the agent reads an entry when it needs it.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

from ruamel.yaml.comments import CommentedMap

from . import launch
from .i18n import _
from .launch import knowledge_dir, rules_dir
from .model import (
    NAME,
    AgentSysError,
    check_name,
    check_scope,
    iso,
    mtime,
    plain,
    read_task_file,
    read_text,
    write_task_file,
    write_text,
)

ENTRY_TYPES = ("pitfall", "env-fact", "method", "negative-result", "rule")
RULE_FILE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.md")
LESSON_FILE = re.compile(r"[^/\\]+\.md")
INDEX_LINES = 60  # past this, the index (injected into every run) should be pruned


def tokens(n_bytes):
    """A rough token count for UTF-8 text that is mostly Chinese."""
    return round(n_bytes / 3.2)


# ---------------------------------------------------------------- rules


def rule_file(path, home):
    return {
        "name": path.name,
        "path": rel(path, home),
        "size": path.stat().st_size,
        "mtime": iso(mtime(path)),
        "content": read_text(path),
    }


def rel(path, home):
    try:
        return str(Path(path).relative_to(home))
    except ValueError:
        return str(path)


def rules(home, project=None):
    """The rules page: global and project rules, the agent_sys protocol (read only, it comes with
    agent_sys's code) and what a run of this project gets injected, part by part, as launch.md lists it."""
    home = Path(home)

    def scope(d):
        return {"dir": rel(d, home), "files": [rule_file(f, home) for f in sorted(d.glob("*.md"))]}

    parts = [
        {
            "kind": p["kind"],
            "title": p["title"],
            "path": rel(p["path"], home),
            "bytes": len(p["text"].encode()),
        }
        for p in launch.system_parts(home, project)
    ]
    total = sum(p["bytes"] for p in parts)
    return {
        "global": scope(rules_dir(home)),
        "project": scope(rules_dir(home, project.id)) if project else None,
        "protocol": {
            **rule_file(launch.protocol(home), home),
            "path": str(launch.protocol(home)),
            "language": launch.agent_language(home),
        },
        "injected": {"parts": parts, "bytes": total, "tokens": tokens(total)},
    }


def check_rule_name(name):
    if not RULE_FILE.fullmatch(name or ""):
        raise AgentSysError(
            _("规则文件名「{name}」不合法：应以 .md 结尾，只能包含字母、数字、_、- 和 .", name=name)
        )


def save_rule(home, pid, name, content, base_mtime=""):
    """Write a rules file. No base_mtime = a new file (refused if it exists); with it, refused if the file changed
    since you read it."""
    check_rule_name(name)
    path = rules_dir(home, pid) / name
    if not base_mtime and path.exists():
        raise AgentSysError(_("规则文件 {name} 已存在", name=name), 409)
    if base_mtime and base_mtime != iso(mtime(path)):
        raise AgentSysError(_("{path} 在你编辑期间已被修改，请载入新版本后再保存", path=name), 409)
    write_text(path, content if content.endswith("\n") or not content else content + "\n")
    return rule_file(path, home)


def append_rule(home, pid, name, text):
    check_rule_name(name)
    path = rules_dir(home, pid) / name
    old = read_text(path).rstrip()
    write_text(path, (old + "\n\n" if old else "") + text.strip() + "\n")
    return path


# ---------------------------------------------------------------- entries and INDEX.md


def read_entry(path):
    """An entry or a candidate -> its fields, with `problems`: what is missing or broken (it is listed anyway)."""
    problems = []
    try:
        fm, body = read_task_file(path)
    except ValueError as e:
        fm, body = {}, read_text(path)
        problems.append(_("frontmatter 无法解析：{error}", error=e))
    if not read_text(path).startswith("---") and not problems:
        problems.append(_("缺少 frontmatter"))
    name = str(fm.get("name") or "").strip()
    if not NAME.fullmatch(name):
        problems.append(_("缺少 name") if not name else _("name「{name}」不合法", name=name))
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", path.stem).strip("-_") or "lesson"
    description = " ".join(str(fm.get("description") or "").split())
    if not description:
        problems.append(_("缺少 description"))
    type_ = str(fm.get("type") or "")
    if type_ not in ENTRY_TYPES:
        problems.append(_("type「{type}」不认识", type=type_) if type_ else _("缺少 type"))
    return {
        "name": name,
        "description": description,
        "type": type_,
        "source": str(fm.get("source") or ""),
        "approved": str(plain(fm.get("approved")) or ""),
        "body": body.strip("\n"),
        "fm": fm,
        "problems": problems,
    }


def entry_files(kdir):
    return [f for f in sorted((kdir / "entries").glob("*.md")) if not f.name.startswith(".")]


def entry_names(kdir):
    return {f.stem for f in entry_files(kdir)}


def entries(kdir):
    return [(f, read_entry(f)) for f in entry_files(kdir)]


def index_text(kdir):
    """INDEX.md made from the entries: `- name — description` by name (the file name is the entry's name)."""
    lines = sorted((f.stem, e["description"]) for f, e in entries(kdir))
    return "".join(f"- {name} — {desc}\n" if desc else f"- {name}\n" for name, desc in lines)


def refresh_index(kdir):
    """Write INDEX.md again if it does not say what the entries say. True if it was written."""
    if not (kdir / "entries").is_dir():
        return False
    text = index_text(kdir)
    if read_text(kdir / "INDEX.md") == text:
        return False
    write_text(kdir / "INDEX.md", text)
    return True


def library(home, pid=None):
    kdir = knowledge_dir(home, pid)
    index = read_text(kdir / "INDEX.md")
    items = [
        {k: v for k, v in e.items() if k != "fm"}
        | {"name": f.stem, "path": rel(f, home), "mtime": iso(mtime(f))}
        for f, e in entries(kdir)
    ]
    return {
        "scope": "project" if pid else "global",
        "project": pid,
        "dir": rel(kdir, home),
        "entries": items,
        "index": {
            "text": index,
            "lines": len(index.splitlines()),
            "bytes": len(index.encode()),
            "tokens": tokens(len(index.encode())),
            "limit": INDEX_LINES,
        },
    }


def write_entry(kdir, name, description, type_, body, source, approved, fm=None):
    """entries/<name>.md; fm: the entry's frontmatter so far (fields agent_sys does not know are kept)."""
    fm = CommentedMap(fm or {})
    for k, v in (
        ("name", name),
        ("description", description),
        ("type", type_),
        ("source", source),
        ("approved", approved),
    ):
        fm[k] = v
    write_task_file(kdir / "entries" / f"{name}.md", fm, body.strip("\n") + "\n")
    refresh_index(kdir)


def check_entry(name, description, type_):
    check_name(name, _("经验名称"))
    if not description.strip():
        raise AgentSysError(_("请填写描述：经验索引里只有这一行"))
    if type_ not in ENTRY_TYPES:
        raise AgentSysError(
            _("经验类型「{type}」不认识（可用 {types}）", type=type_, types=" / ".join(ENTRY_TYPES))
        )


def entry_path(home, pid, name):
    check_name(name, _("经验名称"))
    path = knowledge_dir(home, pid) / "entries" / f"{name}.md"
    if not path.is_file():
        raise AgentSysError(_("经验「{name}」不存在", name=name), 404)
    return path


def edit_entry(home, pid, name, description, type_, body, base_mtime=""):
    """编辑 an entry of a library: description, type and text; its name, source and approval stay."""
    path = entry_path(home, pid, name)
    check_entry(name, description, type_)
    if base_mtime and base_mtime != iso(mtime(path)):
        raise AgentSysError(
            _("经验「{name}」在你编辑期间已被修改，请载入新版本后再保存", name=name), 409
        )
    e = read_entry(path)
    write_entry(
        path.parent.parent,
        name,
        " ".join(description.split()),
        type_,
        body,
        e["fm"].get("source", ""),
        e["fm"].get("approved", ""),
        e["fm"],
    )


def delete_entry(home, pid, name, base_mtime=""):
    """删除 an entry: it moves to knowledge/.trash/ (moving it back restores it) and INDEX.md is made again."""
    path = entry_path(home, pid, name)
    if base_mtime and base_mtime != iso(mtime(path)):
        raise AgentSysError(_("经验「{name}」在你查看之后已被修改，请刷新后再删除", name=name), 409)
    move_aside(path, path.parent.parent / ".trash")
    refresh_index(path.parent.parent)


def move_aside(path, into):
    """Move a file into a directory; a name already there gets the time added."""
    into.mkdir(parents=True, exist_ok=True)
    target = into / path.name
    if target.exists():
        target = into / f"{path.stem}-{datetime.now():%Y%m%d-%H%M%S}{path.suffix}"
    shutil.move(path, target)
    return target


# ---------------------------------------------------------------- candidates (workspaces' lessons/)


def candidate_id(pid, t, f):
    return f"{pid}:{t.id}/lessons/{f.name}"


def candidates(home, projects):
    """The lessons/*.md of every task of these projects, oldest first, listed before the task is done."""
    home = Path(home)
    in_global = entry_names(knowledge_dir(home))
    out = []
    for p in projects:
        libs = {"project": entry_names(knowledge_dir(home, p.id)), "global": in_global}
        for t in p.tasks.values():
            for f in sorted((t.workspace / "lessons").glob("*.md")):
                if f.name.startswith(".") or not f.is_file():
                    continue
                e = read_entry(f)
                out.append(
                    {
                        **{k: v for k, v in e.items() if k != "fm"},
                        "id": candidate_id(p.id, t, f),
                        "project": p.id,
                        "task": t.id,
                        "task_title": t.title,
                        "task_status": t.status,
                        "path": rel(f, home),
                        "file": f.name,
                        "created": iso(mtime(f)),
                        "source": e["source"] or t.id,
                        "conflict": [s for s in ("project", "global") if e["name"] in libs[s]],
                    }
                )
    return sorted(out, key=lambda c: (c["created"], c["id"]))


def find_candidate(store_project, cid):
    """'<project>:<type>/<task>/lessons/<file>' -> (project, task, path). store_project(pid) gives the scanned
    project (404 if there is none)."""
    m = re.fullmatch(r"([^:/]+):([^/]+)/([^/]+)/lessons/([^/]+)", cid or "")
    if not m or not LESSON_FILE.fullmatch(m[4]) or m[4].startswith("."):
        raise AgentSysError(_("经验候选「{id}」不存在", id=cid), 404)
    p = store_project(m[1])
    t = p.task(f"{m[2]}/{m[3]}")
    path = t.workspace / "lessons" / m[4]
    if not path.is_file():
        raise AgentSysError(_("该经验候选已处理或已不存在"), 404)
    return p, t, path


def done(path):
    """A handled candidate moves to lessons/.done/ (not listed; moving it back makes it a candidate again)."""
    return move_aside(path, path.parent / ".done")


def adopt(
    home, p, t, path, scope, name=None, description=None, type_=None, body=None, replace=False
):
    """采纳 a candidate into the project's or the global library, edited or as it is: entries/<name>.md with source
    (`<project>:<type>/<task>` in the global library) and approved (today), INDEX.md made again, the candidate moved
    to .done/. An entry of the same name is refused unless replace (the page asks first); the old one moves to
    .trash/ as a deleted one does."""
    pid = check_scope(scope, p.id)
    e = read_entry(path)
    name = (name if name is not None else e["name"]).strip()
    description = " ".join((description if description is not None else e["description"]).split())
    type_ = type_ if type_ is not None else e["type"]
    check_entry(name, description, type_)
    kdir = knowledge_dir(home, pid)
    old = kdir / "entries" / f"{name}.md"
    if old.exists():
        if not replace:
            raise AgentSysError(_("经验库中已有「{name}」：换一个名字，或确认覆盖", name=name), 409)
        move_aside(old, kdir / ".trash")
    source = t.id if pid else f"{p.id}:{t.id}"
    # approved: a date, written bare (approved: 2026-10-21)
    body = body if body is not None else e["body"]
    write_entry(kdir, name, description, type_, body, source, datetime.now().date())
    done(path)
    return old


def to_rule(home, p, path, scope, file, text):
    """改写为规则: text appended to a global or project rules file (made if missing); the candidate moves to .done/."""
    pid = check_scope(scope, p.id)
    if not (text or "").strip():
        raise AgentSysError(_("请填写要写入规则的内容"))
    target = append_rule(home, pid, file, text)
    done(path)
    return target


def discard(path):
    done(path)
