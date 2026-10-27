"""A task's workspace as the 工作区文件 tab sees it: list one directory, read a file, write a file.

Paths are relative to the workspace and may not leave it: no absolute paths, no `..`. Reading follows symlinks
(workspaces often link to data elsewhere, and only you can reach the server's socket); writing does not
leave the workspace even through a symlink, and never touches .agent_sys/ (agent_sys's own state).
"""

from __future__ import annotations

import os
from itertools import islice
from pathlib import PurePosixPath

from .i18n import _
from .model import AgentSysError, iso, mtime, write_text

MAX_ENTRIES = 300  # per directory listed (a workspace may hold 100,000 files)
MAX_COUNT = 1000  # entries counted in a subdirectory; more: "1000+"
TAIL_BYTES = 64 * 1024  # text above this is shown as its last TAIL_LINES lines
TAIL_LINES = 200
MAX_MD = 2 * 1024 * 1024  # markdown is shown whole up to this size
IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
LOGS = {".log", ".out", ".err"}


def resolve(workspace, rel, write=False):
    """The path `rel` names inside the workspace; refused if it leaves the workspace. For writing, the real path
    (through symlinks), refused if it is outside the workspace or in .agent_sys/."""
    parts = PurePosixPath(rel or "").parts
    if (rel or "").startswith("/") or ".." in parts:
        raise AgentSysError(_("路径「{path}」不在任务工作区内", path=rel), 400)
    path = workspace.joinpath(*parts)
    if write:
        path, root = path.resolve(), workspace.resolve()
        if not path.is_relative_to(root):
            raise AgentSysError(_("路径「{path}」经软链接指向工作区以外，不能修改", path=rel), 400)
        if path.relative_to(root).parts[:1] == (".agent_sys",):
            raise AgentSysError(_(".agent_sys/ 由 agent_sys 维护，不能通过界面修改"), 400)
    return path


def kind_of(path):
    suffix = path.suffix.lower()
    return (
        "image"
        if suffix in IMAGES
        else "md"
        if suffix == ".md"
        else "log"
        if suffix in LOGS
        else "text"
    )


def visible(path):
    """The entries of a directory that are not hidden, as os.DirEntry: its type comes without a stat on most file
    systems (a big directory on NFS lists ten times faster)."""
    with os.scandir(path) as it:
        yield from (e for e in it if not e.name.startswith("."))


def is_dir(e):
    try:
        return e.is_dir()
    except OSError:
        return False


def count(path):
    """Entries of a subdirectory, up to MAX_COUNT; None if it cannot be read."""
    try:
        return sum(1 for _ in islice(visible(path), MAX_COUNT))
    except OSError:
        return None


def list_dir(workspace, rel=""):
    """One level of a directory: {dir, entries, total, truncated}. Directories first; hidden entries are left out.
    A subdirectory says how many entries it has (count, more: MAX_COUNT or more). A workspace that does not exist
    yet lists as empty."""
    path = resolve(workspace, rel)
    if not path.is_dir():
        if rel:
            raise AgentSysError(_("目录「{path}」不存在", path=rel), 404)
        return {"dir": "", "entries": [], "total": 0, "truncated": False}
    items = sorted(visible(path), key=lambda e: (not is_dir(e), e.name))
    entries = []
    for e in items[:MAX_ENTRIES]:
        out = {"name": e.name, "path": f"{rel}/{e.name}" if rel else e.name, "mtime": iso(mtime(e))}
        if is_dir(e):
            n = count(e.path)
            entries.append({**out, "type": "dir", "count": n, "more": n == MAX_COUNT})
        else:
            try:
                size = e.stat().st_size
            except OSError:  # a broken symlink
                size = 0
            entries.append(
                {**out, "type": "file", "size": size, "kind": kind_of(PurePosixPath(e.name))}
            )
    return {
        "dir": rel,
        "entries": entries,
        "total": len(items),
        "truncated": len(items) > MAX_ENTRIES,
    }


def tail(path, lines=TAIL_LINES):
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 4 * TAIL_BYTES))
        return b"\n".join(f.read().splitlines()[-lines:]).decode("utf-8", "replace")


def read_file(workspace, rel):
    """{path, kind, size, mtime, content}: markdown whole, long text by its tail; images (shown from /raw) and
    binary files by size only."""
    path = resolve(workspace, rel)
    if not path.is_file():
        raise AgentSysError(_("文件「{path}」不存在", path=rel), 404)
    size, kind = path.stat().st_size, kind_of(path)
    out = {"path": rel, "kind": kind, "size": size, "mtime": iso(mtime(path))}
    if kind == "image":
        return out
    with path.open("rb") as f:
        head = f.read(8192)
    if b"\0" in head:
        return {**out, "kind": "binary"}
    if size > (MAX_MD if kind == "md" else TAIL_BYTES):
        return {
            **out,
            "kind": "log",
            "content": tail(path),
            "truncated": True,
            "tail_lines": TAIL_LINES,
        }
    return {**out, "content": path.read_text(encoding="utf-8", errors="replace")}


def write_file(workspace, rel, content, base_mtime=""):
    """Save a text file of the workspace. base_mtime is the file's mtime when you started editing: if it changed
    since (the agent wrote it meanwhile), nothing is written. A symlink stays one (its target is written) and the
    file keeps its mode (a script stays executable)."""
    if not PurePosixPath(rel or "").parts:  # '' or '.': the workspace itself
        raise AgentSysError(_("请指定要保存的文件"), 400)
    path = resolve(workspace, rel, write=True)
    if path.is_dir():
        raise AgentSysError(_("「{path}」是目录", path=rel), 400)
    if base_mtime and path.exists() and iso(mtime(path)) != base_mtime:
        raise AgentSysError(_("{path} 在你编辑期间已被修改，请载入新版本后再保存", path=rel), 409)
    mode = path.stat().st_mode if path.exists() else None
    write_text(path, content)
    if mode is not None:
        path.chmod(mode)
    return {"path": rel, "mtime": iso(mtime(path))}


def raw_headers():
    """Response headers for /raw: never sniff a type, and whatever the file is, it runs sandboxed (opaque origin, no
    scripts): markup the agent wrote cannot call agent_sys's API from agent_sys_v2's origin, under any suffix."""
    return {
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-cache",
        "Content-Security-Policy": "sandbox",
    }
