"""o11y_sources — read-only source adapters shared by the analysis jobs (design §8.2).

Each adapter reads one external format (Claude Code's registry and histories, the machine pool, KernelForge,
e2e-flow, the optional OTel copy) defensively, returns plain values, and reports missing or mistyped fields
through `Problems`, never by guessing. Nothing here writes protocol files; cursors are kept by each job
(o11y_job.CursorStore). Pure rules shared by the jobs (usage totals, pool attribution) live here so each rule
has one home. Standard library only in this module and in `otel`, so bin/o11y-launch can import them."""

from __future__ import annotations

import functools
import grp
import json
import os
import pwd
import stat
from dataclasses import dataclass, field


@dataclass
class Problem:
    source: str  # claude_registry | transcript | pool | forge | e2e_flow | otel
    path: str
    field: str
    detail: str = ""

    @property
    def key(self) -> str:
        return f"{self.source}|{self.path}|{self.field}"


@dataclass
class Problems:
    items: list[Problem] = field(default_factory=list)

    def add(self, source: str, path: str, fld: str, detail: str = "") -> None:
        self.items.append(Problem(source, str(path), fld, detail))

    def require(
        self, source: str, path: str, obj: dict, fields: list[str], types: dict | None = None
    ) -> bool:
        """Check dotted required fields (present; the value may be null unless `types` says otherwise) and,
        for fields named in `types`, the value's type (a tuple of Python types; include type(None) to allow null)."""
        ok = True
        for f in fields:
            cur = obj
            for part in f.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    self.add(source, path, f, "missing required field")
                    ok = False
                    break
                cur = cur[part]
            else:
                want = (types or {}).get(f)
                if want is not None and (
                    not isinstance(cur, want) or (isinstance(cur, bool) and bool not in want)
                ):
                    self.add(source, path, f, f"wrong type {type(cur).__name__}")
                    ok = False
        return ok


def _mode_problem(path: str, st: os.stat_result, kind: str | None, external: bool) -> str | None:
    if stat.S_ISLNK(st.st_mode):
        return f"{path} is a symlink"
    if (kind == "file" and not stat.S_ISREG(st.st_mode)) or (
        kind == "dir" and not stat.S_ISDIR(st.st_mode)
    ):
        return f"{path} is not a {'regular file' if kind == 'file' else 'directory'}"
    if st.st_uid != os.getuid():
        return f"{path} is owned by uid {st.st_uid}"
    if st.st_mode & 0o002:
        return f"{path} is world-writable (mode {st.st_mode & 0o777:o})"
    if st.st_mode & 0o020 and not (external and private_group(st.st_gid)):
        return f"{path} is writable by group {st.st_gid} (mode {st.st_mode & 0o777:o})"
    return None


def owned_problem(
    path: str, kind: str | None = None, follow: bool = False, external: bool = False
) -> str | None:
    """The one ownership rule (None = fine): no symlink (unless `follow`), the expected type (`file` / `dir`),
    owned by this uid, never world-writable, never group-writable — except, for `external` data we only read
    (machine pool, KernelForge, e2e-flow), group-write by our user-private group (umask 002). What we create
    and trust ourselves (OTel dir and files, token, fleet root) stays strict."""
    try:
        st = os.stat(path) if follow else os.lstat(path)
    except OSError as e:
        return f"{path} unreadable ({e.__class__.__name__})"
    return _mode_problem(path, st, kind, external)


def owned_open(path, external: bool = False):
    """(binary file object, None) or (None, problem): open without following a symlink, then apply the
    ownership rule to the OPENED file (fstat), so the checked file is the one read (no check-then-open race)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as e:
        return None, f"{path} unreadable ({e.__class__.__name__})"
    why = _mode_problem(str(path), os.fstat(fd), "file", external)
    if why:
        os.close(fd)
        return None, why
    return os.fdopen(fd, "rb"), None


def owned_json(path, external: bool = False):
    """(document, None) or (None, problem) through `owned_open`."""
    f, why = owned_open(path, external)
    if f is None:
        return None, why
    with f:
        try:
            return json.loads(f.read()), None
        except ValueError:
            return None, f"{path} is not valid JSON"


@functools.lru_cache(maxsize=8)
def private_group(gid: int) -> bool:
    """`gid` is our user-private group: our primary gid, named like our user, no listed members, and no other
    account has it as primary group. Fails closed when an answer cannot be established. Cached per process."""
    try:
        me = pwd.getpwuid(os.getuid())
        g = grp.getgrgid(gid)
        return (
            gid == os.getgid() == me.pw_gid
            and g.gr_name == me.pw_name
            and not g.gr_mem
            and all(p.pw_uid == me.pw_uid for p in pwd.getpwall() if p.pw_gid == gid)
        )
    except (KeyError, OSError):
        return False
