"""The ownership rules for files and directories this package reads or creates outside the fleet root's own
writes: external sources (machine pool, KernelForge, e2e-flow, OTel copy) and the collector's directory.
One home for the rules, used by the analysis jobs, their sources and bin/o11y-launch. Standard library only."""

from __future__ import annotations

import functools
import grp
import json
import os
import pwd
import stat


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


def tree_problem(path: str) -> str | None:
    """Why the directory chain to `path` cannot be trusted (None = fine). Every component: no symlink.
    Components owned by root: not writable by group/others unless sticky. Components owned by this user:
    mode 0700 or stricter (no group/other bits). Components owned by anyone else: refused."""
    cur = os.sep
    for comp in os.path.abspath(path).split(os.sep)[1:]:
        cur = os.path.join(cur, comp)
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            return f"{cur} missing"
        if stat.S_ISLNK(st.st_mode):
            return f"{cur} is a symlink"
        if not stat.S_ISDIR(st.st_mode):
            return f"{cur} is not a directory"
        if st.st_uid == 0:
            if st.st_mode & 0o022 and not st.st_mode & stat.S_ISVTX:
                return f"{cur} is writable by others and not sticky"
        elif st.st_uid == os.getuid():
            if st.st_mode & 0o077:
                return f"{cur} has mode {st.st_mode & 0o777:o} (needs 0700)"
        else:
            return f"{cur} is owned by uid {st.st_uid}"
    return None


def ensure_private_tree(path: str) -> str | None:
    """Create the missing tail of `path` with mode 0700, then check the whole chain; returns the problem."""
    path = os.path.abspath(path)
    missing = []
    cur = path
    while not os.path.lexists(cur):
        missing.append(cur)
        cur = os.path.dirname(cur)
    for d in reversed(missing):
        try:
            os.mkdir(d, 0o700)
        except FileExistsError:
            pass
        except OSError as e:
            return f"cannot create {d} ({e.__class__.__name__})"
    return tree_problem(path)
