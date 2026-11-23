"""Optional: Claude Code native OTel log events from a local collector's JSONL (otel.md).

Accepted line shapes:
- flattened (vector remap): {"time", "event_name", "attributes": {...}, "resource": {...}}
- vector native decoding: {"message": "claude_code.<event>", "attributes": {...}, "timestamp"}
- OTel Collector file exporter (OTLP-JSON): {"resourceLogs": [{"scopeLogs": [{"logRecords": [...]}]}]}
Used events: api_request (tokens/cost per session.id), compaction, api_error, api_retries_exhausted.
"""

from __future__ import annotations

import json
import os

from . import Problems, owned_json, owned_open, owned_problem  # noqa: F401  (also o11y-launch's)

USED = {"api_request", "compaction", "api_error", "api_retries_exhausted"}


def _anyvalue(v: dict):
    if not isinstance(v, dict):
        return v
    for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if k in v:
            return int(v[k]) if k == "intValue" else v[k]
    if "arrayValue" in v:
        return [_anyvalue(x) for x in v["arrayValue"].get("values", [])]
    return None


def records(path: str, lines, problems: Problems) -> list[dict]:
    """Normalise to {event, attrs} for the events we use."""
    out = []
    for off, raw in lines:
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        batch = []
        if "resourceMetrics" in d or "resourceSpans" in d:
            continue  # the collector's metric and trace files: not log events
        if "resourceLogs" in d:
            for rl in d.get("resourceLogs") or []:
                for sl in rl.get("scopeLogs") or []:
                    for r in sl.get("logRecords") or []:
                        batch.append(
                            {
                                a["key"]: _anyvalue(a.get("value"))
                                for a in r.get("attributes") or []
                                if isinstance(a, dict) and "key" in a
                            }
                        )
        elif isinstance(d.get("attributes"), dict):
            batch.append(d["attributes"])
        else:
            problems.add("otel", path, "attributes", "unrecognised line shape")
            continue
        for a in batch:
            name = a.get("event.name")
            if name in USED:
                if "session.id" not in a:
                    problems.add("otel", path, "session.id", "missing required attribute")
                    continue
                out.append({"event": name, "attrs": a, "offset": off})
    return out


# ----------------------------------------------------------------------------------------- private dir checks
# The collector directory lives on local disk under /var/tmp (the container runtime cannot mount NFS homes),
# where other users can create files. One rule, used by o11y_log, o11y_facts and bin/o11y-launch.


def default_otel_dir(environ=None) -> str:
    """`$AGENT_SYS_O11Y_OTEL_DIR`, else `/var/tmp/<user>.infera_agent_sys/o11y/otel`: a top-level directory of
    our own directly under the sticky /var/tmp, so no group-writable parent of someone else's choosing."""
    environ = os.environ if environ is None else environ
    explicit = environ.get("AGENT_SYS_O11Y_OTEL_DIR")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    user = environ.get("USER") or str(os.getuid())
    return os.path.join("/var/tmp", f"{user}.infera_agent_sys", "o11y", "otel")


def tree_problem(path: str) -> str | None:
    """Why the directory chain to `path` cannot be trusted (None = fine). Every component: no symlink.
    Components owned by root: not writable by group/others unless sticky. Components owned by this user:
    mode 0700 or stricter (no group/other bits). Components owned by anyone else: refused."""
    import stat as st_mod

    path = os.path.abspath(path)
    parts = path.split(os.sep)
    cur = os.sep
    for comp in parts[1:]:
        cur = os.path.join(cur, comp)
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            return f"{cur} missing"
        if st_mod.S_ISLNK(st.st_mode):
            return f"{cur} is a symlink"
        if not st_mod.S_ISDIR(st.st_mode):
            return f"{cur} is not a directory"
        if st.st_uid == 0:
            if st.st_mode & 0o022 and not st.st_mode & st_mod.S_ISVTX:
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


def private_jsonl_files(dir_path: str) -> tuple[list[str], list[tuple[str, str]]]:
    """The collector's `*.jsonl` set in name order, keeping only files that pass `owned_problem`, and only when
    the directory chain passes `tree_problem`. A missing directory is not a problem (OTel copy off)."""
    if not os.path.lexists(dir_path):
        return [], []
    bad = tree_problem(dir_path)
    if bad:
        return [], [(dir_path, bad)]
    files, problems = [], []
    for name in sorted(os.listdir(dir_path)):
        if not name.endswith(".jsonl"):
            continue
        p = os.path.join(dir_path, name)
        why = owned_problem(p, "file")
        if why:
            problems.append((p, why))
        else:
            files.append(p)
    return files, problems


def write_private(path: str, data: bytes) -> None:
    """Write `data` to `path` owner-only without following symlinks: fresh tmp (O_EXCL|O_NOFOLLOW, 0600) in the
    same (verified) directory, fsync, rename over the target."""
    d = os.path.dirname(path)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp.{os.getpid()}.{os.urandom(4).hex()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
