"""The shared knowledge repo: where tasks and handoffs are backed up and found.

The layout (`release/`, `<user>/draft/`, `<user>/staging/`, each with
`tasks/` and `handoffs/`) and the item format are in `docs/knowledge_repo.md`.

**How writes are made.** Every change is built as a commit on top of the
remote's tip with git plumbing (`read-tree` into a private index,
`update-index`, `write-tree`, `commit-tree`) and pushed by hash. The clone's
working tree and index are never edited — the clone only ever fast-forwards.
So a failure at any point (a hook, a push race, a bad META) leaves nothing
half-done to clean up, a name is chosen against the remote's tree (no add/add
conflict between two hosts), and a push race is retried from the new tip.
Nothing is ever deleted or overwritten: an existing name gets `.v2`, `.v3`…;
`release/` is written only on a branch proposed as a pull request.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from skillkit import AGENT_SYS, hstore, settings
from skillkit import audit as audit_mod
from skillkit.runner import _now

META = "META.astk.json"
TIERS = ("release", "staging", "draft")
TYPES = ("task", "handoff")
PUSH_TRIES = 4
META_LIMIT = 1 << 20


class RemoteError(Exception):
    pass


class Refused(RemoteError):
    """The content may not be published as it is (a secret, a big file): exit 1."""


class NotConfigured(RemoteError):
    pass


@dataclass
class Config:
    url: str
    clone: str
    user: str
    branch: str = "main"


def config_path() -> Path:
    override = os.environ.get("INFERA_AGENT_SYS_CONFIG")
    return Path(override) if override else settings.config_dir() / "config.json"


def default_clone() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "infera-agent-sys" / "knowledge"


def default_user() -> str:
    try:
        email = _proc(None, "config", "user.email").stdout.decode(errors="replace")
    except OSError:  # no git
        email = ""
    return slug(email.strip().split("@")[0] or getpass.getuser())


def slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-.")
    if not out:
        raise RemoteError(f"{text!r} gives an empty name")
    return out[:80]


def user_slug(text: str) -> str:
    """A user's top-level directory. A tier name is reserved: user `release`
    would write straight into `release/` on the branch, which only a PR may."""
    out = slug(text)
    if out.lower() in TIERS:
        raise RemoteError(f"user name {out!r} is reserved; choose another with --user")
    return out


def load() -> Config:
    path = config_path()
    if not path.is_file():
        raise NotConfigured(
            f"no knowledge repo configured ({path} missing). Configure it once with:\n"
            "  astk remote init --url <git url> [--user <name>] [--clone <dir>]"
        )
    cfg = Config(**json.loads(path.read_text()))
    user_slug(cfg.user)  # a config written before the name was reserved
    return cfg


def _proc(
    cwd: str | Path | None, *args: str, env: dict | None = None, data: bytes | None = None
) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env or {})}
    return subprocess.run(["git", *args], cwd=cwd, env=env, input=data, capture_output=True)


def _stderr(proc: subprocess.CompletedProcess) -> str:
    return (proc.stderr.strip() or proc.stdout.strip()).decode(errors="replace")


def _git_bytes(where: Config | Path, *args: str, check: bool = True, **kw) -> bytes:
    """Raw stdout: for `-z` output, where a path must survive byte for byte."""
    cwd = where.clone if isinstance(where, Config) else where
    proc = _proc(cwd, *args, **kw)
    if check and proc.returncode != 0:
        raise RemoteError(f"git {' '.join(args[:4])} failed in {cwd}:\n{_stderr(proc)}")
    return proc.stdout


def _git(where: Config | Path, *args: str, check: bool = True, **kw) -> str:
    return _git_bytes(where, *args, check=check, **kw).decode(errors="replace").strip()


# -- setup and sync ------------------------------------------------------------- #


def init(
    url: str, *, user: str | None = None, clone: str | None = None, branch: str = "main"
) -> Config:
    """Clone the repo (or adopt an existing clone of the same URL) and save the config."""
    cfg = Config(
        url=url,
        clone=str(Path(clone or default_clone()).expanduser().resolve()),
        user=user_slug(user or default_user()),
        branch=branch,
    )
    path = Path(cfg.clone)
    if (path / ".git").is_dir():
        origin = _git(path, "remote", "get-url", "origin")
        if origin != url:
            raise RemoteError(f"{path} is a clone of {origin}, not {url}; choose another --clone")
    elif path.exists() and any(path.iterdir()):
        raise RemoteError(
            f"{path} exists, is not empty and is not a git clone; choose another --clone"
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(path.parent, "clone", "-q", url, str(path))
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    return cfg


def can_push(cfg: Config) -> tuple[bool, str]:
    """`git push --dry-run`: find a missing write permission now, not at the end of a packup."""
    proc = _proc(cfg.clone, "push", "--dry-run", "origin", f"HEAD:{cfg.branch}")
    return proc.returncode == 0, (_stderr(proc).splitlines() or [""])[-1]


def fetch(cfg: Config) -> str:
    _git(cfg, "fetch", "-q", "origin", cfg.branch)
    return _git(cfg, "rev-parse", f"origin/{cfg.branch}")


def sync(cfg: Config) -> str:
    """Fetch, then fast-forward the clone. A clone that cannot fast-forward
    (someone edited it by hand) is left exactly as it is, with the reason."""
    tip = fetch(cfg)
    if _git(cfg, "rev-parse", "--abbrev-ref", "HEAD", check=False) != cfg.branch:
        raise RemoteError(
            f"{cfg.clone} is not on {cfg.branch}; `git checkout {cfg.branch}` there and retry"
        )
    proc = _proc(cfg.clone, "merge", "-q", "--ff-only", f"origin/{cfg.branch}")
    if proc.returncode != 0:
        raise RemoteError(
            f"{cfg.clone} cannot fast-forward to origin/{cfg.branch} (local edits or commits?); "
            f"nothing was changed — `git -C {cfg.clone} status` shows why.\n{_stderr(proc)}"
        )
    return tip[:7]


def status(cfg: Config, *, offline: bool = False) -> dict:
    if not offline:
        _git(cfg, "fetch", "-q", "origin", cfg.branch, check=False)
    counts = _git(
        cfg, "rev-list", "--left-right", "--count", f"HEAD...origin/{cfg.branch}", check=False
    )
    ahead, _, behind = counts.partition("\t")
    return {
        **asdict(cfg),
        "config": str(config_path()),
        "head": _git(cfg, "rev-parse", "--short", "HEAD", check=False),
        "ahead": int(ahead or 0),
        "behind": int(behind or 0),
        "items": len(items(cfg)),
    }


# -- reading -------------------------------------------------------------------- #


@dataclass
class Item:
    type: str  # task | handoff
    tier: str  # release | staging | draft | local
    user: str | None  # None for release / local
    name: str
    path: str
    meta: dict


_SCALAR = (str, int, float)


def _meta(p: Path) -> dict:
    """META of one item; a malformed one must not take the listing down for everybody.

    Anyone can push a META, so every field a reader uses is normalised to the
    type it expects (summary a str, tags a list of str, context a str -> str
    dict, sealed a bool), and a symlinked META is not followed."""
    f = p / META
    try:
        if f.is_symlink():
            return {"meta_error": f"{META} is a symlink"}
        if not f.is_file():
            meta: object = {}
        elif f.stat().st_size > META_LIMIT:
            return {"meta_error": f"{META} is over {META_LIMIT >> 20} MB"}
        else:
            meta = json.loads(f.read_text(errors="replace"))
    except (OSError, ValueError, RecursionError):
        return {"meta_error": f"unreadable {META}"}
    if not isinstance(meta, dict):
        return {"meta_error": f"{META} is not an object"}
    for key in ("summary", "name", "user", "type", "tier", "created", "handoff_kind", "digest"):
        if key in meta and not isinstance(meta[key], str):
            meta[key] = str(meta[key]) if isinstance(meta[key], _SCALAR) else ""
    if "tags" in meta:
        tags = meta["tags"]
        tags = [tags] if isinstance(tags, str) else tags if isinstance(tags, list) else []
        meta["tags"] = [str(t) for t in tags if isinstance(t, _SCALAR)]
    if "context" in meta:
        ctx = meta["context"] if isinstance(meta["context"], dict) else {}
        meta["context"] = {str(k): str(v) for k, v in ctx.items() if isinstance(v, _SCALAR)}
    if "sealed" in meta and not isinstance(meta["sealed"], bool):
        del meta["sealed"]
    return meta


def _real_dir(p: Path) -> bool:
    """A directory checked out as itself — not a symlink that leaves the clone."""
    return p.is_dir() and not p.is_symlink()


def items(cfg: Config) -> list[Item]:
    """Every item in the clone's working tree (= the last synced remote tip)."""
    root = Path(cfg.clone)
    out: list[Item] = []

    def collect(base: Path, tier: str, user: str | None) -> None:
        for t in TYPES:
            d = base / f"{t}s"
            if not (_real_dir(base) and _real_dir(d)):
                continue
            for p in sorted(d.iterdir()):
                if _real_dir(p) and not p.name.startswith("."):
                    meta = _meta(p)
                    if t == "handoff":
                        meta.setdefault("sealed", (p / "manifest.yaml").is_file())
                    out.append(
                        Item(type=t, tier=tier, user=user, name=p.name, path=str(p), meta=meta)
                    )

    collect(root / "release", "release", None)
    for u in sorted(root.iterdir()):
        if _real_dir(u) and u.name != "release" and not u.name.startswith("."):
            for tier in ("staging", "draft"):
                collect(u / tier, tier, u.name)
    return out


# -- building commits on the remote tip ----------------------------------------- #


class _Tree:
    """A private index on top of a commit. Edits never touch the clone's own index;
    lookups (`exists`, `read`, `copy`) read the base commit."""

    def __init__(self, cfg: Config, base: str) -> None:
        self.cfg, self.base = cfg, base
        fd, self.index = tempfile.mkstemp(prefix="astk-index-")
        os.close(fd)
        os.unlink(self.index)  # git creates it; mkstemp only reserved a unique name
        self.env = {"GIT_INDEX_FILE": self.index}
        _git(cfg, "read-tree", base, env=self.env)

    def exists(self, path: str) -> bool:
        return bool(_git(self.cfg, "ls-tree", "--name-only", self.base, "--", path, check=False))

    def read(self, path: str) -> str:
        return _git(self.cfg, "show", f"{self.base}:{path}", check=False)

    def add_file(self, path: str, data: bytes, mode: str = "100644") -> None:
        self.add_blob(path, _git(self.cfg, "hash-object", "-w", "--stdin", data=data), mode)

    def add_blob(self, path: str, sha: str, mode: str = "100644") -> None:
        _git(self.cfg, "update-index", "--add", "--cacheinfo", f"{mode},{sha},{path}", env=self.env)

    def copy(self, src: str, dst: str, *, remove_src: bool = False) -> None:
        """Copy (or move) every file under `src` in the base commit to `dst`. Paths
        stay NUL separated bytes: a quoted or re-decoded path is another path."""
        raw = _git_bytes(self.cfg, "ls-tree", "-r", "-z", self.base, "--", src)
        cut, to = len(os.fsencode(src)), os.fsencode(dst)
        recs = b""
        for rec in filter(None, raw.split(b"\0")):
            info, path = rec.split(b"\t", 1)
            mode, _type, sha = info.split()
            if remove_src:
                recs += b"0 " + b"0" * 40 + b"\t" + path + b"\0"
            recs += mode + b" " + sha + b"\t" + to + path[cut:] + b"\0"
        _git(self.cfg, "update-index", "-z", "--index-info", env=self.env, data=recs)

    def close(self) -> None:
        Path(self.index).unlink(missing_ok=True)

    def commit(self, message: str) -> str:
        try:
            tree = _git(self.cfg, "write-tree", env=self.env)
            return _git(self.cfg, "commit-tree", tree, "-p", self.base, "-m", message)
        finally:
            self.close()


def _publish(
    cfg: Config, build: Callable[[_Tree], tuple[str, dict]], *, ref: str | None = None
) -> dict:
    """Build a commit on the current remote tip with `build(tree) -> (message, result)`
    and push it; on a push race, rebuild on the new tip. Then fast-forward the clone."""
    last = ""
    for _ in range(PUSH_TRIES):
        tip = fetch(cfg)
        tree = _Tree(cfg, tip)
        try:
            message, result = build(tree)
        except BaseException:
            tree.close()  # nothing is pushed
            raise
        sha = tree.commit(message)
        target = ref or f"refs/heads/{cfg.branch}"
        proc = _proc(cfg.clone, "push", "-q", "origin", f"{sha}:{target}")
        if proc.returncode == 0:
            if ref is None:
                try:
                    sync(cfg)
                except RemoteError:
                    pass  # pushed; the clone catches up on the next sync that can
            return result
        last = _stderr(proc)
    raise RemoteError(f"push failed {PUSH_TRIES} times; nothing was changed on the remote:\n{last}")


def _free(tree: _Tree, base: str, name: str) -> str:
    cand, n = name, 2
    while tree.exists(f"{base}/{cand}"):
        cand, n = f"{name}.v{n}", n + 1
    return cand


def _meta_blob(tree: _Tree, read_from: str, **changes: object) -> bytes:
    try:
        meta = json.loads(tree.read(f"{read_from}/{META}") or "{}")
    except ValueError:
        meta = {}
    meta = {**(meta if isinstance(meta, dict) else {}), **changes}
    return (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode()


# -- writing -------------------------------------------------------------------- #


def _filemode(p: Path) -> str:
    return "100755" if os.access(p, os.X_OK) else "100644"


def _kind_of_source(src: Path) -> str:
    if (src / "main.yaml").is_file() and (src / "assets").is_dir():
        return "task"
    if (src / "content").is_dir() or ((src / "README.md").is_file() and (src / "items").is_dir()):
        return "handoff"
    raise RemoteError(
        f"{src} is neither a task package (main.yaml + assets/) nor a handoff (README.md + items/)"
    )


def _source_files(src: Path, kind: str) -> list[tuple[str, Path | str, str]]:
    """(relative path, file or link target, git mode) — exactly what will be published.

    A task is dereferenced (a package may link files from a sibling package);
    a handoff keeps its links, as the store does (the digest counts them)."""
    out: list[tuple[str, Path | str, str]] = []
    if kind == "task":
        skip = {".astk-validation", "__pycache__"}
        for rel, real in audit_mod.files(src, follow=True):
            if not skip & set(Path(rel).parts):
                out.append((rel, real, _filemode(real)))
        return out
    content = src / "content" if (src / "content").is_dir() else src
    for p in sorted(content.rglob("*")):
        rel = "content/" + str(p.relative_to(content))
        if p.is_symlink():
            out.append((rel, os.readlink(p), "120000"))
        elif p.is_file():
            out.append((rel, p, _filemode(p)))
    for extra in ("manifest.yaml", "validation.yaml"):
        if content is not src and (src / extra).is_file():
            out.append((extra, src / extra, "100644"))
    return out


def _audit_blob(
    cfg: Config, a: audit_mod.Audit, rel: str, what: Path | str, mode: str, placeholders: bool
) -> str:
    """Scan one entry into `a` and write the scanned bytes as a blob; its sha."""
    if mode == "120000":
        data = os.fsencode(what)
        audit_mod.scan(a, rel, data, placeholders=placeholders)
        return _git(cfg, "hash-object", "-w", "--stdin", data=data)
    proc = subprocess.Popen(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=cfg.clone,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        audit_mod.scan(a, rel, Path(what), sink=proc.stdin.write, placeholders=placeholders)
        sha, err = proc.communicate()
    except BaseException as e:
        proc.kill()
        proc.communicate()
        if isinstance(e, OSError):
            raise RemoteError(f"cannot read {what} or store it in {cfg.clone}: {e}") from e
        raise
    if proc.returncode != 0:
        raise RemoteError(f"git hash-object failed for {rel}:\n{err.decode(errors='replace')}")
    return sha.decode().strip()


def backup(
    cfg: Config,
    src: Path,
    *,
    name: str | None = None,
    summary: str = "",
    tags: list[str] | None = None,
    context: dict[str, str] | None = None,
    allow_big: bool = False,
    allow_unscanned: bool = False,
    dry_run: bool = False,
) -> dict:
    """Publish a task package or a handoff as `<user>/draft/<type>s/<name>/`.

    `allow_big`: the user agreed to files over 4 MB (they are still scanned).
    `allow_unscanned`: the user checked by hand what the secret scan cannot
    read (an archive it cannot open, a secret-looking string in a binary)."""
    src = src.resolve()
    kind = _kind_of_source(src)
    content = src / "content" if (src / "content").is_dir() else src
    before = hstore.digest(content) if kind == "handoff" else ""
    # Read once, streaming: each file is scanned and written into the clone's
    # object store in the same pass, so what is published is exactly what was
    # audited (for a task, link targets included; for a handoff, manifest.yaml
    # and validation.yaml included), and memory stays one chunk per file.
    records = ("manifest.yaml", "validation.yaml") if kind == "handoff" else ()
    audited = audit_mod.Audit(root=str(src))
    entries = [
        (rel, _audit_blob(cfg, audited, rel, what, mode, rel not in records), mode)
        for rel, what, mode in _source_files(src, kind)
    ]
    if audited.blocking:
        raise Refused("refusing to back up:\n  " + "\n  ".join(audited.blocking))
    if audited.big_files and not allow_big:
        raise Refused(
            "files over 4 MB (ask the user; then --allow-big, or reference them by "
            "path + sha256):\n  "
            + "\n  ".join(f"{p} ({n / 2**20:.1f} MB)" for p, n in audited.big_files)
        )
    unscanned = audited.unscanned + [f"{h}: secret-looking string" for h in audited.binary_hits]
    if unscanned and not allow_unscanned:
        raise Refused(
            "not scannable for secrets, or a secret-looking string in a binary (ask the user "
            "to check by hand; then --allow-unscanned, or drop them):\n  " + "\n  ".join(unscanned)
        )
    if kind == "handoff" and hstore.digest(content) != before:
        raise RemoteError(f"{content} changed while it was read; back it up once it is quiet")
    meta: dict = {
        "type": kind,
        "user": cfg.user,
        "tier": "draft",
        "summary": summary,
        "tags": tags or [],
        "context": {**_lift_context(src), **(context or {})},
        # Placeholders, not the real path or host: the repo is shared and the
        # reader is an agent, which needs only the item's name to trace it.
        "source": f"<source-dir>/{src.name}",
        "source_host": "<host>",
        "created": _now(),
        "agent_sys_commit": _git(AGENT_SYS, "rev-parse", "HEAD", check=False),
    }
    if kind == "handoff":
        meta["sealed"] = content is not src and (src / "manifest.yaml").is_file()
        meta["digest"] = before
        if meta["sealed"]:
            import yaml  # noqa: PLC0415

            blob = next(sha for rel, sha, _m in entries if rel == "manifest.yaml")
            manifest = _git_bytes(cfg, "cat-file", "blob", blob)
            meta["handoff_kind"] = (yaml.safe_load(manifest) or {}).get("kind")
    base = f"{cfg.user}/draft/{kind}s"

    def build(tree: _Tree) -> tuple[str, dict]:
        item = _free(tree, base, slug(name or src.name))
        root = f"{base}/{item}"
        for rel, sha, mode in entries:
            tree.add_blob(f"{root}/{rel}", sha, mode)
        doc = {**meta, "name": item}
        tree.add_file(
            f"{root}/{META}", (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode()
        )
        return f"backup({cfg.user}/draft): {kind} {item}", {"relative": root, **doc}

    if dry_run:
        tree = _Tree(cfg, fetch(cfg))
        _msg, result = build(tree)
        tree.commit("dry run")  # objects only; no ref points at it
        return {**result, "dry_run": True, "files": len(entries)}
    result = _publish(cfg, build)
    return {"path": str(Path(cfg.clone) / result["relative"]), **result}


#: environment-record keys worth surfacing in search, under the name they get in META.
# fmt: off
_CONTEXT_KEYS = {
    "model": "model", "model_name": "model",
    "image": "image", "image_id": "image_id", "engine": "engine",
    "gpu": "gpu", "gpu_model": "gpu", "gpu_arch": "gpu_arch",
    "tp": "tp", "tp_size": "tp",
    "sglang_version": "sglang", "vllm_version": "vllm", "rocm_version": "rocm",
}
# fmt: on


def _lift_context(root: Path) -> dict[str, str]:
    """Best effort: model / engine / GPU / image from an `environment.yaml` or
    `env.json` in the item, so search can show them. Explicit `--context` wins."""
    import yaml  # noqa: PLC0415

    found: dict[str, str] = {}

    def walk(node: object, depth: int = 0) -> None:
        if depth > 4 or not isinstance(node, dict):
            return
        for k, v in node.items():
            key = _CONTEXT_KEYS.get(str(k).lower())
            if key and isinstance(v, (str, int, float)) and key not in found and str(v).strip():
                found[key] = str(v)[:120]
            walk(v, depth + 1)

    candidates = [
        p for pat in ("environment.yaml", "environment.yml", "env.json") for p in root.rglob(pat)
    ]
    for f in sorted(candidates, key=lambda p: len(p.parts))[:5]:
        try:
            walk(yaml.safe_load(f.read_text(errors="replace")))
        except Exception:  # noqa: BLE001, S112 — a malformed record just contributes nothing
            continue
    return found


def _propose(cfg: Config, kind: str, src: str, dst: str, missing: str, message: str, **kw) -> None:
    """One commit copying item `src` to `dst` with its META updated by `kw`;
    `ref` publishes on another branch, `remove_src` makes it a move."""
    if kind not in TYPES:
        raise RemoteError(f"type must be one of {TYPES}")
    ref, remove_src = kw.pop("ref", None), kw.pop("remove_src", False)

    def build(tree: _Tree) -> tuple[str, dict]:
        if not tree.exists(src):
            raise RemoteError(missing)
        if tree.exists(dst):
            raise RemoteError(f"{dst} already exists; use another name")
        tree.copy(src, dst, remove_src=remove_src)
        tree.add_file(f"{dst}/{META}", _meta_blob(tree, src, **kw))
        return message, {}

    _publish(cfg, build, ref=ref)


def promote(cfg: Config, kind: str, name: str) -> dict:
    """draft -> staging, after the user's own review: one commit moving the item."""
    src, dst = f"{cfg.user}/draft/{kind}s/{name}", f"{cfg.user}/staging/{kind}s/{name}"
    missing = f"no draft {kind} {name!r} of user {cfg.user} on the remote"
    msg = f"promote({cfg.user}): {kind} {name} draft -> staging"
    _propose(cfg, kind, src, dst, missing, msg, remove_src=True, tier="staging", promoted=_now())
    return {"path": str(Path(cfg.clone) / dst), "relative": dst}


def release_pr(cfg: Config, kind: str, name: str) -> dict:
    """Propose a staging item for `release/` as a branch built on the remote tip
    (never from the local clone), and open a pull request when `gh` can."""
    src, dst = f"{cfg.user}/staging/{kind}s/{name}", f"release/{kind}s/{name}"
    branch = f"release/{cfg.user}-{kind}-{name}"
    if _git(cfg, "ls-remote", "--heads", "origin", branch, check=False):
        raise RemoteError(
            f"branch {branch} already exists on the remote: the item was proposed already"
        )
    missing = f"no staging {kind} {name!r} of user {cfg.user} on the remote (promote it first)"
    msg, ref = f"release: {kind} {name} (from {cfg.user}/staging)", f"refs/heads/{branch}"
    _propose(cfg, kind, src, dst, missing, msg, ref=ref, tier="release", proposed_from=src)
    out = {"branch": branch, "path_on_branch": dst}
    if shutil.which("gh") and "github.com" in cfg.url:
        repo = re.sub(r"^(git@github\.com:|https://github\.com/)|\.git$", "", cfg.url)
        cmd = ["gh", "pr", "create", "--repo", repo, "--base", cfg.branch, "--head", branch]
        cmd += ["--title", f"release: {kind} {name}"]
        cmd += ["--body", f"Promote `{src}` to `{dst}`. Merging this PR is the review."]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out["pr"] = proc.stdout.strip() or proc.stderr.strip()
    else:
        out["pr"] = f"open a pull request from {branch} to {cfg.branch}"
    return out
