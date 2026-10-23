"""The shared knowledge repo: where tasks and handoffs are backed up and found.

Layout of the remote (a plain git repo):

    release/{tasks,handoffs}/<name>/          reviewed; changed ONLY through a pull request
    <user>/draft/{tasks,handoffs}/<name>/     a user's backups land here
    <user>/staging/{tasks,handoffs}/<name>/   promoted from draft after the user's own review

Each item carries `META.astk.json` (what it is, who backed it up, from where,
when, its digest, its context). A task item is the package directory; a
handoff item is the store shape — `content/` (+ `manifest.yaml`,
`validation.yaml` when it was sealed).

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

import datetime as dt
import getpass
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from skillkit import AGENT_SYS, hstore
from skillkit import audit as audit_mod

META = "META.astk.json"
TIERS = ("release", "staging", "draft")
TYPES = ("task", "handoff")
PUSH_TRIES = 4


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
    if override:
        return Path(override)
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "infera-agent-sys" / "config.json"


def default_clone() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "infera-agent-sys" / "knowledge"


def default_user() -> str:
    try:
        name = subprocess.run(
            ["git", "config", "user.email"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        name = ""
    return slug(name.strip().split("@")[0] or getpass.getuser())


def slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-.")
    if not out:
        raise RemoteError(f"{text!r} gives an empty name")
    return out[:80]


def load() -> Config:
    path = config_path()
    if not path.is_file():
        raise NotConfigured(
            f"no knowledge repo configured ({path} missing). Configure it once with:\n"
            "  astk remote init --url <git url> [--user <name>] [--clone <dir>]"
        )
    return Config(**json.loads(path.read_text()))


def _git(
    where: Config | Path,
    *args: str,
    check: bool = True,
    env: dict | None = None,
    data: bytes | None = None,
) -> str:
    cwd = Path(where.clone) if isinstance(where, Config) else where
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        check=False,
        env={**os.environ, **(env or {})},
        input=data,
    )
    out = proc.stdout.decode(errors="replace").strip()
    err = proc.stderr.decode(errors="replace").strip()
    if check and proc.returncode != 0:
        raise RemoteError(f"git {' '.join(args[:4])} failed in {cwd}:\n{err or out}")
    return out


# -- setup and sync ------------------------------------------------------------- #


def init(
    url: str, *, user: str | None = None, clone: str | None = None, branch: str = "main"
) -> Config:
    """Clone the repo (or adopt an existing clone of the same URL) and save the config."""
    cfg = Config(
        url=url,
        clone=str(Path(clone or default_clone()).expanduser().resolve()),
        user=slug(user or default_user()),
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
        proc = subprocess.run(
            ["git", "clone", "-q", url, str(path)], capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            raise RemoteError(f"git clone {url} failed:\n{proc.stderr.strip()}")
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    return cfg


def can_push(cfg: Config) -> tuple[bool, str]:
    """`git push --dry-run`: find a missing write permission now, not at the end of a packup."""
    proc = subprocess.run(
        ["git", "push", "--dry-run", "origin", f"HEAD:{cfg.branch}"],
        cwd=cfg.clone,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0, (proc.stderr.strip().splitlines() or [""])[-1]


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
    proc = subprocess.run(
        ["git", "merge", "-q", "--ff-only", f"origin/{cfg.branch}"],
        cwd=cfg.clone,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RemoteError(
            f"{cfg.clone} cannot fast-forward to origin/{cfg.branch} (local edits or commits?); "
            f"nothing was changed — `git -C {cfg.clone} status` shows why.\n{proc.stderr.strip()}"
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


def _meta(p: Path) -> dict:
    """META of one item; a malformed one must not take the listing down for everybody."""
    try:
        meta = json.loads((p / META).read_text()) if (p / META).is_file() else {}
    except (OSError, ValueError):
        return {"meta_error": f"unreadable {META}"}
    if not isinstance(meta, dict):
        return {"meta_error": f"{META} is not an object"}
    if not isinstance(meta.get("context", {}), dict):
        meta["context"] = {}
    return meta


def items(cfg: Config) -> list[Item]:
    """Every item in the clone's working tree (= the last synced remote tip)."""
    root = Path(cfg.clone)
    out: list[Item] = []

    def collect(base: Path, tier: str, user: str | None) -> None:
        for t in TYPES:
            d = base / f"{t}s"
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.is_dir() and not p.name.startswith("."):
                    meta = _meta(p)
                    if t == "handoff":
                        meta.setdefault("sealed", (p / "manifest.yaml").is_file())
                    out.append(
                        Item(type=t, tier=tier, user=user, name=p.name, path=str(p), meta=meta)
                    )

    collect(root / "release", "release", None)
    for u in sorted(root.iterdir()):
        if u.is_dir() and u.name != "release" and not u.name.startswith("."):
            for tier in ("staging", "draft"):
                collect(u / tier, tier, u.name)
    return out


# -- building commits on the remote tip ----------------------------------------- #


class _Tree:
    """A private index on top of a commit. Edits never touch the clone's own index;
    lookups (`exists`, `entries`) read the base commit."""

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
        sha = _git(self.cfg, "hash-object", "-w", "--stdin", data=data)
        _git(self.cfg, "update-index", "--add", "--cacheinfo", f"{mode},{sha},{path}", env=self.env)

    def entries(self, prefix: str) -> list[tuple[str, str, str]]:
        """(mode, sha, path) of every file under `prefix` in the base commit."""
        out = []
        for line in _git(self.cfg, "ls-tree", "-r", self.base, "--", prefix).splitlines():
            info, path = line.split("\t", 1)
            mode, _type, sha = info.split()
            out.append((mode, sha, path))
        return out

    def copy(self, src: str, dst: str, *, remove_src: bool = False) -> None:
        moved = self.entries(src)
        lines = [f"0 {'0' * 40}\t{p}" for _m, _s, p in moved] if remove_src else []
        lines += [f"{m} {s}\t{dst + p[len(src) :]}" for m, s, p in moved]
        _git(
            self.cfg,
            "update-index",
            "--index-info",
            env=self.env,
            data=("\n".join(lines) + "\n").encode(),
        )

    def close(self) -> None:
        if os.path.exists(self.index):
            os.unlink(self.index)

    def commit(self, message: str) -> str:
        try:
            tree = _git(self.cfg, "write-tree", env=self.env)
            return _git(self.cfg, "commit-tree", tree, "-p", self.base, "-m", message)
        finally:
            self.close()


def _publish(
    cfg: Config, build: Callable[[_Tree], tuple[str, dict]], *, ref: str | None = None
) -> tuple[str, dict]:
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
        proc = subprocess.run(
            ["git", "push", "-q", "origin", f"{sha}:{target}"],
            cwd=cfg.clone,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            if ref is None:
                try:
                    sync(cfg)
                except RemoteError:
                    pass  # pushed; the clone catches up on the next sync that can
            return sha, result
        last = proc.stderr.strip()
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


def _kind_of_source(src: Path) -> str:
    if (src / "main.yaml").is_file() and (src / "assets").is_dir():
        return "task"
    if (src / "README.md").is_file() and (src / "items").is_dir():
        return "handoff"
    if (src / "content").is_dir():
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
                out.append((rel, real, "100755" if os.access(real, os.X_OK) else "100644"))
        return out
    content = src / "content" if (src / "content").is_dir() else src
    for p in sorted(content.rglob("*")):
        rel = "content/" + str(p.relative_to(content))
        if p.is_symlink():
            out.append((rel, os.readlink(p), "120000"))
        elif p.is_file():
            out.append((rel, p, "100755" if os.access(p, os.X_OK) else "100644"))
    for extra in ("manifest.yaml", "validation.yaml"):
        if content is not src and (src / extra).is_file():
            out.append((extra, src / extra, "100644"))
    return out


def backup(
    cfg: Config,
    src: Path,
    *,
    name: str | None = None,
    summary: str = "",
    tags: list[str] | None = None,
    context: dict[str, str] | None = None,
    allow_big: bool = False,
    dry_run: bool = False,
) -> dict:
    """Publish a task package or a handoff as `<user>/draft/<type>s/<name>/`."""
    src = src.resolve()
    kind = _kind_of_source(src)
    content = src / "content" if (src / "content").is_dir() else src
    # Audit exactly what is published: for a task, link targets included.
    audited = audit_mod.audit(src if kind == "task" else content, follow=kind == "task")
    if audited.blocking:
        raise Refused("refusing to back up:\n  " + "\n  ".join(audited.blocking))
    if audited.big_files and not allow_big:
        raise Refused(
            "files over 4 MB (ask the user; then --allow-big, or reference them by path + sha256):\n  "
            + "\n  ".join(f"{p} ({n / 2**20:.1f} MB)" for p, n in audited.big_files)
        )
    entries = _source_files(src, kind)
    meta: dict = {
        "type": kind,
        "user": cfg.user,
        "tier": "draft",
        "summary": summary,
        "tags": tags or [],
        "context": {**_lift_context(src), **(context or {})},
        "source": str(src),
        "source_host": platform.node(),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "agent_sys_commit": _git(AGENT_SYS, "rev-parse", "HEAD", check=False),
    }
    if kind == "handoff":
        meta["sealed"] = content is not src and (src / "manifest.yaml").is_file()
        meta["digest"] = hstore.digest(content)
        if meta["sealed"]:
            import yaml  # noqa: PLC0415

            meta["handoff_kind"] = (yaml.safe_load((src / "manifest.yaml").read_text()) or {}).get(
                "kind"
            )
    base = f"{cfg.user}/draft/{kind}s"

    def build(tree: _Tree) -> tuple[str, dict]:
        item = _free(tree, base, slug(name or src.name))
        root = f"{base}/{item}"
        for rel, what, mode in entries:
            data = str(what).encode() if mode == "120000" else Path(what).read_bytes()
            tree.add_file(f"{root}/{rel}", data, mode)
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
    _sha, result = _publish(cfg, build)
    return {"path": str(Path(cfg.clone) / result["relative"]), **result}


#: environment-record keys worth surfacing in search, under the name they get in META.
_CONTEXT_KEYS = {
    "model_name": "model",
    "model": "model",
    "model_path": "model_path",
    "image": "image",
    "image_id": "image_id",
    "gpu_arch": "gpu_arch",
    "gpu": "gpu",
    "gpu_model": "gpu",
    "node": "host",
    "hostname": "host",
    "engine": "engine",
    "sglang_version": "sglang",
    "vllm_version": "vllm",
    "rocm_version": "rocm",
    "tp": "tp",
    "tp_size": "tp",
}


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


def promote(cfg: Config, kind: str, name: str) -> dict:
    """draft -> staging, after the user's own review: one commit moving the item."""
    if kind not in TYPES:
        raise RemoteError(f"type must be one of {TYPES}")
    src = f"{cfg.user}/draft/{kind}s/{name}"
    dst = f"{cfg.user}/staging/{kind}s/{name}"

    def build(tree: _Tree) -> tuple[str, dict]:
        if not tree.exists(src):
            raise RemoteError(f"no draft {kind} {name!r} of user {cfg.user} on the remote")
        if tree.exists(dst):
            raise RemoteError(f"{dst} already exists; promote under another name")
        stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        tree.copy(src, dst, remove_src=True)
        tree.add_file(f"{dst}/{META}", _meta_blob(tree, src, tier="staging", promoted=stamp))
        return f"promote({cfg.user}): {kind} {name} draft -> staging", {"relative": dst}

    _sha, result = _publish(cfg, build)
    return {"path": str(Path(cfg.clone) / result["relative"]), **result}


def release_pr(cfg: Config, kind: str, name: str) -> dict:
    """Propose a staging item for `release/` as a branch built on the remote tip
    (never from the local clone), and open a pull request when `gh` can."""
    src = f"{cfg.user}/staging/{kind}s/{name}"
    dst = f"release/{kind}s/{name}"
    branch = f"release/{cfg.user}-{kind}-{name}"

    def build(tree: _Tree) -> tuple[str, dict]:
        if not tree.exists(src):
            raise RemoteError(
                f"no staging {kind} {name!r} of user {cfg.user} on the remote (promote it first)"
            )
        if tree.exists(dst):
            raise RemoteError(f"{dst} already exists; propose under another name")
        tree.copy(src, dst)
        tree.add_file(f"{dst}/{META}", _meta_blob(tree, src, tier="release", proposed_from=src))
        return f"release: {kind} {name} (from {cfg.user}/staging)", {
            "branch": branch,
            "path_on_branch": dst,
        }

    if _git(cfg, "ls-remote", "--heads", "origin", branch, check=False):
        raise RemoteError(
            f"branch {branch} already exists on the remote: the item was proposed already"
        )
    _sha, result = _publish(cfg, build, ref=f"refs/heads/{branch}")
    out = dict(result)
    if shutil.which("gh") and "github.com" in cfg.url:
        repo = re.sub(r"^(git@github\.com:|https://github\.com/)|\.git$", "", cfg.url)
        proc = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                repo,
                "--base",
                cfg.branch,
                "--head",
                branch,
                "--title",
                f"release: {kind} {name}",
                "--body",
                f"Promote `{src}` to `{dst}`. Merging this PR is the review.",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        out["pr"] = proc.stdout.strip() or proc.stderr.strip()
    else:
        out["pr"] = f"open a pull request from {branch} to {cfg.branch}"
    return out
