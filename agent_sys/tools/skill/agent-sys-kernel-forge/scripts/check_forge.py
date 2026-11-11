#!/usr/bin/env python3
"""Resolve a session's AKLR/Hyperloom sources, then verify its Forge runtime."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HYPERLOOM_REPOSITORY = "https://github.com/AMD-AGI/Hyperloom.git"
AKLR_REPOSITORY = "https://github.com/AMD-AGI/AKLR.git"
SUGGESTED_HYPERLOOM_REVISION = "fa94e001b88d77e69d8d50973c868a8f4a11a15d"
SUGGESTED_AKLR_REVISION = "5d59c935e7e47410e1f84bc4c2a5077816d4071d"
DEFAULT_KB_PATH = "harness/skills/aklr-kb"
COMMANDS = ("forge-loop", "forge-say", "forge-ablate", "forge-finalize", "forge-distill")
LOOP_OPTIONS = (
    "--kernel-backend",
    "--skill-dir",
    "--resume",
    "--paired-keep",
    "--early-stop",
    "--agent-backend",
    "--max-rounds",
)


class PreflightError(ValueError):
    """A source, knowledge reference or runtime cannot serve this workflow."""


class KnowledgeMoved(PreflightError):
    """A discovered replacement requires the user's adoption decision."""

    def __init__(self, aklr: Path, candidates: list[str]):
        self.candidates = candidates
        super().__init__(
            f"AKLR's {DEFAULT_KB_PATH} moved in {aklr}; discovered: {', '.join(candidates)}. "
            "Report these locations to the user, then pass --accept-kb-path <relative path> "
            "only after they choose the replacement. No campaign was started."
        )


def _run(argv: list[str]) -> str:
    if argv[:2] == ["git", "-C"] and len(argv) >= 3:
        # Session checkouts can cross host/container UIDs. Trust only this
        # explicitly selected checkout for this invocation, without changing
        # global configuration or trusting any other directory.
        checkout = str(Path(argv[2]).resolve())
        argv = ["git", "-c", f"safe.directory={checkout}", *argv[1:]]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise PreflightError(f"{' '.join(argv[:4])}: {detail.strip()}") from error
    return result.stdout.strip()


def _source(root: Path, expected: str, paths: list[str]) -> str:
    actual = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    if actual != expected:
        raise PreflightError(f"{root}: expected revision {expected}, found {actual}")
    changed = _run(["git", "-C", str(root), "status", "--porcelain", "--", *paths])
    if changed:
        raise PreflightError(
            f"{root}: referenced runtime/KB sources have local changes; use a clean session checkout"
        )
    return actual


def _remote_revision(repository: str, ref: str) -> str:
    refs = (
        [ref]
        if ref.startswith("refs/")
        else [f"refs/heads/{ref}", f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}"]
    )
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        return ref.lower()
    lines = _run(["git", "ls-remote", repository, *refs]).splitlines()
    found = dict(line.split()[::-1] for line in lines if len(line.split()) == 2)
    for name in refs:
        if re.fullmatch(r"[0-9a-f]{40}", found.get(name, "")):
            return found[name]
    raise PreflightError(f"{repository}: cannot resolve declared ref {ref}")


def _checkout(
    repository: str,
    revision: str,
    candidate: Path | None,
    destination: Path,
    *,
    reuse_candidate: bool = True,
) -> Path:
    """Reuse a matching clean checkout or materialize an owned immutable one."""
    if candidate and candidate.is_dir():
        try:
            head = _run(["git", "-C", str(candidate), "rev-parse", "HEAD"])
            dirty = _run(
                ["git", "-C", str(candidate), "status", "--porcelain", "--untracked-files=no"]
            )
            if reuse_candidate and head == revision and not dirty:
                return candidate.resolve()
        except PreflightError:
            pass
    if destination.exists():
        try:
            _run(["git", "-C", str(destination), "rev-parse", "HEAD"])
        except PreflightError:
            # A failed fetch can leave an initialized directory with no HEAD.
            # Preserve it and retry in a fresh owned sibling, without deleting
            # artifacts or resetting any caller-owned checkout.
            for attempt in range(1, 100):
                retry = destination.with_name(f"{destination.name}-retry-{attempt}")
                if not retry.exists():
                    return _checkout(repository, revision, candidate, retry, reuse_candidate=False)
            raise PreflightError(
                f"too many incomplete source checkouts beside {destination}"
            ) from None
        _source(destination, revision, ["."])
        return destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Local Git objects make the suggested-revision fallback usable offline.
    local = False
    if candidate:
        try:
            _run(["git", "-C", str(candidate), "cat-file", "-e", f"{revision}^{{commit}}"])
            local = True
        except PreflightError:
            pass
    if local:
        _run(["git", "clone", "--no-checkout", "--no-hardlinks", str(candidate), str(destination)])
        _run(["git", "-C", str(destination), "checkout", "--detach", revision])
    else:
        _run(["git", "init", "--quiet", str(destination)])
        _run(["git", "-C", str(destination), "fetch", "--depth=1", repository, revision])
        _run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"])
    return destination.resolve()


def _knowledge_identity(path: Path) -> bool:
    entry = path / "SKILL.md"
    if not entry.is_file():
        return False
    text = entry.read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)
    return len(frontmatter) == 3 and bool(
        re.search(r"^name:\s*['\"]?aklr-kb['\"]?\s*$", frontmatter[1], re.MULTILINE)
    )


def _knowledge_path(aklr: Path, accepted: str = "") -> Path:
    if accepted:
        selected = (aklr / accepted).resolve()
        if not selected.is_relative_to(aklr.resolve()) or not _knowledge_identity(selected):
            raise PreflightError(
                "--accept-kb-path must identify the aklr-kb skill inside the selected AKLR checkout"
            )
        return selected
    canonical = aklr / DEFAULT_KB_PATH
    if _knowledge_identity(canonical):
        return canonical
    candidates = sorted(
        str(path.parent.relative_to(aklr))
        for path in aklr.rglob("SKILL.md")
        if ".git" not in path.parts
        and _knowledge_identity(path.parent)
        and path.resolve().is_relative_to(aklr.resolve())
    )
    if candidates:
        raise KnowledgeMoved(aklr, candidates)
    raise PreflightError("AKLR's kernel knowledge skill could not be located")


def _hyperloom_declaration(aklr: Path) -> dict | None:
    """Find a setup declaration, excluding historical kernel/KB citations."""
    preferred = [
        "harness/scripts/forge_task/README.md",
        "docs/forge_task.md",
        "pyproject.toml",
        "requirements.txt",
    ]
    config = sorted(
        {
            str(path.relative_to(aklr))
            for pattern in (
                "**/Dockerfile*",
                ".github/**/*.yml",
                ".github/**/*.yaml",
                "**/*requirements*.txt",
                "**/*lock*.json",
            )
            for path in aklr.glob(pattern)
            if path.is_file() and ".git" not in path.parts
        }
    )
    documents = [
        (name, (aklr / name).read_text(encoding="utf-8"))
        for name in dict.fromkeys(preferred + config)
        if (aklr / name).is_file()
    ]
    patterns = [
        r"github\.com[:/]AMD-AGI/Hyperloom(?:\.git)?@([A-Za-z0-9._/-]+)",
        r"\bHYPERLOOM_(?:REVISION|COMMIT|REF)\s*[:=]\s*['\"]?([A-Za-z0-9._/-]+)",
    ]
    for pattern in patterns:
        for name, text in documents:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {"path": name, "ref": match[1], "kind": "revision-or-ref"}
    for name, text in documents:
        match = re.search(r"Hyperloom['’]s\s+[`'\"]?([A-Za-z0-9._/-]+)[`'\"]?\s+branch", text)
        if match:
            return {"path": name, "ref": match[1], "kind": "branch"}
    return None


def resolve_sources(
    hyperloom: Path | None,
    aklr: Path | None,
    workspace: Path,
    *,
    accept_kb_path: str = "",
    use_suggested: str = "",
) -> dict:
    """Resolve once at session start; never reset a caller's checkout."""
    workspace = workspace.resolve()
    reasons = []
    if use_suggested:
        reasons.append(f"explicit adaptation fallback: {use_suggested.strip()}")
    else:
        try:
            aklr_sha = _remote_revision(AKLR_REPOSITORY, "refs/heads/main")
            aklr_root = _checkout(
                AKLR_REPOSITORY, aklr_sha, aklr, workspace / f"aklr-{aklr_sha[:12]}"
            )
            kb = _knowledge_path(aklr_root, accept_kb_path)
        except KnowledgeMoved:
            raise
        except (PreflightError, OSError) as error:
            if accept_kb_path:
                raise
            reasons.append(f"AKLR main unavailable or knowledge discovery failed: {error}")
    if reasons:
        aklr_sha = SUGGESTED_AKLR_REVISION
        aklr_root = _checkout(
            AKLR_REPOSITORY,
            aklr_sha,
            aklr,
            workspace / f"aklr-fallback-{aklr_sha[:12]}",
            reuse_candidate=False,
        )
        kb = _knowledge_path(aklr_root)
    declaration = _hyperloom_declaration(aklr_root)
    if reasons or declaration is None:
        hyperloom_sha = SUGGESTED_HYPERLOOM_REVISION
        if declaration is None:
            reasons.append(
                "AKLR declares no discoverable Hyperloom revision/ref; using the suggested revision"
            )
    else:
        try:
            hyperloom_sha = _remote_revision(HYPERLOOM_REPOSITORY, declaration["ref"])
        except PreflightError as error:
            reasons.append(
                f"AKLR's declared Hyperloom ref could not be resolved: {error}; using the suggested revision"
            )
            hyperloom_sha = SUGGESTED_HYPERLOOM_REVISION
    hyperloom_root = _checkout(
        HYPERLOOM_REPOSITORY,
        hyperloom_sha,
        hyperloom,
        workspace / f"hyperloom-{hyperloom_sha[:12]}",
        reuse_candidate=not reasons,
    )
    return {
        "hyperloom_revision": hyperloom_sha,
        "aklr_revision": aklr_sha,
        "hyperloom_repo": str(hyperloom_root),
        "aklr_repo": str(aklr_root),
        "aklr_knowledge": {"path": str(kb), "relative_path": str(kb.relative_to(aklr_root))},
        "resolution": {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "aklr_requested_ref": "suggested" if use_suggested else "main",
            "hyperloom_declaration": declaration,
            "fallback_reasons": reasons,
        },
    }


def _backends(hyperloom: Path) -> list[str]:
    source = hyperloom / "src/kernelforge/kernel_backends/constants.py"
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "KERNEL_BACKEND_PROMPT_MODULES"
            for target in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise PreflightError(f"kernel backend registry was not found in {source}")


def _copied_source(original: Path, installed: Path) -> None:
    tracked = _run(
        ["git", "-C", str(original), "ls-files", "-z", "--", "pyproject.toml", "src/kernelforge"]
    )
    for name in tracked.split("\0"):
        if name and (original / name).read_bytes() != (installed / name).read_bytes():
            raise PreflightError(f"installed source copy differs from pinned Hyperloom: {name}")
    original_names = {name for name in tracked.split("\0") if name}
    for path in (installed / "src/kernelforge").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            if path.relative_to(installed).as_posix() not in original_names:
                raise PreflightError(f"installed source copy contains an unpinned file: {path}")


def check(
    hyperloom: Path,
    aklr: Path,
    backend: str = "",
    *,
    source_only: bool = False,
    installed_source: Path | None = None,
    resolution: dict | None = None,
) -> dict:
    selected = resolution or resolve_sources(hyperloom, aklr, aklr.parent / "forge-sources")
    hyperloom, aklr = Path(selected["hyperloom_repo"]), Path(selected["aklr_repo"])
    kb = Path(selected["aklr_knowledge"]["path"])
    relative_kb = str(kb.relative_to(aklr))
    if not _knowledge_identity(kb) or not kb.resolve().is_relative_to(aklr.resolve()):
        raise PreflightError(
            "the session's selected AKLR KB is absent, moved or outside its checkout; resolve a new session"
        )
    hyperloom_sha = _source(
        hyperloom, selected["hyperloom_revision"], ["pyproject.toml", "src/kernelforge"]
    )
    aklr_sha = _source(aklr, selected["aklr_revision"], [relative_kb, "harness/scripts/forge_task"])
    backends = _backends(hyperloom)
    if backend and backend not in backends:
        raise PreflightError(
            f"unsupported kernel backend {backend!r}; registered: {', '.join(backends)}"
        )
    language = "triton" if backend == "gluon" else backend
    skill_dirs = [kb]
    if language and (kb / language / "SKILL.md").is_file():
        skill_dirs.append(kb / language)
    if not (kb / "SKILL.md").is_file():
        raise PreflightError(f"AKLR kernel knowledge entry is absent: {kb}/SKILL.md")
    report = {
        "hyperloom_revision": hyperloom_sha,
        "aklr_revision": aklr_sha,
        "hyperloom_repo": str(hyperloom),
        "aklr_repo": str(aklr),
        "hyperloom": {
            "revision": hyperloom_sha,
            "source_path": str(hyperloom),
            "repository": "https://github.com/AMD-AGI/Hyperloom",
        },
        "aklr_knowledge": {
            "revision": aklr_sha,
            "source_path": str(aklr),
            "path": str(kb),
            "relative_path": relative_kb,
            "reference": f"https://github.com/AMD-AGI/AKLR/blob/{aklr_sha}/{relative_kb}/SKILL.md",
        },
        "resolution": selected["resolution"],
        "kernel_backend": backend,
        "backends": backends,
        "skill_dirs": [str(path) for path in skill_dirs],
        "installed": None,
    }
    if source_only:
        return report
    runtime_source = (installed_source or hyperloom).resolve()
    if runtime_source != hyperloom:
        _copied_source(hyperloom, runtime_source)
    probe = _run(
        [
            sys.executable,
            "-c",
            (
                "import json, kernelforge; "
                "from kernelforge.kernel_backends.constants import KERNEL_BACKENDS; "
                "print(json.dumps({'module': kernelforge.__file__, 'backends': KERNEL_BACKENDS}))"
            ),
        ]
    )
    try:
        runtime = json.loads(probe)
        module = Path(runtime["module"]).resolve()
        installed_backends = runtime["backends"]
    except (ValueError, KeyError, TypeError) as error:
        raise PreflightError(
            "installed Forge did not report its module path and backends"
        ) from error
    if not module.is_relative_to(runtime_source / "src/kernelforge"):
        raise PreflightError(
            f"installed kernelforge resolves to {module}, outside {runtime_source}/src/kernelforge"
        )
    if installed_backends != backends:
        raise PreflightError("installed Forge backend registry differs from the pinned source")
    help_text = _run([sys.executable, "-m", "kernelforge.cli", "--help"])
    missing = [command for command in COMMANDS if command not in help_text.split()]
    if missing:
        raise PreflightError(f"installed Forge is missing commands: {', '.join(missing)}")
    loop_help = _run([sys.executable, "-m", "kernelforge.cli", "forge-loop", "--help"])
    missing = [option for option in LOOP_OPTIONS if option not in loop_help]
    if missing:
        raise PreflightError(f"installed forge-loop is missing options: {', '.join(missing)}")
    report.update(
        module=str(module),
        commands=list(COMMANDS),
        installed={
            "source_path": str(runtime_source),
            "module": str(module),
            "commands": list(COMMANDS),
            "loop_options": list(LOOP_OPTIONS),
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyperloom", default=os.environ.get("HYPERLOOM_REPO"))
    parser.add_argument("--aklr", default=os.environ.get("AKLR_REPO"))
    parser.add_argument("--backend", default="")
    parser.add_argument(
        "--session", type=Path, help="Reuse this run's recorded source selection; defaults to --out"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Owned source checkout directory; defaults to <session parent>/sources",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Resolve sources at session start without importing Forge",
    )
    parser.add_argument(
        "--accept-kb-path", default="", help="User-approved moved aklr-kb path, relative to AKLR"
    )
    parser.add_argument(
        "--use-suggested",
        default="",
        metavar="REASON",
        help="Use separate suggested-revision snapshots after explaining why current AKLR cannot be adapted",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Check sources before installation; do not import Forge",
    )
    parser.add_argument(
        "--installed-source",
        type=Path,
        help="Editable scratch copy installed instead of the original checkout",
    )
    parser.add_argument(
        "--out", type=Path, help="Also write JSON evidence into an existing task directory"
    )
    args = parser.parse_args()
    session = args.session or args.out
    if session is None:
        parser.error("pass --session or --out to record the session's resolved sources")
    if args.use_suggested and (not args.use_suggested.strip() or args.accept_kb_path):
        parser.error("--use-suggested needs a reason and cannot be combined with --accept-kb-path")
    try:
        if session.is_file():
            selected = json.loads(session.read_text(encoding="utf-8"))
            if "resolution" not in selected:
                raise PreflightError(
                    f"{session} predates session source resolution; choose a new session file"
                )
            if args.use_suggested and selected["resolution"]["aklr_requested_ref"] != "suggested":
                raise PreflightError(
                    "--use-suggested changes source selection; choose a new --session file to preserve earlier evidence"
                )
        else:
            selected = resolve_sources(
                Path(args.hyperloom) if args.hyperloom else None,
                Path(args.aklr) if args.aklr else None,
                args.workspace or session.parent / "sources",
                accept_kb_path=args.accept_kb_path,
                use_suggested=args.use_suggested,
            )
            session.parent.mkdir(parents=True, exist_ok=True)
            session.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        report = check(
            Path(selected["hyperloom_repo"]),
            Path(selected["aklr_repo"]),
            args.backend,
            source_only=args.source_only or args.resolve,
            installed_source=args.installed_source,
            resolution=selected,
        )
        payload = json.dumps(report, indent=2) + "\n"
        if args.out:
            args.out.write_text(payload, encoding="utf-8")
    except KnowledgeMoved as error:
        print(f"Forge source discovery needs user input: {error}", file=sys.stderr)
        return 3
    except (PreflightError, OSError, ValueError, KeyError) as error:
        print(f"Forge preflight: {error}", file=sys.stderr)
        return 1
    print(payload, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
