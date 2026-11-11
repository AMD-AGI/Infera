#!/usr/bin/env python3
"""Check the pinned Hyperloom Forge runtime and externally referenced AKLR KB."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

HYPERLOOM_REVISION = "fa94e001b88d77e69d8d50973c868a8f4a11a15d"
AKLR_REVISION = "5d59c935e7e47410e1f84bc4c2a5077816d4071d"
COMMANDS = ("forge-loop", "forge-say", "forge-ablate", "forge-finalize", "forge-distill")
LOOP_OPTIONS = ("--kernel-backend", "--skill-dir", "--resume", "--paired-keep", "--early-stop")


class PreflightError(ValueError):
    """A source, knowledge reference or runtime cannot serve this workflow."""


def _run(argv: list[str]) -> str:
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
        raise PreflightError(f"{root}: referenced runtime/KB sources have local changes; use a clean pinned checkout")
    return actual


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
    tracked = _run(["git", "-C", str(original), "ls-files", "-z", "--", "pyproject.toml", "src/kernelforge"])
    for name in tracked.split("\0"):
        if name and (original / name).read_bytes() != (installed / name).read_bytes():
            raise PreflightError(f"installed source copy differs from pinned Hyperloom: {name}")
    original_names = {name for name in tracked.split("\0") if name}
    for path in (installed / "src/kernelforge").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            if path.relative_to(installed).as_posix() not in original_names:
                raise PreflightError(f"installed source copy contains an unpinned file: {path}")


def check(hyperloom: Path, aklr: Path, backend: str = "", *,
          source_only: bool = False, installed_source: Path | None = None) -> dict:
    hyperloom, aklr = hyperloom.resolve(), aklr.resolve()
    hyperloom_sha = _source(hyperloom, HYPERLOOM_REVISION, ["pyproject.toml", "src/kernelforge"])
    aklr_sha = _source(aklr, AKLR_REVISION, ["harness/skills/aklr-kb", "harness/scripts/forge_task"])
    backends = _backends(hyperloom)
    if backend and backend not in backends:
        raise PreflightError(f"unsupported kernel backend {backend!r}; registered: {', '.join(backends)}")
    kb = aklr / "harness/skills/aklr-kb"
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
        "hyperloom": {"revision": hyperloom_sha, "source_path": str(hyperloom),
                      "repository": "https://github.com/AMD-AGI/Hyperloom"},
        "aklr_knowledge": {"revision": aklr_sha, "source_path": str(aklr), "path": str(kb),
                           "reference": f"https://github.com/AMD-AGI/AKLR/blob/{aklr_sha}/harness/skills/aklr-kb/SKILL.md"},
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
    probe = _run([sys.executable, "-c", (
        "import json, kernelforge; "
        "from kernelforge.kernel_backends.constants import KERNEL_BACKENDS; "
        "print(json.dumps({'module': kernelforge.__file__, 'backends': KERNEL_BACKENDS}))"
    )])
    try:
        runtime = json.loads(probe)
        module = Path(runtime["module"]).resolve()
        installed_backends = runtime["backends"]
    except (ValueError, KeyError, TypeError) as error:
        raise PreflightError("installed Forge did not report its module path and backends") from error
    if not module.is_relative_to(runtime_source / "src/kernelforge"):
        raise PreflightError(f"installed kernelforge resolves to {module}, outside {runtime_source}/src/kernelforge")
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
    report.update(module=str(module), commands=list(COMMANDS),
                  installed={"source_path": str(runtime_source), "module": str(module),
                             "commands": list(COMMANDS), "loop_options": list(LOOP_OPTIONS)})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyperloom", default=os.environ.get("HYPERLOOM_REPO"))
    parser.add_argument("--aklr", default=os.environ.get("AKLR_REPO"))
    parser.add_argument("--backend", default="")
    parser.add_argument("--source-only", action="store_true", help="Check sources before installation; do not import Forge")
    parser.add_argument("--installed-source", type=Path, help="Editable scratch copy installed instead of the original checkout")
    parser.add_argument("--out", type=Path, help="Also write JSON evidence into an existing task directory")
    args = parser.parse_args()
    if not args.hyperloom or not args.aklr:
        parser.error("set --hyperloom/HYPERLOOM_REPO and --aklr/AKLR_REPO to pinned checkouts")
    try:
        report = check(Path(args.hyperloom), Path(args.aklr), args.backend,
                       source_only=args.source_only, installed_source=args.installed_source)
        payload = json.dumps(report, indent=2) + "\n"
        if args.out:
            args.out.write_text(payload, encoding="utf-8")
    except (PreflightError, OSError) as error:
        print(f"Forge preflight: {error}", file=sys.stderr)
        return 1
    print(payload, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
