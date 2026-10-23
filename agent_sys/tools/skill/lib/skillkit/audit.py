"""Audit a handoff (or any deliverable dir) before it is shipped.

agent_sys has no size limit and no secret scan (`handoff/` enforces neither),
so these are the skills' own rules, taken from experiment-result-packup:

* a file over 4 MB is a question for the user, not a default inclusion;
* a logs directory is a question for the user;
* a secret VALUE must never be in it — names and sources only;
* an unfilled template placeholder is a defect the seal would also refuse.

Absolute paths are reported as information only: the runtime's locality
check was switched off after measuring 97% false positives, and an
`environment.yaml` naming `model_path` is correct, not a leak.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from skillkit.secretscan import looks_secret

BIG_FILE = 4 << 20
#: Placeholder text in source code is code (a validator's own pattern), not a gap.
CODE_SUFFIXES = frozenset(
    ".py .sh .bash .js .mjs .ts .c .cc .cpp .h .hpp .go .rs .java .cu .hip".split()
)
TEXT_SCAN_LIMIT = 4 << 20
PLACEHOLDERS = re.compile(r"\[More Information Needed\]|<fill[^>]*>|\bTODO: fill\b", re.I)


@dataclass
class Audit:
    root: str
    files: int = 0
    total_bytes: int = 0
    big_files: list[tuple[str, int]] = field(default_factory=list)
    log_paths: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    symlinks_out: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[str]:
        out = [f"secret-looking value at {s}" for s in self.secrets]
        out += [f"unfilled placeholder at {p}" for p in self.placeholders]
        return out

    @property
    def ask_user(self) -> list[str]:
        out = [
            f"{p} is {n / 2**20:.1f} MB (> 4 MB): include, reference by path+sha256, or drop?"
            for p, n in self.big_files
        ]
        if self.log_paths:
            out.append(
                f"{len(self.log_paths)} log file(s) included (e.g. {self.log_paths[0]}): keep, trim, gzip?"
            )
        return out

    def render(self) -> str:
        lines = [f"audit {self.root}: {self.files} files, {self.total_bytes / 2**20:.1f} MB"]
        for b in self.blocking:
            lines.append(f"  BLOCK  {b}")
        for a in self.ask_user:
            lines.append(f"  ASK    {a}")
        for s in self.symlinks_out:
            lines.append(f"  note   symlink leaves the dir: {s}")
        if not self.blocking and not self.ask_user:
            lines.append("  clean")
        return "\n".join(lines)


def files(root: Path, *, follow: bool = True) -> list[tuple[str, Path]]:
    """`(relative path, real file)` for every file under `root`.

    `follow=True` walks INTO symlinked files and directories, because that is
    what a dereferencing copy (a task backup) publishes: a link to a secret
    outside the package is the secret. Directory cycles are cut."""
    root = root.resolve()
    out: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow):
        real = Path(dirpath).resolve()
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.is_symlink() and not follow:
                continue
            if p.is_file():
                out.append((str(p.relative_to(root)), p.resolve()))
    return out


def audit(root: Path, *, follow: bool = True) -> Audit:
    root = root.resolve()
    a = Audit(root=str(root))
    for p in sorted(root.rglob("*")):
        if p.is_symlink() and not str(p.resolve()).startswith(str(root)):
            a.symlinks_out.append(f"{p.relative_to(root)} -> {p.resolve()}")
    for rel, p in files(root, follow=follow):
        size = p.stat().st_size
        a.files += 1
        a.total_bytes += size
        if size > BIG_FILE:
            a.big_files.append((rel, size))
        if p.suffix in (".log",) or "/logs/" in f"/{rel}" or rel.startswith("logs/"):
            a.log_paths.append(rel)
        if size > TEXT_SCAN_LIMIT:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        prose = {} if p.suffix in CODE_SUFFIXES else _prose(text)
        for n, line in enumerate(text.splitlines(), 1):
            if looks_secret(line):
                a.secrets.append(f"{rel}:{n}")
            if PLACEHOLDERS.search(prose.get(n, "")):
                a.placeholders.append(f"{rel}:{n}")
    return a


def _prose(text: str) -> dict[int, str]:
    """Lines outside code fences, `inline code` removed: quoting a placeholder
    is a mention, not an unfilled section (same rule as handoff_check)."""
    out, fence = {}, False
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            continue
        if not fence:
            out[n] = re.sub(r"`[^`]*`", "", line)
    return out
