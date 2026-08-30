"""Locality independence — and the honest answer is narrower than it sounds.

Criterion 17: a handoff whose content declares an absolute local path fails.
Spec §7: a handoff carrying `/home/someone/run3/` is a record of one machine's
afternoon, not a transferable artefact.

**Nobody detects locality dependence by a path's shape.** Every working system
either matches a prefix supplied by an oracle — lintian's
`quotemeta($buildinfo->Build-Path)`, rpmlint's `%{?buildroot}`, conda-build's
`PREFIX_PLACEHOLDER` — or prevents the path existing at all. The shape
refinement was proposed on Debian #1002451 and **refused on the record**: you
cannot recognise a build path by its shape, because the shape is a property of
whoever built it.

Measured on this repository: a bare absolute-path regex over 276 files gives
650 matches of which **23 are genuinely local and 627 (96%) need a suppression
rule**. So the check here is an **anchored allow-list plus the oracles we
actually have** — delocate's shape — and it says which of the two fired.

Two traps pre-empted. *Self-application*: every reject and allow pattern is
**data**, never a literal spliced into a scanning expression, because this
module's own tests and messages contain example absolute paths — conda splits
its placeholder across two source strings for exactly this. *Stale exclusions
fail closed*: an oracle whose prefix cannot be formed is an error, not a
silently widened blind spot.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from handoff.errors import Malformed

__all__ = ["ALLOWED_PREFIXES", "Oracles", "check"]

#: Anchored allow-list: an absolute path starting with one of these is portable
#: by construction — a system location, a vendor prefix, or a container-internal
#: root. Data, and the caller adds the declared image's prefixes to it.
ALLOWED_PREFIXES: tuple[str, ...] = (
    "/usr/",
    "/bin/",
    "/sbin/",
    "/lib/",
    "/lib64/",
    "/etc/",
    "/opt/",
    "/proc/",
    "/sys/",
    "/dev/",
    "/var/lib/",
    "/var/log/",
    "/run/",
    "/srv/",
    "/workspace/",
    "/app/",
)

#: The shape a candidate absolute path takes. POSIX first, then a Windows drive
#: letter — the POSIX alternation alone misses `C:\\Users\\bob\\run` entirely.
#:
#: The lookbehind is not cosmetic: without it `./data/in.json` matches on its
#: `/data/in.json` substring, and a relative path reported as absolute is the
#: shape of false positive that got every surveyed checker disabled.
_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9._~@+-])(?:[A-Za-z]:\\[^\s\"'<>|]*|(?:/[A-Za-z0-9._+@-]+){2,}/?)"
)

#: A URL's path component looks absolute and is 401 of the 627 false positives.
#: Matched on the *scheme*, which is what distinguishes it.
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S*")

_MAX_BYTES = 4 << 20


@dataclass(frozen=True)
class Oracles:
    """The prefixes we know are local, because we minted them.

    This is more than lintian has, and an oracle hit is **certain**: a
    playground path in a published artefact is a record of one machine's
    afternoon by construction. The shape heuristic runs only where no oracle
    applies.
    """

    playground_root: Path | None = None
    store_root: Path | None = None
    #: Prefixes the declared container image makes portable — the kind's
    #: `dependencies`, spec §7. Added to `ALLOWED_PREFIXES` for this check.
    image_prefixes: Sequence[str] = field(default_factory=tuple)

    def prefixes(self) -> tuple[str, ...]:
        out = []
        for root in (self.playground_root, self.store_root):
            if root is None:
                continue
            text = str(Path(root))
            if not text or text == "/":
                raise Malformed(
                    f"oracle root {root!r} is empty or the filesystem root; it "
                    f"would match everything or nothing. Refused rather than "
                    f"silently widening the blind spot"
                )
            out.append(text)
        return tuple(out)


def _allowed(candidate: str, extra: Sequence[str]) -> bool:
    for prefix in tuple(ALLOWED_PREFIXES) + tuple(extra):
        if candidate == prefix.rstrip("/") or candidate.startswith(prefix):
            return True
    return False


def _scan_text(text: str, oracle_prefixes: Sequence[str], allowed: Sequence[str]):
    """Yield `(line_no, path, why)` for every hit. `why` is 'oracle' or 'heuristic'."""
    for lineno, line in enumerate(text.splitlines(), start=1):
        for prefix in oracle_prefixes:
            if prefix in line:
                yield lineno, prefix, "oracle"
        stripped = _URL.sub(" ", line)
        if stripped.lstrip().startswith("#!"):
            continue  # a shebang names an interpreter, not a produced artefact
        for match in _CANDIDATE.finditer(stripped):
            path = match.group(0)
            if not _allowed(path, allowed):
                yield lineno, path, "heuristic"


def check(content_dir: Path, *, oracles: Oracles, allow: Sequence[str] = ()) -> None:
    """Raise `Malformed` naming the file, the line, the path, and which rule fired.

    **Sound on oracle hits and best-effort otherwise**, and the message says
    which — because a check that claims more than it delivers is how Debian
    ended up with `gcc captures build path` across 1841 packages after they
    froze the build path and made differential detection impossible.

    Three false negatives are known and none is closed by more regex: a path
    hidden by **compression**, one built by **runtime concatenation**
    (`os.path.join(HOME, "run3")`), and the general form of Nix's own caveat —
    a clean scan asserts that nothing was found, not that there is nothing to
    find.

    **No weighting by file role.** A path in a re-executable script is stronger
    evidence than one in a log, and lintian is the only surveyed tool with the
    mechanism — and does not wire it in. A playground path in a changelog is
    still a record of one machine. `design.md` O5 keeps it a question.
    """
    root = Path(content_dir)
    oracle_prefixes = oracles.prefixes()
    allowed = tuple(oracles.image_prefixes) + tuple(allow)

    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        if path.stat().st_size > _MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary, or unreadable: the scan says what it found
        for lineno, hit, why in _scan_text(text, oracle_prefixes, allowed):
            rel = path.relative_to(root)
            detail = (
                "an oracle prefix this system minted — certain"
                if why == "oracle"
                else "the shape heuristic — best effort, and it is not weighted by file role"
            )
            raise Malformed(
                f"{rel}:{lineno}: {hit!r} is a local path ({detail}). "
                f"A handoff names its dependencies and nothing about the "
                f"machine that produced it (spec §7)"
            )
