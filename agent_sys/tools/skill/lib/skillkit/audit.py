"""Audit a handoff (or any deliverable dir) before it is shipped.

agent_sys has no size limit and no secret scan (`handoff/` enforces neither),
so these are the skills' own rules, taken from experiment-result-packup:

* a file over 4 MB is a question for the user, not a default inclusion;
* a logs directory is a question for the user;
* a secret VALUE must never be in it — names and sources only. Every file is
  scanned, whatever its size, in one streaming pass: gzip / bzip2 / xz are
  decompressed (bounded) first, UTF-16 is decoded, and in a binary only the
  printable strings are read. A hit in a binary, or a file that cannot be
  scanned (another archive format, a decompression bomb), is a question for
  the user, never a silent pass;
* an unfilled template placeholder is a defect the seal would also refuse.

Absolute paths are reported as information only: the runtime's locality
check was switched off after measuring 97% false positives, and an
`environment.yaml` naming `model_path` is correct, not a leak.
"""

from __future__ import annotations

import bz2
import codecs
import gzip
import io
import lzma
import os
import re
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from skillkit.secretscan import looks_secret

BIG_FILE = 4 << 20
#: Placeholder text in source code is code (a validator's own pattern), not a gap.
CODE_SUFFIXES = frozenset(
    ".py .sh .bash .js .mjs .ts .c .cc .cpp .h .hpp .go .rs .java .cu .hip".split()
)
TEXT_SCAN_LIMIT = 4 << 20
#: A compressed stream that inflates past this is reported as unscanned (a bomb).
DECOMPRESS_LIMIT = 1 << 30
CHUNK = 1 << 20
#: What decides text, UTF-16 or binary (git's own rule: a NUL in the first 8000 bytes).
SAMPLE = 8000
#: Around a hint, at most this much of a line (minified JSON) goes through the patterns.
_PIECE, _OVERLAP = 8192, 512
#: No secretscan pattern can match without one of these (ASCII, any case), so
#: only the text around one is checked: a multi-GB file scans at memchr speed,
#: the patterns alone run at ~3 MB/s. Keep in step with secretscan.PATTERNS
#: (`tests/tools/test_skillkit_remote_safety.py` checks one sample per pattern).
_HINTS = tuple(
    b"key secret token pass pwd auth bearer basic login curl :// sk- ghp_ gho_ ghu_ ghs_ "
    b"github_pat glpat- xox hf_ dckr_pat akia eyj private".split()
)
_PRINTABLE = re.compile(rb"[\t\x20-\x7e]+")
_COMPRESSED = (
    (b"\x1f\x8b", lambda fh: gzip.GzipFile(fileobj=fh)),
    (b"BZh", bz2.BZ2File),
    (b"\xfd7zXZ\x00", lzma.LZMAFile),
)
#: Small archives of these may hold no NUL in their first 8000 bytes.
_OPAQUE = (
    (b"PK\x03\x04", "zip archive"),
    (b"(\xb5/\xfd", "zstd stream"),
    (b"7z\xbc\xaf'\x1c", "7z archive"),
)
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
    unscanned: list[str] = field(default_factory=list)
    binary_hits: list[str] = field(default_factory=list)

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
        out += [
            f"{u} — not scanned for secrets: check it by hand; include, or drop?"
            for u in self.unscanned
        ]
        out += [
            f"{h} — secret-looking string in a binary file: check it by hand; include, or drop?"
            for h in self.binary_hits
        ]
        if self.log_paths:
            out.append(
                f"{len(self.log_paths)} log file(s) included (e.g. {self.log_paths[0]}): keep, trim, gzip?"
            )
        return out

    def render(self) -> str:
        lines = [f"audit {self.root}: {self.files} files, {self.total_bytes / 2**20:.1f} MB"]
        lines += [f"  BLOCK  {b}" for b in self.blocking]
        lines += [f"  ASK    {a}" for a in self.ask_user]
        lines += [f"  note   symlink leaves the dir: {s}" for s in self.symlinks_out]
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
        try:
            scan(a, rel, p)
        except OSError as e:
            a.unscanned.append(f"{rel}: unreadable ({e.strerror or e})")
    return a


def scan(
    a: Audit,
    rel: str,
    src: bytes | Path,
    *,
    sink: Callable[[bytes], object] | None = None,
    placeholders: bool = True,
) -> None:
    """Add one file (its path, or its bytes) to `a`, read once in chunks of any
    size; `sink` (if given) receives every byte that was scanned, in order.
    `placeholders=False`: a record, not prose to fill in. Raises OSError."""
    with io.BytesIO(src) if isinstance(src, bytes) else Path(src).open("rb") as fh:
        raw = _Source(fh.read, sink)
        _scan(a, rel, raw.read, 0)
        while raw.read(CHUNK):  # what a decompressor left unread is still published
            pass
    a.files += 1
    a.total_bytes += raw.size
    if raw.size > BIG_FILE:
        a.big_files.append((rel, raw.size))
    if Path(rel).suffix == ".log" or "/logs/" in f"/{rel}":
        a.log_paths.append(rel)
    if not placeholders or raw.size > TEXT_SCAN_LIMIT or Path(rel).suffix in CODE_SUFFIXES:
        return
    try:
        prose = _prose(bytes(raw.head).decode("utf-8"))
    except UnicodeDecodeError:
        return
    a.placeholders += [f"{rel}:{n}" for n, line in prose.items() if PLACEHOLDERS.search(line)]


class _Source:
    """The raw file: counts it, hands each chunk to `sink`, keeps the first
    TEXT_SCAN_LIMIT + 1 bytes for the placeholder check."""

    def __init__(self, read: Callable[[int], bytes], sink) -> None:
        self._read, self._sink, self.size, self.head = read, sink, 0, bytearray()

    def read(self, n: int) -> bytes:
        data = self._read(n)
        self.size += len(data)
        if self._sink is not None and data:
            self._sink(data)
        if len(self.head) <= TEXT_SCAN_LIMIT:
            self.head += data[: TEXT_SCAN_LIMIT + 1 - len(self.head)]
        return data


class _Stream:
    """A forward-only file object for a decompressor: `head` (already read), then `read`."""

    def __init__(self, head: bytes, read: Callable[[int], bytes]) -> None:
        self._head, self._read = head, read

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        n = CHUNK if n is None or n < 0 else n
        if self._head:
            out, self._head = self._head[:n], self._head[n:]
            return out
        return self._read(n)


class _TooBig(Exception):
    pass


def _bounded(inner) -> Callable[[int], bytes]:
    """`inner.read`, stopped once the decompressed position passes the limit."""

    def read(n: int) -> bytes:
        if inner.tell() > DECOMPRESS_LIMIT:
            raise _TooBig
        return inner.read(n)

    return read


def _scan(a: Audit, rel: str, read: Callable[[int], bytes], depth: int) -> None:
    """Scan one stream; a compressed one is decompressed (bounded) and scanned inside."""
    head = read(SAMPLE)
    for magic, opener in _COMPRESSED:
        if head.startswith(magic):
            if depth >= 3:
                a.unscanned.append(f"{rel}: compressed more than 3 times")
                return
            try:
                with opener(_Stream(head, read)) as inner:
                    _scan(a, rel, _bounded(inner), depth + 1)
            except _TooBig:
                a.unscanned.append(f"{rel}: decompresses to over {DECOMPRESS_LIMIT >> 20} MB")
            except (OSError, EOFError, lzma.LZMAError, zlib.error) as e:
                a.unscanned.append(f"{rel}: cannot decompress ({e})")
            return
    for magic, what in _OPAQUE:
        if head.startswith(magic):
            a.unscanned.append(f"{rel}: {what}, cannot be scanned")
            return
    # The raw bytes are always scanned; a stream that looks like UTF-16 (a BOM
    # can also be a stray byte pair) is scanned decoded as well.
    raw = _Scanner(a, rel, binary=b"\0" in head)
    enc = _utf16(head)
    wide = _Scanner(a, rel, binary=False) if enc else None
    dec = codecs.getincrementaldecoder(enc)(errors="replace") if enc else None
    # Inside a compressed stream, small reads: what came before a corrupt or
    # truncated part is still scanned (the flush below runs on the error too).
    data, step = head, CHUNK if depth == 0 else 1 << 16
    try:
        while data:
            raw.feed(data)
            if wide:
                wide.feed(dec.decode(data).encode())
            data = read(step)
    finally:
        raw.feed(b"", final=True)
        if wide:
            wide.feed(dec.decode(b"", final=True).encode(), final=True)


def _utf16(head: bytes) -> str | None:
    """The codec of a UTF-16 stream (PowerShell's `>`), or None. There every
    ASCII character is paired with a NUL, so a token is not ASCII bytes."""
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    half = len(head) // 2
    if half < 8:
        return None
    even, odd = head[0 : 2 * half : 2].count(0), head[1 : 2 * half : 2].count(0)
    if odd > 0.4 * half and even < 0.05 * half:
        return "utf-16-le"
    if even > 0.4 * half and odd < 0.05 * half:
        return "utf-16-be"
    return None


class _Scanner:
    """Secret scan of a byte stream fed in chunks of any size, at memchr speed.

    Only the text around a hint (`_HINTS`) goes through the patterns: in text,
    the hint's line (an 8 KB window of a longer one); in a binary (a NUL in the
    first 8 KB, as git decides), the printable run around it — so NULs never
    become part of a value. A hit in text blocks; a hit in a binary is a string
    table as often as a secret, so it is a question for the user."""

    def __init__(self, a: Audit, rel: str, *, binary: bool) -> None:
        self.a, self.rel, self.binary = a, rel, binary
        self.buf, self.base, self.lines, self.pos = b"", 0, 0, 0
        self.last = (-1, -1)  # the window checked last: a line with many hints is checked once

    def feed(self, data: bytes, *, final: bool = False) -> None:
        buf = self.buf = self.buf + data
        limit = len(buf) if final else len(buf) - _PIECE
        if limit > self.pos:
            low = buf.lower()
            for s in sorted(i for h in _HINTS for i in _find_all(low, h, self.pos, limit)):
                self._check(buf, s)
            self.pos = limit
        cut = max(0, self.pos - _PIECE)
        if cut:
            self.lines += buf.count(b"\n", 0, cut)
            self.base += cut
            self.buf, self.pos = buf[cut:], self.pos - cut
            self.last = (self.last[0] - cut, self.last[1] - cut)

    def _check(self, buf: bytes, s: int) -> None:
        half = _PIECE // 2
        if self.binary:
            if s < self.last[1]:
                return
            runs = _PRINTABLE.finditer(buf, max(0, s - half), s + half)
            run = next((m for m in runs if m.start() <= s < m.end()), None)
            if run is None:
                return
            lo, hi = run.span()
        else:
            nl = buf.rfind(b"\n", max(0, s - half), s)
            lo = nl + 1 if nl >= 0 else max(0, s - half)
            end = buf.find(b"\n", s, s + half)
            hi = end if end >= 0 else min(len(buf), s + half)
            if (lo, hi) == self.last or s < self.last[1] - _OVERLAP:
                return
        self.last = (lo, hi)
        if not looks_secret(buf[lo:hi].decode("utf-8", errors="replace")):
            return
        if self.binary:
            self.a.binary_hits.append(f"{self.rel} (byte {self.base + s})")
        else:
            line = self.lines + buf.count(b"\n", 0, s) + 1
            where = f"{self.rel}:{line}"
            if where not in self.a.secrets:
                self.a.secrets.append(where)


def _find_all(buf: bytes, needle: bytes, start: int, stop: int) -> Iterator[int]:
    i = buf.find(needle, start, stop + len(needle) - 1)
    while i != -1:
        yield i
        i = buf.find(needle, i + 1, stop + len(needle) - 1)


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
